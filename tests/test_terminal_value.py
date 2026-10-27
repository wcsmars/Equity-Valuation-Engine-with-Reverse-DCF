"""Terminal-value reinvestment checks and the opt-in value-driver terminal value.

The DCF splits the first stable year (N+1) into NOPAT and net reinvestment
(capex - D&A + dNWC), records the reinvestment rate and the return on new
invested capital (RONIC) the Gordon terminal value implies, and offers
``terminal_method="value_driver"``: NOPAT x (1 - g / RONIC) / (WACC - g).
Gordon stays the default, so the pinned demo figures do not move. A
value-driver headline's sensitivity cells run the value-driver method at the
headline's RONIC (WACC when the headline replaced the one given), so its
grids are centred on it; the engine drops any grid priced with another
terminal method. Offline: everything runs on the synthetic company. Run with:
python -m unittest tests.test_terminal_value
"""

from __future__ import annotations

import contextlib
from dataclasses import replace
import io
import math
import os
import tempfile
import unittest
from unittest.mock import patch

from equity_valuation import engine, value_company
from equity_valuation.data.synthetic import DEMO_TICKER, SyntheticProvider, make_company
from equity_valuation.models import sensitivity
from equity_valuation.models.dcf import (
    TERMINAL_METHODS,
    latest_roic,
    resolve_terminal_method,
    run_dcf,
)
from equity_valuation.models.sensitivity import dcf_sensitivity
from equity_valuation.schemas import (
    AnnualFinancials,
    DCFAssumptions,
    MacroAssumptions,
    SensitivityResult,
)


def _rel_close(test, a, b, rel=1e-9):
    test.assertTrue(abs(a - b) <= rel * max(abs(a), abs(b)), f"{a} != {b} (rel {rel})")


def _company(capex_share=0.05, da_share=0.04, nwc_share=0.0, revenue=None):
    """The synthetic company on a steady 6% revenue ramp with capex, D&A and
    dNWC set as shares of revenue (25% EBIT margin, 21% tax)."""
    base = make_company()
    revenue = revenue or [100e9 * 1.06 ** i for i in range(5)]
    n = len(revenue)
    ebit = [r * 0.25 for r in revenue]
    pretax = [e * 0.95 for e in ebit]
    tax = [p * 0.21 for p in pretax]
    fin = AnnualFinancials(
        fiscal_years=list(range(2025 - n + 1, 2026)),
        revenue=list(revenue),
        ebit=ebit,
        ebitda=[e + r * da_share for e, r in zip(ebit, revenue)],
        net_income=[p - t for p, t in zip(pretax, tax)],
        dep_amort=[r * da_share for r in revenue],
        capex=[r * capex_share for r in revenue],
        change_in_nwc=[r * nwc_share for r in revenue],
        interest_expense=[e * 0.05 for e in ebit],
        tax_expense=tax,
        pretax_income=pretax,
        dividends_paid=[(p - t) * 0.3 for p, t in zip(pretax, tax)],
        diluted_shares=[base.financials.diluted_shares[-1]] * n,
    )
    return replace(base, financials=fin)


def _with_kind(company, kind):
    fin = replace(company.financials)
    fin._financial_kind = kind
    return replace(company, financials=fin)


def _notes(dcf):
    return dcf.assumptions["notes"]


def _run_cli(argv):
    """(exit code, stdout, stderr) of cli.main, with SystemExit mapped to its code."""
    from equity_valuation.cli import main

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = main(argv)
        except SystemExit as exc:
            code = exc.code
    return code, out.getvalue(), err.getvalue()


def _tmpdir(test):
    tmp = tempfile.TemporaryDirectory()
    test.addCleanup(tmp.cleanup)
    return tmp.name


GROWTH_WITHOUT_REINVESTMENT = "implies growth without reinvestment"
HIGH_RONIC = "implied return on new invested capital"
# Both Gordon notes name the remedy in model terms: the dashboard cannot pick
# the value-driver method, so they do not point to it.
REVIEW_DRIVERS = "review the terminal capex, D&A and working-capital drivers or the terminal growth"
GRIDS_OMITTED = "Sensitivity: grids omitted"


class GordonReinvestmentDiagnosticsTests(unittest.TestCase):
    """The first stable year's split and the implied RONIC on SYNT."""

    @classmethod
    def setUpClass(cls):
        cls.company = SyntheticProvider().get_company_data(DEMO_TICKER)
        cls.macro = MacroAssumptions()
        cls.price = cls.company.market.price
        cls.dcf = run_dcf(cls.company, cls.macro, DCFAssumptions(), cls.price)

    def test_terminal_fcff_is_nopat_less_net_reinvestment(self):
        a = self.dcf.assumptions
        _rel_close(self, a["terminal_fcff"],
                   a["terminal_nopat"] - a["terminal_net_reinvestment"])
        # The split also reproduces the Gordon terminal value.
        _rel_close(self, self.dcf.terminal_value,
                   a["terminal_fcff"] / (a["wacc"] - a["terminal_growth_used"]))

    def test_synt_reinvestment_rate_and_implied_ronic(self):
        a = self.dcf.assumptions
        self.assertAlmostEqual(a["terminal_reinvestment_rate"], 0.0673047, places=6)
        self.assertAlmostEqual(a["implied_terminal_ronic"], 0.3714450, places=6)
        _rel_close(self, a["implied_terminal_ronic"],
                   a["terminal_growth_used"] / a["terminal_reinvestment_rate"])
        self.assertIsNone(a["terminal_ronic_used"])  # Gordon: no RONIC input
        bs, fin = self.company.balance_sheet, self.company.financials
        expected_roic = (fin.ebit[-1] * (1 - a["tax_rate"])
                         / (bs.total_equity + bs.total_debt - bs.cash_and_investments))
        _rel_close(self, a["latest_roic"], expected_roic)

    def test_demo_adds_no_warning_and_keeps_its_pins(self):
        # 37% implied RONIC is below the 100% floor: no note on the demo.
        self.assertEqual(_notes(self.dcf), [])
        report = value_company(DEMO_TICKER, provider=SyntheticProvider())
        for w in report.warnings:
            self.assertNotIn("RONIC", w)
            self.assertNotIn("reinvestment", w)
            self.assertNotIn("value-driver", w)
        self.assertEqual(round(report.dcf.implied_price, 2), 33.38)
        self.assertEqual(round(report.summary["blended_target"], 2), 32.40)

    def test_exit_multiple_leaves_the_diagnostics_empty(self):
        dcf = run_dcf(self.company, self.macro,
                      DCFAssumptions(terminal_method="exit_multiple", exit_ev_ebitda=12.0),
                      self.price)
        for key in ("terminal_nopat", "terminal_net_reinvestment", "terminal_reinvestment_rate",
                    "implied_terminal_ronic", "terminal_ronic_used", "latest_roic"):
            self.assertIsNone(dcf.assumptions[key], key)


class ValueDriverTerminalValueTests(unittest.TestCase):
    """The opt-in value-driver terminal value."""

    @classmethod
    def setUpClass(cls):
        cls.company = SyntheticProvider().get_company_data(DEMO_TICKER)
        cls.macro = MacroAssumptions()
        cls.price = cls.company.market.price
        cls.gordon = run_dcf(cls.company, cls.macro, DCFAssumptions(), cls.price)

    def _vd(self, **kwargs):
        return run_dcf(self.company, self.macro,
                       DCFAssumptions(terminal_method="value_driver", **kwargs), self.price)

    def test_method_is_accepted_and_normalised(self):
        self.assertIn("value_driver", TERMINAL_METHODS)
        self.assertEqual(resolve_terminal_method(DCFAssumptions(terminal_method=" Value_Driver ")),
                         ("value_driver", None))
        self.assertEqual(self._vd().assumptions["terminal_method"], "value_driver")

    def test_ronic_at_the_gordon_implied_level_reproduces_the_gordon_tv(self):
        ronic = self.gordon.assumptions["implied_terminal_ronic"]
        vd = self._vd(terminal_ronic=ronic)
        _rel_close(self, vd.terminal_value, self.gordon.terminal_value)
        _rel_close(self, vd.implied_price, self.gordon.implied_price)
        self.assertEqual(vd.assumptions["terminal_ronic_used"], ronic)
        self.assertTrue(any("value-driver terminal value at RONIC 37.1%" in n
                            for n in _notes(vd)), _notes(vd))

    def test_default_ronic_is_wacc_and_the_tv_ignores_growth(self):
        tvs = []
        for g in (0.015, 0.025):
            with self.subTest(terminal_growth=g):
                vd = self._vd(terminal_growth=g)
                a = vd.assumptions
                self.assertEqual(a["terminal_ronic_used"], a["wacc"])
                _rel_close(self, vd.terminal_value, a["terminal_nopat"] / a["wacc"])
                self.assertIn("terminal RONIC defaults to WACC: growth adds no value",
                              " ".join(_notes(vd)))
                tvs.append(vd.terminal_value / a["terminal_nopat"])
        _rel_close(self, tvs[0], tvs[1])  # TV / NOPAT_{N+1} = 1 / WACC either way

    def test_ronic_not_above_growth_falls_back_to_wacc(self):
        for ronic in (0.02, 0.025):
            with self.subTest(terminal_ronic=ronic):
                vd = self._vd(terminal_ronic=ronic)
                a = vd.assumptions
                self.assertEqual(a["terminal_ronic_used"], a["wacc"])
                self.assertTrue(any("must exceed terminal growth" in n and "using WACC" in n
                                    for n in _notes(vd)), _notes(vd))
                _rel_close(self, vd.terminal_value, a["terminal_nopat"] / a["wacc"])

    def test_non_positive_or_non_numeric_ronic_falls_back_to_wacc(self):
        vd = self._vd(terminal_growth=-0.02, terminal_ronic=-0.01)
        self.assertEqual(vd.assumptions["terminal_ronic_used"], vd.assumptions["wacc"])
        self.assertTrue(any("must be positive" in n for n in _notes(vd)), _notes(vd))
        vd = self._vd(terminal_ronic=float("nan"))
        self.assertEqual(vd.assumptions["terminal_ronic_used"], vd.assumptions["wacc"])
        self.assertTrue(any("is not a number" in n for n in _notes(vd)), _notes(vd))

    def test_negative_growth_notes_the_capital_release(self):
        # g -1% at RONIC 0.1%: 10x NOPAT of capital released a year, TV ~105x
        # NOPAT against ~10.5x for NOPAT / WACC.
        vd = self._vd(terminal_growth=-0.01, terminal_ronic=0.001)
        a = vd.assumptions
        w = a["wacc"]
        _rel_close(self, vd.terminal_value, a["terminal_nopat"] * 11.0 / (w + 0.01))
        note = next(n for n in _notes(vd) if n.startswith("value-driver terminal value"))
        self.assertIn("shrinking 1.0% a year releases invested capital worth 1000.0% of "
                      "first stable-year NOPAT a year", note)
        self.assertIn(f"the TV ({11.0 / (w + 0.01):.1f}x NOPAT) exceeds the no-growth "
                      f"NOPAT / WACC ({1 / w:.1f}x)", note)
        self.assertNotIn("reinvested to grow", note)
        # A RONIC above WACC releases less than the NOPAT lost: no excess clause.
        vd = self._vd(terminal_growth=-0.01, terminal_ronic=0.15)
        self.assertLess(vd.terminal_value, vd.assumptions["terminal_nopat"] / w)
        note = next(n for n in _notes(vd) if n.startswith("value-driver terminal value"))
        self.assertIn("releases invested capital worth 6.7%", note)
        self.assertNotIn("exceeds", note)
        # At the WACC default shrinking adds no value: TV = NOPAT / WACC.
        vd = self._vd(terminal_growth=-0.01)
        _rel_close(self, vd.terminal_value, vd.assumptions["terminal_nopat"] / w)
        self.assertFalse(any("releases" in n for n in _notes(vd)), _notes(vd))

    def test_formula_and_gordon_timing(self):
        for mid_year in (True, False):
            with self.subTest(mid_year=mid_year):
                vd = self._vd(terminal_ronic=0.15, mid_year_convention=mid_year)
                a = vd.assumptions
                g, w = a["terminal_growth_used"], a["wacc"]
                _rel_close(self, a["terminal_fcff"], a["terminal_nopat"] * (1 - g / 0.15))
                _rel_close(self, vd.terminal_value, a["terminal_fcff"] / (w - g))
                n = a["forecast_years"]
                self.assertEqual(a["terminal_discount_exponent"], n - 0.5 if mid_year else n)
                self.assertIn("value-driver TV", a["terminal_discount_exponent_rationale"])
                _rel_close(self, vd.pv_terminal,
                           vd.terminal_value / (1 + w) ** a["terminal_discount_exponent"])
                # The driver-based diagnostics are still reported for comparison.
                self.assertEqual(a["implied_terminal_ronic"],
                                 self.gordon.assumptions["implied_terminal_ronic"])

    def test_growth_too_close_to_wacc_is_clamped_as_for_gordon(self):
        vd = self._vd(terminal_growth=0.094, terminal_ronic=0.5)
        a = vd.assumptions
        self.assertAlmostEqual(a["wacc"] - a["terminal_growth_used"], 0.01, places=12)
        self.assertTrue(any("clamped" in n for n in _notes(vd)))
        self.assertGreater(vd.terminal_value, 0)

    def test_negative_nopat_is_noted(self):
        company = _company()
        vd = run_dcf(company, MacroAssumptions(),
                     DCFAssumptions(terminal_method="value_driver", target_ebit_margin=-0.05),
                     company.market.price)
        self.assertLess(vd.terminal_value, 0)
        self.assertTrue(any("NOPAT is negative" in n for n in _notes(vd)), _notes(vd))

    def test_engine_runs_a_value_driver_valuation(self):
        a = DCFAssumptions(terminal_method="value_driver", terminal_ronic=0.15)
        report = value_company(DEMO_TICKER, provider=SyntheticProvider(), dcf_assumptions=a,
                               run_sensitivity=False)
        direct = run_dcf(self.company, self.macro, a, self.price)
        self.assertAlmostEqual(report.dcf.implied_price, direct.implied_price, places=9)
        self.assertTrue(any(w.startswith("DCF: value-driver terminal value at RONIC 15.0%")
                            for w in report.warnings), report.warnings)
        # The reverse DCF re-runs the same terminal method.
        self.assertTrue(report.reverse_dcf["converged"])


class ReinvestmentNoteTests(unittest.TestCase):
    """The Gordon consistency notes."""

    def _dcf(self, company, **kwargs):
        return run_dcf(company, MacroAssumptions(), DCFAssumptions(**kwargs),
                       company.market.price)

    def test_capex_below_da_is_growth_without_reinvestment(self):
        dcf = self._dcf(_company(capex_share=0.036, da_share=0.04))
        a = dcf.assumptions
        self.assertLess(a["terminal_reinvestment_rate"], 0)
        self.assertIsNone(a["implied_terminal_ronic"])
        note = next((n for n in _notes(dcf) if GROWTH_WITHOUT_REINVESTMENT in n), None)
        self.assertIsNotNone(note, _notes(dcf))
        self.assertIn("(capex 0.90x D&A)", note)
        self.assertIn(REVIEW_DRIVERS, note)
        self.assertNotIn("value-driver", note)  # the dashboard cannot select it
        self.assertNotIn("--", note)  # no CLI flags: the dashboard shows the note too
        self.assertFalse(any(HIGH_RONIC in n for n in _notes(dcf)))

    def test_negative_nwc_is_named_beside_capex(self):
        # Capex above D&A, but an explicit negative dNWC share drives the
        # rate below zero; the note names both drivers.
        dcf = self._dcf(_company(capex_share=0.042, da_share=0.04), nwc_pct_revenue=-0.5)
        self.assertLess(dcf.assumptions["terminal_reinvestment_rate"], 0)
        note = next((n for n in _notes(dcf) if GROWTH_WITHOUT_REINVESTMENT in n), None)
        self.assertIsNotNone(note, _notes(dcf))
        self.assertIn("(capex 1.05x D&A, dNWC -50.0% of the revenue change)", note)

    def test_high_implied_ronic_is_noted_against_latest_roic(self):
        dcf = self._dcf(_company(capex_share=0.041, da_share=0.04))
        a = dcf.assumptions
        self.assertGreater(a["implied_terminal_ronic"], 1.0)
        note = next((n for n in _notes(dcf) if HIGH_RONIC in n), None)
        self.assertIsNotNone(note, _notes(dcf))
        self.assertIn(f"latest ROIC {a['latest_roic']:.1%}", note)
        self.assertIn(REVIEW_DRIVERS, note)
        self.assertNotIn("value-driver", note)
        self.assertNotIn("--", note)
        self.assertFalse(any(GROWTH_WITHOUT_REINVESTMENT in n for n in _notes(dcf)))

    def test_without_a_latest_roic_the_100_percent_floor_applies(self):
        def no_roic(company):
            # Net cash above equity: invested capital <= 0, so no latest ROIC.
            bs = company.balance_sheet
            return replace(company, balance_sheet=replace(
                bs, total_equity=-(bs.total_debt - bs.cash_and_investments) - 1.0))

        dcf = self._dcf(no_roic(_company(capex_share=0.041, da_share=0.04)))
        self.assertIsNone(dcf.assumptions["latest_roic"])
        self.assertGreater(dcf.assumptions["implied_terminal_ronic"], 1.0)
        note = next((n for n in _notes(dcf) if HIGH_RONIC in n), None)
        self.assertIsNotNone(note, _notes(dcf))
        self.assertIn("vs latest ROIC n/a:", note)
        # SYNT's 37% stays below the floor.
        synt = no_roic(SyntheticProvider().get_company_data(DEMO_TICKER))
        dcf = self._dcf(synt)
        self.assertIsNone(dcf.assumptions["latest_roic"])
        self.assertAlmostEqual(dcf.assumptions["implied_terminal_ronic"], 0.3714450, places=6)
        self.assertEqual(_notes(dcf), [])

    def test_high_ronic_note_needs_three_times_the_latest_roic(self):
        company = _company(capex_share=0.041, da_share=0.04)
        ronic = self._dcf(company).assumptions["implied_terminal_ronic"]
        # Shrink invested capital until 3x ROIC exceeds the implied RONIC.
        bs = company.balance_sheet
        fin = company.financials
        nopat = fin.ebit[-1] * (1 - self._dcf(company).assumptions["tax_rate"])
        invested = nopat / (ronic / 2.5)  # ROIC = RONIC / 2.5, so 3x ROIC > RONIC
        high_roic = replace(company, balance_sheet=replace(
            bs, total_equity=invested - bs.total_debt + bs.cash_and_investments))
        dcf = self._dcf(high_roic)
        self.assertAlmostEqual(dcf.assumptions["latest_roic"], ronic / 2.5, places=9)
        self.assertFalse(any(HIGH_RONIC in n for n in _notes(dcf)), _notes(dcf))

    def test_no_note_for_value_driver_exit_multiple_or_no_growth(self):
        company = _company(capex_share=0.036, da_share=0.04)
        for kwargs in ({"terminal_method": "value_driver"},
                       {"terminal_method": "exit_multiple", "exit_ev_ebitda": 12.0},
                       {"terminal_growth": 0.0}):
            with self.subTest(**kwargs):
                notes = " ".join(_notes(self._dcf(company, **kwargs)))
                self.assertNotIn(GROWTH_WITHOUT_REINVESTMENT, notes)
                self.assertNotIn(HIGH_RONIC, notes)

    def test_flagged_company_gets_no_note(self):
        # A bank's FCFF DCF is reference-only; the diagnostics are still kept.
        company = _with_kind(_company(capex_share=0.036, da_share=0.04), "bank")
        dcf = self._dcf(company)
        self.assertLess(dcf.assumptions["terminal_reinvestment_rate"], 0)
        self.assertFalse(any(GROWTH_WITHOUT_REINVESTMENT in n for n in _notes(dcf)))


class LatestRoicTests(unittest.TestCase):
    def test_missing_or_non_positive_inputs_give_none(self):
        company = make_company()
        fin, bs = company.financials, company.balance_sheet
        self.assertIsNotNone(latest_roic(fin, bs, 0.21))
        self.assertIsNone(latest_roic(replace(fin, ebit=fin.ebit[:-1] + [0.0]), bs, 0.21))
        self.assertIsNone(latest_roic(fin, None, 0.21))
        self.assertIsNone(latest_roic(None, bs, 0.21))
        self.assertIsNone(latest_roic(fin, replace(bs, total_equity=None), 0.21))
        negative = replace(bs, total_equity=-(bs.total_debt - bs.cash_and_investments) - 1.0)
        self.assertIsNone(latest_roic(fin, negative, 0.21))


def _grid_centre(grid):
    """The cell at no WACC (or margin) shift and the headline's column."""
    from equity_valuation import config

    return grid.grid[list(config.SENSITIVITY_WACC_DELTAS).index(0.0)][
        list(config.SENSITIVITY_GROWTH_DELTAS).index(0.0)]


def _value_driver_report(ronic=0.15):
    return value_company(DEMO_TICKER, provider=SyntheticProvider(),
                         dcf_assumptions=DCFAssumptions(terminal_method="value_driver",
                                                        terminal_ronic=ronic))


class ValueDriverGridGuardTests(unittest.TestCase):
    """The engine keeps a value-driver headline's grids only when their centre
    cell is the headline, so neither the grids nor the football-field DCF bar
    mix two terminal methods."""

    def _report_with(self, fake_sensitivity, ronic=0.15):
        with patch.object(sensitivity, "dcf_sensitivity", fake_sensitivity):
            return _value_driver_report(ronic)

    def test_value_driver_grids_are_kept_and_centred(self):
        # The cells keep the value-driver method, so a real run keeps both
        # grids, centred on the headline, and the DCF bar spans their cells.
        for ronic in (None, 0.15):
            with self.subTest(terminal_ronic=ronic):
                report = _value_driver_report(ronic)
                headline = report.dcf.implied_price
                self.assertEqual(len(report.sensitivities), 2)
                for grid in report.sensitivities:
                    self.assertAlmostEqual(_grid_centre(grid), headline, places=6)
                self.assertFalse(any(GRIDS_OMITTED in w for w in report.warnings))
                bar = next(r for r in report.football_field if r.method == "DCF")
                self.assertAlmostEqual(bar.base, headline, places=9)
                self.assertNotAlmostEqual(bar.low, headline * 0.85, places=6)

    def test_gordon_priced_grids_are_dropped(self):
        def gordon_cells(company, macro, assumptions, current_price):
            return dcf_sensitivity(company, macro,
                                   replace(assumptions, terminal_method="gordon"),
                                   current_price)

        report = self._report_with(gordon_cells)
        headline = report.dcf.implied_price
        self.assertAlmostEqual(headline, 30.70, places=2)
        self.assertEqual(report.sensitivities, [])
        warning = next(w for w in report.warnings if w.startswith(GRIDS_OMITTED))
        self.assertIn("centre cell (33.38) does not reproduce the value-driver DCF price "
                      "(30.70)", warning)
        # The DCF bar falls back to the +/-15% band around the headline.
        bar = next(r for r in report.football_field if r.method == "DCF")
        self.assertAlmostEqual(bar.low, headline * 0.85, places=9)
        self.assertAlmostEqual(bar.base, headline, places=9)
        self.assertAlmostEqual(bar.high, headline * 1.15, places=9)

    def _fake_grid(self, centre_value):
        """A WACC grid around ``centre_value`` (None: the headline itself)."""

        def fake(company, macro, assumptions, current_price):
            headline = run_dcf(company, macro, assumptions, current_price).implied_price
            centre = headline if centre_value is None else centre_value
            grid = [[headline * (1 + 0.05 * (i - 2) + 0.02 * (j - 2)) for j in range(5)]
                    for i in range(5)]
            grid[2][2] = centre
            return [SensitivityResult(title="DCF implied price: WACC vs terminal growth",
                                      row_label="WACC", col_label="Terminal growth",
                                      row_values=[0.08, 0.0875, 0.095, 0.1025, 0.11],
                                      col_values=[0.015, 0.02, 0.025, 0.03, 0.035],
                                      grid=grid)]
        return fake

    def test_centred_or_blank_centre_grids_are_kept(self):
        for centre in (None, float("nan")):
            with self.subTest(centre=centre):
                report = self._report_with(self._fake_grid(centre))
                self.assertEqual(len(report.sensitivities), 1)
                self.assertFalse(any(GRIDS_OMITTED in w for w in report.warnings))
                bar = next(r for r in report.football_field if r.method == "DCF")
                headline = report.dcf.implied_price
                self.assertAlmostEqual(bar.low, headline * 0.86, places=9)
                self.assertAlmostEqual(bar.high, headline * 1.14, places=9)

    def test_gordon_headline_is_not_checked(self):
        # Even an off-centre grid is left alone for a Gordon headline (its
        # cells run the headline's method by construction).
        with patch.object(sensitivity, "dcf_sensitivity", self._fake_grid(1.0)):
            report = value_company(DEMO_TICKER, provider=SyntheticProvider())
        self.assertEqual(len(report.sensitivities), 1)
        self.assertFalse(any(GRIDS_OMITTED in w for w in report.warnings))


class ValueDriverSensitivityTests(unittest.TestCase):
    """A value-driver headline's cells run the value-driver method, so
    both grids' centre cells are the headline, and a column whose growth is
    not below an explicit RONIC is blank rather than priced at RONIC = WACC.
    A RONIC the headline itself replaced with WACC is replaced in every cell."""

    @classmethod
    def setUpClass(cls):
        cls.company = SyntheticProvider().get_company_data(DEMO_TICKER)
        cls.macro = MacroAssumptions()
        cls.price = cls.company.market.price

    def test_centre_cells_equal_the_headline(self):
        gordon = run_dcf(self.company, self.macro, DCFAssumptions(), self.price).implied_price
        for ronic in (None, 0.15):
            with self.subTest(terminal_ronic=ronic):
                a = DCFAssumptions(terminal_method="value_driver", terminal_ronic=ronic)
                headline = run_dcf(self.company, self.macro, a, self.price).implied_price
                self.assertNotAlmostEqual(headline, gordon, places=2)
                grids = dcf_sensitivity(self.company, self.macro, a, self.price)
                self.assertEqual(len(grids), 2)
                for grid in grids:
                    self.assertEqual(grid.col_label, "Terminal growth")
                    self.assertAlmostEqual(_grid_centre(grid), headline, places=6)
                # The engine keeps them.
                report = _value_driver_report(ronic)
                self.assertEqual(len(report.sensitivities), 2)
                self.assertFalse(any(GRIDS_OMITTED in w for w in report.warnings))

    def test_every_cell_runs_the_value_driver_method_and_ronic(self):
        seen = set()
        real = sensitivity.run_dcf

        def spy(company, macro, assumptions, *args, **kwargs):
            seen.add((assumptions.terminal_method, assumptions.terminal_ronic))
            return real(company, macro, assumptions, *args, **kwargs)

        a = DCFAssumptions(terminal_method="value_driver", terminal_ronic=0.15)
        with patch.object(sensitivity, "run_dcf", spy):
            dcf_sensitivity(self.company, self.macro, a, self.price)
        self.assertEqual(seen, {("value_driver", 0.15)})

    def test_columns_not_below_an_explicit_ronic_are_blank(self):
        # RONIC 3%: the 3.0% and 3.5% columns would fall back to WACC.
        a = DCFAssumptions(terminal_method="value_driver", terminal_ronic=0.03)
        headline = run_dcf(self.company, self.macro, a, self.price).implied_price
        for grid in dcf_sensitivity(self.company, self.macro, a, self.price):
            self.assertEqual([round(c, 4) for c in grid.col_values],
                             [0.015, 0.02, 0.025, 0.03, 0.035])
            for row in grid.grid:
                self.assertTrue(all(math.isfinite(c) for c in row[:3]), row)
                self.assertTrue(all(math.isnan(c) for c in row[3:]), row)
            # The 2.5% column is below the RONIC, so the centre is the headline.
            self.assertAlmostEqual(_grid_centre(grid), headline, places=6)

    def _assert_same_grid(self, got, want):
        self.assertEqual(got.col_values, want.col_values)
        self.assertEqual(got.row_values, want.row_values)
        for got_row, want_row in zip(got.grid, want.grid):
            for g, w in zip(got_row, want_row):
                if math.isnan(w):
                    self.assertTrue(math.isnan(g), (got_row, want_row))
                else:
                    self.assertEqual(g, w)

    def test_a_ronic_the_headline_replaced_gives_the_wacc_grids(self):
        # RONIC at or below the headline's 2.5% growth, or not positive: the
        # headline runs at WACC, so every cell does too (not at the RONIC,
        # which priced the low-growth columns at 12-17 and blanked the rest).
        base = DCFAssumptions(terminal_method="value_driver")
        headline = run_dcf(self.company, self.macro, base, self.price).implied_price
        self.assertAlmostEqual(headline, 28.09, places=2)
        want = dcf_sensitivity(self.company, self.macro, base, self.price)
        want_bar = next(r for r in _value_driver_report(None).football_field
                        if r.method == "DCF")
        for ronic in (0.02, 0.025, 0.0, -0.05):
            with self.subTest(terminal_ronic=ronic):
                a = replace(base, terminal_ronic=ronic)
                dcf = run_dcf(self.company, self.macro, a, self.price)
                self.assertEqual(dcf.implied_price, headline)
                self.assertEqual(dcf.assumptions["terminal_ronic_used"],
                                 dcf.assumptions["wacc"])
                grids = dcf_sensitivity(self.company, self.macro, a, self.price)
                self.assertEqual(len(grids), 2)
                for got, ref in zip(grids, want):
                    self._assert_same_grid(got, ref)
                    self.assertAlmostEqual(_grid_centre(got), headline, places=6)
                    self.assertTrue(all(math.isfinite(c) for row in got.grid for c in row))
                report = _value_driver_report(ronic)
                self.assertEqual(len(report.sensitivities), 2)
                bar = next(r for r in report.football_field if r.method == "DCF")
                self.assertEqual((bar.low, bar.base, bar.high),
                                 (want_bar.low, want_bar.base, want_bar.high))
                self.assertGreater(bar.low, 20.0)  # not the RONIC-priced 12.42

    def test_replaced_ronic_cells_run_at_wacc(self):
        calls = []
        real = sensitivity.run_dcf

        def spy(company, macro, assumptions, *args, **kwargs):
            calls.append((assumptions.terminal_method, assumptions.terminal_ronic))
            return real(company, macro, assumptions, *args, **kwargs)

        a = DCFAssumptions(terminal_method="value_driver", terminal_ronic=0.02)
        with patch.object(sensitivity, "run_dcf", spy):
            dcf_sensitivity(self.company, self.macro, a, self.price)
        # One headline check at the RONIC given, then every cell at WACC.
        self.assertEqual(calls[0], ("value_driver", 0.02))
        self.assertEqual(set(calls[1:]), {("value_driver", None)})

    def test_blanked_columns_would_otherwise_price_at_wacc(self):
        # What the blanked 3.5% column would show: the DCF replaces RONIC 3%
        # with WACC (growth adds no value), a valuation the headline's RONIC
        # does not give.
        a = DCFAssumptions(terminal_method="value_driver", terminal_ronic=0.03,
                           terminal_growth=0.035)
        dcf = run_dcf(self.company, self.macro, a, self.price)
        self.assertTrue(math.isfinite(dcf.implied_price))
        self.assertEqual(dcf.assumptions["terminal_ronic_used"], dcf.assumptions["wacc"])
        self.assertTrue(any("must exceed terminal growth" in n for n in _notes(dcf)))
        price, _ = sensitivity._safe_implied_price(self.company, self.macro, a, self.price)
        self.assertTrue(math.isnan(price))
        # Without an explicit RONIC the same column keeps its price.
        price, _ = sensitivity._safe_implied_price(
            self.company, self.macro, replace(a, terminal_ronic=None), self.price)
        self.assertTrue(math.isfinite(price))


class TerminalRonicCliTests(unittest.TestCase):
    """--terminal-method value_driver and --terminal-ronic."""

    def test_value_driver_with_a_ronic_runs(self):
        seen = {}
        real = engine.value_company

        def spy(*args, **kwargs):
            seen.update(kwargs)
            return real(*args, **{**kwargs, "run_sensitivity": False})

        out = _tmpdir(self)
        with patch.object(engine, "value_company", spy):
            code, _out, err = _run_cli(["--demo", "--terminal-method", "value_driver",
                                        "--terminal-ronic", "0.15", "--quiet", "--html",
                                        "--out", out])
        self.assertEqual(code, 0, err)
        self.assertEqual(seen["dcf_assumptions"].terminal_method, "value_driver")
        self.assertEqual(seen["dcf_assumptions"].terminal_ronic, 0.15)
        self.assertNotIn("ignored", err)
        self.assertEqual(os.listdir(out), ["SYNT_valuation.html"])

    def test_bad_ronic_is_an_argument_error(self):
        cases = [
            (["--terminal-ronic", "15"], "looks like a percentage"),
            (["--terminal-ronic", "5"], "looks like a percentage"),
            (["--terminal-ronic", "0"], "> 0"),
            (["--terminal-ronic", "-0.1"], "> 0"),
            (["--terminal-ronic", "nan"], "finite"),
            (["--terminal-ronic", "0.02"], "must exceed --terminal-growth"),
            (["--terminal-ronic", "0.03", "--terminal-growth", "0.03"],
             "must exceed --terminal-growth"),
        ]
        out = _tmpdir(self)
        for extra, hint in cases:
            with self.subTest(args=extra):
                code, _out, err = _run_cli(["--demo", "--quiet", "--out", out,
                                            "--terminal-method", "value_driver"] + extra)
                self.assertEqual(code, 2)
                self.assertIn(hint, err)
                self.assertNotIn("Traceback", err)
        self.assertEqual(os.listdir(out), [])

    def test_ronic_without_value_driver_is_noted_and_ignored(self):
        seen = {}
        real = engine.value_company

        def spy(*args, **kwargs):
            seen.update(kwargs)
            return real(*args, **{**kwargs, "run_sensitivity": False})

        # A RONIC at or below the growth is no error when it is ignored, as
        # for --exit-ev-ebitda without exit_multiple.
        for ronic in ("0.15", "0.02"):
            with self.subTest(terminal_ronic=ronic), patch.object(engine, "value_company", spy):
                code, _out, err = _run_cli(["--demo", "--terminal-ronic", ronic, "--quiet",
                                            "--html", "--out", _tmpdir(self)])
                self.assertEqual(code, 0, err)
                self.assertIn("--terminal-ronic is ignored unless --terminal-method "
                              "value_driver", err)
                self.assertNotIn("must exceed", err)
                self.assertEqual(seen["dcf_assumptions"].terminal_method, "gordon")

    def test_ronic_of_one_or_more_is_read_as_a_decimal_with_a_note(self):
        seen = {}
        real = engine.value_company

        def spy(*args, **kwargs):
            seen.update(kwargs)
            return real(*args, **{**kwargs, "run_sensitivity": False})

        cases = [("3", 3.0, "note: --terminal-ronic 3 is read as 300%; use 0.03 for 3%."),
                 ("1", 1.0, "note: --terminal-ronic 1 is read as 100%; use 0.01 for 1%."),
                 ("4.99", 4.99, "is read as 499%; use 0.0499 for 4.99%."),
                 ("0.5", 0.5, None)]
        for text, value, note in cases:
            with self.subTest(terminal_ronic=text), patch.object(engine, "value_company", spy):
                code, _out, err = _run_cli(["--demo", "--terminal-method", "value_driver",
                                            "--terminal-ronic", text, "--quiet", "--html",
                                            "--out", _tmpdir(self)])
                self.assertEqual(code, 0, err)
                self.assertEqual(seen["dcf_assumptions"].terminal_ronic, value)
                if note is None:
                    self.assertNotIn("is read as", err)
                else:
                    self.assertIn(note, err)


if __name__ == "__main__":
    unittest.main()
