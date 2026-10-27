"""Offline tests for the engine's reverse DCF (``equity_valuation.models.reverse_dcf``).

What the market price implies for year-1 revenue growth, the target EBIT
margin and the discount rate, other DCF inputs held, plus the ``run_dcf``
``wacc_override`` keyword it relies on, the engine hook and the CLI block.
Everything runs on the synthetic company, so nothing touches the network.
Run with:  python -m unittest tests.test_reverse_dcf
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import math
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend import valuation_service as vs
from equity_valuation import value_company
from equity_valuation.cli import _print_summary, main
from equity_valuation.data.synthetic import (
    DEMO_PEERS,
    DEMO_TICKER,
    SyntheticProvider,
    make_company,
)
from equity_valuation.models import reverse_dcf as rd
from equity_valuation.models.dcf import run_dcf
from equity_valuation.models.reverse_dcf import growth_path_from_y1, solve_reverse_dcf
from equity_valuation.schemas import DCFAssumptions, MacroAssumptions
from equity_valuation.utils import fade_path

# The demo's solved values. The growth figure was produced by the dashboard's
# own solver before it moved into the engine; the margin and discount rate
# were pinned when they were added.
DEMO_IMPLIED_GROWTH = 0.17470256313681604
DEMO_IMPLIED_MARGIN = 0.3084244632720947
DEMO_IMPLIED_WACC = 0.08236103469729422


class _FlaggedProvider(SyntheticProvider):
    """The synthetic company flagged as a bank by its industry, or as a
    captive-finance group or lessor by the data layer's kind."""

    def __init__(self, industry=None, kind=None):
        super().__init__()
        self.industry, self.kind = industry, kind

    def get_company_data(self, ticker):
        company = make_company()
        if self.industry:
            company.market.sector = "Financial Services"
            company.market.industry = self.industry
        if self.kind:
            company.financials._financial_kind = self.kind
        return company


class _ZeroSharesProvider(SyntheticProvider):
    def get_company_data(self, ticker):
        company = make_company()
        company.market = dataclasses.replace(company.market, shares_outstanding=0.0)
        company.financials = dataclasses.replace(
            company.financials, diluted_shares=[0.0] * len(company.financials.diluted_shares))
        return company


def _demo(**kwargs):
    kwargs.setdefault("peers", DEMO_PEERS)
    return value_company(DEMO_TICKER, provider=kwargs.pop("provider", SyntheticProvider()),
                         **kwargs)


def _held(report, **changes):
    """The report's DCF inputs with the resolved growth path and target
    margin made explicit, as the margin and discount-rate legs hold them."""
    a = report.dcf.assumptions
    base = dataclasses.replace(DCFAssumptions(), revenue_growth=list(a["revenue_growth_path"]),
                               target_ebit_margin=a["target_ebit_margin"])
    return dataclasses.replace(base, **changes)


def _price(report, assumptions, **kwargs):
    return run_dcf(report.company, report.macro, assumptions, report.current_price,
                   **kwargs).implied_price


class DemoReverseDCFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = _demo()
        cls.r = cls.report.reverse_dcf
        cls.price = cls.report.current_price

    def test_growth_leg_matches_the_dashboard_solver_and_reprices(self):
        r = self.r
        self.assertTrue(r["converged"])
        self.assertAlmostEqual(r["implied_growth_y1"], DEMO_IMPLIED_GROWTH, delta=1e-9)
        self.assertEqual(r["current_assumption_y1"],
                         self.report.dcf.assumptions["revenue_growth_path"][0])
        self.assertNotIn("note", r)
        a = dataclasses.replace(DCFAssumptions(), revenue_growth=r["implied_revenue_growth"])
        self.assertAlmostEqual(_price(self.report, a), self.price, delta=self.price * 1e-6)
        # The shifted path fades the year-1 delta to zero by the final year.
        self.assertAlmostEqual(r["implied_revenue_growth"][-1],
                               self.report.dcf.assumptions["revenue_growth_path"][-1], places=12)
        # The dashboard's wrapper gives the same growth keys exactly.
        macro, dcf_a, *_ = vs.parse_assumptions({})
        backend = vs._reverse_dcf(self.report, dcf_a, macro)
        for key in ("current_assumption_y1", "converged", "implied_growth_y1",
                    "implied_revenue_growth"):
            self.assertEqual(backend[key], r[key], key)
        self.assertEqual(backend, r)

    def test_margin_leg_converges_and_reprices(self):
        m = self.r["margin"]
        self.assertTrue(m["converged"])
        self.assertIsNone(m["note"])
        self.assertAlmostEqual(m["implied_target_ebit_margin"], DEMO_IMPLIED_MARGIN, delta=1e-6)
        self.assertTrue(0.30 < m["implied_target_ebit_margin"] < 0.32)
        self.assertEqual(m["current_target_ebit_margin"],
                         self.report.dcf.assumptions["target_ebit_margin"])
        self.assertEqual(m["start_ebit_margin"], self.report.dcf.assumptions["start_ebit_margin"])
        a = _held(self.report, target_ebit_margin=m["implied_target_ebit_margin"])
        self.assertAlmostEqual(_price(self.report, a), self.price, delta=self.price * 1e-6)

    def test_wacc_leg_converges_and_reprices(self):
        w = self.r["wacc"]
        self.assertTrue(w["converged"])
        self.assertIsNone(w["note"])
        self.assertAlmostEqual(w["implied_wacc"], DEMO_IMPLIED_WACC, delta=1e-6)
        self.assertTrue(0.075 < w["implied_wacc"] < 0.09)
        self.assertEqual(w["current_wacc"], self.report.dcf.wacc.wacc)
        self.assertAlmostEqual(_price(self.report, _held(self.report),
                                      wacc_override=w["implied_wacc"]),
                               self.price, delta=self.price * 1e-6)

    def test_held_inputs_reproduce_the_forward_run(self):
        # Holding growth and margin explicitly at their resolved values, and
        # the WACC at the computed one, gives the DCF's own price: the legs
        # move only the driver they solve for.
        dcf_price = self.report.dcf.implied_price
        self.assertAlmostEqual(_price(self.report, _held(self.report)), dcf_price, places=9)
        self.assertAlmostEqual(_price(self.report, _held(self.report),
                                      wacc_override=self.report.dcf.wacc.wacc),
                               dcf_price, places=9)

    def test_price_rises_strictly_with_the_target_margin(self):
        lo, hi, step = rd.MARGIN_RANGE
        prices = [_price(self.report, _held(self.report, target_ebit_margin=m))
                  for m in rd._grid(lo, hi, step)]
        self.assertEqual(len(prices), 51)
        self.assertTrue(all(b > a for a, b in zip(prices, prices[1:])))

    def test_reverse_dcf_leaves_the_pinned_demo_untouched(self):
        off = _demo(run_reverse_dcf=False)
        self.assertIsNone(off.reverse_dcf)
        self.assertEqual(off.summary, self.report.summary)
        self.assertEqual(off.warnings, self.report.warnings)
        s = self.report.summary
        self.assertAlmostEqual(s["methods"]["DCF"], 33.38, delta=0.005)
        self.assertAlmostEqual(s["blended_target"], 32.40, delta=0.005)

    def test_serialized_report_carries_every_leg(self):
        from backend.serialization import report_to_dict

        d = report_to_dict(self.report)
        json.dumps(d, allow_nan=False)
        self.assertEqual(d["reverse_dcf"]["margin"]["implied_target_ebit_margin"],
                         self.r["margin"]["implied_target_ebit_margin"])
        with patch.object(vs, "_get_provider", lambda t, refresh=False: SyntheticProvider()):
            api = vs.run_valuation(DEMO_TICKER, {"peers": ",".join(DEMO_PEERS)})
        self.assertEqual(api["reverse_dcf"], self.r)


class WaccOverrideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.company = make_company()
        cls.macro = MacroAssumptions()
        cls.a = DCFAssumptions()
        cls.price = cls.company.market.price
        cls.base = run_dcf(cls.company, cls.macro, cls.a, cls.price)

    def test_no_override_is_todays_dcf(self):
        same = run_dcf(self.company, self.macro, self.a, self.price, wacc_override=None)
        self.assertEqual(same, self.base)
        self.assertAlmostEqual(self.base.implied_price, 33.38, delta=0.005)
        self.assertNotIn("wacc_override", self.base.wacc.detail)

    def test_override_discounts_at_the_given_rate(self):
        r = run_dcf(self.company, self.macro, self.a, self.price, wacc_override=0.08)
        self.assertEqual(r.wacc.wacc, 0.08)
        self.assertEqual(r.wacc.detail["wacc"], 0.08)
        self.assertEqual(r.wacc.detail["wacc_computed"], self.base.wacc.wacc)
        self.assertIs(r.wacc.detail["wacc_override"], True)
        self.assertEqual(r.assumptions["wacc"], 0.08)
        self.assertAlmostEqual(r.discount_factors[0], 1 / 1.08 ** 0.5, places=12)
        self.assertGreater(r.implied_price, self.base.implied_price)
        # The rest of the WACC breakdown is the computed one.
        self.assertEqual(r.wacc.cost_of_equity, self.base.wacc.cost_of_equity)

    def test_override_too_close_to_terminal_growth_is_clamped_as_usual(self):
        r = run_dcf(self.company, self.macro, self.a, self.price, wacc_override=0.03)
        self.assertAlmostEqual(r.assumptions["terminal_growth_used"], 0.02, places=12)
        self.assertTrue(any("clamped" in n for n in r.assumptions["notes"]))

    def test_invalid_override_is_ignored_with_a_note(self):
        for bad in (0.0, -0.05, float("nan"), float("inf"), True):
            with self.subTest(override=bad):
                r = run_dcf(self.company, self.macro, self.a, self.price, wacc_override=bad)
                self.assertEqual(r.implied_price, self.base.implied_price)
                self.assertEqual(r.wacc.wacc, self.base.wacc.wacc)
                self.assertTrue(any(n.startswith("wacc_override") and "ignored" in n
                                    for n in r.assumptions["notes"]))

    def test_override_is_keyword_only(self):
        with self.assertRaises(TypeError):
            run_dcf(self.company, self.macro, self.a, self.price, 0.08)


class EngineHookTests(unittest.TestCase):
    def test_skipped_on_request_or_without_an_available_dcf(self):
        self.assertIsNone(_demo(run_reverse_dcf=False).reverse_dcf)
        self.assertIsNone(_demo(run_dcf=False).reverse_dcf)
        report = _demo(provider=_ZeroSharesProvider(), run_comps=False)
        self.assertFalse(report.dcf.assumptions["valuation_available"])
        self.assertIsNone(report.reverse_dcf)

    def test_a_failing_solver_becomes_a_warning(self):
        with patch("equity_valuation.models.reverse_dcf.solve_reverse_dcf",
                   side_effect=RuntimeError("boom")):
            report = _demo(run_comps=False)
        self.assertIsNone(report.reverse_dcf)
        self.assertIn("Reverse DCF failed: boom", report.warnings)
        self.assertAlmostEqual(report.dcf.implied_price, 33.38, delta=0.005)


class FlaggedCompanyTests(unittest.TestCase):
    def test_bank_is_not_solved(self):
        report = _demo(provider=_FlaggedProvider(industry="Banks - Diversified"))
        self.assertEqual(set(report.summary["excluded_from_blend"]), {"DCF", "FCFE"})
        r = report.reverse_dcf
        self.assertFalse(r["converged"])
        self.assertIsNone(r["implied_growth_y1"])
        self.assertIsNone(r["margin"])
        self.assertIsNone(r["wacc"])
        self.assertEqual(r["note"], "Not meaningful for a financial institution: the FCFF DCF "
                         "is reference-only.")
        self.assertEqual(r["current_assumption_y1"],
                         report.dcf.assumptions["revenue_growth_path"][0])

    def test_wording_follows_the_kind(self):
        for kind, whom in (("captive_finance", "a company with a consolidated captive "
                                               "finance arm"),
                           ("lessor", "a debt-funded lessor")):
            with self.subTest(kind=kind):
                r = _demo(provider=_FlaggedProvider(kind=kind)).reverse_dcf
                self.assertEqual(r["note"],
                                 f"Not meaningful for {whom}: the FCFF DCF is reference-only.")

    def test_dashboard_still_inverts_the_reference_only_dcf(self):
        report = _demo(provider=_FlaggedProvider(industry="Banks - Diversified"))
        macro, dcf_a, *_ = vs.parse_assumptions({})
        api = vs._reverse_dcf(report, dcf_a, macro)
        self.assertTrue(api["converged"])
        direct = solve_reverse_dcf(report.company, macro, dcf_a, report.current_price,
                                   report.dcf.assumptions, include_reference_only=True)
        self.assertEqual(api, direct)


class SolverEdgeTests(unittest.TestCase):
    def test_growth_path_from_y1(self):
        self.assertEqual(growth_path_from_y1(0.15, 0.025, 1), [0.15])
        self.assertIsNone(growth_path_from_y1(None, 0.025, 5))
        self.assertIsNone(growth_path_from_y1(0.15, 0.025, 0))
        for n in range(2, 16):
            self.assertEqual(growth_path_from_y1(0.15, 0.025, n), fade_path(0.15, 0.025, n))
        self.assertIs(vs._revenue_growth_path, growth_path_from_y1)

    def test_grid_covers_each_range(self):
        self.assertEqual(len(rd._grid(*rd.GROWTH_RANGE)), 25)
        g = rd._grid(0.0351, rd.WACC_MAX, rd.WACC_STEP)
        self.assertEqual(g[0], 0.0351)
        self.assertEqual(g[-1], rd.WACC_MAX)
        self.assertTrue(all(b > a for a, b in zip(g, g[1:])))

    def test_notes_name_the_driver(self):
        several = rd._solve(lambda x: 50.0 - 400.0 * (x - 0.3) ** 2, 40.0,
                            *rd.MARGIN_RANGE, rd._MARGIN)
        self.assertTrue(several["converged"])
        self.assertAlmostEqual(several["value"], 0.3 - (10 / 400) ** 0.5, delta=1e-7)
        self.assertIn("More than one target EBIT margin", several["note"])
        flat = rd._solve(lambda x: 40.0, 40.0, 0.04, 0.25, 0.0025, rd._WACC)
        self.assertFalse(flat["converged"])
        self.assertIn("the discount rate cannot be uniquely inferred", flat["note"])
        outside = rd._solve(lambda x: 10.0 - x, 40.0, 0.04, 0.25, 0.0025, rd._WACC)
        self.assertIn("solvable discount-rate range (4.00% to 25.00%)", outside["note"])
        jump = rd._solve(lambda x: 30.0 if x < 0.1 else 50.0, 40.0, 0.04, 0.25, 0.0025,
                         rd._WACC)
        self.assertFalse(jump["converged"])
        self.assertIn("no discount rate reproduces it", jump["note"])
        self.assertIsNone(rd._solve(lambda x: None, 40.0, 0.04, 0.25, 0.0025, rd._WACC))

    def test_invalid_price_skips_every_leg(self):
        report = _demo(run_comps=False)
        for price in (None, 0.0, -1.0, float("nan")):
            with self.subTest(price=price):
                r = solve_reverse_dcf(report.company, report.macro, DCFAssumptions(), price,
                                      report.dcf.assumptions)
                self.assertFalse(r["converged"])
                self.assertEqual(r["note"], "No valid market price to solve against.")
                self.assertIsNone(r["margin"])
                self.assertIsNone(r["wacc"])

    def test_each_leg_degrades_on_its_own(self):
        # A run_dcf without the wacc_override keyword (as the dashboard tests'
        # stand-in) leaves only the discount-rate leg empty.
        report = _demo(run_comps=False)

        def four_args(company, macro, a, price):
            return run_dcf(company, macro, a, price)

        with patch("equity_valuation.models.dcf.run_dcf", four_args):
            r = solve_reverse_dcf(report.company, report.macro, DCFAssumptions(),
                                  report.current_price, report.dcf.assumptions)
        self.assertAlmostEqual(r["implied_growth_y1"], DEMO_IMPLIED_GROWTH, delta=1e-9)
        self.assertTrue(r["margin"]["converged"])
        self.assertIsNone(r["wacc"])

        def broken(*args, **kwargs):
            raise RuntimeError("model down")

        with patch("equity_valuation.models.dcf.run_dcf", broken):
            self.assertIsNone(solve_reverse_dcf(report.company, report.macro, DCFAssumptions(),
                                                report.current_price, report.dcf.assumptions))
        # Garbage inputs never raise.
        self.assertIsNone(solve_reverse_dcf(None, None, None, 40.0, {"valuation_available": False}))
        self.assertIsNone(solve_reverse_dcf(object(), None, None, 40.0,
                                            {"revenue_growth_path": "x"}))

    def test_clamped_terminal_growth_is_held_in_the_wacc_leg(self):
        # Terminal growth 20% is clamped to WACC - 1pp in the forward run; the
        # discount-rate leg keeps that clamped rate rather than the input.
        report = _demo(dcf_assumptions=DCFAssumptions(terminal_growth=0.2), run_comps=False)
        used = report.dcf.assumptions["terminal_growth_used"]
        self.assertLess(used, 0.2)
        w = report.reverse_dcf["wacc"]
        self.assertTrue(w["converged"])
        self.assertGreater(w["implied_wacc"], w["current_wacc"])  # model price > market
        held = _held(report, terminal_growth=used)
        self.assertAlmostEqual(_price(report, held, wacc_override=w["implied_wacc"]),
                               report.current_price, delta=report.current_price * 1e-6)

    def test_exit_multiple_and_one_year_runs_solve(self):
        for a in (DCFAssumptions(terminal_method="exit_multiple", exit_ev_ebitda=14.0),
                  DCFAssumptions(forecast_years=1, revenue_growth=[0.15])):
            with self.subTest(assumptions=a):
                report = _demo(dcf_assumptions=a, run_comps=False)
                r = report.reverse_dcf
                for leg, key in ((r, "implied_growth_y1"),
                                 (r["margin"], "implied_target_ebit_margin"),
                                 (r["wacc"], "implied_wacc")):
                    self.assertTrue(leg["converged"], key)
                    self.assertTrue(math.isfinite(leg[key]), key)
                w = r["wacc"]["implied_wacc"]
                self.assertAlmostEqual(_price(report, _held(report, terminal_method=a.terminal_method,
                                                            exit_ev_ebitda=a.exit_ev_ebitda,
                                                            forecast_years=a.forecast_years),
                                              wacc_override=w),
                                       report.current_price, delta=report.current_price * 1e-6)

    def test_no_room_below_the_wacc_ceiling(self):
        report = _demo(dcf_assumptions=DCFAssumptions(terminal_growth=0.245), run_comps=False)
        resolved = dict(report.dcf.assumptions, terminal_growth_used=0.245)
        w = rd._wacc_leg(report.company, report.macro, DCFAssumptions(terminal_growth=0.245),
                         report.current_price, resolved)
        self.assertFalse(w["converged"])
        self.assertIn("No discount rate to test", w["note"])


class CliReverseDCFTests(unittest.TestCase):
    @staticmethod
    def _summary(report) -> str:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _print_summary(report)
        return out.getvalue()

    def test_demo_prints_the_three_implied_drivers(self):
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(out):
            self.assertEqual(main(["--demo", "--out", tmp]), 0)
        lines = out.getvalue().splitlines()
        start = lines.index("  Reverse DCF (what the price implies, other inputs held):")
        # The block follows the blended-target table and precedes the notes.
        self.assertTrue(lines[start - 2].lstrip().startswith("Blended target"))
        self.assertTrue(lines[start - 1].startswith("====="))
        self.assertEqual(lines[start + 1:start + 5], [
            "    Year-1 revenue growth      17.5%   (model 8.0%)",
            "    Target EBIT margin         30.8%   (model 25.0%)",
            "    Discount rate (WACC)        8.2%   (model 9.5%)",
            "=" * 64,
        ])
        self.assertEqual(lines[start + 5], "  Notes:")

    def test_bank_prints_one_line(self):
        text = self._summary(_demo(provider=_FlaggedProvider(industry="Banks - Diversified")))
        block = text.split("Reverse DCF (what the price implies, other inputs held):\n")[1]
        self.assertTrue(block.startswith(
            "    Not meaningful for a financial institution: the FCFF DCF is reference-only.\n"
            + "=" * 64))

    def test_unsolved_driver_prints_na_and_its_note(self):
        report = _demo(dcf_assumptions=DCFAssumptions(terminal_growth=0.2), run_comps=False)
        text = self._summary(report)
        self.assertIn("    Year-1 revenue growth        n/a   (model 8.0%)\n"
                      "      Market price is outside the solvable growth range (-40% to 80%)",
                      text)
        report.reverse_dcf = dict(report.reverse_dcf, wacc=None)
        self.assertIn("    Discount rate (WACC)         n/a\n", self._summary(report))

    def test_no_block_without_a_reverse_dcf(self):
        self.assertNotIn("Reverse DCF", self._summary(_demo(run_dcf=False)))
        summary = _demo(run_comps=False).summary
        self.assertNotIn("Reverse DCF", self._summary(SimpleNamespace(summary=summary,
                                                                      warnings=[])))


if __name__ == "__main__":
    unittest.main()
