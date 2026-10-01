"""Regression tests for model-layer audit fixes (WACC, DCF, DDM/FCFE,
sensitivity, engine blend and warnings).

Each test perturbs the synthetic company (``equity_valuation.data.synthetic``)
in the way a provider gap or an edge-case input would, so nothing touches the
network. Run with:  python -m unittest tests.test_model_fixes
"""

from __future__ import annotations

from dataclasses import replace
import contextlib
import io
import math
from types import SimpleNamespace
import unittest

from equity_valuation import config, value_company
from equity_valuation.cli import _print_summary
from equity_valuation.data.synthetic import DEMO_PEERS, SyntheticProvider, make_company
from equity_valuation.engine import (
    _add_notes,
    _build_summary,
    _fallback_football_field,
    _recommendation,
)
from equity_valuation.models.comps import EARNINGS_COLLAPSE_NOTE_PREFIX, _eps_cagr, run_comps
from equity_valuation.models.dcf import margin_fade_target, run_dcf
from equity_valuation.models.ddm_fcfe import (
    _dividend_cagr,
    _sustainable_growth,
    ebit_charge_collapse,
    net_margin_collapse,
    run_ddm,
    run_fcfe,
)
from equity_valuation.models.sensitivity import (
    _base_latest_ebit_margin,
    build_football_field,
    dcf_sensitivity,
)
from equity_valuation.models.wacc import (
    MIN_DEBT_SPREAD,
    beta_adjustment,
    compute_wacc,
    effective_tax_rate_detail,
)
from equity_valuation.schemas import (
    AnnualFinancials,
    CompRow,
    DCFAssumptions,
    DDMAssumptions,
    MacroAssumptions,
    ValuationReport,
)
from equity_valuation.utils import (
    LOW_DDM_PAYOUT_SHARE,
    MARGIN_COLLAPSE_SHARE,
    REFERENCE_ONLY_KINDS,
    charge_year_margin,
    collapsed_margin,
    ddm_left_alone,
    ddm_payout,
    ddm_reference_only,
    financial_institution,
    financial_institution_detail,
    first_positive_margin,
    growth_capex_path,
    incremental_ratio,
    low_payout_ddm,
    material_capex_fade,
    median,
    pooled_ratio,
    robust_latest_margin,
    screened_incremental_ratio,
    series_cagr,
    trim_outliers,
)

from tests.test_synthetic import make_distressed_company


class _OneCompanyProvider(SyntheticProvider):
    """SyntheticProvider that serves a caller-supplied (perturbed) company."""

    def __init__(self, company):
        self._company = company

    def get_company_data(self, ticker):
        return self._company


def _with_fin(company, **changes):
    return replace(company, financials=replace(company.financials, **changes))


def _with_market(company, **changes):
    return replace(company, market=replace(company.market, **changes))


def _with_bs(company, **changes):
    return replace(company, balance_sheet=replace(company.balance_sheet, **changes))


def _company_from_revenue(revenue, **changes):
    """The synthetic company rebuilt on another revenue history (any length):
    every other line keeps make_company's ratio to revenue unless overridden."""
    base = make_company()
    n = len(revenue)
    ebit = [r * 0.25 for r in revenue]
    pretax = [e * 0.95 for e in ebit]
    tax = [p * 0.21 for p in pretax]
    fin = AnnualFinancials(
        fiscal_years=list(range(2025 - n + 1, 2026)),
        revenue=list(revenue),
        ebit=ebit,
        ebitda=[r * 0.29 for r in revenue],
        net_income=[p - t for p, t in zip(pretax, tax)],
        dep_amort=[r * 0.04 for r in revenue],
        capex=[r * 0.05 for r in revenue],
        change_in_nwc=[r * 0.01 for r in revenue],
        interest_expense=[e * 0.05 for e in ebit],
        tax_expense=tax,
        pretax_income=pretax,
        dividends_paid=[(p - t) * 0.3 for p, t in zip(pretax, tax)],
        diluted_shares=[base.financials.diluted_shares[-1]] * n,
    )
    return replace(base, financials=replace(fin, **changes))


class WACCMarketCapFallbackTests(unittest.TestCase):
    def setUp(self):
        self.company = make_company()
        self.macro = MacroAssumptions()
        self.base = compute_wacc(self.company, self.macro)

    def test_zero_market_cap_uses_price_times_shares(self):
        w = compute_wacc(_with_market(self.company, market_cap=0.0), self.macro)
        self.assertAlmostEqual(w.wacc, self.base.wacc, places=12)
        self.assertIn("market_cap unavailable; using price x shares_outstanding", w.detail["notes"])

    def test_zero_market_cap_and_shares_uses_diluted_shares(self):
        c = _with_market(self.company, market_cap=0.0, shares_outstanding=0.0)
        w = compute_wacc(c, self.macro)
        self.assertAlmostEqual(w.wacc, self.base.wacc, places=12)
        self.assertIn("market_cap unavailable; using price x latest diluted_shares", w.detail["notes"])
        # The DCF no longer inflates on the degraded market data.
        dcf = run_dcf(c, self.macro, DCFAssumptions(), c.market.price)
        base = run_dcf(self.company, self.macro, DCFAssumptions(), c.market.price)
        self.assertAlmostEqual(dcf.implied_price, base.implied_price, places=9)

    def test_no_equity_value_defaults_to_all_equity_not_all_debt(self):
        c = _with_market(self.company, market_cap=None, shares_outstanding=0.0)
        c = _with_fin(c, diluted_shares=[0.0] * 5)
        w = compute_wacc(c, self.macro)
        self.assertEqual((w.weight_equity, w.weight_debt), (1.0, 0.0))
        self.assertAlmostEqual(w.wacc, w.cost_of_equity, places=12)
        self.assertTrue(any("all-equity" in n for n in w.detail["notes"]))

    def test_debt_free_company_has_no_cost_of_debt_note(self):
        w = compute_wacc(_with_bs(self.company, total_debt=0.0), self.macro)
        self.assertEqual(w.weight_debt, 0.0)
        self.assertEqual(w.detail["notes"], [])


class DCFInputGapTests(unittest.TestCase):
    def setUp(self):
        self.company = make_company()
        self.macro = MacroAssumptions()
        self.price = self.company.market.price
        self.base = run_dcf(self.company, self.macro, DCFAssumptions(), self.price)

    def test_no_positive_revenue_fails_instead_of_negative_price(self):
        c = _with_fin(self.company, revenue=[0.0] * 5)
        with self.assertRaises(ValueError):
            run_dcf(c, self.macro, DCFAssumptions(), self.price)
        report = value_company("SYNT", provider=_OneCompanyProvider(c), peers=DEMO_PEERS)
        self.assertIsNone(report.dcf)
        self.assertNotIn("DCF", report.summary["methods"])
        self.assertTrue(any(w.startswith("DCF failed:") for w in report.warnings))

    def test_zero_filled_oldest_revenue_keeps_history_cagr(self):
        rev = list(self.company.financials.revenue)
        c = _with_fin(self.company, revenue=[0.0] + rev[1:])
        dcf = run_dcf(c, self.macro, DCFAssumptions(), self.price)
        fcfe = run_fcfe(c, self.macro, DDMAssumptions(), self.price)
        self.assertAlmostEqual(dcf.assumptions["revenue_growth_path"][0], 0.08, places=9)
        self.assertAlmostEqual(fcfe.detail["revenue_growth_path"][0], 0.08, places=9)

    def test_interior_gap_counts_the_full_period(self):
        rev = list(self.company.financials.revenue)
        c = _with_fin(self.company, revenue=[rev[0], None, rev[2], None, rev[4]])
        dcf = run_dcf(c, self.macro, DCFAssumptions(), self.price)
        self.assertAlmostEqual(dcf.assumptions["revenue_growth_path"][0], 0.08, places=9)
        # Fiscal years take precedence over positions when they are aligned.
        self.assertAlmostEqual(series_cagr([100.0, 0.0, 121.0], [2020, 2021, 2022]), 0.10)
        self.assertAlmostEqual(series_cagr([100.0, 121.0], [2019, 2021]), 0.10)

    def test_nwc_ratio_is_pooled_not_a_mean_of_ratios(self):
        rev = [100e9, 110e9, 110.5e9, 121e9, 133e9]
        dnwc = [0.0, 1.4e9, 0.9e9, 1.5e9, 1.7e9]
        c = _with_fin(self.company, revenue=rev, change_in_nwc=dnwc)
        dcf = run_dcf(c, self.macro, DCFAssumptions(), self.price)
        pooled = sum(dnwc[1:]) / (rev[-1] - rev[0])
        self.assertAlmostEqual(dcf.assumptions["nwc_pct_revenue"], pooled, places=12)

    def test_zero_filled_ebit_rebuilt_from_pretax_plus_interest(self):
        c = _with_fin(self.company, ebit=[0.0] * 5)
        dcf = run_dcf(c, self.macro, DCFAssumptions(), self.price)
        # Synthetic pretax + interest reproduces EBIT exactly.
        self.assertAlmostEqual(dcf.implied_price, self.base.implied_price, places=6)
        self.assertTrue(any("pretax income + interest" in n for n in dcf.assumptions["notes"]))

    def test_zero_effective_tax_rate_is_kept_but_flagged(self):
        c = _with_fin(self.company, tax_expense=[0.0] * 5)
        dcf = run_dcf(c, self.macro, DCFAssumptions(), self.price)
        self.assertEqual(dcf.assumptions["tax_rate"], 0.0)
        self.assertTrue(any("effective tax rate is 0%" in n for n in dcf.assumptions["notes"]))

    def test_zero_filled_capex_is_flagged(self):
        # With D&A zero-filled too there is nothing to add back: capex stays 0
        # (a zero-filled capex with D&A is covered by MaintenanceCapexTests).
        c = _with_fin(self.company, capex=[0.0] * 5, dep_amort=[0.0] * 5)
        dcf = run_dcf(c, self.macro, DCFAssumptions(), self.price)
        self.assertEqual(dcf.assumptions["capex_pct_revenue"], 0.0)
        self.assertIn("capex %revenue unavailable; defaulting to 0", dcf.assumptions["notes"])

    def test_terminal_method_is_normalised_or_falls_back_to_gordon(self):
        dcf = run_dcf(self.company, self.macro, DCFAssumptions(terminal_method=" Gordon "), self.price)
        self.assertEqual(dcf.implied_price, self.base.implied_price)
        self.assertEqual(dcf.assumptions["terminal_method"], "gordon")
        dcf = run_dcf(self.company, self.macro, DCFAssumptions(terminal_method="perpetuity"), self.price)
        self.assertEqual(dcf.implied_price, self.base.implied_price)
        self.assertTrue(any("unknown terminal_method" in n for n in dcf.assumptions["notes"]))
        exit_a = DCFAssumptions(terminal_method="Exit_Multiple", exit_ev_ebitda=12.0)
        dcf = run_dcf(self.company, self.macro, exit_a, self.price)
        self.assertEqual(dcf.assumptions["terminal_method"], "exit_multiple")

    def test_wacc_fallback_is_reported_on_the_result(self):
        c = _with_market(self.company, beta=-2.0)
        dcf = run_dcf(c, self.macro, DCFAssumptions(), self.price)
        self.assertGreater(dcf.wacc.wacc, 0)
        self.assertEqual(dcf.wacc.wacc, dcf.assumptions["wacc"])
        self.assertEqual(dcf.wacc.detail["wacc"], dcf.assumptions["wacc"])
        self.assertLess(dcf.wacc.detail["wacc_computed"], 0)

    def test_missing_cash_keeps_known_debt_and_none_debt_does_not_crash(self):
        c = _with_bs(self.company, cash_and_investments=float("nan"))
        dcf = run_dcf(c, self.macro, DCFAssumptions(), self.price)
        self.assertEqual(dcf.net_debt, self.company.balance_sheet.total_debt)
        c = _with_bs(self.company, total_debt=None)
        dcf = run_dcf(c, self.macro, DCFAssumptions(), self.price)
        self.assertEqual(dcf.net_debt, -self.company.balance_sheet.cash_and_investments)


class DDMFixTests(unittest.TestCase):
    def setUp(self):
        self.company = make_company()
        self.macro = MacroAssumptions()
        self.price = self.company.market.price
        self.base = run_ddm(self.company, self.macro, DDMAssumptions(), self.price)

    def test_negative_book_equity_skips_roe_growth(self):
        for equity in (-3.8e9, -20e9, 0.0):
            c = _with_bs(self.company, total_equity=equity)
            ddm = run_ddm(c, self.macro, DDMAssumptions(), self.price)
            # Falls back to the 8% dividend CAGR, same as with positive equity.
            self.assertAlmostEqual(ddm.detail["high_growth"], 0.08, places=9)
            self.assertAlmostEqual(ddm.implied_price, self.base.implied_price, places=9)
            self.assertTrue(any("Book equity" in n for n in ddm.detail["notes"]))

    def test_thin_book_equity_roe_is_not_used(self):
        c = _with_bs(self.company, total_equity=1e9)  # ROE ~2000%
        notes: list[str] = []
        self.assertIsNone(_sustainable_growth(c.financials, c, notes))
        self.assertTrue(any("not meaningful" in n for n in notes))

    def test_dividend_cagr_uses_the_paying_run_after_a_gap(self):
        # Zero years between paying years break the run (a suspension looks
        # the same as a zero-filled gap); the first year back may be partial,
        # so the CAGR covers the paying years after it, by fiscal year.
        d = self.company.financials.dividends_paid[0]
        fin = replace(self.company.financials, dividends_paid=[d, 0.0, 0.0, 0.0, 1.2 * d])
        notes: list[str] = []
        self.assertIsNone(_dividend_cagr(fin, notes))
        self.assertTrue(any("zero or unreported in FY2023" in n and "no dividend CAGR" in n
                            for n in notes))
        fin = _company_from_revenue([100e9] * 8).financials
        fin = replace(fin, dividends_paid=[d, 0.0, 0.5 * d] + [d * 1.1 ** i for i in range(5)])
        notes = []
        self.assertAlmostEqual(_dividend_cagr(fin, notes), 0.10, places=12)
        self.assertTrue(any("the 5 paying years since (FY2021-FY2025)" in n for n in notes))

    def test_zero_filled_dividends_paid_uses_dps_payout(self):
        c = _with_fin(self.company, dividends_paid=[0.0] * 5)
        notes: list[str] = []
        sg = _sustainable_growth(c.financials, c, notes)
        expected = _sustainable_growth(self.company.financials, self.company)  # 30% payout
        self.assertAlmostEqual(sg, expected, places=12)
        self.assertTrue(any("DPS x shares" in n for n in notes))

    def test_float_horizons_are_accepted(self):
        a = DDMAssumptions(high_growth_years=5.0, forecast_years=5.0)
        self.assertEqual(run_ddm(self.company, self.macro, a, self.price).implied_price,
                         self.base.implied_price)
        self.assertEqual(run_fcfe(self.company, self.macro, a, self.price).implied_price,
                         run_fcfe(self.company, self.macro, DDMAssumptions(), self.price).implied_price)

    def test_dividend_growth_tracks_each_share_not_total_payout(self):
        fin = replace(self.company.financials, dividends_paid=[100.0] * 5,
                      diluted_shares=[100.0 * 0.9 ** i for i in range(5)])
        self.assertAlmostEqual(_dividend_cagr(fin), 1.0 / 0.9 - 1.0)
        # Issuing more shares at an unchanged dividend per share is not a
        # dividend restart or step-change in the investor's cash distribution.
        fin = replace(fin, dividends_paid=[10, 10, 30, 30, 30],
                      diluted_shares=[100, 100, 300, 300, 300])
        self.assertEqual(_dividend_cagr(fin), 0.0)

    def test_missing_dividend_share_history_discloses_proxy(self):
        fin = replace(self.company.financials, diluted_shares=[])
        notes = []
        self.assertAlmostEqual(_dividend_cagr(fin, notes), 0.08)
        self.assertTrue(any("proxy for per-share growth" in n for n in notes))


class FCFEFixTests(unittest.TestCase):
    def setUp(self):
        self.company = make_company()
        self.macro = MacroAssumptions()
        self.price = self.company.market.price

    def test_implausible_nwc_ratio_is_zeroed_like_the_dcf(self):
        rev = [100e9, 100.05e9, 100.1e9, 99.9e9, 100.2e9]
        c = _with_fin(self.company, revenue=rev, change_in_nwc=[1e9] * 5)
        fcfe = run_fcfe(c, self.macro, DDMAssumptions(), self.price)
        dcf = run_dcf(c, self.macro, DCFAssumptions(), self.price)
        self.assertEqual(fcfe.detail["nwc_pct_delta_revenue"], 0.0)
        self.assertEqual(dcf.assumptions["nwc_pct_revenue"], 0.0)
        self.assertTrue(any("implausible" in n for n in fcfe.detail["notes"]))

    def test_net_borrowing_holds_debt_to_revenue_constant(self):
        fcfe = run_fcfe(self.company, self.macro, DDMAssumptions(), self.price)
        d = fcfe.detail
        debt = self.company.balance_sheet.total_debt
        ratio = debt / self.company.financials.revenue[-1]
        prev_rev = self.company.financials.revenue[-1]
        for rev_t, fcfe_t in zip(d["revenue"], fcfe.fcfe):
            d_rev = rev_t - prev_rev
            operating = (d["net_margin"] + d["da_pct_revenue"] - d["capex_pct_revenue"]) * rev_t \
                - d["nwc_pct_delta_revenue"] * d_rev
            debt += fcfe_t - operating  # ΔDebt_t
            self.assertAlmostEqual(debt / rev_t, ratio, places=9)
            prev_rev = rev_t

    def test_common_earnings_are_not_rebuilt_from_consolidated_profit(self):
        fin = _company_from_revenue([100.0, 100.0], net_income=[1.0, 1.0],
                                    pretax_income=[10.0, 10.0],
                                    tax_expense=[2.0, 2.0]).financials
        fin._income_attribution_adjusted = True
        c = replace(self.company, financials=fin)
        result = run_fcfe(c, self.macro, DDMAssumptions(), self.price)
        self.assertEqual(result.detail["net_margin"], 0.01)
        self.assertIsNone(net_margin_collapse(fin))
        self.assertTrue(any("Common earnings attribution retained" in n
                            for n in result.detail["notes"]))


class SensitivityFixTests(unittest.TestCase):
    def setUp(self):
        self.company = make_company()
        self.macro = MacroAssumptions()
        self.price = self.company.market.price

    def test_clamped_growth_cells_are_blank(self):
        wacc_grid, margin_grid = dcf_sensitivity(
            self.company, self.macro, DCFAssumptions(terminal_growth=0.075), self.price)
        base_wacc = wacc_grid.row_values[2]  # the zero-delta row
        for i, row in enumerate(wacc_grid.grid):
            for j, cell in enumerate(row):
                clamped = wacc_grid.row_values[i] - wacc_grid.col_values[j] < 0.01
                self.assertEqual(math.isnan(cell), clamped, (i, j))
        for row in margin_grid.grid:  # margin rows all discount at the base WACC
            for g, cell in zip(margin_grid.col_values, row):
                self.assertEqual(math.isnan(cell), base_wacc - g < 0.01, g)

    def test_wacc_fallback_rows_are_blank_and_labels_stay_ordered(self):
        c = _with_market(self.company, beta=-0.9)
        grid = dcf_sensitivity(c, self.macro, DCFAssumptions(), self.price)[0]
        self.assertEqual(grid.row_values, sorted(grid.row_values))
        for label, row in zip(grid.row_values, grid.grid):
            if label <= 0:
                self.assertTrue(all(math.isnan(p) for p in row))

    def test_exit_multiple_headline_sits_inside_the_dcf_bar(self):
        a = DCFAssumptions(terminal_method="exit_multiple", exit_ev_ebitda=30.0)
        report = value_company("SYNT", provider=SyntheticProvider(), peers=DEMO_PEERS,
                               dcf_assumptions=a)
        bar = next(r for r in report.football_field if r.method == "DCF")
        self.assertEqual(bar.base, report.dcf.implied_price)
        self.assertLessEqual(bar.low, bar.base)
        self.assertLessEqual(bar.base, bar.high)
        # The grids follow the exit multiple, so the headline is their centre.
        for s in report.sensitivities:
            self.assertEqual(s.col_label, "Exit EV/EBITDA")
            self.assertEqual(s.col_values, [28.0, 29.0, 30.0, 31.0, 32.0])
            self.assertAlmostEqual(s.grid[2][2], report.dcf.implied_price, places=9)

    def test_gordon_headline_grids_are_unlabelled_and_centred(self):
        report = value_company("SYNT", provider=SyntheticProvider(), peers=DEMO_PEERS)
        grid = report.sensitivities[0]
        self.assertNotIn("Gordon terminal", grid.title)
        self.assertAlmostEqual(grid.grid[2][2], report.dcf.implied_price, places=9)


class EngineBlendAndWarningTests(unittest.TestCase):
    def test_demo_run_adds_no_model_warnings(self):
        report = value_company("SYNT", provider=SyntheticProvider(), peers=DEMO_PEERS)
        self.assertEqual(report.warnings, ["Synthetic demo data: no live market or filing data."])

    def test_placeholder_prices_are_excluded_from_the_blend(self):
        c = _with_fin(make_company(), revenue=[0.0] * 5)
        report = value_company("SYNT", provider=_OneCompanyProvider(c), peers=DEMO_PEERS)
        methods = report.summary["methods"]
        self.assertEqual(methods["FCFE"], 0.0)  # still shown per method
        positive = sorted(v for v in methods.values() if v > 0)
        self.assertAlmostEqual(report.summary["blended_target"],
                               (positive[0] + positive[1]) / 2.0, places=9)
        self.assertIn("FCFE excluded from blended target: no valuation (0.00).",
                      report.warnings)
        self.assertTrue(any(w.startswith("FCFE: Insufficient revenue") for w in report.warnings))

    def test_negative_dcf_counts_as_zero_with_a_warning(self):
        c = make_distressed_company()
        report = value_company("DSTR", provider=_OneCompanyProvider(c), run_comps=False)
        methods = report.summary["methods"]
        self.assertLess(methods["DCF"], 0)
        # Negative equity is floored at zero, not dropped: median of [0, FCFE].
        self.assertAlmostEqual(report.summary["blended_target"], methods["FCFE"] / 2.0, places=9)
        self.assertTrue(any(w.startswith("DCF implies negative equity") for w in report.warnings))

    def test_model_notes_reach_warnings(self):
        report = value_company("SYNT", provider=SyntheticProvider(), peers=DEMO_PEERS,
                               dcf_assumptions=DCFAssumptions(terminal_growth=0.09))
        self.assertTrue(any(w.startswith("DCF: terminal growth") and "clamped" in w
                            for w in report.warnings))
        c = _with_market(make_company(), beta=None)
        report = value_company("SYNT", provider=_OneCompanyProvider(c), peers=DEMO_PEERS)
        self.assertIn("WACC: beta unavailable; using DEFAULT_BETA=1.0", report.warnings)

    def test_fallback_football_field_keeps_bands_ordered(self):
        c = make_distressed_company()
        report = value_company("DSTR", provider=_OneCompanyProvider(c), run_comps=False)
        rows = _fallback_football_field(report)
        self.assertTrue(any(r.method == "DCF" and r.base < 0 for r in rows))
        for r in rows:
            self.assertLessEqual(r.low, r.base, r.method)
            self.assertLessEqual(r.base, r.high, r.method)


class BlendedTargetZeroTests(unittest.TestCase):
    """A blend of 0 (negative equity floored at 0) is a target; only an empty
    blend is 'N/A'."""

    def test_all_negative_methods_give_a_zero_target_and_overvalued(self):
        c = make_distressed_company()
        report = value_company("DSTR", provider=_OneCompanyProvider(c), run_comps=False,
                               run_ddm=False, run_fcfe=False)
        s = report.summary
        self.assertLess(s["methods"]["DCF"], 0)
        self.assertEqual(s["blended_target"], 0.0)
        self.assertEqual(s["blended_upside"], -1.0)
        self.assertEqual(s["recommendation"], "Overvalued")
        self.assertTrue(any(w.startswith("Blended target is 0.00") for w in report.warnings))

    def test_median_of_two_negatives_and_a_positive_is_zero(self):
        report = ValuationReport(
            company=make_company(), macro=MacroAssumptions(), current_price=10.0,
            dcf=SimpleNamespace(implied_price=-5.0), ddm=SimpleNamespace(implied_price=12.0),
            fcfe=SimpleNamespace(implied_price=-3.0))
        s = _build_summary(report)
        self.assertEqual(s["blended_target"], 0.0)
        self.assertEqual(s["blended_upside"], -1.0)
        self.assertEqual(s["recommendation"], "Overvalued")
        self.assertEqual(s["excluded_from_blend"], {})

    def test_no_method_means_no_target(self):
        report = value_company("SYNT", provider=SyntheticProvider(), run_dcf=False,
                               run_comps=False, run_ddm=False, run_fcfe=False)
        s = report.summary
        self.assertIsNone(s["blended_target"])
        self.assertIsNone(s["blended_upside"])
        self.assertEqual(s["recommendation"], "N/A")

    def test_completed_zero_value_is_distinct_from_missing_data(self):
        base = make_company()
        c = _with_bs(base, total_debt=0.0, cash_and_investments=0.0)
        c = _with_fin(c, net_income=[0.0] * 5, ebit=[0.0] * 5,
                      pretax_income=[0.0] * 5, interest_expense=[0.0] * 5,
                      change_in_nwc=[0.0] * 5, capex=base.financials.dep_amort)
        for dcf_enabled in (True, False):
            with self.subTest(model="DCF" if dcf_enabled else "FCFE"):
                report = value_company("SYNT", provider=_OneCompanyProvider(c),
                                       run_dcf=dcf_enabled, run_fcfe=not dcf_enabled,
                                       run_comps=False, run_ddm=False, run_sensitivity=False)
                self.assertEqual(report.summary["blended_target"], 0.0)
                self.assertEqual(report.summary["blended_upside"], -1.0)
                self.assertEqual(report.summary["excluded_from_blend"], {})
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    _print_summary(report)
                method = "DCF" if dcf_enabled else "FCFE"
                line = next(line for line in out.getvalue().splitlines() if line.strip().startswith(method))
                self.assertIn("-100.0%", line)

    def test_missing_shares_do_not_make_zero_sensitivity_prices(self):
        c = _with_market(make_company(), shares_outstanding=0.0)
        c = _with_fin(c, diluted_shares=[0.0] * 5)
        report = value_company("SYNT", provider=_OneCompanyProvider(c), run_comps=False,
                               run_ddm=False, run_fcfe=False)
        self.assertFalse(report.dcf.assumptions["valuation_available"])
        self.assertTrue(all(math.isnan(v) for grid in report.sensitivities for row in grid.grid for v in row))
        self.assertNotIn("DCF", [r.method for r in report.football_field])
        self.assertNotIn("DCF", [r.method for r in _fallback_football_field(report)])

    def test_recommendation_needs_a_target_and_a_positive_price(self):
        self.assertEqual(_recommendation(0.0, 10.0), "Overvalued")
        self.assertEqual(_recommendation(None, 10.0), "N/A")
        self.assertEqual(_recommendation(10.0, 0.0), "N/A")
        self.assertEqual(_recommendation(10.0, None), "N/A")
        self.assertEqual(_recommendation(12.0, 10.0), "Undervalued")
        self.assertEqual(_recommendation(10.0, 10.0), "Fairly valued")


class FinancialInstitutionTests(unittest.TestCase):
    """Banks, insurers, REITs and lenders/BDCs: DCF and FCFE shown, not blended."""

    @staticmethod
    def _as(industry, company=None, sector="Financial Services"):
        return _with_market(company or make_company(), industry=industry, sector=sector)

    def test_bank_dcf_and_fcfe_are_shown_but_left_out_of_the_blend(self):
        c = self._as("Banks - Diversified")
        report = value_company("BANK", provider=_OneCompanyProvider(c), peers=DEMO_PEERS)
        s = report.summary
        self.assertIn("DCF", s["methods"])
        self.assertIn("FCFE", s["methods"])
        self.assertEqual(set(s["excluded_from_blend"]), {"DCF", "FCFE"})
        self.assertEqual(s["financial_institution"], "Banks - Diversified")
        expected = median([s["methods"]["Comps (median)"], s["methods"]["DDM"]])
        self.assertAlmostEqual(s["blended_target"], expected, places=9)
        self.assertTrue(any(w.startswith("Financial institution (Banks - Diversified)")
                            for w in report.warnings))
        # The CLI marks the methods it shows for reference only.
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _print_summary(report)
        self.assertIn("DCF (excluded)", out.getvalue())
        self.assertIn("FCFE (excluded)", out.getvalue())

    def test_bank_ebit_is_not_rebuilt_from_pretax_plus_interest(self):
        # For a bank interest is an operating cost; adding it back to pretax
        # income counted it twice (JPM: EBIT 93% of revenue, DCF $1,032).
        bank = _with_fin(self._as("Banks - Diversified"), ebit=[0.0] * 5)
        dcf = run_dcf(bank, MacroAssumptions(), DCFAssumptions(), bank.market.price)
        self.assertEqual(dcf.assumptions["start_ebit_margin"], 0.0)
        self.assertIn("EBIT margin unavailable; defaulting to 0", dcf.assumptions["notes"])
        self.assertFalse(any("pretax income + interest" in n for n in dcf.assumptions["notes"]))
        grid = dcf_sensitivity(bank, MacroAssumptions(), DCFAssumptions(), bank.market.price)[1]
        self.assertTrue(all(math.isnan(v) for v in grid.row_values))

    def test_operating_company_is_not_flagged(self):
        report = value_company("SYNT", provider=SyntheticProvider(), peers=DEMO_PEERS)
        self.assertIsNone(report.summary["financial_institution"])
        self.assertEqual(report.summary["excluded_from_blend"], {})

    def test_industries_that_always_count(self):
        cases = {
            "Banks - Regional": True, "Insurance - Life": True,
            "Insurance - Property & Casualty": True, "REIT - Retail": True,
            "REIT - Industrial": True, "Mortgage Finance": True,
            "Insurance Brokers": False, "Financial Data & Stock Exchanges": False,
            "Software": False,
        }
        for industry, flagged in cases.items():
            sector = "Real Estate" if industry.startswith("REIT") else "Financial Services"
            got = financial_institution(self._as(industry, sector=sector))
            self.assertEqual(got is not None, flagged, industry)

    def test_mixed_industries_need_a_lender_signal(self):
        # SYNT's interest is 1.25% of revenue: a fee business such as V or MA.
        for industry in ("Credit Services", "Capital Markets", "Asset Management"):
            self.assertIsNone(financial_institution(self._as(industry)), industry)
        c = self._as("Credit Services")
        lender = _with_fin(c, interest_expense=[0.2 * r for r in c.financials.revenue])
        self.assertIn("interest expense 20% of revenue", financial_institution(lender))
        bdc = _with_fin(self._as("Asset Management"), revenue=[0.0] * 5)
        self.assertIn("no operating revenue line", financial_institution(bdc))

    def test_edgar_classification_and_notes_also_count(self):
        c = make_company()
        fin = replace(c.financials)
        fin._financial_kind = "insurer"
        self.assertEqual(financial_institution(replace(c, financials=fin)),
                         "insurer per its EDGAR tags")
        noted = replace(c, source_notes=[
            "WARNING: EDGAR tags mark this company as bank (deposits are 54% of total assets)"])
        self.assertEqual(financial_institution(noted), "flagged by the data provider")


class NWCOneOffScreenTests(unittest.TestCase):
    """dNWC per unit of revenue change, with one-off years left out."""

    def setUp(self):
        self.macro = MacroAssumptions()
        self.price = make_company().market.price

    def test_two_outsized_years_do_not_set_the_ratio(self):
        # KO-shaped: two late years with dNWC ~10-11% of revenue on a ~1% revenue
        # change (a tax deposit, a settled accrual). Pooled they give 0.90.
        rev = [100e9, 106e9, 112e9, 118e9, 124e9, 130e9, 131e9, 132e9]
        dnwc = [0.0, 0.5e9, -0.6e9, 0.3e9, -0.2e9, 0.4e9, 13.5e9, 15.0e9]
        self.assertAlmostEqual(incremental_ratio(dnwc, rev), 28.9 / 32, places=9)
        ratio, dropped = screened_incremental_ratio(dnwc, rev)
        self.assertEqual(dropped, [6, 7])
        self.assertAlmostEqual(ratio, 0.4 / 30, places=9)
        c = _company_from_revenue(rev, change_in_nwc=dnwc)
        dcf = run_dcf(c, self.macro, DCFAssumptions(), self.price)
        fcfe = run_fcfe(c, self.macro, DDMAssumptions(), self.price)
        self.assertAlmostEqual(dcf.assumptions["nwc_pct_revenue"], 0.4 / 30, places=9)
        self.assertAlmostEqual(fcfe.detail["nwc_pct_delta_revenue"], 0.4 / 30, places=9)
        self.assertTrue(any("FY2024, FY2025" in n and "one-off" in n
                            for n in dcf.assumptions["notes"]))
        self.assertTrue(any("FY2024, FY2025" in n and "one-off" in n
                            for n in fcfe.detail["notes"]))

    def test_a_fast_growers_working_capital_build_is_kept(self):
        # NVDA-shaped: dNWC above 10% of revenue, but in line with its revenue change.
        rev = [10e9, 11e9, 17e9, 27e9, 61e9, 130e9, 216e9]
        dnwc = [0.0, 0.2e9, 1.2e9, 2.0e9, 6.8e9, 13.8e9, 17.2e9]
        ratio, dropped = screened_incremental_ratio(dnwc, rev)
        self.assertEqual(dropped, [])
        self.assertAlmostEqual(ratio, incremental_ratio(dnwc, rev), places=12)
        dcf = run_dcf(_company_from_revenue(rev, change_in_nwc=dnwc), self.macro,
                      DCFAssumptions(), self.price)
        self.assertFalse(any("one-off" in n for n in dcf.assumptions["notes"]))

    def test_flat_revenue_year_with_ordinary_dnwc_is_kept(self):
        rev = [100e9, 100.2e9, 100.1e9, 104e9, 108e9]
        dnwc = [0.0, 0.8e9, 0.4e9, 1.0e9, 0.9e9]
        self.assertEqual(screened_incremental_ratio(dnwc, rev)[1], [])


def _max_phased_capex(capex, da, kept, steps=20000):
    """Brute-force largest phased capex c(x) = D&A + (x - D&A)(1 - w(x)(1 - kept))
    for x from 1.5x D&A up to ``capex`` (w: 0 at 1.5x D&A, 1 from 2x), an
    independent check of ``growth_capex_path``'s closed-form peak."""
    best = None
    for i in range(steps + 1):
        x = 1.5 * da + (capex - 1.5 * da) * i / steps
        w = min(1.0, max(0.0, (x / da - 1.5) / 0.5))
        c = da + (x - da) * (1.0 - w * (1.0 - kept))
        best = c if best is None else max(best, c)
    return best


class CapexIntensityTests(unittest.TestCase):
    """Revenue-weighted capex/D&A and growth-phase capex faded with growth."""

    def setUp(self):
        self.macro = MacroAssumptions()
        self.price = make_company().market.price

    def test_capex_and_da_are_revenue_weighted(self):
        # RIVN-shaped: the first sales year has capex 33x revenue.
        rev = [0.055e9, 1.658e9, 4.434e9, 4.970e9, 5.387e9]
        capex = [1.794e9, 1.370e9, 1.025e9, 1.142e9, 1.710e9]
        da = [0.197e9, 0.652e9, 0.936e9, 1.030e9, 0.787e9]
        c = _company_from_revenue(rev, capex=capex, dep_amort=da)
        dcf = run_dcf(c, self.macro, DCFAssumptions(), self.price)
        fcfe = run_fcfe(c, self.macro, DDMAssumptions(), self.price)
        self.assertAlmostEqual(dcf.assumptions["capex_pct_revenue"], sum(capex) / sum(rev), places=12)
        self.assertAlmostEqual(dcf.assumptions["da_pct_revenue"], sum(da) / sum(rev), places=12)
        self.assertAlmostEqual(fcfe.detail["capex_pct_revenue"], sum(capex) / sum(rev), places=12)
        self.assertLess(dcf.assumptions["capex_pct_revenue"], 0.5)  # a mean of ratios: 6.8
        # Zero-filled years are gaps, not 0% readings.
        self.assertAlmostEqual(pooled_ratio([0.0, 5.0, 6.0], [90.0, 100.0, 110.0]), 11.0 / 210.0)

    def test_growth_capex_fades_so_the_terminal_flow_is_positive(self):
        # CAVA-shaped: 24% growth, EBIT 5%, D&A 7%, capex 15% of revenue. Held
        # flat, capex makes every forecast FCFF negative, including the terminal.
        rev = [1e9 * 1.24 ** i for i in range(5)]
        ebit = [r * 0.05 for r in rev]
        pretax = [e * 0.95 for e in ebit]
        c = _company_from_revenue(
            rev, ebit=ebit, pretax_income=pretax, tax_expense=[p * 0.21 for p in pretax],
            net_income=[p * 0.79 for p in pretax], dep_amort=[r * 0.07 for r in rev],
            capex=[r * 0.15 for r in rev], change_in_nwc=[0.0] * 5)
        dcf = run_dcf(c, self.macro, DCFAssumptions(), self.price)
        path = dcf.assumptions["capex_pct_path"]
        self.assertAlmostEqual(path[0], 0.15, places=9)  # year 1 grows as fast as history
        s = lambda g: g / (1.0 + g)
        # Net capex scaled with growth would reach 8.0% of revenue by year 5,
        # below what capex just past 1.5x D&A keeps; the path holds that.
        f = s(0.025) / s(0.24)
        self.assertLess(0.07 + 0.08 * f, 0.105)
        self.assertAlmostEqual(path[-1], _max_phased_capex(0.15, 0.07, f), places=6)
        self.assertLess(0.105, path[-1])
        self.assertLess(path[-1], 0.106)
        self.assertAlmostEqual(path[1], 0.07 + 0.08 * s(0.24 - 0.215 / 4) / s(0.24), places=9)
        self.assertEqual(path, sorted(path, reverse=True))
        self.assertGreater(dcf.fcff[-1], 0)
        self.assertGreater(dcf.terminal_value, 0)
        self.assertTrue(any("growth-phase" in n for n in dcf.assumptions["notes"]))
        flat = run_dcf(c, self.macro, DCFAssumptions(capex_pct_revenue=0.15), self.price)
        self.assertLess(flat.fcff[-1], 0)
        self.assertTrue(any("stable-year FCFF is negative" in n for n in flat.assumptions["notes"]))
        fcfe = run_fcfe(c, self.macro, DDMAssumptions(), self.price)
        self.assertAlmostEqual(fcfe.detail["capex_pct_path"][-1], path[-1], places=9)
        self.assertTrue(any("growth-phase" in n for n in fcfe.detail["notes"]))

    def test_ordinary_capex_is_held_flat(self):
        # The demo company (capex 1.25x D&A) and anything up to 1.5x D&A.
        dcf = run_dcf(make_company(), self.macro, DCFAssumptions(), self.price)
        self.assertEqual(dcf.assumptions["capex_pct_path"], [0.05] * 5)
        self.assertIsNone(growth_capex_path(0.06, 0.04, 0.10, [0.10, 0.025]))

    def test_fade_is_phased_in_between_1_5x_and_2x_da(self):
        # Growth still near its reference rate: the phase-in rises with capex.
        f = (0.08 / 1.08) / (0.10 / 1.10)
        full = growth_capex_path(0.08, 0.04, 0.10, [0.10, 0.08])
        half = growth_capex_path(0.07, 0.04, 0.10, [0.10, 0.08])  # 1.75x D&A
        self.assertAlmostEqual(full[-1], 0.04 + 0.04 * f, places=12)
        self.assertAlmostEqual(half[-1], 0.04 + 0.03 * (1.0 - 0.5 * (1.0 - f)), places=12)
        self.assertLess(half[-1], full[-1])
        self.assertEqual(full[0], 0.08)  # never above history, even for faster growth
        self.assertEqual(growth_capex_path(0.08, 0.04, 0.10, [0.20])[0], 0.08)
        # At terminal growth the phase-in peaks just past 1.5x D&A and would
        # then fall; 1.75x and 2x hold that peak instead.
        f = (0.025 / 1.025) / (0.10 / 1.10)
        peak = _max_phased_capex(0.08, 0.04, f)
        self.assertLess(0.04 + 0.04 * f, peak)
        for capex in (0.07, 0.08):
            self.assertAlmostEqual(growth_capex_path(capex, 0.04, 0.10, [0.10, 0.025])[-1], peak,
                                   places=6)
        big = growth_capex_path(0.20, 0.04, 0.10, [0.10, 0.025])  # 5x D&A: above the peak
        self.assertAlmostEqual(big[-1], 0.04 + 0.16 * f, places=12)


class StartMarginSpikeTests(unittest.TestCase):
    """A one-off gain or charge in the latest year does not set the margin."""

    def setUp(self):
        self.macro = MacroAssumptions()
        self.price = make_company().market.price

    def test_solv_shaped_disposal_gain_starts_from_the_median(self):
        rev = [8.10e9, 8.13e9, 8.20e9, 8.25e9, 8.33e9]
        margins = [0.208, 0.208, 0.206, 0.126, 0.262]
        c = _company_from_revenue(rev, ebit=[r * m for r, m in zip(rev, margins)])
        dcf = run_dcf(c, self.macro, DCFAssumptions(), self.price)
        self.assertAlmostEqual(dcf.assumptions["start_ebit_margin"], 0.208, places=9)
        self.assertTrue(any(n.startswith("latest EBIT margin 26.2%")
                            for n in dcf.assumptions["notes"]))
        # The margin grid is centred on the same start.
        grid = dcf_sensitivity(c, self.macro, DCFAssumptions(), self.price)[1]
        self.assertAlmostEqual(grid.row_values[2], 0.208, places=9)
        self.assertAlmostEqual(grid.grid[2][2], dcf.implied_price, places=9)
        # The FCFE takes the same one-off out of net income, after tax.
        fcfe = run_fcfe(c, self.macro, DDMAssumptions(), self.price)
        reported = c.financials.net_income[-1] / rev[-1]
        self.assertAlmostEqual(fcfe.detail["net_margin"],
                               reported - (0.262 - 0.208) * (1.0 - 0.21), places=9)
        self.assertTrue(any("net margin adjusted by its after-tax excess" in n
                            for n in fcfe.detail["notes"]))

    def test_fcfe_net_margin_spike_uses_the_median(self):
        rev = [10e9] * 5
        ni = [1.0e9, 1.1e9, 1.0e9, 0.9e9, 2.4e9]
        c = _company_from_revenue(rev, net_income=ni)
        fcfe = run_fcfe(c, self.macro, DDMAssumptions(), self.price)
        self.assertAlmostEqual(fcfe.detail["net_margin"], 0.10, places=9)
        self.assertTrue(any(n.startswith("Latest net margin 24.0%") for n in fcfe.detail["notes"]))

    def test_what_is_not_a_spike(self):
        rev = [100.0] * 5
        cases = {
            "steady": [30.0, 31.0, 29.0, 30.5, 32.0],
            "expanding trend": [5.0, 8.0, 12.0, 17.0, 25.0],
            "turning profitable": [-26.0, -17.0, -43.0, 20.0],
            "gradual drift": [20.0, 22.0, 24.0, 26.0, 28.0],
        }
        for name, ebit in cases.items():
            margin, spike = robust_latest_margin(ebit, rev[:len(ebit)])
            self.assertIsNone(spike, name)
            self.assertAlmostEqual(margin, ebit[-1] / 100.0, msg=name)
        # A revenue step (an acquisition, a revenue-tag change): the latest is kept.
        margin, spike = robust_latest_margin([6.4, 6.3, 7.6, 7.0], [11.1, 11.0, 12.8, 24.2])
        self.assertIsNone(spike)
        self.assertAlmostEqual(margin, 7.0 / 24.2)

    def test_a_one_off_charge_is_also_caught(self):
        margin, spike = robust_latest_margin([20.0, 21.0, 20.0, 5.0], [100.0] * 4)
        self.assertAlmostEqual(margin, 0.20)
        self.assertAlmostEqual(spike["latest"], 0.05)
        # Too little history to judge.
        self.assertEqual(robust_latest_margin([20.0, 40.0], [100.0, 100.0]), (0.40, None))


class ThinTaxHistoryTests(unittest.TestCase):
    """The effective tax rate needs 3 clean profitable years, else marginal."""

    def setUp(self):
        self.macro = MacroAssumptions()
        self.price = make_company().market.price

    @staticmethod
    def _fin(pretax, tax):
        return SimpleNamespace(pretax_income=pretax, tax_expense=tax)

    def test_rate_rules(self):
        # One profitable year (NOL-shielded): marginal rate, with a note.
        rate, source, note = effective_tax_rate_detail(
            self._fin([-5.0, -3.0, 5.0], [0.1, 0.1, 0.01]), 0.21)
        self.assertEqual((rate, source), (0.21, "marginal (fallback)"))
        self.assertIn("only 1 clean year(s) among 1", note)
        # Two profitable years, one with an allowance release: marginal.
        rate, _, note = effective_tax_rate_detail(self._fin([2.5, 2.8], [0.94, -2.05]), 0.21)
        self.assertEqual(rate, 0.21)
        self.assertIn("only 1 clean year(s) among 2", note)
        # A benefit year among enough clean ones is left out of the median.
        rate, source, note = effective_tax_rate_detail(
            self._fin([10.0, 10.0, 10.0, 10.0], [2.0, 2.5, -5.0, 2.2]), 0.21)
        self.assertAlmostEqual(rate, 0.22)
        self.assertEqual(source, "effective (historical)")
        self.assertIsNone(note)
        # No profitable year: marginal, nothing to explain.
        self.assertEqual(effective_tax_rate_detail(self._fin([-1.0], [0.0]), 0.21),
                         (0.21, "marginal (fallback)", None))

    def test_young_issuer_is_taxed_at_the_marginal_rate(self):
        # Losses, then two profitable years: one NOL-shielded, one allowance release.
        c = make_company()
        pretax = [-2e9, -1e9, -0.5e9, 4e9, 5e9]
        tax = [0.0, 0.0, 0.0, 0.1e9, -1.0e9]
        c = _with_fin(c, pretax_income=pretax, tax_expense=tax,
                      net_income=[p - t for p, t in zip(pretax, tax)])
        report = value_company("YOUNG", provider=_OneCompanyProvider(c), peers=DEMO_PEERS)
        a = report.dcf.assumptions
        self.assertEqual((a["tax_rate"], a["tax_source"]), (0.21, "marginal (fallback)"))
        self.assertEqual(report.dcf.wacc.detail["tax_source"], "marginal (fallback)")
        self.assertTrue(any(w.startswith("DCF: effective tax rate: only 1 clean year")
                            for w in report.warnings))
        # FCFE taxes the latest profit at the same rate instead of the release year's.
        d = report.fcfe.detail
        self.assertAlmostEqual(d["net_margin"], 5e9 * 0.79 / c.financials.revenue[-1], places=12)
        self.assertTrue(any(n.startswith("Tax history too thin") for n in d["notes"]))

    def test_clean_history_is_unchanged(self):
        dcf = run_dcf(make_company(), self.macro, DCFAssumptions(), self.price)
        self.assertAlmostEqual(dcf.assumptions["tax_rate"], 0.21, places=12)
        self.assertEqual(dcf.assumptions["tax_source"], "effective (historical)")


class ExitMultipleSensitivityTests(unittest.TestCase):
    """Exit-multiple DCFs get WACC x exit EV/EBITDA grids centred on the headline."""

    def setUp(self):
        self.company = make_company()
        self.macro = MacroAssumptions()
        self.price = self.company.market.price

    def test_grid_contract(self):
        a = DCFAssumptions(terminal_method="exit_multiple", exit_ev_ebitda=12.0)
        headline = run_dcf(self.company, self.macro, a, self.price).implied_price
        wacc_grid, margin_grid = dcf_sensitivity(self.company, self.macro, a, self.price)
        self.assertEqual((wacc_grid.row_label, wacc_grid.col_label), ("WACC", "Exit EV/EBITDA"))
        self.assertEqual(wacc_grid.col_values, [10.0, 11.0, 12.0, 13.0, 14.0])
        self.assertEqual(wacc_grid.grid[2][2], headline)
        self.assertEqual((margin_grid.row_label, margin_grid.col_label),
                         ("EBIT margin", "Exit EV/EBITDA"))
        self.assertEqual(margin_grid.grid[2][2], headline)
        for row in wacc_grid.grid:  # a higher exit multiple is worth more
            self.assertEqual(row, sorted(row))

    def test_low_multiple_keeps_every_column_positive(self):
        a = DCFAssumptions(terminal_method="exit_multiple", exit_ev_ebitda=1.5)
        grid = dcf_sensitivity(self.company, self.macro, a, self.price)[0]
        self.assertTrue(all(v > 0 for v in grid.col_values))
        self.assertEqual(grid.col_values[2], 1.5)
        self.assertEqual(grid.grid[2][2], run_dcf(self.company, self.macro, a, self.price).implied_price)

    def test_gordon_runs_keep_the_growth_grid(self):
        wacc_grid, margin_grid = dcf_sensitivity(self.company, self.macro, DCFAssumptions(),
                                                 self.price)
        self.assertEqual(wacc_grid.col_label, "Terminal growth")
        self.assertEqual(margin_grid.col_label, "Terminal growth")
        # An exit method without a multiple falls back to Gordon, grids included.
        fallback = dcf_sensitivity(self.company, self.macro,
                                   DCFAssumptions(terminal_method="exit_multiple"), self.price)
        self.assertEqual(fallback[0].col_label, "Terminal growth")


class FCFETaxOverrideTests(unittest.TestCase):
    """The FCFE's net-margin adjustments tax at the rate the DCF applies,
    including an explicit macro.tax_rate (the CLI's --tax)."""

    def setUp(self):
        self.price = make_company().market.price

    def test_thin_history_rebuild_uses_the_set_rate(self):
        # The ThinTaxHistoryTests young issuer, with a 30% tax rate set.
        pretax = [-2e9, -1e9, -0.5e9, 4e9, 5e9]
        tax = [0.0, 0.0, 0.0, 0.1e9, -1.0e9]
        c = _with_fin(make_company(), pretax_income=pretax, tax_expense=tax,
                      net_income=[p - t for p, t in zip(pretax, tax)])
        rev = c.financials.revenue[-1]
        macro = MacroAssumptions(tax_rate=0.30)
        dcf = run_dcf(c, macro, DCFAssumptions(), self.price)
        self.assertEqual((dcf.assumptions["tax_rate"], dcf.assumptions["tax_source"]),
                         (0.30, "macro.tax_rate"))
        fcfe = run_fcfe(c, macro, DDMAssumptions(), self.price)
        self.assertAlmostEqual(fcfe.detail["net_margin"], 5e9 * 0.70 / rev, places=12)
        self.assertTrue(any(n.startswith("Tax history too thin") and "the set tax rate 30.0%" in n
                            for n in fcfe.detail["notes"]))
        # Without a set rate the marginal rate is used and named.
        fcfe = run_fcfe(c, MacroAssumptions(), DDMAssumptions(), self.price)
        self.assertAlmostEqual(fcfe.detail["net_margin"], 5e9 * 0.79 / rev, places=12)
        self.assertTrue(any("the marginal rate 21.0%" in n for n in fcfe.detail["notes"]))

    def test_ebit_spike_excess_is_taxed_at_the_set_rate(self):
        rev = [8.10e9, 8.13e9, 8.20e9, 8.25e9, 8.33e9]
        margins = [0.208, 0.208, 0.206, 0.126, 0.262]
        c = _company_from_revenue(rev, ebit=[r * m for r, m in zip(rev, margins)])
        fcfe = run_fcfe(c, MacroAssumptions(tax_rate=0.30), DDMAssumptions(), self.price)
        reported = c.financials.net_income[-1] / rev[-1]
        self.assertAlmostEqual(fcfe.detail["net_margin"],
                               reported - (0.262 - 0.208) * (1.0 - 0.30), places=9)
        self.assertTrue(any("after-tax excess" in n and "taxed at 30.0%" in n
                            for n in fcfe.detail["notes"]))

    def test_clean_history_keeps_reported_net_income_under_a_set_rate(self):
        c = make_company()
        fcfe = run_fcfe(c, MacroAssumptions(tax_rate=0.30), DDMAssumptions(), self.price)
        f = c.financials
        self.assertAlmostEqual(fcfe.detail["net_margin"], f.net_income[-1] / f.revenue[-1],
                               places=12)


class FinancialFootballFieldTests(unittest.TestCase):
    """A financial institution's DCF and FCFE bars say they are not in the blend."""

    def setUp(self):
        self.bank = _with_market(make_company(), industry="Banks - Diversified",
                                 sector="Financial Services")

    def test_bank_bars_are_marked(self):
        report = value_company("BANK", provider=_OneCompanyProvider(self.bank), peers=DEMO_PEERS)
        labels = [r.method for r in report.football_field]
        self.assertIn("DCF (not in blend)", labels)
        self.assertIn("FCFE (not in blend)", labels)
        self.assertIn("DDM", labels)
        self.assertNotIn("DCF", labels)
        self.assertNotIn("FCFE", labels)
        self.assertEqual([r.method for r in build_football_field(report)], labels)
        fallback = [r.method for r in _fallback_football_field(report)]
        self.assertIn("DCF (not in blend)", fallback)
        self.assertIn("FCFE (not in blend)", fallback)

    def test_operating_company_bars_are_unmarked(self):
        report = value_company("SYNT", provider=SyntheticProvider(), peers=DEMO_PEERS)
        labels = [r.method for r in report.football_field]
        self.assertIn("DCF", labels)
        self.assertIn("FCFE", labels)
        self.assertFalse([m for m in labels if "not in blend" in m])


class FinancialDDMOnlyVerdictTests(unittest.TestCase):
    """A financial institution's DDM-only blend keeps its target but no verdict."""

    def setUp(self):
        self.bank = _with_market(make_company(), industry="Banks - Diversified",
                                 sector="Financial Services")

    def test_bank_without_comps_has_no_verdict(self):
        report = value_company("BANK", provider=_OneCompanyProvider(self.bank), run_comps=False)
        s = report.summary
        self.assertEqual(s["blended_target"], s["methods"]["DDM"])
        # The upside is withheld with the verdict (BAC read "N/A" next to a
        # colored -63%), although the DDM alone is well below the price.
        self.assertLess(s["methods"]["DDM"] / s["current_price"] - 1.0, -0.15)
        self.assertIsNone(s["blended_upside"])
        self.assertEqual(s["recommendation"], "N/A")
        self.assertTrue(any(w.startswith("Blended target rests on the DDM alone")
                            and "no verdict or upside is given" in w
                            and "buybacks" in w for w in report.warnings))
        # The console summary shows the target and "n/a" for its upside.
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _print_summary(report)
        row = next(ln for ln in out.getvalue().splitlines()
                   if ln.lstrip().startswith("Blended target"))
        self.assertIn(f"${s['blended_target']:,.2f}", row)
        self.assertTrue(row.rstrip().endswith("n/a"), row)

    def test_bank_with_comps_gets_a_verdict(self):
        report = value_company("BANK", provider=_OneCompanyProvider(self.bank), peers=DEMO_PEERS)
        s = report.summary
        self.assertNotEqual(s["recommendation"], "N/A")
        self.assertFalse(any(w.startswith("Blended target rests on") for w in report.warnings))

    def test_operating_company_ddm_only_keeps_its_verdict(self):
        report = value_company("SYNT", provider=SyntheticProvider(), run_dcf=False,
                               run_comps=False, run_fcfe=False)
        self.assertEqual(report.summary["recommendation"], "Overvalued")
        self.assertFalse(any(w.startswith("Blended target rests on") for w in report.warnings))


class FinancialKindReasonTests(unittest.TestCase):
    """The warning gives the reason that fits the kind of institution."""

    @staticmethod
    def _as(industry, sector="Financial Services", company=None):
        return _with_market(company or make_company(), industry=industry, sector=sector)

    def _warning(self, company):
        report = value_company("FIN", provider=_OneCompanyProvider(company), peers=DEMO_PEERS)
        return next(w for w in report.warnings if w.startswith("Financial institution ("))

    def test_kinds_from_the_industry(self):
        self.assertEqual(financial_institution_detail(self._as("Banks - Regional")),
                         ("Banks - Regional", "bank"))
        self.assertEqual(financial_institution_detail(self._as("Insurance - Life"))[1], "insurer")
        self.assertEqual(financial_institution_detail(self._as("REIT - Retail", "Real Estate"))[1],
                         "reit")
        self.assertEqual(financial_institution_detail(self._as("Mortgage Finance"))[1], "lender")
        c = self._as("Credit Services")
        lender = _with_fin(c, interest_expense=[0.2 * r for r in c.financials.revenue])
        self.assertEqual(financial_institution_detail(lender)[1], "lender")
        self.assertIsNone(financial_institution_detail(make_company()))

    def test_kinds_from_edgar_tags_and_notes(self):
        c = make_company()
        fin = replace(c.financials)
        fin._financial_kind = "bdc"
        self.assertEqual(financial_institution_detail(replace(c, financials=fin)),
                         ("BDC per its EDGAR tags", "lender"))
        reit_note = replace(c, source_notes=[
            "WARNING: EDGAR tags mark this company as a REIT (investment property is 80% of "
            "total assets); its growth comes from buying property, which is not in capex"])
        self.assertEqual(financial_institution_detail(reit_note),
                         ("flagged by the data provider", "reit"))
        vague = replace(c, source_notes=["Financial filer: statements not comparable"])
        self.assertEqual(financial_institution_detail(vague)[1], "financial")

    def test_reit_warning_names_property_not_interest(self):
        w = self._warning(self._as("REIT - Industrial", "Real Estate"))
        self.assertIn("buying property, which is not in capex", w)
        self.assertNotIn("interest", w)
        self.assertIn("interest on deposits", self._warning(self._as("Banks - Diversified")))
        self.assertIn("policyholder funds", self._warning(self._as("Insurance - Diversified")))


class FirstProfitableYearTests(unittest.TestCase):
    """A first positive EBIT margin after losses is kept but noted (BA FY2025)."""

    def test_ba_shaped_year_is_noted(self):
        # Losses, then a year lifted into profit by a disposal gain, on +34% revenue.
        rev = [62.3e9, 66.6e9, 77.8e9, 66.5e9, 89.5e9]
        ebit = [-2.9e9, -3.5e9, -0.8e9, -10.7e9, 4.28e9]
        c = _company_from_revenue(rev, ebit=ebit)
        dcf = run_dcf(c, MacroAssumptions(), DCFAssumptions(), c.market.price)
        self.assertAlmostEqual(dcf.assumptions["start_ebit_margin"], 4.28 / 89.5, places=12)
        self.assertTrue(any(n.startswith("latest EBIT margin 4.8% is the first positive year")
                            for n in dcf.assumptions["notes"]))

    def test_rule(self):
        turn = first_positive_margin([-26.0, -17.0, -43.0, 20.0], [100.0] * 4)
        self.assertAlmostEqual(turn["latest"], 0.20)
        self.assertAlmostEqual(turn["prior_median"], -0.26)
        # Already profitable the year before, a positive prior median, a loss
        # year, or too little history: nothing to note.
        self.assertIsNone(first_positive_margin([-5.0, -4.0, 3.0, 4.0], [100.0] * 4))
        self.assertIsNone(first_positive_margin([5.0, 6.0, -3.0, 4.0], [100.0] * 4))
        self.assertIsNone(first_positive_margin([-5.0, -4.0, -3.0], [100.0] * 3))
        self.assertIsNone(first_positive_margin([4.0], [100.0]))
        dcf = run_dcf(make_company(), MacroAssumptions(), DCFAssumptions(),
                      make_company().market.price)
        self.assertFalse(any("first positive year" in n for n in dcf.assumptions["notes"]))


class CapexFadeNoteTests(unittest.TestCase):
    """A negligible growth-capex fade is applied without a note."""

    def test_capex_just_past_1_5x_da_is_faded_silently(self):
        # AMZN-shaped: capex 12.1%, D&A 8.0% (1.51x), so the fade weight is ~0.03.
        rev = [100e9 * 1.10 ** i for i in range(5)]
        c = _company_from_revenue(rev, dep_amort=[r * 0.08 for r in rev],
                                  capex=[r * 0.121 for r in rev])
        price = c.market.price
        dcf = run_dcf(c, MacroAssumptions(), DCFAssumptions(), price)
        fcfe = run_fcfe(c, MacroAssumptions(), DDMAssumptions(), price)
        for path in (dcf.assumptions["capex_pct_path"], fcfe.detail["capex_pct_path"]):
            self.assertLess(path[-1], 0.121)
            self.assertLess(0.121 - path[-1], 0.005)
        self.assertFalse(any("growth-phase" in n for n in dcf.assumptions["notes"]))
        self.assertFalse(any("growth-phase" in n for n in fcfe.detail["notes"]))

    def test_threshold(self):
        self.assertTrue(material_capex_fade(0.15, [0.15, 0.10]))
        self.assertTrue(material_capex_fade(0.15, [0.15, 0.145]))
        self.assertFalse(material_capex_fade(0.121, [0.121, 0.1203]))
        self.assertFalse(material_capex_fade(0.10, None))


class CapexFadeMonotonicTests(unittest.TestCase):
    """More historical capex never gives a lower capex path or a higher DCF.

    ABG: a backfill took pooled capex from 0.755% to 1.032% of revenue (1.6x to
    2.2x D&A of 0.46%), and its DCF rose from 839.89 to 864.82, because a fade
    weight phased in between 1.5x and 2x D&A took capex at 2x below capex at
    1.5x."""

    GROWTH = [0.147, 0.117, 0.086, 0.056, 0.025]

    def test_path_is_continuous_and_non_decreasing_in_capex(self):
        da, ref = 0.04, 0.147
        prev_capex, prev = None, None
        for i in range(0, 161):
            capex = 0.04 + i * 0.001  # 1x to 5x D&A
            path = growth_capex_path(capex, da, ref, self.GROWTH) or [capex] * len(self.GROWTH)
            if prev is not None:
                for a, b in zip(prev, path):
                    self.assertLessEqual(a, b + 1e-15, capex)
                    self.assertLessEqual(b - a, capex - prev_capex + 1e-15, capex)
            prev_capex, prev = capex, path
        # Held flat up to 1.5x D&A; just past it, barely moved.
        self.assertIsNone(growth_capex_path(0.06, da, ref, self.GROWTH))
        self.assertLess(0.0601 - growth_capex_path(0.0601, da, ref, self.GROWTH)[-1], 1e-4 + 1e-15)
        # Each year is the largest phased capex at any capex up to history.
        s = lambda g: g / (1.0 + g)
        for capex in (0.065, 0.07, 0.08, 0.09, 0.12):
            path = growth_capex_path(capex, da, ref, self.GROWTH)
            for g, c in zip(self.GROWTH, path):
                self.assertAlmostEqual(c, _max_phased_capex(capex, da, s(g) / s(ref)), places=7)

    def test_a_rising_phase_in_is_unchanged(self):
        # UPS-shaped: capex 1.65x D&A, history growing 3%, so little of the
        # net capex fades and the phase-in already rises with capex.
        da, capex, growth = 0.0345, 0.0569, [0.03, 0.0275, 0.025]
        s = lambda g: g / (1.0 + g)
        w = (capex / da - 1.5) / 0.5
        for g, c in zip(growth, growth_capex_path(capex, da, 0.03, growth)):
            f = s(g) / s(0.03)
            self.assertAlmostEqual(c, da + (capex - da) * (1.0 - w * (1.0 - f)), places=12)

    def test_abg_shaped_dcf_and_fcfe_fall_as_capex_rises(self):
        rev = [6.87e9, 7.21e9, 7.13e9, 9.84e9, 15.43e9, 14.80e9, 17.19e9, 18.00e9]
        macro = MacroAssumptions()
        last_dcf = last_fcfe = None
        for capex_pct in (0.0050, 0.00690, 0.00755, 0.0085, 0.00920, 0.01032, 0.0125, 0.02):
            c = _company_from_revenue(rev, dep_amort=[r * 0.0046 for r in rev],
                                      capex=[r * capex_pct for r in rev])
            dcf = run_dcf(c, macro, DCFAssumptions(), c.market.price).implied_price
            fcfe = run_fcfe(c, macro, DDMAssumptions(), c.market.price).implied_price
            if last_dcf is not None:
                self.assertLess(dcf, last_dcf, capex_pct)
                self.assertLess(fcfe, last_fcfe, capex_pct)
            last_dcf, last_fcfe = dcf, fcfe

    def test_faded_capex_never_ends_below_the_start_threshold(self):
        # The running maximum is also a floor: capex above 1.5x D&A fades
        # toward what the remaining growth needs, but never below 1.5x D&A
        # (MSFT: capex 2.6x D&A ends at 1.5x). With no growth left, that is
        # where every such capex ends.
        da, ref = 0.04, 0.147
        for multiple in (1.6, 2.0, 2.6, 4.0):
            path = growth_capex_path(multiple * da, da, ref, self.GROWTH + [0.0])
            self.assertGreaterEqual(min(path), 1.5 * da - 1e-15, multiple)
            self.assertAlmostEqual(path[-1], 1.5 * da, places=12, msg=multiple)

    def test_dcf_and_fcfe_fade_against_the_same_da(self):
        # Capex backfilled for the last four years only (zero-filled before),
        # while D&A rose from 3% to 5% of revenue. Both models set the fade
        # against the all-years D&A they add back, so one report shows one D&A
        # and one capex path for both.
        rev = [100e9 * 1.10 ** i for i in range(8)]
        da = [r * (0.03 if i < 4 else 0.05) for i, r in enumerate(rev)]
        capex = [0.0] * 4 + [r * 0.09 for r in rev[4:]]
        c = _company_from_revenue(rev, dep_amort=da, capex=capex)
        dcf = run_dcf(c, MacroAssumptions(), DCFAssumptions(), c.market.price).assumptions
        fcfe = run_fcfe(c, MacroAssumptions(), DDMAssumptions(), c.market.price).detail
        da_all = pooled_ratio(da, rev)
        for model, notes in ((dcf, dcf["notes"]), (fcfe, fcfe["notes"])):
            self.assertAlmostEqual(model["da_pct_revenue"], da_all, places=12)
            self.assertAlmostEqual(model["capex_pct_revenue"], 0.09, places=12)
            expected = growth_capex_path(0.09, da_all, 0.10, model["revenue_growth_path"])
            self.assertEqual(len(model["capex_pct_path"]), len(expected))
            for a, b in zip(model["capex_pct_path"], expected):
                self.assertAlmostEqual(a, b, places=9)
            self.assertTrue(any(f"growth-phase (D&A {da_all:.1%})" in n for n in notes), notes)
        self.assertAlmostEqual(dcf["capex_pct_path"][-1], fcfe["capex_pct_path"][-1], places=9)


class CostOfDebtFloorTests(unittest.TestCase):
    """A derived cost of debt below rf + MIN_DEBT_SPREAD is floored there."""

    def setUp(self):
        self.macro = MacroAssumptions()
        self.floor = self.macro.risk_free_rate + MIN_DEBT_SPREAD

    @staticmethod
    def _with_interest(rate):
        c = make_company()
        debt = c.balance_sheet.total_debt
        return _with_fin(c, interest_expense=c.financials.interest_expense[:-1] + [rate * debt])

    def test_derived_rate_below_the_floor_is_lifted(self):
        for rate in (0.0156, 0.04):  # KD-shaped, and just below rf
            w = compute_wacc(self._with_interest(rate), self.macro)
            self.assertAlmostEqual(w.detail["pretax_cost_of_debt"], self.floor, places=12)
            self.assertIn("floored", w.detail["cost_of_debt_source"])
            self.assertTrue(any(n.startswith(f"derived cost of debt {rate:.4f} below rf")
                                for n in w.detail["notes"]))

    def test_rates_at_or_above_the_floor_and_overrides_are_unchanged(self):
        w = compute_wacc(make_company(), self.macro)  # demo: ~6.8%
        self.assertEqual(w.detail["cost_of_debt_source"], "interest_expense/total_debt")
        self.assertGreater(w.detail["pretax_cost_of_debt"], self.floor)
        self.assertEqual(w.detail["notes"], [])
        w = compute_wacc(self._with_interest(0.05), self.macro)
        self.assertAlmostEqual(w.detail["pretax_cost_of_debt"], 0.05, places=12)
        w = compute_wacc(self._with_interest(0.0156), replace(self.macro, pretax_cost_of_debt=0.02))
        self.assertEqual(w.detail["pretax_cost_of_debt"], 0.02)
        # Below the plausible band it is still the rf + default-spread proxy.
        w = compute_wacc(self._with_interest(0.005), self.macro)
        self.assertAlmostEqual(w.detail["pretax_cost_of_debt"],
                               self.macro.risk_free_rate + config.DEFAULT_CREDIT_SPREAD, places=12)


class CLISummaryLayoutTests(unittest.TestCase):
    """The method table stays aligned when a long method name is marked."""

    @staticmethod
    def _lines(summary):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _print_summary(SimpleNamespace(summary=summary, warnings=[]))
        return out.getvalue().splitlines()

    def test_long_excluded_name_widens_the_column(self):
        lines = self._lines({
            "name": "Example Corp", "ticker": "EXMP", "currency": "USD", "current_price": 10.0,
            "methods": {"DCF": 12.0, "Comps (median)": float("nan"), "DDM": 9.0},
            "excluded_from_blend": {"Comps (median)": "non-finite implied price"},
            "blended_target": 10.5, "blended_upside": 0.05, "recommendation": "Fairly valued",
        })
        start = next(i for i, ln in enumerate(lines) if ln.lstrip().startswith("Method"))
        table = lines[start:start + 7]  # header, rule, 3 methods, rule, blended target
        self.assertTrue(table[3].startswith("  Comps (median) (excluded) "))
        # Header, method rows and blended target end together (right-aligned
        # upside column), and the rules match the wider table.
        body = [table[0]] + table[2:5] + [table[6]]
        self.assertEqual({len(ln) for ln in body}, {50 + 3}, body)
        self.assertEqual(table[1], "  " + "-" * 53)
        self.assertEqual(table[5], table[1])

    def test_usual_layout_is_unchanged(self):
        report = value_company("SYNT", provider=SyntheticProvider(), peers=DEMO_PEERS)
        lines = self._lines(report.summary)
        self.assertIn("  Method                 Implied price      Upside", lines)
        self.assertIn("  " + "-" * 50, lines)
        self.assertIn("  DCF                            $33.38     -18.3%", lines)


class TrimOutlierTests(unittest.TestCase):
    def test_two_values_are_not_trimmed(self):
        self.assertEqual(trim_outliers([5.0, 30.0], 3.0), [5.0, 30.0])
        self.assertEqual(trim_outliers([2.0, 100.0], 3.0), [2.0, 100.0])
        self.assertEqual(trim_outliers([1.0, 10.0, 12.0, 11.0], 3.0), [10.0, 12.0, 11.0])


def _with_kind(company, kind):
    """``company`` with the data layer's ``financials._financial_kind`` set."""
    fin = replace(company.financials)
    fin._financial_kind = kind
    return replace(company, financials=fin)


class CaptiveFinanceTests(unittest.TestCase):
    """A consolidated captive finance arm (GM, F, TM, DE, CAT): DCF and FCFE are
    shown for reference but left out of the blend, as for a financial."""

    def setUp(self):
        self.captive = _with_kind(_with_market(make_company(), industry="Auto Manufacturers",
                                               sector="Consumer Cyclical"), "captive_finance")

    def test_kind_and_reason(self):
        self.assertEqual(financial_institution_detail(self.captive),
                         ("consolidated captive finance arm", "captive_finance"))
        self.assertEqual(financial_institution(self.captive), "consolidated captive finance arm")

    def test_dcf_and_fcfe_are_reference_only(self):
        # A payout of 73% of net income keeps the DDM in the blend.
        paying = _with_market(self.captive, dividend_per_share=1.5)
        self.assertGreater(ddm_payout(paying), LOW_DDM_PAYOUT_SHARE)
        report = value_company("AUTO", provider=_OneCompanyProvider(paying), peers=DEMO_PEERS)
        s = report.summary
        self.assertIn("DCF", s["methods"])
        self.assertIn("FCFE", s["methods"])
        reason = "not meaningful with a consolidated captive finance arm"
        self.assertEqual(s["excluded_from_blend"], {"DCF": reason, "FCFE": reason})
        self.assertEqual(s["financial_kind"], "captive_finance")
        self.assertEqual(s["financial_institution"], "consolidated captive finance arm")
        self.assertAlmostEqual(s["blended_target"],
                               median([s["methods"]["Comps (median)"], s["methods"]["DDM"]]),
                               places=9)
        self.assertNotEqual(s["recommendation"], "N/A")  # comps + DDM: a verdict
        warning = next(w for w in report.warnings if w.startswith("Captive finance arm:"))
        self.assertIn("finance and leasing arm", warning)
        self.assertIn("left out of the blended target", warning)
        self.assertFalse(any(w.startswith("Financial institution (") for w in report.warnings))
        labels = [r.method for r in report.football_field]
        self.assertIn("DCF (not in blend)", labels)
        self.assertIn("FCFE (not in blend)", labels)
        self.assertIn("DDM", labels)
        self.assertFalse(any(w.startswith("Captive finance arm: the DDM") for w in report.warnings))

    def test_low_payout_ddm_is_left_out_next_to_comps(self):
        # GM-shaped: the dividend is 24% of net income (GM: comps $52.53, DDM
        # $9.81, blend $31.17 "Overvalued -61.6%" at $81.20 before).
        net_income = self.captive.financials.net_income[-1]
        gm = _with_market(self.captive, dividend_per_share=0.24 * net_income / 1e10)
        self.assertAlmostEqual(ddm_payout(gm), 0.24, places=12)
        report = value_company("AUTO", provider=_OneCompanyProvider(gm), peers=DEMO_PEERS)
        s = report.summary
        comps = s["methods"]["Comps (median)"]
        self.assertIn("DDM", s["methods"])  # still shown
        self.assertEqual(s["excluded_from_blend"]["DDM"],
                         "dividends only (24% of net income); buybacks ignored")
        self.assertAlmostEqual(s["blended_target"], comps, places=9)
        self.assertEqual(s["recommendation"], _recommendation(comps, report.current_price))
        self.assertTrue(any(w.startswith("Captive finance arm: the DDM is shown for reference "
                                         "only and left out of the blended target, which rests "
                                         "on comps. Its dividend is 24% of net income")
                            for w in report.warnings))
        self.assertIn("DDM (not in blend)", [r.method for r in report.football_field])
        self.assertIn("DDM (not in blend)", [r.method for r in _fallback_football_field(report)])
        self.assertAlmostEqual(low_payout_ddm(report), 0.24, places=12)

    def test_low_payout_rule_scope(self):
        # Without comps a low-payout DDM is left out too (it no longer needs
        # comps next to it), so there is no blended target at all.
        alone = value_company("AUTO", provider=_OneCompanyProvider(self.captive), run_comps=False)
        self.assertIsNone(alone.summary["blended_target"])
        self.assertIn("DDM", alone.summary["methods"])
        self.assertEqual(alone.summary["excluded_from_blend"]["DDM"],
                         "dividends only (30% of net income); buybacks ignored")
        self.assertAlmostEqual(low_payout_ddm(alone), 0.30, places=12)
        # A bank or an operating company at the same 30% payout keeps its DDM.
        for company in (_with_kind(make_company(), "bank"), make_company()):
            report = value_company("KEEP", provider=_OneCompanyProvider(company),
                                   peers=DEMO_PEERS)
            self.assertNotIn("DDM", report.summary["excluded_from_blend"])
            self.assertIsNone(low_payout_ddm(report))
        # A loss year has no payout to measure: the DDM stays.
        fin = self.captive.financials
        loss = _with_kind(_with_fin(self.captive, net_income=fin.net_income[:-1] + [-1e9]),
                          "captive_finance")
        self.assertIsNone(ddm_payout(loss))
        report = value_company("AUTO", provider=_OneCompanyProvider(loss), peers=DEMO_PEERS)
        self.assertEqual(report.summary["financial_kind"], "captive_finance")
        self.assertNotIn("DDM", report.summary["excluded_from_blend"])
        # The payout falls back to the latest diluted count without market shares.
        no_shares = _with_market(self.captive, shares_outstanding=0.0)
        self.assertAlmostEqual(ddm_payout(no_shares), self.captive.market.dividend_per_share
                               * fin.diluted_shares[-1] / fin.net_income[-1], places=12)
        self.assertIsNone(ddm_payout(_with_market(self.captive, dividend_per_share=None)))

    def test_ddm_only_blend_has_no_target(self):
        # GM-shaped without peers (was "Blended target $9.81", "N/A", -88%):
        # the low-payout DDM is reference-only, so nothing is left to blend.
        report = value_company("AUTO", provider=_OneCompanyProvider(self.captive),
                               run_comps=False)
        s = report.summary
        self.assertIsNone(s["blended_target"])
        self.assertIsNone(s["blended_upside"])
        self.assertEqual(s["recommendation"], "N/A")
        self.assertIn("DDM", s["methods"])
        self.assertTrue(any(w.startswith("Captive finance arm: the DDM is shown for reference "
                                         "only and left out of the blended target. Its dividend "
                                         "is 30% of net income")
                            for w in report.warnings), report.warnings)
        self.assertIn("No blended target: the DCF and FCFE are left out for a company with a "
                      "consolidated captive finance arm; the DDM is shown for reference only, "
                      "and there are no comps. Supply peers with --peers to get a target.",
                      report.warnings)
        self.assertFalse(any(w.startswith("Blended target rests on the DDM alone")
                             for w in report.warnings))
        self.assertIn("DDM (not in blend)", [r.method for r in report.football_field])

    def test_leading_warning_note_also_flags_it(self):
        # The data layer's note (kept even where a copy drops _financial_kind)
        # names the kind before the generic "tags mark this company as" match.
        for note in (
            "WARNING: EDGAR tags mark this company as a group with a captive finance arm "
            "(FY2025 loan and lease originations are 28% of revenue); the finance arm's debt, "
            "leases and receivables are consolidated",
            "WARNING: consolidated statements include a captive finance arm (Example Financial, "
            "a bank-chartered lender): its debt, leases and receivables are consolidated",
        ):
            noted = replace(make_company(), source_notes=[note])
            self.assertEqual(financial_institution_detail(noted),
                             ("consolidated captive finance arm", "captive_finance"))
        # Without the WARNING prefix (the market client's "none is visible"
        # note), or for a plain lessor, nothing is flagged.
        for text in ("finance arm revenue is small",
                     "Auto Manufacturers companies often consolidate a captive finance arm, "
                     "which Yahoo's statements do not always show; none is visible here"):
            plain = replace(make_company(), source_notes=[text])
            self.assertIsNone(financial_institution_detail(plain), text)
        lessor = _with_market(make_company(), industry="Rental & Leasing Services",
                              sector="Industrials")
        self.assertIsNone(financial_institution_detail(lessor))
        self.assertIsNone(financial_institution_detail(_with_kind(lessor, None)))

    def test_unknown_kind_counts_as_financial(self):
        self.assertEqual(financial_institution_detail(_with_kind(make_company(), "conglomerate")),
                         ("conglomerate per the data provider", "financial"))
        self.assertIsNone(financial_institution_detail(_with_kind(make_company(), "")))


class FinancialCompsTests(unittest.TestCase):
    """Lenders and insurers get P/E and P/B comps only; EV multiples carry the
    funding debt (JPM: EV/Sales $106 vs P/E $277 and P/B $200)."""

    def _comps(self, company):
        return run_comps(company, SyntheticProvider(), DEMO_PEERS, company.market.price)

    def test_bank_uses_pe_and_pb_only(self):
        bank = _with_market(make_company(), industry="Banks - Diversified",
                            sector="Financial Services")
        full = self._comps(make_company())
        comps = self._comps(bank)
        for m in ("ev_ebitda", "ev_sales"):
            self.assertIsNone(comps.implied[m], m)
            self.assertIsNotNone(full.implied[m], m)
        self.assertAlmostEqual(comps.implied["pe"], full.implied["pe"], places=9)
        self.assertAlmostEqual(comps.implied_price_summary["median"],
                               median([full.implied["pe"], full.implied["pb"]]), places=9)
        self.assertTrue(comps.notes[-1].startswith("Equity multiples only (Banks - Diversified)"))
        # Peer EV multiples stay on display.
        self.assertEqual(comps.stats["ev_ebitda"]["n"], 3)
        report = value_company("BANK", provider=_OneCompanyProvider(bank), peers=DEMO_PEERS)
        self.assertTrue(any(w.startswith("Comps: Equity multiples only") for w in report.warnings))
        self.assertNotIn("EV/EBITDA comps", [r.method for r in report.football_field])

    def test_insurer_lender_and_captive_finance_too(self):
        for company in (_with_kind(make_company(), "insurer"), _with_kind(make_company(), "bdc"),
                        _with_kind(make_company(), "captive_finance")):
            comps = self._comps(company)
            self.assertIsNone(comps.implied["ev_ebitda"])
            self.assertIsNotNone(comps.implied["pb"])

    def test_reits_and_operating_companies_keep_standard_multiples(self):
        reit = _with_market(make_company(), industry="REIT - Retail", sector="Real Estate")
        for company in (make_company(), reit):
            comps = self._comps(company)
            self.assertTrue(all(comps.implied[m] is not None
                                for m in ("ev_ebitda", "ev_sales", "pe", "pb")))
            self.assertIsNone(comps.implied["peg"])
            self.assertFalse(any(n.startswith("Equity multiples only") for n in comps.notes))
        report = value_company("SYNT", provider=SyntheticProvider(), peers=DEMO_PEERS)
        self.assertAlmostEqual(report.summary["methods"]["Comps (median)"], 46.96, places=2)

    def test_mortgage_reits_are_lenders(self):
        # AGNC, NLY, STWD: repo and warehouse debt funds a loan book, not property.
        for industry in ("REIT - Mortgage", "REIT\u2014Mortgage", "REIT \u2013 Mortgage"):
            mreit = _with_market(make_company(), industry=industry, sector="Real Estate")
            self.assertEqual(financial_institution_detail(mreit), (industry, "lender"))
            comps = self._comps(mreit)
            for m in ("ev_ebitda", "ev_sales", "peg"):
                self.assertIsNone(comps.implied[m], (industry, m))
            self.assertTrue(comps.notes[-1].startswith(f"Equity multiples only ({industry})"))
        report = value_company("MREIT", provider=_OneCompanyProvider(mreit), peers=DEMO_PEERS)
        self.assertEqual(report.summary["financial_kind"], "lender")
        self.assertEqual(set(report.summary["excluded_from_blend"]), {"DCF", "FCFE"})
        # Equity REITs (either dash) and the older insurer spelling keep their kinds.
        for industry, kind in (("REIT - Residential", "reit"), ("REIT\u2014Office", "reit"),
                               ("Insurance\u2014Life", "insurer")):
            company = _with_market(make_company(), industry=industry)
            self.assertEqual(financial_institution_detail(company), (industry, kind))


def _f_shaped(**changes):
    """F FY2021-25: EBIT margin +3.3%, +4.0%, +3.1%, +2.8%, then -4.9% on +1.2%
    revenue (impairments); net margin 13.2%, -1.3%, 2.5%, 3.2%, -4.4%."""
    rev = [136.3e9, 158.1e9, 176.2e9, 185.0e9, 187.3e9]
    ebit_m = [0.0332, 0.0397, 0.0310, 0.0282, -0.0490]
    ni_m = [0.1316, -0.0125, 0.0247, 0.0318, -0.0436]
    ebit = [r * m for r, m in zip(rev, ebit_m)]
    fields = dict(ebit=ebit, net_income=[r * m for r, m in zip(rev, ni_m)],
                  pretax_income=[e * 0.9 for e in ebit],
                  tax_expense=[max(e * 0.9, 0.0) * 0.21 for e in ebit])
    fields.update(changes)
    return _company_from_revenue(rev, **fields)


class ChargeYearMarginTests(unittest.TestCase):
    """A loss year after a profitable run starts the projection but fades to the
    prior median, capped at the last profitable year's margin (F FY2025: blend
    $0.00 'Overvalued -100%' before)."""

    F_PRIOR_EBIT = [0.0332, 0.0397, 0.0310, 0.0282]  # median 3.21%, last 2.82%

    def setUp(self):
        self.macro = MacroAssumptions()
        self.company = _f_shaped()
        self.price = self.company.market.price

    def test_rule(self):
        rev = [136.3, 158.1, 176.2, 185.0, 187.3]
        got = charge_year_margin([4.525, 6.277, 5.462, 5.217, -9.178], rev)
        self.assertAlmostEqual(got["latest"], -9.178 / 187.3, places=12)
        self.assertAlmostEqual(got["prior_median"],
                               median([4.525 / 136.3, 6.277 / 158.1, 5.462 / 176.2, 5.217 / 185.0]),
                               places=12)
        self.assertEqual(got["profitable_years"], 4)
        self.assertAlmostEqual(got["last_profitable"], 5.217 / 185.0, places=12)
        self.assertAlmostEqual(got["target"], 5.217 / 185.0, places=12)  # below the median
        # Rising margins: the median is below the last year and is the target.
        rising = charge_year_margin([2.0, 3.0, 4.0, 5.0, -3.0], [100.0] * 5)
        self.assertAlmostEqual(rising["target"], 0.035, places=12)
        # A business already in decline (INTC-shaped) is not faded back to a
        # four-year median of 14.4%, but to its last profitable year's 0.2%.
        decline = charge_year_margin([30.0, 25.0, 3.7, 0.2, -5.0], [100.0] * 5)
        self.assertAlmostEqual(decline["prior_median"], 0.1435, places=12)
        self.assertAlmostEqual(decline["target"], 0.002, places=12)
        # Not a charge year: a loss in the last three priors, only two prior
        # years, a revenue collapse, or a profitable latest year.
        self.assertIsNone(charge_year_margin([3.0, -1.0, 3.0, 3.0, -5.0], [100.0] * 5))
        self.assertIsNone(charge_year_margin([3.0, 3.0, -5.0], [100.0] * 3))
        self.assertIsNone(charge_year_margin([3.0, 3.0, 3.0, 3.0, -5.0], [100.0] * 4 + [70.0]))
        self.assertIsNone(charge_year_margin([3.0, 3.0, 3.0, 3.0, 1.0], [100.0] * 5))

    def test_dcf_fades_from_the_loss_to_the_capped_prior_median(self):
        dcf = run_dcf(self.company, self.macro, DCFAssumptions(), self.price)
        a = dcf.assumptions
        target = min(median(self.F_PRIOR_EBIT), self.F_PRIOR_EBIT[-1])
        self.assertAlmostEqual(target, 0.0282, places=12)
        self.assertAlmostEqual(a["start_ebit_margin"], -0.0490, places=9)
        self.assertAlmostEqual(a["target_ebit_margin"], target, places=9)
        self.assertAlmostEqual(a["ebit_margin_path"][-1], target, places=9)
        note = next(n for n in a["notes"] if n.startswith("WARNING: latest EBIT margin -4.9%"))
        self.assertIn("is a loss after 4 profitable years (prior median 3.2%)", note)
        self.assertIn("fades to 2.8%, the last profitable year's margin (below the prior median) "
                      "by year 5. Impairments are non-cash", note)
        # With rising margins the target is the prior median, named as such.
        rising = _f_shaped(ebit=[r * m for r, m in zip([136.3e9, 158.1e9, 176.2e9, 185.0e9,
                                                         187.3e9],
                                                        [0.020, 0.030, 0.040, 0.050, -0.049])])
        up = run_dcf(rising, self.macro, DCFAssumptions(), rising.market.price).assumptions
        self.assertAlmostEqual(up["target_ebit_margin"], 0.035, places=9)
        self.assertTrue(any("fades to the prior median 3.5% by year 5" in n for n in up["notes"]))
        held = run_dcf(self.company, self.macro, DCFAssumptions(target_ebit_margin=-0.049),
                       self.price)
        self.assertLess(held.implied_price, dcf.implied_price)
        self.assertFalse(any("charge year" in n for n in held.assumptions["notes"]))
        # The engine keeps the WARNING prefix in front of the model label.
        report = value_company("CHRG", provider=_OneCompanyProvider(self.company),
                               run_comps=False)
        self.assertTrue(any(w.startswith("WARNING: DCF: latest EBIT margin -4.9%")
                            for w in report.warnings))

    def test_margin_grid_centres_on_the_faded_target(self):
        headline = run_dcf(self.company, self.macro, DCFAssumptions(), self.price).implied_price
        grid = dcf_sensitivity(self.company, self.macro, DCFAssumptions(), self.price)[1]
        self.assertAlmostEqual(grid.row_values[2], 0.0282, places=9)
        self.assertAlmostEqual(grid.grid[2][2], headline, places=9)

    def test_fcfe_net_margin_fades_too(self):
        # F's net margin had a loss in FY2022, so its own series does not
        # qualify; the EBIT series does, and the note says so.
        fcfe = run_fcfe(self.company, self.macro, DDMAssumptions(), self.price)
        d = fcfe.detail
        prior = median([0.1316, -0.0125, 0.0247, 0.0318])  # 2.8%, below FY2024's 3.2%
        self.assertAlmostEqual(d["net_margin"], -0.0436, places=9)
        self.assertAlmostEqual(d["net_margin_path"][0], -0.0436, places=9)
        self.assertAlmostEqual(d["net_margin_path"][-1], prior, places=9)
        self.assertTrue(any(n == "WARNING: latest net margin -4.4% is a loss, as is the EBIT "
                                 "margin -4.9% after 4 years of operating profit, likely a charge "
                                 "year (impairments, restructuring); the projection starts from "
                                 "it and fades to the prior median net margin 2.8%."
                            for n in d["notes"]), d["notes"])
        self.assertFalse(any("profitable years" in n for n in d["notes"]))
        # With a profitable FY2022 the net series itself qualifies; its last
        # profitable year (3.2%) is below the prior median (4.8%) and caps it.
        rev = [136.3e9, 158.1e9, 176.2e9, 185.0e9, 187.3e9]
        net = _f_shaped(net_income=[r * m for r, m in zip(rev, [0.1316, 0.0640, 0.0247,
                                                                 0.0318, -0.0436])])
        d = run_fcfe(net, self.macro, DDMAssumptions(), net.market.price).detail
        self.assertAlmostEqual(d["net_margin_path"][-1], 0.0318, places=9)
        self.assertTrue(any(n.startswith("WARNING: latest net margin -4.4% is a loss after 4 "
                                         "profitable years, likely a charge year")
                            and "fades to 3.2%, the last profitable year's net margin (below "
                                "the prior median 4.8%)" in n for n in d["notes"]), d["notes"])
        self.assertNotIn("net_margin_path",
                         run_fcfe(make_company(), self.macro, DDMAssumptions(),
                                  make_company().market.price).detail)

    def test_steady_companies_are_unchanged(self):
        dcf = run_dcf(make_company(), self.macro, DCFAssumptions(), make_company().market.price)
        self.assertEqual(dcf.assumptions["start_ebit_margin"], dcf.assumptions["target_ebit_margin"])
        self.assertFalse(any("charge year" in n for n in dcf.assumptions["notes"]))


class NetMarginCollapseTests(unittest.TestCase):
    """FCFE: a net margin that collapsed below what EBIT implies (BP FY2025:
    0.03% on an 8.3% EBIT margin) is not held for the whole projection."""

    def setUp(self):
        self.macro = MacroAssumptions()
        self.rev = [181.8e9, 158.3e9, 142.5e9, 142.6e9]

    def _company(self, ni_margins):
        ebit = [r * m for r, m in zip(self.rev, [0.1717, 0.1478, 0.0720, 0.0835])]
        return _company_from_revenue(self.rev, ebit=ebit, interest_expense=[3.86e9] * 4,
                                     net_income=[r * m for r, m in zip(self.rev, ni_margins)])

    def _implied(self, c):
        fin = c.financials
        tax, _, _ = effective_tax_rate_detail(fin, 0.21)
        return (fin.ebit[-1] - 3.86e9) * (1.0 - tax) / fin.revenue[-1]

    def _note(self, fcfe):
        return next(n for n in fcfe.detail["notes"] if "far below" in n)

    def test_depressed_history_uses_the_implied_margin(self):
        c = self._company([-0.0103, 0.0725, 0.0020, 0.0003])
        fcfe = run_fcfe(c, self.macro, DDMAssumptions(), c.market.price)
        self.assertAlmostEqual(fcfe.detail["net_margin"], self._implied(c), places=12)
        note = self._note(fcfe)
        self.assertTrue(note.startswith("WARNING: latest net margin 0.0% is far below"), note)
        self.assertIn("projecting from the margin EBIT implies after interest and tax (prior "
                      "median 0.2% is no better), 4.5%.", note)
        # The engine lists it with the other WARNING notes.
        report = value_company("COLL", provider=_OneCompanyProvider(c), run_comps=False)
        self.assertTrue(any(w.startswith("WARNING: FCFE: latest net margin 0.0% is far below")
                            for w in report.warnings))

    def test_prior_median_above_the_implied_margin_is_capped(self):
        # Prior median 5.5% against an implied 4.5%: the implied margin is used
        # and named as such (GIS: a prior median near 125% from a mis-tagged
        # revenue line was shown as "the prior median net margin, 1.4%").
        c = self._company([0.060, 0.055, 0.050, -0.010])
        implied = self._implied(c)
        self.assertLess(implied, 0.055)
        fcfe = run_fcfe(c, self.macro, DDMAssumptions(), c.market.price)
        self.assertAlmostEqual(fcfe.detail["net_margin"], implied, places=12)
        note = self._note(fcfe)
        self.assertIn(f"projecting from the margin EBIT implies after interest and tax (below "
                      f"the prior median 5.5%), {implied:.1%}.", note)
        self.assertNotIn("the prior median net margin", note)
        implausible = self._company([1.30, 1.25, 1.20, -0.010])
        fcfe = run_fcfe(implausible, self.macro, DDMAssumptions(), implausible.market.price)
        self.assertAlmostEqual(fcfe.detail["net_margin"], self._implied(implausible), places=12)
        self.assertIn("(below the prior median 125.0%)", self._note(fcfe))

    def test_prior_median_between_the_line_and_the_implied_margin_is_used(self):
        c = self._company([0.030, 0.035, 0.040, -0.010])  # prior median 3.5% < implied 4.5%
        fcfe = run_fcfe(c, self.macro, DDMAssumptions(), c.market.price)
        self.assertAlmostEqual(fcfe.detail["net_margin"], 0.035, places=12)
        self.assertIn("projecting from the prior median net margin, 3.5%.", self._note(fcfe))

    def test_a_merely_lower_net_margin_is_kept(self):
        c = self._company([0.060, 0.055, 0.050, 0.030])  # well above 25% of implied
        fcfe = run_fcfe(c, self.macro, DDMAssumptions(), c.market.price)
        self.assertAlmostEqual(fcfe.detail["net_margin"], 0.030, places=12)
        self.assertFalse(any("far below" in n for n in fcfe.detail["notes"]))


class DividendRunTests(unittest.TestCase):
    """The dividend CAGR covers only the paying run since the last break (GM:
    -16.1%/yr across the 2020-21 suspension before)."""

    GM = [2.242e9, 2.35e9, 0.669e9, 0.186e9, 0.397e9, 0.597e9, 0.653e9, 0.657e9]

    def _fin(self, divs):
        return replace(_company_from_revenue([100e9] * len(divs)).financials,
                       dividends_paid=list(divs))

    def test_gm_shaped_suspension_falls_back_to_sustainable_growth(self):
        notes: list[str] = []
        self.assertIsNone(_dividend_cagr(self._fin(self.GM), notes))
        self.assertTrue(any(n.startswith("Dividends paid per share more than doubled in FY2022")
                            and "no dividend CAGR" in n for n in notes))
        c = _with_bs(_company_from_revenue([100e9] * 8, dividends_paid=list(self.GM)),
                     total_equity=400e9)  # ROE ~4.7%, as for GM
        ddm = run_ddm(c, MacroAssumptions(), DDMAssumptions(), c.market.price)
        sg = _sustainable_growth(c.financials, c)
        self.assertAlmostEqual(ddm.detail["high_growth"], sg, places=12)
        self.assertGreater(ddm.detail["high_growth"], 0.0)

    def test_a_long_run_after_a_cut_sets_the_cagr(self):
        divs = [2.0, 0.9] + [1.0 * 1.05 ** i for i in range(6)]
        notes: list[str] = []
        self.assertAlmostEqual(_dividend_cagr(self._fin(divs), notes), 0.05, places=12)
        self.assertTrue(any(n.startswith("Dividends paid per share fell by more than half in FY2019")
                            for n in notes))

    def test_one_year_specials_are_skipped_not_breaks(self):
        # COST FY2018-25 (millions): specials in FY2021 and FY2024. Read as
        # breaks, the normal FY2025 ended the run and no CAGR was left.
        cost = [689, 1038, 1479, 5748, 1498, 1251, 9041, 2183]
        notes: list[str] = []
        got = _dividend_cagr(self._fin([v * 1e6 for v in cost]), notes)
        self.assertAlmostEqual(got, (2183 / 689) ** (1 / 7) - 1, places=12)
        self.assertEqual(notes, ["Dividends paid per share in FY2021, FY2024 were more than double both "
                                 "neighbouring years (a special dividend); left out of the "
                                 "dividend CAGR as one-offs, not read as breaks in the paying "
                                 "run."])
        # A special next to the first paying year: that year stays the start.
        notes = []
        self.assertAlmostEqual(_dividend_cagr(self._fin([1.0, 3.5, 1.21, 1.331, 1.4641]),
                                              notes), 0.10, places=12)
        self.assertEqual(len(notes), 1)
        # After a break, the run since can skip a special too (4 paying years).
        notes = []
        divs = [5.0, 0.9, 1.0, 1.05, 3.0, 1.157625, 1.21550625]
        self.assertAlmostEqual(_dividend_cagr(self._fin(divs), notes), 0.05, places=12)
        self.assertTrue(notes[0].startswith("Dividends paid per share in FY2023 were more than double"))
        self.assertTrue(notes[1].startswith("Dividends paid per share fell by more than half in FY2020")
                        and "the CAGR covers the 4 paying years since (FY2021-FY2025)" in notes[1])

    def test_what_still_breaks_the_run(self):
        # AGCO: specials over four years in a row, then a normal FY2025.
        notes: list[str] = []
        self.assertIsNone(_dividend_cagr(self._fin([47, 48, 48, 358, 404, 457, 273, 86]), notes))
        self.assertTrue(notes[0].startswith("Dividends paid per share fell by more than half in FY2025"))
        # A special in the latest year: one neighbour cannot show it is a one-off.
        notes = []
        self.assertIsNone(_dividend_cagr(self._fin([1.0, 1.05, 1.1, 1.15, 3.5]), notes))
        self.assertTrue(notes[0].startswith("Dividends paid per share more than doubled in FY2025"))
        # A spike whose neighbours are not in line with each other is a break.
        notes = []
        self.assertIsNone(_dividend_cagr(self._fin([1.0, 1.0, 1.0, 5.0, 2.2, 2.3]), notes))
        self.assertTrue(notes[0].startswith("Dividends paid per share fell by more than half in FY2024"))
        # The trough of a suspension is not a one-year dip to skip (GM).
        notes = []
        self.assertIsNone(_dividend_cagr(self._fin(self.GM), notes))
        self.assertEqual(len(notes), 1)

    def test_unbroken_histories_are_unchanged(self):
        # A steady payer, and a new payer (leading zero years) with two years.
        steady = [1.0 * 1.08 ** i for i in range(5)]
        notes: list[str] = []
        self.assertAlmostEqual(_dividend_cagr(self._fin(steady), notes), 0.08, places=12)
        self.assertAlmostEqual(_dividend_cagr(self._fin([0.0] * 6 + [5.0, 5.25]), notes),
                               0.05, places=12)
        self.assertEqual(notes, [])
        base = run_ddm(make_company(), MacroAssumptions(), DDMAssumptions(),
                       make_company().market.price)
        self.assertAlmostEqual(base.implied_price, 10.99, places=2)  # pinned demo DDM


class BetaAdjustmentNoteTests(unittest.TestCase):
    """The WACC notes say when the market data adjusted the beta."""

    def setUp(self):
        self.macro = MacroAssumptions()

    def _adjusted(self, note):
        c = _with_market(make_company(), beta=1.816)
        return replace(c, source_notes=c.source_notes + [note])

    def test_market_data_note_and_raw_beta(self):
        # The market client's note, with the raw figure kept on MarketData.
        note = ("Beta 1.816 is Yahoo's 5-year monthly beta 2.217 Blume-adjusted toward 1.0 "
                "(0.67 x raw + 0.33) for CAPM")
        c = self._adjusted(note)
        rf, erp = self.macro.risk_free_rate, self.macro.equity_risk_premium
        for company in (c, self._with_raw(c, 2.217)):
            w = compute_wacc(company, self.macro)
            self.assertEqual(w.detail["beta_source"], "market data (adjusted)")
            self.assertAlmostEqual(w.detail["beta_raw"], 2.217, places=12)
            msg = next(n for n in w.detail["notes"] if n.startswith("beta 1.816 is Blume-adjusted"))
            self.assertIn("toward 1 from the raw beta 2.217", msg)
            self.assertIn(f"raw beta: {rf + 2.217 * erp:.2%}", msg)
            self.assertAlmostEqual(w.cost_of_equity, rf + 1.816 * erp, places=12)  # not re-adjusted
        report = value_company("ADJ", provider=_OneCompanyProvider(c), run_comps=False)
        self.assertTrue(any(w.startswith("WACC: beta 1.816 is Blume-adjusted")
                            for w in report.warnings))
        # The WACC note restates the market data's note, which then leaves the
        # warnings (one line per adjustment) but stays with the source notes.
        self.assertNotIn(note, report.warnings)
        self.assertIn(note, report.company.source_notes)
        no_dcf = value_company("ADJ", provider=_OneCompanyProvider(c), run_comps=False,
                               run_dcf=False)
        self.assertIn(note, no_dcf.warnings)

    @staticmethod
    def _with_raw(company, raw):
        market = replace(company.market)
        market.raw_beta = raw
        return replace(company, market=market)

    def test_note_without_a_raw_value_and_the_attribute_alone(self):
        c = self._adjusted("Beta adjusted toward 1 (Blume) for mean reversion")
        self.assertEqual(beta_adjustment(c), {"raw": None, "blume": True,
                                              "note": c.source_notes[-1]})
        self.assertTrue(any(n.startswith("beta 1.816 is Blume-adjusted toward 1 (see the data "
                                         "notes)") for n in compute_wacc(c, self.macro).detail["notes"]))
        # That WACC note points at the data note, so the data note is kept.
        report = value_company("ADJ", provider=_OneCompanyProvider(c), run_comps=False)
        self.assertIn(c.source_notes[-1], report.warnings)
        kept = self._with_raw(make_company(), 2.0)
        self.assertEqual(beta_adjustment(kept)["raw"], 2.0)
        self.assertTrue(any(n.startswith("beta 1.100 is adjusted by the market data from the raw "
                                         "beta 2.000") for n in compute_wacc(kept, self.macro).detail["notes"]))
        self.assertIsNone(beta_adjustment(self._with_raw(make_company(), None)))

    def test_unadjusted_and_dropped_betas_add_no_adjustment_note(self):
        w = compute_wacc(make_company(), self.macro)
        self.assertEqual((w.detail["beta_source"], w.detail["beta_raw"]), ("market data", None))
        self.assertEqual(w.detail["notes"], [])
        dropped = replace(_with_market(make_company(), beta=None), source_notes=[
            "Yahoo beta -0.22 is not usable for CAPM (not positive); left unset so the models "
            "use the default beta"])
        w = compute_wacc(dropped, self.macro)
        self.assertEqual(w.detail["beta_source"], "DEFAULT_BETA")
        self.assertEqual(w.detail["notes"], [f"beta unavailable; using DEFAULT_BETA={config.DEFAULT_BETA}"])


class MaintenanceCapexTests(unittest.TestCase):
    """No usable capex year but D&A above 0 (PSX, NEE, AER): capex is set equal
    to D&A (maintenance only) with a WARNING in the DCF and the FCFE, instead
    of 0 with D&A still added back (PSX: DCF 339 'Undervalued +22%' before)."""

    def setUp(self):
        self.macro = MacroAssumptions()
        base = make_company()
        self.price = base.market.price
        self.psx = _with_fin(base, capex=[0.0] * 5)  # D&A 4.0% of revenue, capex untagged
        self.da_pct = pooled_ratio(base.financials.dep_amort, base.financials.revenue)

    def test_dcf_uses_maintenance_capex(self):
        dcf = run_dcf(self.psx, self.macro, DCFAssumptions(), self.price)
        a = dcf.assumptions
        self.assertAlmostEqual(a["capex_pct_revenue"], self.da_pct, places=12)
        self.assertEqual(a["capex_pct_path"], [a["capex_pct_revenue"]] * 5)
        note = next(n for n in a["notes"] if n.startswith("WARNING: no usable capex history"))
        self.assertIn("while D&A is 4.0% of revenue; capex set equal to D&A (maintenance only) "
                      "rather than 0", note)
        self.assertNotIn("capex %revenue unavailable; defaulting to 0", a["notes"])
        # Same value as a filer whose reported capex equals its D&A; the old
        # reading (capex 0, D&A added back) overstated it.
        same = _with_fin(self.psx, capex=list(self.psx.financials.dep_amort))
        self.assertAlmostEqual(run_dcf(same, self.macro, DCFAssumptions(), self.price)
                               .implied_price, dcf.implied_price, places=9)
        zero = run_dcf(self.psx, self.macro, DCFAssumptions(capex_pct_revenue=0.0), self.price)
        self.assertGreater(zero.implied_price, dcf.implied_price)
        self.assertFalse(any("no usable capex" in n for n in zero.assumptions["notes"]))
        # Unreported (None) years are the same gap as zero-filled ones.
        none = _with_fin(self.psx, capex=[None] * 5)
        self.assertAlmostEqual(run_dcf(none, self.macro, DCFAssumptions(), self.price)
                               .implied_price, dcf.implied_price, places=9)
        # An explicit D&A share is what capex follows.
        set_da = run_dcf(self.psx, self.macro, DCFAssumptions(da_pct_revenue=0.06), self.price)
        self.assertAlmostEqual(set_da.assumptions["capex_pct_revenue"], 0.06, places=12)

    def test_fcfe_uses_maintenance_capex(self):
        fcfe = run_fcfe(self.psx, self.macro, DDMAssumptions(), self.price)
        d = fcfe.detail
        self.assertAlmostEqual(d["capex_pct_revenue"], d["da_pct_revenue"], places=12)
        self.assertAlmostEqual(d["capex_pct_revenue"], self.da_pct, places=12)
        self.assertTrue(any(n.startswith("WARNING: No usable capex history (zero or unreported in "
                                         "every year) while D&A is 4.0% of revenue")
                            for n in d["notes"]), d["notes"])
        # Same value as a filer whose reported capex equals its D&A: D&A and
        # capex cancel, as they do with neither line (the old reading added
        # D&A back with no capex against it).
        same = _with_fin(self.psx, capex=list(self.psx.financials.dep_amort))
        self.assertAlmostEqual(run_fcfe(same, self.macro, DDMAssumptions(), self.price)
                               .implied_price, fcfe.implied_price, places=9)
        no_da = _with_fin(self.psx, dep_amort=[0.0] * 5)
        self.assertAlmostEqual(run_fcfe(no_da, self.macro, DDMAssumptions(), self.price)
                               .implied_price, fcfe.implied_price, places=9)

    def test_engine_lists_both_warnings(self):
        report = value_company("PSX", provider=_OneCompanyProvider(self.psx), run_comps=False)
        self.assertTrue(any(w.startswith("WARNING: DCF: no usable capex history")
                            for w in report.warnings))
        self.assertTrue(any(w.startswith("WARNING: FCFE: No usable capex history")
                            for w in report.warnings))

    def test_what_the_guard_leaves_alone(self):
        # One reported year is history: its ratio is used, with no warning.
        rev = self.psx.financials.revenue
        one = _with_fin(self.psx, capex=[0.0] * 4 + [rev[-1] * 0.05])
        dcf = run_dcf(one, self.macro, DCFAssumptions(), self.price)
        self.assertAlmostEqual(dcf.assumptions["capex_pct_revenue"], 0.05, places=12)
        self.assertFalse(any("no usable capex" in n for n in dcf.assumptions["notes"]))
        fcfe = run_fcfe(one, self.macro, DDMAssumptions(), self.price)
        self.assertAlmostEqual(fcfe.detail["capex_pct_revenue"], 0.05, places=12)
        # No D&A either: nothing is added back, so capex stays 0.
        bare = _with_fin(self.psx, dep_amort=[0.0] * 5)
        fcfe = run_fcfe(bare, self.macro, DDMAssumptions(), self.price)
        self.assertEqual(fcfe.detail["capex_pct_revenue"], 0.0)
        self.assertIn("No usable capex history; capex set to 0% of revenue.", fcfe.detail["notes"])
        # The demo company reports capex: unchanged.
        demo = run_dcf(make_company(), self.macro, DCFAssumptions(), self.price)
        self.assertAlmostEqual(demo.assumptions["capex_pct_revenue"], 0.05, places=12)

    def test_banks_insurers_reits_and_lenders_keep_capex_at_zero(self):
        # A REIT's D&A is mostly property depreciation, not a capex proxy (O's
        # reference DCF 156 -> 52 with capex = D&A), and a bank's says nothing
        # about reinvestment: their reference-only DCF and FCFE keep capex 0,
        # with the plain note rather than a WARNING.
        reit = _with_market(self.psx, industry="REIT - Retail", sector="Real Estate")
        # "other" is a kind the data layer does not name: flagged as "financial".
        flagged = [reit] + [_with_kind(self.psx, k)
                            for k in ("bank", "insurer", "bdc", "reit", "other")]
        for company in flagged:
            with self.subTest(financial_institution_detail(company)):
                dcf = run_dcf(company, self.macro, DCFAssumptions(), self.price)
                self.assertEqual(dcf.assumptions["capex_pct_revenue"], 0.0)
                self.assertIn("capex %revenue unavailable; defaulting to 0",
                              dcf.assumptions["notes"])
                fcfe = run_fcfe(company, self.macro, DDMAssumptions(), self.price)
                self.assertEqual(fcfe.detail["capex_pct_revenue"], 0.0)
                self.assertIn("No usable capex history; capex set to 0% of revenue.",
                              fcfe.detail["notes"])
        report = value_company("O", provider=_OneCompanyProvider(reit), run_comps=False)
        self.assertFalse([w for w in report.warnings
                          if w.startswith("WARNING") and "capex" in w], report.warnings)
        # A captive-finance group and a lessor still get maintenance capex.
        for kind in REFERENCE_ONLY_KINDS:
            with self.subTest(kind):
                company = _with_kind(self.psx, kind)
                dcf = run_dcf(company, self.macro, DCFAssumptions(), self.price)
                self.assertAlmostEqual(dcf.assumptions["capex_pct_revenue"], self.da_pct,
                                       places=12)
                fcfe = run_fcfe(company, self.macro, DDMAssumptions(), self.price)
                self.assertAlmostEqual(fcfe.detail["capex_pct_revenue"], self.da_pct,
                                       places=12)


class LessorTests(unittest.TestCase):
    """A debt-funded operating lessor (AerCap: 'Undervalued +413%' from an FCFF
    DCF before) takes the captive-finance path: DCF and FCFE for reference
    only, comps on P/E and P/B, and its DDM never sets the target alone."""

    def setUp(self):
        self.aer = _with_kind(_with_market(make_company(), industry="Rental & Leasing Services",
                                           sector="Industrials"), "lessor")
        self.label = "operating lessor per its filings"

    def test_kind_and_label(self):
        self.assertIn("lessor", REFERENCE_ONLY_KINDS)
        self.assertEqual(financial_institution_detail(self.aer), (self.label, "lessor"))
        self.assertEqual(financial_institution(self.aer), self.label)

    def test_dcf_and_fcfe_are_reference_only_and_comps_equity_only(self):
        paying = _with_market(self.aer, dividend_per_share=1.5)  # a 73% payout
        report = value_company("LESS", provider=_OneCompanyProvider(paying), peers=DEMO_PEERS)
        s = report.summary
        reason = "not meaningful for a debt-funded lessor"
        self.assertEqual(s["excluded_from_blend"], {"DCF": reason, "FCFE": reason})
        self.assertEqual(s["financial_kind"], "lessor")
        self.assertAlmostEqual(s["blended_target"],
                               median([s["methods"]["Comps (median)"], s["methods"]["DDM"]]),
                               places=9)
        self.assertNotEqual(s["recommendation"], "N/A")
        self.assertIn(f"Debt-funded lessor ({self.label}): its borrowing funds the fleet it leases "
                      "out (aircraft, railcars, vehicles), so interest is an operating cost",
                      next(w for w in report.warnings if w.startswith("Debt-funded lessor (")))
        self.assertFalse(any(w.startswith(("Financial institution (", "Captive finance arm"))
                             for w in report.warnings))
        for m in ("ev_ebitda", "ev_sales", "peg"):
            self.assertIsNone(report.comps.implied[m], m)
        self.assertTrue(any(n.startswith(f"Equity multiples only ({self.label})")
                            and "the fleet it leases out" in n for n in report.comps.notes))
        labels = [r.method for r in report.football_field]
        self.assertIn("DCF (not in blend)", labels)
        self.assertIn("FCFE (not in blend)", labels)
        self.assertIn("DDM", labels)

    def test_low_payout_and_no_comps(self):
        # AerCap pays 7% of net income; the demo company's 30% is also low.
        report = value_company("LESS", provider=_OneCompanyProvider(self.aer), run_comps=False)
        s = report.summary
        self.assertIsNone(s["blended_target"])
        self.assertIsNone(s["blended_upside"])
        self.assertEqual(s["recommendation"], "N/A")
        self.assertEqual(s["excluded_from_blend"]["DDM"],
                         "dividends only (30% of net income); buybacks ignored")
        self.assertTrue(any(w.startswith("Debt-funded lessor: the DDM is shown for reference only")
                            for w in report.warnings))
        self.assertIn("No blended target: the DCF and FCFE are left out for a debt-funded lessor; "
                      "the DDM is shown for reference only, and there are no comps. Supply peers "
                      "with --peers to get a target.", report.warnings)
        # With peers the target rests on comps (the DDM stays out).
        with_comps = value_company("LESS", provider=_OneCompanyProvider(self.aer),
                                   peers=DEMO_PEERS).summary
        self.assertAlmostEqual(with_comps["blended_target"],
                               with_comps["methods"]["Comps (median)"], places=9)

    def test_rental_and_leasing_industry_needs_heavy_interest(self):
        base = _with_market(make_company(), industry="Rental & Leasing Services",
                            sector="Industrials")
        rev = base.financials.revenue
        aer = _with_fin(base, interest_expense=[0.22 * r for r in rev])
        self.assertEqual(financial_institution_detail(aer),
                         ("Rental & Leasing Services; interest expense 22% of revenue", "lessor"))
        report = value_company("AER", provider=_OneCompanyProvider(aer), run_comps=False)
        self.assertEqual(report.summary["financial_kind"], "lessor")
        self.assertEqual(set(report.summary["excluded_from_blend"]), {"DCF", "FCFE", "DDM"})
        # URI (interest ~4% of revenue) and Ryder rent equipment on ordinary
        # leverage: not flagged. Neither is heavy interest in another industry.
        uri = _with_fin(base, interest_expense=[0.044 * r for r in rev])
        self.assertIsNone(financial_institution_detail(uri))
        other = _with_market(aer, industry="Railroads")
        self.assertIsNone(financial_institution_detail(other))

    def test_a_leading_warning_note_also_flags_it(self):
        note = ("WARNING: EDGAR tags mark this company as a debt-funded operating lessor (FY2025 "
                "lease income is 91% and interest expense 22% of revenue); its borrowing funds "
                "the fleet it leases out")
        noted = replace(make_company(), source_notes=[note])
        self.assertEqual(financial_institution_detail(noted), (self.label, "lessor"))
        plain = replace(make_company(), source_notes=["Lessors hold their fleets as property"])
        self.assertIsNone(financial_institution_detail(plain))
        # Another WARNING that mentions lessors in passing does not flag it.
        passing = replace(make_company(), source_notes=[
            "WARNING: capex for FY2025 not backfilled: Yahoo's figure is 7x EDGAR's D&A, as "
            "when a lessor's fleet purchases are included",
            "WARNING: lessors and rental companies hold their fleets as property"])
        self.assertIsNone(financial_institution_detail(passing))
        # The captive-finance note keeps its kind.
        captive = replace(make_company(), source_notes=[
            "WARNING: EDGAR tags mark this company as a group with a captive finance arm "
            "(FY2025 loan and lease originations are 28% of revenue)"])
        self.assertEqual(financial_institution_detail(captive)[1], "captive_finance")


    def test_missing_ebit_is_rebuilt_as_the_data_layer_does(self):
        # AerCap on older data: EBIT zero-filled, pretax income and fleet
        # interest reported. Left at 0, the reference DCF was only the debt
        # (-268 a share); the data layer derives EBIT as pretax + interest.
        rev = self.aer.financials.revenue
        interest = [0.22 * r for r in rev]
        pretax = [0.10 * r for r in rev]
        gap = _with_kind(_with_fin(self.aer, ebit=[0.0] * len(rev), pretax_income=pretax,
                                   interest_expense=interest), "lessor")
        macro = MacroAssumptions()
        dcf = run_dcf(gap, macro, DCFAssumptions(), gap.market.price)
        a = dcf.assumptions
        self.assertAlmostEqual(a["start_ebit_margin"], 0.32, places=12)
        self.assertIn("EBIT not reported for some years; approximated as pretax income + "
                      "interest expense", a["notes"])
        self.assertNotIn("EBIT margin unavailable; defaulting to 0", a["notes"])
        explicit = _with_kind(_with_fin(gap, ebit=[p + i for p, i in zip(pretax, interest)]),
                              "lessor")
        self.assertAlmostEqual(run_dcf(explicit, macro, DCFAssumptions(), gap.market.price)
                               .implied_price, dcf.implied_price, places=9)
        # The sensitivity grid's margin axis centres on the same margin.
        self.assertAlmostEqual(_base_latest_ebit_margin(gap), 0.32, places=12)
        # A bank's gaps stay: its interest is the cost of its lending book.
        bank = _with_kind(gap, "bank")
        self.assertIn("EBIT margin unavailable; defaulting to 0",
                      run_dcf(bank, macro, DCFAssumptions(), gap.market.price)
                      .assumptions["notes"])
        self.assertTrue(math.isnan(_base_latest_ebit_margin(bank)))


class CaptiveDDMAloneTests(unittest.TestCase):
    """A captive-finance group's DDM never sets the target on its own (PCAR:
    'Blended target $56.03', N/A, -50% before): without comps there is no
    blended target and a warning asks for peers."""

    def setUp(self):
        captive = _with_kind(_with_market(make_company(), industry="Farm & Heavy Construction "
                                          "Machinery", sector="Industrials"), "captive_finance")
        net_income = captive.financials.net_income[-1]
        # PCAR-shaped: regular plus year-end extra dividends, 87% of net income.
        self.pcar = _with_market(captive, dividend_per_share=0.87 * net_income / 1e10)

    def test_high_payout_ddm_alone_has_no_target(self):
        self.assertAlmostEqual(ddm_payout(self.pcar), 0.87, places=12)
        report = value_company("PCAR", provider=_OneCompanyProvider(self.pcar), run_comps=False)
        s = report.summary
        self.assertIsNone(s["blended_target"])
        self.assertIsNone(s["blended_upside"])
        self.assertEqual(s["recommendation"], "N/A")
        self.assertIn("DDM", s["methods"])
        self.assertEqual(s["excluded_from_blend"]["DDM"],
                         "dividends only and no comps to check it; supply peers")
        self.assertIsNone(low_payout_ddm(report))
        self.assertTrue(ddm_left_alone(report))
        self.assertTrue(ddm_reference_only(report))
        self.assertIn("Captive finance arm: the DDM is shown for reference only and left out of "
                      "the blended target, which it would otherwise set on its own: it counts "
                      "regular dividends only (buybacks and special dividends are ignored) and no "
                      "second method checks it.", report.warnings)
        self.assertTrue(any(w.startswith("No blended target:") and "--peers" in w
                            for w in report.warnings))
        self.assertIn("DDM (not in blend)", [r.method for r in report.football_field])
        self.assertIn("DDM (not in blend)", [r.method for r in _fallback_football_field(report)])
        # The console shows n/a for the target and its upside.
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _print_summary(report)
        text = out.getvalue()
        self.assertIn("  DDM (excluded)", text)
        row = next(ln for ln in text.splitlines() if ln.lstrip().startswith("Blended target"))
        self.assertEqual(row.split()[-2:], ["n/a", "n/a"])

    def test_with_comps_the_ddm_joins_the_blend(self):
        report = value_company("PCAR", provider=_OneCompanyProvider(self.pcar), peers=DEMO_PEERS)
        s = report.summary
        self.assertFalse(ddm_left_alone(report))
        self.assertNotIn("DDM", s["excluded_from_blend"])
        self.assertAlmostEqual(s["blended_target"],
                               median([s["methods"]["Comps (median)"], s["methods"]["DDM"]]),
                               places=9)
        self.assertIsNotNone(s["blended_upside"])
        self.assertFalse(any(w.startswith("No blended target") for w in report.warnings))

    def test_banks_and_operating_companies_are_not_affected(self):
        bank = _with_kind(make_company(), "bank")
        report = value_company("BANK", provider=_OneCompanyProvider(bank), run_comps=False)
        self.assertFalse(ddm_left_alone(report))
        self.assertEqual(report.summary["blended_target"], report.summary["methods"]["DDM"])
        demo = value_company("SYNT", provider=SyntheticProvider(), run_dcf=False,
                             run_comps=False, run_fcfe=False)
        self.assertFalse(ddm_reference_only(demo))
        self.assertIsNotNone(demo.summary["blended_upside"])


class _PeerRowsProvider(_OneCompanyProvider):
    """Serves a caller-supplied company and peer rows."""

    def __init__(self, company, rows):
        super().__init__(company)
        self._rows = rows

    def get_peer_comp_rows(self, tickers):
        return list(self._rows)


class NoTargetWordingTests(unittest.TestCase):
    """The no-target and DDM-alone warnings name only the reference-only
    models that ran, and ask for peers only when none were supplied or none
    was usable: peers that ran but gave no price (a lessor's peers with EV
    multiples only, since it is valued on P/E and P/B) are not a missing
    input."""

    PEERS = ["P0", "P1", "P2"]

    def setUp(self):
        self.lessor = _with_kind(make_company(), "lessor")
        self.ev_only = [CompRow(ticker=t, name=t, market_cap=5e10, enterprise_value=5.2e10,
                                ev_ebitda=18.0 + i, ev_sales=4.0)
                        for i, t in enumerate(self.PEERS)]

    def test_peers_that_give_no_price(self):
        report = value_company("LESS", provider=_PeerRowsProvider(self.lessor, self.ev_only),
                               peers=self.PEERS)
        self.assertTrue(report.comps.peers)
        self.assertIsNone(report.summary["blended_target"])
        self.assertIn("No blended target: the DCF and FCFE are left out for a debt-funded "
                      "lessor; the DDM is shown for reference only, and the peers supplied give "
                      "no usable comps price.", report.warnings)
        self.assertFalse([w for w in report.warnings if "--peers" in w], report.warnings)
        # A high-payout lessor's DDM, left out because it would stand alone:
        # its exclusion reason does not ask for peers either.
        net_income = self.lessor.financials.net_income[-1]
        high = _with_market(self.lessor, dividend_per_share=0.87 * net_income / 1e10)
        report = value_company("LESS", provider=_PeerRowsProvider(high, self.ev_only),
                               peers=self.PEERS)
        self.assertTrue(ddm_left_alone(report))
        self.assertIsNone(report.summary["blended_target"])
        self.assertEqual(report.summary["excluded_from_blend"]["DDM"],
                         "dividends only and no comps price from the peers supplied to check it")
        self.assertFalse([w for w in report.warnings if "--peers" in w], report.warnings)
        # A bank whose blend is the DDM alone, with the DCF off: only the FCFE
        # is named, and supplying peers is not what is missing.
        bank = _with_kind(make_company(), "bank")
        report = value_company("BANK", provider=_PeerRowsProvider(bank, self.ev_only),
                               peers=self.PEERS, run_dcf=False)
        self.assertEqual(report.summary["recommendation"], "N/A")
        self.assertIn("Blended target rests on the DDM alone (FCFE left out for a financial "
                      "institution; no usable comps price from the peers supplied), so no "
                      "verdict or upside is given: the DDM counts regular dividends only "
                      "(buybacks and special dividends are ignored) and there is no second "
                      "method to check it against.", report.warnings)

    def test_only_the_models_that_ran_are_named(self):
        provider = _OneCompanyProvider(self.lessor)
        report = value_company("LESS", provider=provider, run_comps=False, run_dcf=False)
        self.assertIn("No blended target: the FCFE is left out for a debt-funded lessor; the "
                      "DDM is shown for reference only, and there are no comps. Supply peers "
                      "with --peers to get a target.", report.warnings)
        report = value_company("LESS", provider=provider, run_comps=False, run_dcf=False,
                               run_fcfe=False)
        self.assertIn("No blended target: the DDM of a debt-funded lessor is shown for "
                      "reference only, and there are no comps. Supply peers with --peers to "
                      "get a target.", report.warnings)
        report = value_company("LESS", provider=provider, run_comps=False, run_dcf=False,
                               run_fcfe=False, run_ddm=False)
        self.assertIn("No blended target for a debt-funded lessor: there are no comps. Supply "
                      "peers with --peers to get a target.", report.warnings)
        # Comps that ran but found no usable peer still ask for peers.
        report = value_company("LESS", provider=_PeerRowsProvider(self.lessor, []),
                               peers=self.PEERS)
        self.assertFalse(report.comps.peers)
        self.assertTrue(any(w.startswith("No blended target:") and w.endswith(
            "and there are no comps. Supply peers with --peers to get a target.")
            for w in report.warnings), report.warnings)


class CompsEarningsCollapseTests(unittest.TestCase):
    """Comps leave out the P/E-implied price when the latest net margin has
    collapsed below operating profit (BP FY2025: a P/E-implied GBP 0.04 took
    the comps median from 9.15 to 6.97)."""

    REV = [181.8e9, 158.3e9, 142.5e9, 142.6e9]

    def _company(self, ni_margins):
        ebit = [r * m for r, m in zip(self.REV, [0.1717, 0.1478, 0.0720, 0.0835])]
        return _company_from_revenue(self.REV, ebit=ebit, interest_expense=[3.86e9] * 4,
                                     net_income=[r * m for r, m in zip(self.REV, ni_margins)])

    @staticmethod
    def _comps(company, **kwargs):
        return run_comps(company, SyntheticProvider(), DEMO_PEERS, company.market.price, **kwargs)

    def test_bp_shaped_pe_is_left_out(self):
        bp = self._company([-0.0103, 0.0725, 0.0020, 0.0003])
        collapse = net_margin_collapse(bp.financials)
        self.assertAlmostEqual(collapse["latest"], 0.0003, places=12)
        self.assertAlmostEqual(collapse["prior_median"], 0.0020, places=12)
        comps = self._comps(bp)
        self.assertIsNone(comps.implied["pe"])
        self.assertIsNotNone(comps.implied["pb"])
        self.assertIsNotNone(comps.implied["ev_ebitda"])
        rest = [p for p in comps.implied.values() if p is not None and p > 0]
        self.assertAlmostEqual(comps.implied_price_summary["median"], median(rest), places=9)
        note = next(n for n in comps.notes if n.startswith(EARNINGS_COLLAPSE_NOTE_PREFIX))
        self.assertIn("the latest net margin 0.03% is under a quarter of both its prior median "
                      "0.20% and the 4.5% its EBIT margin 8.3% implies", note)
        self.assertIn("so P/E gives no implied price", note)
        # Without the screen the P/E price would be about a thousandth of the others.
        normal = self._comps(self._company([0.06, 0.07, 0.065, 0.06]))
        self.assertIsNotNone(normal.implied["pe"])
        self.assertFalse(any(n.startswith(EARNINGS_COLLAPSE_NOTE_PREFIX) for n in normal.notes))
        report = value_company("BP", provider=_OneCompanyProvider(bp), peers=DEMO_PEERS)
        self.assertTrue(any(w.startswith("Comps: P/E-implied price left out")
                            for w in report.warnings))
        self.assertNotIn("P/E comps", [r.method for r in report.football_field])

    def test_peg_on_the_same_eps_goes_too(self):
        c = self._company([0.0001, 0.0700, 0.0700, 0.0003])  # a positive earnings CAGR
        full = self._comps(c, tax_rate=0.99)  # a 99% tax leaves no collapse to screen
        self.assertIsNone(full.implied["peg"])
        comps = self._comps(c)
        self.assertIsNone(comps.implied["pe"])
        self.assertIsNone(comps.implied["peg"])
        self.assertTrue(any("so P/E gives no implied price" in n for n in comps.notes))

    def test_what_keeps_the_pe(self):
        # A net margin that has always been thin (under a quarter of what EBIT
        # implies, but in line with its own history) is the earnings base.
        thin = self._comps(self._company([0.0012, 0.0010, 0.0011, 0.0009]))
        self.assertIsNotNone(thin.implied["pe"])
        # A merely lower margin, and an explicit tax rate that removes the gap.
        self.assertIsNotNone(self._comps(self._company([0.06, 0.055, 0.05, 0.03])).implied["pe"])
        bp = self._company([-0.0103, 0.0725, 0.0020, 0.0003])
        self.assertIsNotNone(self._comps(bp, tax_rate=0.99).implied["pe"])
        # The engine passes the macro tax rate (--tax) to the screen, as to
        # the FCFE.
        report = value_company("BP", provider=_OneCompanyProvider(bp), peers=DEMO_PEERS,
                               macro=MacroAssumptions(tax_rate=0.99))
        self.assertIsNotNone(report.comps.implied["pe"])
        self.assertFalse(any(w.startswith("Comps: P/E-implied price left out")
                             for w in report.warnings))
        self.assertIn("P/E comps", [r.method for r in report.football_field])
        # SYNT comps are unchanged.
        self.assertAlmostEqual(value_company("SYNT", provider=SyntheticProvider(),
                                             peers=DEMO_PEERS).summary["methods"]["Comps (median)"],
                               46.96, places=2)


class CompsChargeInsideEBITTests(unittest.TestCase):
    """A charge inside operating profit also drops the P/E: GM FY2025 (EBIT
    margin 1.6% against 6.7%, net margin 1.46% against 6.1%) got a P/E-implied
    25.48 against 79.91 from P/B, a comps median of 52.70 and 'Overvalued
    -35%', while the DCF and FCFE start from the median margin."""

    REV = [127.0e9, 156.7e9, 171.8e9, 187.4e9, 185.0e9]
    EBIT = [0.0734, 0.0658, 0.0541, 0.0682, 0.0157]
    NI = [0.0789, 0.0634, 0.0589, 0.0321, 0.0146]

    def _company(self, ebit_m=EBIT, ni_m=NI, kind=None):
        # EBITDA is EBIT + D&A, as every provider builds it, so it carries a
        # charge inside EBIT too.
        ebit = [r * m for r, m in zip(self.REV, ebit_m)]
        da = [r * 0.04 for r in self.REV]
        c = _company_from_revenue(self.REV, ebit=ebit, dep_amort=da,
                                  ebitda=[e + d for e, d in zip(ebit, da)],
                                  net_income=[r * m for r, m in zip(self.REV, ni_m)])
        return _with_kind(c, kind) if kind else c

    @staticmethod
    def _comps(company):
        return run_comps(company, SyntheticProvider(), DEMO_PEERS, company.market.price)

    def _note(self, comps):
        return next(n for n in comps.notes if n.startswith(EARNINGS_COLLAPSE_NOTE_PREFIX))

    def test_gm_shaped_captive_comps_are_pb_only(self):
        gm = self._company(kind="captive_finance")
        self.assertIsNone(net_margin_collapse(gm.financials))  # EBIT is itself a one-off
        charge = ebit_charge_collapse(gm.financials)
        self.assertEqual(charge["ebit_move"], "drop")
        self.assertAlmostEqual(charge["latest"], 0.0146, places=12)
        self.assertAlmostEqual(charge["ebit_margin"], 0.0157, places=12)
        self.assertAlmostEqual(charge["prior_median"], median(self.NI[:-1]), places=12)
        self.assertAlmostEqual(charge["ebit_prior_median"], median(self.EBIT[:-1]), places=12)
        comps = self._comps(gm)
        for m in ("pe", "peg", "ev_ebitda", "ev_sales"):
            self.assertIsNone(comps.implied[m], m)
        self.assertEqual(comps.implied_price_summary["median"], comps.implied["pb"])
        note = self._note(comps)
        self.assertIn("the latest net margin 1.46% is under a quarter of its prior median "
                      "6.11%, and the latest EBIT margin 1.6% is a one-off drop from its prior "
                      "median 6.7% (likely a charge inside operating profit", note)
        # The captive's EV multiples were already out, so only the P/E goes here.
        self.assertIn("so P/E gives no implied price: a peer multiple of that EPS would", note)
        report = value_company("GM", provider=_OneCompanyProvider(gm), peers=DEMO_PEERS)
        self.assertTrue(any(w.startswith("Comps: P/E-implied price left out")
                            for w in report.warnings))
        self.assertNotIn("P/E comps", [r.method for r in report.football_field])
        self.assertAlmostEqual(report.summary["methods"]["Comps (median)"],
                               comps.implied["pb"], places=9)

    def test_a_smaller_charge_is_caught_by_the_collapse_rule(self):
        # EBIT 1.8% or 2.0% against 6.7%: a drop under 5pp, so not a one-off
        # spike, but under a third of the prior median, which the DCF fades
        # from (utils.collapsed_margin). The comps must not price it as normal.
        for ebit_latest in (0.0180, 0.0200):
            c = self._company(ebit_m=self.EBIT[:-1] + [ebit_latest], kind="captive_finance")
            fin = c.financials
            self.assertIsNone(robust_latest_margin(fin.ebit, fin.revenue)[1])
            self.assertIsNotNone(collapsed_margin(fin.ebit, fin.revenue))
            charge = ebit_charge_collapse(fin)
            self.assertEqual(charge["ebit_move"], "collapse", ebit_latest)
            self.assertAlmostEqual(charge["ebit_margin"], ebit_latest, places=12)
            self.assertAlmostEqual(charge["ebit_prior_median"], median(self.EBIT[:-1]),
                                   places=12)
            dcf = run_dcf(c, MacroAssumptions(), DCFAssumptions(), c.market.price)
            self.assertTrue(any("is below a third of the prior median" in n
                                for n in dcf.assumptions["notes"]))
            comps = self._comps(c)
            self.assertIsNone(comps.implied["pe"])
            self.assertEqual(comps.implied_price_summary["median"], comps.implied["pb"])
            self.assertIn(f"the latest EBIT margin {ebit_latest:.2%} is under a third of its "
                          "prior median 6.70% (likely a charge inside operating profit",
                          self._note(comps))

    def test_an_operating_company_also_loses_ev_ebitda(self):
        c = self._company()
        comps = self._comps(c)
        for m in ("pe", "ev_ebitda"):
            self.assertIsNone(comps.implied[m], m)
        for m in ("ev_sales", "pb"):
            self.assertIsNotNone(comps.implied[m], m)
        rest = [comps.implied["ev_sales"], comps.implied["pb"]]
        self.assertAlmostEqual(comps.implied_price_summary["median"], median(rest), places=9)
        note = self._note(comps)
        self.assertRegex(note, r"so P/E(, PEG)? and EV/EBITDA give no implied price: a peer "
                               r"multiple of that EPS or EBITDA would price the charge")
        # What EV/EBITDA says on the charge year's EBITDA (net margin only
        # halved, so the screen keeps it): about half of what it says with
        # EBIT back at its prior median (19.41 against 38.39).
        charged = self._comps(self._company(ni_m=self.NI[:-1] + [0.0200]))
        normal = self._comps(self._company(ebit_m=self.EBIT[:-1] + [0.0670]))
        self.assertLess(charged.implied["ev_ebitda"], 0.6 * normal.implied["ev_ebitda"])

    def test_a_one_off_ebit_gain_with_a_collapsed_net_margin(self):
        # EBIT up to 14% against 6.7% while the net margin fell to 1.46%: a
        # charge below operating profit next to a one-off gain within it. The
        # P/E goes; EBITDA holds no charge, so EV/EBITDA stays.
        gain = self._company(ebit_m=self.EBIT[:-1] + [0.1400])
        self.assertIsNone(net_margin_collapse(gain.financials))
        charge = ebit_charge_collapse(gain.financials)
        self.assertEqual(charge["ebit_move"], "gain")
        comps = self._comps(gain)
        self.assertIsNone(comps.implied["pe"])
        self.assertIsNotNone(comps.implied["ev_ebitda"])
        note = self._note(comps)
        self.assertIn("the latest EBIT margin 14.0% is a one-off rise above its prior median "
                      "6.7% (likely a charge below operating profit", note)
        self.assertNotIn("EV/EBITDA", note)
        self.assertIn("a peer multiple of that EPS would price the charge", note)

    def test_what_keeps_the_pe(self):
        # A net margin down by half, not three quarters, with the EBIT drop.
        kept = self._company(ni_m=self.NI[:-1] + [0.0200])
        self.assertIsNone(ebit_charge_collapse(kept.financials))
        self.assertIsNotNone(self._comps(kept).implied["pe"])
        self.assertIsNotNone(self._comps(kept).implied["ev_ebitda"])
        # A one-off EBIT gain with the net margin in line.
        gain = self._company(ebit_m=self.EBIT[:-1] + [0.1400], ni_m=self.NI[:-1] + [0.0600])
        self.assertIsNone(ebit_charge_collapse(gain.financials))
        self.assertIsNotNone(self._comps(gain).implied["pe"])
        # EBIT in line (a steady decline is neither a one-off nor a collapse):
        # the BP screen's job.
        steady = self._company(ebit_m=[0.0734, 0.0658, 0.0541, 0.0450, 0.0380])
        self.assertIsNone(ebit_charge_collapse(steady.financials))
        # A loss year leaves no P/E price to drop, so no note.
        loss = self._comps(self._company(ni_m=self.NI[:-1] + [-0.0100]))
        self.assertIsNone(loss.implied["pe"])
        self.assertIsNotNone(loss.implied["ev_ebitda"])
        self.assertFalse(any(n.startswith(EARNINGS_COLLAPSE_NOTE_PREFIX) for n in loss.notes))


def _jd_shaped(**changes):
    """JD FY2022-25: EBIT margin 1.75%, 2.67%, 3.41%, then 0.28% on +13% revenue."""
    rev = [156.2e9, 161.9e9, 173.0e9, 195.4e9]
    ebit = [r * m for r, m in zip(rev, [0.0175, 0.0267, 0.0341, 0.0028])]
    fields = dict(ebit=ebit, ebitda=[e + r * 0.0074 for e, r in zip(ebit, rev)])
    fields.update(changes)
    return _company_from_revenue(rev, **fields)


class CollapsedMarginTests(unittest.TestCase):
    """A positive EBIT margin below a third of the prior median after a
    profitable run, on stable revenue, is faded back like a charge year (JD
    FY2025: DCF -0.11 on a 0.28% margin, blend 'Overvalued -46%' from the DDM)."""

    def setUp(self):
        self.macro = MacroAssumptions()
        self.jd = _jd_shaped()
        self.price = self.jd.market.price

    def test_rule(self):
        rev = [156.2, 161.9, 173.0, 195.4]
        got = collapsed_margin([r * m for r, m in zip(rev, [0.0175, 0.0267, 0.0341, 0.0028])],
                               rev)
        self.assertAlmostEqual(got["latest"], 0.0028, places=12)
        self.assertAlmostEqual(got["prior_median"], 0.0267, places=12)
        self.assertAlmostEqual(got["target"], 0.0267, places=12)
        self.assertEqual(got["profitable_years"], 3)
        self.assertAlmostEqual(got["revenue_move"], 195.4 / 173.0 - 1.0, places=12)
        self.assertAlmostEqual(MARGIN_COLLAPSE_SHARE, 1.0 / 3.0, places=12)
        flat = [100.0] * 5
        # Capped at the year before when that is below the median.
        self.assertAlmostEqual(collapsed_margin([6.0, 5.0, 4.0, 3.0, 0.5], flat)["target"], 0.03)
        # A steady decline stays above a third of the median (TSLA: 12.1%,
        # 16.8%, 9.2%, 7.2%, 4.6%), a loss is a charge year, a latest-year
        # revenue collapse or a revenue step inside the window (GIS: a
        # revenue-tag change) is not a trough, fewer than three prior years
        # is too little history, and a loss among them is not a profitable run.
        tsla_rev = [53.8, 81.5, 96.8, 97.7, 94.8]
        self.assertIsNone(collapsed_margin([r * m for r, m in zip(
            tsla_rev, [0.1212, 0.1676, 0.0919, 0.0724, 0.0459])], tsla_rev))
        self.assertIsNone(collapsed_margin([3.0, 3.0, 3.0, 3.0, -1.0], flat))
        self.assertIsNone(collapsed_margin([3.0, 3.0, 3.0, 3.0, 0.5], flat[:4] + [70.0]))
        self.assertIsNone(collapsed_margin([3.4, 3.4, 3.4, 3.3, 0.9],
                                           [2.0, 2.1, 2.0, 19.5, 18.4]))
        self.assertIsNone(collapsed_margin([3.0, 3.0, 0.5], flat[:3]))
        self.assertIsNone(collapsed_margin([3.0, -1.0, 3.0, 3.0, 0.5], flat))
        # A drop of more than five points is a spike: the median is the start.
        self.assertIsNone(collapsed_margin([7.3, 6.6, 5.4, 6.8, 1.6], flat))
        self.assertIsNotNone(robust_latest_margin([7.3, 6.6, 5.4, 6.8, 1.6], flat)[1])

    def test_dcf_fades_from_the_trough_to_the_prior_median(self):
        dcf = run_dcf(self.jd, self.macro, DCFAssumptions(), self.price)
        a = dcf.assumptions
        self.assertAlmostEqual(a["start_ebit_margin"], 0.0028, places=9)
        self.assertAlmostEqual(a["target_ebit_margin"], 0.0267, places=9)
        self.assertAlmostEqual(a["ebit_margin_path"][-1], 0.0267, places=9)
        note = next(n for n in a["notes"] if n.startswith("WARNING: latest EBIT margin 0.28%"))
        self.assertIn("is below a third of the prior median 2.67% after 3 profitable years, on a "
                      "+12.9% revenue change", note)
        self.assertIn("fades to the prior median 2.67% by year 5", note)
        fade = margin_fade_target(self.jd.financials, self.jd.financials.revenue)
        self.assertEqual(fade["rule"], "collapse")
        held = run_dcf(self.jd, self.macro, DCFAssumptions(target_ebit_margin=0.0028), self.price)
        self.assertGreater(dcf.implied_price, held.implied_price)
        self.assertFalse(any("below a third" in n for n in held.assumptions["notes"]))
        report = value_company("JD", provider=_OneCompanyProvider(self.jd), run_comps=False)
        self.assertTrue(any(w.startswith("WARNING: DCF: latest EBIT margin 0.28% is below a third")
                            for w in report.warnings))
        # The FCFE projects net income, which did not collapse: unchanged.
        self.assertNotIn("net_margin_path",
                         run_fcfe(self.jd, self.macro, DDMAssumptions(), self.price).detail)

    def test_margin_grid_centres_on_the_target(self):
        headline = run_dcf(self.jd, self.macro, DCFAssumptions(), self.price).implied_price
        grid = dcf_sensitivity(self.jd, self.macro, DCFAssumptions(), self.price)[1]
        self.assertAlmostEqual(grid.row_values[2], 0.0267, places=9)
        self.assertAlmostEqual(grid.grid[2][2], headline, places=9)

    def test_a_charge_year_is_still_named_as_one(self):
        fade = margin_fade_target(_f_shaped().financials, _f_shaped().financials.revenue)
        self.assertEqual(fade["rule"], "charge")
        self.assertIsNone(margin_fade_target(make_company().financials,
                                             make_company().financials.revenue))


class WarningPrefixTests(unittest.TestCase):
    def test_a_model_warning_note_keeps_its_prefix_in_front(self):
        report = SimpleNamespace(warnings=[])
        _add_notes(report, "DCF", ["WARNING: charge year", "plain note", "WARNING: charge year"])
        self.assertEqual(report.warnings, ["WARNING: DCF: charge year", "DCF: plain note"])



class TerminalCashFlowTests(unittest.TestCase):
    def setUp(self):
        self.company = make_company()
        self.macro = MacroAssumptions()

    def dcf(self, assumptions, company=None):
        return run_dcf(company or self.company, self.macro, assumptions, self.company.market.price)

    def test_clamped_terminal_growth_also_ends_automatic_fade(self):
        result = self.dcf(DCFAssumptions(terminal_growth=0.20))
        self.assertAlmostEqual(result.assumptions["revenue_growth_path"][-1],
                               result.assumptions["terminal_growth_used"])
        self.assertLess(result.assumptions["terminal_growth_used"], 0.20)

    def test_terminal_working_capital_uses_stable_revenue_change(self):
        a = DCFAssumptions(revenue_growth=[0.20] * 3, forecast_years=3,
                           nwc_pct_revenue=0.40, capex_pct_revenue=0.05,
                           da_pct_revenue=0.04, tax_rate=0.21, terminal_growth=0.025)
        result = self.dcf(a)
        self.assertEqual(result.assumptions["revenue_growth_path"], [0.20] * 3)
        stable_revenue = result.revenue[-1] * 1.025
        stable_ebit = stable_revenue * result.assumptions["ebit_margin_path"][-1]
        stable_fcff = stable_ebit * 0.79 + stable_revenue * (0.04 - 0.05) \
            - result.revenue[-1] * 0.025 * 0.40
        self.assertAlmostEqual(result.assumptions["terminal_fcff"] / stable_fcff, 1.0)
        self.assertAlmostEqual(result.terminal_value * (result.wacc.wacc - 0.025)
                               / stable_fcff, 1.0)

    def test_moving_one_stable_year_into_forecast_preserves_value(self):
        # This identity checks the boundary between explicit and perpetual
        # periods, including a growth-capex fade and working capital.
        c = replace(self.company, financials=replace(
            self.company.financials,
            capex=[r * 0.20 for r in self.company.financials.revenue],
        ))
        a = DCFAssumptions(forecast_years=2, revenue_growth=[0.20, 0.20],
                           terminal_growth=0.025, mid_year_convention=False,
                           nwc_pct_revenue=0.40)
        short = self.dcf(a, c)
        longer = self.dcf(replace(a, forecast_years=3, revenue_growth=[0.20, 0.20, 0.025]), c)
        self.assertAlmostEqual(short.enterprise_value / longer.enterprise_value, 1.0, places=12)

    def test_invalid_explicit_growth_is_not_silently_removed(self):
        for growth in ([0.10, float("nan"), 0.02], [-1.0], [-1.5]):
            with self.subTest(growth=growth), self.assertRaises(ValueError):
                self.dcf(DCFAssumptions(revenue_growth=growth))

    def test_invalid_exit_multiple_and_terminal_growth_fail(self):
        for multiple in (0.0, -2.0):
            with self.subTest(multiple=multiple), self.assertRaises(ValueError):
                self.dcf(DCFAssumptions(terminal_method="exit_multiple", exit_ev_ebitda=multiple))
        with self.assertRaises(ValueError):
            self.dcf(DCFAssumptions(terminal_growth=-1.0))

    def test_negative_share_fallback_does_not_reverse_equity_value(self):
        c = replace(self.company,
                    market=replace(self.company.market, shares_outstanding=0.0),
                    financials=replace(self.company.financials, diluted_shares=[-100.0]))
        result = self.dcf(DCFAssumptions(), c)
        self.assertGreater(result.equity_value, 0)
        self.assertEqual(result.implied_price, 0.0)
        self.assertEqual(result.shares, 0.0)

class DividendDomainTests(unittest.TestCase):
    def setUp(self):
        self.company = make_company()
        self.macro = MacroAssumptions()

    def test_finite_high_growth_can_exceed_discount_rate(self):
        a = DDMAssumptions(high_growth_rate=0.25, high_growth_years=3, terminal_growth=0.025)
        result = run_ddm(self.company, self.macro, a, self.company.market.price)
        ke, d0 = result.cost_of_equity, self.company.market.dividend_per_share
        self.assertLess(ke, 0.25)
        expected = sum(d0 * 1.25 ** t / (1.0 + ke) ** t for t in range(1, 4))
        expected += d0 * 1.25 ** 3 * 1.025 / (ke - 0.025) / (1.0 + ke) ** 3
        self.assertAlmostEqual(result.implied_price, expected)
        self.assertEqual(result.detail["high_growth"], 0.25)

    def test_h_model_keeps_finite_high_growth(self):
        a = DDMAssumptions(method=" H_Model ", high_growth_rate=0.25,
                           high_growth_years=4, terminal_growth=0.025)
        result = run_ddm(self.company, self.macro, a, self.company.market.price)
        d0 = self.company.market.dividend_per_share
        expected = d0 * (1.025 + 2 * (0.25 - 0.025)) / (result.cost_of_equity - 0.025)
        self.assertAlmostEqual(result.implied_price, expected)

    def test_nonpositive_discount_rate_is_unavailable(self):
        macro = MacroAssumptions(risk_free_rate=-0.20, equity_risk_premium=0.01)
        for model in (run_ddm, run_fcfe):
            with self.subTest(model=model.__name__), self.assertRaises(ValueError):
                model(self.company, macro, DDMAssumptions(), self.company.market.price)

    def test_impossible_growth_and_negative_dividend_are_unavailable(self):
        for model in (run_ddm, run_fcfe):
            with self.subTest(model=model.__name__), self.assertRaises(ValueError):
                model(self.company, self.macro, DDMAssumptions(terminal_growth=-1.0), 30)
        with self.assertRaises(ValueError):
            run_ddm(self.company, self.macro, DDMAssumptions(high_growth_rate=-1.0), 30)
        c = replace(self.company, market=replace(self.company.market, dividend_per_share=-1.0))
        self.assertIsNone(run_ddm(c, self.macro, DDMAssumptions(), 30))

class PeerWeightingTests(unittest.TestCase):
    def test_duplicate_peer_requests_and_rows_get_one_vote(self):
        class Provider:
            def get_peer_comp_rows(self, tickers):
                self.requested = tickers
                return [CompRow("P1", "One", pe=10), CompRow(" p1 ", "One", pe=10),
                        CompRow("P2", "Two", pe=20), CompRow("P3", "Three", pe=30)]

        provider = Provider()
        company = make_company()
        result = run_comps(company, provider, [" p1 ", "P1", "P2", "P3", " synt "], 30)
        self.assertEqual(provider.requested, ["P1", "P2", "P3"])
        self.assertEqual(len(result.peers), 3)
        self.assertEqual(result.stats["pe"]["median"], 20)

    def test_earnings_cagr_uses_elapsed_fiscal_years(self):
        fin = replace(make_company().financials, fiscal_years=[2019, 2022, 2025],
                      net_income=[100, 133.1, 177.1561], diluted_shares=[10, 10, 10])
        self.assertAlmostEqual(_eps_cagr(fin), 0.10)

    def test_eps_growth_accounts_for_dilution_and_peg_stays_reference_only(self):
        company = make_company()
        fin = replace(company.financials, fiscal_years=[2020, 2025],
                      net_income=[100.0, 200.0], diluted_shares=[10.0, 20.0])
        self.assertEqual(_eps_cagr(fin), 0.0)
        comps = run_comps(company, SyntheticProvider(), DEMO_PEERS, company.market.price)
        self.assertIsNotNone(comps.target.peg)
        self.assertIsNone(comps.implied["peg"])
        self.assertEqual(comps.implied_price_summary["median"],
                         median([v for v in comps.implied.values() if v is not None]))
        self.assertTrue(any("growth bases are not comparable" in n for n in comps.notes))

if __name__ == "__main__":
    unittest.main()
