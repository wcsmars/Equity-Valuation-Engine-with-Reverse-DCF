"""Pin the valuation formulas on the synthetic company.

``tests.test_synthetic`` checks signs, bounds, and ordering. These tests check
the arithmetic itself, so a refactor that quietly changes a discounting
convention or drops a term from the enterprise-to-equity bridge fails loudly.
Run with:  python -m unittest tests.test_models
"""

from __future__ import annotations

from dataclasses import replace
import math
import unittest

from equity_valuation.data.synthetic import DEMO_PEERS, SyntheticProvider, make_company
from equity_valuation.models.comps import run_comps
from equity_valuation.models.dcf import run_dcf
from equity_valuation.models.ddm_fcfe import run_ddm, run_fcfe
from equity_valuation.models.wacc import compute_wacc
from equity_valuation.schemas import DCFAssumptions, DDMAssumptions, MacroAssumptions


def _close(a: float, b: float, rel: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=1e-6)


class WACCFormulaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.company = make_company()
        self.macro = MacroAssumptions()
        self.wacc = compute_wacc(self.company, self.macro)

    def test_capm_cost_of_equity(self) -> None:
        expected = self.macro.risk_free_rate + self.company.market.beta * self.macro.equity_risk_premium
        self.assertTrue(_close(self.wacc.cost_of_equity, expected))

    def test_wacc_is_weighted_average(self) -> None:
        w = self.wacc
        self.assertTrue(_close(w.weight_equity + w.weight_debt, 1.0))
        expected = w.weight_equity * w.cost_of_equity + w.weight_debt * w.after_tax_cost_of_debt
        self.assertTrue(_close(w.wacc, expected))


class DCFFormulaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.company = make_company()
        self.macro = MacroAssumptions()
        self.price = self.company.market.price

    def test_mid_year_discount_factors(self) -> None:
        dcf = run_dcf(self.company, self.macro, DCFAssumptions(), self.price)
        w = dcf.assumptions["wacc"]
        exponents = dcf.assumptions["discount_exponents"]
        self.assertEqual(exponents, [t + 0.5 for t in range(len(dcf.years))])
        for df, exponent in zip(dcf.discount_factors, exponents):
            self.assertTrue(_close(df, (1.0 + w) ** -exponent))

    def test_gordon_terminal_value_and_timing(self) -> None:
        dcf = run_dcf(self.company, self.macro, DCFAssumptions(), self.price)
        w = dcf.assumptions["wacc"]
        g = dcf.assumptions["terminal_growth_used"]
        n = dcf.assumptions["forecast_years"]
        self.assertGreater(w, g)
        self.assertTrue(_close(dcf.terminal_value, dcf.fcff[-1] * (1.0 + g) / (w - g)))
        # A Gordon perpetuity shares the last explicit flow's mid-year timing.
        self.assertEqual(dcf.assumptions["terminal_discount_exponent"], n - 0.5)
        self.assertTrue(_close(dcf.pv_terminal, dcf.terminal_value / (1.0 + w) ** (n - 0.5)))
        self.assertTrue(_close(dcf.enterprise_value, sum(dcf.pv_fcff) + dcf.pv_terminal))

    def test_exit_multiple_terminal_value_discounted_at_full_period(self) -> None:
        dcf = run_dcf(
            self.company, self.macro,
            DCFAssumptions(terminal_method="exit_multiple", exit_ev_ebitda=12.0),
            self.price,
        )
        w = dcf.assumptions["wacc"]
        n = dcf.assumptions["forecast_years"]
        # EBITDA_N rebuilt independently: final-year revenue x (final EBIT
        # margin + the pooled historical D&A share), so a wrong EBITDA (EBIT
        # only, another year, half the D&A) fails here, not just the TV.
        fin = self.company.financials
        da_pct = sum(fin.dep_amort) / sum(fin.revenue)
        ebitda_n = dcf.revenue[-1] * (dcf.assumptions["ebit_margin_path"][-1] + da_pct)
        self.assertTrue(_close(dcf.assumptions["ebitda_terminal"], ebitda_n))
        self.assertTrue(_close(dcf.terminal_value, ebitda_n * 12.0))
        self.assertAlmostEqual(dcf.implied_price, 39.2443, places=3)  # demo at 12x
        # An exit-multiple TV is a year-end sale value: full period N, not N-0.5.
        self.assertEqual(dcf.assumptions["terminal_discount_exponent"], float(n))
        self.assertTrue(_close(dcf.pv_terminal, dcf.terminal_value / (1.0 + w) ** n))

    def test_equity_bridge_subtracts_minority_and_preferred(self) -> None:
        base = run_dcf(self.company, self.macro, DCFAssumptions(), self.price)
        minority, preferred = 5_000_000_000.0, 2_000_000_000.0
        levered = replace(
            self.company,
            balance_sheet=replace(
                self.company.balance_sheet,
                minority_interest=minority, preferred_equity=preferred,
            ),
        )
        dcf = run_dcf(levered, self.macro, DCFAssumptions(), self.price)
        # Non-common claims do not enter WACC, so enterprise value is unchanged.
        self.assertTrue(_close(dcf.enterprise_value, base.enterprise_value))
        self.assertTrue(_close(
            dcf.equity_value,
            dcf.enterprise_value - dcf.net_debt - minority - preferred,
        ))
        self.assertTrue(_close(dcf.implied_price, dcf.equity_value / dcf.shares))
        self.assertTrue(_close(
            base.implied_price - dcf.implied_price,
            (minority + preferred) / dcf.shares,
        ))


class CompsBridgeTests(unittest.TestCase):
    def test_ev_multiples_subtract_minority_and_preferred(self) -> None:
        # The comps EV->equity bridge, like the DCF's: EV multiples lose
        # (minority + preferred) / shares, equity multiples do not move, and
        # the target's own EV carries both claims.
        company = make_company()
        price = company.market.price
        base = run_comps(company, SyntheticProvider(), DEMO_PEERS, price)
        minority, preferred = 5_000_000_000.0, 2_000_000_000.0
        levered = replace(company, balance_sheet=replace(
            company.balance_sheet, minority_interest=minority, preferred_equity=preferred))
        comps = run_comps(levered, SyntheticProvider(), DEMO_PEERS, price)
        shares = company.market.shares_outstanding
        for m in ("ev_ebitda", "ev_sales"):
            self.assertIsNotNone(base.implied[m], m)
            self.assertAlmostEqual(base.implied[m] - comps.implied[m],
                                   (minority + preferred) / shares, places=9, msg=m)
        for m in ("pe", "pb"):
            self.assertEqual(comps.implied[m], base.implied[m], m)
        self.assertAlmostEqual(comps.target.enterprise_value - base.target.enterprise_value,
                               minority + preferred, delta=1.0)


class DDMFormulaTests(unittest.TestCase):
    def test_perpetual_growth_at_or_above_ke_is_clamped(self) -> None:
        # Every perpetuity needs ke - g >= 0.005: a terminal growth above ke
        # (or within the spread) is clamped to ke - 0.005, with a note, in all
        # three DDM forms and in the FCFE (fade path and terminal value).
        company, macro = make_company(), MacroAssumptions()
        price = company.market.price
        ke = run_ddm(company, macro, DDMAssumptions(method="gordon"), price).cost_of_equity
        d0 = company.market.dividend_per_share
        for g in (0.20, ke - 0.001):
            for method, key in (("gordon", "growth"), ("h_model", "terminal_growth_used"),
                                ("two_stage", "terminal_growth_used")):
                with self.subTest(g=g, method=method):
                    r = run_ddm(company, macro, DDMAssumptions(method=method, terminal_growth=g),
                                price)
                    self.assertAlmostEqual(r.detail[key], ke - 0.005, places=12)
                    self.assertTrue(any("clamped to" in n for n in r.detail["notes"]))
                    self.assertGreater(r.implied_price, 0)
                    if method == "gordon":
                        self.assertTrue(_close(r.implied_price, d0 * (1 + ke - 0.005) / 0.005))
            with self.subTest(g=g, model="fcfe"):
                f = run_fcfe(company, macro, DDMAssumptions(terminal_growth=g), price)
                self.assertAlmostEqual(f.detail["terminal_growth_used"], ke - 0.005, places=12)
                self.assertAlmostEqual(f.detail["revenue_growth_path"][-1], ke - 0.005, places=12)
                self.assertGreater(f.terminal_value, 0)
                self.assertTrue(any("clamped to" in n for n in f.detail["notes"]))

    def test_gordon_growth_uses_next_dividend(self) -> None:
        company = make_company()
        macro = MacroAssumptions()
        ddm = run_ddm(company, macro, DDMAssumptions(method="gordon"), company.market.price)
        self.assertEqual(ddm.method, "gordon")
        g = ddm.detail["growth"]
        ke = ddm.cost_of_equity
        d0 = company.market.dividend_per_share
        self.assertGreater(ke, g)
        self.assertTrue(_close(ddm.implied_price, d0 * (1.0 + g) / (ke - g)))


if __name__ == "__main__":
    unittest.main()
