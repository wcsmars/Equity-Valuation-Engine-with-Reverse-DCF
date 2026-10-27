"""Offline tests for the opt-in trailing-twelve-month comps basis.

Peers' multiples are Yahoo's trailing ones. By default comps apply them to the
target's latest fiscal-year revenue, EBITDA and net income, and a note flags a
trailing net income or EBITDA more than 15% away from the fiscal year's;
``run_comps(basis="ttm")`` (``value_company(comps_basis="ttm")``, the CLI's
``--comps-basis ttm``) applies them to Yahoo's trailing figures instead,
which ``YFinanceClient.get_market_data`` copies from the ``.info`` it already
fetches (``MarketData.ttm_*``), only when Yahoo's reporting currency is the
quote currency. Covered: the market-data fields (epoch dates to ISO, a
different reporting currency, NaN and missing values, a pence quote, the
hybrid provider's backfill copy), hand-computed implied prices on the
trailing basis, the gap note's threshold and direction, stale trailing
figures, a latest quarter that is the fiscal year end (and the newer year
when the statements still end at the prior one), a per-metric fallback, a
different reporting currency, the fiscal-year collapse screen, a lender's
P/E-only gap note and display-only EV metrics, a trailing loss on either
basis, the gap note's "mostly below EBITDA" wording (one-off items, which the
trailing figures are not screened for), datetime dates, the unchanged default
demo, an unknown basis, the engine warnings, the backend payload and the CLI
flag. "Today" is pinned wherever the trailing figures' age matters. No
network access is needed.
Run with:  python -m unittest tests.test_ttm_comps
"""

from __future__ import annotations

import contextlib
import datetime
import io
import tempfile
import unittest
from dataclasses import replace
from unittest import mock

from backend.serialization import report_to_dict
from equity_valuation import value_company
from equity_valuation.cli import build_parser, main
from equity_valuation.data.provider import HybridProvider
from equity_valuation.data.synthetic import (
    DEMO_PEERS,
    DEMO_TICKER,
    SyntheticProvider,
    make_company,
)
from equity_valuation.models import comps as comps_mod
from equity_valuation.models.comps import (
    COMPS_BASES,
    COMPS_GAP_NOTE_PREFIX,
    COMPS_TTM_NOTE_PREFIX,
    EARNINGS_COLLAPSE_NOTE_PREFIX,
    TTM_GAP_THRESHOLD,
    TTM_MAX_AGE_DAYS,
    run_comps,
)
from equity_valuation.schemas import MarketData
from tests.test_market_data import client, ticker

NaN = float("nan")
TODAY = datetime.date(2025, 6, 1)  # pinned "today": the synthetic FY2024 ended 2024-12-31
AS_OF = "2025-03-31"               # the latest quarter Yahoo reports
FY_END = "2024-12-31"
# The synthetic peers' medians: EV/EBITDA 20, EV/Sales 4.8, P/E 21, P/B 5.8.
MED_EV_EBITDA, MED_EV_SALES, MED_PE, MED_PB = 20.0, 4.8, 21.0, 5.8
NET_DEBT = 20e9 - 8e9  # the synthetic balance sheet: debt 20B, cash 8B
SHARES = 10e9
EQUITY = 60e9
# The gap note's cause clauses and its closing words.
CAUSE = "the gap may reflect the business or one-off items (gains, impairments, tax)"
BELOW = ("the net income gap is mostly below EBITDA, so it may be non-operating or one-off "
         "items (gains, impairments, tax)")
TAIL = ("(an opt-in trailing basis, --comps-basis ttm in the CLI, uses Yahoo's trailing "
        "figures, which are not screened for one-off items).")
BANK = {"industry": "Banks - Diversified", "sector": "Financial Services"}


def _epoch(day: str) -> int:
    """Epoch seconds of `day` at midnight UTC, as Yahoo's date fields carry it."""
    d = datetime.date.fromisoformat(day)
    return int(datetime.datetime(d.year, d.month, d.day,
                                 tzinfo=datetime.timezone.utc).timestamp())


def _company(ni=1.0, ebitda=1.0, revenue=1.0, **market):
    """The synthetic company with Yahoo trailing figures at the given multiples
    of its FY2024 ones (None leaves that field out), dated AS_OF after FY_END."""
    co = make_company()
    fin = co.financials
    fields = {"ttm_as_of": AS_OF, "fiscal_year_end": FY_END, "ttm_currency": "USD"}
    for name, factor, series in (("ttm_net_income", ni, fin.net_income),
                                 ("ttm_ebitda", ebitda, fin.ebitda),
                                 ("ttm_revenue", revenue, fin.revenue)):
        fields[name] = series[-1] * factor if factor is not None else None
    fields.update(market)
    co.market = replace(co.market, **fields)
    return co


def _comps(company, basis="fy", today=TODAY):
    with mock.patch.object(comps_mod, "_today", return_value=today):
        return run_comps(company, SyntheticProvider(), DEMO_PEERS, company.market.price,
                         basis=basis)


def _with(notes, prefix):
    return [n for n in notes or [] if str(n).startswith(prefix)]


class _Provider(SyntheticProvider):
    """The synthetic peers around a given target company."""

    def __init__(self, company):
        self.company = company

    def get_company_data(self, ticker):
        return self.company


# --------------------------------------------------------------------------- #
#  Market data
# --------------------------------------------------------------------------- #
class MarketDataTrailingFieldsTests(unittest.TestCase):
    BASE = {"currency": "USD", "currentPrice": 100.0, "sharesOutstanding": 10.0,
            "marketCap": 1000.0}

    def _md(self, **info) -> MarketData:
        return client({"X": ticker({**self.BASE, **info})}).get_market_data("X")

    def test_trailing_fields_and_dates_are_read(self) -> None:
        md = self._md(financialCurrency="USD", totalRevenue=500.0, ebitda=120.0,
                      netIncomeToCommon=80.0, mostRecentQuarter=_epoch("2026-07-26"),
                      lastFiscalYearEnd=1780185600)
        self.assertEqual((md.ttm_revenue, md.ttm_ebitda, md.ttm_net_income),
                         (500.0, 120.0, 80.0))
        self.assertEqual(md.ttm_as_of, "2026-07-26")
        self.assertEqual(md.fiscal_year_end, "2026-05-31")  # epoch seconds, UTC
        self.assertEqual(md.ttm_currency, "USD")

    def test_a_negative_trailing_net_income_is_kept(self) -> None:
        md = self._md(financialCurrency="USD", netIncomeToCommon=-40.0)
        self.assertEqual(md.ttm_net_income, -40.0)

    def test_other_reporting_currency_leaves_the_figures_none(self) -> None:
        # TSM: TWD statements behind a USD quote; the trailing figures are not converted.
        md = self._md(financialCurrency="TWD", totalRevenue=3.0e12, ebitda=2.0e12,
                      netIncomeToCommon=1.5e12, mostRecentQuarter=_epoch("2026-06-30"))
        self.assertIsNone(md.ttm_revenue)
        self.assertIsNone(md.ttm_ebitda)
        self.assertIsNone(md.ttm_net_income)
        self.assertEqual(md.ttm_currency, "TWD")
        self.assertEqual(md.ttm_as_of, "2026-06-30")

    def test_missing_reporting_currency_leaves_the_figures_none(self) -> None:
        md = self._md(totalRevenue=500.0, ebitda=120.0, netIncomeToCommon=80.0)
        self.assertIsNone(md.ttm_revenue)
        self.assertIsNone(md.ttm_net_income)
        self.assertIsNone(md.ttm_currency)

    def test_nan_strings_and_missing_fields_give_none(self) -> None:
        md = self._md(financialCurrency="USD", totalRevenue=NaN, ebitda="Infinity",
                      mostRecentQuarter=None, lastFiscalYearEnd="n/a")
        self.assertIsNone(md.ttm_revenue)
        self.assertIsNone(md.ttm_ebitda)
        self.assertIsNone(md.ttm_net_income)  # netIncomeToCommon missing
        self.assertIsNone(md.ttm_as_of)
        self.assertIsNone(md.fiscal_year_end)

    def test_pence_quote_with_pound_statements_is_the_same_currency(self) -> None:
        md = client({"ULVR.L": ticker({"currency": "GBp", "financialCurrency": "GBP",
                                        "currentPrice": 4500.0, "sharesOutstanding": 10.0,
                                        "marketCap": 450.0, "totalRevenue": 600.0,
                                        "ebitda": 110.0, "netIncomeToCommon": 60.0})}
                     ).get_market_data("ULVR.L")
        self.assertEqual(md.currency, "GBP")
        self.assertEqual((md.ttm_revenue, md.ttm_ebitda, md.ttm_net_income),
                         (600.0, 110.0, 60.0))

    def test_hybrid_backfill_copy_keeps_the_trailing_fields(self) -> None:
        co = _company(ni=1.2)
        market = replace(co.market, shares_outstanding=0.0, market_cap=0.0)
        out = HybridProvider._backfill_market(market, co.financials, False, [])
        self.assertIsNot(out, market)
        self.assertGreater(out.shares_outstanding, 0)
        self.assertEqual(out.ttm_net_income, co.market.ttm_net_income)
        self.assertEqual((out.ttm_as_of, out.fiscal_year_end, out.ttm_currency),
                         (AS_OF, FY_END, "USD"))

    def test_synthetic_provider_has_no_trailing_figures(self) -> None:
        md = SyntheticProvider().get_market_data(DEMO_TICKER)
        self.assertIsNone(md.ttm_revenue)
        self.assertIsNone(md.ttm_ebitda)
        self.assertIsNone(md.ttm_net_income)
        self.assertIsNone(md.ttm_as_of)
        self.assertIsNone(md.fiscal_year_end)


# --------------------------------------------------------------------------- #
#  Comps on the trailing basis
# --------------------------------------------------------------------------- #
class TrailingBasisTests(unittest.TestCase):
    def test_implied_prices_use_the_trailing_figures(self) -> None:
        co = _company(ni=1.2, ebitda=1.1, revenue=1.05)
        m = co.market
        r = _comps(co, "ttm")
        self.assertAlmostEqual(r.implied["pe"], MED_PE * m.ttm_net_income / SHARES)
        self.assertAlmostEqual(r.implied["ev_ebitda"],
                               (MED_EV_EBITDA * m.ttm_ebitda - NET_DEBT) / SHARES)
        self.assertAlmostEqual(r.implied["ev_sales"],
                               (MED_EV_SALES * m.ttm_revenue - NET_DEBT) / SHARES)
        self.assertAlmostEqual(r.implied["pb"], MED_PB * EQUITY / SHARES)  # book value as is
        # The target's own row is on the same basis.
        self.assertAlmostEqual(r.target.pe, m.price / (m.ttm_net_income / SHARES))
        self.assertAlmostEqual(r.target.ev_ebitda, r.target.enterprise_value / m.ttm_ebitda)
        self.assertAlmostEqual(r.target.ev_sales, r.target.enterprise_value / m.ttm_revenue)
        self.assertIn(
            f"{COMPS_TTM_NOTE_PREFIX} to {AS_OF} (Yahoo) for the target's revenue, EBITDA and "
            "net income, matching peers' trailing multiples; the DCF stays on fiscal year "
            "2024.", r.notes)
        self.assertEqual(_with(r.notes, COMPS_GAP_NOTE_PREFIX), [])

    def test_hand_computed_pe_price(self) -> None:
        co = _company()
        co.market = replace(co.market, ttm_net_income=25e9)
        r = _comps(co, "ttm")
        self.assertAlmostEqual(r.implied["pe"], 21.0 * 2.5)  # 21x a TTM EPS of 2.50

    def test_a_missing_trailing_ebitda_falls_back_for_ev_ebitda_only(self) -> None:
        co = _company(ni=1.2, ebitda=None, revenue=1.05)
        fy = _comps(make_company(), "fy")
        r = _comps(co, "ttm")
        self.assertAlmostEqual(r.implied["ev_ebitda"], fy.implied["ev_ebitda"])
        self.assertAlmostEqual(r.implied["pe"], MED_PE * co.market.ttm_net_income / SHARES)
        self.assertAlmostEqual(r.implied["ev_sales"],
                               (MED_EV_SALES * co.market.ttm_revenue - NET_DEBT) / SHARES)
        self.assertIn(
            f"{COMPS_TTM_NOTE_PREFIX} to {AS_OF} (Yahoo) for the target's revenue and net "
            "income, matching peers' trailing multiples; the DCF stays on fiscal year 2024.",
            r.notes)
        self.assertIn(
            f"{COMPS_TTM_NOTE_PREFIX}: Yahoo gives no trailing EBITDA for the target, so "
            "EV/EBITDA uses the fiscal year 2024 figure.", r.notes)
        two = _comps(_company(ni=1.2, ebitda=None, revenue=None), "ttm")
        self.assertIn(
            f"{COMPS_TTM_NOTE_PREFIX}: Yahoo gives no trailing revenue and EBITDA for the "
            "target, so EV/Sales and EV/EBITDA use the fiscal year 2024 figures.", two.notes)

    def test_a_lenders_missing_trailing_ebitda_is_not_named(self) -> None:
        # JPM: Yahoo has no EBITDA for a bank, whose EV multiples give no price.
        co = _company(ni=1.2, ebitda=None, industry="Banks - Diversified",
                      sector="Financial Services")
        r = _comps(co, "ttm")
        self.assertIsNone(r.implied["ev_ebitda"])
        self.assertAlmostEqual(r.implied["pe"], MED_PE * co.market.ttm_net_income / SHARES)
        self.assertEqual(len(_with(r.notes, COMPS_TTM_NOTE_PREFIX)), 1)  # the basis note only

    def test_a_lenders_ev_metrics_are_display_only(self) -> None:
        # JPM: EV/Sales gives a bank no price; its trailing revenue only changes
        # the displayed EV/Sales.
        r = _comps(_company(ni=1.2, ebitda=None, revenue=1.1, **BANK), "ttm")
        self.assertIsNone(r.implied["ev_sales"])
        self.assertAlmostEqual(r.target.ev_sales,
                               r.target.enterprise_value / (1.1 * make_company().financials
                                                            .revenue[-1]))
        self.assertIn(
            f"{COMPS_TTM_NOTE_PREFIX} to {AS_OF} (Yahoo) for the target's revenue (display "
            "only) and net income, matching peers' trailing multiples; the DCF stays on "
            "fiscal year 2024.", r.notes)
        both = _comps(_company(ni=1.2, ebitda=1.1, **BANK), "ttm")
        self.assertTrue(_with(both.notes, f"{COMPS_TTM_NOTE_PREFIX} to {AS_OF} (Yahoo) for the "
                              "target's revenue (display only), EBITDA (display only) and net "
                              "income, "))

    def test_a_trailing_loss_gives_no_pe_price_and_says_so(self) -> None:
        co = _company(ni=-0.5)
        r = _comps(co, "ttm")
        self.assertIsNone(r.implied["pe"])
        self.assertIsNone(r.target.pe)
        self.assertEqual(_with(r.notes, COMPS_TTM_NOTE_PREFIX), [
            f"{COMPS_TTM_NOTE_PREFIX} to {AS_OF} (Yahoo) for the target's revenue, EBITDA and "
            "net income (a loss, so P/E gives no price), matching peers' trailing multiples; "
            "the DCF stays on fiscal year 2024."])
        self.assertEqual(_with(r.notes, COMPS_GAP_NOTE_PREFIX), [])
        neg = _comps(_company(ebitda=-0.1, ni=0.0), "ttm")
        self.assertIsNone(neg.implied["ev_ebitda"])
        self.assertIsNone(neg.implied["pe"])
        self.assertTrue(_with(neg.notes, f"{COMPS_TTM_NOTE_PREFIX} to {AS_OF} (Yahoo) for the "
                              "target's revenue, EBITDA (negative, so EV/EBITDA gives no price) "
                              "and net income (zero, so P/E gives no price), "))

    def test_stale_trailing_figures_fall_back_to_the_fiscal_year(self) -> None:
        co = _company(ni=1.6, ebitda=1.4)
        fy = _comps(make_company(), "fy")
        late = datetime.date.fromisoformat(AS_OF) + datetime.timedelta(days=TTM_MAX_AGE_DAYS + 1)
        r = _comps(co, "ttm", today=late)
        self.assertEqual(r.implied, fy.implied)
        notes = _with(r.notes, COMPS_TTM_NOTE_PREFIX)
        self.assertEqual(len(notes), 1)
        self.assertTrue(notes[0].startswith(f"{COMPS_TTM_NOTE_PREFIX} not used: "))
        self.assertIn("stale", notes[0])
        self.assertIn(f"{TTM_MAX_AGE_DAYS + 1} days ago", notes[0])
        self.assertTrue(notes[0].endswith("comps use the target's fiscal year 2024 figures."))
        # The fiscal-year basis gives no gap note on stale figures either.
        self.assertEqual(_with(_comps(co, "fy", today=late).notes, COMPS_GAP_NOTE_PREFIX), [])

    def test_trailing_figures_at_the_age_limit_are_used(self) -> None:
        edge = datetime.date.fromisoformat(AS_OF) + datetime.timedelta(days=TTM_MAX_AGE_DAYS)
        r = _comps(_company(ni=1.2), "ttm", today=edge)
        self.assertTrue(_with(r.notes, f"{COMPS_TTM_NOTE_PREFIX} to {AS_OF}"))

    def test_latest_quarter_at_the_fiscal_year_end_uses_the_fiscal_year(self) -> None:
        co = _company(ni=1.3, ttm_as_of=FY_END)
        fy = _comps(make_company(), "fy")
        r = _comps(co, "ttm")
        self.assertEqual(r.implied, fy.implied)
        notes = _with(r.notes, COMPS_TTM_NOTE_PREFIX)
        self.assertEqual(notes, [
            f"{COMPS_TTM_NOTE_PREFIX} not used: the latest quarter Yahoo reports ({FY_END}) is "
            f"the fiscal year end ({FY_END}), so its trailing twelve months are that fiscal "
            "year; comps use the target's fiscal year 2024 figures."])
        before = _comps(_company(ni=1.3, ttm_as_of="2024-09-30"), "ttm")
        self.assertEqual(before.implied, fy.implied)
        self.assertIn("ends before the fiscal year end", _with(before.notes,
                                                               COMPS_TTM_NOTE_PREFIX)[0])

    def test_a_newer_year_than_the_statements_is_used(self) -> None:
        # A bank between its year-end results and its 10-K: Yahoo's latest
        # quarter and fiscal year end are 2025-12-31, the statements FY2024.
        day = datetime.date(2026, 1, 20)
        co = _company(ni=1.2, ttm_as_of="2025-12-31", fiscal_year_end="2025-12-31")
        r = _comps(co, "ttm", today=day)
        self.assertAlmostEqual(r.implied["pe"], MED_PE * co.market.ttm_net_income / SHARES)
        self.assertIn(
            f"{COMPS_TTM_NOTE_PREFIX} to 2025-12-31 (Yahoo) for the target's revenue, EBITDA "
            "and net income, matching peers' trailing multiples; the DCF stays on fiscal year "
            "2024.", r.notes)
        gap = _with(_comps(co, "fy", today=day).notes, COMPS_GAP_NOTE_PREFIX)
        self.assertEqual(len(gap), 1)
        self.assertIn("net income is +20.0% vs fiscal year 2024 (Yahoo, to 2025-12-31)", gap[0])

    def test_newer_year_needs_an_older_statement_label(self) -> None:
        def run(years, end):
            co = _company(ni=1.2, ttm_as_of=end, fiscal_year_end=end)
            co.financials = replace(co.financials, fiscal_years=years)
            return _with(_comps(co, "ttm", today=datetime.date(2025, 2, 1)).notes,
                         COMPS_TTM_NOTE_PREFIX)[0]
        # A 52/53-week year ending 2025-01-03 is labelled 2024 by the EDGAR
        # client: statements labelled 2024 may be that year, so it is not used.
        self.assertIn("not used", run([2020, 2021, 2022, 2023, 2024], "2025-01-03"))
        self.assertIn("to 2025-01-03 (Yahoo)", run([2019, 2020, 2021, 2022, 2023],
                                                   "2025-01-03"))
        self.assertIn("not used", run([0, 0, 0, 0, 0], "2024-12-31"))  # no usable label
        self.assertFalse(comps_mod._statements_end_before(None, datetime.date(2025, 1, 31)))

    def test_datetime_dates_are_read_as_dates(self) -> None:
        stamp = datetime.datetime(2025, 3, 31, 16, 30)
        self.assertEqual(comps_mod._iso_date(stamp), datetime.date(2025, 3, 31))
        co = _company(ni=1.2, ttm_as_of=stamp,
                      fiscal_year_end=datetime.datetime(2024, 12, 31))
        r = _comps(co, "ttm")
        self.assertAlmostEqual(r.implied["pe"], MED_PE * co.market.ttm_net_income / SHARES)
        self.assertTrue(_with(r.notes, f"{COMPS_TTM_NOTE_PREFIX} to {AS_OF} (Yahoo)"))

    def test_no_fiscal_year_end_needs_only_a_recent_date(self) -> None:
        r = _comps(_company(ni=1.2, fiscal_year_end=None), "ttm")
        self.assertTrue(_with(r.notes, f"{COMPS_TTM_NOTE_PREFIX} to {AS_OF}"))

    def test_undated_trailing_figures_are_not_used(self) -> None:
        r = _comps(_company(ni=1.2, ttm_as_of="not a date"), "ttm")
        self.assertIn("no date", _with(r.notes, COMPS_TTM_NOTE_PREFIX)[0])
        self.assertEqual(r.implied, _comps(make_company(), "fy").implied)

    def test_other_reporting_currency_is_named(self) -> None:
        co = _company(ni=None, ebitda=None, revenue=None, ttm_currency="TWD")
        r = _comps(co, "ttm")
        self.assertEqual(_with(r.notes, COMPS_TTM_NOTE_PREFIX), [
            f"{COMPS_TTM_NOTE_PREFIX} not used: Yahoo reports the target's trailing figures in "
            "TWD, a different reporting currency from the USD quote, and they are not "
            "converted; comps use the target's fiscal year 2024 figures."])

    def test_no_trailing_figures_on_the_demo(self) -> None:
        demo = make_company()
        r = _comps(demo, "ttm")
        self.assertEqual(r.implied, _comps(demo, "fy").implied)
        self.assertEqual(_with(r.notes, COMPS_TTM_NOTE_PREFIX), [
            f"{COMPS_TTM_NOTE_PREFIX} not used: Yahoo's trailing-twelve-month figures are not "
            "available for the target; comps use the target's fiscal year 2024 figures."])

    def test_the_collapse_screen_reads_the_fiscal_year(self) -> None:
        # FY2024 net income collapsed to 5% of normal; Yahoo's trailing is healthy.
        co = _company()
        ni = list(co.financials.net_income)
        normal = ni[-1]
        ni[-1] = normal * 0.05
        co.financials = replace(co.financials, net_income=ni)
        co.market = replace(co.market, ttm_net_income=normal * 1.1)
        r = _comps(co, "ttm")
        self.assertIsNone(r.implied["pe"])
        self.assertAlmostEqual(r.target.pe, co.market.price / (normal * 1.1 / SHARES))
        collapse = _with(r.notes, EARNINGS_COLLAPSE_NOTE_PREFIX)
        self.assertEqual(len(collapse), 1)
        self.assertIn("(screened on fiscal year 2024, also under the trailing basis)",
                      collapse[0])
        # The fiscal-year basis's wording is unchanged.
        fy = _with(_comps(co, "fy").notes, EARNINGS_COLLAPSE_NOTE_PREFIX)
        self.assertNotIn("screened on", fy[0])

    def test_basis_is_case_insensitive_and_unknown_raises(self) -> None:
        self.assertEqual(COMPS_BASES, ("fy", "ttm"))
        co = _company(ni=1.2)
        self.assertEqual(_comps(co, "TTM").implied, _comps(co, "ttm").implied)
        self.assertEqual(_comps(co, None).implied, _comps(co, "fy").implied)
        for bad in ("ltm", "", 1):
            with self.assertRaises(ValueError):
                _comps(co, bad)


# --------------------------------------------------------------------------- #
#  Fiscal-year basis: the gap note
# --------------------------------------------------------------------------- #
class GapNoteTests(unittest.TestCase):
    def test_threshold(self) -> None:
        self.assertEqual(TTM_GAP_THRESHOLD, 0.15)
        r = _comps(_company(ni=1.2))
        self.assertEqual(_with(r.notes, COMPS_GAP_NOTE_PREFIX), [
            f"{COMPS_GAP_NOTE_PREFIX}: trailing-twelve-month net income is +20.0% vs fiscal "
            f"year 2024 (Yahoo, to {AS_OF}); {CAUSE}; comps apply peers' trailing multiples "
            f"to fiscal-year figures, so the P/E-implied price may be understated {TAIL}"])
        self.assertEqual(_with(_comps(_company(ni=1.1)).notes, COMPS_GAP_NOTE_PREFIX), [])

    def test_fiscal_year_prices_are_unchanged(self) -> None:
        self.assertEqual(_comps(_company(ni=1.6, ebitda=1.4)).implied,
                         _comps(make_company()).implied)

    def test_both_metrics_and_the_direction(self) -> None:
        both = _with(_comps(_company(ni=1.6, ebitda=1.3)).notes, COMPS_GAP_NOTE_PREFIX)[0]
        self.assertIn("net income is +60.0% and EBITDA +30.0% vs fiscal year 2024", both)
        self.assertIn("so the P/E- and EV/EBITDA-implied prices may be understated", both)
        lower = _with(_comps(_company(ebitda=0.8)).notes, COMPS_GAP_NOTE_PREFIX)[0]
        self.assertIn("trailing-twelve-month EBITDA is -20.0% vs", lower)
        self.assertIn("the EV/EBITDA-implied price may be overstated", lower)
        mixed = _with(_comps(_company(ni=1.2, ebitda=0.8)).notes, COMPS_GAP_NOTE_PREFIX)[0]
        self.assertIn("the P/E-implied price may be understated and the EV/EBITDA-implied "
                      "price may be overstated", mixed)

    def test_a_trailing_loss_against_a_fiscal_year_profit(self) -> None:
        # EBITDA flat, so the swing to a loss is below EBITDA.
        r = _comps(_company(ni=-0.5))
        self.assertEqual(r.implied, _comps(make_company()).implied)  # FY prices unchanged
        self.assertEqual(_with(r.notes, COMPS_GAP_NOTE_PREFIX), [
            f"{COMPS_GAP_NOTE_PREFIX}: trailing-twelve-month net income is a loss, against a "
            f"profit in fiscal year 2024 (Yahoo, to {AS_OF}); the net income gap is mostly "
            "below EBITDA (trailing EBITDA +0.0%), so it may be non-operating or one-off items "
            "(gains, impairments, tax); comps apply peers' trailing multiples to fiscal-year "
            f"figures, so the P/E-implied price may be overstated {TAIL}"])
        ebitda = _with(_comps(_company(ebitda=-0.1)).notes, COMPS_GAP_NOTE_PREFIX)[0]
        self.assertIn("trailing-twelve-month EBITDA is negative, against positive EBITDA in "
                      f"fiscal year 2024 (Yahoo, to {AS_OF}); {CAUSE}; ", ebitda)
        self.assertIn("the EV/EBITDA-implied price may be overstated", ebitda)
        mixed = _with(_comps(_company(ni=-0.5, ebitda=0.7)).notes, COMPS_GAP_NOTE_PREFIX)[0]
        self.assertIn("trailing-twelve-month EBITDA is -30.0% vs fiscal year 2024 and net "
                      "income is a loss, against a profit in that year (Yahoo", mixed)
        self.assertIn("the P/E- and EV/EBITDA-implied prices may be overstated", mixed)
        both = _with(_comps(_company(ni=-0.5, ebitda=-0.2)).notes, COMPS_GAP_NOTE_PREFIX)[0]
        self.assertIn("net income is a loss and EBITDA is negative, against a profit and "
                      "positive EBITDA in fiscal year 2024", both)

    def test_a_fiscal_year_loss_gives_no_gap(self) -> None:
        # No positive fiscal-year figure (and no P/E price) to compare with.
        co = _company(ni=1.0)
        ni = list(co.financials.net_income)
        ni[-1] = -abs(ni[-1])
        co.financials = replace(co.financials, net_income=ni)
        co.market = replace(co.market, ttm_net_income=5e9)
        self.assertFalse([n for n in _with(_comps(co).notes, COMPS_GAP_NOTE_PREFIX)
                          if "net income" in n])

    def test_net_income_gap_mostly_below_ebitda(self) -> None:
        # GOOGL on fiscal year 2025: trailing net income +84.7%, EBITDA +15.3%.
        googl = _with(_comps(_company(ni=1.847, ebitda=1.153)).notes, COMPS_GAP_NOTE_PREFIX)
        self.assertEqual(googl, [
            f"{COMPS_GAP_NOTE_PREFIX}: trailing-twelve-month net income is +84.7% and EBITDA "
            f"+15.3% vs fiscal year 2024 (Yahoo, to {AS_OF}); {BELOW}; comps apply peers' "
            "trailing multiples to fiscal-year figures, so the P/E- and EV/EBITDA-implied "
            f"prices may be understated {TAIL}"])
        # An EBITDA gap within the threshold is given in the clause.
        hidden = _with(_comps(_company(ni=1.4, ebitda=1.05)).notes, COMPS_GAP_NOTE_PREFIX)[0]
        self.assertIn("; the net income gap is mostly below EBITDA (trailing EBITDA +5.0%), so "
                      "it may be non-operating or one-off items", hidden)
        self.assertNotIn(CAUSE, hidden)
        # Operating leverage (NVDA: +60.6% against +51.1%; or +30% against
        # +16%) and net income within the threshold keep the general wording.
        for ni, ebitda in ((1.606, 1.511), (1.30, 1.16)):
            note = _with(_comps(_company(ni=ni, ebitda=ebitda)).notes, COMPS_GAP_NOTE_PREFIX)[0]
            self.assertIn(f"; {CAUSE}; ", note)
        ebitda_only = _with(_comps(_company(ni=1.1, ebitda=0.6)).notes, COMPS_GAP_NOTE_PREFIX)[0]
        self.assertIn(f"; {CAUSE}; ", ebitda_only)

    def test_the_note_names_the_cli_flag_as_such(self) -> None:
        # The dashboard and exports show the note too, where no flag can be set.
        note = _with(_comps(_company(ni=1.6)).notes, COMPS_GAP_NOTE_PREFIX)[0]
        self.assertTrue(note.endswith(TAIL))
        self.assertNotIn("is available: --comps-basis ttm)", note)

    def test_no_gap_note_on_the_trailing_basis(self) -> None:
        self.assertEqual(_with(_comps(_company(ni=1.6), "ttm").notes, COMPS_GAP_NOTE_PREFIX),
                         [])

    def test_a_lenders_gap_note_names_pe_only(self) -> None:
        # A bank's EV multiples give no implied price, so the EBITDA gap is moot.
        co = _company(ni=1.3, ebitda=1.3, **BANK)
        r = _comps(co)
        self.assertIsNone(r.implied["ev_ebitda"])
        note = _with(r.notes, COMPS_GAP_NOTE_PREFIX)[0]
        self.assertIn("net income is +30.0% vs", note)
        self.assertNotIn("EBITDA", note)
        # Nor is its net income gap set against its EBITDA.
        flat = _with(_comps(_company(ni=1.4, ebitda=1.0, **BANK)).notes,
                     COMPS_GAP_NOTE_PREFIX)[0]
        self.assertIn(f"; {CAUSE}; ", flat)
        self.assertNotIn("EBITDA", flat)


# --------------------------------------------------------------------------- #
#  Engine, payload and CLI
# --------------------------------------------------------------------------- #
class EngineTests(unittest.TestCase):
    def _value(self, company, **kwargs):
        with mock.patch.object(comps_mod, "_today", return_value=TODAY):
            return value_company(DEMO_TICKER, provider=_Provider(company), peers=DEMO_PEERS,
                                 run_sensitivity=False, run_reverse_dcf=False, **kwargs)

    def test_default_demo_is_unchanged(self) -> None:
        report = value_company(DEMO_TICKER, provider=SyntheticProvider(), peers=DEMO_PEERS)
        self.assertAlmostEqual(report.comps.implied_price_summary["median"], 46.96, places=2)
        self.assertAlmostEqual(report.summary["blended_target"], 32.40, places=2)
        for prefix in (COMPS_TTM_NOTE_PREFIX, COMPS_GAP_NOTE_PREFIX):
            self.assertEqual(_with(report.comps.notes, prefix), [])
            self.assertFalse([w for w in report.warnings if prefix in w])

    def test_trailing_basis_reaches_the_warnings_and_the_blend(self) -> None:
        co = _company(ni=1.6, ebitda=1.4, revenue=1.2)
        fy = self._value(co)
        ttm = self._value(co, comps_basis="ttm")
        self.assertIn(
            f"Comps: {COMPS_TTM_NOTE_PREFIX} to {AS_OF} (Yahoo) for the target's revenue, "
            "EBITDA and net income, matching peers' trailing multiples; the DCF stays on "
            "fiscal year 2024.", ttm.warnings)
        self.assertGreater(ttm.comps.implied_price_summary["median"],
                           fy.comps.implied_price_summary["median"])
        self.assertAlmostEqual(ttm.dcf.implied_price, fy.dcf.implied_price)  # DCF stays FY
        self.assertFalse([w for w in ttm.warnings if COMPS_GAP_NOTE_PREFIX in w])
        gap = [w for w in fy.warnings if w.startswith(f"Comps: {COMPS_GAP_NOTE_PREFIX}: ")]
        self.assertEqual(len(gap), 1)
        self.assertIn("net income is +60.0% and EBITDA +40.0%", gap[0])

    def test_ev_bars_hold_the_implied_price_on_either_basis(self) -> None:
        # Trailing EBITDA and revenue well above the fiscal year's: the EV bars
        # must be priced on the figures comps used, or the trailing price falls
        # outside its own bar.
        co = _company(ni=1.6, ebitda=1.4, revenue=1.2)
        for basis in ("fy", "ttm"):
            report = self._value(co, comps_basis=basis)
            implied = report.comps.implied
            bars = {row.method: row for row in report.football_field}
            for key, label in (("ev_ebitda", "EV/EBITDA comps"), ("ev_sales", "EV/Sales comps")):
                with self.subTest(basis=basis, multiple=key):
                    bar = bars[label]
                    self.assertLessEqual(bar.low - 1e-9, implied[key])
                    self.assertLessEqual(implied[key], bar.high + 1e-9)
                    self.assertAlmostEqual(bar.base, implied[key])
        ttm = self._value(co, comps_basis="ttm").comps.target_metrics
        self.assertAlmostEqual(ttm["ebitda"], co.market.ttm_ebitda)
        self.assertAlmostEqual(ttm["revenue"], co.market.ttm_revenue)
        fy = self._value(co).comps.target_metrics
        self.assertAlmostEqual(fy["ebitda"], co.financials.ebitda[-1])

    def test_unusable_trailing_basis_is_in_the_warnings(self) -> None:
        report = self._value(make_company(), comps_basis="ttm")
        self.assertTrue([w for w in report.warnings
                         if w.startswith(f"Comps: {COMPS_TTM_NOTE_PREFIX} not used: ")])
        self.assertAlmostEqual(report.summary["blended_target"], 32.40, places=2)

    def test_unknown_basis_raises_before_any_fetch(self) -> None:
        provider = mock.Mock(spec=SyntheticProvider)
        with self.assertRaises(ValueError):
            value_company(DEMO_TICKER, provider=provider, comps_basis="quarterly")
        provider.get_company_data.assert_not_called()

    def test_payload_carries_the_trailing_fields(self) -> None:
        d = report_to_dict(self._value(_company(ni=1.2)))
        market = d["company"]["market"]
        self.assertEqual(market["ttm_as_of"], AS_OF)
        self.assertEqual(market["fiscal_year_end"], FY_END)
        self.assertAlmostEqual(market["ttm_net_income"],
                               make_company().financials.net_income[-1] * 1.2)
        demo = report_to_dict(value_company(DEMO_TICKER, provider=SyntheticProvider(),
                                            peers=DEMO_PEERS))
        self.assertIsNone(demo["company"]["market"]["ttm_net_income"])


def _run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = main(argv)
        except SystemExit as exc:
            code = exc.code
    return code, out.getvalue(), err.getvalue()


class CliCompsBasisTests(unittest.TestCase):
    def _kwargs(self, argv):
        with mock.patch("equity_valuation.engine.value_company",
                        side_effect=RuntimeError("stop")) as vc:
            code, _, err = _run_cli(argv + ["--quiet"])
        self.assertEqual(code, 1)
        return vc.call_args.kwargs, err

    def test_flag_is_forwarded(self) -> None:
        kwargs, _ = self._kwargs(["NVDA", "--peers", "AMD", "--comps-basis", "ttm"])
        self.assertEqual(kwargs["comps_basis"], "ttm")

    def test_fiscal_year_by_default(self) -> None:
        kwargs, err = self._kwargs(["NVDA", "--peers", "AMD"])
        self.assertEqual(kwargs["comps_basis"], "fy")
        self.assertNotIn("--comps-basis", err)

    def test_invalid_value_exits_2(self) -> None:
        with mock.patch("equity_valuation.engine.value_company") as vc:
            code, _, err = _run_cli(["NVDA", "--comps-basis", "ltm"])
        self.assertEqual(code, 2)
        self.assertIn("--comps-basis", err)
        vc.assert_not_called()

    def test_ignored_with_no_comps(self) -> None:
        _, err = self._kwargs(["NVDA", "--no-comps", "--comps-basis", "ttm"])
        self.assertIn("note: --comps-basis is ignored with --no-comps.", err)

    def test_help_text(self) -> None:
        text = build_parser().format_help()
        self.assertIn("--comps-basis {fy,ttm}", text)
        self.assertIn("trailing twelve months", text)

    def test_demo_runs_offline_on_the_fiscal_year(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            code, stdout, _ = _run_cli(["--demo", "--comps-basis", "ttm", "--html",
                                        "--out", out])
        self.assertEqual(code, 0)
        self.assertIn(f"Comps: {COMPS_TTM_NOTE_PREFIX} not used: ", stdout)
        self.assertIn("$32.40", stdout)  # no trailing figures: the pinned blend holds


if __name__ == "__main__":
    unittest.main()
