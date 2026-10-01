"""Offline tests for the yfinance market-data client and its hybrid hand-off.

yfinance ``Ticker`` objects are replaced with ``unittest.mock`` stand-ins that
carry hand-built ``.info`` dicts, ``fast_info`` fields and statement DataFrames
shaped like yfinance 1.x output (period-end Timestamp columns, newest first;
pretty row labels) and dividend histories shaped like ``get_dividends()``.
Covered: minor-unit quote currencies (GBp, ZAc) including their dividends,
the DDM D0 choice (Yahoo's indicated rate plus the 3-year average of recurring
variable/special dividends, steady as ex-dates drift; one-off specials, cuts,
large final dividends, ADRs, the no-rate path), the Blume beta adjustment and
non-positive betas (and the raw beta in the API payload), reporting vs quote
currency conversion for foreign issuers,
the ``.info`` outage path, multi-class share counts, comps rows rebuilt from
market cap, debt and cash on one currency and share basis (also without a
balance sheet, or when the check fails), statement tables in another currency
than Yahoo's financialCurrency (VALE, PBR), the statement fallback's column
and label handling and its captive-finance mark, HybridProvider's note
hand-off and market-data backfills, its rejection of stale EDGAR histories and
its Yahoo backfills of what EDGAR's tags miss: debt (with zero cash), and
capex years no capex tag covers, matched by fiscal year (not for banks,
insurers, BDCs and REITs, nor when Yahoo's capex is on another definition than
EDGAR's or, for a lessor, far above its D&A; a failed Yahoo request is told
apart from a statement without capex). The EDGAR gap notes those backfills
parse are also produced by the real EDGAR client on a companyfacts fixture.
"Today" is pinned wherever a date window matters.
No network access is needed. Run with:  python -m unittest tests.test_market_data
"""

from __future__ import annotations

import datetime
import unittest
from unittest import mock

from backend.fmp_client import FMPClient
from equity_valuation.data.synthetic import make_company

import numpy as np
import pandas as pd

from equity_valuation.data import market as market_mod
from equity_valuation.data import provider as provider_mod
from equity_valuation.data.base import DataError
from equity_valuation.data.edgar import EdgarClient
from equity_valuation.data.market import YFinanceClient, major_currency
from equity_valuation.data.provider import HybridProvider
from equity_valuation.schemas import AnnualFinancials, BalanceSheetSnapshot

NaN = np.nan
COLS = pd.to_datetime(["2024-12-31", "2023-12-31", "2022-12-31", "2021-12-31", "2020-12-31"])
TODAY = datetime.date(2026, 9, 28)  # pinned "today" for the dividend and staleness windows


def _frame(rows: dict, scale: float = 1.0, keep_unscaled: tuple = ()) -> pd.DataFrame:
    df = pd.DataFrame(rows, index=COLS).T.astype(float)
    scaled = [r for r in df.index if r not in keep_unscaled]
    df.loc[scaled] *= scale
    return df


def income(scale: float = 1.0, **over) -> pd.DataFrame:
    """Four full years plus a sparse oldest (2020) column, as yfinance 1.x returns."""
    rows = {
        "Total Revenue": [1000, 900, 800, 700, 600],
        "EBIT": [230, 205, 180, 160, NaN],
        "Operating Income": [200, 180, 160, 140, NaN],
        "Pretax Income": [210, 190, 170, 150, NaN],
        "Tax Provision": [42, 38, -34, 30, NaN],
        "Interest Expense": [20, 15, 10, 10, NaN],
        "Net Income": [168, 152, 136, 120, NaN],
        "Reconciled Depreciation": [50, 45, 40, 35, NaN],
        "Diluted Average Shares": [10, 10, 10, 10, NaN],
        "Basic Average Shares": [10, 10, 10, 10, NaN],
    }
    rows.update(over)
    return _frame(rows, scale, keep_unscaled=("Diluted Average Shares", "Basic Average Shares"))


def cashflow(scale: float = 1.0) -> pd.DataFrame:
    return _frame({
        "Capital Expenditure": [-60, -55, -50, -45, NaN],
        "Cash Dividends Paid": [-40, -38, -36, -34, NaN],
        "Change In Working Capital": [-10, -8, -6, -5, NaN],
    }, scale)


def balance(scale: float = 1.0, drop: tuple = (), **over) -> pd.DataFrame:
    rows = {
        "Total Debt": [300, 280, 260, 240, NaN],
        "Cash And Cash Equivalents": [100, 90, 80, 70, NaN],
        "Other Short Term Investments": [50, 40, 30, 20, NaN],
        "Cash Cash Equivalents And Short Term Investments": [150, 130, 110, 90, NaN],
        "Stockholders Equity": [800, 700, 600, 500, NaN],
        "Common Stock Equity": [800, 700, 600, 500, NaN],
        "Total Equity Gross Minority Interest": [850, 750, 650, 550, NaN],
        "Minority Interest": [50, 50, 50, 50, NaN],
    }
    rows.update(over)
    return _frame(rows, scale).drop(index=list(drop))


def fast(last_price=None, currency=None, shares=None):
    """yfinance FastInfo stand-in (attribute access; missing fields -> None)."""
    return mock.NonCallableMock(spec=["last_price", "currency", "shares"],
                                last_price=last_price, currency=currency, shares=shares)


def ticker(info=None, fast_info=None, fin=None, cf=None, bs=None, info_error=None, divs=None,
           qbs=None):
    """A mocked yfinance Ticker (`divs`: what ``get_dividends()`` returns; None -> no method;
    `qbs`: the quarterly balance sheet)."""
    spec = ["info", "fast_info", "financials", "cashflow", "balance_sheet",
            "quarterly_balance_sheet"]
    tk = mock.NonCallableMock(spec=spec + (["get_dividends"] if divs is not None else []))
    if info_error is not None:
        type(tk).info = mock.PropertyMock(side_effect=info_error)
    else:
        tk.info = info if info is not None else {}
    tk.fast_info = fast_info if fast_info is not None else fast()
    tk.financials = fin if fin is not None else pd.DataFrame()
    tk.cashflow = cf if cf is not None else pd.DataFrame()
    tk.balance_sheet = bs if bs is not None else pd.DataFrame()
    tk.quarterly_balance_sheet = qbs if qbs is not None else pd.DataFrame()
    if divs is not None:
        tk.get_dividends = mock.Mock(return_value=divs)
    return tk


def dividends(payments: dict) -> pd.Series:
    """A ``get_dividends()`` Series: {ISO ex-date: amount}, exchange-tz index."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(payments)), name="Date").tz_localize("America/New_York")
    return pd.Series(list(payments.values()), index=idx, name="Dividends", dtype=float)


def client(tickers: dict) -> YFinanceClient:
    """YFinanceClient whose yf.Ticker lookups hit `tickers` (unknown -> KeyError)."""
    c = YFinanceClient()
    c._ticker = mock.Mock(side_effect=lambda sym: tickers[sym])
    return c


def fx_ticker(rate: float):
    return ticker({}, fast(last_price=rate))


class MinorUnitCurrencyTests(unittest.TestCase):
    def test_minor_unit_codes_map_to_major_currency(self) -> None:
        self.assertEqual(major_currency("GBp"), ("GBP", 0.01))
        self.assertEqual(major_currency("GBX"), ("GBP", 0.01))
        self.assertEqual(major_currency("ZAc"), ("ZAR", 0.01))
        self.assertEqual(major_currency("ILA"), ("ILS", 0.01))

    def test_major_codes_are_unchanged(self) -> None:
        self.assertEqual(major_currency("GBP"), ("GBP", 1.0))
        self.assertEqual(major_currency("usd"), ("USD", 1.0))
        self.assertEqual(major_currency(None), (None, 1.0))


class MarketDataTests(unittest.TestCase):
    def test_pence_quote_is_converted_to_pounds(self) -> None:
        # marketCap and dividendRate arrive in pounds while the price is in
        # pence; the trailing fields are in the (USD) reporting currency.
        info = {"currency": "GBp", "financialCurrency": "USD", "currentPrice": 2500.0,
                "sharesOutstanding": 10.0, "marketCap": 250.0, "fiftyTwoWeekLow": 2000.0,
                "fiftyTwoWeekHigh": 3000.0, "dividendRate": 1.0, "dividendYield": 4.0,
                "trailingAnnualDividendRate": 1.35, "trailingAnnualDividendYield": 0.00054}
        md = client({"SHEL.L": ticker(info)}).get_market_data("SHEL.L")
        self.assertEqual(md.currency, "GBP")
        self.assertAlmostEqual(md.price, 25.0)
        self.assertAlmostEqual(md.market_cap, 250.0)
        self.assertAlmostEqual(md.shares_outstanding, 10.0)
        self.assertAlmostEqual(md.fifty_two_week_low, 20.0)
        self.assertAlmostEqual(md.fifty_two_week_high, 30.0)
        self.assertAlmostEqual(md.dividend_per_share, 1.0)
        self.assertTrue(any("GBp" in n for n in md._source_notes))

    def test_pence_market_cap_and_pound_dividend_are_detected(self) -> None:
        # The other reading of each field: marketCap in pence, dividend in pounds.
        info = {"currency": "GBp", "currentPrice": 2500.0, "sharesOutstanding": 10.0,
                "marketCap": 25000.0, "dividendRate": 1.0, "trailingAnnualDividendYield": 0.04}
        md = client({"X.L": ticker(info)}).get_market_data("X.L")
        self.assertAlmostEqual(md.market_cap, 250.0)
        self.assertAlmostEqual(md.dividend_per_share, 1.0)

    def test_info_outage_uses_fast_info_and_never_fabricates_a_cap(self) -> None:
        tk = ticker(info_error=RuntimeError("HTTP 401"),
                    fast_info=fast(last_price=40.0, currency="USD", shares=5.0))
        md = client({"X": tk}).get_market_data("X")
        self.assertEqual((md.price, md.shares_outstanding, md.market_cap), (40.0, 5.0, 200.0))
        self.assertTrue(any(".info) unavailable" in n for n in md._source_notes))

    def test_no_share_count_anywhere_leaves_zero_for_the_provider_to_fill(self) -> None:
        tk = ticker(info_error=RuntimeError("HTTP 429"), fast_info=fast(last_price=40.0))
        md = client({"X": tk}).get_market_data("X")
        self.assertEqual((md.shares_outstanding, md.market_cap), (0.0, 0.0))
        self.assertIsNone(md.dividend_per_share)

    def test_no_price_still_raises(self) -> None:
        with self.assertRaises(DataError):
            client({"X": ticker({"currency": "USD"})}).get_market_data("X")

    def test_multi_class_share_count_is_replaced_by_market_cap_over_price(self) -> None:
        info = {"currency": "USD", "currentPrice": 100.0, "sharesOutstanding": 48.0,
                "marketCap": 10_000.0}
        md = client({"GOOGL": ticker(info)}).get_market_data("GOOGL")
        self.assertAlmostEqual(md.shares_outstanding, 100.0)
        self.assertAlmostEqual(md.market_cap, 10_000.0)
        self.assertTrue(any("marketCap/price" in n for n in md._source_notes))

    def test_consistent_share_count_is_kept(self) -> None:
        info = {"currency": "USD", "currentPrice": 100.0, "sharesOutstanding": 99.0,
                "marketCap": 10_000.0}
        md = client({"X": ticker(info)}).get_market_data("X")
        self.assertEqual(md.shares_outstanding, 99.0)
        self.assertEqual(md._source_notes, [])

    def test_indicated_dividend_rate_is_d0_with_trailing_as_fallback(self) -> None:
        # Without a payment history the indicated rate leads and the trailing
        # rate only fills in (its reporting currency is unknown here).
        info = {"currency": "USD", "currentPrice": 50.0, "sharesOutstanding": 1.0,
                "dividendRate": 1.0, "trailingAnnualDividendRate": 1.75}
        self.assertEqual(client({"X": ticker(info)}).get_market_data("X").dividend_per_share, 1.0)
        del info["dividendRate"]
        self.assertEqual(client({"X": ticker(info)}).get_market_data("X").dividend_per_share, 1.75)

    def test_non_positive_beta_is_left_to_the_default(self) -> None:
        # BP.L / SHEL.L: Yahoo beta -0.22 would put ke below the risk-free rate.
        info = {"currency": "USD", "currentPrice": 45.0, "sharesOutstanding": 1.0, "beta": -0.22}
        md = client({"BP": ticker(info)}).get_market_data("BP")
        self.assertIsNone(md.beta)
        self.assertIsNone(md.raw_beta)
        self.assertEqual(md._source_notes, [
            "Yahoo beta -0.22 is not usable for CAPM (not positive); left unset so the "
            "models use the default beta"])
        info["beta"] = 0.0
        self.assertIsNone(client({"BP": ticker(info)}).get_market_data("BP").beta)
        # A low positive beta is kept, Blume-adjusted (PGR 0.259 -> 0.504).
        info["beta"] = 0.259
        md = client({"PGR": ticker(info)}).get_market_data("PGR")
        self.assertAlmostEqual(md.beta, 0.67 * 0.259 + 0.33)
        self.assertEqual(md.raw_beta, 0.259)

    def test_positive_beta_is_blume_adjusted_and_the_raw_beta_kept(self) -> None:
        # NVDA: Yahoo's raw 2.217 gave ke 15.3% and a 15.2% WACC.
        info = {"currency": "USD", "currentPrice": 180.0, "sharesOutstanding": 1.0, "beta": 2.217}
        md = client({"NVDA": ticker(info)}).get_market_data("NVDA")
        self.assertAlmostEqual(md.beta, 1.81539)
        self.assertEqual(md.raw_beta, 2.217)
        self.assertEqual(md._source_notes, [
            "Beta 1.815 is Yahoo's 5-year monthly beta 2.217 Blume-adjusted toward 1.0 "
            "(0.67 x raw + 0.33) for CAPM"])
        # JNJ 0.235 is pulled up, a beta of 1.0 is unchanged.
        self.assertAlmostEqual(market_mod.blume_adjusted_beta(0.235), 0.48745)
        self.assertAlmostEqual(market_mod.blume_adjusted_beta(1.0), 1.0)

    def test_missing_beta_stays_missing(self) -> None:
        info = {"currency": "USD", "currentPrice": 10.0, "sharesOutstanding": 1.0}
        md = client({"X": ticker(info)}).get_market_data("X")
        self.assertIsNone(md.beta)
        self.assertIsNone(md.raw_beta)
        self.assertEqual(md._source_notes, [])

    def test_synthetic_and_supplied_betas_are_not_adjusted(self) -> None:
        # The adjustment belongs to the Yahoo client only: the demo's synthetic
        # company and a MarketData built by hand keep their beta, so the
        # pinned demo figures do not move.
        from equity_valuation.data.synthetic import SyntheticProvider
        from equity_valuation.models.wacc import compute_wacc
        from equity_valuation.schemas import MacroAssumptions, MarketData

        company = SyntheticProvider().get_company_data("SYNT")
        self.assertEqual(company.market.beta, 1.1)
        self.assertIsNone(company.market.raw_beta)
        self.assertEqual(compute_wacc(company, MacroAssumptions()).beta, 1.1)
        md = MarketData(ticker="X", name="X", currency="USD", price=1.0,
                        shares_outstanding=1.0, market_cap=1.0, beta=2.217)
        self.assertEqual((md.beta, md.raw_beta), (2.217, None))

    def test_raw_beta_reaches_the_api_payload_next_to_the_adjusted_beta(self) -> None:
        # The UI labels `beta` as adjusted and shows Yahoo's raw figure beside it.
        import json

        from backend.serialization import report_to_dict
        from equity_valuation.schemas import (CompanyData, MacroAssumptions, MarketData,
                                              ValuationReport)

        info = {"currency": "USD", "currentPrice": 180.0, "sharesOutstanding": 1.0, "beta": 2.217}
        fin, bs, cik, name = edgar_fundamentals([2024, 2025], "2025-12-31")
        for md, beta, raw in (
            (client({"NVDA": ticker(info)}).get_market_data("NVDA"), 1.81539, 2.217),
            (MarketData(ticker="X", name="X", currency="USD", price=1.0,
                        shares_outstanding=1.0, market_cap=1.0, beta=1.1), 1.1, None),
        ):
            report = ValuationReport(company=CompanyData(md.ticker, name, cik, fin, bs, md),
                                     macro=MacroAssumptions(), current_price=md.price)
            market = json.loads(json.dumps(report_to_dict(report)))["company"]["market"]
            self.assertAlmostEqual(market["beta"], beta)
            self.assertIn("raw_beta", market)
            self.assertEqual(market["raw_beta"], raw)


class DividendPerShareTests(unittest.TestCase):
    """D0 selection; Yahoo shapes taken from live quotes (September 2026)."""

    def setUp(self) -> None:
        patcher = mock.patch.object(market_mod, "_today", return_value=TODAY)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _md(sym, info, divs=None):
        return client({sym: ticker(info, divs=divs)}).get_market_data(sym)

    # --- minor-unit quotes ------------------------------------------------ #
    def test_london_dividend_rate_is_already_in_pounds(self) -> None:
        # BP.L: dividendRate 0.25 GBP (4.49% yield) against a 566.3p price. The
        # trailing yield mixes USD and pence (0.336 / 558.5); it must not flip
        # the dividend to pence (the old reading gave 0.0025 GBP).
        info = {"currency": "GBp", "financialCurrency": "USD", "currentPrice": 566.3,
                "sharesOutstanding": 15452053728, "marketCap": 87504977920,
                "dividendRate": 0.25, "dividendYield": 4.49, "trailingAnnualDividendRate": 0.336,
                "trailingAnnualDividendYield": 0.00060161145}
        divs = dividends({"2025-02-20": 6.1761, "2025-05-15": 5.8993, "2025-08-14": 6.1942,
                          "2025-11-13": 6.2394, "2026-02-19": 6.226, "2026-05-14": 6.1844,
                          "2026-08-13": 6.4059})
        md = self._md("BP.L", info, divs)
        self.assertEqual(md.currency, "GBP")
        self.assertAlmostEqual(md.price, 5.663)
        self.assertAlmostEqual(md.dividend_per_share, 0.25)
        self.assertAlmostEqual(md.market_cap, 87504977920)
        self.assertAlmostEqual(md.market_cap / md.price, md.shares_outstanding, delta=1e3)
        self.assertAlmostEqual(md.dividend_per_share / md.price, 0.0441, places=4)
        self.assertFalse(any("D0" in n for n in md._source_notes), md._source_notes)

    def test_johannesburg_cents_quote_keeps_the_rand_dividend_rate(self) -> None:
        # SBK.JO: dividendRate 18.04 ZAR, history in cents (17.80 ZAR over 12 months).
        info = {"currency": "ZAc", "financialCurrency": "ZAR", "currentPrice": 30139.0,
                "sharesOutstanding": 1622887230, "marketCap": 489121939456,
                "dividendRate": 18.04, "dividendYield": 6.0, "trailingAnnualDividendRate": 17.8,
                "trailingAnnualDividendYield": 0.0005919127}
        divs = dividends({"2025-04-09": 763.0, "2025-09-10": 817.0, "2026-04-15": 878.0,
                          "2026-09-09": 902.0})
        md = self._md("SBK.JO", info, divs)
        self.assertEqual(md.currency, "ZAR")
        self.assertAlmostEqual(md.price, 301.39)
        self.assertAlmostEqual(md.dividend_per_share, 18.04)

    def test_dividend_rate_in_pence_is_detected_by_its_yield(self) -> None:
        info = {"currency": "GBp", "currentPrice": 566.3, "sharesOutstanding": 10.0,
                "dividendRate": 25.0}
        md = self._md("X.L", info)
        self.assertAlmostEqual(md.dividend_per_share, 0.25)
        self.assertTrue(any("read as GBp and converted" in n for n in md._source_notes))

    def test_pence_history_is_converted_when_the_rate_is_missing(self) -> None:
        info = {"currency": "GBp", "financialCurrency": "USD", "currentPrice": 566.3,
                "sharesOutstanding": 10.0, "trailingAnnualDividendRate": 0.336}
        divs = dividends({"2025-11-13": 6.25, "2026-02-19": 6.25, "2026-05-14": 6.25,
                          "2026-08-13": 6.25})
        md = self._md("BP.L", info, divs)
        self.assertAlmostEqual(md.dividend_per_share, 0.25)
        self.assertTrue(any("indicated dividend rate unavailable" in n for n in md._source_notes))

    # --- variable / special dividends -------------------------------------- #
    # Five-year histories as ``get_dividends(period='5y')`` returned them
    # (September 2026). PGR and BKE pay the annual variable/special dividend
    # together with a regular quarterly; CME paid it on its own in late
    # December until its 2025 one went ex with the Q1 quarterly in March 2026.
    PGR_INFO = {"currency": "USD", "financialCurrency": "USD", "currentPrice": 207.88,
                "sharesOutstanding": 580400000, "marketCap": 120653553664,
                "dividendRate": 0.4, "trailingAnnualDividendRate": 0.4}
    PGR_DIVS = {"2021-10-06": 0.1, "2021-12-17": 1.5, "2022-01-06": 0.1, "2022-04-06": 0.1,
                "2022-07-06": 0.1, "2022-10-06": 0.1, "2023-01-05": 0.1, "2023-04-05": 0.1,
                "2023-07-06": 0.1, "2023-10-04": 0.1, "2024-01-18": 0.85, "2024-04-03": 0.1,
                "2024-07-03": 0.1, "2024-10-03": 0.1, "2025-01-10": 4.6, "2025-04-03": 0.1,
                "2025-07-03": 0.1, "2025-10-02": 0.1, "2026-01-02": 13.6, "2026-04-02": 0.1,
                "2026-07-02": 0.1}
    CME_INFO = {"currency": "USD", "financialCurrency": "USD", "currentPrice": 264.5,
                "sharesOutstanding": 359576125, "marketCap": 95108718592, "dividendRate": 5.2}
    CME_DIVS = {"2021-12-09": 0.9, "2021-12-27": 3.25, "2022-03-09": 1.0, "2022-06-09": 1.0,
                "2022-09-08": 1.0, "2022-12-08": 1.0, "2022-12-27": 4.5, "2023-03-09": 1.1,
                "2023-06-08": 1.1, "2023-09-07": 1.1, "2023-12-07": 1.1, "2023-12-27": 5.25,
                "2024-03-07": 1.15, "2024-06-07": 1.15, "2024-09-09": 1.15, "2024-12-09": 1.15,
                "2024-12-27": 5.8, "2025-03-07": 1.25, "2025-06-09": 1.25, "2025-09-09": 1.25,
                "2025-12-12": 1.25, "2026-03-10": 7.45, "2026-06-09": 1.3, "2026-09-09": 1.3}
    BKE_INFO = {"currency": "USD", "financialCurrency": "USD", "currentPrice": 42.385,
                "sharesOutstanding": 51512434, "marketCap": 2183354368, "dividendRate": 1.4}
    BKE_DIVS = {"2021-10-14": 0.33, "2021-12-17": 6.0, "2022-04-13": 0.35, "2022-07-14": 0.35,
                "2022-10-13": 0.35, "2023-01-12": 3.0, "2023-04-13": 0.35, "2023-07-13": 0.35,
                "2023-10-12": 0.35, "2024-01-11": 2.85, "2024-04-11": 0.35, "2024-07-12": 0.35,
                "2024-10-11": 0.35, "2025-01-15": 2.85, "2025-04-15": 0.35, "2025-07-15": 0.35,
                "2025-10-15": 0.35, "2026-01-15": 3.35, "2026-04-15": 0.35, "2026-07-15": 0.35}

    def test_recurring_variable_dividend_adds_its_three_year_average(self) -> None:
        # PGR: 0.10 quarterly; January payments of 0.85, 4.60 and 13.60 each
        # include that quarterly. D0 = 0.40 + (0.75 + 4.50 + 13.50) / 3.
        md = self._md("PGR", self.PGR_INFO, dividends(self.PGR_DIVS))
        self.assertAlmostEqual(md.dividend_per_share, 6.65)
        note = next(n for n in md._source_notes if n.startswith("DDM D0 ="))
        self.assertTrue(note.startswith(
            "DDM D0 = Yahoo's indicated regular rate (0.4) + 6.25 USD a year of recurring "
            "special/variable dividends = 6.65 USD per share"), note)
        self.assertIn("0.75 on 2024-01-18, 4.5 on 2025-01-10, 13.5 on 2026-01-02", note)

    def test_variable_dividend_paid_on_its_own_counts_in_full(self) -> None:
        # CME: late-December payments stand apart from the December quarterly;
        # the March 2026 one (7.45) carries the 1.30 Q1 quarterly.
        md = self._md("CME", self.CME_INFO, dividends(self.CME_DIVS))
        self.assertAlmostEqual(md.dividend_per_share, 5.2 + (5.25 + 5.8 + 6.15) / 3)
        self.assertTrue(any("5.25 on 2023-12-27, 5.8 on 2024-12-27, 6.15 on 2026-03-10" in n
                            for n in md._source_notes), md._source_notes)

    def test_d0_is_steady_across_anniversaries_and_ex_date_drift(self) -> None:
        # Replays of the live histories: D0 moves only when a new variable
        # payment goes ex, never because a strict 12-month window gains or
        # loses one (CME's 2025 variable went ex 438 days after the 2024 one;
        # BKE's and PGR's January payments straddle the anniversaries).
        cases = [  # (symbol, info, history, [(first day, last day, D0)])
            ("PGR", self.PGR_INFO, self.PGR_DIVS,
             [("2025-06-01", "2026-01-01", 0.4 + 5.25 / 3), ("2026-01-02", "2026-12-31", 6.65)]),
            ("CME", self.CME_INFO, self.CME_DIVS,
             [("2025-06-01", "2026-03-09", 5.2 + 15.55 / 3),
              ("2026-03-10", "2026-12-31", 5.2 + 17.2 / 3)]),
            ("BKE", self.BKE_INFO, self.BKE_DIVS,
             [("2025-06-01", "2026-01-14", 1.4 + 7.65 / 3), ("2026-01-15", "2026-12-31", 1.4 + 8.0 / 3)]),
        ]
        for sym, info, divs, spans in cases:
            history = dividends(divs)
            for first, last, expected in spans:
                day = datetime.date.fromisoformat(first)
                while day <= datetime.date.fromisoformat(last):
                    with mock.patch.object(market_mod, "_today", return_value=day):
                        md = self._md(sym, info, history)
                    # CME's March 2026 row counts 7.45 - 1.25 until its 1.30 Q1
                    # quarterly is confirmed by the June one: 0.017 more.
                    self.assertAlmostEqual(md.dividend_per_share, expected, delta=0.02,
                                           msg=f"{sym} {day}")
                    self.assertFalse(any("one-off" in n for n in md._source_notes),
                                     f"{sym} {day}: {md._source_notes}")
                    day += datetime.timedelta(days=1)

    # A 15.00 special once, regular 1.16 quarterlies otherwise.
    ONE_OFF_DIVS = {"2024-11-01": 1.16, "2025-02-07": 1.16, "2025-05-02": 1.16,
                    "2025-08-01": 1.16, "2025-10-31": 1.16, "2025-12-26": 15.0,
                    "2026-01-30": 1.16, "2026-05-01": 1.16, "2026-07-24": 1.16}

    def test_one_off_special_keeps_the_indicated_rate(self) -> None:
        info = {"currency": "USD", "currentPrice": 900.0, "sharesOutstanding": 1.0,
                "dividendRate": 4.64}
        md = self._md("X", info, dividends(self.ONE_OFF_DIVS))
        self.assertEqual(md.dividend_per_share, 4.64)
        self.assertTrue(any("include a one-off special payment (15 on 2025-12-26 above the "
                            "regular dividend; none in the two years before)" in n
                            for n in md._source_notes), md._source_notes)

    def test_info_outage_leaves_a_one_off_special_out_of_d0(self) -> None:
        # No indicated rate (.info down): D0 is the trailing regular dividends.
        tk = ticker(info_error=RuntimeError("HTTP 429"),
                    fast_info=fast(last_price=900.0, currency="USD", shares=1.0),
                    divs=dividends(self.ONE_OFF_DIVS))
        md = client({"X": tk}).get_market_data("X")
        self.assertAlmostEqual(md.dividend_per_share, 4.64)
        self.assertIn("Yahoo quote summary (.info) unavailable; market data limited to the "
                      "price endpoint (no beta, indicated dividend rate or 52-week range from "
                      "Yahoo)", md._source_notes)
        self.assertIn("Yahoo's indicated dividend rate unavailable; DDM D0 = trailing-12-month "
                      "regular cash dividends from the payment history (4.64 USD); a one-off "
                      "special payment (15 on 2025-12-26 above the regular dividend; none in "
                      "the two years before) is left out", md._source_notes)

    def test_recurring_variable_dividend_without_a_rate(self) -> None:
        info = dict(self.PGR_INFO, dividendRate=None)
        md = self._md("PGR", info, dividends(self.PGR_DIVS))
        self.assertAlmostEqual(md.dividend_per_share, 0.4 + 6.25)
        self.assertTrue(any("(0.4 USD) + 6.25 USD a year of recurring special/variable" in n
                            for n in md._source_notes), md._source_notes)

    def test_large_final_dividend_already_in_the_rate_is_not_added(self) -> None:
        # HSBC: 0.50 quarterlies and a larger March dividend that Yahoo's
        # indicated rate (3.75 = 2.25 + 3 x 0.50) already holds.
        info = {"currency": "USD", "financialCurrency": "USD", "currentPrice": 100.93,
                "sharesOutstanding": 3421317591, "marketCap": 345313574912, "dividendRate": 3.75}
        divs = dividends({"2022-03-10": 0.9, "2022-08-18": 0.45, "2023-03-02": 1.15,
                          "2023-05-11": 0.5, "2023-08-10": 0.5, "2023-11-09": 0.5,
                          "2024-03-07": 1.55, "2024-05-09": 1.55, "2024-08-16": 0.5,
                          "2024-11-08": 0.5, "2025-03-07": 1.8, "2025-05-09": 0.5,
                          "2025-08-15": 0.5, "2025-11-07": 0.5, "2026-03-13": 2.25,
                          "2026-05-15": 0.5, "2026-08-14": 0.5})
        md = self._md("HSBC", info, divs)
        self.assertEqual(md.dividend_per_share, 3.75)
        self.assertEqual(md._source_notes, [])

    def test_dividend_cut_keeps_the_indicated_rate(self) -> None:
        # Quarterly 0.50 cut to 0.25: TTM 1.50 is 1.5x the indicated 1.00.
        info = {"currency": "USD", "currentPrice": 40.0, "sharesOutstanding": 1.0,
                "dividendRate": 1.0}
        divs = dividends({"2025-03-14": 0.5, "2025-06-13": 0.5, "2025-10-15": 0.5,
                          "2026-01-15": 0.5, "2026-04-15": 0.25, "2026-07-15": 0.25})
        md = self._md("X", info, divs)
        self.assertEqual(md.dividend_per_share, 1.0)
        self.assertIn("trailing-12-month cash dividends of 1.5 USD per share exceed Yahoo's "
                      "indicated rate (1) (a cut, supplemental or special payments, or an extra "
                      "ex-date in the window); DDM D0 keeps the indicated rate", md._source_notes)

    def test_fast_growing_regular_dividend_is_not_special(self) -> None:
        # 10% more each quarter for five years (0.10 -> 0.61): each payment is
        # judged against the payments within a year of it, not the 5-year
        # median (0.25), so the latest ones are not stripped as specials.
        start = datetime.date(2021, 10, 15)
        divs = dividends({(start + datetime.timedelta(days=91 * i)).isoformat(): 0.1 * 1.1 ** i
                          for i in range(20)})
        info = {"currency": "USD", "currentPrice": 40.0, "sharesOutstanding": 1.0}
        md = self._md("X", info, divs)
        self.assertAlmostEqual(md.dividend_per_share, sum(0.1 * 1.1 ** i for i in range(16, 20)))
        self.assertFalse(any("special" in n for n in md._source_notes), md._source_notes)

    def test_regular_payer_keeps_the_indicated_rate_silently(self) -> None:
        # KO: indicated 2.12 after a raise; TTM 2.10.
        info = {"currency": "USD", "currentPrice": 87.07, "sharesOutstanding": 1.0,
                "dividendRate": 2.12, "trailingAnnualDividendRate": 2.08}
        divs = dividends({"2025-09-15": 0.51, "2025-12-01": 0.51, "2026-03-13": 0.53,
                          "2026-06-15": 0.53, "2026-09-15": 0.53})
        md = self._md("KO", info, divs)
        self.assertEqual(md.dividend_per_share, 2.12)
        self.assertEqual(md._source_notes, [])

    def test_history_unavailable_keeps_the_indicated_rate(self) -> None:
        tk = ticker(self.PGR_INFO)
        tk.get_dividends = mock.Mock(side_effect=RuntimeError("HTTP 429"))
        self.assertEqual(client({"PGR": tk}).get_market_data("PGR").dividend_per_share, 0.4)

    def test_history_in_another_currency_is_not_used(self) -> None:
        frame = pd.DataFrame({"Dividends": [5.0, 5.0, 5.0, 5.0],
                              "currency": ["EUR"] * 4},
                             index=dividends({"2025-11-01": 0, "2026-02-01": 0,
                                              "2026-05-01": 0, "2026-08-01": 0}).index)
        md = self._md("X", {"currency": "USD", "currentPrice": 10.0, "sharesOutstanding": 1.0,
                            "dividendRate": 0.4}, frame)
        self.assertEqual(md.dividend_per_share, 0.4)
        self.assertEqual(md._source_notes, [])

    # --- ADRs -------------------------------------------------------------- #
    def test_adr_trailing_rate_in_the_reporting_currency_is_not_d0(self) -> None:
        # TSM: trailingAnnualDividendRate 26.0 is TWD per ordinary share.
        info = {"currency": "USD", "financialCurrency": "TWD", "currentPrice": 444.68,
                "sharesOutstanding": 5186474013, "marketCap": 2306321154048,
                "trailingAnnualDividendRate": 26.0}
        md = self._md("TSM", info)
        self.assertIsNone(md.dividend_per_share)
        self.assertTrue(any("in TWD per ordinary share" in n for n in md._source_notes))
        # The USD per-ADS payment history is used instead when available.
        divs = dividends({"2025-12-11": 0.835, "2026-03-17": 0.956, "2026-06-11": 0.955,
                          "2026-09-16": 1.107})
        self.assertAlmostEqual(self._md("TSM", info, divs).dividend_per_share, 3.853)

    def test_usd_adr_uses_the_per_ads_rate(self) -> None:
        # BP ADR: 2.02 per ADS; the trailing 0.336 is per ordinary share.
        info = {"currency": "USD", "financialCurrency": "USD", "currentPrice": 45.045,
                "sharesOutstanding": 2575342288, "marketCap": 116006289408,
                "dividendRate": 2.02, "trailingAnnualDividendRate": 0.336}
        divs = dividends({"2025-11-14": 0.499, "2026-02-20": 0.499, "2026-05-15": 0.499,
                          "2026-08-14": 0.52})
        self.assertEqual(self._md("BP", info, divs).dividend_per_share, 2.02)

    def test_suspended_dividend_without_a_rate_gives_zero(self) -> None:
        # INTC: no indicated rate and no payment in the last 12 months.
        info = {"currency": "USD", "financialCurrency": "USD", "currentPrice": 115.0,
                "sharesOutstanding": 1.0, "trailingAnnualDividendRate": 0.0}
        md = self._md("INTC", info, dividends({"2024-05-06": 0.125, "2024-08-07": 0.125}))
        self.assertIsInstance(md.dividend_per_share, float)
        self.assertEqual(md.dividend_per_share, 0.0)

    def test_zero_trailing_rate_is_kept_whatever_its_currency(self) -> None:
        info = {"currency": "USD", "financialCurrency": "CNY", "currentPrice": 10.0,
                "sharesOutstanding": 1.0, "trailingAnnualDividendRate": 0.0}
        md = self._md("X", info, dividends({}))
        self.assertEqual(md.dividend_per_share, 0.0)
        self.assertEqual(md._source_notes, [])


class FxRateTests(unittest.TestCase):
    def test_direct_pair(self) -> None:
        c = client({"TWDUSD=X": fx_ticker(0.03125)})
        self.assertEqual(c.get_fx_rate("TWD", "USD"), (0.03125, "TWDUSD=X"))

    def test_inverse_pair(self) -> None:
        rate, how = client({"USDTWD=X": fx_ticker(32.0)}).get_fx_rate("TWD", "USD")
        self.assertAlmostEqual(rate, 1 / 32.0)
        self.assertEqual(how, "1/USDTWD=X")

    def test_cross_through_usd(self) -> None:
        c = client({"DKKUSD=X": fx_ticker(0.15), "USDEUR=X": fx_ticker(0.9)})
        rate, how = c.get_fx_rate("DKK", "EUR")
        self.assertAlmostEqual(rate, 0.135)
        self.assertEqual(how, "DKKUSD=X x USDEUR=X")

    def test_same_currency_and_unavailable_rate(self) -> None:
        c = client({})
        self.assertEqual(c.get_fx_rate("usd", "USD"), (1.0, "same currency"))
        self.assertIsNone(c.get_fx_rate("TWD", "USD"))


class StatementFallbackTests(unittest.TestCase):
    def _fallback(self, info=None, fin=None, cf=None, bs=None, extra=None):
        tickers = {"X": ticker(info if info is not None else {"currency": "USD",
                                                               "financialCurrency": "USD"},
                               fin=fin if fin is not None else income(),
                               cf=cf if cf is not None else cashflow(),
                               bs=bs if bs is not None else balance())}
        tickers.update(extra or {})
        return client(tickers).get_annual_financials_fallback("X")

    def test_sparse_oldest_column_is_dropped(self) -> None:
        fin, _bs = self._fallback()
        self.assertEqual(fin.fiscal_years, [2021, 2022, 2023, 2024])
        self.assertEqual(fin.capex, [45.0, 50.0, 55.0, 60.0])
        self.assertTrue(any("dropped 2020-12-31" in n for n in fin._source_notes))

    def test_operating_income_is_preferred_over_yahoo_ebit(self) -> None:
        fin, _bs = self._fallback()
        self.assertEqual(fin.ebit, [140.0, 160.0, 180.0, 200.0])

    def test_nan_in_preferred_row_falls_through_per_period(self) -> None:
        fin, _bs = self._fallback(fin=income(**{"Diluted Average Shares": [10, 10, 10, NaN, NaN],
                                                "Basic Average Shares": [10, 10, 10, 9, NaN]}))
        self.assertEqual(fin.diluted_shares, [9.0, 10.0, 10.0, 10.0])
        bs_frame = balance()
        bs_frame.loc["Stockholders Equity", COLS[0]] = NaN
        _fin, bs = self._fallback(bs=bs_frame)
        self.assertEqual(bs.total_equity, 800.0)  # from Common Stock Equity

    def test_missing_line_is_zero_filled_with_a_note(self) -> None:
        fin, _bs = self._fallback(fin=income(**{"Reconciled Depreciation": [NaN] * 5}))
        self.assertEqual(fin.dep_amort, [0.0] * 4)
        self.assertIn("yfinance fallback: D&A unavailable; filled with 0.0", fin._source_notes)

    def test_tax_benefit_keeps_its_sign(self) -> None:
        fin, _bs = self._fallback()
        self.assertEqual(fin.tax_expense, [30.0, -34.0, 38.0, 42.0])

    def test_combined_cash_line_is_not_double_counted(self) -> None:
        _fin, bs = self._fallback(bs=balance(drop=("Cash And Cash Equivalents",)))
        self.assertEqual(bs.cash_and_investments, 150.0)
        _fin, bs = self._fallback(bs=balance(drop=("Cash Cash Equivalents And Short Term Investments",)))
        self.assertEqual(bs.cash_and_investments, 150.0)

    def test_gross_equity_line_excludes_minority_interest(self) -> None:
        _fin, bs = self._fallback(bs=balance(drop=("Stockholders Equity", "Common Stock Equity")))
        self.assertEqual(bs.total_equity, 800.0)

    def test_foreign_statements_are_converted_to_the_quote_currency(self) -> None:
        info = {"currency": "USD", "financialCurrency": "TWD"}
        fin, bs = self._fallback(info=info, fin=income(32), cf=cashflow(32), bs=balance(32),
                                 extra={"TWDUSD=X": fx_ticker(1 / 32)})
        self.assertEqual(fin.revenue, [700.0, 800.0, 900.0, 1000.0])
        self.assertEqual(fin.dividends_paid, [34.0, 36.0, 38.0, 40.0])
        self.assertEqual(fin.diluted_shares, [10.0] * 4)  # share counts untouched
        self.assertEqual((bs.total_debt, bs.cash_and_investments, bs.total_equity,
                          bs.minority_interest), (300.0, 150.0, 800.0, 50.0))
        self.assertTrue(fin._source_notes[0].startswith(
            "Fundamentals converted from TWD to USD at spot 0.03125"))

    def test_missing_fx_rate_stops_unit_mixed_valuation(self) -> None:
        info = {"currency": "USD", "financialCurrency": "TWD"}
        with self.assertRaisesRegex(DataError, "no TWD->USD exchange rate.*retry"):
            self._fallback(info=info, fin=income(32), cf=cashflow(32), bs=balance(32))

    def test_pence_quote_with_pound_statements_needs_no_fx(self) -> None:
        fin, _bs = self._fallback(info={"currency": "GBp", "financialCurrency": "GBP"})
        self.assertEqual(fin.revenue[-1], 1000.0)
        self.assertFalse(any("convert" in n or "WARNING" in n for n in fin._source_notes))

    def test_pence_quote_with_dollar_statements_converts_to_pounds(self) -> None:
        fin, _bs = self._fallback(info={"currency": "GBp", "financialCurrency": "USD"},
                                  extra={"USDGBP=X": fx_ticker(0.8)})
        self.assertAlmostEqual(fin.revenue[-1], 800.0)

    # VALE-like: financialCurrency and .info amounts in BRL, statement tables
    # in USD (table debt 300 against totalDebt 1,500; revenue 1,000 against 5,600).
    VALE_INFO = {"currency": "USD", "financialCurrency": "BRL", "totalDebt": 1500.0,
                 "totalRevenue": 5600.0}
    VALE_NOTE = ("yfinance fallback: Yahoo gives BRL as the reporting currency "
                 "(financialCurrency), but its statement tables are in USD (table Total Debt / "
                 "totalDebt 0.2 and Total Revenue / totalRevenue 0.1786, against BRL->USD 0.19 "
                 "(BRLUSD=X)); they are read as USD")

    def test_usd_tables_under_a_local_financial_currency_are_not_converted_twice(self) -> None:
        fin, bs = self._fallback(info=self.VALE_INFO, extra={"BRLUSD=X": fx_ticker(0.19)})
        self.assertEqual(fin.revenue, [700.0, 800.0, 900.0, 1000.0])  # not x 0.19
        self.assertEqual((bs.total_debt, bs.total_equity), (300.0, 800.0))
        self.assertEqual(fin._source_notes[0], self.VALE_NOTE)
        self.assertFalse(any("converted" in n for n in fin._source_notes))
        # The Sao Paulo line (BRL quote): the USD tables are converted to BRL.
        info = dict(self.VALE_INFO, currency="BRL")
        fin, bs = self._fallback(info=info, extra={"BRLUSD=X": fx_ticker(0.19)})
        self.assertAlmostEqual(fin.revenue[-1], 1000.0 / 0.19)
        self.assertAlmostEqual(bs.total_debt, 300.0 / 0.19)
        self.assertEqual(fin._source_notes[0], self.VALE_NOTE)
        self.assertTrue(fin._source_notes[1].startswith(
            "Fundamentals converted from USD to BRL at spot 5.26316 (1/BRLUSD=X)"))

    def test_one_table_comparison_alone_does_not_move_the_currency(self) -> None:
        # Table debt at a fifth of totalDebt (a bank's definitions) while
        # revenue agrees: the tables stay BRL and are converted as before.
        info = dict(self.VALE_INFO, totalRevenue=1000.0)
        fin, _bs = self._fallback(info=info, extra={"BRLUSD=X": fx_ticker(0.19)})
        self.assertAlmostEqual(fin.revenue[-1], 190.0)
        self.assertTrue(fin._source_notes[0].startswith("Fundamentals converted from BRL to USD"))
        # Near-1 currency pairs are never switched (EUR tables, GBP quote).
        info = {"currency": "GBp", "financialCurrency": "EUR", "totalDebt": 345.0,
                "totalRevenue": 1150.0}
        fin, _bs = self._fallback(info=info, extra={"EURGBP=X": fx_ticker(0.87)})
        self.assertAlmostEqual(fin.revenue[-1], 870.0)
        self.assertFalse(any("tables are in" in n for n in fin._source_notes))

    def test_january_period_end_takes_the_prior_fiscal_year(self) -> None:
        cols = pd.to_datetime(["2023-12-31", "2023-01-01", "2022-01-02"])
        fin_frame = pd.DataFrame({"Total Revenue": [3.0, 2.0, 1.0], "Net Income": [1.0] * 3},
                                 index=cols).T
        fin, _bs = self._fallback(fin=fin_frame, cf=pd.DataFrame(), bs=pd.DataFrame())
        self.assertEqual(fin.fiscal_years, [2021, 2022, 2023])


class HybridProviderMarketTests(unittest.TestCase):
    def setUp(self) -> None:
        # Keeps the FY2024 EDGAR fixtures inside the staleness window.
        patcher = mock.patch.object(provider_mod, "_today", return_value=TODAY)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _edgar_fails(self):
        edgar = mock.Mock()
        edgar.get_annual_financials.side_effect = DataError("not in SEC map")
        return edgar

    def test_foreign_adr_is_valued_in_one_currency(self) -> None:
        info = {"currency": "USD", "financialCurrency": "TWD", "currentPrice": 200.0,
                "sharesOutstanding": 10.0, "marketCap": 2000.0, "dividendRate": 4.0}
        c = client({"TSM": ticker(info, fin=income(32), cf=cashflow(32), bs=balance(32)),
                    "TWDUSD=X": fx_ticker(1 / 32)})
        cd = HybridProvider(edgar=self._edgar_fails(), market=c).get_company_data("TSM")
        self.assertEqual(cd.market.currency, "USD")
        self.assertEqual(cd.financials.revenue[-1], 1000.0)
        self.assertEqual(cd.balance_sheet.total_debt, 300.0)
        self.assertEqual(cd.source_notes[:2], ["EDGAR unavailable: not in SEC map",
                                               "Fundamentals: yfinance fallback"])
        self.assertTrue(any("converted from TWD to USD" in n for n in cd.source_notes))

    def test_vale_like_adr_fundamentals_keep_their_usd_tables(self) -> None:
        # VALE: EDGAR stale, so the yfinance fallback supplies USD tables that
        # Yahoo labels BRL; they must not be scaled down about 5x.
        info = dict(StatementFallbackTests.VALE_INFO, currentPrice=13.6,
                    sharesOutstanding=100.0, marketCap=1360.0)
        c = client({"VALE": ticker(info, fin=income(), cf=cashflow(), bs=balance()),
                    "BRLUSD=X": fx_ticker(0.19)})
        cd = HybridProvider(edgar=self._edgar_fails(), market=c).get_company_data("VALE")
        self.assertEqual(cd.financials.revenue[-1], 1000.0)
        self.assertEqual(cd.balance_sheet.total_debt, 300.0)
        self.assertIn(StatementFallbackTests.VALE_NOTE, cd.source_notes)
        self.assertFalse(any("converted" in n for n in cd.source_notes))

    def test_unconverted_currencies_stop_hybrid_valuation(self) -> None:
        info = {"currency": "USD", "financialCurrency": "TWD", "currentPrice": 200.0,
                "sharesOutstanding": 10.0, "marketCap": 2000.0}
        c = client({"TSM": ticker(info, fin=income(32), cf=cashflow(32), bs=balance(32))})
        with self.assertRaisesRegex(DataError, "No usable fundamentals.*no TWD->USD exchange rate"):
            HybridProvider(edgar=self._edgar_fails(), market=c).get_company_data("TSM")

    def test_info_outage_backfills_shares_cap_and_dps_from_statements(self) -> None:
        fin = AnnualFinancials(
            fiscal_years=[2023, 2024], revenue=[90.0, 100.0], ebit=[9.0, 10.0],
            ebitda=[12.0, 13.0], net_income=[7.0, 8.0], dep_amort=[3.0, 3.0],
            capex=[4.0, 4.0], change_in_nwc=[1.0, 1.0], interest_expense=[1.0, 1.0],
            tax_expense=[2.0, 2.0], pretax_income=[9.0, 10.0], dividends_paid=[2.0, 3.0],
            diluted_shares=[20.0, 25.0])
        bs = BalanceSheetSnapshot(as_of="2024-12-31", total_debt=50.0,
                                  cash_and_investments=10.0, total_equity=80.0)
        edgar = mock.Mock()
        edgar.get_annual_financials.return_value = (fin, bs, "0000000001", "Fixture Co")
        tk = ticker(info_error=RuntimeError("HTTP 401"), fast_info=fast(last_price=40.0))
        cd = HybridProvider(edgar=edgar, market=client({"FIX": tk})).get_company_data("FIX")
        self.assertEqual(cd.market.shares_outstanding, 25.0)
        self.assertEqual(cd.market.market_cap, 1000.0)
        self.assertAlmostEqual(cd.market.dividend_per_share, 3.0 / 25.0)
        notes = " | ".join(cd.source_notes)
        self.assertIn(".info) unavailable", notes)
        self.assertIn("market cap unavailable from Yahoo; set to price x shares", notes)

    def test_variable_dividend_d0_reaches_the_company_notes(self) -> None:
        edgar = mock.Mock()
        edgar.get_annual_financials.return_value = edgar_fundamentals(
            [2022, 2023, 2024, 2025], "2026-06-30")
        tk = ticker(DividendPerShareTests.PGR_INFO,
                    divs=dividends(DividendPerShareTests.PGR_DIVS))
        with mock.patch.object(market_mod, "_today", return_value=TODAY):
            cd = HybridProvider(edgar=edgar, market=client({"PGR": tk})).get_company_data("PGR")
        self.assertAlmostEqual(cd.market.dividend_per_share, 6.65)
        self.assertEqual(cd.source_notes[0], "Fundamentals: SEC EDGAR (CIK 0001094517)")
        self.assertTrue(any(n.startswith("DDM D0 = Yahoo's indicated regular rate (0.4) + 6.25")
                            for n in cd.source_notes), cd.source_notes)

    def test_no_share_count_anywhere_warns(self) -> None:
        fin = AnnualFinancials(
            fiscal_years=[2024], revenue=[100.0], ebit=[10.0], ebitda=[13.0],
            net_income=[8.0], dep_amort=[3.0], capex=[4.0], change_in_nwc=[1.0],
            interest_expense=[1.0], tax_expense=[2.0], pretax_income=[10.0],
            dividends_paid=[0.0], diluted_shares=[0.0])
        bs = BalanceSheetSnapshot(as_of="2024-12-31", total_debt=50.0,
                                  cash_and_investments=10.0, total_equity=80.0)
        edgar = mock.Mock()
        edgar.get_annual_financials.return_value = (fin, bs, "0000000001", "Fixture Co")
        tk = ticker(info_error=RuntimeError("HTTP 401"), fast_info=fast(last_price=40.0))
        cd = HybridProvider(edgar=edgar, market=client({"FIX": tk})).get_company_data("FIX")
        self.assertEqual(cd.market.market_cap, 0.0)
        self.assertTrue(cd.source_notes[1].startswith("WARNING: market cap unavailable"),
                        cd.source_notes)


def edgar_fundamentals(years: list[int], as_of: str) -> tuple:
    """An EdgarClient.get_annual_financials result with flat fixture series."""
    n = len(years)
    fin = AnnualFinancials(
        fiscal_years=list(years), revenue=[100.0] * n, ebit=[10.0] * n, ebitda=[13.0] * n,
        net_income=[8.0] * n, dep_amort=[3.0] * n, capex=[4.0] * n, change_in_nwc=[1.0] * n,
        interest_expense=[1.0] * n, tax_expense=[2.0] * n, pretax_income=[10.0] * n,
        dividends_paid=[2.0] * n, diluted_shares=[10.0] * n)
    fin._source_notes = ["only 4 annual period(s) available on EDGAR (target is 5-8)"]
    bs = BalanceSheetSnapshot(as_of=as_of, total_debt=50.0, cash_and_investments=10.0,
                              total_equity=80.0)
    return fin, bs, "0001094517", "Fixture Motor Corp"


class StaleEdgarTests(unittest.TestCase):
    """EDGAR histories that ended years ago (Toyota: us-gaap facts stop at FY2013)."""

    INFO = {"currency": "USD", "financialCurrency": "USD", "currentPrice": 188.0,
            "sharesOutstanding": 10.0, "marketCap": 1880.0}

    def setUp(self) -> None:
        patcher = mock.patch.object(provider_mod, "_today", return_value=TODAY)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _provider(self, edgar_result, fallback: bool = True):
        edgar = mock.Mock()
        edgar.get_annual_financials.return_value = edgar_result
        frames = dict(fin=income(), cf=cashflow(), bs=balance()) if fallback else {}
        market = client({"TM": ticker(self.INFO, **frames)})
        return HybridProvider(edgar=edgar, market=market), market

    def test_stale_edgar_history_gives_way_to_newer_yfinance_statements(self) -> None:
        prov, _ = self._provider(edgar_fundamentals([2010, 2011, 2012, 2013], "2013-03-31"))
        cd = prov.get_company_data("TM")
        self.assertEqual(cd.financials.fiscal_years, [2021, 2022, 2023, 2024])
        self.assertEqual(cd.balance_sheet.as_of, "2024-12-31")
        self.assertIsNone(cd.cik)
        self.assertTrue(cd.source_notes[0].startswith(
            "WARNING: EDGAR fundamentals (CIK 0001094517) end FY2013 (balance sheet 2013-03-31)"),
            cd.source_notes)
        self.assertIn("using the yfinance fallback (to FY2024) instead", cd.source_notes[0])
        self.assertEqual(cd.source_notes[1], "Fundamentals: yfinance fallback")
        # EDGAR's own parsing notes no longer apply.
        self.assertNotIn("only 4 annual period(s) available on EDGAR (target is 5-8)",
                         cd.source_notes)

    def test_stale_edgar_history_is_kept_with_a_warning_without_a_newer_fallback(self) -> None:
        prov, _ = self._provider(edgar_fundamentals([2010, 2011, 2012, 2013], "2013-03-31"),
                                 fallback=False)
        cd = prov.get_company_data("TM")
        self.assertEqual(cd.financials.fiscal_years, [2010, 2011, 2012, 2013])
        self.assertEqual(cd.cik, "0001094517")
        self.assertEqual(cd.source_notes[0], "Fundamentals: SEC EDGAR (CIK 0001094517)")
        self.assertTrue(cd.source_notes[1].startswith("WARNING: EDGAR fundamentals"), cd.source_notes)
        self.assertIn("the valuation uses this stale history", cd.source_notes[1])

    def test_fallback_that_is_not_newer_does_not_replace_edgar(self) -> None:
        prov, _ = self._provider(edgar_fundamentals([2021, 2022, 2023, 2024], "2024-06-30"))
        with mock.patch.object(provider_mod, "_today", return_value=datetime.date(2027, 6, 1)):
            cd = prov.get_company_data("TM")
        self.assertEqual(cd.cik, "0001094517")
        self.assertEqual(cd.balance_sheet.as_of, "2024-06-30")
        self.assertTrue(any("has no newer fiscal year" in n for n in cd.source_notes))

    def test_current_edgar_history_is_used_without_the_fallback(self) -> None:
        prov, market = self._provider(edgar_fundamentals([2022, 2023, 2024, 2025], "2026-06-30"))
        with mock.patch.object(market, "get_annual_financials_fallback") as fallback:
            cd = prov.get_company_data("TM")
        fallback.assert_not_called()
        self.assertEqual(cd.source_notes[0], "Fundamentals: SEC EDGAR (CIK 0001094517)")
        self.assertFalse(any(n.startswith("WARNING") for n in cd.source_notes), cd.source_notes)

    def test_current_balance_sheet_does_not_make_stale_flows_current(self) -> None:
        # The DCF runs on the flows: FY2010-2013 next to a 2026 balance sheet
        # is still stale.
        prov, _ = self._provider(edgar_fundamentals([2010, 2011, 2012, 2013], "2026-06-30"))
        cd = prov.get_company_data("TM")
        self.assertEqual(cd.financials.fiscal_years, [2021, 2022, 2023, 2024])
        self.assertTrue(cd.source_notes[0].startswith(
            "WARNING: EDGAR fundamentals (CIK 0001094517) end FY2013 (balance sheet 2026-06-30)"),
            cd.source_notes)

    def test_age_is_measured_from_the_latest_fiscal_year(self) -> None:
        # 31 December of the latest label, whatever the balance-sheet date.
        fin, bs, cik, _name = edgar_fundamentals([2023, 2024], "2021-12-31")
        self.assertEqual(provider_mod._latest_period_end(fin, bs), datetime.date(2024, 12, 31))
        self.assertIsNone(HybridProvider._edgar_staleness(fin, bs, cik))
        fin, bs, cik, _name = edgar_fundamentals([2012, 2013], "2026-06-30")
        self.assertEqual(provider_mod._latest_period_end(fin, bs), datetime.date(2013, 12, 31))
        self.assertIsNotNone(HybridProvider._edgar_staleness(fin, bs, cik))
        # Two years (730 days) is the limit; one day more is stale.
        fin, bs, cik, _name = edgar_fundamentals([2023], "")
        with mock.patch.object(provider_mod, "_today", return_value=datetime.date(2025, 12, 30)):
            self.assertIsNone(HybridProvider._edgar_staleness(fin, bs, cik))
        with mock.patch.object(provider_mod, "_today", return_value=datetime.date(2025, 12, 31)):
            self.assertIn("(balance sheet undated), more than 2 years before 2025-12-31",
                          HybridProvider._edgar_staleness(fin, bs, cik))
        # Without a fiscal-year label the balance-sheet date decides.
        fin, bs, cik, _name = edgar_fundamentals([], "2024-09-28")
        self.assertEqual(provider_mod._latest_period_end(fin, bs), datetime.date(2024, 9, 28))
        self.assertIsNone(HybridProvider._edgar_staleness(fin, bs, cik))
        bs.as_of = "2024-09-27"
        self.assertIn("more than 2 years before 2026-09-28",
                      HybridProvider._edgar_staleness(fin, bs, cik))
        bs.as_of = ""
        self.assertIsNone(provider_mod._latest_period_end(fin, bs))
        self.assertIsNone(HybridProvider._edgar_staleness(fin, bs, cik))


class CompRowTests(unittest.TestCase):
    """Peer comps rows on one currency and share basis (Yahoo shapes, September 2026)."""

    # BP ADR: USD statements, but Yahoo's EV (510B) is on another share basis.
    BP = {"currency": "USD", "financialCurrency": "USD", "currentPrice": 44.79,
          "sharesOutstanding": 2575342288, "marketCap": 115349585920,
          "enterpriseValue": 510331060224, "totalDebt": 72692998144, "totalCash": 37225000960,
          "totalRevenue": 215465000960, "ebitda": 39197999104, "enterpriseToEbitda": 13.019,
          "enterpriseToRevenue": 2.369, "priceToBook": 7.93305, "trailingPE": 21.43,
          "pegRatio": 0.06}
    # TSM ADR: TWD statements against a USD quote.
    TSM = {"currency": "USD", "financialCurrency": "TWD", "currentPrice": 451.98,
           "sharesOutstanding": 5186474013, "marketCap": 2344182611968,
           "enterpriseValue": 16289117503488, "totalDebt": 1068558516224,
           "totalCash": 3518010228736, "totalRevenue": 4440492343296, "ebitda": 3167467077632,
           "enterpriseToEbitda": 5.143, "enterpriseToRevenue": 3.668, "priceToBook": 92.815,
           "trailingPE": 33.65, "pegRatio": 0.86}

    def test_consistent_row_keeps_yahoo_figures(self) -> None:
        info = {"currency": "USD", "financialCurrency": "USD", "currentPrice": 340.7,
                "sharesOutstanding": 14594180000, "marketCap": 4972237291520,
                "enterpriseValue": 4999582056448, "totalDebt": 84343996416,
                "totalCash": 62399000576, "totalRevenue": 466822987776, "ebitda": 167959003136,
                "enterpriseToEbitda": 29.767, "enterpriseToRevenue": 10.71,
                "priceToBook": 46.29, "trailingPE": 39.03, "pegRatio": 2.74}
        row = client({"AAPL": ticker(info)}).get_comp_row("AAPL")
        self.assertEqual((row.market_cap, row.enterprise_value, row.ev_ebitda, row.ev_sales,
                          row.pe, row.pb, row.peg),
                         (4972237291520, 4999582056448, 29.767, 10.71, 39.03, 46.29, 2.74))
        self.assertFalse(hasattr(row, "_source_notes"))

    def test_adr_ev_on_another_share_basis_is_rebuilt(self) -> None:
        row = client({"BP": ticker(self.BP)}).get_comp_row("BP")
        ev = 115349585920 + 72692998144 - 37225000960
        self.assertAlmostEqual(row.enterprise_value, ev)
        self.assertAlmostEqual(row.ev_ebitda, ev / 39197999104)  # 3.83, not 13.0
        self.assertAlmostEqual(row.ev_sales, ev / 215465000960)
        self.assertIsNone(row.pb)                                 # 7.9 on ordinary shares
        self.assertEqual((row.pe, row.peg, row.market_cap), (21.43, 0.06, 115349585920))
        self.assertEqual(len(row._source_notes), 1)
        self.assertTrue(row._source_notes[0].startswith(
            "BP: EV rebuilt as market cap + (total debt - total cash) = 150,817,583,104 USD "
            "(Yahoo's 510,331,060,224 is more than 25% away"), row._source_notes)
        self.assertIn("EV/EBITDA 13 -> 3.85, EV/Sales 2.37 -> 0.7; P/B 7.93 left out",
                      row._source_notes[0])

    def test_foreign_statements_are_converted_before_the_multiples(self) -> None:
        c = client({"TSM": ticker(self.TSM), "TWDUSD=X": fx_ticker(1 / 32)})
        row = c.get_comp_row("TSM")
        ev = 2344182611968 + (1068558516224 - 3518010228736) / 32
        self.assertAlmostEqual(row.enterprise_value, ev)
        self.assertAlmostEqual(row.ev_ebitda, ev / (3167467077632 / 32))  # ~23, not 5.1
        self.assertAlmostEqual(row.ev_sales, ev / (4440492343296 / 32))
        self.assertIsNone(row.pb)                                         # 92.8
        self.assertEqual(row.pe, 33.65)
        self.assertIn("x 0.03125 (TWDUSD=X)", row._source_notes[0])
        self.assertIn("Yahoo mixes TWD statements with the USD quote", row._source_notes[0])

    def test_foreign_statements_without_an_fx_rate_leave_the_ev_fields_out(self) -> None:
        row = client({"TSM": ticker(self.TSM)}).get_comp_row("TSM")
        self.assertEqual((row.enterprise_value, row.ev_ebitda, row.ev_sales, row.pb),
                         (None, None, None, None))
        self.assertEqual((row.pe, row.peg), (33.65, 0.86))  # still a usable peer
        self.assertEqual(row._source_notes, [
            "TSM: Yahoo's enterprise value, EV/EBITDA, EV/Sales and P/B mix TWD statements "
            "with the USD quote and could not be rebuilt (no TWD->USD exchange rate); left "
            "out of the comps"])

    def test_pence_line_with_dollar_statements(self) -> None:
        # BP.L: marketCap in GBP (price in pence), statements in USD.
        info = {"currency": "GBp", "financialCurrency": "USD", "currentPrice": 563.5,
                "sharesOutstanding": 15452053728, "marketCap": 87072317440,
                "enterpriseValue": 140151668736, "totalDebt": 72692998144,
                "totalCash": 37225000960, "totalRevenue": 215465000960, "ebitda": 39197999104,
                "enterpriseToEbitda": 3.575, "enterpriseToRevenue": 0.65, "priceToBook": 2.0168,
                "trailingPE": 21.67}
        row = client({"BP.L": ticker(info), "USDGBP=X": fx_ticker(0.75)}).get_comp_row("BP.L")
        ev = 87072317440 + (72692998144 - 37225000960) * 0.75
        self.assertEqual(row.market_cap, 87072317440)
        self.assertAlmostEqual(row.enterprise_value, ev)
        self.assertAlmostEqual(row.ev_ebitda, ev / (39197999104 * 0.75))
        self.assertIsNone(row.pb)
        # A market cap Yahoo gives in pence is scaled to pounds.
        info = dict(info, marketCap=8707231744000)
        row = client({"BP.L": ticker(info), "USDGBP=X": fx_ticker(0.75)}).get_comp_row("BP.L")
        self.assertAlmostEqual(row.market_cap, 87072317440)
        self.assertAlmostEqual(row.enterprise_value, ev)

    def test_missing_yahoo_ev_is_filled_and_book_value_kept(self) -> None:
        info = {"currency": "USD", "financialCurrency": "USD", "marketCap": 1000.0,
                "totalDebt": 300.0, "totalCash": 100.0, "totalRevenue": 600.0, "ebitda": -50.0,
                "priceToBook": 2.5, "trailingPE": 12.0}
        row = client({"X": ticker(info)}).get_comp_row("X")
        self.assertEqual((row.enterprise_value, row.ev_sales, row.ev_ebitda, row.pb),
                         (1200.0, 2.0, None, 2.5))  # negative EBITDA: no EV/EBITDA
        self.assertIn("(Yahoo reports no enterprise value)", row._source_notes[0])

    def test_minority_interest_that_explains_yahoo_ev_keeps_it(self) -> None:
        # BN: Yahoo's EV (460.5B) = market cap + debt - cash + minority interest
        # + preferred; without the last two it looks 37% too high.
        info = {"currency": "USD", "financialCurrency": "USD", "currentPrice": 36.0,
                "sharesOutstanding": 2258000000, "marketCap": 81300000000,
                "enterpriseValue": 460500000000, "totalDebt": 274200000000,
                "totalCash": 20000000000, "totalRevenue": 80600000000, "ebitda": 32700000000,
                "enterpriseToEbitda": 14.1, "enterpriseToRevenue": 5.71, "priceToBook": 1.92}
        qbs = yahoo_quarters(**{"Minority Interest": [120065000000.0, 119000000000.0, NaN],
                                "Preferred Stock": [4088000000.0, 4088000000.0, NaN],
                                "Common Stock Equity": [42300000000.0, 41000000000.0, NaN]})
        row = client({"BN": ticker(info, qbs=qbs)}).get_comp_row("BN")
        self.assertEqual((row.enterprise_value, row.ev_ebitda, row.ev_sales, row.pb),
                         (460500000000, 14.1, 5.71, 1.92))
        self.assertFalse(hasattr(row, "_source_notes"))

    def test_balance_sheet_adds_minority_interest_and_rebuilds_book_value(self) -> None:
        # TSM with its (TWD) balance sheet: minority interest joins the EV and
        # P/B becomes market cap / (common equity x FX) instead of 92.8.
        qbs = yahoo_quarters(**{"Minority Interest": [NaN, 32000000000.0, 30000000000.0],
                                "Common Stock Equity": [NaN, 6400000000000.0, 6000000000000.0]})
        c = client({"TSM": ticker(self.TSM, qbs=qbs), "TWDUSD=X": fx_ticker(1 / 32)})
        row = c.get_comp_row("TSM")
        ev = 2344182611968 + (1068558516224 - 3518010228736 + 32000000000.0) / 32
        self.assertAlmostEqual(row.enterprise_value, ev)
        self.assertAlmostEqual(row.ev_ebitda, ev / (3167467077632 / 32))
        self.assertAlmostEqual(row.pb, 2344182611968 / (6400000000000.0 / 32))  # 11.7
        self.assertIn("EV rebuilt as market cap + (total debt - total cash + minority "
                      "interest + preferred stock) x 0.03125 (TWDUSD=X)", row._source_notes[0])
        self.assertIn("P/B 92.8 -> 11.7 (market cap / common equity at 2026-03-31)",
                      row._source_notes[0])  # the sparse 2026-06-30 column is skipped

    def test_negative_yahoo_ev_against_a_positive_rebuild_is_replaced(self) -> None:
        # BRK-B: Yahoo's EV is -234B and its P/B 0.001 (class A book value
        # per share against the class B price).
        info = {"currency": "USD", "financialCurrency": "USD", "marketCap": 1078.4,
                "enterpriseValue": -233.9, "totalDebt": 128.6, "totalCash": 365.5,
                "totalRevenue": 384.0, "ebitda": 130.0, "enterpriseToEbitda": -1.79,
                "enterpriseToRevenue": -0.608, "priceToBook": 0.000965}
        qbs = yahoo_quarters(**{"Minority Interest": [12.0, 12.0, 11.0],
                                "Common Stock Equity": [700.0, 690.0, 680.0]})
        row = client({"BRK-B": ticker(info, qbs=qbs)}).get_comp_row("BRK-B")
        self.assertAlmostEqual(row.enterprise_value, 1078.4 + 128.6 - 365.5 + 12.0)
        self.assertAlmostEqual(row.ev_ebitda, (1078.4 + 128.6 - 365.5 + 12.0) / 130.0)
        self.assertAlmostEqual(row.pb, 1078.4 / 700.0)
        self.assertIn("(Yahoo's -234 is more than 25% away", row._source_notes[0])

    def test_rows_without_debt_or_cash_figures_are_left_alone(self) -> None:
        info = {"currency": "USD", "financialCurrency": "USD", "marketCap": 1000.0,
                "enterpriseValue": 5000.0, "enterpriseToEbitda": 10.0, "priceToBook": 3.0}
        row = client({"X": ticker(info)}).get_comp_row("X")
        self.assertEqual((row.enterprise_value, row.ev_ebitda, row.pb), (5000.0, 10.0, 3.0))

    # VALE ADR: .info amounts in BRL (financialCurrency), statement tables in USD.
    VALE = {"currency": "USD", "financialCurrency": "BRL", "currentPrice": 13.595,
            "sharesOutstanding": 4255762795, "marketCap": 57857093632,
            "enterpriseValue": 142917926912, "totalDebt": 110007001088,
            "totalCash": 29841999872, "totalRevenue": 218068992000, "ebitda": 77332996096,
            "enterpriseToEbitda": 1.848, "enterpriseToRevenue": 0.655,
            "priceToBook": 1.493468, "trailingPE": 27.19}

    @staticmethod
    def _annual_revenue(*values: float) -> pd.DataFrame:
        cols = pd.to_datetime(["2025-12-31", "2024-12-31"])
        return pd.DataFrame({"Total Revenue": list(values)}, index=cols).T.astype(float)

    def test_usd_tables_under_a_brl_financial_currency_take_no_fx(self) -> None:
        qbs = yahoo_quarters(**{"Total Debt": [NaN, 22195000000.0, 21801000000.0],
                                "Common Stock Equity": [NaN, 36651000000.0, 33509000000.0],
                                "Minority Interest": [NaN, 901000000.0, 841000000.0]})
        c = client({"VALE": ticker(self.VALE, qbs=qbs,
                                   fin=self._annual_revenue(38403000000.0, 38056000000.0)),
                    "BRLUSD=X": fx_ticker(0.19181)})
        row = c.get_comp_row("VALE")
        ev = 57857093632 + (110007001088 - 29841999872) * 0.19181 + 901000000.0
        self.assertAlmostEqual(row.enterprise_value, ev)
        self.assertAlmostEqual(row.ev_ebitda, ev / (77332996096 * 0.19181))
        self.assertAlmostEqual(row.pb, 57857093632 / 36651000000.0)  # 1.58, not 8.2
        self.assertTrue(row._source_notes[0].startswith(
            "VALE: EV rebuilt as market cap + (total debt - total cash) x 0.19181 (BRLUSD=X) "
            "+ minority interest + preferred stock = "), row._source_notes)
        self.assertIn("P/B 1.49 -> 1.58 (market cap / common equity at 2026-03-31)",
                      row._source_notes[0])
        self.assertEqual(row._source_notes[1], (
            "VALE: Yahoo's statement tables are in USD, not the BRL of its financialCurrency "
            "(table Total Debt / totalDebt 0.2018 and Total Revenue / totalRevenue 0.1761, "
            "against BRL->USD 0.1918 (BRLUSD=X)); minority interest, preferred stock and "
            "common equity are read in USD"))

    def test_tables_in_the_reporting_currency_are_converted_as_before(self) -> None:
        # ITUB: a bank's table debt is half of Yahoo's totalDebt, but revenue
        # agrees, so its tables are BRL like its .info amounts.
        info = {"currency": "USD", "financialCurrency": "BRL", "marketCap": 88395415552,
                "enterpriseValue": 731363409920, "totalDebt": 1115894054912,
                "totalCash": 484164993024, "totalRevenue": 143713992704,
                "priceToBook": 2.0810442, "trailingPE": 9.9}
        qbs = yahoo_quarters(**{"Total Debt": [560389000000.0, 560875000000.0, NaN],
                                "Common Stock Equity": [217779000000.0, 209705000000.0, NaN],
                                "Minority Interest": [10247000000.0, 10312000000.0, NaN]})
        c = client({"ITUB": ticker(info, qbs=qbs,
                                   fin=self._annual_revenue(165243000000.0, 158568000000.0)),
                    "BRLUSD=X": fx_ticker(0.19181)})
        row = c.get_comp_row("ITUB")
        self.assertAlmostEqual(row.pb, 88395415552 / (217779000000.0 * 0.19181))  # 2.12
        self.assertAlmostEqual(row.enterprise_value, 88395415552 + (
            1115894054912 - 484164993024 + 10247000000.0) * 0.19181)
        self.assertEqual(len(row._source_notes), 1)

    def test_same_currency_ev_is_kept_when_the_balance_sheet_cannot_be_read(self) -> None:
        # BN without Yahoo's balance sheet (an outage): the 37% gap may be the
        # minority interest and preferred stock its EV includes.
        info = {"currency": "USD", "financialCurrency": "USD", "marketCap": 81300000000,
                "enterpriseValue": 460500000000, "totalDebt": 274200000000,
                "totalCash": 20000000000, "totalRevenue": 80600000000, "ebitda": 32700000000,
                "enterpriseToEbitda": 14.1, "enterpriseToRevenue": 5.71, "priceToBook": 1.92}
        row = client({"BN": ticker(info)}).get_comp_row("BN")
        self.assertEqual((row.enterprise_value, row.ev_ebitda, row.ev_sales, row.pb),
                         (460500000000, 14.1, 5.71, 1.92))
        self.assertEqual(row._source_notes, [
            "BN: Yahoo's enterprise value 460,500,000,000 is 37% above market cap + total "
            "debt - total cash (335,500,000,000); kept with its EV multiples and P/B, since "
            "the minority interest and preferred stock that Yahoo's EV includes could not be "
            "checked without Yahoo's balance sheet"])
        # Gaps minority interest cannot explain are still rebuilt.
        for label, yahoo_ev in (("below the check", 200000000000),
                                ("over twice the check", 700000000000)):
            with self.subTest(label):
                tk = ticker(dict(info, enterpriseValue=yahoo_ev))
                row = client({"BN": tk}).get_comp_row("BN")
                self.assertAlmostEqual(row.enterprise_value, 335500000000)
                self.assertIsNone(row.pb)
        # A balance sheet without minority interest settles it: rebuilt.
        qbs = yahoo_quarters(**{"Common Stock Equity": [42300000000.0, 41000000000.0, NaN]})
        row = client({"BN": ticker(info, qbs=qbs)}).get_comp_row("BN")
        self.assertAlmostEqual(row.enterprise_value, 335500000000)
        self.assertAlmostEqual(row.pb, 81300000000 / 42300000000.0)

    def test_a_failed_check_leaves_mixed_currency_figures_out(self) -> None:
        c = client({"TSM": ticker(self.TSM), "BP": ticker(self.BP)})
        c._comp_ev_fields = mock.Mock(side_effect=RuntimeError("boom"))
        row = c.get_comp_row("TSM")
        self.assertEqual((row.enterprise_value, row.ev_ebitda, row.ev_sales, row.pb),
                         (None, None, None, None))
        self.assertEqual((row.market_cap, row.pe), (2344182611968, 33.65))
        self.assertEqual(row._source_notes, [
            "TSM: the EV and P/B check failed (RuntimeError); Yahoo's enterprise value, "
            "EV/EBITDA, EV/Sales and P/B mix TWD statements with the USD quote and are left "
            "out of the comps"])
        # A single-currency row keeps Yahoo's figures.
        row = c.get_comp_row("BP")
        self.assertEqual((row.enterprise_value, row.ev_ebitda, row.pb),
                         (510331060224, 13.019, 7.93305))
        self.assertEqual(row._source_notes, [
            "BP: the EV and P/B check failed (RuntimeError); Yahoo's enterprise value, EV "
            "multiples and P/B are kept as reported"])


def captive_balance(scale: float = 1.0, **rows) -> pd.DataFrame:
    """A newest-column balance sheet; `rows` as {label: value} (NaN elsewhere)."""
    data = {"Total Assets": 1000.0, "Total Debt": 300.0, "Stockholders Equity": 300.0,
            "Cash And Cash Equivalents": 100.0}
    data.update(rows)
    frame = pd.DataFrame({COLS[0]: data, COLS[1]: {k: NaN for k in data}})
    return (frame * scale).astype(float)


class CaptiveFinanceFallbackTests(unittest.TestCase):
    """The yfinance fallback marks a consolidated captive finance arm like EDGAR does."""

    AUTO = {"currency": "USD", "financialCurrency": "USD", "sector": "Consumer Cyclical",
            "industry": "Auto Manufacturers"}

    def _fallback(self, info, bs, extra=None):
        tickers = {"X": ticker(info, fin=income(), cf=cashflow(), bs=bs)}
        tickers.update(extra or {})
        return client(tickers).get_annual_financials_fallback("X")

    def test_toyota_like_noncurrent_finance_receivables_mark_captive_finance(self) -> None:
        # TM (JPY statements, USD ADR): non-current receivables 24%, debt 41%.
        info = dict(self.AUTO, financialCurrency="JPY")
        bs = captive_balance(150.0, **{"Non Current Accounts Receivable": 240.0,
                                       "Total Debt": 410.0})
        fin, snap = self._fallback(info, bs, {"JPYUSD=X": fx_ticker(1 / 150)})
        self.assertEqual(fin._financial_kind, "captive_finance")  # survives the FX copy
        self.assertAlmostEqual(snap.total_debt, 410.0)
        self.assertEqual(fin._source_notes[0], (
            "WARNING: yfinance statements mark this company as a group with a captive "
            "finance arm (non-current receivables are 24% of total assets and total debt "
            "41%); the finance arm's debt, leases and receivables are consolidated (total "
            "debt includes the borrowing that funds its customer loans and leases), so the "
            "FCFF DCF and FCFE do not fit it and the DDM and comps are the better guides"))
        self.assertTrue(fin._source_notes[1].startswith("Fundamentals converted from JPY"))

    def test_finance_book_reported_as_current_receivables_in_an_auto_industry(self) -> None:
        # Nissan / Renault: Yahoo reports the sales-finance book as current.
        bs = captive_balance(**{"Accounts Receivable": 30.0, "Other Receivables": 370.0,
                                "Total Debt": 450.0})
        fin, _ = self._fallback(self.AUTO, bs)
        self.assertEqual(fin._financial_kind, "captive_finance")
        self.assertIn("(receivables are 40% of total assets and total debt 45%, for a maker "
                      "in Auto Manufacturers)", fin._source_notes[0])

    def test_hidden_finance_arm_of_an_automaker_gets_a_note_only(self) -> None:
        # BMW / Hyundai: Yahoo lumps the finance book into unlabelled lines.
        bs = captive_balance(**{"Accounts Receivable": 150.0, "Total Debt": 330.0})
        fin, _ = self._fallback(self.AUTO, bs)
        self.assertIsNone(fin._financial_kind)
        self.assertFalse(any(n.startswith("WARNING") for n in fin._source_notes))
        self.assertIn("Auto Manufacturers companies often consolidate a captive finance arm, "
                      "which Yahoo's statements do not always show; none is visible here "
                      "(receivables 15% and total debt 33% of total assets), but if there is "
                      "one, its debt and receivables are in the DCF, FCFE and WACC inputs",
                      fin._source_notes)

    def test_non_captive_filers_are_not_marked(self) -> None:
        cases = {
            # NVO: a drug maker with 1% non-current receivables.
            "pharma": ({"sector": "Healthcare", "industry": "Drug Manufacturers - General"},
                       {"Non Current Accounts Receivable": 10.0, "Accounts Receivable": 130.0}),
            # URI / AerCap: the rental fleet is property, not receivables.
            "lessor": ({"sector": "Industrials", "industry": "Rental & Leasing Services"},
                       {"Accounts Receivable": 80.0, "Total Debt": 520.0, "Net PPE": 620.0}),
            # Dell: a small financing line (6% non-current).
            "small finance line": ({"sector": "Technology", "industry": "Computer Hardware"},
                                   {"Non Current Accounts Receivable": 60.0}),
            # Large long-term receivables not funded by debt.
            "unfunded": ({"sector": "Industrials", "industry": "Aerospace & Defense"},
                         {"Non Current Accounts Receivable": 200.0, "Total Debt": 50.0}),
            # Concession receivables of a utility: outside the durable-goods sectors.
            "utility": ({"sector": "Utilities", "industry": "Utilities - Regulated Water"},
                        {"Non Current Accounts Receivable": 300.0, "Total Debt": 500.0}),
            # Ferrari-like: an automaker with a modest finance book.
            "small auto finance": (dict(self.AUTO), {"Receivables": 220.0}),
        }
        for label, (meta, rows) in cases.items():
            with self.subTest(label):
                info = dict({"currency": "USD", "financialCurrency": "USD"}, **meta)
                fin, _ = self._fallback(info, captive_balance(**rows))
                self.assertIsNone(fin._financial_kind)
                self.assertFalse(any("WARNING" in n for n in fin._source_notes),
                                 fin._source_notes)

    def test_captive_mark_reaches_the_company_and_the_models(self) -> None:
        from equity_valuation.utils import financial_institution_detail

        edgar = mock.Mock()
        edgar.get_annual_financials.side_effect = DataError("statements are in JPY")
        info = dict(self.AUTO, currentPrice=188.0, sharesOutstanding=10.0, marketCap=1880.0)
        bs = captive_balance(**{"Non Current Accounts Receivable": 240.0, "Total Debt": 410.0})
        c = client({"TM": ticker(info, fin=income(), cf=cashflow(), bs=bs)})
        with mock.patch.object(provider_mod, "_today", return_value=TODAY):
            cd = HybridProvider(edgar=edgar, market=c).get_company_data("TM")
        self.assertEqual(cd.financials._financial_kind, "captive_finance")
        self.assertEqual(cd.source_notes[1], "Fundamentals: yfinance fallback")
        self.assertTrue(cd.source_notes[2].startswith(
            "WARNING: yfinance statements mark this company as a group with a captive "
            "finance arm"), cd.source_notes)
        self.assertEqual(financial_institution_detail(cd)[1], "captive_finance")

    def test_fx_copy_keeps_the_financial_kind_marker(self) -> None:
        fin, bs, _cik, _name = edgar_fundamentals([2024], "2024-12-31")
        fin._financial_kind = "bank"
        scaled, _ = market_mod.scale_fundamentals(fin, bs, 1.3)
        self.assertEqual(scaled._financial_kind, "bank")
        self.assertAlmostEqual(scaled.revenue[0], 130.0)
        self.assertFalse(hasattr(scaled, "_source_notes"))


# The EDGAR client's warning (edgar.py) when interest implies debt its tags miss.
EDGAR_DEBT_WARNING = (
    "WARNING: FY2025 interest expense is 1,254,000,000 but no debt was found under the SEC "
    "debt tags this parser reads (the filer may tag its debt with company-specific "
    "elements); net debt and the WACC debt weight may be understated")


def yahoo_quarters(**rows) -> pd.DataFrame:
    """A quarterly balance sheet {label: [2026-06-30, 2026-03-31, 2025-12-31]}."""
    cols = pd.to_datetime(["2026-06-30", "2026-03-31", "2025-12-31"])
    return pd.DataFrame(rows, index=cols).T.astype(float)


class EdgarDebtBackfillTests(unittest.TestCase):
    """Debt (and T-bills) under company-specific EDGAR tags come from Yahoo (F, BRK-B)."""

    INFO = {"currency": "USD", "financialCurrency": "USD", "currentPrice": 12.0,
            "sharesOutstanding": 100.0, "marketCap": 1200.0, "sector": "Consumer Cyclical",
            "industry": "Auto Manufacturers"}
    # Ford-like: debt 157B at 2026-03-31 (long-term 106 + current 51; leases apart).
    FORD = dict(**{"Total Assets": [285.5, 288.0, 289.2], "Total Debt": [163.3, 159.5, 165.7],
                   "Long Term Debt": [109.6, 106.1, 106.0], "Current Debt": [51.3, 51.0, 57.3],
                   "Capital Lease Obligations": [2.3, 2.4, 2.4],
                   "Non Current Accounts Receivable": [59.4, 60.0, 61.5],
                   "Cash And Cash Equivalents": [18.6, 17.9, 23.4],
                   "Cash Cash Equivalents And Short Term Investments": [31.3, 30.5, 38.5]})

    def setUp(self) -> None:
        patcher = mock.patch.object(provider_mod, "_today", return_value=TODAY)
        patcher.start()
        self.addCleanup(patcher.stop)

    # The EDGAR client's gap notes that put Ford's debt at 0 (live, September 2026).
    ZERO_DEBT_NOTES = [
        "total debt unavailable on EDGAR; set to 0.0",
        "long-term debt (noncurrent) last reported 2020-12-31 (291,000,000); treated as 0 at "
        "the 2026-03-31 balance sheet",
        "total debt last reported 2020-12-31 (471,000,000); treated as 0 at the 2026-03-31 "
        "balance sheet",
        "debt line (UnsecuredLongTermDebt) last reported 2020-12-31 (294,000,000); treated as "
        "0 at the 2026-03-31 balance sheet",
    ]
    OTHER_NOTES = [
        "capex unavailable on EDGAR; filled with 0.0",
        "inventory last reported 2020-12-31 (10,000,000); treated as 0 at the 2026-03-31 "
        "balance sheet",
    ]

    def _company(self, as_of="2026-03-31", debt=0.0, cash=30.5, notes=None, kind=None,
                 info=None, qbs=None, extra=None, fin=None):
        edgar_fin, bs, cik, name = edgar_fundamentals([2022, 2023, 2024, 2025], as_of)
        edgar_fin._source_notes = list(notes if notes is not None else [EDGAR_DEBT_WARNING])
        edgar_fin._financial_kind = kind
        bs.total_debt, bs.cash_and_investments = debt, cash
        edgar = mock.Mock()
        edgar.get_annual_financials.return_value = (edgar_fin, bs, cik, name)
        tk = ticker(info or self.INFO, qbs=qbs if qbs is not None else yahoo_quarters(**self.FORD),
                    fin=fin)
        tickers = {"F": tk}
        tickers.update(extra or {})
        return HybridProvider(edgar=edgar, market=client(tickers)).get_company_data("F"), tk

    def test_ford_like_debt_is_backfilled_at_the_edgar_date_and_captive_marked(self) -> None:
        cd, _ = self._company(notes=[EDGAR_DEBT_WARNING] + self.ZERO_DEBT_NOTES
                              + self.OTHER_NOTES)
        self.assertAlmostEqual(cd.balance_sheet.total_debt, 157.1)   # 106.1 + 51.0
        self.assertEqual(cd.balance_sheet.cash_and_investments, 30.5)  # not clearly missing
        self.assertEqual(cd.financials._financial_kind, "captive_finance")
        self.assertTrue(cd.source_notes[1].startswith(
            "WARNING: Yahoo's balance-sheet lines mark this company as a group with a captive "
            "finance arm (non-current receivables are 21% of total assets and total debt "
            "55%)"), cd.source_notes)
        self.assertEqual(cd.source_notes[2], EDGAR_DEBT_WARNING + (
            "; backfilled from Yahoo's balance sheet at 2026-03-31: total debt 157 "
            "(long-term plus current debt, lease obligations excluded)"))
        # EDGAR's notes that put debt at 0 no longer contradict the backfill.
        for note in self.ZERO_DEBT_NOTES:
            self.assertIn(note + "; replaced by the Yahoo balance-sheet backfill (see the "
                          "interest-expense WARNING)", cd.source_notes)
        for note in self.OTHER_NOTES:
            self.assertIn(note, cd.source_notes)
        self.assertEqual(sum("replaced by the Yahoo" in n for n in cd.source_notes), 4)

    def test_berkshire_like_treasury_bills_are_backfilled_with_the_debt(self) -> None:
        info = dict(self.INFO, sector="Financial Services", industry="Insurance - Diversified")
        qbs = yahoo_quarters(**{"Total Assets": [1263.1, 1240.0, 1222.2],
                                "Long Term Debt": [125.9, 126.0, 125.8],
                                "Current Debt": [2.7, 3.0, 3.3],
                                "Cash And Cash Equivalents": [40.6, 45.0, 51.9],
                                "Other Short Term Investments": [324.9, 320.0, 321.4]})
        cd, _ = self._company(as_of="2026-06-30", cash=40.6, info=info, qbs=qbs)
        self.assertAlmostEqual(cd.balance_sheet.total_debt, 128.6)
        self.assertAlmostEqual(cd.balance_sheet.cash_and_investments, 365.5)
        self.assertIsNone(cd.financials._financial_kind)  # no captive arm shown
        warning = next(n for n in cd.source_notes if n.startswith(EDGAR_DEBT_WARNING))
        self.assertIn("backfilled from Yahoo's balance sheet at 2026-06-30: total debt 129 "
                      "(long-term plus current debt, lease obligations excluded); cash and "
                      "short-term investments 366 (EDGAR's 41 is cash and equivalents alone",
                      warning)

    # PACCAR's EDGAR cash gap notes before the disposal-group cash tag was read (its
    # cash tag stale since 2019).
    ZERO_CASH_NOTES = [
        "cash & equivalents unavailable on EDGAR; set to 0.0",
        "cash & equivalents last reported 2019-06-30 (3,219,400,000); treated as 0 at the "
        "2026-03-31 balance sheet",
    ]

    def test_paccar_like_zero_cash_is_backfilled_with_the_debt(self) -> None:
        cd, _ = self._company(cash=0.0, notes=[EDGAR_DEBT_WARNING, self.ZERO_DEBT_NOTES[0]]
                              + self.ZERO_CASH_NOTES + self.OTHER_NOTES[1:])
        self.assertAlmostEqual(cd.balance_sheet.total_debt, 157.1)
        # Yahoo's cash and short-term investments at the same 2026-03-31 column.
        self.assertAlmostEqual(cd.balance_sheet.cash_and_investments, 30.5)
        warning = next(n for n in cd.source_notes if n.startswith(EDGAR_DEBT_WARNING))
        self.assertTrue(warning.endswith(
            "; backfilled from Yahoo's balance sheet at 2026-03-31: total debt 157 (long-term "
            "plus current debt, lease obligations excluded); cash and short-term investments "
            "30 (EDGAR's are 0: its cash tags are stale or company-specific, like its debt "
            "tags)"), warning)
        for note in [self.ZERO_DEBT_NOTES[0]] + self.ZERO_CASH_NOTES:
            self.assertIn(note + "; replaced by the Yahoo balance-sheet backfill (see the "
                          "interest-expense WARNING)", cd.source_notes)
        self.assertIn(self.OTHER_NOTES[1], cd.source_notes)  # other gaps are untouched

    def test_zero_cash_is_not_backfilled_without_the_debt(self) -> None:
        # Yahoo shows no debt either: nothing comes from its balance sheet.
        rows = dict(self.FORD, **{"Long Term Debt": [0.0, 0.0, 0.0],
                                  "Current Debt": [0.0, 0.0, 0.0]})
        cd, _ = self._company(cash=0.0, qbs=yahoo_quarters(**rows),
                              notes=[EDGAR_DEBT_WARNING] + self.ZERO_CASH_NOTES)
        self.assertEqual((cd.balance_sheet.total_debt, cd.balance_sheet.cash_and_investments),
                         (0.0, 0.0))
        self.assertIn(EDGAR_DEBT_WARNING + "; Yahoo's balance sheet at 2026-03-31 shows no debt "
                      "either, so nothing was backfilled", cd.source_notes)
        for note in self.ZERO_CASH_NOTES:
            self.assertIn(note, cd.source_notes)
        # EDGAR debt found (no interest WARNING): EDGAR's zero cash stays as reported.
        cd, _ = self._company(debt=50.0, cash=0.0, notes=list(self.ZERO_CASH_NOTES))
        self.assertEqual((cd.balance_sheet.total_debt, cd.balance_sheet.cash_and_investments),
                         (50.0, 0.0))

    def test_edgar_debt_is_never_overwritten(self) -> None:
        notes = [EDGAR_DEBT_WARNING.replace("no debt was found", "total debt read is only 5")]
        cd, tk = self._company(debt=5.0, notes=notes)
        self.assertEqual(cd.balance_sheet.total_debt, 5.0)
        self.assertEqual(cd.source_notes[1], notes[0])
        self.assertIsNone(cd.financials._financial_kind)

    def test_no_backfill_without_the_edgar_warning(self) -> None:
        cd, _ = self._company(debt=0.0, notes=["capex unavailable on EDGAR; filled with 0.0"])
        self.assertEqual(cd.balance_sheet.total_debt, 0.0)
        self.assertIsNone(cd.financials._financial_kind)

    def test_edgar_captive_mark_is_not_repeated(self) -> None:
        edgar_warning = ("WARNING: EDGAR tags mark this company as a group with a captive "
                         "finance arm (finance receivables ... are 48% of total assets)")
        cd, _ = self._company(notes=[edgar_warning, EDGAR_DEBT_WARNING], kind="captive_finance")
        self.assertAlmostEqual(cd.balance_sheet.total_debt, 157.1)
        self.assertEqual(sum("captive finance arm" in n for n in cd.source_notes), 1)

    def test_no_yahoo_balance_sheet_near_the_edgar_date(self) -> None:
        # Nearest Yahoo column is 184 days away; EDGAR's zero-debt note stands.
        cd, _ = self._company(as_of="2025-06-30",
                              notes=[EDGAR_DEBT_WARNING, self.ZERO_DEBT_NOTES[0]])
        self.assertEqual(cd.balance_sheet.total_debt, 0.0)
        self.assertIn(EDGAR_DEBT_WARNING + "; no Yahoo balance sheet within 100 days of "
                      "2025-06-30 to backfill it from", cd.source_notes)
        self.assertIn(self.ZERO_DEBT_NOTES[0], cd.source_notes)
        self.assertFalse(any("replaced by the Yahoo" in n for n in cd.source_notes))

    def test_yahoo_statements_in_another_currency_are_converted_or_skipped(self) -> None:
        info = dict(self.INFO, financialCurrency="EUR")
        cd, _ = self._company(info=info, extra={"EURUSD=X": fx_ticker(1.1)})
        self.assertAlmostEqual(cd.balance_sheet.total_debt, 157.1 * 1.1)
        cd, _ = self._company(info=info)
        self.assertEqual(cd.balance_sheet.total_debt, 0.0)
        self.assertTrue(any("Yahoo's balance sheet is in EUR and no EUR->USD rate could be "
                            "fetched, so nothing was backfilled" in n for n in cd.source_notes))

    def test_total_debt_less_leases_when_the_split_is_missing(self) -> None:
        rows = {k: v for k, v in self.FORD.items() if k not in ("Long Term Debt", "Current Debt")}
        cd, _ = self._company(qbs=yahoo_quarters(**rows))
        self.assertAlmostEqual(cd.balance_sheet.total_debt, 159.5 - 2.4)
        self.assertIn("(total debt less lease obligations)", cd.source_notes[2])

    def test_current_debt_reported_only_with_its_leases(self) -> None:
        # No "Current Debt" line: the current side comes from the line that
        # includes leases, less the current lease line when Yahoo has one.
        rows = {k: v for k, v in self.FORD.items() if k != "Current Debt"}
        rows["Current Debt And Capital Lease Obligation"] = [53.0, 52.4, 58.9]
        rows["Current Capital Lease Obligation"] = [1.7, 1.4, 1.6]
        cd, _ = self._company(qbs=yahoo_quarters(**rows))
        self.assertAlmostEqual(cd.balance_sheet.total_debt, 106.1 + 51.0)
        self.assertIn("(long-term plus current debt, lease obligations excluded)",
                      cd.source_notes[2])
        del rows["Current Capital Lease Obligation"]
        cd, _ = self._company(qbs=yahoo_quarters(**rows))
        self.assertAlmostEqual(cd.balance_sheet.total_debt, 106.1 + 52.4)
        self.assertIn("(long-term plus current debt, current lease obligations included)",
                      cd.source_notes[2])

    def test_yahoo_tables_in_usd_under_a_local_financial_currency(self) -> None:
        # Table Total Debt 163.3 against totalDebt 816.5 and revenue 1,000
        # against 5,555.6 at BRL->USD 0.19: the tables are USD, not converted.
        info = dict(self.INFO, financialCurrency="BRL", totalDebt=816.5,
                    totalRevenue=5555.6)
        c = client({"F": ticker(info, qbs=yahoo_quarters(**self.FORD), fin=income()),
                    "BRLUSD=X": fx_ticker(0.19)})
        items = c.get_balance_sheet_items("F", "2026-03-31")
        self.assertEqual(items["currency"], "USD")
        self.assertEqual(items["currency_note"], (
            "Yahoo's statement tables are in USD, not the BRL of its financialCurrency (table "
            "Total Debt / totalDebt 0.2 and Total Revenue / totalRevenue 0.18, against "
            "BRL->USD 0.19 (BRLUSD=X))"))
        cd, _ = self._company(info=info, fin=income(), extra={"BRLUSD=X": fx_ticker(0.19)})
        self.assertAlmostEqual(cd.balance_sheet.total_debt, 157.1)  # no BRL->USD scaling
        self.assertTrue(cd.source_notes[2].endswith(
            "(long-term plus current debt, lease obligations excluded) (Yahoo's statement "
            "tables are in USD, not the BRL of its financialCurrency (table Total Debt / "
            "totalDebt 0.2 and Total Revenue / totalRevenue 0.18, against BRL->USD 0.19 "
            "(BRLUSD=X)))"), cd.source_notes)
        # With tables in the reporting currency the amounts are converted.
        items = client({"F": ticker(dict(info, totalDebt=163.3), fin=income(),
                                    qbs=yahoo_quarters(**self.FORD))}
                       ).get_balance_sheet_items("F", "2026-03-31")
        self.assertEqual((items["currency"], items["currency_note"]), ("BRL", None))


def yahoo_cashflow(dates=("2025-12-31", "2024-12-31", "2023-12-31", "2022-12-31"),
                   **rows) -> pd.DataFrame:
    """An annual cash-flow statement {label: [one value per date]}, newest first."""
    return pd.DataFrame(rows, index=pd.to_datetime(list(dates))).T.astype(float)


class EdgarCapexBackfillTests(unittest.TestCase):
    """Capex years no EDGAR tag covers come from Yahoo's cash flow (PSX, NEE, AER)."""

    INFO = {"currency": "USD", "financialCurrency": "USD", "currentPrice": 250.0,
            "sharesOutstanding": 10.0, "marketCap": 2500.0}
    YEARS = [2021, 2022, 2023, 2024, 2025]
    # Phillips 66's Yahoo 'Capital Expenditure' (millions), FY2025 back to FY2022.
    CAPEX = {"Capital Expenditure": [-2233.0, -1859.0, -2155.0, -1888.0]}
    ALL_GAP = "capex unavailable on EDGAR; filled with 0.0"

    def setUp(self) -> None:
        patcher = mock.patch.object(provider_mod, "_today", return_value=TODAY)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _company(self, capex, notes, cf=None, info=None, extra=None, years=None,
                 kind="captive_finance", da=None):
        fin, bs, cik, name = edgar_fundamentals(list(years or self.YEARS), "2026-06-30")
        fin.capex = list(capex)
        if da is not None:
            fin.dep_amort = list(da)
        fin._source_notes = list(notes)
        fin._financial_kind = kind  # any marker: it must survive the copy
        edgar = mock.Mock()
        edgar.get_annual_financials.return_value = (fin, bs, cik, name)
        tickers = {"PSX": ticker(info or self.INFO,
                                 cf=cf if cf is not None else yahoo_cashflow(**self.CAPEX))}
        tickers.update(extra or {})
        cd = HybridProvider(edgar=edgar, market=client(tickers)).get_company_data("PSX")
        return cd, fin

    def test_psx_like_capex_without_any_tag_is_backfilled_by_fiscal_year(self) -> None:
        cd, fin = self._company([0.0] * 5, [self.ALL_GAP])
        self.assertEqual(cd.financials.capex, [0.0, 1888.0, 2155.0, 1859.0, 2233.0])
        self.assertEqual(fin.capex, [0.0] * 5)  # EDGAR's object is not mutated
        self.assertEqual(cd.financials.fiscal_years, self.YEARS)
        self.assertEqual(cd.financials.dep_amort, fin.dep_amort)
        self.assertEqual(cd.financials._financial_kind, "captive_finance")
        self.assertIn(self.ALL_GAP + (
            "; backfilled from Yahoo's annual cash-flow statement (capital expenditure for the "
            "same fiscal year): FY2022 1,888 (year to 2022-12-31), FY2023 2,155 (year to "
            "2023-12-31), FY2024 1,859 (year to 2024-12-31), FY2025 2,233 (year to 2025-12-31); "
            "FY2021 not in Yahoo's statement, left at 0.0"), cd.source_notes)
        self.assertFalse(any(n == self.ALL_GAP for n in cd.source_notes))

    def test_only_the_years_edgar_zero_filled_are_backfilled(self) -> None:
        # VZ/AER-like: a tag gap in some years; reported years keep EDGAR's figure
        # (here within 2% of Yahoo's, so the two define capex alike).
        gap = "capex not reported on EDGAR for FY2023-2024; filled with 0.0"
        cd, _ = self._company([1800.0, 1850.0, 0.0, 0.0, 2250.0], [gap])
        self.assertEqual(cd.financials.capex, [1800.0, 1850.0, 2155.0, 1859.0, 2250.0])
        self.assertIn(gap + "; backfilled from Yahoo's annual cash-flow statement (capital "
                      "expenditure for the same fiscal year): FY2023 2,155 (year to 2023-12-31), "
                      "FY2024 1,859 (year to 2024-12-31)", cd.source_notes)
        # Non-contiguous gaps in the note ("FY2021, FY2023") are both read.
        gap = "capex not reported on EDGAR for FY2021, FY2023; filled with 0.0"
        cd, _ = self._company([0.0, 1888.0, 0.0, 1859.0, 2233.0], [gap])
        self.assertEqual(cd.financials.capex, [0.0, 1888.0, 2155.0, 1859.0, 2233.0])

    def test_yahoo_capex_on_another_definition_is_not_used(self) -> None:
        # Oportun-like: EDGAR's FY2022 capex is 5,995 and Yahoo's 48,892 (8.2x),
        # so Yahoo's FY2023-2025 figures would not be comparable with EDGAR's.
        gap = "capex not reported on EDGAR for FY2023-2025; filled with 0.0"
        cf = yahoo_cashflow(**{"Capital Expenditure": [-24330.0, -19187.0, -31261.0,
                                                       -48892.0]})
        cd, fin = self._company([12296.0, 5995.0, 0.0, 0.0, 0.0], [gap], cf=cf)
        self.assertEqual(cd.financials.capex, fin.capex)
        self.assertIn(gap + "; not backfilled from Yahoo's annual cash-flow statement: its "
                      "capital expenditure for FY2022, which EDGAR also reports, is 8.2x "
                      "EDGAR's, so the two likely define capex differently; FY2023-2025 left "
                      "at 0.0", cd.source_notes)
        # Narrower than EDGAR's (a median 0.50x over two years): not used either.
        cf = yahoo_cashflow(**{"Capital Expenditure": [-2233.0, -1859.0, -1000.0, -900.0]})
        cd, _ = self._company([2000.0, 1800.0, 0.0, 0.0, 0.0], [gap], cf=cf)
        self.assertEqual(cd.financials.capex, [2000.0, 1800.0, 0.0, 0.0, 0.0])
        self.assertTrue(any("capital expenditure for FY2022, which EDGAR also reports, is "
                            "0.50x EDGAR's" in n for n in cd.source_notes), cd.source_notes)

    def test_yahoo_capex_matching_edgar_where_both_report_is_used(self) -> None:
        # United Rentals-like: Yahoo/EDGAR 1.00, 1.05 and 1.00 for FY2022-2024
        # (a median, so one wider year does not block the fill), FY2025 missing.
        gap = "capex not reported on EDGAR for FY2025; filled with 0.0"
        cf = yahoo_cashflow(**{"Capital Expenditure": [-4528.0, -4130.0, -4057.2, -3690.0]})
        cd, _ = self._company([3198.0, 3690.0, 3864.0, 4130.0, 0.0], [gap], cf=cf)
        self.assertEqual(cd.financials.capex, [3198.0, 3690.0, 3864.0, 4130.0, 4528.0])
        self.assertIn(gap + "; backfilled from Yahoo's annual cash-flow statement (capital "
                      "expenditure for the same fiscal year): FY2025 4,528 (year to "
                      "2025-12-31)", cd.source_notes)
        # One year at 1.9x is outvoted by two that match.
        cf = yahoo_cashflow(**{"Capital Expenditure": [-4528.0, -4130.0, -7341.6, -3690.0]})
        cd, _ = self._company([3198.0, 3690.0, 3864.0, 4130.0, 0.0], [gap], cf=cf)
        self.assertEqual(cd.financials.capex[-1], 4528.0)

    def test_banks_insurers_bdcs_and_reits_are_not_backfilled(self) -> None:
        # Their DCF and FCFE are reference-only: no Yahoo request, EDGAR's gap stays.
        for kind, label in (("bank", "a bank"), ("insurer", "an insurer"),
                            ("bdc", "a business development company"), ("reit", "a REIT")):
            with self.subTest(kind=kind), \
                    mock.patch.object(YFinanceClient, "get_capex_by_year") as get_capex:
                cd, _ = self._company([0.0] * 5, [self.ALL_GAP], kind=kind)
                self.assertEqual(cd.financials.capex, [0.0] * 5)
                self.assertIn(self.ALL_GAP + "; not backfilled from Yahoo: capex enters only "
                              f"the FCFF DCF and FCFE, which for {label} are shown for "
                              "reference only", cd.source_notes)
                get_capex.assert_not_called()

    def test_lessor_capex_far_above_edgar_da_is_not_backfilled(self) -> None:
        # Avis Budget-like: vehicle purchases (1,859-2,233) against an EDGAR D&A
        # of 3 that leaves out the fleet's depreciation.
        cd, _ = self._company([0.0] * 5, [self.ALL_GAP], kind="lessor")
        self.assertEqual(cd.financials.capex, [0.0] * 5)
        self.assertIn(self.ALL_GAP + "; not backfilled from Yahoo's annual cash-flow statement: "
                      "its capital expenditure for FY2022-2025 is a median 673.8x EDGAR's D&A "
                      "for the same years, so it likely includes purchases of the lease fleet, "
                      "whose depreciation EDGAR's D&A leaves out; FY2021-2025 left at 0.0",
                      cd.source_notes)
        # AerCap-like: aircraft purchases about 2.4x a D&A that includes the fleet.
        cd, _ = self._company([0.0] * 5, [self.ALL_GAP], kind="lessor",
                              da=[700.0, 790.0, 900.0, 800.0, 950.0])
        self.assertEqual(cd.financials.capex, [0.0, 1888.0, 2155.0, 1859.0, 2233.0])

    def test_reported_or_unexplained_capex_is_never_replaced(self) -> None:
        with mock.patch.object(YFinanceClient, "get_capex_by_year") as get_capex:
            # A non-zero EDGAR figure stays, whatever the gap note says.
            cd, _ = self._company([4.0] * 5, [self.ALL_GAP])
            self.assertEqual(cd.financials.capex, [4.0] * 5)
            self.assertIn(self.ALL_GAP, cd.source_notes)
            # A 0 without EDGAR's gap note is a reading (e.g. a finance arm's lease
            # fleet sold down), not a tag gap.
            cd, _ = self._company([4.0, 4.0, 4.0, 4.0, 0.0], [])
            self.assertEqual(cd.financials.capex, [4.0, 4.0, 4.0, 4.0, 0.0])
            get_capex.assert_not_called()

    def test_no_yahoo_capex_either(self) -> None:
        # The statement was read but has no capex line (a bank-style layout).
        cf = yahoo_cashflow(**{"Operating Cash Flow": [10.0, 9.0, 8.0, 7.0]})
        cd, _ = self._company([0.0] * 5, [self.ALL_GAP], cf=cf)
        self.assertEqual(cd.financials.capex, [0.0] * 5)
        self.assertIn(self.ALL_GAP + "; Yahoo's cash-flow statement has no capital expenditure "
                      "for FY2021-2025 either, so nothing was backfilled", cd.source_notes)

    def test_a_failed_yahoo_request_is_not_reported_as_no_capex(self) -> None:
        # yfinance returns an empty frame when rate-limited, or the request raises.
        failed = ticker(self.INFO)
        type(failed).cashflow = mock.PropertyMock(side_effect=RuntimeError("HTTP 429"))
        for cf, extra in ((pd.DataFrame(), None), (None, {"PSX": failed})):
            with self.subTest(cf=cf):
                cd, _ = self._company([0.0] * 5, [self.ALL_GAP], cf=cf, extra=extra)
                self.assertEqual(cd.financials.capex, [0.0] * 5)
                self.assertIn(self.ALL_GAP + "; Yahoo's annual cash-flow statement could not be "
                              "read (the request failed or returned nothing), so nothing was "
                              "backfilled", cd.source_notes)
                self.assertFalse(any("has no capital expenditure" in n for n in cd.source_notes))

    def test_yahoo_capex_in_another_currency_is_converted_or_skipped(self) -> None:
        info = dict(self.INFO, financialCurrency="EUR")
        cd, _ = self._company([0.0] * 5, [self.ALL_GAP], info=info,
                              extra={"EURUSD=X": fx_ticker(1.1)})
        self.assertEqual(cd.financials.capex,
                         [0.0] + [v * 1.1 for v in (1888.0, 2155.0, 1859.0, 2233.0)])
        self.assertTrue(any("FY2025 2,456 (year to 2025-12-31) (Yahoo's EUR figures at "
                            "EUR->USD spot 1.1)" in n for n in cd.source_notes), cd.source_notes)
        cd, _ = self._company([0.0] * 5, [self.ALL_GAP], info=info)
        self.assertEqual(cd.financials.capex, [0.0] * 5)
        self.assertIn(self.ALL_GAP + "; Yahoo's cash-flow statement is in EUR and no EUR->USD "
                      "rate could be fetched, so nothing was backfilled", cd.source_notes)

    def test_52_53_week_year_ending_in_january_matches_the_prior_fiscal_year(self) -> None:
        # A retailer's FY2024 ends on 2025-01-03 (the EDGAR parser's label rule).
        cf = yahoo_cashflow(("2025-01-03", "2023-12-29", "2022-12-30"),
                            **{"Capital Expenditure": [-30.0, -20.0, -10.0]})
        cd, _ = self._company([0.0] * 3, [self.ALL_GAP], cf=cf, years=[2022, 2023, 2024])
        self.assertEqual(cd.financials.capex, [10.0, 20.0, 30.0])
        self.assertTrue(any("FY2024 30 (year to 2025-01-03)" in n for n in cd.source_notes))

    def test_capex_by_year_reads_the_first_row_with_a_value_per_column(self) -> None:
        cf = yahoo_cashflow(("2025-12-31", "2024-12-31", "2024-06-30", "2023-06-30"),
                            **{"Capital Expenditure": [NaN, -5.0, -7.0, 0.0],
                               "Purchase Of PPE": [-6.0, -4.0, -7.0, 0.0]})
        items = client({"X": ticker(self.INFO, cf=cf)}).get_capex_by_year("X")
        # FY2025 from Purchase Of PPE; FY2024 is shared by two columns (a changed
        # year end) and left out; FY2023's 0 is not a reading.
        self.assertEqual(items, {"currency": "USD", "currency_note": None,
                                 "by_year": {2025: ("2025-12-31", 6.0)}})
        # Read, but no capital expenditure in it: an empty by_year.
        cf = yahoo_cashflow(**{"Capital Expenditure": [NaN, 0.0, NaN, NaN]})
        self.assertEqual(client({"X": ticker(self.INFO, cf=cf)}).get_capex_by_year("X"),
                         {"currency": None, "currency_note": None, "by_year": {}})
        # Not read (an empty frame, or the request raised): None.
        self.assertIsNone(client({"X": ticker(self.INFO)}).get_capex_by_year("X"))
        failed = ticker(self.INFO)
        type(failed).cashflow = mock.PropertyMock(side_effect=RuntimeError("HTTP 429"))
        self.assertIsNone(client({"X": failed}).get_capex_by_year("X"))


# --------------------------------------------------------------------------- #
#  The EDGAR notes the backfills parse, as the real EDGAR client words them
# --------------------------------------------------------------------------- #
FIX_YEARS = list(range(2019, 2025))


def _edgar_flow(year: int, val: float) -> dict:
    """A companyfacts annual (10-K) flow fact for calendar `year`."""
    return {"start": f"{year}-01-01", "end": f"{year}-12-31", "val": val,
            "filed": f"{year + 1}-02-15", "fp": "FY", "form": "10-K", "fy": year}


def _edgar_instant(end: str, val: float) -> dict:
    """A companyfacts balance-sheet fact at `end` from a 10-K."""
    return {"end": end, "val": val, "filed": f"{end[:4]}-12-31", "form": "10-K", "fp": "FY"}


def edgar_client(**usd) -> EdgarClient:
    """A real EdgarClient serving one companyfacts fixture: FY2019-2024 flows and
    a 2024-12-31 balance sheet. `usd` replaces tags; None removes one."""
    tags = {
        "Revenues": [_edgar_flow(y, 1000.0 + 100 * i) for i, y in enumerate(FIX_YEARS)],
        "NetIncomeLoss": [_edgar_flow(y, 100.0) for y in FIX_YEARS],
        "OperatingIncomeLoss": [_edgar_flow(y, 150.0) for y in FIX_YEARS],
        "DepreciationDepletionAndAmortization": [_edgar_flow(y, 50.0) for y in FIX_YEARS],
        "PaymentsToAcquirePropertyPlantAndEquipment": [_edgar_flow(y, 60.0) for y in FIX_YEARS],
        "InterestExpense": [_edgar_flow(y, 10.0) for y in FIX_YEARS],
        "IncomeTaxExpenseBenefit": [_edgar_flow(y, 30.0) for y in FIX_YEARS],
        "StockholdersEquity": [_edgar_instant("2024-12-31", 1000.0)],
        "CashAndCashEquivalentsAtCarryingValue": [_edgar_instant("2024-12-31", 200.0)],
        "LongTermDebtNoncurrent": [_edgar_instant("2024-12-31", 480.0)],
    }
    tags.update(usd)
    gaap = {tag: {"units": {"USD": facts}} for tag, facts in tags.items() if facts is not None}
    gaap["WeightedAverageNumberOfDilutedSharesOutstanding"] = {
        "units": {"shares": [_edgar_flow(y, 400.0) for y in FIX_YEARS]}}
    edgar = EdgarClient(user_agent="equity-research-tests test@example.com")
    edgar._ticker_map = {"FIX": ("0000000001", "Fixture Co")}  # no directory fetch
    edgar.company_facts = mock.Mock(return_value={"facts": {"us-gaap": gaap}})  # no network
    return edgar


class EdgarNoteContractTests(unittest.TestCase):
    """The provider's capex, debt and cash backfills act on the EDGAR client's
    gap notes; here the real client writes them, so a change of wording on
    either side fails a test instead of silently stopping a backfill."""

    INFO = {"currency": "USD", "financialCurrency": "USD", "currentPrice": 20.0,
            "sharesOutstanding": 400.0, "marketCap": 8000.0}
    # Yahoo's capex FY2024 back to FY2021, within 7% of the fixture's 60 a year.
    CF = {"Capital Expenditure": [-64.0, -62.0, -61.0, -59.0]}
    CF_DATES = ("2024-12-31", "2023-12-31", "2022-12-31", "2021-12-31")
    BACKFILLED = ("; backfilled from Yahoo's annual cash-flow statement (capital expenditure "
                  "for the same fiscal year): ")

    def setUp(self) -> None:
        patcher = mock.patch.object(provider_mod, "_today", return_value=TODAY)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _company(self, edgar: EdgarClient, qbs=None):
        tk = ticker(self.INFO, cf=yahoo_cashflow(self.CF_DATES, **self.CF), qbs=qbs)
        return HybridProvider(edgar=edgar, market=client({"FIX": tk})).get_company_data("FIX")

    def test_capex_without_any_tag_is_backfilled(self) -> None:
        cd = self._company(edgar_client(PaymentsToAcquirePropertyPlantAndEquipment=None))
        self.assertEqual(cd.financials.capex, [0.0, 0.0, 59.0, 61.0, 62.0, 64.0])
        self.assertIn(
            "capex unavailable on EDGAR; filled with 0.0" + self.BACKFILLED + "FY2021 59 (year "
            "to 2021-12-31), FY2022 61 (year to 2022-12-31), FY2023 62 (year to 2023-12-31), "
            "FY2024 64 (year to 2024-12-31); FY2019-2020 not in Yahoo's statement, left at 0.0",
            cd.source_notes)

    def test_capex_gap_years_are_backfilled(self) -> None:
        flows = [_edgar_flow(y, 60.0) for y in FIX_YEARS if y not in (2020, 2022)]
        cd = self._company(edgar_client(PaymentsToAcquirePropertyPlantAndEquipment=flows))
        # FY2022 from Yahoo (FY2021, 2023, 2024 match EDGAR's 60); FY2020 predates it.
        self.assertEqual(cd.financials.capex, [60.0, 0.0, 60.0, 61.0, 60.0, 60.0])
        self.assertIn("capex not reported on EDGAR for FY2020, FY2022; filled with 0.0"
                      + self.BACKFILLED + "FY2022 61 (year to 2022-12-31); FY2020 not in "
                      "Yahoo's statement, left at 0.0", cd.source_notes)

    def test_zero_debt_and_cash_notes_are_superseded_by_the_yahoo_backfill(self) -> None:
        # Debt and cash tags last used in 2019 (PACCAR-like); interest still paid.
        edgar = edgar_client(
            LongTermDebtNoncurrent=[_edgar_instant("2019-12-31", 300.0)],
            CashAndCashEquivalentsAtCarryingValue=[_edgar_instant("2019-12-31", 200.0)])
        qbs = pd.DataFrame(
            {"Total Assets": [3000.0, 2950.0], "Long Term Debt": [450.0, 440.0],
             "Current Debt": [30.0, 35.0], "Cash And Cash Equivalents": [150.0, 140.0],
             "Cash Cash Equivalents And Short Term Investments": [210.0, 200.0]},
            index=pd.to_datetime(["2024-12-31", "2024-09-30"])).T
        cd = self._company(edgar, qbs=qbs)
        self.assertEqual((cd.balance_sheet.total_debt, cd.balance_sheet.cash_and_investments),
                         (480.0, 210.0))
        self.assertIn(
            "WARNING: FY2024 interest expense is 10 but no debt was found under the SEC debt "
            "tags this parser reads (the filer may tag its debt with company-specific "
            "elements); net debt and the WACC debt weight may be understated; backfilled from "
            "Yahoo's balance sheet at 2024-12-31: total debt 480 (long-term plus current debt, "
            "lease obligations excluded); cash and short-term investments 210 (EDGAR's are 0: "
            "its cash tags are stale or company-specific, like its debt tags)", cd.source_notes)
        superseded = [n for n in cd.source_notes if n.endswith(
            "; replaced by the Yahoo balance-sheet backfill (see the interest-expense WARNING)")]
        for note in ("total debt unavailable on EDGAR; set to 0.0",
                     "cash & equivalents unavailable on EDGAR; set to 0.0",
                     "cash & equivalents last reported 2019-12-31 (200); treated as 0 at the "
                     "2024-12-31 balance sheet"):
            self.assertTrue(any(n.startswith(note) for n in superseded), (note, cd.source_notes))
        self.assertTrue(any(n.startswith("long-term debt") and "last reported 2019-12-31" in n
                            for n in superseded), cd.source_notes)


class SuggestPeersTests(unittest.TestCase):
    def test_no_network_call_and_no_peers(self) -> None:
        c = client({})
        self.assertEqual(c.suggest_peers("AAPL"), [])
        c._ticker.assert_not_called()


class YahooBoundaryTests(unittest.TestCase):
    def test_common_income_precedes_parent_income(self):
        fin = income(**{"Net Income Common Stockholders": [120, 110, 100, 90, NaN]})
        c = client({"T": ticker(fin=fin, cf=cashflow(), bs=balance())})
        normalized, _bs = c.get_annual_financials_fallback("T")
        self.assertEqual(normalized.net_income, [90, 100, 110, 120])
        self.assertTrue(normalized._income_attribution_adjusted)

    def test_missing_common_income_subtracts_reported_preferred_adjustments(self):
        fin = income(**{"Preferred Stock Dividends": [5] * 5,
                        "Otherunder Preferred Stock Dividend": [2] * 5})
        c = client({"T": ticker(fin=fin, cf=cashflow(), bs=balance())})
        normalized, _bs = c.get_annual_financials_fallback("T")
        self.assertEqual(normalized.net_income[-1], 161)
        self.assertTrue(normalized._income_attribution_adjusted)

    def test_edgar_conversion_failure_stops_valuation(self):
        company = make_company()
        company.market.currency = "GBP"
        provider = HybridProvider(edgar=mock.Mock(), market=mock.Mock())
        provider.market.get_fx_rate.return_value = None
        with self.assertRaisesRegex(DataError, "no USD->GBP exchange rate.*retry"):
            provider._edgar_to_quote_currency(
                company.financials, company.balance_sheet, company.market, [])

    def test_numeric_coercion_rejects_boolean_and_overflow(self):
        self.assertIsNone(market_mod._num(True))
        self.assertIsNone(market_mod._num(10 ** 1000))

    def test_common_equity_is_used_without_subtracting_preferred_twice(self):
        frame = balance(**{"Preferred Stock": [100] * 5})
        bs = YFinanceClient()._build_balance_sheet(frame)
        self.assertEqual(bs.total_equity, 800.0)
        self.assertEqual(bs.preferred_equity, 100.0)

    def test_parent_equity_excludes_preferred_in_fundamentals_and_comps(self):
        frame = balance(drop=("Common Stock Equity",), **{"Preferred Stock": [100] * 5})
        c = YFinanceClient()
        self.assertEqual(c._build_balance_sheet(frame).total_equity, 700.0)
        self.assertEqual(c._comp_balance_items(ticker(bs=frame))["equity"], 700.0)

    def test_preferred_outside_equity_is_not_subtracted_from_parent(self):
        frame = balance(drop=("Common Stock Equity",),
                        **{"Preferred Securities Outside Stock Equity": [100] * 5})
        bs = YFinanceClient()._build_balance_sheet(frame)
        self.assertEqual(bs.total_equity, 800.0)
        self.assertEqual(bs.preferred_equity, 100.0)

    def test_comp_rows_report_major_quote_currency(self):
        tk = ticker(info={"currency": "GBp", "financialCurrency": "GBP", "marketCap": 10000,
                          "currentPrice": 100, "sharesOutstanding": 100, "trailingPE": 12})
        row = client({"T.L": tk}).get_comp_row("T.L")
        self.assertEqual(row.currency, "GBP")
        self.assertEqual(row.market_cap, 100.0)

class FmpBoundaryTests(unittest.TestCase):
    def test_first_rejects_non_mapping_rows(self):
        self.assertIsNone(FMPClient._first(["upstream service unavailable"]))
        self.assertIsNone(FMPClient._first([None]))

    def test_enrichment_lists_drop_non_records(self):
        client = FMPClient(api_key="test-token")
        payload = [None, "error", {"title": "A news item"}]
        with mock.patch.object(client, "_get", return_value=payload):
            self.assertEqual(client.news("T"), [{"title": "A news item"}])
            self.assertEqual(client.analyst_estimates("T"), [{"title": "A news item"}])
            self.assertEqual(client.peers("T"), [])


if __name__ == "__main__":
    unittest.main()
