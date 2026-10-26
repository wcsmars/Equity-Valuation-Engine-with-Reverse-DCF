"""Data-boundary regressions from the October 2026 audit; no live providers."""

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from equity_valuation import config
from equity_valuation.data.base import DataError
from equity_valuation.data.edgar import EdgarClient
from equity_valuation.data.market import YFinanceClient
from tests.test_market_data import balance, cashflow, client, income, ticker


class FallbackStatementAuditTests(unittest.TestCase):
    def test_common_cash_dividends_precede_total_distribution(self):
        flows = cashflow()
        flows.loc["Common Stock Dividend Paid"] = [-30, -28, -26, -24, np.nan]
        provider = client({"T": ticker(fin=income(), cf=flows, bs=balance())})
        financials, _ = provider.get_annual_financials_fallback("T")
        self.assertEqual(financials.dividends_paid, [24, 26, 28, 30])

    def test_invalid_and_undated_income_columns_are_not_fiscal_years(self):
        frame = income()
        for bad in ("TTM", "2025-99-99", pd.NaT):
            frame[bad] = frame.iloc[:, 0] * 2
        provider = client({"T": ticker(fin=frame, cf=cashflow(), bs=balance())})
        financials, _ = provider.get_annual_financials_fallback("T")
        self.assertEqual(financials.fiscal_years, [2021, 2022, 2023, 2024])
        self.assertEqual(financials.revenue[-1], 1000)
        self.assertIn("ignored undated", " ".join(financials._source_notes))

    def test_fiscal_year_change_does_not_create_two_annual_growth_observations(self):
        frame = income()
        frame[pd.Timestamp("2024-06-30")] = frame.iloc[:, 0] / 2
        provider = client({"T": ticker(fin=frame, cf=cashflow(), bs=balance())})
        financials, _ = provider.get_annual_financials_fallback("T")
        self.assertEqual(financials.fiscal_years, [2021, 2022, 2023])
        self.assertIn("multiple periods sharing FY2024", " ".join(financials._source_notes))

    def test_empty_newer_balance_column_does_not_erase_real_debt(self):
        frame = balance()
        frame[pd.Timestamp("2025-12-31")] = np.nan
        notes = []
        result = YFinanceClient()._build_balance_sheet(frame, notes)
        self.assertEqual(result.as_of, "2024-12-31")
        self.assertEqual(result.total_debt, 300)
        self.assertEqual(result.cash_and_investments, 150)
        self.assertIn("empty newer balance-sheet", " ".join(notes))

    def test_real_zero_balance_column_remains_current(self):
        frame = balance()
        frame[pd.Timestamp("2025-12-31")] = 0
        result = YFinanceClient()._build_balance_sheet(frame)
        self.assertEqual(result.as_of, "2025-12-31")
        self.assertEqual(result.total_debt, 0)

    def test_empty_newer_balance_column_preserves_captive_finance_classification(self):
        frame = balance(**{"Total Assets": [2000] * 5,
                           "Total Debt": [800] * 5,
                           "Non Current Accounts Receivable": [600] * 5})
        frame[pd.Timestamp("2025-12-31")] = np.nan
        provider = client({"T": ticker(info={"sector": "Industrials"},
                                       fin=income(), cf=cashflow(), bs=frame)})
        financials, snapshot = provider.get_annual_financials_fallback("T")
        self.assertEqual(snapshot.as_of, "2024-12-31")
        self.assertEqual(financials._financial_kind, "captive_finance")

    def test_fiscal_year_requires_a_real_calendar_date(self):
        for invalid in ("2024-02-31", "2024-01-00", "2024-TTM", pd.NaT, None):
            with self.subTest(invalid=invalid):
                self.assertIsNone(YFinanceClient._year_of(invalid))
        self.assertEqual(YFinanceClient._year_of("2024-01-13"), 2023)


class EdgarCurrencyAndPrivacyAuditTests(unittest.TestCase):
    @staticmethod
    def facts(extra_usd):
        return {"facts": {"us-gaap": {
            "Assets": {"units": {
                "EUR": [{"end": "2024-12-31", "filed": "2025-02-01", "val": 100}] * 5,
                "USD": [{"end": "2024-12-31", "filed": "2025-02-01", "val": 110}]
                       + extra_usd,
            }},
        }}}

    def test_malformed_date_cannot_disable_currency_majority_guard(self):
        facts = self.facts([{"end": "9999-99-99", "filed": "9999-99-99", "val": 1}])
        result = EdgarClient()._foreign_reporting_currency(facts)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "EUR")

    def test_non_numeric_facts_cannot_outvote_reported_currency(self):
        invalid = [{"end": "2024-12-31", "filed": "2025-02-01", "val": val}
                   for val in (None, True, float("nan"), float("inf"), "bad")]
        result = EdgarClient()._foreign_reporting_currency(self.facts(invalid * 5))
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "EUR")

    def test_sec_failure_does_not_echo_contact_into_exportable_notes(self):
        contact = "private-contact@example.com"
        provider = EdgarClient(user_agent=f"test app {contact}")
        with mock.patch.object(provider._session, "get", return_value=mock.Mock(status_code=403)), \
                mock.patch.object(config, "SEC_MAX_RETRIES", 1):
            with self.assertRaises(DataError) as caught:
                provider._get_json("https://data.sec.gov/test")
        self.assertNotIn(contact, str(caught.exception))
        self.assertIn("User-Agent", str(caught.exception))


class FmpStableContractTests(unittest.TestCase):
    def setUp(self):
        from backend.fmp_client import FMPClient
        self.client = FMPClient(api_key="test-token")

    def test_all_requests_use_documented_stable_paths_and_query_parameters(self):
        cases = [
            ("profile", ("aapl",), "profile", {"symbol": "AAPL"}),
            ("quote", ("aapl",), "quote", {"symbol": "AAPL"}),
            ("ratios_ttm", ("aapl",), "ratios-ttm", {"symbol": "AAPL"}),
            ("key_metrics_ttm", ("aapl",), "key-metrics-ttm", {"symbol": "AAPL"}),
            ("price_target", ("aapl",), "price-target-consensus", {"symbol": "AAPL"}),
            ("rating", ("aapl",), "ratings-snapshot", {"symbol": "AAPL"}),
            ("peers", ("aapl",), "stock-peers", {"symbol": "AAPL"}),
            ("news", ("aapl", 5), "news/stock", {"symbols": "AAPL", "limit": 5}),
            ("analyst_estimates", ("aapl", 6), "analyst-estimates",
             {"symbol": "AAPL", "limit": 6, "period": "annual", "page": 0}),
            ("transcripts_list", ("aapl",), "earning-call-transcript-dates", {"symbol": "AAPL"}),
            ("transcript", ("aapl", 2025, 3), "earning-call-transcript",
             {"symbol": "AAPL", "year": 2025, "quarter": 3}),
        ]
        for method, args, path, params in cases:
            with self.subTest(method=method), mock.patch("backend.fmp_client.requests.get") as get:
                get.return_value.status_code = 200
                get.return_value.json.return_value = []
                getattr(self.client, method)(*args)
                get.assert_called_once_with(
                    f"https://financialmodelingprep.com/stable/{path}",
                    params={**params, "apikey": "test-token"}, timeout=12.0,
                )

    def test_stable_peer_rows_are_deduplicated_and_exclude_target(self):
        payload = [{"symbol": "MSFT"}, {"symbol": " msft "}, {"symbol": "AAPL"},
                   {"symbol": None}, {"symbol": 10}, "error", {"symbol": "GOOGL"}]
        with mock.patch.object(self.client, "_get", return_value=payload):
            self.assertEqual(self.client.peers("AAPL"), ["MSFT", "GOOGL"])

    def test_stable_ratios_keep_dashboard_keys_and_missing_is_not_zero(self):
        payload = {"priceToEarningsRatioTTM": 20, "priceToFreeCashFlowRatioTTM": 18,
                   "dividendPayoutRatioTTM": 0, "debtToEquityRatioTTM": 0.4}
        with mock.patch.object(self.client, "_get", return_value=[payload]):
            result = self.client.ratios_ttm("T")
        self.assertEqual(result["peRatioTTM"], 20)
        self.assertEqual(result["priceToFreeCashFlowsRatioTTM"], 18)
        self.assertEqual(result["payoutRatioTTM"], 0)
        self.assertEqual(result["debtEquityRatioTTM"], 0.4)
        self.assertNotIn("returnOnEquityTTM", result)
        self.assertNotIn("peRatioTTM", payload)

    def test_estimate_aliases_preserve_reported_currency_without_guessing_it(self):
        payload = [{"revenueAvg": 100, "epsAvg": 2, "reportedCurrency": "TWD"},
                   {"revenueAvg": 200, "epsAvg": 3}]
        with mock.patch.object(self.client, "_get", return_value=payload):
            result = self.client.analyst_estimates("T")
        self.assertEqual(result[0]["estimatedRevenueAvg"], 100)
        self.assertEqual(result[0]["estimatedEpsAvg"], 2)
        self.assertEqual(result[0]["reportedCurrency"], "TWD")
        self.assertNotIn("reportedCurrency", result[1])

    def test_stable_transcript_fiscal_year_maps_to_existing_contract(self):
        with mock.patch.object(self.client, "_get", return_value=[
                {"quarter": 2, "fiscalYear": 2025, "date": "2025-04-30"}]):
            result = self.client.transcripts_list("T")
        self.assertEqual(result, [{"quarter": 2, "year": 2025, "date": "2025-04-30"}])

    def test_roe_relocated_to_key_metrics_remains_available_to_dashboard(self):
        with mock.patch.object(self.client, "_get", side_effect=lambda path, params: (
                [{"returnOnEquityTTM": 0.15, "returnOnInvestedCapitalTTM": 0.1}]
                if path == "/stable/key-metrics-ttm" else [])):
            result = self.client.enrichment("T")
        self.assertEqual(result["ratios_ttm"]["returnOnEquityTTM"], 0.15)
        self.assertEqual(result["key_metrics_ttm"]["roicTTM"], 0.1)

    def test_errors_and_key_strings_are_not_returned_as_data(self):
        with mock.patch("backend.fmp_client.requests.get") as get:
            get.return_value.status_code = 200
            get.return_value.json.return_value = {"Error Message": "invalid test-token"}
            self.assertIsNone(self.client.profile("T"))
            get.side_effect = RuntimeError("private request contents")
            self.assertIsNone(self.client.profile("T"))

    def test_news_text_fields_reject_non_string_provider_values(self):
        payload = [{"title": {}, "text": 123, "url": [], "publisher": "Publisher",
                    "image": False, "publishedDate": None}]
        with mock.patch.object(self.client, "_get", return_value=payload):
            result = self.client.news("T")[0]
        self.assertEqual(result["site"], "Publisher")
        for key in ("title", "text", "url", "image", "publishedDate"):
            self.assertIsNone(result[key])
        self.assertEqual(payload[0]["text"], 123)


class ForeignFilingSectionsTests(unittest.TestCase):
    @staticmethod
    def document():
        return (
            "Item 3. Key Information 10\nItem 4. Information on the Company 20\n"
            "Item 5. Operating and Financial Review and Prospects 30\n"
            + "Foreword. " * 100 + "\n"
        ) + (
            "Item 3. Key Information\n" + "Risk discussion. " * 80
            + "\nItem 4. Information on the Company\n" + "Business operations. " * 80
            + "\nItem 4A. Unresolved Staff Comments\nComments excluded.\n"
            + "Item 5. Operating and Financial Review and Prospects\n"
            + "Operating cash flows. " * 80
            + "\nItem 6. Directors, Senior Management and Employees\nDirectors excluded."
        )

    def test_20f_uses_foreign_annual_report_item_numbers_and_boundaries(self):
        from backend import filings
        for form in ("20-F", "20-F/A"):
            with self.subTest(form=form):
                result = filings.extract_sections(self.document(), form)
                self.assertEqual(set(result), {"risk_factors", "business", "mdna"})
                self.assertIn("Risk discussion.", result["risk_factors"])
                self.assertNotIn("Business operations.", result["risk_factors"])
                self.assertNotIn("Comments excluded.", result["business"])
                self.assertIn("Operating cash flows.", result["mdna"])
                self.assertNotIn("Directors excluded.", result["mdna"])

    def test_20f_material_heading_and_total_budget_match_actual_content(self):
        from backend import filings
        with mock.patch.object(filings, "fetch_filing_text", return_value=self.document()):
            text, meta = filings.build_filing_material("T", "20-F", "2025-03-01", "example", "report.htm")
        self.assertIn("ITEM 3 - KEY INFORMATION AND RISK FACTORS", text)
        self.assertIn("ITEM 5 - OPERATING AND FINANCIAL REVIEW", text)
        self.assertNotIn("ITEM 1A", text)
        self.assertEqual(meta["material_chars"], len(text))
        self.assertLessEqual(len(text), filings._TOTAL_CAP)


if __name__ == "__main__":
    unittest.main()
