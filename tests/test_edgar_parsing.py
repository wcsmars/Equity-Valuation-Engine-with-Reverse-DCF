"""Offline tests for the EDGAR companyfacts normalizer.

A hand-built ``companyfacts`` payload exercises the rules that matter with
real filings: restated comparatives win, quarterly and non-annual forms are
ignored, 52/53-week years are accepted, and a company that switched revenue
tags still gets one complete series with the preferred tag winning on overlap.
Further fixtures cover the full `get_annual_financials` path: EBIT fallbacks
and the filers they must skip (banks, BDCs, lender-scale interest), the
financial-filer flags (banks, insurers, BDCs, REITs, including those known
only by their Schedule III real estate, debt-funded lessors, captive finance
arms and the rental companies, lenders and small finance lines that are
neither), D&A and capex as the largest of their tags unless that figure
predates a restatement (for capex, only a tag larger in every year, not a
line that adds acquisitions or a separate line), lease-fleet capex (not added
again to a total that holds it), other-PP&E capex and the
other-productive-assets line (gap-fills only, the latter named in a note),
pretax income from its domestic and foreign parts,
statements reported in another currency, signed tax,
working capital, stale balance-sheet facts, the cash and debt tag families,
the interest-versus-debt warning, preferred stock, stale revenue axes,
amendments, the hand-off of parsing notes to HybridProvider, and the SEC
filings list shown in the app.
No network access is needed. Run with:  python -m unittest tests.test_edgar_parsing
"""

from __future__ import annotations

import unittest
from unittest import mock

from backend import filings

from equity_valuation.data.base import DataError
from equity_valuation.data.edgar import _TAGS_REVENUE, EdgarClient
from equity_valuation.data.provider import HybridProvider, _missing_debt_warning
from equity_valuation.schemas import MarketData
from equity_valuation.utils import financial_institution_detail

PREFERRED = "RevenueFromContractWithCustomerExcludingAssessedTax"
FALLBACK = "Revenues"
_TOTAL_CASH = "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"
_TOTAL_CASH_DISPOSAL = _TOTAL_CASH + "IncludingDisposalGroupAndDiscontinuedOperations"


def _entry(start: str, end: str, val: float, filed: str,
           fp: str = "FY", form: str = "10-K") -> dict:
    return {"start": start, "end": end, "val": val, "filed": filed,
            "fp": fp, "form": form, "fy": int(end[:4])}


def _facts(**tags: list[dict]) -> dict:
    return {"facts": {"us-gaap": {
        tag: {"units": {"USD": entries}} for tag, entries in tags.items()
    }}}


class AnnualFlowNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = EdgarClient(user_agent="equity-research-tests test@example.com")
        self.facts = _facts(**{
            PREFERRED: [
                _entry("2023-01-01", "2023-12-31", 300.0, "2024-02-01"),
                _entry("2022-01-01", "2022-12-31", 200.0, "2023-02-01"),
            ],
            FALLBACK: [
                # Original FY2021 report, then the restated comparative filed a
                # year later inside the FY2022 10-K.
                _entry("2021-01-01", "2021-12-31", 100.0, "2022-02-01"),
                _entry("2021-01-01", "2021-12-31", 111.0, "2023-02-01"),
                # Same period as the preferred tag; must lose to it.
                _entry("2023-01-01", "2023-12-31", 999.0, "2024-02-01"),
                # A 53-week fiscal year (371 days) is still an annual period.
                _entry("2019-12-26", "2020-12-31", 90.0, "2021-02-01"),
                # A one-quarter span labelled FY is not an annual flow.
                _entry("2019-10-01", "2019-12-31", 12.0, "2020-02-01"),
                # Quarterly and interim forms never feed the annual series.
                _entry("2018-01-01", "2018-12-31", 80.0, "2019-02-01", fp="Q4"),
                _entry("2017-01-01", "2017-12-31", 70.0, "2018-02-01", form="10-Q"),
            ],
        })
        self.series = self.client._annual_flow_by_fy(self.facts, (PREFERRED, FALLBACK))

    def test_latest_filing_wins_for_a_restated_period(self) -> None:
        self.assertEqual(self.series[2021], 111.0)

    def test_preferred_tag_wins_on_overlapping_periods(self) -> None:
        self.assertEqual(self.series[2023], 300.0)
        self.assertEqual(self.series[2022], 200.0)

    def test_fallback_tag_backfills_years_the_preferred_tag_lacks(self) -> None:
        self.assertEqual(self.series[2020], 90.0)

    def test_short_periods_and_non_annual_forms_are_dropped(self) -> None:
        self.assertNotIn(2019, self.series)
        self.assertNotIn(2018, self.series)
        self.assertNotIn(2017, self.series)

    def test_series_is_exactly_the_expected_years(self) -> None:
        self.assertEqual(self.series, {2020: 90.0, 2021: 111.0, 2022: 200.0, 2023: 300.0})

    def test_missing_tags_yield_an_empty_series(self) -> None:
        self.assertEqual(self.client._annual_flow_by_fy(self.facts, ("NoSuchTag",)), {})


# --------------------------------------------------------------------------- #
#  Full get_annual_financials fixtures
# --------------------------------------------------------------------------- #
_PRETAX = "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"
_YEARS = list(range(2019, 2025))
_SHARES = {"WeightedAverageNumberOfDilutedSharesOutstanding": [
    _entry(f"{y}-01-01", f"{y}-12-31", 400.0, f"{y + 1}-02-15") for y in _YEARS
]}


def _fy(year: int, val: float, filed: str = "", form: str = "10-K") -> dict:
    return _entry(f"{year}-01-01", f"{year}-12-31", val, filed or f"{year + 1}-02-15", form=form)


def _inst(end: str, val: float, filed: str = "", form: str = "10-K") -> dict:
    return {"end": end, "val": val, "filed": filed or f"{end[:4]}-12-31",
            "form": form, "fp": "FY"}


def _q2(end: str, val: float) -> dict:
    """An instant fact from a later 10-Q."""
    return {"end": end, "val": val, "filed": "2025-08-01", "form": "10-Q", "fp": "Q2"}


def _base_usd() -> dict:
    """Six clean fiscal years (2019-2024) plus a FY2024 balance sheet."""
    return {
        "Revenues": [_fy(y, 1000.0 + 100 * i) for i, y in enumerate(_YEARS)],
        "NetIncomeLoss": [_fy(y, 100.0 + 10 * i) for i, y in enumerate(_YEARS)],
        "OperatingIncomeLoss": [_fy(y, 150.0 + 10 * i) for i, y in enumerate(_YEARS)],
        "DepreciationDepletionAndAmortization": [_fy(y, 50.0) for y in _YEARS],
        "PaymentsToAcquirePropertyPlantAndEquipment": [_fy(y, 60.0) for y in _YEARS],
        "InterestExpense": [_fy(y, 10.0) for y in _YEARS],
        # A tax benefit (valuation-allowance release) in 2020.
        "IncomeTaxExpenseBenefit": [_fy(y, -60.0 if y == 2020 else 30.0) for y in _YEARS],
        _PRETAX: [_fy(y, 130.0 + 10 * i) for i, y in enumerate(_YEARS)],
        "PaymentsOfDividendsCommonStock": [_fy(y, 20.0) for y in _YEARS],
        "StockholdersEquity": [_inst("2024-12-31", 1000.0)],
        "CashAndCashEquivalentsAtCarryingValue": [_inst("2024-12-31", 200.0)],
        "LongTermDebtNoncurrent": [_inst("2024-12-31", 480.0)],
    }


def _company(usd: dict, shares: dict | None = None) -> dict:
    facts = _facts(**usd)
    for tag, entries in (shares if shares is not None else _SHARES).items():
        facts["facts"]["us-gaap"][tag] = {"units": {"shares": entries}}
    return facts


def _client(facts: dict) -> EdgarClient:
    client = EdgarClient(user_agent="equity-research-tests test@example.com")
    client._ticker_map = {"FIX": ("0000000001", "Fixture Co")}  # no directory fetch
    client.company_facts = mock.Mock(return_value=facts)        # no network
    return client


def _parse(usd: dict, shares: dict | None = None):
    fin, bs, _cik, _name = _client(_company(usd, shares)).get_annual_financials("FIX")
    return fin, bs, fin._source_notes


def _without(usd: dict, *tags: str) -> dict:
    return {k: v for k, v in usd.items() if k not in tags}


class FiscalYearLabelTests(unittest.TestCase):
    def test_52_53_week_years_ending_in_early_january_keep_their_own_label(self) -> None:
        # Saturday-nearest-Dec-31 calendar: three years end on Jan 1-3.
        ends = [("2017-12-31", "2018-12-30", 100.0), ("2018-12-31", "2019-12-29", 110.0),
                ("2019-12-30", "2021-01-03", 121.0), ("2021-01-04", "2022-01-02", 133.1),
                ("2022-01-03", "2023-01-01", 146.4), ("2023-01-02", "2023-12-31", 161.1)]
        facts = _facts(Revenues=[_entry(s, e, v, "2024-02-15") for s, e, v in ends])
        series = _client(facts)._annual_flow_by_fy(facts, ("Revenues",))
        self.assertEqual(series, {2018: 100.0, 2019: 110.0, 2020: 121.0,
                                  2021: 133.1, 2022: 146.4, 2023: 161.1})

    def test_late_january_year_end_keeps_the_calendar_year_label(self) -> None:
        facts = _facts(Revenues=[_entry("2023-02-01", "2024-01-31", 5.0, "2024-03-20")])
        self.assertEqual(_client(facts)._annual_flow_by_fy(facts, ("Revenues",)), {2024: 5.0})


class AmendmentAndPrecedenceTests(unittest.TestCase):
    def test_10k_amendment_restates_the_latest_year(self) -> None:
        usd = _base_usd()
        usd["Revenues"] = usd["Revenues"] + [_fy(2024, 1450.0, "2025-06-01", form="10-K/A")]
        fin, _bs, _notes = _parse(usd)
        self.assertEqual(fin.revenue[-1], 1450.0)

    def test_20f_amendment_and_40f_are_annual_forms(self) -> None:
        facts = _facts(Revenues=[_fy(2022, 7.0, form="40-F"),
                                 _fy(2023, 8.0, form="20-F"),
                                 _fy(2023, 9.0, "2024-09-01", form="20-F/A")])
        self.assertEqual(_client(facts)._annual_flow_by_fy(facts, ("Revenues",)),
                         {2022: 7.0, 2023: 9.0})

    def test_total_revenues_beats_the_asc606_subset(self) -> None:
        # A lessor/finance-arm filer: total Revenues includes lease income, the
        # ASC 606 contract-revenue tag does not.
        facts = _facts(**{
            "Revenues": [_fy(2023, 1000.0), _fy(2024, 1100.0)],
            PREFERRED: [_fy(2022, 280.0), _fy(2023, 300.0), _fy(2024, 320.0)],
        })
        series = _client(facts)._annual_flow_by_fy(facts, _TAGS_REVENUE)
        self.assertEqual(series, {2022: 280.0, 2023: 1000.0, 2024: 1100.0})


class AnnualFinancialsTests(unittest.TestCase):
    def test_clean_filer_parses_without_data_gap_notes(self) -> None:
        fin, bs, notes = _parse(_base_usd())
        self.assertEqual(fin.fiscal_years, _YEARS)
        self.assertEqual(fin.ebit, [150.0, 160.0, 170.0, 180.0, 190.0, 200.0])
        self.assertEqual(bs.total_debt, 480.0)
        self.assertFalse([n for n in notes if "EBIT" in n or "D&A" in n])

    def test_ebit_derived_from_pretax_plus_interest_when_operating_income_missing(self) -> None:
        fin, _bs, notes = _parse(_without(_base_usd(), "OperatingIncomeLoss"))
        self.assertEqual(fin.ebit, [140.0, 150.0, 160.0, 170.0, 180.0, 190.0])
        self.assertEqual(fin.ebitda, [e + 50.0 for e in fin.ebit])
        self.assertTrue(any("FY2019-2024; derived as pretax income + interest expense" in n
                            for n in notes), notes)

    def test_missing_latest_operating_income_is_derived_not_zero(self) -> None:
        usd = _base_usd()
        usd["OperatingIncomeLoss"] = [e for e in usd["OperatingIncomeLoss"]
                                      if not e["end"].startswith("2024")]
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin.ebit[-1], 190.0)  # pretax 180 + interest 10
        self.assertEqual(fin.ebit[:-1], [150.0, 160.0, 170.0, 180.0, 190.0])
        self.assertTrue(any("for FY2024; derived" in n for n in notes), notes)

    def test_ebit_from_revenue_minus_costs_when_no_pretax_income(self) -> None:
        usd = _without(_base_usd(), "OperatingIncomeLoss", _PRETAX)
        usd["CostsAndExpenses"] = [_fy(y, 900.0 + 100 * i) for i, y in enumerate(_YEARS)]
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin.ebit, [100.0] * 6)
        self.assertTrue(any("derived as revenue - CostsAndExpenses" in n for n in notes), notes)

    def test_ebit_without_any_source_is_zero_and_noted(self) -> None:
        fin, _bs, notes = _parse(_without(_base_usd(), "OperatingIncomeLoss", _PRETAX))
        self.assertEqual(fin.ebit, [0.0] * 6)
        self.assertIn("EBIT (operating income) unavailable on EDGAR; filled with 0.0", notes)

    def test_tax_benefit_keeps_its_sign(self) -> None:
        fin, _bs, _notes = _parse(_base_usd())
        self.assertEqual(fin.tax_expense, [30.0, -60.0, 30.0, 30.0, 30.0, 30.0])

    def test_partial_gap_in_a_series_is_noted(self) -> None:
        usd = _base_usd()
        usd["DepreciationDepletionAndAmortization"] = [_fy(y, 50.0) for y in (2019, 2020, 2021)]
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin.dep_amort, [50.0, 50.0, 50.0, 0.0, 0.0, 0.0])
        self.assertIn("D&A not reported on EDGAR for FY2022-2024; filled with 0.0", notes)

    def test_depreciation_backfills_years_without_a_d_and_a_tag(self) -> None:
        usd = _base_usd()
        usd["DepreciationDepletionAndAmortization"] = [_fy(y, 50.0) for y in (2019, 2020, 2021)]
        usd["Depreciation"] = [_fy(y, 45.0) for y in _YEARS]
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin.dep_amort, [50.0, 50.0, 50.0, 45.0, 45.0, 45.0])
        self.assertFalse([n for n in notes if n.startswith("D&A")], notes)

    def test_other_ppe_payments_fill_capex_without_the_main_tags(self) -> None:
        # Eli Lilly tags its capex only as PaymentsToAcquireOtherPropertyPlantAndEquipment.
        usd = _without(_base_usd(), "PaymentsToAcquirePropertyPlantAndEquipment")
        usd["PaymentsToAcquireOtherPropertyPlantAndEquipment"] = [_fy(y, 70.0) for y in _YEARS]
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin.capex, [70.0] * 6)
        self.assertFalse([n for n in notes if n.startswith("capex")], notes)

    def test_main_capex_tag_wins_over_other_ppe_payments(self) -> None:
        # A smaller other-PP&E line does not replace the main tag; it fills
        # the years without it.
        usd = _base_usd()
        usd["PaymentsToAcquirePropertyPlantAndEquipment"] = [
            _fy(y, 60.0) for y in _YEARS if y >= 2021]
        usd["PaymentsToAcquireOtherPropertyPlantAndEquipment"] = [_fy(y, 5.0) for y in _YEARS]
        fin, _bs, _notes = _parse(usd)
        self.assertEqual(fin.capex, [5.0, 5.0, 60.0, 60.0, 60.0, 60.0])

    def test_other_productive_assets_fill_capex_after_a_tag_switch(self) -> None:
        # Verizon: PaymentsToAcquireProductiveAssets until FY2018, then only
        # PaymentsToAcquireOtherProductiveAssets (17B a year), which read 0.
        usd = _without(_base_usd(), "PaymentsToAcquirePropertyPlantAndEquipment")
        usd["PaymentsToAcquireProductiveAssets"] = [_fy(2019, 166.0)]
        usd["PaymentsToAcquireOtherProductiveAssets"] = [
            _fy(y, 999.0 if y == 2019 else 170.0 + y - 2020) for y in _YEARS]
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin.capex, [166.0, 170.0, 171.0, 172.0, 173.0, 174.0])
        # Named for the years it alone fills, since for another filer the
        # line may be only part of capex; no gap note.
        self.assertEqual(
            [n for n in notes if n.startswith("capex")],
            ["capex for FY2020-2024 read from PaymentsToAcquireOtherProductiveAssets "
             "(payments for other productive assets), the only capex line tagged for "
             "those years; for some filers it is all of capex, for others only a part, "
             "so capex there may be understated"])

    def test_other_productive_assets_never_replace_a_main_capex_tag(self) -> None:
        # Outside the capex tags: a filer's "other productive assets" line
        # does not displace any of them (see CapexTagTests for a larger one).
        usd = _base_usd()
        usd["PaymentsToAcquireOtherPropertyPlantAndEquipment"] = [_fy(2019, 7.0)]
        usd["PaymentsToAcquirePropertyPlantAndEquipment"] = [
            _fy(y, 60.0) for y in _YEARS if y != 2019]
        usd["PaymentsToAcquireOtherProductiveAssets"] = [_fy(y, 3.0) for y in _YEARS]
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin.capex, [7.0, 60.0, 60.0, 60.0, 60.0, 60.0])
        self.assertFalse([n for n in notes if n.startswith("capex")], notes)

    def test_pretax_income_is_summed_from_domestic_and_foreign_parts(self) -> None:
        # McDonald's tags pretax income only as its domestic and foreign parts.
        # FY2024 has a total (it wins); FY2019 lacks the foreign part.
        usd = _base_usd()
        usd[_PRETAX] = [_fy(2024, 500.0)]
        usd["IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic"] = [
            _fy(y, 80.0) for y in _YEARS]
        usd["IncomeLossFromContinuingOperationsBeforeIncomeTaxesForeign"] = [
            _fy(y, 60.0) for y in _YEARS if y != 2019]
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin.pretax_income, [0.0, 140.0, 140.0, 140.0, 140.0, 500.0])
        self.assertIn("pretax income not tagged as a total on EDGAR for FY2020-2023; summed "
                      "from its domestic and foreign parts", notes)
        self.assertIn("pretax income not reported on EDGAR for FY2019; filled with 0.0", notes)

    def test_nwc_components_are_not_summed_without_the_aggregate_tag(self) -> None:
        # A GE Vernova-like year: receivables, inventories and payables would
        # sum to a +11 cash use, while customer advances (a contract-liability
        # source of 20) make the full working-capital change -9. A partial
        # sum would publish the wrong sign, so dNWC stays 0.0 with a note.
        usd = _base_usd()
        usd["IncreaseDecreaseInAccountsReceivable"] = [_fy(y, 10.0) for y in _YEARS]
        usd["IncreaseDecreaseInInventories"] = [_fy(y, 5.0) for y in _YEARS]
        usd["IncreaseDecreaseInAccountsPayable"] = [_fy(y, 4.0) for y in _YEARS]
        usd["IncreaseDecreaseInContractWithCustomerLiability"] = [_fy(y, 20.0) for y in _YEARS]
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin.change_in_nwc, [0.0] * 6)
        self.assertTrue(any(n.startswith("change in net working capital "
                                         "(IncreaseDecreaseInOperatingCapital) not reported")
                            for n in notes), notes)

    def test_aggregate_nwc_tag_wins_over_components(self) -> None:
        usd = _base_usd()
        usd["IncreaseDecreaseInOperatingCapital"] = [_fy(y, 7.0) for y in _YEARS]
        usd["IncreaseDecreaseInAccountsReceivable"] = [_fy(y, 10.0) for y in _YEARS]
        fin, _bs, _notes = _parse(usd)
        self.assertEqual(fin.change_in_nwc, [7.0] * 6)

    def test_revenue_moved_to_an_unread_tag_is_rejected_as_stale(self) -> None:
        usd = {
            "SalesRevenueNet": [_fy(y, 1000.0) for y in range(2010, 2018)],
            "RevenuesNetOfInterestExpense": [_fy(y, 2000.0) for y in range(2018, 2025)],
            "NetIncomeLoss": [_fy(y, 100.0) for y in range(2010, 2025)],
        }
        with self.assertRaisesRegex(DataError, "FY2017 but net income to FY2024"):
            _parse(usd)

    def test_one_year_revenue_lag_is_noted(self) -> None:
        usd = _base_usd()
        usd["Revenues"] = usd["Revenues"][:-1]  # FY2024 revenue under an unread tag
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin.fiscal_years[-1], 2023)
        self.assertTrue(any("revenue on EDGAR ends FY2023" in n for n in notes), notes)


class DepreciationTagTests(unittest.TestCase):
    """D&A takes the largest figure the D&A tags report for each period.

    Each tag is either the total or one part of it, and which one differs by
    filer, so the largest is the best lower bound; the tags are never added,
    since a part and the total would then be counted twice.
    """

    _DDA = "DepreciationDepletionAndAmortization"
    _COGS_DA = "CostOfGoodsAndServicesSoldDepreciationAndAmortization"

    def test_total_d_and_a_beats_a_smaller_preferred_tag(self) -> None:
        # McDonald's: DepreciationDepletionAndAmortization is 0.46B, one part;
        # DepreciationAndAmortization is the 2.20B total.
        usd = _base_usd()
        usd["DepreciationAndAmortization"] = [_fy(y, 220.0 + i) for i, y in enumerate(_YEARS)]
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin.dep_amort, [220.0, 221.0, 222.0, 223.0, 224.0, 225.0])
        self.assertEqual(fin.ebitda, [e + d for e, d in zip(fin.ebit, fin.dep_amort)])
        self.assertIn(
            "D&A for FY2019-2024 read from DepreciationAndAmortization, the largest D&A "
            f"figure tagged for those years, over the smaller {self._DDA} (each D&A tag is "
            "the total or a part of it, so the largest is used and none are added)", notes)

    def test_cost_of_sales_d_and_a_that_is_the_total_is_not_added_to_the_others(self) -> None:
        # Home Depot: cost-of-sales D&A (tagged from FY2023) holds the cash-
        # flow total of 406; the other tags are parts of it. Summing any two
        # would count D&A twice.
        usd = _base_usd()
        usd[self._DDA] = [_fy(y, 350.0) for y in _YEARS]
        usd["DepreciationAndAmortization"] = [_fy(y, 327.0) for y in _YEARS]
        usd["DepreciationNonproduction"] = [_fy(y, 345.0) for y in _YEARS]
        usd[self._COGS_DA] = [_fy(y, 406.0) for y in (2023, 2024)]
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin.dep_amort, [350.0, 350.0, 350.0, 350.0, 406.0, 406.0])
        self.assertEqual(
            [n for n in notes if n.startswith("D&A")],
            [f"D&A for FY2023-2024 read from {self._COGS_DA}, the largest D&A figure "
             f"tagged for those years, over the smaller {self._DDA} (each D&A tag is the "
             "total or a part of it, so the largest is used and none are added)"])

    def test_rental_fleet_depreciation_in_cost_of_sales_is_read(self) -> None:
        # United Rentals: rental-fleet depreciation (267) is tagged only as
        # cost-of-sales D&A and non-rental D&A (44) as DepreciationAndAmortization.
        # The fleet figure is used; the parts are not added (the total, 311,
        # is not tagged), so D&A stays a lower bound.
        usd = _without(_base_usd(), self._DDA)
        usd["DepreciationAndAmortization"] = [_fy(y, 44.0) for y in _YEARS]
        usd[self._COGS_DA] = [_fy(y, 267.0) for y in _YEARS]
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin.dep_amort, [267.0] * 6)
        self.assertTrue(any(n.startswith(f"D&A for FY2019-2024 read from {self._COGS_DA}")
                            for n in notes), notes)

    def test_matching_figures_keep_the_preferred_tag_without_a_note(self) -> None:
        # Within the 0.5% match tolerance (e.g. accretion rounding), the first
        # tag's figure stands; a larger figure in one year only replaces that year.
        usd = _base_usd()
        usd["DepreciationAmortizationAndAccretionNet"] = [
            _fy(y, 60.0 if y == 2022 else 50.2) for y in _YEARS]
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin.dep_amort, [50.0, 50.0, 50.0, 60.0, 50.0, 50.0])
        da_notes = [n for n in notes if n.startswith("D&A")]
        self.assertEqual(len(da_notes), 1, notes)
        self.assertTrue(da_notes[0].startswith(
            "D&A for FY2022 read from DepreciationAmortizationAndAccretionNet, the largest"))

    def test_only_a_smaller_tag_in_one_year_leaves_the_preferred_figure(self) -> None:
        # A part tagged alone for a year is still used there (the backfill),
        # and never replaces a larger preferred figure.
        usd = _base_usd()
        usd[self._DDA] = [_fy(y, 50.0) for y in _YEARS if y != 2021]
        usd["Depreciation"] = [_fy(y, 40.0) for y in _YEARS]
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin.dep_amort, [50.0, 50.0, 40.0, 50.0, 50.0, 50.0])
        self.assertFalse([n for n in notes if n.startswith("D&A")], notes)

    def test_a_dropped_tag_does_not_beat_a_later_restated_figure(self) -> None:
        # GE: FY2019-2020 D&A was restated (20.2, 21.3) in the 10-Ks filed
        # after its aircraft-leasing arm moved to discontinued operations, with
        # revenue and capex. Depreciation (40.3, 46.4), last filed in 2021,
        # still includes that arm, so it must not replace the restated figures.
        # In FY2021 a larger Depreciation filed after the D&A figure still wins.
        usd = _base_usd()
        usd[self._DDA] = [_fy(y, 50.0) for y in _YEARS] + [
            _fy(2019, 20.2, filed="2023-02-10"), _fy(2020, 21.3, filed="2023-02-10")]
        usd["Depreciation"] = [_fy(2019, 40.3, filed="2021-02-12"),
                               _fy(2020, 46.4, filed="2021-02-12"),
                               _fy(2021, 60.0, filed="2023-02-10")]
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin.dep_amort, [20.2, 21.3, 60.0, 50.0, 50.0, 50.0])
        self.assertEqual(
            [n for n in notes if n.startswith("D&A")],
            [f"D&A for FY2021 read from Depreciation, the largest D&A figure tagged for "
             f"those years, over the smaller {self._DDA} (each D&A tag is the total or a "
             "part of it, so the largest is used and none are added)"])

    def test_a_third_tag_must_be_filed_no_earlier_than_the_figure_it_replaces(self) -> None:
        # FY2021: the second tag's 60, filed in 2023, replaced the first tag's
        # 50 (filed 2022); the third tag's 70, filed in 2022, is older than
        # the 60 it would replace, so it does not win (though it is no older
        # than the first tag's figure). FY2022: all filed together, 70 wins.
        usd = _base_usd()
        usd["DepreciationAmortizationAndAccretionNet"] = [
            _fy(2021, 60.0, filed="2023-02-15"), _fy(2022, 60.0)]
        usd["DepreciationAndAmortization"] = [
            _fy(2021, 70.0, filed="2022-02-15"), _fy(2022, 70.0)]
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin.dep_amort, [50.0, 50.0, 60.0, 70.0, 50.0, 50.0])
        self.assertEqual(
            [n.split(", the largest")[0] for n in notes if n.startswith("D&A")],
            ["D&A for FY2021 read from DepreciationAmortizationAndAccretionNet",
             "D&A for FY2022 read from DepreciationAndAmortization"])


class CapexTagTests(unittest.TestCase):
    """Capex takes the larger figure of the two main capex tags for each
    period, under the same tolerance and filed-date guard as D&A, but only
    for a filer whose second tag is larger in every year both report.

    Either tag can be the total and the other a part of it; they are never
    added. A second tag that is larger only in some years (capex including
    acquisitions) or zero where the first is not is not the total. The
    other-PP&E and other-productive-assets lines stay outside the comparison
    and only fill the years neither main tag reports.
    """

    _PPE = "PaymentsToAcquirePropertyPlantAndEquipment"
    _PA = "PaymentsToAcquireProductiveAssets"
    _OTHER_PPE = "PaymentsToAcquireOtherPropertyPlantAndEquipment"
    _OTHER = "PaymentsToAcquireOtherProductiveAssets"
    _WHY = ("(it is larger in every year both tags report, so it is read as the total "
            "and the smaller as a part of it; the two are not added)")

    def test_total_capex_beats_a_smaller_component_tag_in_the_same_filing(self) -> None:
        # United Rentals: the first tag holds only non-rental capex (12-15);
        # the total with the rental fleet (190-220) sits under
        # PaymentsToAcquireProductiveAssets in the same 10-Ks. From FY2023
        # only the total is tagged.
        usd = _base_usd()
        usd[self._PPE] = [_fy(y, 12.0 + y - 2019) for y in _YEARS if y <= 2022]
        usd[self._PA] = [_fy(y, 190.0 + 10 * (y - 2019)) for y in _YEARS]
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin.capex, [190.0, 200.0, 210.0, 220.0, 230.0, 240.0])
        self.assertEqual(
            [n for n in notes if n.startswith("capex")],
            [f"capex for FY2019-2022 read from {self._PA}, the largest capex figure "
             f"tagged for those years, over the smaller {self._PPE} {self._WHY}"])

    def test_a_dropped_capex_tag_does_not_beat_a_later_restated_figure(self) -> None:
        # After a spin-off the first tag's FY2020-2021 capex is restated
        # (40, 45) in the 10-K filed in 2023; PaymentsToAcquireProductiveAssets
        # (75, 80), last filed in 2022, still includes the business that
        # left, so it must not replace the restated figures. In FY2022 its
        # larger figure, filed in the same 10-K as the first tag's, still wins.
        usd = _base_usd()
        usd[self._PPE] = usd[self._PPE] + [
            _fy(2020, 40.0, filed="2023-02-15"), _fy(2021, 45.0, filed="2023-02-15")]
        usd[self._PA] = [_fy(2020, 75.0, filed="2022-02-15"),
                         _fy(2021, 80.0, filed="2022-02-15"),
                         _fy(2022, 90.0, filed="2023-02-15")]
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin.capex, [60.0, 40.0, 45.0, 90.0, 60.0, 60.0])
        self.assertEqual(
            [n for n in notes if n.startswith("capex")],
            [f"capex for FY2022 read from {self._PA}, the largest capex figure tagged "
             f"for those years, over the smaller {self._PPE} {self._WHY}"])

    def test_matching_capex_figures_keep_the_preferred_tag_without_a_note(self) -> None:
        # Within the 0.5% match tolerance the first tag's figure stands.
        usd = _base_usd()
        usd[self._PA] = [_fy(y, 60.2) for y in _YEARS]
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin.capex, [60.0] * 6)
        self.assertFalse([n for n in notes if n.startswith("capex")], notes)

    def test_a_larger_other_productive_assets_line_only_fills_gaps(self) -> None:
        # Kept out of the largest-figure rule: where a main capex tag reports
        # the year, even a larger other-productive-assets figure is not used.
        usd = _base_usd()
        usd[self._PPE] = [_fy(y, 60.0) for y in _YEARS if y != 2024]
        usd[self._OTHER] = [_fy(y, 500.0 if y != 2024 else 70.0) for y in _YEARS]
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin.capex, [60.0, 60.0, 60.0, 60.0, 60.0, 70.0])
        capex_notes = [n for n in notes if n.startswith("capex")]
        self.assertEqual(len(capex_notes), 1, notes)
        self.assertTrue(capex_notes[0].startswith(
            f"capex for FY2024 read from {self._OTHER} (payments for other productive "
            "assets), the only capex line tagged for those years"), capex_notes)

    def test_capex_including_acquisitions_is_not_read_as_the_total(self) -> None:
        # Copart tags its segment note's capex "including acquisitions" as
        # PaymentsToAcquireProductiveAssets: its PP&E purchases plus the
        # year's acquisitions (FY2019, FY2021, FY2024), and the same figure in
        # years without one. Not larger every year, so not the total.
        usd = _base_usd()
        usd[self._PA] = [_fy(y, {2019: 75.0, 2021: 70.0, 2024: 90.0}.get(y, 60.0))
                         for y in _YEARS]
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin.capex, [60.0] * 6)
        self.assertFalse([n for n in notes if n.startswith("capex")], notes)

    def test_a_second_tag_reported_as_zero_is_not_read_as_the_total(self) -> None:
        # Larger in FY2019-2022 but zero in FY2023-2024 while the first tag
        # reports 60: a total is never below its part, so this is another line.
        usd = _base_usd()
        usd[self._PA] = [_fy(y, 0.0 if y >= 2023 else 150.0) for y in _YEARS]
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin.capex, [60.0] * 6)
        self.assertFalse([n for n in notes if n.startswith("capex")], notes)

    def test_a_separate_other_ppe_line_only_fills_gaps(self) -> None:
        # D.R. Horton: PP&E purchases and "expenditures related to rental
        # properties" (the other-PP&E element) are two lines that add up; the
        # second is larger in some years, smaller or zero in others. It is
        # used only for FY2019, which has no PP&E figure.
        usd = _base_usd()
        usd[self._PPE] = [_fy(y, 60.0) for y in _YEARS if y != 2019]
        usd[self._OTHER_PPE] = [
            _fy(y, v) for y, v in zip(_YEARS, (70.0, 62.0, 50.0, 120.0, 0.0, 0.0))]
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin.capex, [70.0, 60.0, 60.0, 60.0, 60.0, 60.0])
        self.assertFalse([n for n in notes if n.startswith("capex")], notes)

    def test_lease_fleet_purchases_inside_the_total_are_not_added_again(self) -> None:
        # The second tag's total (160) exceeds the first tag's 60 by exactly
        # the lease-fleet purchases (100) in FY2019-2021, so it holds them.
        # In FY2022 the excess (90) is not those purchases, and FY2023-2024
        # have no total: there the net purchases (100 - 70) are added.
        usd = _base_usd()
        usd[self._PA] = [_fy(y, 150.0 if y == 2022 else 160.0) for y in _YEARS if y <= 2022]
        usd["PaymentsToAcquireLeasesHeldForInvestment"] = [_fy(y, 100.0) for y in _YEARS]
        usd["ProceedsFromLeasesHeldForInvestment"] = [_fy(y, 70.0) for y in _YEARS]
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin.capex, [160.0, 160.0, 160.0, 180.0, 90.0, 90.0])
        self.assertEqual(
            [n.split(" read from")[0].split(" includes")[0]
             for n in notes if n.startswith("capex")],
            ["capex for FY2019-2022", "capex for FY2022-2024"])


class BalanceSheetTests(unittest.TestCase):
    def _bs(self, **instants: list[dict]):
        usd = _base_usd()
        for tag in ("StockholdersEquity", "CashAndCashEquivalentsAtCarryingValue",
                    "LongTermDebtNoncurrent"):
            usd.pop(tag)
        usd.update(instants)
        _fin, bs, notes = _parse(usd)
        return bs, notes

    def test_lines_no_longer_reported_are_not_summed_into_today(self) -> None:
        bs, notes = self._bs(
            StockholdersEquity=[_inst("2024-12-31", 1000.0), _q2("2025-06-30", 1050.0)],
            CashAndCashEquivalentsAtCarryingValue=[_inst("2024-12-31", 200.0),
                                                   _q2("2025-06-30", 210.0)],
            ShortTermInvestments=[_inst("2015-12-31", 900.0), _inst("2016-12-31", 800.0)],
            MinorityInterest=[_inst("2019-12-31", 70.0)],
        )
        self.assertEqual(bs.as_of, "2025-06-30")
        self.assertEqual(bs.cash_and_investments, 210.0)
        self.assertEqual(bs.total_equity, 1050.0)
        self.assertEqual(bs.minority_interest, 0.0)
        self.assertTrue(any(n.startswith("short-term investments last reported 2016-12-31")
                            for n in notes), notes)

    def test_item_only_in_the_last_10k_is_used_at_a_later_10q_date(self) -> None:
        bs, notes = self._bs(
            StockholdersEquity=[_q2("2025-06-30", 1050.0)],
            CashAndCashEquivalentsAtCarryingValue=[_q2("2025-06-30", 210.0)],
            MinorityInterest=[_inst("2024-12-31", 70.0)],
        )
        self.assertEqual(bs.minority_interest, 70.0)
        self.assertIn("minority interest taken from 2024-12-31; not reported at the "
                      "2025-06-30 balance sheet", notes)

    def test_debt_current_is_not_added_to_its_own_components(self) -> None:
        # DebtCurrent 70 = short-term borrowings 40 + current LTD 30.
        bs, _notes = self._bs(
            StockholdersEquity=[_inst("2024-12-31", 1000.0)],
            LongTermDebtNoncurrent=[_inst("2024-12-31", 470.0)],
            LongTermDebtCurrent=[_inst("2024-12-31", 30.0)],
            DebtCurrent=[_inst("2024-12-31", 70.0)],
        )
        self.assertEqual(bs.total_debt, 540.0)

    def test_lease_inclusive_noncurrent_tag_is_combined_with_current_portion(self) -> None:
        bs, _notes = self._bs(
            StockholdersEquity=[_inst("2024-12-31", 1000.0)],
            LongTermDebt=[_inst("2024-12-31", 1000.0)],
            LongTermDebtAndCapitalLeaseObligations=[_inst("2024-12-31", 950.0)],
            LongTermDebtCurrent=[_inst("2024-12-31", 50.0)],
        )
        self.assertEqual(bs.total_debt, 1000.0)

    def test_commercial_paper_counts_as_debt(self) -> None:
        bs, _notes = self._bs(
            StockholdersEquity=[_inst("2024-12-31", 1000.0)],
            CommercialPaper=[_inst("2024-12-31", 10.0)],
            LongTermDebtCurrent=[_inst("2024-12-31", 10.9)],
            LongTermDebtNoncurrent=[_inst("2024-12-31", 85.8)],
        )
        self.assertAlmostEqual(bs.total_debt, 106.7)

    def test_current_fallback_tag_beats_a_stale_preferred_tag(self) -> None:
        bs, _notes = self._bs(
            StockholdersEquity=[_inst("2024-12-31", 1000.0)],
            ShortTermBorrowings=[_inst("2016-12-31", 300.0)],
            DebtCurrent=[_inst("2024-12-31", 20.0)],
            LongTermDebtNoncurrent=[_inst("2024-12-31", 400.0)],
        )
        self.assertEqual(bs.total_debt, 420.0)

    def test_long_term_debt_total_plus_short_term_borrowings(self) -> None:
        # LongTermDebt already includes its current maturities (50).
        bs, _notes = self._bs(
            StockholdersEquity=[_inst("2024-12-31", 1000.0)],
            LongTermDebt=[_inst("2024-12-31", 500.0)],
            LongTermDebtCurrent=[_inst("2024-12-31", 50.0)],
            ShortTermBorrowings=[_inst("2024-12-31", 25.0)],
        )
        self.assertEqual(bs.total_debt, 525.0)

    def test_preferred_stock_is_read(self) -> None:
        bs, _notes = self._bs(
            StockholdersEquity=[_inst("2024-12-31", 900.0)],
            PreferredStockValue=[_inst("2024-12-31", 250.0)],
        )
        self.assertEqual(bs.preferred_equity, 250.0)


class CashTagTests(unittest.TestCase):
    """Cash at the snapshot across the cash tags filers actually use."""

    def _bs(self, **instants: list[dict]):
        usd = _without(_base_usd(), "CashAndCashEquivalentsAtCarryingValue")
        usd["StockholdersEquity"] = [_inst("2023-12-31", 950.0), _inst("2024-12-31", 1000.0)]
        usd.update(instants)
        _fin, bs, notes = _parse(usd)
        return bs, notes

    def test_total_including_restricted_cash_is_used_when_it_is_the_only_tag(self) -> None:
        # GE Vernova: no CashAndCashEquivalentsAtCarryingValue fact at all.
        bs, notes = self._bs(**{_TOTAL_CASH: [_inst("2024-12-31", 131.2)]})
        self.assertAlmostEqual(bs.cash_and_investments, 131.2)
        self.assertNotIn("cash & equivalents unavailable on EDGAR; set to 0.0", notes)
        self.assertTrue(any("restricted cash is not tagged separately" in n for n in notes), notes)

    def test_stale_plain_tag_gives_way_to_the_current_total_less_restricted(self) -> None:
        # GE: the plain tag was last used in 2017; the 2024 total includes a
        # noncurrent restricted balance tagged at the same date.
        bs, notes = self._bs(**{
            "CashAndCashEquivalentsAtCarryingValue": [_inst("2017-12-31", 433.0)],
            _TOTAL_CASH: [_inst("2024-12-31", 93.45)],
            "RestrictedCashAndCashEquivalentsNoncurrent": [_inst("2024-12-31", 11.68)],
        })
        self.assertAlmostEqual(bs.cash_and_investments, 81.77)
        self.assertTrue(any("less restricted cash (12)" in n for n in notes), notes)

    def test_current_and_noncurrent_restricted_cash_are_both_removed(self) -> None:
        # Chevron: 95.82 total = 85.27 cash + 2.36 current + 8.19 noncurrent.
        bs, _notes = self._bs(**{
            _TOTAL_CASH: [_inst("2024-12-31", 95.82)],
            "RestrictedCashCurrent": [_inst("2024-12-31", 2.36)],
            "RestrictedCashNoncurrent": [_inst("2024-12-31", 8.19)],
        })
        self.assertAlmostEqual(bs.cash_and_investments, 85.27)

    def test_restricted_cash_from_another_date_is_not_subtracted(self) -> None:
        bs, _notes = self._bs(**{
            _TOTAL_CASH: [_inst("2024-12-31", 131.2)],
            "RestrictedCash": [_inst("2023-12-31", 3.79)],
        })
        self.assertAlmostEqual(bs.cash_and_investments, 131.2)

    def test_balance_sheet_cash_beats_a_larger_total_at_the_same_date(self) -> None:
        # PayPal / Airbnb: the total also holds customer funds.
        bs, _notes = self._bs(**{
            "CashAndCashEquivalentsAtCarryingValue": [_inst("2024-12-31", 69.8)],
            _TOTAL_CASH: [_inst("2024-12-31", 224.4)],
        })
        self.assertAlmostEqual(bs.cash_and_investments, 69.8)

    def test_cash_line_equal_to_the_total_keeps_restricted_cash_outside(self) -> None:
        # Travelers: Cash 621 = the total 621; its restricted cash sits elsewhere.
        bs, _notes = self._bs(**{
            _TOTAL_CASH: [_inst("2024-12-31", 621.0)],
            "Cash": [_inst("2024-12-31", 621.0)],
            "RestrictedCash": [_inst("2024-12-31", 139.0)],
        })
        self.assertAlmostEqual(bs.cash_and_investments, 621.0)

    def test_total_with_disposal_group_cash_is_the_last_fallback(self) -> None:
        # PACCAR: the balance-sheet cash tag stopped in 2019; since then only
        # the cash-flow total including disposal-group cash is tagged.
        bs, notes = self._bs(**{
            "CashAndCashEquivalentsAtCarryingValue": [_inst("2019-06-30", 32.2)],
            _TOTAL_CASH_DISPOSAL: [_inst("2024-12-31", 55.7)],
        })
        self.assertAlmostEqual(bs.cash_and_investments, 55.7)
        self.assertNotIn("cash & equivalents unavailable on EDGAR; set to 0.0", notes)
        self.assertIn("cash & equivalents from the cash-flow total including restricted cash "
                      "and the cash of disposal groups (56) at 2024-12-31; restricted cash is "
                      "not tagged separately, so any restricted balance is included", notes)

    def test_restricted_cash_is_taken_out_of_the_disposal_group_total(self) -> None:
        bs, notes = self._bs(**{
            _TOTAL_CASH_DISPOSAL: [_inst("2024-12-31", 55.7)],
            "RestrictedCashCurrent": [_inst("2024-12-31", 5.2)],
        })
        self.assertAlmostEqual(bs.cash_and_investments, 50.5)
        self.assertTrue(any("disposal groups (56) at 2024-12-31, less restricted cash (5)" in n
                            for n in notes), notes)

    def test_every_other_cash_tag_beats_the_disposal_group_total(self) -> None:
        bs, _notes = self._bs(**{
            _TOTAL_CASH_DISPOSAL: [_inst("2024-12-31", 90.0)],
            "CashAndDueFromBanks": [_inst("2024-12-31", 70.0)],
        })
        self.assertAlmostEqual(bs.cash_and_investments, 70.0)

    def test_bank_cash_and_due_from_banks_is_a_fallback(self) -> None:
        bs, notes = self._bs(CashAndDueFromBanks=[_inst("2024-12-31", 247.2)])
        self.assertAlmostEqual(bs.cash_and_investments, 247.2)
        self.assertIn("cash & equivalents read from the CashAndDueFromBanks line", notes)

    def test_short_term_investments_inside_cash_equivalents_are_not_added_twice(self) -> None:
        # Target: cash, cash equivalents and short-term investments = cash alone
        # at the 10-K date, so the investments are already in cash.
        bs, notes = self._bs(**{
            "StockholdersEquity": [_inst("2024-12-31", 1000.0), _q2("2025-06-30", 1010.0)],
            _TOTAL_CASH: [_inst("2024-12-31", 54.9), _q2("2025-06-30", 54.1)],
            "CashCashEquivalentsAndShortTermInvestments": [_inst("2024-12-31", 54.9),
                                                           _q2("2025-06-30", 54.1)],
            "ShortTermInvestments": [_inst("2024-12-31", 46.1)],
        })
        self.assertAlmostEqual(bs.cash_and_investments, 54.1)
        self.assertTrue(any("are part of cash & equivalents" in n for n in notes), notes)

    def test_separate_short_term_investments_are_still_added(self) -> None:
        # Microsoft: 76.8 = 20.9 cash + 55.9 short-term investments.
        bs, _notes = self._bs(**{
            "CashAndCashEquivalentsAtCarryingValue": [_inst("2024-12-31", 20.9)],
            "ShortTermInvestments": [_inst("2024-12-31", 55.9)],
            "CashCashEquivalentsAndShortTermInvestments": [_inst("2024-12-31", 76.8)],
        })
        self.assertAlmostEqual(bs.cash_and_investments, 76.8)


class DebtTagTests(unittest.TestCase):
    """Total debt from the tag families filers use, without double counting."""

    def _bs(self, **instants: list[dict]):
        usd = _without(_base_usd(), "LongTermDebtNoncurrent")
        usd["Assets"] = [_inst("2023-12-31", 4000.0), _inst("2024-12-31", 5000.0)]
        usd.update(instants)
        _fin, bs, notes = _parse(usd)
        return bs, notes

    def test_reit_debt_lines_are_summed_with_commercial_paper(self) -> None:
        # Realty Income: notes, term loans, mortgages and commercial paper, no total.
        bs, notes = self._bs(
            NotesPayable=[_inst("2024-12-31", 250.92)],
            LoansPayable=[_inst("2024-12-31", 27.60)],
            SecuredDebt=[_inst("2024-12-31", 0.37)],
            CommercialPaper=[_inst("2024-12-31", 14.0)],
        )
        self.assertAlmostEqual(bs.total_debt, 292.89)
        self.assertTrue(any("summed from the balance-sheet debt lines (NotesPayable, "
                            "LoansPayable, SecuredDebt) plus short-term borrowings" in n
                            for n in notes), notes)

    def test_a_line_that_totals_other_lines_is_not_double_counted(self) -> None:
        # Mid-America: notes payable 56.57 = unsecured 52.96 + secured 3.60.
        bs, _notes = self._bs(
            NotesPayable=[_inst("2024-12-31", 56.57)],
            UnsecuredDebt=[_inst("2024-12-31", 52.96)],
            SecuredDebt=[_inst("2024-12-31", 3.60)],
        )
        self.assertAlmostEqual(bs.total_debt, 56.57)

    def test_senior_notes_inside_notes_payable_count_once(self) -> None:
        bs, _notes = self._bs(
            NotesPayable=[_inst("2024-12-31", 100.0)],
            SeniorNotes=[_inst("2024-12-31", 80.0)],
        )
        self.assertEqual(bs.total_debt, 100.0)

    def test_debt_lines_never_fall_below_current_debt(self) -> None:
        # Deere: 171 of current borrowings, 61 of secured borrowings tagged.
        bs, _notes = self._bs(
            DebtCurrent=[_inst("2024-12-31", 171.0)],
            SecuredDebt=[_inst("2024-12-31", 61.0)],
        )
        self.assertEqual(bs.total_debt, 171.0)

    def test_long_term_total_including_current_maturities_plus_short_term(self) -> None:
        # MetLife: long-term debt incl. current maturities + short-term debt.
        bs, _notes = self._bs(
            LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities=[
                _inst("2024-12-31", 142.44)],
            ShortTermBorrowings=[_inst("2024-12-31", 4.60)],
            SubordinatedDebt=[_inst("2024-12-31", 51.44)],
        )
        self.assertAlmostEqual(bs.total_debt, 147.04)

    def test_debt_and_lease_obligation_total_is_read(self) -> None:
        # Aflac: one short- plus long-term figure.
        bs, _notes = self._bs(DebtAndCapitalLeaseObligations=[_inst("2024-12-31", 87.29)])
        self.assertAlmostEqual(bs.total_debt, 87.29)

    def test_noncurrent_notes_line_plus_current_debt(self) -> None:
        # Oracle: noncurrent notes payable + current debt.
        bs, _notes = self._bs(
            LongTermNotesPayable=[_inst("2024-12-31", 1223.42)],
            DebtCurrent=[_inst("2024-12-31", 71.99)],
        )
        self.assertAlmostEqual(bs.total_debt, 1295.41)

    def test_boeing_current_debt_plus_noncurrent_lease_inclusive_line(self) -> None:
        # Boeing: DebtCurrent 45.65 + noncurrent 413.35; the LongTermDebt
        # total (455.96) must not be added on top.
        bs, _notes = self._bs(
            DebtCurrent=[_inst("2024-12-31", 45.65)],
            LongTermDebt=[_inst("2024-12-31", 455.96)],
            LongTermDebtAndCapitalLeaseObligations=[_inst("2024-12-31", 413.35)],
        )
        self.assertAlmostEqual(bs.total_debt, 459.0)

    def test_stale_zero_current_portion_does_not_hide_the_combined_total(self) -> None:
        # Progressive: a zero LongTermDebtCurrent last tagged years ago must not
        # stand in for the current combined total.
        bs, _notes = self._bs(
            LongTermDebtCurrent=[_inst("2019-12-31", 0.0)],
            DebtLongtermAndShorttermCombinedAmount=[_inst("2024-12-31", 83.87)],
        )
        self.assertAlmostEqual(bs.total_debt, 83.87)

    def test_debt_fact_dated_off_a_balance_sheet_date_is_ignored(self) -> None:
        # NNN REIT: a 2.0 "LongTermDebt" dated mid-quarter describes one note
        # issue, not the balance; the balance-sheet lines give the debt.
        bs, _notes = self._bs(
            LongTermDebt=[_inst("2024-10-15", 2.0)],
            NotesPayable=[_inst("2024-12-31", 44.76)],
            LoansPayable=[_inst("2024-12-31", 4.97)],
        )
        self.assertAlmostEqual(bs.total_debt, 49.73)

    def test_noncurrent_unsecured_debt_line_is_read(self) -> None:
        # CME: its whole long-term debt is tagged UnsecuredLongTermDebt.
        bs, notes = self._bs(UnsecuredLongTermDebt=[_inst("2024-12-31", 34.24)])
        self.assertAlmostEqual(bs.total_debt, 34.24)
        self.assertTrue(any("debt lines (UnsecuredLongTermDebt)" in n for n in notes), notes)

    def test_noncurrent_only_lines_take_all_current_debt(self) -> None:
        # The noncurrent line excludes its current maturities, so the current
        # portion of long-term debt is added, not just short-term borrowings.
        bs, notes = self._bs(
            UnsecuredLongTermDebt=[_inst("2024-12-31", 30.0)],
            LongTermDebtCurrent=[_inst("2024-12-31", 5.0)],
        )
        self.assertAlmostEqual(bs.total_debt, 35.0)
        self.assertTrue(any("(UnsecuredLongTermDebt) plus current debt" in n for n in notes),
                        notes)

    def test_senior_long_term_notes_are_summed_with_other_lines(self) -> None:
        # Credit Acceptance: senior notes, secured financing and a revolver.
        bs, _notes = self._bs(
            SeniorLongTermNotes=[_inst("2024-12-31", 10.89)],
            SecuredDebt=[_inst("2024-12-31", 50.19)],
            LineOfCredit=[_inst("2024-12-31", 1.78)],
        )
        self.assertAlmostEqual(bs.total_debt, 62.86)

    def test_including_current_line_beats_a_noncurrent_line_in_its_group(self) -> None:
        bs, _notes = self._bs(
            UnsecuredDebt=[_inst("2024-12-31", 40.0)],
            UnsecuredLongTermDebt=[_inst("2024-12-31", 35.0)],
            SecuredLongTermDebt=[_inst("2024-12-31", 8.0)],
        )
        self.assertAlmostEqual(bs.total_debt, 48.0)


class DebtPlausibilityTests(unittest.TestCase):
    _WARNING_END = ("company-specific elements); net debt and the WACC debt weight may "
                    "be understated")

    @staticmethod
    def _usd(interest: float, **instants: list[dict]) -> dict:
        usd = _without(_base_usd(), "LongTermDebtNoncurrent")
        usd["InterestExpense"] = [_fy(y, interest) for y in _YEARS]
        usd.update(instants)
        return usd

    def _debt_warnings(self, notes: list[str]) -> list[str]:
        return [n for n in notes if n.startswith("WARNING") and "interest expense" in n]

    def test_interest_without_debt_is_a_leading_warning(self) -> None:
        # Ford / Berkshire: debt tagged only with company-specific elements.
        _fin, bs, notes = _parse(self._usd(60.0))
        self.assertEqual(bs.total_debt, 0.0)
        self.assertTrue(notes[0].startswith("WARNING: FY2024 interest expense is 60 but no "
                                            "debt was found"), notes)
        self.assertTrue(notes[0].endswith(self._WARNING_END), notes[0])

    def test_small_interest_of_a_debt_free_company_is_not_flagged(self) -> None:
        # Copart: a little interest on leases and fees, no borrowings.
        _fin, _bs, notes = _parse(self._usd(0.5))
        self.assertFalse([n for n in notes if n.startswith("WARNING")], notes)

    def test_plausible_interest_is_not_flagged(self) -> None:
        _fin, _bs, notes = _parse(_base_usd())  # 10 interest on 480 of debt
        self.assertFalse([n for n in notes if n.startswith("WARNING")], notes)

    def test_debt_repaid_during_the_year_is_a_note_not_a_warning(self) -> None:
        # SanDisk / SailPoint: a term loan repaid within the fiscal year leaves
        # no debt at the snapshot, but the year's balance sheets carried it.
        dates = ("2023-12-31", "2024-06-30", "2024-12-31")
        _fin, bs, notes = _parse(self._usd(
            40.0,
            Assets=[_inst(d, 5000.0) for d in dates],
            LongTermDebtNoncurrent=[_inst(d, v) for d, v in zip(dates, (600.0, 300.0, 0.0))],
        ))
        self.assertEqual(bs.total_debt, 0.0)
        self.assertFalse(self._debt_warnings(notes), notes)
        self.assertIn("FY2024 interest expense (40) fits the debt carried during that year "
                      "(total debt averaged 300 on the balance sheets from 2023-12-31 to "
                      "2024-12-31); total debt is 0 at the 2024-12-31 balance sheet, so it "
                      "was repaid or is tagged differently there", notes)

    def test_one_large_opening_balance_does_not_hide_missing_debt(self) -> None:
        # A filer that moved most of its debt to company-specific tags at the
        # start of the year: only the opening balance sheet shows it.
        dates = ("2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31")
        _fin, _bs, notes = _parse(self._usd(
            90.0,
            Assets=[_inst(d, 5000.0) for d in dates],
            LongTermDebtNoncurrent=[_inst(d, 1600.0 if d == dates[0] else 100.0)
                                    for d in dates],
        ))
        self.assertTrue(notes[0].startswith("WARNING: FY2024 interest expense is 90 but total "
                                            "debt read is only 100"), notes)

    def test_bank_interest_is_not_checked_against_debt(self) -> None:
        # Deposits, not debt, carry a bank's interest expense.
        usd = self._usd(900.0, Assets=[_inst("2024-12-31", 10000.0)],
                        Deposits=[_inst("2024-12-31", 6000.0)])
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin._financial_kind, "bank")
        self.assertFalse(self._debt_warnings(notes), notes)

    def test_hybrid_provider_finds_the_warning_to_backfill_debt(self) -> None:
        # HybridProvider backfills debt from Yahoo when its matcher finds this
        # WARNING, also behind a leading captive-finance WARNING (Ford).
        self.assertIsNone(_missing_debt_warning(_parse(_base_usd())[2]))
        self.assertEqual(_missing_debt_warning(_parse(self._usd(60.0))[2]), 0)
        fin, _bs, notes = _parse(self._usd(
            60.0, Assets=[_inst("2024-12-31", 10000.0)],
            NotesReceivableNet=[_inst("2024-12-31", 4000.0)],
            InventoryNet=[_inst("2024-12-31", 800.0)],
        ))
        self.assertEqual(fin._financial_kind, "captive_finance")
        self.assertEqual(_missing_debt_warning(notes), 1, notes)


class FinancialFilerTests(unittest.TestCase):
    """Banks, insurers, BDCs, REITs and lenders.

    All four financial kinds are flagged with a leading WARNING. EBIT is still
    derived as pretax income + interest expense where interest is a financing
    cost (insurers, equity REITs), but not where it is a cost of the lending
    book: banks, BDCs, and other filers with lender-scale interest expense.
    """

    @staticmethod
    def _usd(**extra: list[dict]) -> dict:
        usd = _without(_base_usd(), "OperatingIncomeLoss")
        usd["Assets"] = [_inst("2023-12-31", 9000.0), _inst("2024-12-31", 10000.0)]
        usd.update(extra)
        return usd

    def _assert_not_derived(self, usd: dict, kind, reason: str,
                            years: str = "FY2019-2024") -> tuple:
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin._financial_kind, kind)
        prefix = (f"EBIT (OperatingIncomeLoss) not reported on EDGAR for {years}; left at "
                  f"0.0 rather than derived as pretax income + interest expense: {reason}")
        self.assertTrue(any(n.startswith(prefix) for n in notes), notes)
        self.assertFalse([n for n in notes if "; derived as" in n], notes)
        return fin, notes

    def _assert_flagged(self, notes: list[str], label: str) -> None:
        self.assertTrue(notes[0].startswith(f"WARNING: EDGAR tags mark this company as {label}"),
                        notes)

    def test_bank_ebit_is_not_rebuilt_from_pretax_plus_interest(self) -> None:
        # JPM-like: deposits fund most of the balance sheet, and pretax +
        # interest (a deposit cost) would exceed revenue.
        usd = self._usd(Deposits=[_inst("2024-12-31", 5400.0)])
        usd["InterestExpense"] = [_fy(y, 900.0) for y in _YEARS]
        fin, notes = self._assert_not_derived(
            usd, "bank", "interest is an operating cost of a bank, so adding it back "
                         "would count it twice")
        self.assertEqual(fin.ebit, [0.0] * 6)
        self._assert_flagged(notes, "a bank")
        self.assertIn("deposits are 54% of total assets", notes[0])

    def test_bdc_ebit_is_not_rebuilt_from_pretax_plus_interest(self) -> None:
        fin, notes = self._assert_not_derived(
            self._usd(InvestmentOwnedAtFairValue=[_inst("2024-12-31", 9600.0)]),
            "bdc", "interest is an operating cost of a business development company")
        self.assertEqual(fin.ebit, [0.0] * 6)
        self._assert_flagged(notes, "a business development company")

    def test_bank_keeps_its_reported_operating_income_years(self) -> None:
        # Only the years without OperatingIncomeLoss are left at 0.0 and noted.
        usd = self._usd(Deposits=[_inst("2024-12-31", 5400.0)],
                        OperatingIncomeLoss=[_fy(y, 150.0 + 10 * i)
                                             for i, y in enumerate(_YEARS) if y >= 2022])
        fin, _notes = self._assert_not_derived(
            usd, "bank", "interest is an operating cost of a bank", years="FY2019-2021")
        self.assertEqual(fin.ebit, [0.0, 0.0, 0.0, 180.0, 190.0, 200.0])

    def test_insurer_is_flagged_and_its_ebit_derived(self) -> None:
        # Progressive: interest expense is on corporate debt, a financing cost.
        fin, _bs, notes = _parse(self._usd(
            PremiumsEarnedNet=[_fy(y, 900.0 + 90 * i) for i, y in enumerate(_YEARS)],
            NetInvestmentIncome=[_fy(y, 60.0) for y in _YEARS],
        ))
        self.assertEqual(fin._financial_kind, "insurer")
        self._assert_flagged(notes, "an insurer")
        self.assertEqual(fin.ebit, [140.0, 150.0, 160.0, 170.0, 180.0, 190.0])
        self.assertTrue(any("FY2019-2024; derived as pretax income + interest expense" in n
                            for n in notes), notes)

    def test_insurer_keeps_reported_years_and_derives_only_the_missing_ones(self) -> None:
        # MetLife: OperatingIncomeLoss from FY2022 on; earlier years derived.
        fin, _bs, notes = _parse(self._usd(
            PremiumsEarnedNet=[_fy(y, 900.0 + 90 * i) for i, y in enumerate(_YEARS)],
            NetInvestmentIncome=[_fy(y, 60.0) for y in _YEARS],
            OperatingIncomeLoss=[_fy(y, 150.0 + 10 * i)
                                 for i, y in enumerate(_YEARS) if y >= 2022],
        ))
        self.assertEqual(fin.ebit, [140.0, 150.0, 160.0, 180.0, 190.0, 200.0])
        self.assertIn("EBIT (OperatingIncomeLoss) not reported on EDGAR for FY2019-2021; "
                      "derived as pretax income + interest expense", notes)

    def test_reit_is_flagged_and_its_ebit_derived_despite_high_interest(self) -> None:
        # Realty Income: interest near 19% of revenue is on property debt, so
        # pretax income + interest is its EBIT and EBITDA is not D&A alone.
        usd = self._usd(RealEstateInvestmentPropertyNet=[_inst("2024-12-31", 7200.0)],
                        LongTermDebtNoncurrent=[_inst("2024-12-31", 4000.0)])
        usd["InterestExpense"] = [_fy(y, 280.0) for y in _YEARS]
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin._financial_kind, "reit")
        self._assert_flagged(notes, "a REIT")
        self.assertEqual(fin.ebit, [410.0, 420.0, 430.0, 440.0, 450.0, 460.0])
        self.assertEqual(fin.ebitda[-1], 510.0)

    def test_schedule_iii_real_estate_marks_a_reit_without_a_property_tag(self) -> None:
        # Equinix: no RealEstateInvestmentPropertyNet, but real estate at gross
        # carrying value on its Schedule III is 84% of total assets.
        usd = self._usd(RealEstateGrossAtCarryingValue=[_inst("2023-12-31", 7000.0),
                                                        _inst("2024-12-31", 8400.0)])
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin._financial_kind, "reit")
        self.assertTrue(notes[0].startswith(
            "WARNING: EDGAR tags mark this company as a REIT (real estate at gross "
            "carrying value (its Schedule III) is 84% of total assets)"), notes)

    def test_lender_scale_interest_is_not_added_back(self) -> None:
        # No deposits or fair-value investments to classify the filer, but
        # interest expense of 32% (Jefferies-like) or 172% (Interactive
        # Brokers-like) of revenue is a funding cost of the book.
        for interest, share in ((480.0, "32%"), (2580.0, "172%")):
            with self.subTest(share=share):
                usd = self._usd()
                usd["InterestExpense"] = [_fy(y, interest) for y in _YEARS]
                fin, notes = self._assert_not_derived(
                    usd, None, f"FY2024 interest expense is {share} of revenue, as for a "
                               "lender, broker or mortgage REIT")
                self.assertEqual(fin.ebit, [0.0] * 6)
                self.assertFalse([n for n in notes if n.startswith("WARNING: EDGAR tags")],
                                 notes)

    def test_interest_without_a_revenue_line_is_not_added_back(self) -> None:
        # Annaly-like mortgage REIT: interest income is not tagged as revenue.
        usd = _without(self._usd(), "Revenues")
        usd["InterestExpense"] = [_fy(y, 4800.0) for y in _YEARS]
        fin, _notes = self._assert_not_derived(
            usd, None, "FY2024 interest expense is 4,800 with no revenue")
        self.assertEqual(fin.ebit, [0.0] * 6)

    def test_pretax_plus_interest_above_revenue_is_not_used(self) -> None:
        # A gain on a sale lifts FY2022 pretax income above revenue; that year
        # falls back to revenue - CostsAndExpenses.
        usd = self._usd(CostsAndExpenses=[_fy(2022, 1100.0)])
        usd[_PRETAX] = [_fy(y, 1400.0 if y == 2022 else 130.0 + 10 * i)
                        for i, y in enumerate(_YEARS)]
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin.ebit, [140.0, 150.0, 160.0, 200.0, 180.0, 190.0])
        self.assertIn("EBIT (OperatingIncomeLoss) not reported on EDGAR for FY2022, and pretax "
                      "income + interest expense exceeds revenue there, so it is not used as "
                      "EBIT", notes)
        self.assertIn("EBIT (OperatingIncomeLoss) not reported on EDGAR for FY2022; derived "
                      "as revenue - CostsAndExpenses", notes)

    def _assert_derived(self, usd: dict) -> None:
        fin, _bs, notes = _parse(usd)
        self.assertIsNone(fin._financial_kind)
        self.assertEqual(fin.ebit, [140.0, 150.0, 160.0, 170.0, 180.0, 190.0])
        self.assertFalse([n for n in notes if n.startswith("WARNING")], notes)

    def test_operating_company_without_operating_income_is_still_derived(self) -> None:
        # Eli Lilly: no OperatingIncomeLoss, no financial lines.
        self._assert_derived(self._usd())

    def test_operating_company_below_the_lender_share_is_still_derived(self) -> None:
        # Deere-like: interest expense at 7% of revenue.
        usd = self._usd()
        usd["InterestExpense"] = [_fy(y, 105.0) for y in _YEARS]
        fin, _bs, _notes = _parse(usd)
        self.assertEqual(fin.ebit, [235.0, 245.0, 255.0, 265.0, 275.0, 285.0])

    def test_captive_finance_deposits_do_not_make_a_bank(self) -> None:
        # Harley-Davidson: deposits are 7% of total assets.
        self._assert_derived(self._usd(Deposits=[_inst("2024-12-31", 700.0)]))

    def test_deposits_no_longer_reported_do_not_make_a_bank(self) -> None:
        # GE: bank deposits last tagged in 2015.
        usd = self._usd(Deposits=[_inst("2015-12-31", 5000.0)])
        usd["Assets"] = usd["Assets"] + [_inst("2015-12-31", 10000.0)]
        self._assert_derived(usd)

    def test_managed_care_premiums_without_float_income_are_not_an_insurer(self) -> None:
        # UnitedHealth: premiums dominate revenue, investment income ~1%.
        self._assert_derived(self._usd(
            PremiumsEarnedNet=[_fy(y, 800.0 + 80 * i) for i, y in enumerate(_YEARS)],
            NetInvestmentIncome=[_fy(y, 10.0) for y in _YEARS],
        ))

    def test_small_policyholder_benefit_line_is_not_an_insurer(self) -> None:
        # Deere: a finance arm's small insurance line.
        self._assert_derived(self._usd(
            PolicyholderBenefitsAndClaimsIncurredNet=[_fy(y, 10.0) for y in _YEARS],
            NetInvestmentIncome=[_fy(y, 30.0) for y in _YEARS],
        ))


class CaptiveFinanceTests(unittest.TestCase):
    """Industrial groups that consolidate a finance and leasing arm.

    Flagged ``_financial_kind == "captive_finance"`` with a leading WARNING
    when a group with a product business (inventory) has finance receivables
    of at least 10% of total assets that, with operating-lease property, are
    a quarter of total assets or more, or when the latest year's loan and
    lease originations are 15-100% of revenue. Lessors, lenders, lending
    platforms, rental fleets, small finance lines and filers already
    classified as financial are not flagged. FY2024 revenue is 1,500, total
    assets 10,000 and inventory 800 throughout unless removed.
    """

    _PREFIX = "WARNING: EDGAR tags mark this company as a group with a captive finance arm ("
    _LEASES = ("the finance arm's debt, leases and receivables are consolidated (total debt "
               "includes the borrowing that funds its customer loans and leases, and D&A the "
               "depreciation of its lease fleet), so the FCFF DCF")
    _LOANS_ONLY = ("the finance arm's debt and receivables are consolidated (total debt "
                   "includes the borrowing that funds its customer loans), so the FCFF DCF")

    @staticmethod
    def _usd(**extra: list[dict]) -> dict:
        usd = _base_usd()
        usd["Assets"] = [_inst("2023-12-31", 9000.0), _inst("2024-12-31", 10000.0)]
        usd["InventoryNet"] = [_inst("2023-12-31", 700.0), _inst("2024-12-31", 800.0)]
        usd.update(extra)
        return usd

    def _assert_captive(self, usd: dict, evidence: str, leases: bool = True) -> tuple:
        fin, bs, notes = _parse(usd)
        self.assertEqual(fin._financial_kind, "captive_finance")
        self.assertEqual(notes[0].split("); ")[0], self._PREFIX + evidence, notes)
        self.assertIn(self._LEASES if leases else self._LOANS_ONLY, notes[0])
        if not leases:
            self.assertNotIn("lease", notes[0].split("); ")[1])
        return fin, bs, notes

    def _assert_not_captive(self, usd: dict, kind=None) -> list[str]:
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin._financial_kind, kind)
        self.assertFalse([n for n in notes if "captive finance" in n], notes)
        return notes

    def test_receivables_and_lease_fleet_mark_a_captive_finance_arm(self) -> None:
        # Ford: Ford Credit's receivables plus leased vehicles are 48% of assets.
        fin, _bs, _notes = self._assert_captive(self._usd(
            NotesAndLoansReceivableNetCurrent=[_inst("2024-12-31", 1600.0)],
            NotesAndLoansReceivableNetNoncurrent=[_inst("2024-12-31", 2200.0)],
            PropertySubjectToOrAvailableForOperatingLeaseNet=[_inst("2024-12-31", 1000.0)],
        ), "finance receivables and operating-lease property are 48% of total assets")
        # Operating income is reported, so EBIT is untouched.
        self.assertEqual(fin.ebit, [150.0, 160.0, 170.0, 180.0, 190.0, 200.0])

    def test_originations_mark_a_captive_whose_receivables_use_company_tags(self) -> None:
        # Deere / PACCAR: the receivables sit under company-specific elements,
        # but the cash-flow statement tags the year's originations.
        self._assert_captive(self._usd(
            PaymentsToAcquireFinanceReceivables=[_fy(y, 870.0) for y in _YEARS],
            PropertySubjectToOrAvailableForOperatingLeaseNet=[_inst("2024-12-31", 700.0)],
        ), "FY2024 loan and lease originations are 58% of revenue, operating-lease "
           "property 7% of total assets")

    def test_lease_fleet_purchases_net_of_proceeds_join_capex(self) -> None:
        # GM: GM Financial's lease-fleet purchases less the proceeds of ended
        # leases; the fleet shrank in FY2022 (proceeds above purchases).
        fin, _bs, notes = self._assert_captive(self._usd(
            PaymentsToAcquireFinanceReceivables=[_fy(y, 330.0) for y in _YEARS],
            PaymentsToAcquireLeasesHeldForInvestment=[_fy(y, 100.0) for y in _YEARS],
            ProceedsFromLeasesHeldForInvestment=[
                _fy(y, 130.0 if y == 2022 else 70.0) for y in _YEARS],
        ), "FY2024 loan and lease originations are 29% of revenue")
        self.assertEqual(fin.capex, [90.0, 90.0, 90.0, 30.0, 90.0, 90.0])
        self.assertIn(
            "capex for FY2019-2024 includes net purchases of vehicles and equipment for "
            "operating leases (PaymentsToAcquireLeasesHeldForInvestment less "
            "ProceedsFromLeasesHeldForInvestment), since D&A includes the lease fleet's "
            "depreciation", notes)

    def test_lease_fleet_capex_needs_a_reported_capex_year_and_never_goes_negative(self) -> None:
        usd = self._usd(
            PaymentsToAcquireLeasesHeldForInvestment=[_fy(y, 10.0) for y in _YEARS],
            ProceedsFromLeasesHeldForInvestment=[_fy(y, 90.0 if y == 2024 else 0.0)
                                                 for y in _YEARS],
        )
        usd["PaymentsToAcquirePropertyPlantAndEquipment"] = [
            _fy(y, 60.0) for y in _YEARS if y != 2019]
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin.capex, [0.0, 70.0, 70.0, 70.0, 70.0, 0.0])
        self.assertIn("capex not reported on EDGAR for FY2019; filled with 0.0", notes)

    def test_captive_warning_leads_the_interest_without_debt_warning(self) -> None:
        # Ford: no us-gaap debt tags, so the interest check fires too; the
        # classification stays first and the debt warning keeps its wording.
        usd = _without(self._usd(
            NotesAndLoansReceivableNetCurrent=[_inst("2024-12-31", 4000.0)],
        ), "LongTermDebtNoncurrent")
        usd["InterestExpense"] = [_fy(y, 60.0) for y in _YEARS]
        _fin, bs, notes = _parse(usd)
        self.assertEqual(bs.total_debt, 0.0)
        self.assertTrue(notes[0].startswith(self._PREFIX), notes)
        self.assertTrue(notes[1].startswith(
            "WARNING: FY2024 interest expense is 60 but no debt was found under the SEC "
            "debt tags this parser reads"), notes)

    def test_loans_only_captive_is_not_told_about_a_lease_fleet(self) -> None:
        # CarMax / CNH / Harley-Davidson: customer loans, no lease fleet tagged.
        self._assert_captive(self._usd(
            FinancingReceivableExcludingAccruedInterestAfterAllowanceForCreditLoss=[
                _inst("2024-12-31", 5300.0)],
        ), "finance receivables are 53% of total assets", leases=False)

    def test_operating_lease_income_keeps_the_lease_fleet_reason(self) -> None:
        # Caterpillar: the leased equipment sits in PP&E, but its operating-
        # lease income is tagged.
        self._assert_captive(self._usd(
            PaymentsToAcquireFinanceReceivables=[_fy(y, 340.0) for y in _YEARS],
            OperatingLeaseLeaseIncome=[_fy(y, 27.0) for y in _YEARS],
        ), "FY2024 loan and lease originations are 23% of revenue")

    def test_inventory_change_in_the_cash_flow_shows_a_product_business(self) -> None:
        # PACCAR / Marriott Vacations: the inventory balance uses a company-
        # specific element, but the cash flow tags the change in inventories.
        usd = _without(self._usd(
            PaymentsToAcquireFinanceReceivables=[_fy(y, 320.0) for y in _YEARS],
            IncreaseDecreaseInInventories=[_fy(y, -30.0) for y in _YEARS],
        ), "InventoryNet")
        self._assert_captive(usd, "FY2024 loan and lease originations are 21% of revenue",
                             leases=False)

    def test_originations_above_revenue_leave_the_receivables_evidence(self) -> None:
        # A revolving floorplan book turns over several times a year, so its
        # originations pass revenue; the receivables still mark the arm.
        self._assert_captive(self._usd(
            PaymentsToAcquireFinanceReceivables=[_fy(y, 3400.0) for y in _YEARS],
            NotesReceivableNet=[_inst("2024-12-31", 3000.0)],
        ), "finance receivables are 30% of total assets", leases=False)

    def test_lessor_is_not_a_captive_finance_arm(self) -> None:
        # AerCap: lease income is most of revenue, whatever its receivables.
        self._assert_not_captive(self._usd(
            NotesReceivableNet=[_inst("2024-12-31", 3000.0)],
            PropertySubjectToOrAvailableForOperatingLeaseNet=[_inst("2024-12-31", 5500.0)],
            OperatingLeaseLeaseIncome=[_fy(y, 1300.0) for y in _YEARS],
        ))

    def test_lease_fleet_without_lending_is_not_a_finance_arm(self) -> None:
        # Hertz / WillScot: a rental fleet and no finance receivables, with
        # no lease-income tag to identify the lessor.
        self._assert_not_captive(self._usd(
            PropertySubjectToOrAvailableForOperatingLeaseNet=[_inst("2024-12-31", 5500.0)],
        ))

    def test_rental_fleet_with_an_immaterial_note_is_not_a_finance_arm(self) -> None:
        # WillScot: a rental fleet at 55% of assets and a note receivable of
        # 0.1%; receivables below 10% of assets do not make a finance arm.
        self._assert_not_captive(self._usd(
            PropertySubjectToOrAvailableForOperatingLeaseNet=[_inst("2024-12-31", 5500.0)],
            NotesReceivableNet=[_inst("2024-12-31", 10.0)],
        ))

    def test_small_finance_line_is_not_a_captive(self) -> None:
        # Dell: financing receivables 16% and leased equipment 3% of assets;
        # originations 10% of revenue.
        self._assert_not_captive(self._usd(
            NotesAndLoansReceivableNetCurrent=[_inst("2024-12-31", 1000.0)],
            NotesAndLoansReceivableNetNoncurrent=[_inst("2024-12-31", 600.0)],
            PropertySubjectToOrAvailableForOperatingLeaseNet=[_inst("2024-12-31", 250.0)],
            PaymentsToAcquireFinanceReceivables=[_fy(y, 150.0) for y in _YEARS],
        ))

    def test_lender_is_not_a_captive(self) -> None:
        # OneMain / CarMax-like book, but interest at 20% of revenue: a lender.
        usd = self._usd(NotesReceivableNet=[_inst("2024-12-31", 8000.0)])
        usd["InterestExpense"] = [_fy(y, 300.0) for y in _YEARS]
        self._assert_not_captive(usd)

    def test_originations_above_revenue_are_not_evidence(self) -> None:
        # Short advances originated at twice revenue are a lender's volume;
        # the small book left on the balance sheet is not a finance arm.
        self._assert_not_captive(self._usd(
            PaymentsToAcquireFinanceReceivables=[_fy(y, 3400.0) for y in _YEARS],
            FinancingReceivableExcludingAccruedInterestAfterAllowanceForCreditLoss=[
                _inst("2024-12-31", 500.0)],
        ))

    def test_lending_platform_without_a_product_business_is_not_a_captive(self) -> None:
        # Upstart: loans 34% of assets, interest 4% of revenue, originations
        # barely tagged and no inventory, so nothing for an arm to finance.
        usd = _without(self._usd(
            NotesReceivableNet=[_inst("2024-12-31", 3400.0)],
            PaymentsToAcquireFinanceReceivables=[_fy(y, 12.0) for y in _YEARS],
        ), "InventoryNet")
        usd["InterestExpense"] = [_fy(y, 60.0) for y in _YEARS]
        self._assert_not_captive(usd)
        # Chime: advances originated at twice revenue, no product business.
        self._assert_not_captive(_without(self._usd(
            PaymentsToAcquireFinanceReceivables=[_fy(y, 3400.0) for y in _YEARS],
            FinancingReceivableExcludingAccruedInterestAfterAllowanceForCreditLoss=[
                _inst("2024-12-31", 3000.0)],
        ), "InventoryNet"))

    def test_receivables_no_longer_reported_do_not_count(self) -> None:
        # Deere: its receivable tags stopped in 2020 (company tags since).
        usd = self._usd(NotesReceivableNet=[_inst("2020-12-31", 4500.0)])
        usd["Assets"] = usd["Assets"] + [_inst("2020-12-31", 9000.0)]
        self._assert_not_captive(usd)

    def test_bank_is_not_reclassified_as_captive(self) -> None:
        notes = self._assert_not_captive(self._usd(
            Deposits=[_inst("2024-12-31", 5400.0)],
            FinancingReceivableExcludingAccruedInterestAfterAllowanceForCreditLoss=[
                _inst("2024-12-31", 6000.0)],
        ), kind="bank")
        self.assertTrue(notes[0].startswith("WARNING: EDGAR tags mark this company as a bank"))


class LessorTests(unittest.TestCase):
    """Debt-funded operating lessors.

    Flagged ``_financial_kind == "lessor"`` with a leading WARNING when a
    filer that is not a bank, insurer, BDC or REIT has lease income of at
    least 50% of revenue and interest expense (the largest interest figure
    tagged for the year) of at least 10% of it in the latest fiscal year.
    Their EBIT is still derived as pretax income + interest expense, as for an
    equity REIT, for display and the reference models: the fleet interest is
    an operating cost of the leasing business, which is why the engine keeps
    their FCFF DCF and FCFE out of the blend. FY2024 revenue is 1,500 and
    pretax income 180.
    """

    _PREFIX = "WARNING: EDGAR tags mark this company as a debt-funded operating lessor ("
    _REASON = ("); the assets it leases out are bought with debt, whose interest is an "
               "operating cost of the leasing business, and its purchases of them (its growth "
               "investment) are often tagged outside capex, so the FCFF DCF and FCFE do not "
               "fit it and the DDM and comps are the better guides")

    @staticmethod
    def _usd(lease: float = 1300.0, interest: float = 330.0, **extra: list[dict]) -> dict:
        usd = _without(_base_usd(), "OperatingIncomeLoss")
        usd["Assets"] = [_inst("2023-12-31", 9000.0), _inst("2024-12-31", 10000.0)]
        usd["OperatingLeaseLeaseIncome"] = [_fy(y, lease) for y in _YEARS]
        usd["InterestExpense"] = [_fy(y, interest) for y in _YEARS]
        usd.update(extra)
        return usd

    def _assert_lessor(self, usd: dict, evidence: str) -> tuple:
        fin, bs, notes = _parse(usd)
        self.assertEqual(fin._financial_kind, "lessor")
        self.assertEqual(notes[0], self._PREFIX + evidence + self._REASON, notes)
        self.assertFalse([n for n in notes if "captive finance" in n], notes)
        return fin, bs, notes

    def _assert_not_lessor(self, usd: dict, kind=None) -> list[str]:
        fin, _bs, notes = _parse(usd)
        self.assertEqual(fin._financial_kind, kind)
        self.assertFalse([n for n in notes if "operating lessor" in n], notes)
        return notes

    def test_aircraft_lessor_is_flagged_and_its_ebit_derived(self) -> None:
        # AerCap: lease income 87% and interest 22% of revenue, no operating-
        # income subtotal. EBIT = pretax + interest (before the fleet interest,
        # for display and the reference DCF) rather than 0 as for a lender.
        fin, _bs, notes = self._assert_lessor(
            self._usd(), "FY2024 lease income is 87% and interest expense 22% of revenue")
        self.assertEqual(fin.ebit, [460.0, 470.0, 480.0, 490.0, 500.0, 510.0])
        self.assertEqual(fin.interest_expense, [330.0] * 6)
        self.assertIn("EBIT (OperatingIncomeLoss) not reported on EDGAR for FY2019-2024; "
                      "derived as pretax income + interest expense", notes)
        self.assertFalse([n for n in notes if "as for a lender" in n], notes)

    def test_vehicle_interest_tagged_as_operating_interest_counts(self) -> None:
        # Hertz: its vehicle interest is tagged only as InterestExpenseOperating,
        # which the interest-expense series does not read.
        usd = _without(self._usd(lease=1450.0), "InterestExpense")
        usd["InterestExpenseOperating"] = [_fy(y, 195.0) for y in _YEARS]
        fin, _bs, _notes = self._assert_lessor(
            usd, "FY2024 lease income is 97% and interest expense 13% of revenue")
        self.assertEqual(fin.interest_expense, [0.0] * 6)

    def test_interest_paid_above_the_corporate_interest_line_counts(self) -> None:
        # Avis Budget: the tagged interest expense (3.6% of revenue) is the
        # corporate debt's; interest paid (11%) includes the vehicle debt's.
        usd = _without(self._usd(lease=1470.0), "InterestExpense")
        usd["InterestExpenseDebt"] = [_fy(y, 54.0) for y in _YEARS]
        usd["InterestPaidNet"] = [_fy(y, 167.0) for y in _YEARS]
        fin, _bs, _notes = self._assert_lessor(
            usd, "FY2024 lease income is 98% and interest expense 11% of revenue")
        self.assertEqual(fin.interest_expense, [54.0] * 6)

    def test_equipment_rental_company_with_modest_interest_is_not_a_lessor(self) -> None:
        # United Rentals / Herc: rental revenue, but interest is 4% of revenue.
        notes = self._assert_not_lessor(self._usd(lease=1200.0, interest=60.0))
        self.assertFalse([n for n in notes if n.startswith("WARNING")], notes)

    def test_lender_scale_interest_without_lease_income_is_not_a_lessor(self) -> None:
        # Credit Acceptance-like: interest 20% of revenue but no lease income;
        # still a lender for the EBIT derivation.
        notes = self._assert_not_lessor(self._usd(lease=0.0, interest=300.0))
        self.assertTrue(any("as for a lender, broker or mortgage REIT" in n for n in notes),
                        notes)

    def test_lease_income_below_half_of_revenue_is_not_a_lessor(self) -> None:
        # Ryder: lease income 31% of revenue.
        self._assert_not_lessor(self._usd(lease=460.0))

    def test_reit_stays_a_reit(self) -> None:
        notes = self._assert_not_lessor(self._usd(
            RealEstateInvestmentPropertyNet=[_inst("2024-12-31", 7000.0)]), kind="reit")
        self.assertTrue(notes[0].startswith("WARNING: EDGAR tags mark this company as a REIT"))

    def test_tower_reit_known_by_its_schedule_iii_is_a_reit(self) -> None:
        # American Tower: lease income and interest shares of a lessor, but no
        # investment-property tag; its Schedule III real estate is 47% of
        # total assets, so it is a REIT, as its industry says.
        notes = self._assert_not_lessor(self._usd(
            RealEstateGrossAtCarryingValue=[_inst("2024-12-31", 4700.0)]), kind="reit")
        self.assertTrue(notes[0].startswith(
            "WARNING: EDGAR tags mark this company as a REIT (real estate at gross "
            "carrying value (its Schedule III) is 47% of total assets); its growth comes "
            "from buying property"), notes)

    def test_stale_or_small_schedule_iii_leaves_a_lessor(self) -> None:
        # Hertz's only Schedule III figure is from 2020; a small real-estate
        # schedule (30% of assets) is not a REIT's either.
        stale = self._usd(RealEstateGrossAtCarryingValue=[_inst("2020-12-31", 6000.0)])
        stale["Assets"] = stale["Assets"] + [_inst("2020-12-31", 9000.0)]
        self._assert_lessor(stale, "FY2024 lease income is 87% and interest expense 22% "
                                   "of revenue")
        self._assert_lessor(
            self._usd(RealEstateGrossAtCarryingValue=[_inst("2024-12-31", 3000.0)]),
            "FY2024 lease income is 87% and interest expense 22% of revenue")

    def test_lessor_warning_leads_the_interest_without_debt_warning(self) -> None:
        # The fleet debt under company-specific tags: the interest check fires
        # too, second, with its wording kept for the debt backfill.
        usd = _without(self._usd(), "LongTermDebtNoncurrent")
        _fin, bs, notes = self._assert_lessor(
            usd, "FY2024 lease income is 87% and interest expense 22% of revenue")
        self.assertEqual(bs.total_debt, 0.0)
        self.assertTrue(notes[1].startswith(
            "WARNING: FY2024 interest expense is 330 but no debt was found under the SEC "
            "debt tags this parser reads"), notes)
        self.assertEqual(_missing_debt_warning(notes), 1)


class ReportingCurrencyTests(unittest.TestCase):
    """US-GAAP filers that report in another currency go to the fallback."""

    @staticmethod
    def _foreign(ccy: str, usd_tags: dict, ccy_tags: dict) -> dict:
        facts = _company(usd_tags)
        gaap = facts["facts"]["us-gaap"]
        for tag, entries in ccy_tags.items():
            gaap.setdefault(tag, {"units": {}})["units"][ccy] = entries
        return facts

    def _raises(self, facts: dict, text: str) -> None:
        with self.assertRaises(DataError) as ctx:
            _client(facts).get_annual_financials("FIX")
        self.assertIn(text, str(ctx.exception))

    def test_revenue_in_cny_with_usd_convenience_figures_is_rejected(self) -> None:
        # Alibaba / JD.com / PDD / Baidu: CNY statements, and each 20-F adds USD
        # translations of a few latest-year figures (no D&A or capex).
        cny = {tag: [_fy(y, 7.0 * v["val"], form="20-F") for y, v in zip(_YEARS, entries)]
               for tag, entries in _base_usd().items() if "start" in entries[0]}
        usd = {"Revenues": [_fy(2024, 1500.0, form="20-F")],
               "NetIncomeLoss": [_fy(2024, 150.0, form="20-F")],
               "CashAndCashEquivalentsAtCarryingValue": [_inst("2024-12-31", 200.0)]}
        self._raises(self._foreign("CNY", usd, cny),
                     "are reported in CNY (revenue is tagged in CNY); only a few figures "
                     "are also tagged in USD as convenience translations")

    def test_mostly_foreign_monetary_facts_are_rejected(self) -> None:
        # Canadian National: CAD statements with no revenue tag this parser
        # reads, and a couple of USD-denominated notes.
        cad = {f"Line{i}": [_inst("2024-12-31", 1.0 + i)] for i in range(5)}
        usd = {"LongTermDebtNoncurrent": [_inst("2024-12-31", 480.0)]}
        self._raises(self._foreign("CAD", usd, cad),
                     "are reported in CAD (83% of its monetary facts dated within a year "
                     "of 2024-12-31 are in CAD); only a few figures are tagged in USD, so "
                     "the USD statements would be incomplete")
        # A mistyped far-future USD fact does not move the window off them.
        usd["RestructuringAndRelatedCostExpectedCost"] = [
            _inst("2199-12-31", 5.0, filed="2024-12-31")]
        self._raises(self._foreign("CAD", usd, cad),
                     "83% of its monetary facts dated within a year of 2024-12-31")

    def test_statements_without_usd_facts_are_not_called_translations(self) -> None:
        # No USD figures in the latest year at all.
        cad = {f"Line{i}": [_inst("2024-12-31", 1.0 + i)] for i in range(5)}
        facts = self._foreign("CAD", {}, cad)
        with self.assertRaises(DataError) as ctx:
            _client(facts).get_annual_financials("FIX")
        self.assertTrue(str(ctx.exception).endswith(
            "are reported in CAD (100% of its monetary facts dated within a year of "
            "2024-12-31 are in CAD); its recent statements are not tagged in USD, so they "
            "are not used"), str(ctx.exception))
        # Canadian National: 3 USD commercial-paper facts among ~640 in CAD are
        # rounded down to 99%, never shown as 100%.
        cad = {f"Line{i}": [_inst("2024-12-31", 1.0 + i)] for i in range(199)}
        facts = self._foreign("CAD", {"CommercialPaper": [_inst("2024-12-31", 90.0)]}, cad)
        self._raises(facts, "(99% of its monetary facts dated within a year of 2024-12-31 "
                            "are in CAD); only a few figures are tagged in USD")

    def test_usd_filer_with_a_few_foreign_facts_is_parsed(self) -> None:
        # Coca-Cola: one euro-denominated note among USD statements.
        facts = self._foreign("EUR", _base_usd(),
                              {"DebtInstrumentFaceAmount": [_inst("2024-12-31", 500.0)]})
        fin, _bs, _cik, _name = _client(facts).get_annual_financials("FIX")
        self.assertEqual(fin.revenue[-1], 1500.0)

    def test_future_dated_foreign_fact_does_not_take_over_the_window(self) -> None:
        # One euro fact mistyped as ending 2199-12-31 (Oracle tags a USD one),
        # reported in a 2010 filing: it is not the latest period.
        facts = self._foreign("EUR", _base_usd(), {
            "RestructuringAndRelatedCostExpectedCost": [
                _inst("2199-12-31", 50.0, filed="2010-09-29")]})
        fin, _bs, _cik, _name = _client(facts).get_annual_financials("FIX")
        self.assertEqual(fin.revenue[-1], 1500.0)

    def test_filer_that_moved_to_usd_reporting_is_parsed(self) -> None:
        # Revenue in CAD up to FY2020, USD from FY2021 on.
        usd = _base_usd()
        usd["Revenues"] = [e for e in usd["Revenues"] if e["end"] >= "2021"]
        facts = self._foreign("CAD", usd, {"Revenues": [_fy(y, 900.0) for y in (2019, 2020)]})
        fin, _bs, _cik, _name = _client(facts).get_annual_financials("FIX")
        self.assertEqual(fin.fiscal_years, [2021, 2022, 2023, 2024])


class HybridProviderEdgarTests(unittest.TestCase):
    def _provider(self, usd: dict, market: MarketData, fx=None) -> HybridProvider:
        client = mock.Mock()
        client.get_market_data.return_value = market
        client.get_fx_rate.return_value = fx
        return HybridProvider(edgar=_client(_company(usd)), market=client)

    @staticmethod
    def _market(**kw) -> MarketData:
        base = dict(ticker="FIX", name="Fixture", currency="USD", price=10.0,
                    shares_outstanding=400.0, market_cap=4000.0, dividend_per_share=0.05)
        base.update(kw)
        return MarketData(**base)

    def test_edgar_parsing_notes_reach_company_source_notes(self) -> None:
        usd = _without(_base_usd(), "OperatingIncomeLoss", "DepreciationDepletionAndAmortization")
        cd = self._provider(usd, self._market()).get_company_data("FIX")
        self.assertEqual(cd.source_notes[0], "Fundamentals: SEC EDGAR (CIK 0000000001)")
        self.assertIn("D&A unavailable on EDGAR; filled with 0.0", cd.source_notes)
        self.assertTrue(any("derived as pretax income + interest expense" in n
                            for n in cd.source_notes), cd.source_notes)

    def test_financial_filer_warning_leads_the_source_notes(self) -> None:
        usd = _without(_base_usd(), "OperatingIncomeLoss")
        usd["Assets"] = [_inst("2024-12-31", 10000.0)]
        usd["Deposits"] = [_inst("2024-12-31", 6000.0)]
        cd = self._provider(usd, self._market()).get_company_data("FIX")
        self.assertTrue(cd.source_notes[1].startswith(
            "WARNING: EDGAR tags mark this company as a bank"), cd.source_notes)
        self.assertEqual(cd.financials.ebit, [0.0] * 6)
        self.assertEqual(getattr(cd.financials, "_financial_kind", None), "bank")

    def test_captive_finance_kind_and_warning_reach_company_data(self) -> None:
        usd = _base_usd()
        usd["Assets"] = [_inst("2024-12-31", 10000.0)]
        usd["NotesReceivableNet"] = [_inst("2024-12-31", 4000.0)]
        usd["InventoryNet"] = [_inst("2024-12-31", 800.0)]
        cd = self._provider(usd, self._market()).get_company_data("FIX")
        self.assertEqual(getattr(cd.financials, "_financial_kind", None), "captive_finance")
        self.assertTrue(cd.source_notes[1].startswith(
            "WARNING: EDGAR tags mark this company as a group with a captive finance arm "
            "(finance receivables are 40% of total assets)"), cd.source_notes)

    def test_lessor_kind_and_warning_reach_company_data(self) -> None:
        usd = LessorTests._usd()
        cd = self._provider(usd, self._market()).get_company_data("FIX")
        self.assertEqual(getattr(cd.financials, "_financial_kind", None), "lessor")
        self.assertTrue(cd.source_notes[1].startswith(
            "WARNING: EDGAR tags mark this company as a debt-funded operating lessor "
            "(FY2024 lease income is 87% and interest expense 22% of revenue)"), cd.source_notes)
        # Exactly the lessor kind: the models then leave its FCFF DCF and FCFE
        # out of the blend and value it on equity multiples.
        self.assertEqual(financial_institution_detail(cd),
                         ("operating lessor per its filings", "lessor"))

    def test_statements_in_another_currency_use_the_yfinance_fallback(self) -> None:
        # Alibaba-like: CNY statements; the fallback converts them itself.
        facts = _company({"Revenues": [_fy(2024, 1500.0, form="20-F")]})
        facts["facts"]["us-gaap"]["Revenues"]["units"]["CNY"] = [
            _fy(y, 10500.0, form="20-F") for y in _YEARS]
        fallback = _parse(_base_usd())[:2]
        market = mock.Mock()
        market.get_market_data.return_value = self._market()
        market.get_annual_financials_fallback.return_value = fallback
        client = _client(facts)
        cd = HybridProvider(edgar=client, market=market).get_company_data("FIX")
        self.assertIsNone(cd.cik)
        self.assertIs(cd.financials, fallback[0])
        self.assertTrue(any(n.startswith("EDGAR unavailable: EDGAR us-gaap statements for 'FIX' "
                                         "(CIK 0000000001) are reported in CNY")
                            for n in cd.source_notes), cd.source_notes)
        self.assertIn("Fundamentals: yfinance fallback", cd.source_notes)

    def test_missing_market_shares_cap_and_dps_are_backfilled_from_edgar(self) -> None:
        market = self._market(shares_outstanding=0.0, market_cap=0.0, dividend_per_share=None)
        cd = self._provider(_base_usd(), market).get_company_data("FIX")
        self.assertEqual(cd.market.shares_outstanding, 400.0)
        self.assertEqual(cd.market.market_cap, 4000.0)
        self.assertAlmostEqual(cd.market.dividend_per_share, 20.0 / 400.0)
        self.assertEqual(market.market_cap, 0.0)  # caller's object untouched
        joined = " | ".join(cd.source_notes)
        self.assertIn("FY2024 diluted weighted-average shares from EDGAR", joined)
        self.assertIn("market cap unavailable from Yahoo; set to price x shares", joined)
        self.assertIn("dividend per share unavailable from Yahoo", joined)

    def test_reported_zero_dividend_is_kept(self) -> None:
        cd = self._provider(_base_usd(), self._market(dividend_per_share=0.0)).get_company_data("FIX")
        self.assertEqual(cd.market.dividend_per_share, 0.0)

    def test_usd_statements_are_converted_for_a_non_usd_quote(self) -> None:
        market = self._market(currency="CAD", price=13.0, market_cap=5200.0)
        cd = self._provider(_base_usd(), market, fx=(1.3, "USDCAD=X")).get_company_data("FIX")
        self.assertAlmostEqual(cd.financials.revenue[-1], 1500.0 * 1.3)
        self.assertAlmostEqual(cd.balance_sheet.total_debt, 480.0 * 1.3)
        self.assertEqual(cd.financials.diluted_shares[-1], 400.0)
        self.assertTrue(any("converted from USD to CAD at spot 1.3" in n
                            for n in cd.source_notes), cd.source_notes)

    def test_missing_fx_rate_stops_unit_mixed_valuation(self) -> None:
        market = self._market(currency="CAD", price=13.0, market_cap=5200.0)
        with self.assertRaisesRegex(DataError, "no USD->CAD exchange rate.*retry"):
            self._provider(_base_usd(), market, fx=None).get_company_data("FIX")


class FilingsListTests(unittest.TestCase):
    """backend.filings.list_filings: prospectus noise must not crowd out reports."""

    @staticmethod
    def _submissions(forms: list[str]) -> dict:
        n = len(forms)
        return {"filings": {"recent": {
            "form": forms,
            "filingDate": [f"2026-09-{28 - i % 28:02d}" for i in range(n)],
            "reportDate": [""] * n,
            "accessionNumber": [f"0000000001-26-{i:06d}" for i in range(n)],
            "primaryDocument": [f"doc{i}.htm" for i in range(n)],
            "primaryDocDescription": [""] * n,
        }}}

    def _list(self, forms: list[str], limit: int = 40) -> list[str]:
        from backend import filings

        resp = mock.Mock(status_code=200)
        resp.json.return_value = self._submissions(forms)
        with mock.patch.object(filings._edgar, "resolve_cik",
                               return_value=("0000000001", "Fixture Bank")), \
                mock.patch.object(filings.requests, "get", return_value=resp) as get:
            out = filings.list_filings("FIX", limit=limit)
        self.assertTrue(get.call_args.args[0].endswith("CIK0000000001.json"))
        return [f["form"] for f in out["filings"]]

    def test_prospectus_supplements_do_not_crowd_out_the_annual_report(self) -> None:
        # A bank issuer: dozens of 424B2 / FWP structured-note filings a week.
        forms = ["424B2", "FWP"] * 60 + ["10-Q", "8-K"] + ["424B2"] * 30 + ["10-K", "DEF 14A"]
        self.assertEqual(self._list(forms), ["10-Q", "8-K", "10-K", "DEF 14A"])

    def test_reports_amendments_and_registrations_are_kept(self) -> None:
        forms = ["10-K/A", "10-Q", "8-K", "6-K", "20-F", "40-F", "S-1/A", "424B4",
                 "424B3", "FWP", "SC 13G", "4"]
        self.assertEqual(self._list(forms),
                         ["10-K/A", "10-Q", "8-K", "6-K", "20-F", "40-F", "S-1/A"])

    def test_limit_counts_only_the_listed_forms(self) -> None:
        forms = ["424B2"] * 10 + ["8-K"] * 5
        self.assertEqual(self._list(forms, limit=3), ["8-K"] * 3)


class EdgarBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.client = EdgarClient(user_agent="audit tests@example.com")

    def test_balance_sheet_ignores_future_and_malformed_instant_facts(self):
        usd = _base_usd()
        usd["StockholdersEquity"] += [
            _inst("2199-12-31", 99e12, filed="2025-02-01"),
            _inst("2025-13-40", 99e12, filed="2026-02-01"),
        ]
        bs = self.client._build_balance_sheet(_company(usd), [])
        self.assertEqual(bs.as_of, "2024-12-31")
        self.assertEqual(bs.total_equity, 1000.0)

    def test_flow_cannot_end_after_its_filing(self):
        fact = {"start": "2030-01-01", "end": "2030-12-31", "filed": "2025-02-01",
                "val": 100, "form": "10-K", "fp": "FY"}
        self.assertIsNone(self.client._annual_fact(fact))

    def test_gross_equity_excludes_noncontrolling_interest(self):
        usd = _base_usd()
        del usd["StockholdersEquity"]
        usd["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"] = [
            _inst("2024-12-31", 1200)]
        usd["MinorityInterest"] = [_inst("2024-12-31", 200)]
        bs = self.client._build_balance_sheet(_company(usd), [])
        self.assertEqual(bs.total_equity, 1000.0)
        self.assertEqual(bs.minority_interest, 200.0)

    def test_parent_equity_excludes_preferred_book_value_only(self):
        usd = _base_usd()
        usd["PreferredStockValue"] = [_inst("2024-12-31", 100)]
        usd["PreferredStockLiquidationPreferenceValue"] = [_inst("2024-12-31", 150)]
        bs = self.client._build_balance_sheet(_company(usd), [])
        self.assertEqual(bs.total_equity, 900.0)
        self.assertEqual(bs.preferred_equity, 100.0)
        del usd["PreferredStockValue"]
        bs = self.client._build_balance_sheet(_company(usd), [])
        self.assertEqual(bs.total_equity, 1000.0)  # a liquidation claim is not book value
        self.assertEqual(bs.preferred_equity, 150.0)

    def test_preferred_capital_includes_additional_paid_in_capital(self):
        usd = _base_usd()
        usd["PreferredStockValue"] = [_inst("2024-12-31", 1)]
        usd["PreferredStockIncludingAdditionalPaidInCapitalNetOfDiscount"] = [
            _inst("2024-12-31", 100)]
        bs = self.client._build_balance_sheet(_company(usd), [])
        self.assertEqual(bs.total_equity, 900.0)
        self.assertEqual(bs.preferred_equity, 100.0)

    def test_common_income_precedes_parent_and_consolidated_income(self):
        usd = _base_usd()
        usd["NetIncomeLossAvailableToCommonStockholdersBasic"] = [_fy(2024, 90)]
        usd["ProfitLoss"] = [_fy(2024, 160)]
        usd["NetIncomeLossAttributableToNoncontrollingInterest"] = [_fy(2024, 10)]
        usd["PreferredStockDividendsAndOtherAdjustments"] = [_fy(2024, 5)]
        fin, _bs, _notes = _parse(usd)
        self.assertEqual(fin.net_income[-1], 90)
        self.assertTrue(fin._income_attribution_adjusted)

    def test_parent_income_only_subtracts_preferred_adjustments(self):
        facts = _facts(NetIncomeLoss=[_fy(2024, 100)],
                       NetIncomeLossAttributableToNoncontrollingInterest=[_fy(2024, 20)],
                       PreferredStockDividendsAndOtherAdjustments=[_fy(2024, 5)])
        self.assertEqual(self.client._common_net_income(facts, []), {2024: 95})

    def test_consolidated_income_subtracts_signed_minority_earnings(self):
        for minority, expected in ((20, 75), (-20, 115)):
            with self.subTest(minority=minority):
                facts = _facts(ProfitLoss=[_fy(2024, 100)],
                               NetIncomeLossAttributableToNoncontrollingInterest=[_fy(2024, minority)],
                               PreferredStockDividendsAndOtherAdjustments=[_fy(2024, 5)])
                self.assertEqual(self.client._common_net_income(facts, []), {2024: expected})

    def test_unresolved_gross_income_prevents_consolidated_profit_rebuild(self):
        facts = _facts(ProfitLoss=[_fy(2024, 100)])
        common = self.client._common_net_income(facts, [])
        self.assertTrue(self.client._income_attribution_adjusted(facts, common, [2024]))

    def test_fx_scaling_preserves_common_income_attribution(self):
        from equity_valuation.data.market import scale_fundamentals

        usd = _base_usd()
        usd["NetIncomeLossAvailableToCommonStockholdersBasic"] = [_fy(2024, 90)]
        fin, bs, _notes = _parse(usd)
        converted, _bs = scale_fundamentals(fin, bs, 1.3)
        self.assertEqual(converted.net_income[-1], 117)
        self.assertTrue(converted._income_attribution_adjusted)

class FilingsBoundaryTests(unittest.TestCase):
    def test_line_start_checks_entire_prefix(self):
        text = "Cross reference:  Item 1A. Risk Factors"
        self.assertFalse(filings._at_line_start(text, text.index("Item")))
        text = "Prior line\n    Item 1A. Risk Factors"
        self.assertTrue(filings._at_line_start(text, text.index("Item")))

    def test_sparse_optional_submissions_columns_do_not_crash(self):
        recent = {"form": ["10-K", "8-K", "10-Q"],
                  "accessionNumber": ["0000000001-25-000001", "0000000001-25-000002"],
                  "primaryDocument": ["annual.htm", "current.htm"],
                  "filingDate": ["2025-02-01"]}
        response = mock.Mock(status_code=200)
        response.json.return_value = {"filings": {"recent": recent}}
        with mock.patch.object(filings._edgar, "resolve_cik", return_value=("0000000001", "Test")), \
                mock.patch.object(filings.requests, "get", return_value=response):
            result = filings.list_filings("TEST")
        self.assertEqual(len(result["filings"]), 2)
        self.assertEqual(result["filings"][1]["filed"], "")
        self.assertEqual(result["filings"][0]["description"], "")

    def test_material_cap_counts_section_headings(self):
        sections = {"business": "b" * 20, "risk_factors": "r" * 20, "mdna": "m" * 20}
        with mock.patch.object(filings, "fetch_filing_text", return_value="source"), \
                mock.patch.object(filings, "extract_sections", return_value=sections), \
                mock.patch.object(filings, "_TOTAL_CAP", 140):
            material, meta = filings.build_filing_material("T", "10-K", "2025-01-01", "acc", "doc")
        self.assertLessEqual(len(material), 140)
        self.assertEqual(meta["material_chars"], len(material))

    def test_self_closing_skip_tag_does_not_hide_filing(self):
        self.assertEqual(filings._html_to_text('<head/><p>Operating results</p>'), "Operating results")


if __name__ == "__main__":
    unittest.main()
