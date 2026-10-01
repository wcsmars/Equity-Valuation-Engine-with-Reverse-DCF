"""SEC EDGAR fundamentals provider.

Pulls normalized annual financials from the public, key-less SEC `data.sec.gov`
XBRL JSON API and maps the raw us-gaap facts into the package's canonical
`AnnualFinancials` / `BalanceSheetSnapshot` dataclasses.

Two HTTP endpoints are used (both require a descriptive `User-Agent` header or
the SEC returns HTTP 403):
  * ticker -> CIK directory: https://www.sec.gov/files/company_tickers.json
  * company facts:           https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json

Design conventions honored from schemas.py:
  * All amounts are absolute reporting-currency units (the USD facts are already
    absolute, so no scaling is applied).
  * `capex`, `dep_amort`, `interest_expense`, `dividends_paid` are stored as
    POSITIVE magnitudes regardless of XBRL sign. `tax_expense` keeps the XBRL
    sign of IncomeTaxExpenseBenefit (expense positive, a tax BENEFIT negative).
  * Annual series are ordered OLDEST -> NEWEST; the last element is most recent
    and aligns with the `BalanceSheetSnapshot`.

Only `revenue` or `net_income` being entirely unavailable (or years out of date
relative to the other), or statements reported in a currency other than USD
(any USD facts are then a partial set of convenience translations or
USD-denominated items), is fatal (raises `DataError`); every other gap degrades
gracefully (zeros or a derived value + a source note). The notes are attached
to the returned `AnnualFinancials` as ``_source_notes`` and the hybrid provider
copies them into `CompanyData.source_notes`. Banks, insurers, BDCs, REITs,
debt-funded operating lessors and industrial groups that consolidate a
captive finance arm are recognised from their own tags and marked as
``_financial_kind`` ("bank", "insurer", "bdc", "reit", "lessor",
"captive_finance"; None otherwise), with a leading WARNING note.
"""

from __future__ import annotations

import time
from itertools import combinations
from typing import Optional

import requests

from .. import config
from ..schemas import AnnualFinancials, BalanceSheetSnapshot
from ..utils import is_num
from .base import DataError


# --------------------------------------------------------------------------- #
#  Endpoints & parsing constants
# --------------------------------------------------------------------------- #
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Annual filing forms we accept for FLOW (income/cash-flow) frames. Amendments
# (/A) carry restated XBRL and win through the latest-`filed` rule; 40-F is the
# Canadian MJDS annual report.
_ANNUAL_FORMS = ("10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A")

# A "full-year" period: end - start measured in days. We allow a generous band
# to absorb 52/53-week fiscal calendars and minor reporting drift.
_MIN_PERIOD_DAYS = 330
_MAX_PERIOD_DAYS = 400

# 52/53-week years that end on the weekend nearest Dec 31 sometimes end on
# Jan 1-3. A period ending this early in January is labelled with the PRIOR
# year, so it cannot collide with the next fiscal year ending in late December.
_EARLY_JANUARY_DAYS = 14

# Balance-sheet items are read at ONE snapshot date. A tag last reported more
# than this many days before the snapshot is treated as no longer outstanding
# (a line the company stopped reporting); one reported within the window (e.g.
# only in the last 10-K while the snapshot is a later 10-Q) is still used.
_INSTANT_GRACE_DAYS = 300

# How many of the most recent fiscal years to retain in the aligned series.
_MAX_YEARS = 8
_MIN_YEARS = 5  # informational target; we keep whatever (>=1) is available

# Tag-fallback lists: try each in order, first present wins (D&A and capex
# excepted, see `_TAGS_DA` and `_TAGS_CAPEX`).
# `Revenues` is the income-statement total. The ASC 606 tag covers contract
# revenue only (no lease, interest or insurance revenue), so it just backfills
# periods where the total is not tagged.
_TAGS_REVENUE = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
)
_TAGS_EBIT = ("OperatingIncomeLoss",)
# Total costs of sales and operating expenses; EBIT fallback = revenue - this.
_TAGS_COSTS_AND_EXPENSES = ("CostsAndExpenses",)
_TAG_COMMON_INCOME = "NetIncomeLossAvailableToCommonStockholdersBasic"
_TAG_NCI_INCOME = "NetIncomeLossAttributableToNoncontrollingInterest"
_TAGS_PREFERRED_INCOME_ADJUSTMENTS = (
    "PreferredStockDividendsAndOtherAdjustments",
    "PreferredStockDividendsIncomeStatementImpact",
)
# D&A: each of these tags is either the filer's total D&A or one part of it,
# and which one differs by filer (McDonald's DepreciationDepletionAndAmortization
# is 0.46B against a 2.20B DepreciationAndAmortization total; United Rentals
# tags its 2.67B rental-fleet depreciation only as cost-of-sales D&A; Home
# Depot's cost-of-sales D&A tag holds its cash-flow total). So each fiscal
# year takes the LARGEST figure any of them reports, the best available lower bound
# on the total (see `_largest_flow_by_fy`); they are never added, since a part
# and the total would then be counted twice. Figures within
# `_AMOUNT_MATCH_TOLERANCE` of each other keep the earlier tag, and a larger
# figure replaces the one held only if it was filed no earlier: a tag the
# filer stopped using keeps its last figure, which can predate a restatement
# (GE's FY2019-2020 Depreciation, last filed in 2021, still includes the
# aircraft-leasing arm later moved to discontinued operations, while its D&A
# tag was restated without it, in the filings that restated revenue and capex).
_TAGS_DA = (
    "DepreciationDepletionAndAmortization",
    "DepreciationAmortizationAndAccretionNet",
    "DepreciationAndAmortization",
    "DepreciationNonproduction",
    # PP&E depreciation only (no amortization).
    "Depreciation",
    "CostOfGoodsAndServicesSoldDepreciationAndAmortization",
)
# Capex: either tag can hold the filer's total capex, the other a part of it
# (United Rentals tags only its FY2017-2020 non-rental capex, 0.12-0.22B, as
# PaymentsToAcquirePropertyPlantAndEquipment and the 1.16-2.35B total, rental
# fleet included, as PaymentsToAcquireProductiveAssets; Trinity, Oshkosh and
# PACCAR likewise leave their lease fleets out of the first tag). So, as with
# D&A, each fiscal year takes the LARGER figure, under the same tolerance and
# filed-date guard (see `_TAGS_DA` and `_largest_flow_by_fy`), and the two are
# never added. Unlike D&A, the second tag replaces the first only for a filer
# whose second tag is larger in every year both report, as a total is: Copart
# tags its segment note's capex "including acquisitions" there (the same as
# its PP&E purchases in years without an acquisition), and Schwab a figure
# that is smaller in some years, so neither is read as the total.
_TAGS_CAPEX = (
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
)
# Payments for "other" PP&E: Eli Lilly's whole capex line, but for D.R. Horton
# a separate line (rental properties) that adds to its PP&E purchases and is
# larger in some years, smaller in others. So it only fills the years neither
# of `_TAGS_CAPEX` reports.
_TAG_OTHER_PPE = "PaymentsToAcquireOtherPropertyPlantAndEquipment"
# A generic line for "other" productive assets: Verizon's whole capex (17B a
# year), but for other filers only a part of it (Robinhood tags 15M there and
# its 39M of capitalised software under another element; its capex on Yahoo,
# 54M, is both). So it only fills the years none of the tags above reports,
# and those years are named in an info note (see
# `_OTHER_PRODUCTIVE_ASSETS_NOTE`).
_TAG_OTHER_PRODUCTIVE_ASSETS = "PaymentsToAcquireOtherProductiveAssets"
# Start of the info note naming the years whose capex comes only from
# `_TAG_OTHER_PRODUCTIVE_ASSETS`. Other capex notes start with "capex for
# {years}" too, so a reader of the notes should match the tag name.
_OTHER_PRODUCTIVE_ASSETS_NOTE = "capex for {years} read from " + _TAG_OTHER_PRODUCTIVE_ASSETS
# A finance arm's purchases of vehicles and equipment for operating leases,
# and the proceeds when those leases end (GM Financial). The lease fleet's
# depreciation is inside D&A, so its net purchases are added to capex, except
# in a year whose capex is a total that exceeds the first capex tag by exactly
# these purchases: it already includes them, so they are not counted twice.
_TAG_LEASE_FLEET_PURCHASES = "PaymentsToAcquireLeasesHeldForInvestment"
_TAG_LEASE_FLEET_PROCEEDS = "ProceedsFromLeasesHeldForInvestment"
_TAGS_INTEREST = (
    "InterestExpense",
    "InterestAndDebtExpense",
    "InterestExpenseNonoperating",
    "InterestExpenseDebt",
    "InterestPaidNet",
    "InterestPaid",
)
_TAGS_TAX = ("IncomeTaxExpenseBenefit",)
_TAGS_PRETAX = (
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
)
# Some filers (McDonald's) tag pretax income only as its domestic and foreign
# parts; their sum is used for a year without one of the totals above.
_TAGS_PRETAX_PARTS = (
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesForeign",
)
_TAGS_DIVIDENDS = (
    "PaymentsOfDividendsCommonStock",
    "PaymentsOfDividends",
    "DividendsCommonStockCash",
)
_TAGS_DILUTED_SHARES = (
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic",
)
# Only the aggregate operating-capital tag is read. Most filers tag just the
# individual cash-flow lines instead, and a partial sum of them (receivables +
# inventories - payables) leaves out contract liabilities, accruals and other
# operating lines. On large caps such a sum overstated the working-capital
# drag more often than a zero did, so it is not used.
_TAGS_NWC = ("IncreaseDecreaseInOperatingCapital",)

# Instant (balance-sheet) tags.
# Noncurrent long-term debt (LongTermDebtAndCapitalLeaseObligations is also the
# noncurrent line, including lease obligations; LongTermNotesPayable is the
# noncurrent notes line some filers use instead).
_TAGS_LTD_NONCURRENT = (
    "LongTermDebtNoncurrent",
    "LongTermDebtAndCapitalLeaseObligations",
    "LongTermNotesPayable",
)
_TAGS_LTD_CURRENT = ("LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent")
# Short-term borrowings (which by definition include commercial paper; Apple-
# style filers tag only CommercialPaper).
_TAGS_ST_BORROWINGS = ("ShortTermBorrowings", "CommercialPaper")
# DebtCurrent = current maturities of long-term debt + short-term borrowings.
_TAGS_DEBT_CURRENT = ("DebtCurrent",)
# Long-term debt INCLUDING its current maturities (short-term borrowings are
# added separately).
_TAGS_LTD_TOTAL = (
    "LongTermDebt",
    "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
)
# Short- and long-term debt in one figure.
_TAGS_DEBT_COMBINED = (
    "DebtLongtermAndShorttermCombinedAmount",
    "DebtAndCapitalLeaseObligations",
)
# Last resort for filers with no debt total (many REITs): the separate
# balance-sheet debt lines. One tag per group is used, so a notes total and a
# subset of it (senior or convertible notes) are never both counted. Each line
# includes its current portion, except the noncurrent-only tags listed last in
# their groups (used only when the group has no including-current line).
_DEBT_LINE_GROUPS = (
    ("NotesPayable", "SeniorNotes", "UnsecuredDebt", "ConvertibleNotesPayable",
     "UnsecuredLongTermDebt", "SeniorLongTermNotes"),
    ("LoansPayable",),
    ("SecuredDebt", "SecuredLongTermDebt"),
    ("LineOfCredit",),
    ("SubordinatedDebt", "JuniorSubordinatedNotes"),
    ("OtherLongTermDebt",),
)
# Debt lines that exclude their current portion. When every line summed is
# one of these, all current debt is added (not just short-term borrowings).
_DEBT_LINES_NONCURRENT = frozenset(
    ("UnsecuredLongTermDebt", "SeniorLongTermNotes", "SecuredLongTermDebt")
)
# Two amounts within this relative distance are read as the same figure.
_AMOUNT_MATCH_TOLERANCE = 0.005
# Balance-sheet cash & equivalents first. Filers that tag only the cash-flow
# total including restricted cash (GE, P&G, Chevron, GE Vernova, most banks)
# get that total less restricted cash tagged at the same date; some insurers
# tag only a plain `Cash` line, and banks `CashAndDueFromBanks`. Last, the
# cash-flow total that also includes the cash of disposal groups and
# discontinued operations (PACCAR since 2019), treated the same way.
_TAG_CASH_WITH_RESTRICTED = "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"
_TAG_CASH_WITH_RESTRICTED_AND_DISPOSAL = (
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"
    "IncludingDisposalGroupAndDiscontinuedOperations"
)
_TAGS_CASH_WITH_RESTRICTED = (_TAG_CASH_WITH_RESTRICTED, _TAG_CASH_WITH_RESTRICTED_AND_DISPOSAL)
_TAGS_CASH = (
    "CashAndCashEquivalentsAtCarryingValue",
    _TAG_CASH_WITH_RESTRICTED,
    "Cash",
    "CashAndDueFromBanks",
    _TAG_CASH_WITH_RESTRICTED_AND_DISPOSAL,
)
# Restricted cash inside that total: a total tag, else current + noncurrent.
_TAGS_RESTRICTED_CASH = ("RestrictedCashAndCashEquivalents", "RestrictedCash")
_TAGS_RESTRICTED_CASH_CURRENT = (
    "RestrictedCashAndCashEquivalentsAtCarryingValue",
    "RestrictedCashCurrent",
)
_TAGS_RESTRICTED_CASH_NONCURRENT = (
    "RestrictedCashAndCashEquivalentsNoncurrent",
    "RestrictedCashNoncurrent",
)
_TAGS_ST_INVEST = ("ShortTermInvestments", "MarketableSecuritiesCurrent")
_TAGS_EQUITY = (
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
)
_TAGS_MINORITY = ("MinorityInterest",)
# Prefer full preferred book capital, including additional paid-in capital:
# PreferredStockValue alone can be just par value (JPM stopped tagging it in
# 2009, but still reports its full preferred capital under the first tag).
_TAGS_PREFERRED_BOOK = (
    "PreferredStockIncludingAdditionalPaidInCapitalNetOfDiscount",
    "PreferredStockValue",
)
_TAGS_PREFERRED = _TAGS_PREFERRED_BOOK + ("PreferredStockLiquidationPreferenceValue",)

# Financial filers. For banks, insurers, BDCs and REITs, borrowing, lending,
# investing or buying property is the business itself, so the operating-
# company EBIT derivations and the FCFF/FCFE models do not describe them. A
# filer is classified by a line that must be a material share of its latest
# balance sheet or revenue (a small captive-finance or run-off-insurance line
# in an industrial company does not count):
#   bank     Deposits >= 15% of total assets;
#   insurer  premiums earned or policyholder benefits >= 40% of revenue AND
#            net investment income >= 2% of revenue (the second test leaves
#            out managed-care companies, whose float income is ~1%);
#   BDC      investments at fair value >= 50% of total assets;
#   REIT     real-estate investment property >= 50% of total assets, or real
#            estate at gross carrying value on its Schedule III (the SEC's
#            real-estate schedule) >= 40% of them: tower and data-centre REITs
#            (American Tower 47%, Equinix 84%) tag no investment-property
#            line, and a tower REIT carries much of its assets as acquired
#            intangibles. Tested before the lessor test below, whose lease
#            income and interest shares a tower REIT also meets.
_TAGS_ASSETS = ("Assets",)
_TAGS_DEPOSITS = ("Deposits",)
_TAGS_PREMIUMS = ("PremiumsEarnedNet", "PolicyholderBenefitsAndClaimsIncurredNet")
_TAGS_INVESTMENT_INCOME = (
    "NetInvestmentIncome",
    "InvestmentIncomeInterestAndDividend",
    "InterestAndDividendIncomeOperating",
)
_TAGS_BDC_INVESTMENTS = ("InvestmentOwnedAtFairValue",)
_TAGS_REIT_PROPERTY = ("RealEstateInvestmentPropertyNet",)
_TAGS_REIT_SCHEDULE_III = ("RealEstateGrossAtCarryingValue",)
_BANK_MIN_DEPOSIT_SHARE = 0.15
_INSURER_MIN_PREMIUM_SHARE = 0.40
_INSURER_MIN_INVESTMENT_INCOME_SHARE = 0.02
_BDC_MIN_INVESTMENT_SHARE = 0.50
_REIT_MIN_PROPERTY_SHARE = 0.50
_REIT_MIN_SCHEDULE_III_SHARE = 0.40
# kind -> (label with article, why FCFF-style operating models do not fit it).
_FINANCIAL_KINDS = {
    "bank": ("a bank", "interest on deposits and borrowings is its main operating cost"),
    "insurer": ("an insurer", "investment income on policyholder funds is part of its operations"),
    "bdc": (
        "a business development company",
        "interest on the borrowing that funds its loans is an operating cost",
    ),
    "reit": ("a REIT", "its growth comes from buying property, which is not in capex"),
    "captive_finance": (
        "a group with a captive finance arm",
        "the finance arm's debt, leases and receivables are consolidated (total debt "
        "includes the borrowing that funds its customer loans and leases, and D&A the "
        "depreciation of its lease fleet)",
    ),
    "lessor": (
        "a debt-funded operating lessor",
        "the assets it leases out are bought with debt, whose interest is an "
        "operating cost of the leasing business, and its purchases of them (its "
        "growth investment) are often tagged outside capex",
    ),
}
# The captive-finance reason when the tags show no lease fleet (no operating-
# lease property, fleet purchases or operating-lease income): CarMax, CNH,
# Harley-Davidson, Snap-on and the timeshare groups finance customer loans.
_CAPTIVE_LOANS_ONLY_REASON = (
    "the finance arm's debt and receivables are consolidated (total debt includes "
    "the borrowing that funds its customer loans)"
)
# Captive finance arms (GM Financial, Ford Credit, John Deere Financial, Cat
# Financial, PACCAR Financial, Harley-Davidson Financial Services): a group
# that is none of the kinds above, sells products and lends to its customers
# and dealers to finance them. A product business is required: inventory on
# the latest balance sheet or a change in inventories in the latest fiscal
# year's cash flow (every captive group holds inventory; a lending platform
# such as Upstart does not). Evidence must include lending, since operating-
# lease vehicles alone are what any lessor or rental company holds; they only
# add to the arm's size. At the latest data, either
#   * finance receivables are at least `_CAPTIVE_MIN_RECEIVABLE_SHARE` of total
#     assets and, with property on operating leases, at least
#     `_CAPTIVE_MIN_ASSET_SHARE` (Ford 38% + 10% = 48%, Harley 29%; Lithia 20%,
#     Dell 19% and HPE 16% are smaller finance lines; a rental fleet with an
#     immaterial note receivable is not a finance arm); or
#   * loans and leases originated in the latest fiscal year are at least
#     `_CAPTIVE_MIN_ORIGINATION_SHARE` of revenue (Deere 58%, GM 28%,
#     Caterpillar 23%, PACCAR 21%; Dell 10%). Originations above
#     `_CAPTIVE_MAX_ORIGINATION_SHARE` of revenue are what a consumer lender or
#     payments company books, so they are not counted as this evidence (the
#     balance-sheet test still applies: a floorplan book turns over several
#     times a year).
# Lenders (the `_lender_reason` interest test) and lessors, whose lease income
# is at least `_LESSOR_MIN_LEASE_INCOME_SHARE` of revenue (AerCap, Hertz,
# GATX), are not captive finance arms.
_TAGS_FINANCE_RECEIVABLES = (
    ("FinancingReceivableExcludingAccruedInterestAfterAllowanceForCreditLoss",),
    ("FinancingReceivableExcludingAccruedInterestAfterAllowanceForCreditLossCurrent",
     "FinancingReceivableExcludingAccruedInterestAfterAllowanceForCreditLossNoncurrent"),
    ("NotesAndLoansReceivableNetCurrent", "NotesAndLoansReceivableNetNoncurrent"),
    ("NotesReceivableNet",),
    ("LoansAndLeasesReceivableNetReportedAmount",),
)
_TAGS_OPERATING_LEASE_PROPERTY = ("PropertySubjectToOrAvailableForOperatingLeaseNet",)
_TAGS_FINANCE_ORIGINATIONS = (
    "PaymentsToAcquireFinanceReceivables",
    _TAG_LEASE_FLEET_PURCHASES,
    "PaymentsToAcquireLoansAndLeasesHeldForInvestment",
)
_TAGS_LEASE_INCOME = ("LeaseIncome", "OperatingLeaseLeaseIncome")
_TAG_OPERATING_LEASE_INCOME = "OperatingLeaseLeaseIncome"
# Inventory balances (timeshare groups tag theirs as real-estate inventory),
# and the cash-flow change in inventories for a filer whose balance-sheet line
# uses a company-specific element (PACCAR, Marriott Vacations).
_TAGS_INVENTORY = ("InventoryNet", "InventoryGross", "InventoryRealEstate")
_TAG_INVENTORY_CHANGE = "IncreaseDecreaseInInventories"
_CAPTIVE_MIN_RECEIVABLE_SHARE = 0.10
_CAPTIVE_MIN_ASSET_SHARE = 0.25
_CAPTIVE_MIN_ORIGINATION_SHARE = 0.15
_CAPTIVE_MAX_ORIGINATION_SHARE = 1.0
_LESSOR_MIN_LEASE_INCOME_SHARE = 0.50
# Debt-funded operating lessors (AerCap, GATX, Avis Budget, Hertz), marked
# "lessor": none of the kinds above, with lease income at least
# `_LESSOR_MIN_LEASE_INCOME_SHARE` of revenue and interest expense at least
# `_LENDER_MIN_INTEREST_SHARE` of it in the latest fiscal year. Their fleet
# purchases are the growth investment and often sit outside the capex tags,
# and the debt that funds them grows with the fleet. Rental companies with
# less interest (United Rentals 4% of revenue, Herc 9.6%) are not marked. The
# interest tested is the largest figure tagged for that year among
# `_TAGS_LESSOR_INTEREST` (as for D&A, each is the total or a part): Hertz
# tags its vehicle interest only as operating interest expense, and Avis
# Budget's shows only in its interest paid, beside its corporate interest.
_TAGS_LESSOR_INTEREST = _TAGS_INTEREST + ("InterestExpenseOperating",)
# Where interest expense is a cost of the lending or trading book rather than
# of financing, pretax income + interest expense would count it twice as EBIT,
# so that derivation is skipped: always for banks and BDCs, and for any other
# filer whose latest interest expense is at least this share of revenue, or
# that pays interest with no revenue line (lenders, brokers, mortgage REITs;
# the same share as the lender test in utils.financial_institution). Insurers
# and equity REITs are derived: their interest expense is on corporate or
# property debt, a financing cost, and their revenue is gross premiums or
# rent, not a net interest margin. Lessors are derived as well, so that EBIT
# and EBITDA show the margin on their gross lease income before the interest
# tagged as interest expense; but the interest on their fleet debt is really
# an operating cost of the leasing business (their WARNING), which is why
# the engine keeps their FCFF DCF and FCFE out of the blend, and this EBIT is
# for display and those reference figures only. Fleet interest tagged
# outside the interest-expense tags (Hertz's operating interest expense,
# Avis Budget's vehicle interest) is not added back and stays inside it.
# Equity REITs and lessors are also exempt from the share test
# (`_SHARE_TEST_EXEMPT_KINDS`), since their interest often passes 10% of
# revenue.
_EBIT_NOT_DERIVED_KINDS = ("bank", "bdc")
_SHARE_TEST_EXEMPT_KINDS = ("reit", "lessor")
_LENDER_MIN_INTEREST_SHARE = 0.10

# A latest-year interest expense above this share of total debt means debt
# the tags above do not cover (company-specific tags, unread line items). It
# is flagged when the debt that interest implies at this rate exceeds the debt
# found by more than the given share of revenue (so a debt-free company paying
# a little interest on leases or fees is not flagged), and the average debt on
# that fiscal year's balance sheets (from the opening one, dated within this
# many days before the year starts, to the closing one) falls short as well,
# i.e. the debt was not repaid during the year.
_MAX_PLAUSIBLE_INTEREST_RATE = 0.15
_MIN_DEBT_SHORTFALL_REVENUE_SHARE = 0.01
_YEAR_START_BALANCE_SHEET_DAYS = 7

# Share counts report under unit "shares", everything else under "USD".
_UNIT_USD = "USD"
_UNIT_SHARES = "shares"

# Foreign private issuers that report under US GAAP in their own currency
# (Alibaba, JD.com, PDD and Baidu in CNY; Toyota in JPY until it moved to
# IFRS; Canadian National in CAD) tag the statements in that currency and
# often add USD convenience translations of a few figures for the latest year
# only. The USD facts are then a partial set (no D&A or capex, debt and cash
# under-read), so such a filer is not parsed here: EDGAR raises DataError and
# the hybrid provider uses the yfinance fallback, which converts at spot. A
# filer counts as such when revenue is tagged in another currency for a
# period at least as recent as its latest USD revenue, or when one other
# currency carries more than `_FOREIGN_UNIT_MIN_SHARE` of the monetary facts
# dated within a year of the latest one. Only facts whose period ends on or
# before their filing date count: a mistyped far-future date (Oracle tags an
# expected restructuring cost at 2199-12-31) must not set the window.
_FOREIGN_UNIT_MIN_SHARE = 0.5
_RECENT_FACT_DAYS = 366

_DAY_SECONDS = 86400.0


def _days_between(start: str, end: str) -> Optional[float]:
    """Number of days between two ISO `YYYY-MM-DD` dates, or None on bad input."""
    try:
        # Parse without pulling datetime's full strptime overhead per call; the
        # SEC dates are always strict ISO calendar dates.
        sy, sm, sd = (int(p) for p in start.split("-"))
        ey, em, ed = (int(p) for p in end.split("-"))
    except (AttributeError, ValueError):
        return None
    # Convert both to a day ordinal via the proleptic Gregorian calendar.
    import datetime

    try:
        d0 = datetime.date(sy, sm, sd)
        d1 = datetime.date(ey, em, ed)
    except ValueError:
        return None
    return (d1 - d0).days


def _fiscal_year_label(end: str) -> Optional[int]:
    """Fiscal-year label for a period ending on ISO date `end`, or None.

    The calendar year of the period end, except that a period ending in the
    first `_EARLY_JANUARY_DAYS` of January (a 52/53-week year ending on the
    weekend nearest Dec 31) takes the prior year.
    """
    try:
        y, m, d = (int(p) for p in str(end)[:10].split("-"))
    except (TypeError, ValueError):
        return None
    if m == 1 and d <= _EARLY_JANUARY_DAYS:
        return y - 1
    return y


def _reported_period(end: object, filed: object) -> bool:
    """Whether a fact has a real period end no later than its filing date.

    SEC facts can include expected future amounts and mistyped dates. Neither
    may determine a historical balance-sheet date or an annual flow. Missing
    filing dates remain usable, but a supplied date must be a valid ISO date.
    """
    if not isinstance(end, str) or len(end) != 10 or _days_between(end, end) != 0:
        return False
    if not filed:
        return True
    return (isinstance(filed, str) and len(filed) == 10
            and _days_between(filed, filed) == 0 and end <= filed)


def _fy_list(years: list[int]) -> str:
    """Compact 'FY2019-2021, FY2024' rendering of a sorted year list for notes."""
    runs: list[list[int]] = []
    for y in sorted(years):
        if runs and y == runs[-1][-1] + 1:
            runs[-1].append(y)
        else:
            runs.append([y])
    return ", ".join(
        f"FY{r[0]}" if len(r) == 1 else f"FY{r[0]}-{r[-1]}" for r in runs
    )


# --------------------------------------------------------------------------- #
#  Client
# --------------------------------------------------------------------------- #
class EdgarClient:
    """Thin client over the SEC EDGAR XBRL JSON API.

    The ticker->CIK directory is fetched once and cached on the instance, so a
    single client can resolve many tickers cheaply.
    """

    def __init__(self, user_agent: str = config.SEC_USER_AGENT) -> None:
        self.user_agent = user_agent or config.SEC_USER_AGENT
        # Lazily-populated cache: upper-cased ticker -> (padded cik, name).
        self._ticker_map: Optional[dict[str, tuple[str, str]]] = None
        # A pooled session reuses the TCP connection across the directory +
        # facts calls and keeps the required header on every request.
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/json",
                # data.sec.gov is content-negotiated; Host is set by requests.
            }
        )

    # ----------------------------- HTTP ----------------------------------- #
    def _get_json(self, url: str) -> dict:
        """GET `url` as JSON with the mandatory User-Agent, retries and backoff.

        Retries on transient network errors and on HTTP 429/5xx, sleeping with
        an exponential backoff between attempts. Raises `DataError` once the
        retry budget (`config.SEC_MAX_RETRIES`) is exhausted.
        """
        if not self.user_agent:
            raise DataError("Set SEC_USER_AGENT to an application name and contact email before requesting EDGAR data.")
        last_err: Optional[str] = None
        attempts = max(1, int(config.SEC_MAX_RETRIES))
        for attempt in range(attempts):
            try:
                resp = self._session.get(
                    url,
                    timeout=config.SEC_REQUEST_TIMEOUT,
                    headers={"User-Agent": self.user_agent},
                )
            except requests.RequestException as exc:  # network / timeout error
                last_err = f"network error: {exc}"
            else:
                status = resp.status_code
                if status == 200:
                    try:
                        return resp.json()
                    except ValueError as exc:
                        last_err = f"invalid JSON: {exc}"
                        # Malformed body is unlikely to fix itself; stop early.
                        break
                elif status == 404:
                    # Not found is definitive — no point retrying.
                    raise DataError(f"SEC returned 404 (not found) for {url}")
                elif status == 403:
                    # Almost always a missing/blocked User-Agent. Retrying with
                    # the same header rarely helps, but the budget is small.
                    last_err = (
                        "SEC returned 403 (forbidden) — verify the User-Agent "
                        f"header ({self.user_agent!r}) includes contact info"
                    )
                elif status == 429 or 500 <= status < 600:
                    last_err = f"SEC returned HTTP {status}"
                else:
                    last_err = f"SEC returned unexpected HTTP {status}"

            # Backoff before the next attempt (skip the sleep after the last).
            if attempt < attempts - 1:
                backoff = config.HTTP_RETRY_BACKOFF * (2 ** attempt)
                time.sleep(backoff)

        raise DataError(f"failed to GET {url}: {last_err or 'unknown error'}")

    # --------------------------- ticker -> CIK ---------------------------- #
    def _load_ticker_map(self) -> dict[str, tuple[str, str]]:
        """Fetch (once) and cache the ticker -> (cik, name) directory.

        The SEC payload is a dict keyed by an arbitrary index, each value a
        record like {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}.
        """
        if self._ticker_map is not None:
            return self._ticker_map

        raw = self._get_json(_TICKERS_URL)
        mapping: dict[str, tuple[str, str]] = {}
        # The directory is normally a dict-of-records, but guard for a list too.
        records = raw.values() if isinstance(raw, dict) else raw
        for rec in records:
            if not isinstance(rec, dict):
                continue
            ticker = rec.get("ticker")
            cik_raw = rec.get("cik_str")
            if not ticker or cik_raw is None:
                continue
            try:
                cik = str(int(cik_raw)).zfill(10)
            except (TypeError, ValueError):
                continue
            name = rec.get("title") or ticker
            # First occurrence wins; the directory is effectively unique by
            # ticker, so duplicates (rare) keep the earliest listing.
            mapping.setdefault(str(ticker).upper(), (cik, str(name)))

        self._ticker_map = mapping
        return mapping

    def resolve_cik(self, ticker: str) -> tuple[str, str]:
        """Return ``(10-digit zero-padded CIK, company name)`` for `ticker`.

        Raises `DataError` if the ticker is absent from the SEC directory
        (e.g. a non-US issuer with no EDGAR registration).
        """
        if not ticker or not isinstance(ticker, str):
            raise DataError(f"invalid ticker: {ticker!r}")
        key = ticker.strip().upper()
        mapping = self._load_ticker_map()
        hit = mapping.get(key)
        if hit is None:
            # Some tickers carry a class/exchange suffix (e.g. "BRK.B"); the SEC
            # directory uses "BRK-B". Try a dotted->dashed normalization.
            alt = key.replace(".", "-")
            hit = mapping.get(alt)
        if hit is None:
            raise DataError(f"ticker {ticker!r} not found in SEC EDGAR directory")
        return hit

    # --------------------------- company facts ---------------------------- #
    def company_facts(self, cik: str) -> dict:
        """GET the XBRL company-facts document for a zero-padded `cik`."""
        if not cik:
            raise DataError("company_facts called with empty CIK")
        padded = str(cik).strip().zfill(10)
        url = _FACTS_URL.format(cik=padded)
        data = self._get_json(url)
        if not isinstance(data, dict):
            raise DataError(f"unexpected company-facts payload for CIK {padded}")
        return data

    # ------------------------- fact extraction ---------------------------- #
    @staticmethod
    def _unit_entries(facts: dict, tag: str, unit: str) -> list[dict]:
        """Return the list of fact entries for ``us-gaap[tag].units[unit]``.

        Returns an empty list if the tag, the unit, or the path is absent.
        """
        try:
            gaap = facts.get("facts", {}).get("us-gaap", {})
            node = gaap.get(tag)
            if not node:
                return []
            units = node.get("units", {})
            entries = units.get(unit)
            return entries if isinstance(entries, list) else []
        except AttributeError:
            return []

    @staticmethod
    def _annual_fact(e) -> Optional[tuple[str, str, str, float]]:
        """``(start, end, filed, value)`` of a full-year annual-report flow fact.

        None unless the entry has ``fp == 'FY'``, a form in `_ANNUAL_FORMS`, a
        numeric value and a period of `_MIN_PERIOD_DAYS`-`_MAX_PERIOD_DAYS`.
        """
        if not isinstance(e, dict):
            return None
        if e.get("fp") != "FY" or e.get("form") not in _ANNUAL_FORMS:
            return None
        start, end, val = e.get("start"), e.get("end"), e.get("val")
        filed = e.get("filed") or ""
        if not start or not _reported_period(end, filed) or not is_num(val):
            return None
        span = _days_between(start, end)
        if span is None or not (_MIN_PERIOD_DAYS <= span <= _MAX_PERIOD_DAYS):
            return None
        return start, end, filed, float(val)

    def _fiscal_year_period(
        self, facts: dict, tags: tuple[str, ...], year: int
    ) -> Optional[tuple[str, str]]:
        """``(start, end)`` of the full-year period labelled `year` in `tags`.

        The latest period end with that label wins, as in `_annual_flow_by_fy`;
        None when no such period is reported.
        """
        best: Optional[tuple[str, str]] = None  # (end, start)
        for tag in tags:
            for e in self._unit_entries(facts, tag, _UNIT_USD):
                fact = self._annual_fact(e)
                if fact is None or _fiscal_year_label(fact[1]) != year:
                    continue
                if best is None or fact[1] > best[0]:
                    best = (fact[1], fact[0])
        return (best[1], best[0]) if best else None

    def _annual_flow_by_fy(
        self, facts: dict, tags: tuple[str, ...], unit: str = _UNIT_USD
    ) -> dict[int, float]:
        """Map fiscal-year label -> value for a FLOW item.

        Keyed by the reporting PERIOD (``end`` date), NOT the XBRL ``fy`` field.
        The same period is reported in several filings -- the original plus
        comparatives in later 10-Ks -- each carrying a *different* ``fy`` (the
        filing's fiscal year, not the period's). Keying by ``fy`` therefore both
        collides the current/prior year onto one key and hides restatements
        (e.g. a switched dividend tag, or the comparative years a post-split
        10-K restates). We instead:

          * keep entries with ``fp == 'FY'`` and ``form`` in the annual forms
            (including amendments), whose period length (``end`` - ``start``)
            is a full year;
          * dedupe by ``end`` date, keeping the LATEST ``filed`` value so
            restatements and amendments win over the original report;
          * BACKFILL across the tag-fallback list: the highest-preference tag
            wins for any period it covers, and lower-preference tags fill the
            periods it doesn't (so a company that changed tags over time still
            gets a complete series);
          * collapse each ``end`` date to a fiscal-year label (the calendar year
            of the period end, or the prior year for a 52/53-week year ending in
            early January), keeping the latest period if two share a label.

        Only periods some later filing re-reported are restated: e.g. share
        counts older than a split's restated comparatives stay unadjusted.
        """
        return {
            year: vf[0]
            for year, vf in self._annual_flow_filed_by_fy(facts, tags, unit).items()
        }

    def _annual_flow_filed_by_fy(
        self, facts: dict, tags: tuple[str, ...], unit: str = _UNIT_USD
    ) -> dict[int, tuple[float, str]]:
        """``fy -> (value, filed)``: `_annual_flow_by_fy` with the filing date
        of each value kept (the latest filing of that period under its tag)."""
        # end-date -> (filed, value). Later (lower-preference) tags only fill in
        # periods the earlier tags did not already cover.
        by_end: dict[str, tuple[str, float]] = {}
        for tag in tags:
            entries = self._unit_entries(facts, tag, unit)
            if not entries:
                continue
            tag_by_end: dict[str, tuple[str, float]] = {}
            for e in entries:
                fact = self._annual_fact(e)
                if fact is None:
                    continue
                _start, end, filed, val = fact
                prev = tag_by_end.get(end)
                if prev is None or filed >= prev[0]:
                    tag_by_end[end] = (filed, val)
            # Backfill: keep the higher-preference tag's value for shared periods.
            for end, fv in tag_by_end.items():
                by_end.setdefault(end, fv)

        if not by_end:
            return {}

        # Collapse end-date -> fiscal-year label, keeping the latest end date if
        # two periods share a label (which only happens across a fiscal-year-
        # end change).
        by_year: dict[int, tuple[str, float, str]] = {}  # (end, value, filed)
        for end, (filed, val) in by_end.items():
            year = _fiscal_year_label(end)
            if year is None:
                continue
            prev = by_year.get(year)
            if prev is None or end > prev[0]:
                by_year[year] = (end, val, filed)
        return {year: (evf[1], evf[2]) for year, evf in by_year.items()}

    def _largest_flow_by_fy(
        self, facts: dict, tags: tuple[str, ...], *, larger_every_year: bool = False
    ) -> tuple[dict[int, float], dict[int, tuple[str, str]]]:
        """``(fy -> value, fy -> (tag used, tag passed over))`` for a flow whose
        tags each report either the total or a part of it (see `_TAGS_DA` and
        `_TAGS_CAPEX`).

        As `_annual_flow_by_fy`, except that each fiscal year takes the value
        of the largest magnitude among the tags' own series (each built as
        `_annual_flow_by_fy` builds one), not the first tag's. A later tag
        replaces the value held only if it is larger by more than
        `_AMOUNT_MATCH_TOLERANCE` (so matching figures keep the earlier tag)
        and was filed no earlier than it: a figure left under a tag the filer
        stopped using must not beat the same year restated under a tag it
        still files, as `_annual_flow_by_fy` lets the latest filing win. The
        second map lists only the years where a later tag replaced an earlier
        one, with the earlier tag the first-tag rule would have used.

        With `larger_every_year`, a later tag replaces values only if it is
        larger in that way than each earlier tag in every fiscal year both
        report (years where both are zero aside), as a total is larger than
        a part the filer tags every year. A tag that is larger only in some
        years (a line that adds acquisitions, or a separate line) only fills
        the years the earlier tags leave empty.
        """
        series = [self._annual_flow_filed_by_fy(facts, (tag,)) for tag in tags]

        def larger_throughout(
            later: dict[int, tuple[float, str]], earlier: dict[int, tuple[float, str]]
        ) -> bool:
            return all(
                abs(later[y][0]) > abs(earlier[y][0]) * (1.0 + _AMOUNT_MATCH_TOLERANCE)
                for y in later.keys() & earlier.keys()
                if later[y][0] or earlier[y][0]
            )

        held: dict[int, tuple[float, str]] = {}  # fy -> (value, filed)
        first: dict[int, str] = {}
        used: dict[int, str] = {}
        for i, (tag, by_fy) in enumerate(zip(tags, series)):
            may_replace = not larger_every_year or all(
                larger_throughout(by_fy, earlier) for earlier in series[:i]
            )
            for year, (val, filed) in by_fy.items():
                prev = held.get(year)
                if prev is None:
                    held[year], first[year], used[year] = (val, filed), tag, tag
                elif (may_replace
                      and abs(val) > abs(prev[0]) * (1.0 + _AMOUNT_MATCH_TOLERANCE)
                      and filed >= prev[1]):
                    held[year], used[year] = (val, filed), tag
        replaced = {y: (used[y], first[y]) for y in held if used[y] != first[y]}
        return {y: vf[0] for y, vf in held.items()}, replaced

    def _instant_by_end(
        self, facts: dict, tag: str, unit: str = _UNIT_USD
    ) -> dict[str, float]:
        """``end date -> value`` for an INSTANT tag, the latest filing winning."""
        best: dict[str, tuple[str, float]] = {}
        for e in self._unit_entries(facts, tag, unit):
            if not isinstance(e, dict):
                continue
            end = e.get("end")
            val = e.get("val")
            filed = e.get("filed") or ""
            if (e.get("start") or not _reported_period(end, filed)
                    or not is_num(val)):
                continue
            prev = best.get(end)
            if prev is None or filed >= prev[0]:
                best[end] = (filed, float(val))
        return {end: fv[1] for end, fv in best.items()}

    def _instant_at(
        self,
        facts: dict,
        tags: tuple[str, ...],
        as_of: str,
        notes: list[str],
        label: str,
        bs_dates: Optional[set[str]] = None,
    ) -> Optional[float]:
        """Value of a balance-sheet item at the snapshot date `as_of`, or None.

        See `_instant_fact`, which also reports the tag and date used.
        """
        return self._instant_fact(facts, tags, as_of, notes, label, bs_dates)[0]

    def _instant_fact(
        self,
        facts: dict,
        tags: tuple[str, ...],
        as_of: str,
        notes: list[str],
        label: str,
        bs_dates: Optional[set[str]] = None,
    ) -> tuple[Optional[float], Optional[str], Optional[str]]:
        """``(value, tag, end)`` of a balance-sheet item at `as_of`, or Nones.

        The first tag (in preference order) with a fact dated exactly `as_of`
        wins. Failing that, the most recent fact dated within
        `_INSTANT_GRACE_DAYS` before `as_of` is used (noted). An item whose
        latest fact is older than that is treated as no longer reported: None,
        with a note when the ignored value was non-zero. Facts dated after
        `as_of` are never used. With `bs_dates` (the filer's balance-sheet
        dates), an earlier fact must fall on one of them: a debt amount dated
        mid-quarter describes one note issue, not the balance.
        """
        series = [self._instant_by_end(facts, tag) for tag in tags]
        for tag, by_end in zip(tags, series):
            if as_of in by_end:
                return by_end[as_of], tag, as_of
        older: list[tuple[str, int, float]] = []  # (end, -preference, value)
        for pref, by_end in enumerate(series):
            for end, val in by_end.items():
                if end < as_of and (not bs_dates or end in bs_dates):
                    older.append((end, -pref, val))
        if not older:
            return None, None, None
        end, neg_pref, val = max(older)
        age = _days_between(end, as_of)
        if age is not None and age <= _INSTANT_GRACE_DAYS:
            notes.append(f"{label} taken from {end}; not reported at the {as_of} balance sheet")
            return val, tags[-neg_pref], end
        if val != 0.0:
            notes.append(
                f"{label} last reported {end} ({val:,.0f}); treated as 0 at the "
                f"{as_of} balance sheet"
            )
        return None, None, None

    # ------------------------ financial-filer check ----------------------- #
    def _financial_kind(
        self, facts: dict, revenue_by_fy: dict[int, float]
    ) -> tuple[Optional[str], str]:
        """``(kind, evidence)`` for a bank, insurer, BDC or REIT, else (None, "").

        `kind` is a key of `_FINANCIAL_KINDS`; `evidence` states the share that
        qualified it (e.g. "deposits are 54% of total assets"). Each test uses
        the filer's latest data: a balance-sheet share at the latest date on
        which both the line and total assets are reported (within
        `_INSTANT_GRACE_DAYS` of the latest total assets), and a revenue share
        in the latest (or prior) fiscal year, so a line the company stopped
        reporting years ago (GE's former bank deposits) does not count.
        """
        assets = self._instant_by_end(facts, _TAGS_ASSETS[0])
        latest_assets = max(assets) if assets else None

        def asset_share(tags: tuple[str, ...]) -> Optional[float]:
            if latest_assets is None:
                return None
            for tag in tags:
                by_end = self._instant_by_end(facts, tag)
                common = [e for e in by_end if assets.get(e, 0.0) > 0]
                if not common:
                    continue
                end = max(common)
                age = _days_between(end, latest_assets)
                if age is not None and age <= _INSTANT_GRACE_DAYS:
                    return by_end[end] / assets[end]
            return None

        def revenue_share(tags: tuple[str, ...]) -> Optional[float]:
            if not revenue_by_fy:
                return None
            last = max(revenue_by_fy)
            best: Optional[float] = None
            for tag in tags:
                series = self._annual_flow_by_fy(facts, (tag,))
                for year in (last, last - 1):
                    rev = revenue_by_fy.get(year)
                    if year in series and rev is not None and rev > 0:
                        share = abs(series[year]) / rev
                        best = share if best is None else max(best, share)
                        break
            return best

        share = asset_share(_TAGS_DEPOSITS)
        if share is not None and share >= _BANK_MIN_DEPOSIT_SHARE:
            return "bank", f"deposits are {share:.0%} of total assets"
        premiums = revenue_share(_TAGS_PREMIUMS)
        if premiums is not None and premiums >= _INSURER_MIN_PREMIUM_SHARE:
            invest = revenue_share(_TAGS_INVESTMENT_INCOME)
            if invest is not None and invest >= _INSURER_MIN_INVESTMENT_INCOME_SHARE:
                return "insurer", (
                    f"premiums or policy benefits are {premiums:.0%} and net "
                    f"investment income {invest:.0%} of revenue"
                )
        share = asset_share(_TAGS_BDC_INVESTMENTS)
        if share is not None and share >= _BDC_MIN_INVESTMENT_SHARE:
            return "bdc", f"investments at fair value are {share:.0%} of total assets"
        share = asset_share(_TAGS_REIT_PROPERTY)
        if share is not None and share >= _REIT_MIN_PROPERTY_SHARE:
            return "reit", f"investment property is {share:.0%} of total assets"
        share = asset_share(_TAGS_REIT_SCHEDULE_III)
        if share is not None and share >= _REIT_MIN_SCHEDULE_III_SHARE:
            return "reit", (f"real estate at gross carrying value (its Schedule III) is "
                            f"{share:.0%} of total assets")
        return None, ""

    def _latest_asset_share(
        self, facts: dict, tag_groups: tuple[tuple[str, ...], ...]
    ) -> float:
        """Largest share of total assets among `tag_groups`, or 0.0.

        A group's tags (e.g. a current and a noncurrent line) are summed at the
        latest date on which total assets and at least one of them are
        reported, provided that date is within `_INSTANT_GRACE_DAYS` of the
        latest total assets, so a line the filer stopped reporting does not
        count.
        """
        assets = self._instant_by_end(facts, _TAGS_ASSETS[0])
        if not assets:
            return 0.0
        latest_assets = max(assets)
        best = 0.0
        for group in tag_groups:
            series = [self._instant_by_end(facts, tag) for tag in group]
            dates = [d for by_end in series for d in by_end if assets.get(d, 0.0) > 0]
            if not dates:
                continue
            end = max(dates)
            age = _days_between(end, latest_assets)
            if age is None or age > _INSTANT_GRACE_DAYS:
                continue
            best = max(best, sum(by_end.get(end, 0.0) for by_end in series) / assets[end])
        return best

    def _lessor_evidence(
        self, facts: dict, year: int, revenue_by_fy: dict[int, float]
    ) -> Optional[str]:
        """Evidence that the filer is a debt-funded operating lessor, or None.

        Applies the test described at `_TAGS_LESSOR_INTEREST` to fiscal
        `year` (the latest); call it only for a filer that is not a bank,
        insurer, BDC or REIT.
        """
        revenue = revenue_by_fy.get(year) or 0.0
        if revenue <= 0:
            return None
        lease = max(abs(self._annual_flow_by_fy(facts, (t,)).get(year, 0.0))
                    for t in _TAGS_LEASE_INCOME)
        if lease < _LESSOR_MIN_LEASE_INCOME_SHARE * revenue:
            return None
        interest_by_fy, _ = self._largest_flow_by_fy(facts, _TAGS_LESSOR_INTEREST)
        interest = abs(interest_by_fy.get(year, 0.0))
        if interest < _LENDER_MIN_INTEREST_SHARE * revenue:
            return None
        return (f"FY{year} lease income is {lease / revenue:.0%} and interest expense "
                f"{interest / revenue:.0%} of revenue")

    def _captive_finance_evidence(
        self,
        facts: dict,
        year: int,
        revenue_by_fy: dict[int, float],
        interest_by_fy: dict[int, float],
    ) -> Optional[tuple[str, str]]:
        """``(evidence, reason)`` for a group with a captive finance arm, or None.

        Applies the tests described at `_TAGS_FINANCE_RECEIVABLES` to fiscal
        `year` (the latest) and the latest balance sheet; call it only for a
        filer that is not a bank, insurer, BDC or REIT. `reason` mentions a
        lease fleet only when the tags show one (operating-lease property,
        fleet purchases or operating-lease income), else it is
        `_CAPTIVE_LOANS_ONLY_REASON`.
        """
        revenue = revenue_by_fy.get(year) or 0.0
        if revenue <= 0 or self._lender_reason(None, year, revenue_by_fy, interest_by_fy):
            return None

        def latest_flow(tag: str) -> float:
            return abs(self._annual_flow_by_fy(facts, (tag,)).get(year, 0.0))

        if max(latest_flow(t) for t in _TAGS_LEASE_INCOME) >= (
                _LESSOR_MIN_LEASE_INCOME_SHARE * revenue):
            return None  # a lessor or rental company
        inventory = self._latest_asset_share(facts, tuple((t,) for t in _TAGS_INVENTORY))
        if inventory <= 0 and latest_flow(_TAG_INVENTORY_CHANGE) <= 0:
            return None  # no products to finance: a lender or lending platform
        originated = sum(latest_flow(t) for t in _TAGS_FINANCE_ORIGINATIONS) / revenue
        evidence: list[str] = []
        receivables = self._latest_asset_share(facts, _TAGS_FINANCE_RECEIVABLES)
        fleet = self._latest_asset_share(facts, (_TAGS_OPERATING_LEASE_PROPERTY,))
        if (receivables >= _CAPTIVE_MIN_RECEIVABLE_SHARE
                and receivables + fleet >= _CAPTIVE_MIN_ASSET_SHARE):
            evidence.append(
                "finance receivables"
                + (" and operating-lease property" if fleet > 0 else "")
                + f" are {receivables + fleet:.0%} of total assets"
            )
        # Originations above revenue are a lender's volume (or a revolving
        # floorplan book), so they are not evidence on their own.
        if _CAPTIVE_MIN_ORIGINATION_SHARE <= originated <= _CAPTIVE_MAX_ORIGINATION_SHARE:
            evidence.append(f"FY{year} loan and lease originations are {originated:.0%} of revenue")
            if fleet > 0 and len(evidence) == 1:
                evidence[0] += f", operating-lease property {fleet:.0%} of total assets"
        if not evidence:
            return None
        leases = (fleet > 0 or latest_flow(_TAG_LEASE_FLEET_PURCHASES) > 0
                  or latest_flow(_TAG_OPERATING_LEASE_INCOME) > 0)
        reason = _FINANCIAL_KINDS["captive_finance"][1] if leases else _CAPTIVE_LOANS_ONLY_REASON
        return " and ".join(evidence), reason

    def _foreign_reporting_currency(
        self, facts: dict
    ) -> Optional[tuple[str, str, Optional[str]]]:
        """``(currency, evidence, usd)`` when the us-gaap facts are not in USD,
        else None.

        See `_FOREIGN_UNIT_MIN_SHARE`: revenue tagged in another currency for a
        period at least as recent as the latest USD revenue, or one other
        currency carrying most of the monetary facts of the latest year. Facts
        dated after their own filing date are ignored. `usd` says what the
        latest year has in USD: "revenue" (USD convenience translations that
        include revenue), "facts" (a few other USD figures) or None.
        """
        try:
            gaap = facts.get("facts", {}).get("us-gaap", {}) or {}
            nodes = list(gaap.values())
        except AttributeError:
            return None

        def reported(end: str, filed) -> bool:
            # A period cannot end after the filing that reports it, so such a
            # date is a forecast or a typo (Oracle's 2199-12-31).
            return bool(filed) and end <= str(filed)[:10]

        # Monetary units are ISO currency codes ("USD", "CNY"); share counts,
        # ratios and per-share units ("shares", "pure", "USD/shares") are not.
        ends_by_unit: dict[str, list[str]] = {}
        for node in nodes:
            units = node.get("units") if isinstance(node, dict) else None
            for unit, entries in (units.items() if isinstance(units, dict) else ()):
                if not (isinstance(unit, str) and len(unit) == 3 and unit.isalpha()
                        and unit.isupper() and isinstance(entries, list)):
                    continue
                ends_by_unit.setdefault(unit, []).extend(
                    str(e["end"])[:10] for e in entries
                    if isinstance(e, dict) and e.get("end")
                    and reported(str(e["end"])[:10], e.get("filed"))
                )
        if not any(u != _UNIT_USD and ends for u, ends in ends_by_unit.items()):
            return None

        def latest_revenue_end(unit: str) -> Optional[str]:
            ends = [
                fact[1]
                for tag in _TAGS_REVENUE
                for fact in map(self._annual_fact, self._unit_entries(facts, tag, unit))
                if fact is not None and reported(fact[1], fact[2])
            ]
            return max(ends) if ends else None

        import datetime

        latest = max(max(ends) for ends in ends_by_unit.values() if ends)
        try:
            cutoff = (datetime.date.fromisoformat(latest)
                      - datetime.timedelta(days=_RECENT_FACT_DAYS)).isoformat()
        except ValueError:
            cutoff = None  # malformed date: only the revenue test applies
        recent = ({u: sum(1 for d in ends if d >= cutoff) for u, ends in ends_by_unit.items()}
                  if cutoff else {})
        usd_revenue_end = latest_revenue_end(_UNIT_USD)
        if usd_revenue_end is not None and (cutoff is None or usd_revenue_end >= cutoff):
            usd: Optional[str] = "revenue"
        else:
            usd = "facts" if recent.get(_UNIT_USD, 0) > 0 else None

        for unit in sorted(ends_by_unit):
            end = None if unit == _UNIT_USD else latest_revenue_end(unit)
            if end is not None and (usd_revenue_end is None or end >= usd_revenue_end):
                return unit, f"revenue is tagged in {unit}", usd

        total = sum(recent.values())
        if not total:
            return None
        unit, count = max(recent.items(), key=lambda kv: kv[1])
        if unit != _UNIT_USD and count > _FOREIGN_UNIT_MIN_SHARE * total:
            # Rounded down, so "100%" means every fact of that year is in `unit`.
            return unit, (
                f"{count * 100 // total}% of its monetary facts dated within a year "
                f"of {latest} are in {unit}"
            ), usd
        return None

    @staticmethod
    def _lender_reason(
        fin_kind: Optional[str],
        year: int,
        revenue_by_fy: dict[int, float],
        interest_by_fy: dict[int, float],
    ) -> Optional[str]:
        """Why pretax income + interest expense is not EBIT here, else None.

        A reason is given for a bank or BDC, and for any other filer but an
        equity REIT or a lessor when its fiscal-`year` interest expense is at
        least `_LENDER_MIN_INTEREST_SHARE` of revenue or comes with no revenue
        (see `_EBIT_NOT_DERIVED_KINDS`). It completes a source note.
        """
        if fin_kind in _EBIT_NOT_DERIVED_KINDS:
            return (
                f"interest is an operating cost of {_FINANCIAL_KINDS[fin_kind][0]}, "
                "so adding it back would count it twice"
            )
        interest = abs(interest_by_fy.get(year, 0.0))
        if fin_kind in _SHARE_TEST_EXEMPT_KINDS or interest <= 0:
            return None
        revenue = revenue_by_fy.get(year) or 0.0
        if revenue <= 0:
            share = f"{interest:,.0f} with no revenue"
        elif interest >= _LENDER_MIN_INTEREST_SHARE * revenue:
            share = f"{interest / revenue:.0%} of revenue"
        else:
            return None
        return (
            f"FY{year} interest expense is {share}, as for a lender, broker or "
            "mortgage REIT, so it is likely a funding cost that adding it back "
            "would count twice"
        )

    # ------------------------ public entry point -------------------------- #
    def _common_net_income(self, facts: dict, notes: list[str]) -> dict[int, float]:
        """Annual earnings attributable to common shareholders when tagged.

        Parent earnings exclude NCI but precede preferred distributions.
        ProfitLoss includes NCI: subtract its signed earnings (a subsidiary
        loss is added back). Explicit common earnings already incorporate both
        adjustments and must never be adjusted a second time.
        """
        common = self._annual_flow_by_fy(facts, (_TAG_COMMON_INCOME,))
        parent = self._annual_flow_by_fy(facts, ("NetIncomeLoss",))
        gross = self._annual_flow_by_fy(facts, ("ProfitLoss",))
        minority = self._annual_flow_by_fy(facts, (_TAG_NCI_INCOME,))
        preferred = self._annual_flow_by_fy(facts, _TAGS_PREFERRED_INCOME_ADJUSTMENTS)
        result = dict(common)
        adjusted: list[int] = []
        unadjusted_gross: list[int] = []
        for year in parent.keys() | gross.keys():
            if year in result:
                continue
            value = parent.get(year, gross.get(year))
            changed = False
            if year not in parent:
                if year in minority:
                    value -= minority[year]
                    changed = True
                else:
                    unadjusted_gross.append(year)
            if year in preferred:
                value -= preferred[year]
                changed = True
            result[year] = value
            if changed:
                adjusted.append(year)
        if adjusted:
            notes.append(
                f"net income for {_fy_list(adjusted)} adjusted for reported noncontrolling "
                "income and/or preferred distributions to obtain common-share earnings"
            )
        if unadjusted_gross:
            notes.append(
                f"net income for {_fy_list(unadjusted_gross)} uses consolidated ProfitLoss; "
                "income attributable to noncontrolling interests is not tagged separately, "
                "so earnings may include their share"
            )
        return result

    def _income_attribution_adjusted(
        self, facts: dict, common: dict[int, float], years: list[int]
    ) -> bool:
        """Whether rebuilding common profit from consolidated EBIT is unsafe.

        Keep the reported common-income basis when minority/preferred claims
        matter, or when only common or gross profit is tagged and their
        attribution cannot be reconciled. The models use this marker to avoid
        undoing the provider's common-income normalization.
        """
        parent = self._annual_flow_by_fy(facts, ("NetIncomeLoss",))
        gross = self._annual_flow_by_fy(facts, ("ProfitLoss",))
        minority = self._annual_flow_by_fy(facts, (_TAG_NCI_INCOME,))
        preferred = self._annual_flow_by_fy(facts, _TAGS_PREFERRED_INCOME_ADJUSTMENTS)
        for year in years:
            if year not in common:
                continue
            if minority.get(year, 0.0) or preferred.get(year, 0.0):
                return True
            if any(year in series and common[year] != series[year] for series in (parent, gross)):
                return True
            if year not in parent and (year not in gross or year not in minority):
                return True
        return False

    def get_annual_financials(
        self, ticker: str
    ) -> tuple[AnnualFinancials, BalanceSheetSnapshot, str, str]:
        """Build normalized fundamentals for `ticker` from SEC EDGAR.

        Returns ``(AnnualFinancials, BalanceSheetSnapshot, cik, company_name)``.
        Raises `DataError` if the ticker can't be resolved, if the statements
        are reported in a currency other than USD (see
        `_FOREIGN_UNIT_MIN_SHARE`), if both revenue and net income are entirely
        unavailable, or if one of them ends more than a fiscal year before the
        other (a tag switch we can't follow would otherwise value the company
        on years-old statements).
        """
        cik, name = self.resolve_cik(ticker)
        facts = self.company_facts(cik)
        foreign = self._foreign_reporting_currency(facts)
        if foreign is not None:
            ccy, why, usd = foreign
            incomplete = ("so the USD statements would be incomplete (e.g. D&A, capex, "
                          "debt or cash missing) and are not used")
            if usd == "revenue":
                tail = ("only a few figures are also tagged in USD as convenience "
                        f"translations, {incomplete}")
            elif usd:
                tail = f"only a few figures are tagged in USD, {incomplete}"
            else:
                tail = "its recent statements are not tagged in USD, so they are not used"
            raise DataError(
                f"EDGAR us-gaap statements for {ticker!r} (CIK {cik}) are reported in "
                f"{ccy} ({why}); {tail}"
            )
        notes: list[str] = []
        # Informational notes (expected derivations) go after the data gaps so
        # the material ones lead the list the UI and exports show.
        info_notes: list[str] = []

        # ---- pull each FLOW item as fy -> value -------------------------- #
        revenue_by_fy = self._annual_flow_by_fy(facts, _TAGS_REVENUE)
        ebit_by_fy = self._annual_flow_by_fy(facts, _TAGS_EBIT)
        ni_by_fy = self._common_net_income(facts, info_notes)
        da_by_fy, da_replaced = self._largest_flow_by_fy(facts, _TAGS_DA)
        capex_by_fy, capex_replaced = self._largest_flow_by_fy(
            facts, _TAGS_CAPEX, larger_every_year=True
        )
        interest_by_fy = self._annual_flow_by_fy(facts, _TAGS_INTEREST)
        tax_by_fy = self._annual_flow_by_fy(facts, _TAGS_TAX)
        pretax_by_fy = self._annual_flow_by_fy(facts, _TAGS_PRETAX)
        div_by_fy = self._annual_flow_by_fy(facts, _TAGS_DIVIDENDS)
        shares_by_fy = self._annual_flow_by_fy(facts, _TAGS_DILUTED_SHARES, _UNIT_SHARES)
        nwc_by_fy = self._annual_flow_by_fy(facts, _TAGS_NWC)

        # ---- fatal guard: need revenue OR net income --------------------- #
        if not revenue_by_fy and not ni_by_fy:
            raise DataError(
                f"no annual revenue or net income available on EDGAR for "
                f"{ticker!r} (CIK {cik}); not usable for valuation"
            )

        # ---- staleness guard: the essentials must end in the same year ---- #
        # If a filer moved revenue (or net income) to a tag we don't read, the
        # intersection below would silently end years in the past.
        if revenue_by_fy and ni_by_fy:
            rev_last, ni_last = max(revenue_by_fy), max(ni_by_fy)
            if abs(rev_last - ni_last) > 1:
                raise DataError(
                    f"EDGAR revenue runs to FY{rev_last} but net income to "
                    f"FY{ni_last} for {ticker!r} (CIK {cik}); the lagging series "
                    "moved to a tag this parser does not read, so the EDGAR "
                    "history is stale"
                )
            if rev_last != ni_last:
                lagging = "revenue" if rev_last < ni_last else "net income"
                notes.append(
                    f"{lagging} on EDGAR ends FY{min(rev_last, ni_last)} while the "
                    f"other runs to FY{max(rev_last, ni_last)}; the latest fiscal "
                    "year is left out of the aligned history"
                )

        # ---- choose the aligned fiscal-year axis ------------------------- #
        # Anchor on whichever of the two essential series is present; intersect
        # with the other essential series when both exist so every retained year
        # has at least revenue and net income.
        if revenue_by_fy and ni_by_fy:
            year_set = set(revenue_by_fy) & set(ni_by_fy)
            if not year_set:
                # No overlap: fall back to the union of the essentials and note
                # the gaps (each missing essential is filled below).
                year_set = set(revenue_by_fy) | set(ni_by_fy)
                notes.append(
                    "revenue and net income reported for disjoint fiscal years; "
                    "missing essentials filled with 0.0"
                )
        elif revenue_by_fy:
            year_set = set(revenue_by_fy)
            notes.append("net income unavailable on EDGAR; filled with 0.0")
        else:
            year_set = set(ni_by_fy)
            notes.append("revenue unavailable on EDGAR; filled with 0.0")

        # Keep the most recent _MAX_YEARS, ordered oldest -> newest.
        years = sorted(year_set)[-_MAX_YEARS:]
        if not years:
            # Defensive: the essentials existed but produced no usable fy axis.
            raise DataError(
                f"could not align any fiscal year for {ticker!r} (CIK {cik})"
            )
        if len(years) < _MIN_YEARS:
            notes.append(
                f"only {len(years)} annual period(s) available on EDGAR "
                f"(target is {_MIN_YEARS}-{_MAX_YEARS})"
            )

        # ---- helper to align one flow series onto `years` ---------------- #
        def _align(
            by_fy: dict[int, float],
            *,
            positive: bool = False,
            label: str = "",
            note_if_empty: bool = False,
            gap_notes: Optional[list[str]] = None,
        ) -> list[float]:
            """Project `by_fy` onto `years`, filling gaps with 0.0.

            `positive=True` stores the absolute magnitude (capex, D&A, etc. are
            signed differently across filers; the schemas want +X).
            With `note_if_empty`, a wholly missing series and any individual
            missing years are both noted (in `gap_notes`, default `notes`).
            """
            out: list[float] = []
            missing: list[int] = []
            for y in years:
                v = by_fy.get(y)
                if v is None or not is_num(v):
                    out.append(0.0)
                    missing.append(y)
                else:
                    out.append(abs(v) if positive else float(v))
            if note_if_empty:
                if not by_fy:
                    notes.append(f"{label} unavailable on EDGAR; filled with 0.0")
                elif missing:
                    (notes if gap_notes is None else gap_notes).append(
                        f"{label} not reported on EDGAR for {_fy_list(missing)}; "
                        "filled with 0.0"
                    )
            return out

        def _largest_tag_notes(
            replaced: dict[int, tuple[str, str]], item: str, why: str
        ) -> None:
            """Info notes naming the years where a later tag's larger `item`
            figure replaced the first tag's (see `_largest_flow_by_fy`),
            grouped by the two tags, with `why` the larger one is used."""
            swaps: dict[tuple[str, str], list[int]] = {}
            for y in years:
                if y in replaced:
                    swaps.setdefault(replaced[y], []).append(y)
            for (used, passed_over), swap_years in swaps.items():
                info_notes.append(
                    f"{item} for {_fy_list(swap_years)} read from {used}, the largest "
                    f"{item} figure tagged for those years, over the smaller {passed_over} "
                    f"({why})"
                )

        # D&A where a later tag's larger figure replaced the first tag's (see
        # _TAGS_DA).
        _largest_tag_notes(
            da_replaced, "D&A",
            "each D&A tag is the total or a part of it, so the largest is used and none "
            "are added",
        )

        # Pretax income tagged only as its domestic and foreign parts (see
        # _TAGS_PRETAX_PARTS): their sum fills the years without a total.
        pretax_parts = [self._annual_flow_by_fy(facts, (t,)) for t in _TAGS_PRETAX_PARTS]
        summed_pretax = [
            y for y in years
            if y not in pretax_by_fy and all(y in part for part in pretax_parts)
        ]
        if summed_pretax:
            pretax_by_fy = dict(pretax_by_fy)
            for y in summed_pretax:
                pretax_by_fy[y] = sum(part[y] for part in pretax_parts)
            info_notes.append(
                f"pretax income not tagged as a total on EDGAR for "
                f"{_fy_list(summed_pretax)}; summed from its domestic and foreign parts"
            )

        # Banks, insurers, BDCs, REITs, debt-funded lessors and groups with a
        # captive finance arm: flagged up front, because the FCFF-style models
        # do not fit them (nor, for banks and BDCs, the EBIT derivation below).
        fin_kind, fin_evidence = self._financial_kind(facts, revenue_by_fy)
        fin_reason = _FINANCIAL_KINDS[fin_kind][1] if fin_kind is not None else ""
        lessor = (self._lessor_evidence(facts, years[-1], revenue_by_fy)
                  if fin_kind is None else None)
        if lessor is not None:
            fin_kind, fin_evidence = "lessor", lessor
            fin_reason = _FINANCIAL_KINDS[fin_kind][1]
        elif fin_kind is None:
            captive = self._captive_finance_evidence(
                facts, years[-1], revenue_by_fy, interest_by_fy
            )
            if captive is not None:
                fin_kind = "captive_finance"
                fin_evidence, fin_reason = captive
        if fin_kind is not None:
            fin_label = _FINANCIAL_KINDS[fin_kind][0]
            notes.insert(0, (
                f"WARNING: EDGAR tags mark this company as {fin_label} "
                f"({fin_evidence}); {fin_reason}, so the FCFF DCF and FCFE do "
                "not fit it and the DDM and comps are the better guides"
            ))

        # EBIT: OperatingIncomeLoss where tagged. Filers without an operating-
        # income subtotal (many pharma, insurers and REITs) get it derived per
        # year instead of a silent 0.0: pretax income + interest expense (the
        # textbook EBIT, both already parsed), else revenue - CostsAndExpenses.
        # Not where interest is a cost of the lending book (see
        # _EBIT_NOT_DERIVED_KINDS): adding it back would count it twice, so the
        # missing years stay 0.0 with a note. A pretax-based figure above
        # revenue (a gain in pretax income, or interest income left out of
        # revenue) is not used either.
        lender_reason = self._lender_reason(
            fin_kind, years[-1], revenue_by_fy, interest_by_fy
        )
        ebit_by_fy = dict(ebit_by_fy)
        costs_by_fy: Optional[dict[int, float]] = None
        from_pretax: list[int] = []
        from_costs: list[int] = []
        not_derived: list[int] = []
        above_revenue: list[int] = []
        for y in years:
            if ebit_by_fy.get(y) is not None:
                continue
            if lender_reason is not None:
                not_derived.append(y)
                continue
            pti = pretax_by_fy.get(y)
            rev = revenue_by_fy.get(y)
            if pti is not None:
                derived = pti + abs(interest_by_fy.get(y, 0.0))
                if derived <= (rev or 0.0):
                    ebit_by_fy[y] = derived
                    from_pretax.append(y)
                    continue
                above_revenue.append(y)
            if costs_by_fy is None:
                costs_by_fy = self._annual_flow_by_fy(facts, _TAGS_COSTS_AND_EXPENSES)
            costs = costs_by_fy.get(y)
            if rev is not None and costs is not None:
                ebit_by_fy[y] = rev - abs(costs)
                from_costs.append(y)
        if from_pretax:
            notes.append(
                f"EBIT (OperatingIncomeLoss) not reported on EDGAR for "
                f"{_fy_list(from_pretax)}; derived as pretax income + interest expense"
            )
        if above_revenue:
            notes.append(
                f"EBIT (OperatingIncomeLoss) not reported on EDGAR for "
                f"{_fy_list(above_revenue)}, and pretax income + interest expense "
                "exceeds revenue there, so it is not used as EBIT"
            )
        if from_costs:
            notes.append(
                f"EBIT (OperatingIncomeLoss) not reported on EDGAR for "
                f"{_fy_list(from_costs)}; derived as revenue - CostsAndExpenses"
            )
        if not_derived:
            notes.append(
                f"EBIT (OperatingIncomeLoss) not reported on EDGAR for "
                f"{_fy_list(not_derived)}; left at 0.0 rather than derived as "
                f"pretax income + interest expense: {lender_reason}"
            )

        # Capex where the second tag, larger in every year both report,
        # replaced the first tag's figure (see _TAGS_CAPEX). The years neither
        # reports take the other-PP&E line (see _TAG_OTHER_PPE), then the
        # generic other-productive-assets line (see
        # _TAG_OTHER_PRODUCTIVE_ASSETS), which may be only part of capex:
        # named, so they can be told from a main capex tag.
        _largest_tag_notes(
            capex_replaced, "capex",
            "it is larger in every year both tags report, so it is read as the total and "
            "the smaller as a part of it; the two are not added",
        )
        main_capex = {**self._annual_flow_by_fy(facts, (_TAG_OTHER_PPE,)), **capex_by_fy}
        other_productive = self._annual_flow_by_fy(facts, (_TAG_OTHER_PRODUCTIVE_ASSETS,))
        other_only = [y for y in years if y in other_productive and y not in main_capex]
        capex_by_fy = {**other_productive, **main_capex}
        if other_only:
            info_notes.append(
                _OTHER_PRODUCTIVE_ASSETS_NOTE.format(years=_fy_list(other_only))
                + " (payments for other productive assets), the only capex line tagged "
                "for those years; for some filers it is all of capex, for others only "
                "a part, so capex there may be understated"
            )

        # Net purchases of a finance arm's lease fleet join capex in the years
        # capex is reported (see _TAG_LEASE_FLEET_PURCHASES), so capex matches
        # a D&A that includes the fleet's depreciation. A shrinking fleet
        # (proceeds above purchases) lowers capex, never below 0. A year whose
        # capex is the second tag's total, above the first tag's figure by
        # these purchases, already includes them, so nothing is added there.
        fleet_buys = self._annual_flow_by_fy(facts, (_TAG_LEASE_FLEET_PURCHASES,))
        fleet_years = [y for y in years if y in fleet_buys and y in capex_by_fy]
        if fleet_years and capex_replaced:
            first_capex = self._annual_flow_by_fy(facts, _TAGS_CAPEX[:1])
            fleet_years = [
                y for y in fleet_years
                if y not in capex_replaced or abs(
                    abs(capex_by_fy[y]) - abs(first_capex[y]) - abs(fleet_buys[y])
                ) > _AMOUNT_MATCH_TOLERANCE * abs(fleet_buys[y])
            ]
        if fleet_years:
            fleet_proceeds = self._annual_flow_by_fy(facts, (_TAG_LEASE_FLEET_PROCEEDS,))
            capex_by_fy = dict(capex_by_fy)
            for y in fleet_years:
                net = abs(fleet_buys[y]) - abs(fleet_proceeds.get(y, 0.0))
                capex_by_fy[y] = max(abs(capex_by_fy[y]) + net, 0.0)
            info_notes.append(
                f"capex for {_fy_list(fleet_years)} includes net purchases of "
                f"vehicles and equipment for operating leases ({_TAG_LEASE_FLEET_PURCHASES} "
                f"less {_TAG_LEASE_FLEET_PROCEEDS}), since D&A includes the lease "
                "fleet's depreciation"
            )

        revenue = _align(revenue_by_fy, label="revenue")
        ebit = _align(
            ebit_by_fy, label="EBIT (operating income)", note_if_empty=not not_derived
        )
        net_income = _align(ni_by_fy, label="net income")
        dep_amort = _align(da_by_fy, positive=True, label="D&A", note_if_empty=True)
        capex = _align(capex_by_fy, positive=True, label="capex", note_if_empty=True)
        interest_expense = _align(
            interest_by_fy, positive=True, label="interest expense",
            note_if_empty=True, gap_notes=info_notes,
        )
        # Signed: IncomeTaxExpenseBenefit is negative for a tax benefit, and the
        # effective-tax-rate median needs that sign.
        tax_expense = _align(tax_by_fy, label="tax expense", note_if_empty=True)
        pretax_income = _align(
            pretax_by_fy, label="pretax income", note_if_empty=True
        )
        dividends_paid = _align(
            div_by_fy, positive=True, label="dividends paid",
            note_if_empty=True, gap_notes=info_notes,
        )
        diluted_shares = _align(
            shares_by_fy, label="diluted shares", note_if_empty=True
        )

        # ΔNWC (positive = increase = cash use) from the aggregate operating-
        # capital tag only; see _TAGS_NWC for why the component lines are not
        # summed instead.
        if nwc_by_fy:
            change_in_nwc = _align(nwc_by_fy, label="change in NWC")
        else:
            change_in_nwc = [0.0 for _ in years]
            info_notes.append(
                "change in net working capital (IncreaseDecreaseInOperatingCapital) "
                "not reported on EDGAR; left as 0.0, so the derived working-capital "
                "drag is zero unless nwc_pct_revenue is set (a sum of only the "
                "receivables, inventories and payables lines would leave out "
                "contract liabilities and accruals)"
            )

        # EBITDA is computed, never looked up: EBITDA = EBIT + D&A per year.
        ebitda = [e + d for e, d in zip(ebit, dep_amort)]

        financials = AnnualFinancials(
            fiscal_years=list(years),
            revenue=revenue,
            ebit=ebit,
            ebitda=ebitda,
            net_income=net_income,
            dep_amort=dep_amort,
            capex=capex,
            change_in_nwc=change_in_nwc,
            interest_expense=interest_expense,
            tax_expense=tax_expense,
            pretax_income=pretax_income,
            dividends_paid=dividends_paid,
            diluted_shares=diluted_shares,
        )

        # ---- balance sheet (INSTANT facts at one snapshot date) ----------- #
        balance_sheet = self._build_balance_sheet(facts, notes, info_notes)

        # Debt the tags do not cover shows up as interest expense far above
        # any plausible rate on the debt found (Ford and Berkshire tag their
        # debt only with company-specific elements). Banks are skipped: their
        # interest expense is mostly on deposits, which are not debt here. A
        # company that repaid its debt during the year (SanDisk, SailPoint)
        # gets an informational note instead: the interest fits the average
        # debt on that fiscal year's balance sheets. A one-date peak would not
        # do: a filer that moves its debt to company-specific tags early in
        # the year leaves one large opening figure. Keep the WARNING's wording
        # stable: HybridProvider (data/provider.py, `_missing_debt_warning`)
        # finds it to backfill debt from Yahoo by the "WARNING" prefix, the words
        # "interest expense" and one of its `_MISSING_DEBT_PHRASES` ("no debt was
        # found", "company-specific", "debt weight may be understated").
        latest_interest = interest_expense[-1] if interest_expense else 0.0
        if fin_kind != "bank" and latest_interest > 0:
            debt = balance_sheet.total_debt
            implied = latest_interest / _MAX_PLAUSIBLE_INTEREST_RATE
            tolerance = _MIN_DEBT_SHORTFALL_REVENUE_SHARE * max(revenue[-1], 0.0)
            if implied - debt > tolerance:
                during = self._mean_debt_in_year(facts, years[-1])
                if during is not None and implied - during[0] <= tolerance:
                    info_notes.append(
                        f"FY{years[-1]} interest expense ({latest_interest:,.0f}) "
                        f"fits the debt carried during that year (total debt "
                        f"averaged {during[0]:,.0f} on the balance sheets from "
                        f"{during[1]} to {during[2]}); total debt is {debt:,.0f} "
                        f"at the {balance_sheet.as_of} balance sheet, so it was "
                        "repaid or is tagged differently there"
                    )
                else:
                    found = (
                        "no debt was found" if debt <= 0
                        else f"total debt read is only {debt:,.0f}"
                    )
                    # After the financial-kind WARNING, which stays first.
                    notes.insert(0 if fin_kind is None else 1, (
                        f"WARNING: FY{years[-1]} interest expense is "
                        f"{latest_interest:,.0f} but {found} under the SEC debt "
                        "tags this parser reads (the filer may tag its debt with "
                        "company-specific elements); net debt and the WACC debt "
                        "weight may be understated"
                    ))

        # The schemas have no notes field on these dataclasses, so the parsing
        # notes ride on the financials object as a dynamic attribute that
        # HybridProvider copies into CompanyData.source_notes (data gaps first,
        # informational derivations last). `_financial_kind` ("bank",
        # "insurer", "bdc", "reit", "lessor", "captive_finance" or None) rides
        # the same way, for models that rebuild EBIT or choose methods.
        try:
            setattr(financials, "_source_notes", notes + info_notes)
            setattr(financials, "_financial_kind", fin_kind)
            setattr(financials, "_income_attribution_adjusted",
                    self._income_attribution_adjusted(facts, ni_by_fy, years))
        except Exception:  # pragma: no cover - dataclasses allow attr set
            pass

        return financials, balance_sheet, cik, name

    # --------------------------- balance sheet ---------------------------- #
    def _build_balance_sheet(
        self, facts: dict, notes: list[str], info_notes: Optional[list[str]] = None
    ) -> BalanceSheetSnapshot:
        """Assemble the latest `BalanceSheetSnapshot` from instant facts.

        Every item is read at ONE snapshot date (the latest date at which equity
        or cash is reported), so a line the company stopped reporting years ago
        is not summed into today's balance sheet (see `_instant_at`). Missing
        essentials are noted in `notes`; date adjustments in `info_notes`.
        """
        date_notes = info_notes if info_notes is not None else notes

        def latest_end(tag_groups: tuple[tuple[str, ...], ...]) -> Optional[str]:
            ends = [
                end
                for tags in tag_groups
                for tag in tags
                for end in self._instant_by_end(facts, tag)
            ]
            return max(ends) if ends else None

        # The snapshot date follows equity and balance-sheet cash & equivalents
        # (the other cash tags only back that line up).
        as_of = latest_end((_TAGS_EQUITY, _TAGS_CASH[:1])) or latest_end((
            _TAGS_LTD_NONCURRENT, _TAGS_LTD_CURRENT, _TAGS_ST_BORROWINGS,
            _TAGS_DEBT_CURRENT, _TAGS_LTD_TOTAL, _TAGS_DEBT_COMBINED,
            _TAGS_CASH[1:], _TAGS_ST_INVEST, _TAGS_MINORITY, _TAGS_PREFERRED,
        ))
        if as_of is None:
            notes.append("no balance-sheet facts on EDGAR; debt, cash and equity set to 0.0")
            return BalanceSheetSnapshot(
                as_of="", total_debt=0.0, cash_and_investments=0.0, total_equity=0.0,
            )

        # Dates of the filer's balance sheets (total assets reported; no
        # restriction for a filer that never tags total assets).
        bs_dates = set(self._instant_by_end(facts, _TAGS_ASSETS[0])) or None

        def at(tags: tuple[str, ...], label: str) -> Optional[float]:
            return self._instant_at(facts, tags, as_of, date_notes, label, bs_dates)

        total_debt = self._total_debt(facts, as_of, bs_dates, notes, date_notes)

        cash_and_investments = self._cash_and_investments(
            facts, as_of, notes, date_notes, bs_dates
        )

        # Total common equity (book value).
        equity, equity_tag, equity_end = self._instant_fact(
            facts, _TAGS_EQUITY, as_of, date_notes, "stockholders' equity", bs_dates
        )
        if equity is None:
            notes.append("total equity unavailable on EDGAR; set to 0.0")

        # Minority (noncontrolling) interest and preferred stock -- optional,
        # default 0; both are claims the equity bridges subtract from EV.
        minority = at(_TAGS_MINORITY, "minority interest")
        preferred = at(_TAGS_PREFERRED, "preferred stock")

        # The schema and P/B/ROE models require common equity. Parent equity
        # includes preferred stock; the consolidated fallback also includes
        # noncontrolling interests. Subtract claims at the equity's own date
        # when a stale equity line is used, rather than a newer bridge amount.
        if equity is not None:
            if equity_tag == _TAGS_EQUITY[1]:
                equity -= self._instant_at(
                    facts, _TAGS_MINORITY, equity_end, date_notes,
                    "minority interest for common equity", bs_dates,
                ) or 0.0
            equity -= self._instant_at(
                facts, _TAGS_PREFERRED_BOOK, equity_end, date_notes,
                "preferred stock for common equity", bs_dates,
            ) or 0.0

        return BalanceSheetSnapshot(
            as_of=as_of,
            total_debt=float(total_debt),
            cash_and_investments=float(cash_and_investments),
            total_equity=float(equity or 0.0),
            minority_interest=float(minority or 0.0),
            preferred_equity=float(preferred or 0.0),
        )

    def _total_debt(
        self,
        facts: dict,
        as_of: str,
        bs_dates: Optional[set[str]],
        notes: list[str],
        date_notes: list[str],
    ) -> float:
        """Total debt at the balance-sheet date `as_of` (see `_instant_fact`).

        Gaps are noted in `notes`, date adjustments and derivations in
        `date_notes`.
        """
        def at(tags: tuple[str, ...], label: str) -> Optional[float]:
            return self._instant_at(facts, tags, as_of, date_notes, label, bs_dates)

        # Total debt = noncurrent LTD + current debt, where current debt is
        # DebtCurrent if tagged (it already includes the current maturities of
        # LTD and short-term borrowings, so never add those to it) or else the
        # sum of those components. Without a noncurrent tag: a long-term total
        # that includes its current maturities plus short-term borrowings, then
        # a short- plus long-term total, then the separate debt lines.
        ltd_nc = at(_TAGS_LTD_NONCURRENT, "long-term debt (noncurrent)")
        ltd_cur = at(_TAGS_LTD_CURRENT, "current portion of long-term debt")
        st_borrow = at(_TAGS_ST_BORROWINGS, "short-term borrowings")
        debt_cur = at(_TAGS_DEBT_CURRENT, "current debt")
        if debt_cur is not None:
            current_debt: Optional[float] = debt_cur
        elif ltd_cur is not None or st_borrow is not None:
            current_debt = (ltd_cur or 0.0) + (st_borrow or 0.0)
        else:
            current_debt = None

        if ltd_nc is not None:
            total_debt = ltd_nc + (current_debt or 0.0)
        else:
            # Short-term borrowings only (current maturities are already in a
            # long-term total or a debt line).
            if st_borrow is not None:
                short_only = st_borrow
            elif debt_cur is not None and ltd_cur is not None:
                short_only = max(debt_cur - ltd_cur, 0.0)
            else:
                short_only = 0.0
            ltd_total = at(_TAGS_LTD_TOTAL, "long-term debt")
            combined = None if ltd_total is not None else at(
                _TAGS_DEBT_COMBINED, "total debt")
            lines = (
                None if ltd_total is not None or combined is not None
                else self._debt_lines(at, st_borrow)
            )
            if ltd_total is not None:
                total_debt = ltd_total + short_only
            elif combined is not None:
                total_debt = combined
            elif lines is not None:
                # Lines that all exclude their current portion (CME's
                # UnsecuredLongTermDebt) take all current debt on top. Never
                # below the current debt alone: the lines may leave out debt
                # tagged only as DebtCurrent.
                lines_total, used = lines
                if all(t in _DEBT_LINES_NONCURRENT for t in used):
                    extra, extra_label = current_debt or 0.0, "current debt"
                else:
                    extra, extra_label = short_only, "short-term borrowings"
                total_debt = max(lines_total + extra, current_debt or 0.0)
                date_notes.append(
                    "no total-debt tag on EDGAR; total debt summed from the "
                    f"balance-sheet debt lines ({', '.join(used)})"
                    + (f" plus {extra_label}" if extra else "")
                    + (" (current debt alone is larger and is used)"
                       if total_debt > lines_total + extra else "")
                )
            elif current_debt is not None:
                total_debt = current_debt
                notes.append(
                    "no long-term debt found under the debt tags read from EDGAR; "
                    "total debt counts current debt only"
                )
            else:
                total_debt = 0.0
                notes.append("total debt unavailable on EDGAR; set to 0.0")
        return total_debt

    def _mean_debt_in_year(
        self, facts: dict, year: int
    ) -> Optional[tuple[float, str, str]]:
        """``(average total debt, first date, last date)`` over fiscal `year`.

        Averages total debt on every balance sheet (a date with total assets
        reported) from the one that opens fiscal `year` (dated within
        `_YEAR_START_BALANCE_SHEET_DAYS` before its first day) to the one that
        closes it: the debt that year's interest expense was paid on. None
        without at least two such balance sheets.
        """
        period = self._fiscal_year_period(facts, _TAGS_INTEREST + _TAGS_REVENUE, year)
        bs_dates = set(self._instant_by_end(facts, _TAGS_ASSETS[0]))
        if period is None or not bs_dates:
            return None
        start, end = period
        dates = []
        for d in sorted(bs_dates):
            lead = _days_between(d, start)
            if d <= end and lead is not None and lead <= _YEAR_START_BALANCE_SHEET_DAYS:
                dates.append(d)
        if len(dates) < 2:
            return None
        debts = [self._total_debt(facts, d, bs_dates, [], []) for d in dates]
        return sum(debts) / len(debts), dates[0], dates[-1]

    @staticmethod
    def _debt_lines(at, st_borrow: Optional[float]) -> Optional[tuple[float, list[str]]]:
        """``(sum, tags used)`` of the balance-sheet debt lines, or None.

        `at(tags, label)` reads one item at the snapshot date. Each group of
        `_DEBT_LINE_GROUPS` contributes its first tag present. Before that, a
        line equal to another line, or to the short-term borrowings the caller
        adds, is the same debt and counts once, and a line equal to the sum of
        two or more others is their total (e.g. NotesPayable = UnsecuredDebt +
        SecuredDebt), so those others are dropped. None when no debt line is
        reported.
        """
        found: dict[str, float] = {}
        for group in _DEBT_LINE_GROUPS:
            for tag in group:
                val = at((tag,), f"debt line ({tag})")
                if val is not None:
                    found[tag] = val
        if not any(v > 0 for v in found.values()):
            return None

        def close(a: float, b: float) -> bool:
            return abs(a - b) <= _AMOUNT_MATCH_TOLERANCE * max(abs(a), abs(b))

        # Drop a line already contained in another: a duplicate of a line (or
        # of the short-term borrowings added later), or a component of a line
        # that equals the sum of two or more others.
        pool = {t: v for t, v in found.items() if v > 0}
        dropped: set[str] = set()
        tags = sorted(pool, key=pool.get, reverse=True)
        for i, tag in enumerate(tags):
            if tag in dropped:
                continue
            if st_borrow and close(pool[tag], st_borrow):
                dropped.add(tag)
                continue
            for other in tags[i + 1:]:
                if other not in dropped and close(pool[tag], pool[other]):
                    dropped.add(other)
            rest = [t for t in tags[i + 1:] if t not in dropped]
            for size in range(2, len(rest) + 1):
                hit = next(
                    (combo for combo in combinations(rest, size)
                     if close(pool[tag], sum(pool[t] for t in combo))),
                    None,
                )
                if hit:
                    dropped.update(hit)
                    break

        total = 0.0
        used: list[str] = []
        for group in _DEBT_LINE_GROUPS:
            tag = next((t for t in group if t in pool and t not in dropped), None)
            if tag is not None:
                total += pool[tag]
                used.append(tag)
        return (total, used) if used else None

    def _cash_and_investments(
        self,
        facts: dict,
        as_of: str,
        notes: list[str],
        date_notes: list[str],
        bs_dates: Optional[set[str]] = None,
    ) -> float:
        """Cash & equivalents plus short-term investments at `as_of`.

        Cash comes from the first `_TAGS_CASH` tag at the snapshot (see
        `_instant_fact`). When that is a cash-flow total including restricted
        cash (`_TAGS_CASH_WITH_RESTRICTED`), restricted cash tagged at the
        same date is taken out (a total tag, else current + noncurrent),
        unless a plain `Cash` line at that date equals the total (then
        nothing restricted is inside it); without a restricted-cash tag the
        total is used as is, with a note. A missing cash line is noted and
        counted as 0.0.
        """
        cash, tag, end = self._instant_fact(
            facts, _TAGS_CASH, as_of, date_notes, "cash & equivalents", bs_dates
        )
        if cash is None:
            notes.append("cash & equivalents unavailable on EDGAR; set to 0.0")
        elif tag in _TAGS_CASH_WITH_RESTRICTED:
            def same_date(tags: tuple[str, ...]) -> Optional[float]:
                for t in tags:
                    val = self._instant_by_end(facts, t).get(end)
                    if val is not None:
                        return val
                return None

            restricted = same_date(_TAGS_RESTRICTED_CASH)
            plain = same_date(("Cash",))
            if plain is not None and abs(plain - cash) <= _AMOUNT_MATCH_TOLERANCE * abs(cash):
                # The balance-sheet Cash line equals the total, so any restricted
                # cash the filer tags is held outside it (e.g. in investments).
                restricted = 0.0
            elif restricted is None:
                cur = same_date(_TAGS_RESTRICTED_CASH_CURRENT)
                noncur = same_date(_TAGS_RESTRICTED_CASH_NONCURRENT)
                if cur is not None or noncur is not None:
                    restricted = (cur or 0.0) + (noncur or 0.0)
            disposal = (" and the cash of disposal groups"
                        if tag == _TAG_CASH_WITH_RESTRICTED_AND_DISPOSAL else "")
            total_note = (
                f"cash & equivalents from the cash-flow total including "
                f"restricted cash{disposal} ({cash:,.0f}) at {end}"
            )
            if restricted is not None and 0.0 < restricted <= cash:
                date_notes.append(f"{total_note}, less restricted cash ({restricted:,.0f})")
                cash -= restricted
            elif restricted is None:
                date_notes.append(
                    f"{total_note}; restricted cash is not tagged separately, so "
                    "any restricted balance is included"
                )
            else:
                date_notes.append(total_note)
        elif tag != _TAGS_CASH[0]:
            date_notes.append(f"cash & equivalents read from the {tag} line")
        st_inv, _, st_end = self._instant_fact(
            facts, _TAGS_ST_INVEST, as_of, date_notes, "short-term investments", bs_dates
        )
        if st_inv:
            # Some filers (Target) count short-term investments inside cash
            # equivalents: their cash + short-term investments total then
            # equals cash alone, and adding the investments would double count.
            combined = self._instant_by_end(
                facts, "CashCashEquivalentsAndShortTermInvestments"
            ).get(st_end)
            cash_then = next(
                (v for v in (self._instant_by_end(facts, t).get(st_end)
                             for t in _TAGS_CASH[:1] + _TAGS_CASH_WITH_RESTRICTED)
                 if v is not None),
                None,
            )
            if (combined is not None and cash_then is not None
                    and abs(combined - cash_then) <= _AMOUNT_MATCH_TOLERANCE * abs(combined)):
                date_notes.append(
                    f"short-term investments ({st_inv:,.0f}) are part of cash & "
                    f"equivalents at {st_end}; not added again"
                )
                st_inv = None
        return (cash or 0.0) + (st_inv or 0.0)
