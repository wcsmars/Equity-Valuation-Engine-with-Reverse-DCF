"""Canonical data contracts for the equity-valuation engine.

Every module in the package binds to the dataclasses defined here. They are the
single source of truth for field names and units, so DO NOT redefine or rename
fields in other modules -- import them from here.

UNIT CONVENTIONS (read carefully):
  * All monetary amounts are in the quote currency (`MarketData.currency`, major
    unit), in *absolute* units (NOT millions), e.g. revenue of
    391_035_000_000.0 for AAPL FY2024. Statements reported in another currency
    are converted at one spot rate by the provider, or flagged with a WARNING
    note when no rate is available.
  * Share counts are absolute (e.g. 15_300_000_000.0 shares), never millions.
  * Per-share values are in currency units per share.
  * Rates / percentages are decimals: 8% -> 0.08, never 8.0.
  * `capex`, `dep_amort`, `dividends_paid`, `interest_expense`
    are stored as POSITIVE magnitudes (a cash outflow for capex is stored as +X,
    the models apply the sign). `tax_expense` keeps its sign: a tax benefit is
    negative. `change_in_nwc` is signed: an increase is a positive cash use,
    and a decrease is a negative cash use (a cash release).
  * Annual series (lists) are ordered OLDEST -> NEWEST. The last element is the
    most recent fiscal year and aligns with `BalanceSheetSnapshot`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
#  Raw / normalized company data (output of the data providers)
# --------------------------------------------------------------------------- #
@dataclass
class AnnualFinancials:
    """Normalized annual income-statement & cash-flow history (oldest -> newest)."""

    fiscal_years: list[int]
    revenue: list[float]
    ebit: list[float]               # operating income (EBIT)
    ebitda: list[float]             # EBIT + D&A
    net_income: list[float]
    dep_amort: list[float]          # depreciation & amortization (positive add-back)
    capex: list[float]              # capital expenditures (positive magnitude)
    change_in_nwc: list[float]      # increase in net working capital (positive = cash use)
    interest_expense: list[float]   # gross interest expense (positive)
    tax_expense: list[float]        # income-tax expense (a tax benefit is negative)
    pretax_income: list[float]      # pre-tax income (EBT)
    dividends_paid: list[float]     # total common dividends paid (positive magnitude)
    diluted_shares: list[float]     # weighted-average diluted shares

    def latest(self, attr: str) -> float:
        return getattr(self, attr)[-1]


@dataclass
class BalanceSheetSnapshot:
    """Most-recent balance-sheet items needed for net debt & WACC weights."""

    as_of: str                      # ISO date of the snapshot
    total_debt: float               # short-term + long-term interest-bearing debt
    cash_and_investments: float     # cash + cash equivalents + short-term investments
    total_equity: float             # common stockholders' equity (book value)
    minority_interest: float = 0.0
    preferred_equity: float = 0.0
    # Operating-lease liabilities that total_debt leaves out (EDGAR's
    # borrowings), for the comps EV bridge on the peers' lease-inclusive
    # basis. None when unknown or already in total_debt (Yahoo's and IFRS
    # filers' debt includes leases). Not part of net_debt.
    operating_lease_liability: Optional[float] = None

    @property
    def net_debt(self) -> float:
        return self.total_debt - self.cash_and_investments


@dataclass
class MarketData:
    """Live market data (typically from yfinance)."""

    ticker: str
    name: str
    currency: str
    price: float
    shares_outstanding: float
    market_cap: float
    # The CAPM beta. From yfinance it is Blume-adjusted (0.67 x raw + 0.33);
    # `raw_beta` keeps Yahoo's unadjusted figure (None when not adjusted).
    beta: Optional[float] = None
    # Annual DPS used as the DDM's D0: Yahoo's indicated regular rate, plus the
    # 3-year average of recurring variable/special dividends when it omits them.
    dividend_per_share: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    raw_beta: Optional[float] = None
    # Yahoo's industry identifier (info['industryKey'], e.g.
    # "beverages-non-alcoholic"); opt-in peer discovery reads that industry's list.
    industry_key: Optional[str] = None
    # Yahoo's trailing-twelve-month revenue, EBITDA and net income to common
    # (info totalRevenue, ebitda, netIncomeToCommon), absolute amounts in the
    # quote currency. Filled only when Yahoo's reporting currency
    # (`ttm_currency`) is `currency`: no FX conversion is applied. Read only by
    # comps: the opt-in trailing basis (``models.comps.run_comps(basis="ttm")``)
    # and the default basis's fiscal-year vs trailing gap note; every other
    # model uses the fiscal-year statements. Not screened for one-off items.
    ttm_revenue: Optional[float] = None
    ttm_ebitda: Optional[float] = None
    ttm_net_income: Optional[float] = None
    # ISO dates (UTC) of the quarter the trailing figures run to
    # (info mostRecentQuarter) and of the latest fiscal year end
    # (info lastFiscalYearEnd).
    ttm_as_of: Optional[str] = None
    fiscal_year_end: Optional[str] = None
    # Yahoo's reporting currency (info financialCurrency, major unit), which
    # the trailing figures are in; when it is not `currency` they are None.
    ttm_currency: Optional[str] = None


@dataclass
class CompanyData:
    """Everything the engine needs about one company."""

    ticker: str
    name: str
    cik: Optional[str]
    financials: AnnualFinancials
    balance_sheet: BalanceSheetSnapshot
    market: MarketData
    source_notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Assumptions
# --------------------------------------------------------------------------- #
@dataclass
class MacroAssumptions:
    """Market-wide assumptions feeding CAPM / WACC."""

    risk_free_rate: float = 0.042            # 10y treasury, decimal
    equity_risk_premium: float = 0.05        # market ERP, decimal
    tax_rate: Optional[float] = None         # marginal tax; None -> derive effective
    pretax_cost_of_debt: Optional[float] = None  # None -> derive interest/debt or rf+spread


@dataclass
class DCFAssumptions:
    """Drivers for the unlevered (FCFF) DCF projection."""

    forecast_years: int = 5
    # Per-year revenue growth (decimals). If None, the model derives a fading path
    # from historical CAGR toward `terminal_growth`.
    revenue_growth: Optional[list[float]] = None
    terminal_growth: float = 0.025
    terminal_method: str = "gordon"          # "gordon" | "exit_multiple" | "value_driver"
    exit_ev_ebitda: Optional[float] = None   # required if terminal_method == "exit_multiple"
    # Operating drivers. If None, derived from the trailing historical average and
    # held flat (or faded to the target where given).
    target_ebit_margin: Optional[float] = None
    capex_pct_revenue: Optional[float] = None
    da_pct_revenue: Optional[float] = None
    nwc_pct_revenue: Optional[float] = None  # incremental NWC as % of revenue change
    # Overrides the tax rate on NOPAT in the FCFF projection. The WACC debt tax
    # shield keeps using MacroAssumptions.tax_rate (else the historical effective rate).
    tax_rate: Optional[float] = None
    mid_year_convention: bool = True
    # Return on new invested capital (decimal) for terminal_method
    # "value_driver": the terminal FCFF is NOPAT x (1 - g / RONIC). None, or
    # anything but a positive number above the terminal growth used, uses WACC
    # (growth adds no value; a note says so). Ignored by the other methods.
    terminal_ronic: Optional[float] = None
    # Year-1 revenue growth (decimal) for the derived path: when revenue_growth
    # is None the path starts here instead of the historical CAGR and fades to
    # the terminal growth actually used (after any clamp below WACC), as the
    # derived path does. Ignored when revenue_growth is given.
    revenue_growth_start: Optional[float] = None


@dataclass
class DDMAssumptions:
    """Drivers for dividend-discount and FCFE models."""

    method: str = "two_stage"                # "gordon" | "two_stage" | "h_model"
    high_growth_years: int = 5
    high_growth_rate: Optional[float] = None # None -> derive from ROE * retention or DPS CAGR
    terminal_growth: float = 0.025
    forecast_years: int = 5                  # for the FCFE projection


# --------------------------------------------------------------------------- #
#  Model results
# --------------------------------------------------------------------------- #
@dataclass
class WACCResult:
    cost_of_equity: float
    after_tax_cost_of_debt: float
    weight_equity: float
    weight_debt: float
    wacc: float
    beta: float
    detail: dict = field(default_factory=dict)


@dataclass
class DCFResult:
    wacc: WACCResult
    years: list[int]
    revenue: list[float]
    ebit: list[float]
    nopat: list[float]
    fcff: list[float]
    discount_factors: list[float]
    pv_fcff: list[float]
    terminal_value: float            # undiscounted TV at end of horizon
    pv_terminal: float
    enterprise_value: float
    net_debt: float
    equity_value: float
    shares: float
    implied_price: float
    current_price: float
    upside: float                    # implied/current - 1
    assumptions: dict = field(default_factory=dict)


@dataclass
class CompRow:
    """One row of a trading-comps table (target or a peer)."""

    ticker: str
    name: str
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    ev_ebitda: Optional[float] = None
    ev_sales: Optional[float] = None
    pe: Optional[float] = None
    pb: Optional[float] = None
    peg: Optional[float] = None
    # Currency of this issuer's market cap / enterprise value. Multiples are
    # dimensionless; peer monetary amounts need not use the target's currency.
    currency: Optional[str] = None


@dataclass
class CompsResult:
    target: CompRow
    peers: list[CompRow]
    # stats[multiple] = {'median':.., 'mean':.., 'min':.., 'max':.., 'p25':.., 'p75':..}
    stats: dict
    # implied[multiple] = implied price per share by applying the peer median multiple
    implied: dict
    implied_price_summary: dict      # {'low':.., 'median':.., 'high':..}
    notes: list[str] = field(default_factory=list)
    # peer_status[TICKER] = what comps did with each requested peer (upper-case
    # ticker, request order): 'used', 'not found' (no data came back), 'no
    # usable multiple' (data with no positive multiple), 'is target' (the
    # target itself) or 'not fetched' (the peer-data fetch failed). Empty when
    # no peers were requested. See ``models.comps.run_comps``.
    peer_status: dict = field(default_factory=dict)
    # target_metrics = the target's {'revenue', 'ebitda', 'net_income'} the
    # implied prices were computed on: the latest fiscal year's, or Yahoo's
    # trailing figures under the opt-in trailing basis. The football field
    # re-prices its EV bars on them. Empty when comps stopped before pricing.
    target_metrics: dict = field(default_factory=dict)


@dataclass
class DDMResult:
    method: str
    implied_price: float
    cost_of_equity: float
    detail: dict = field(default_factory=dict)


@dataclass
class FCFEResult:
    years: list[int]
    fcfe: list[float]
    pv_fcfe: list[float]
    terminal_value: float
    pv_terminal: float
    equity_value: float
    shares: float
    implied_price: float
    current_price: float
    cost_of_equity: float
    detail: dict = field(default_factory=dict)


@dataclass
class SensitivityResult:
    title: str
    row_label: str                   # what the rows vary (e.g. "WACC")
    col_label: str                   # what the columns vary (e.g. "Terminal growth")
    row_values: list[float]
    col_values: list[float]
    grid: list[list[float]]          # grid[i][j] = implied price at row_values[i], col_values[j]


@dataclass
class FootballFieldRow:
    method: str                      # e.g. "DCF", "EV/EBITDA comps", "52-wk range"
    low: float
    base: float
    high: float


@dataclass
class ValuationReport:
    """Top-level container assembled by the engine and consumed by the exporters."""

    company: CompanyData
    macro: MacroAssumptions
    current_price: float
    dcf: Optional[DCFResult] = None
    comps: Optional[CompsResult] = None
    ddm: Optional[DDMResult] = None
    fcfe: Optional[FCFEResult] = None
    sensitivities: list[SensitivityResult] = field(default_factory=list)
    football_field: list[FootballFieldRow] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    # UTC calculation timestamp, preserved when the same valuation is exported.
    valued_at: Optional[str] = None
    # What the market price implies for year-1 revenue growth, the target EBIT
    # margin and the discount rate, other DCF inputs held
    # (``models.reverse_dcf.solve_reverse_dcf``); None without an available DCF.
    reverse_dcf: Optional[dict] = None
