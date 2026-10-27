// TypeScript mirror of backend/serialization.py (which mirrors
// equity_valuation/schemas.py). This is the binding contract between the
// FastAPI backend and the UI — keep field names identical to the engine.

export interface MarketData {
  ticker: string;
  name: string;
  currency: string;
  price: number;
  shares_outstanding: number;
  market_cap: number;
  // The beta the models use. For a live quote it is Yahoo's 5-year monthly
  // beta Blume-adjusted toward 1 (0.67 x raw + 0.33); raw_beta keeps Yahoo's
  // unadjusted figure. raw_beta is null when no adjustment was made (the
  // synthetic company, a beta supplied by hand, or no usable beta).
  beta: number | null;
  raw_beta: number | null;
  dividend_per_share: number | null;
  fifty_two_week_low: number | null;
  fifty_two_week_high: number | null;
  sector: string | null;
  industry: string | null;
  // Yahoo's industry identifier (what "Suggest peers" reads), and its
  // trailing-twelve-month figures (read only by comps; null when Yahoo
  // reports in another currency). Optional: older backends omit them.
  industry_key?: string | null;
  ttm_revenue?: number | null;
  ttm_ebitda?: number | null;
  ttm_net_income?: number | null;
  ttm_as_of?: string | null; // ISO date of the latest quarter
  fiscal_year_end?: string | null; // ISO date
  ttm_currency?: string | null;
}

export interface AnnualFinancials {
  fiscal_years: number[];
  revenue: number[];
  ebit: number[];
  ebitda: number[];
  net_income: number[];
  dep_amort: number[];
  capex: number[];
  change_in_nwc: number[];
  interest_expense: number[];
  tax_expense: number[];
  pretax_income: number[];
  dividends_paid: number[];
  diluted_shares: number[];
}

export interface BalanceSheet {
  as_of: string;
  total_debt: number;
  cash_and_investments: number;
  total_equity: number;
  minority_interest: number;
  preferred_equity: number;
  // Operating-lease liabilities total_debt leaves out (EDGAR filers), which
  // comps add to the target's EV on the peers' basis; null when unknown or
  // already in total_debt. Not part of net_debt.
  operating_lease_liability?: number | null;
  net_debt: number;
}

export interface Company {
  ticker: string;
  name: string;
  cik: string | null;
  financials: AnnualFinancials;
  balance_sheet: BalanceSheet;
  market: MarketData;
  source_notes: string[];
}

export interface Macro {
  risk_free_rate: number;
  equity_risk_premium: number;
  tax_rate: number | null;
  pretax_cost_of_debt: number | null;
}

export interface WACC {
  cost_of_equity: number;
  after_tax_cost_of_debt: number;
  weight_equity: number;
  weight_debt: number;
  wacc: number;
  beta: number;
  detail: Record<string, unknown>;
}

// The DCF's echo of what it resolved (DCFResult.assumptions). Only the keys
// the dashboard reads are named; the rest pass through.
export interface DCFAssumptionsEcho {
  [key: string]: unknown;
  forecast_years?: number;
  revenue_growth_path?: number[];
  target_ebit_margin?: number | null;
  start_ebit_margin?: number | null;
  tax_rate?: number | null;
  terminal_method?: "gordon" | "exit_multiple" | "value_driver";
  terminal_growth?: number;
  terminal_growth_used?: number;
  exit_ev_ebitda?: number | null;
  // Terminal-value check of the first stable year (all null for an exit
  // multiple): NOPAT, the net reinvestment the drivers carry into it (capex -
  // D&A + change in NWC), its share of NOPAT, the return on new invested
  // capital that growth implies (g / rate; null unless both are positive),
  // the RONIC a value-driver terminal value used, and the latest ROIC.
  terminal_nopat?: number | null;
  terminal_net_reinvestment?: number | null;
  terminal_reinvestment_rate?: number | null;
  implied_terminal_ronic?: number | null;
  terminal_ronic_used?: number | null;
  latest_roic?: number | null;
  wacc?: number;
  valuation_available?: boolean;
  notes?: string[];
}

export interface DCF {
  wacc: WACC;
  years: number[];
  revenue: number[];
  ebit: number[];
  nopat: number[];
  fcff: number[];
  discount_factors: number[];
  pv_fcff: number[];
  terminal_value: number;
  pv_terminal: number;
  enterprise_value: number;
  net_debt: number;
  equity_value: number;
  shares: number;
  implied_price: number | null;
  current_price: number;
  upside: number | null;
  assumptions: DCFAssumptionsEcho;
}

export interface CompRow {
  ticker: string;
  name: string;
  currency?: string | null;
  market_cap: number | null;
  enterprise_value: number | null;
  ev_ebitda: number | null;
  ev_sales: number | null;
  pe: number | null;
  pb: number | null;
  peg: number | null;
}

export interface StatRow {
  median: number | null;
  mean: number | null;
  min: number | null;
  max: number | null;
  p25: number | null;
  p75: number | null;
  n?: number; // peers with a usable value of this multiple
}

// What comps did with each requested peer (upper-case ticker).
export type PeerStatus =
  | "used"
  | "not found"
  | "no usable multiple"
  | "is target"
  | "not fetched";

export interface Comps {
  target: CompRow;
  peers: CompRow[];
  stats: Record<string, StatRow>;
  implied: Record<string, number | null>;
  implied_price_summary: {
    low?: number | null;
    median?: number | null;
    high?: number | null;
  };
  notes: string[];
  // Each requested peer's fate, in request order; empty without peers.
  peer_status?: Record<string, PeerStatus | string>;
}

export interface DDM {
  method: string;
  implied_price: number | null;
  cost_of_equity: number;
  detail: Record<string, unknown>;
}

export interface FCFE {
  years: number[];
  fcfe: number[];
  pv_fcfe: number[];
  terminal_value: number;
  pv_terminal: number;
  equity_value: number;
  shares: number;
  implied_price: number | null;
  current_price: number;
  cost_of_equity: number;
  detail: Record<string, unknown>;
}

export interface Sensitivity {
  title: string;
  row_label: string;
  col_label: string;
  // Axis values can be null (e.g. the margin axis for a pre-revenue company).
  row_values: (number | null)[];
  col_values: (number | null)[];
  grid: (number | null)[][];
}

export interface FootballRow {
  method: string;
  low: number;
  base: number;
  high: number;
}

export interface Summary {
  ticker: string;
  name: string;
  currency: string;
  current_price: number;
  methods: Record<string, number | null>;
  // null when no method produced a usable value, or when the engine gives no
  // target (e.g. a captive-finance group or a lessor whose only method left
  // is the DDM: it asks for peers instead).
  blended_target: number | null;
  // null when there is no target, no price, or the verdict is withheld (a
  // blend that rests on the DDM alone, or a corporate action after the latest
  // fiscal year); the UI then shows "n/a", uncoloured.
  blended_upside: number | null;
  recommendation: string;
  // Why there is a target but no verdict, e.g. "a corporate action after FY2025
  // (likely a spin-off)" or "the target rests on the DDM alone"; else null.
  verdict_withheld?: string | null;
  // Methods shown for reference but left out of the blended target, with the
  // engine's reason (e.g. a bank's, captive-finance group's or lessor's DCF
  // and FCFE, or a low-payout DDM).
  excluded_from_blend?: Record<string, string>;
  // Why the company is flagged (bank, insurer, REIT, lender, captive finance
  // arm, lessor, ...); else null.
  financial_institution?: string | null;
  // The flag's kind: "bank", "insurer", "reit", "lender", "captive_finance",
  // "lessor" or "financial" (flagged, kind unknown); null when not flagged.
  financial_kind?: string | null;
}

// Reverse DCF: the value of one driver at which the DCF equals the price,
// other inputs held. The top-level keys solve year-1 revenue growth; `margin`
// the target EBIT margin and `wacc` the discount rate (null when not solved).
export interface ReverseDCFMargin {
  converged: boolean;
  implied_target_ebit_margin: number | null;
  current_target_ebit_margin?: number | null;
  start_ebit_margin?: number | null;
  note?: string | null;
}

export interface ReverseDCFWacc {
  converged: boolean;
  implied_wacc: number | null;
  current_wacc?: number | null;
  note?: string | null;
}

export interface ReverseDCF {
  converged: boolean;
  implied_growth_y1: number | null;
  current_assumption_y1?: number | null;
  implied_revenue_growth?: number[];
  note?: string;
  margin?: ReverseDCFMargin | null;
  wacc?: ReverseDCFWacc | null;
}

export interface Report {
  report_id?: string;
  valued_at?: string;
  summary: Summary;
  company: Company;
  macro: Macro;
  current_price: number;
  dcf: DCF | null;
  comps: Comps | null;
  ddm: DDM | null;
  fcfe: FCFE | null;
  sensitivities: Sensitivity[];
  football_field: FootballRow[];
  warnings: string[];
  assumptions_used: AssumptionsUsed;
  // The same echo for an empty request (the engine's defaults), so the
  // header can flag custom inputs. Optional: older backends omit it.
  assumptions_default?: AssumptionsUsed;
  reverse_dcf?: ReverseDCF | null;
}

// --- Peer suggestions (GET /api/peers/{ticker}) ---------------------------- //
export interface PeerSuggestion {
  ticker: string;
  name?: string | null;
  market_weight?: number | null;
  size_ratio?: number | null; // market weight vs the target's
}

export interface PeerSuggestions {
  ticker: string;
  industry_key?: string | null;
  industry?: string | null;
  peers: PeerSuggestion[];
  notes: string[];
  source?: string;
}

// --- FMP enrichment ------------------------------------------------------- //
export interface NewsItem {
  title: string;
  text: string;
  site: string;
  url: string;
  image?: string;
  publishedDate: string;
  symbol?: string;
}

export interface Enrichment {
  enabled: boolean;
  profile?: Record<string, unknown> | null;
  quote?: Record<string, unknown> | null;
  ratios_ttm?: Record<string, unknown> | null;
  key_metrics_ttm?: Record<string, unknown> | null;
  price_target?: Record<string, unknown> | null;
  rating?: Record<string, unknown> | null;
  peers?: string[];
  estimates?: Record<string, unknown>[];
  news?: NewsItem[];
}

// --- Editable assumptions (sent to /api/valuation) ------------------------ //
export type TerminalMethod = "gordon" | "exit_multiple" | "value_driver";
export type DDMMethod = "gordon" | "two_stage" | "h_model";
export type CompsBasis = "fy" | "ttm";

export interface Assumptions {
  rf?: number; // risk-free rate (decimal)
  erp?: number; // equity risk premium (decimal)
  tax_rate?: number; // decimal; null => engine derives effective
  cost_of_debt?: number;
  forecast_years?: number;
  terminal_growth?: number; // decimal
  terminal_method?: TerminalMethod;
  terminal_ronic?: number; // decimal; value-driver RONIC, unset => WACC
  exit_ev_ebitda?: number; // multiple
  target_ebit_margin?: number; // decimal
  revenue_growth_y1?: number; // decimal; near-term growth, fades to terminal
  revenue_growth?: number[]; // explicit path restored from saved research
  peers?: string; // comma-separated tickers
  ddm_method?: DDMMethod;
  ddm_high_growth_rate?: number; // decimal; unset => the DDM derives it
  ddm_high_growth_years?: number;
  comps_basis?: CompsBasis; // target figures comps use (fiscal year or TTM)
}

// The engine's echo of what it actually used (parse_assumptions). Optional
// inputs the user left unset come back as null.
export interface AssumptionsUsed {
  rf: number;
  erp: number;
  tax_rate: number | null;
  cost_of_debt: number | null;
  forecast_years: number;
  terminal_growth: number;
  terminal_method: TerminalMethod;
  exit_ev_ebitda: number | null;
  target_ebit_margin: number | null;
  revenue_growth_y1: number | null;
  revenue_growth: number[] | null;
  peers: string | null;
  // Optional: older backends do not echo them.
  terminal_ronic?: number | null;
  ddm_method?: DDMMethod;
  ddm_high_growth_rate?: number | null;
  ddm_high_growth_years?: number;
  comps_basis?: CompsBasis;
}

// --- AI researcher -------------------------------------------------------- //
export type AssumptionField =
  | "terminal_growth"
  | "forecast_years"
  | "target_ebit_margin"
  | "risk_free_rate"
  | "equity_risk_premium"
  | "tax_rate"
  | "exit_ev_ebitda"
  | "revenue_growth_y1";

export interface AssumptionSuggestion {
  field: AssumptionField;
  label: string;
  current_value: number | null;
  suggested_value: number;
  unit: "percent" | "number" | "years" | "multiple";
  rationale: string;
}

export interface CitedFact {
  fact: string;
  source: string;
}

export interface Digest {
  summary: string;
  sentiment: "bullish" | "bearish" | "neutral" | "mixed";
  key_facts: CitedFact[];
  risks: string[];
  catalysts: string[];
  suggested_assumptions: AssumptionSuggestion[];
  // Suggestions the backend removed (an unknown field, a value that is not a
  // number or is outside the field's range); the summary already names them.
  // Optional: absent when nothing was removed.
  dropped_suggestions?: {
    field: string | null;
    suggested_value: unknown;
    reason: string;
  }[];
}

// A digest plus where it came from (pasted text, a named filing, a call…).
export interface DigestEntry {
  source: string;
  digest: Digest;
  at?: string;
}

export interface Citation {
  source: string;
  note: string;
}

export interface ResearchNote {
  title: string;
  stance: "constructive" | "cautious" | "balanced";
  executive_summary: string;
  thesis: string[];
  valuation_view: string;
  key_drivers: string[];
  risks: string[];
  red_flags: string[];
  catalysts: string[];
  what_would_change_my_mind: string[];
  citations: Citation[];
}

export interface PdfAttachment {
  name: string;
  data_base64: string;
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

// --- SEC filings & transcripts --------------------------------------------- //
export interface Filing {
  form: string;
  filed: string;
  report_date: string;
  accession_number: string;
  primary_document: string;
  description: string;
  url: string;
}

export interface FilingsList {
  ticker: string;
  cik: string;
  name: string;
  filings: Filing[];
}

export interface TranscriptMeta {
  quarter: number;
  year: number;
  date: string;
}

// --- watchlist & persisted research ---------------------------------------- //
export interface WatchlistItem {
  ticker: string;
  name: string | null;
  currency: string | null;
  price: number | null;
  blended_target: number | null;
  recommendation: string | null;
  updated_at: string;
}

export interface ResearchState {
  notes?: string;
  digests?: DigestEntry[];
  note?: ResearchNote | null;
  assumptions?: Assumptions | null;
  updated_at?: string;
}
