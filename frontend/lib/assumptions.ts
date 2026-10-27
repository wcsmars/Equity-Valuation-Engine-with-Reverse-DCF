// Helpers for the assumption controls (components/DCFPanel.tsx, the DDM
// drivers in components/DDMFCFEPanel.tsx, the comps card, the
// AssumptionSlider in components/ui.tsx) and the page's resets, plus the
// comparisons behind the "unapplied changes" bar and the header's "Custom
// inputs" badge.

import type { AssumptionField as SuggestionField, Assumptions, AssumptionsUsed, Report } from "./types";
import { fmtMult, fmtRate } from "./format";

// The editable draft that reproduces what the engine used (its echo,
// report.assumptions_used). Unset optional inputs (null in the echo) stay
// unset. An explicit growth path is kept only when no year-1 growth was
// given, as the backend builds the path from the year-1 rate otherwise.
// After a recompute the page replaces the draft it sent with this, so the
// clamped and normalised values are what the controls show (and what
// diffAssumptions compares against).
export function assumptionsFromUsed(used?: AssumptionsUsed | null): Assumptions {
  if (!used) return {};
  return {
    rf: used.rf,
    erp: used.erp,
    tax_rate: used.tax_rate ?? undefined,
    cost_of_debt: used.cost_of_debt ?? undefined,
    forecast_years: used.forecast_years,
    terminal_growth: used.terminal_growth,
    terminal_method: used.terminal_method,
    terminal_ronic: used.terminal_ronic ?? undefined,
    exit_ev_ebitda: used.exit_ev_ebitda ?? undefined,
    target_ebit_margin: used.target_ebit_margin ?? undefined,
    revenue_growth_y1: used.revenue_growth_y1 ?? undefined,
    revenue_growth: used.revenue_growth_y1 == null ? used.revenue_growth ?? undefined : undefined,
    peers: used.peers ?? undefined,
    ddm_method: used.ddm_method ?? undefined,
    ddm_high_growth_rate: used.ddm_high_growth_rate ?? undefined,
    ddm_high_growth_years: used.ddm_high_growth_years ?? undefined,
    comps_basis: used.comps_basis ?? undefined,
  };
}

// Only the peer set of `a`. A reset of the DCF drivers ("Reset defaults", or
// "Retry with default assumptions" after a failed load) keeps it: peers are
// edited in the comps card, are never found by the engine (the page fills in
// FMP's suggested peers only while the field is blank), and cannot make the
// assumptions invalid, so dropping them would silently remove the comps (and,
// for a bank or a captive-finance group, the only verdict).
export function peersOnly(a: Assumptions | null | undefined): Assumptions {
  const p = a?.peers;
  return typeof p === "string" && p.trim() ? { peers: p } : {};
}

// What the DCF card's "Reset defaults" keeps: the inputs other cards own,
// i.e. the peer set (see peersOnly) and comps basis (the comps card) and the
// DDM drivers (the DDM card). Everything else returns to the engine's
// defaults.
export function keepOnDcfReset(a: Assumptions | null | undefined): Assumptions {
  const out: Assumptions = peersOnly(a);
  if (a?.comps_basis !== undefined) out.comps_basis = a.comps_basis;
  if (a?.ddm_method !== undefined) out.ddm_method = a.ddm_method;
  if (a?.ddm_high_growth_rate !== undefined) out.ddm_high_growth_rate = a.ddm_high_growth_rate;
  if (a?.ddm_high_growth_years !== undefined) out.ddm_high_growth_years = a.ddm_high_growth_years;
  return out;
}

// A slider's range: its usual [min, max], widened to take in each finite value
// of `include` (the engine's own figures and the value shown) and rounded
// outward onto the step grid that starts at `min`. A browser clamps a range
// input's value into [min, max], so without this a loss-maker's -66.5% target
// margin would sit at a 0% thumb and the first nudge would jump to +0.5%.
export function sliderBounds(
  min: number,
  max: number,
  step: number,
  include: (number | null | undefined)[]
): [number, number] {
  const round = (x: number) => Number(x.toFixed(10));
  let lo = min;
  let hi = max;
  for (const v of include) {
    if (typeof v !== "number" || !Number.isFinite(v)) continue;
    if (v < lo) lo = round(min - Math.ceil(round((min - v) / step)) * step);
    if (v > hi) hi = round(min + Math.ceil(round((v - min) / step)) * step);
  }
  return [lo, Math.max(hi, lo)];
}

// --- Comparing assumptions -------------------------------------------------- //
// How each input is compared and shown:
//   rate   a decimal, equal within RATE_TOLERANCE, shown as a percentage;
//   number a plain number (a multiple), equal within RATE_TOLERANCE;
//   int    a whole number (years);
//   text   a choice (terminal method, DDM method, comps basis), case-blind;
//   list   a list of rates (an explicit revenue-growth path);
//   peers  comma-separated tickers, compared as a set in order after
//          trimming, upper-casing and dropping repeats.
// `unset` is how a missing value reads (the engine then derives it).
export type FieldKind = "rate" | "number" | "int" | "text" | "list" | "peers";

export interface AssumptionField {
  key: keyof Assumptions;
  label: string;
  kind: FieldKind;
  format: (v: unknown) => string;
  unset: string;
}

export const RATE_TOLERANCE = 1e-9;

const CHOICES: Record<string, string> = {
  gordon: "Gordon",
  exit_multiple: "Exit multiple",
  value_driver: "Value driver",
  two_stage: "Two-stage",
  h_model: "H-model",
  fy: "Fiscal year",
  ttm: "Trailing 12 months",
};

const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);
const rate = (v: unknown) => fmtRate(num(v));
const years = (v: unknown) => (num(v) == null ? "—" : `${Math.round(num(v) as number)}y`);
const choice = (v: unknown) => (typeof v === "string" ? CHOICES[v.toLowerCase()] ?? v : "—");
const path = (v: unknown) =>
  Array.isArray(v) ? v.map((x) => fmtRate(num(x))).join(", ") : "—";

// A plausible exchange ticker (BRK.B, BRK-B, 7203.T): the backend's own check
// (backend/app.py _TICKER_RE, which a test pins), so a typo is refused before
// any request and the ticker on screen stays.
export const TICKER_PATTERN = /^[A-Za-z0-9.\-]{1,12}$/;

export function isPlausibleTicker(t: unknown): boolean {
  return typeof t === "string" && TICKER_PATTERN.test(t.trim());
}

// Upper-cased, trimmed tickers without repeats, in the order given.
export function normalizePeers(p: unknown): string[] {
  if (typeof p !== "string") return [];
  return [...new Set(p.split(",").map((t) => t.trim().toUpperCase()).filter(Boolean))];
}

export const ASSUMPTION_FIELDS: AssumptionField[] = [
  { key: "forecast_years", label: "Forecast years", kind: "int", format: years, unset: "default" },
  { key: "terminal_growth", label: "Terminal growth", kind: "rate", format: rate, unset: "default" },
  { key: "revenue_growth_y1", label: "Near-term revenue growth", kind: "rate", format: rate,
    unset: "derived" },
  { key: "revenue_growth", label: "Revenue growth path", kind: "list", format: path,
    unset: "derived" },
  { key: "target_ebit_margin", label: "Target EBIT margin", kind: "rate", format: rate,
    unset: "derived" },
  { key: "rf", label: "Risk-free rate", kind: "rate", format: rate, unset: "default" },
  { key: "erp", label: "Equity risk premium", kind: "rate", format: rate, unset: "default" },
  { key: "tax_rate", label: "Tax rate", kind: "rate", format: rate, unset: "derived" },
  { key: "cost_of_debt", label: "Cost of debt", kind: "rate", format: rate, unset: "derived" },
  { key: "terminal_method", label: "Terminal method", kind: "text", format: choice,
    unset: "Gordon" },
  { key: "exit_ev_ebitda", label: "Exit EV / EBITDA", kind: "number",
    format: (v) => fmtMult(num(v)), unset: "none" },
  { key: "terminal_ronic", label: "Terminal RONIC", kind: "rate", format: rate, unset: "WACC" },
  { key: "ddm_method", label: "DDM method", kind: "text", format: choice, unset: "Two-stage" },
  { key: "ddm_high_growth_rate", label: "DDM high growth", kind: "rate", format: rate,
    unset: "derived" },
  { key: "ddm_high_growth_years", label: "DDM high-growth years", kind: "int", format: years,
    unset: "default" },
  { key: "comps_basis", label: "Comps basis", kind: "text", format: choice,
    unset: "Fiscal year" },
  { key: "peers", label: "Peers", kind: "peers",
    format: (v) => normalizePeers(v).join(", ") || "none", unset: "none" },
];

// A value in the form it is compared in (null when unset).
function canonical(kind: FieldKind, v: unknown): unknown {
  if (v === undefined || v === null) return null;
  switch (kind) {
    case "rate":
    case "number":
    case "int":
      return num(v);
    case "text":
      return typeof v === "string" && v.trim() ? v.trim().toLowerCase() : null;
    case "list":
      return Array.isArray(v) && v.length > 0 ? v.map(num) : null;
    case "peers": {
      const p = normalizePeers(v);
      return p.length > 0 ? p.join(",") : null;
    }
  }
}

function same(kind: FieldKind, a: unknown, b: unknown): boolean {
  const x = canonical(kind, a);
  const y = canonical(kind, b);
  if (x === null || y === null) return x === y;
  if (kind === "rate" || kind === "number" || kind === "int") {
    return Math.abs((x as number) - (y as number)) <= RATE_TOLERANCE;
  }
  if (kind === "list") {
    const xs = x as (number | null)[];
    const ys = y as (number | null)[];
    return xs.length === ys.length && xs.every((v, i) =>
      v === null || ys[i] === null ? v === ys[i] : Math.abs(v - (ys[i] as number)) <= RATE_TOLERANCE);
  }
  return x === y;
}

export interface AssumptionChange {
  key: keyof Assumptions;
  label: string;
  from: string;
  to: string;
}

type Values = Partial<Record<keyof Assumptions, unknown>>;

function show(f: AssumptionField, v: unknown): string {
  return canonical(f.kind, v) === null ? f.unset : f.format(v);
}

// What the engine derived for the inputs a report leaves unset (null in its
// assumptions_used): the DCF's echo (target margin, year-1 growth, tax rate,
// the value-driver RONIC), the WACC's pre-tax cost of debt and the DDM's
// high-growth rate. The unapplied-changes bar shows these beside "derived"
// ("derived 31.97% → 30.0%"), the number the slider started from.
export function derivedValues(report: Report | null | undefined): Values {
  const out: Values = {};
  if (!report) return out;
  const fin = (x: unknown) => (typeof x === "number" && Number.isFinite(x) ? x : undefined);
  const dcfA = (report.dcf?.assumptions || {}) as Record<string, unknown>;
  const path = dcfA.revenue_growth_path;
  const wacc = (report.dcf?.wacc?.detail || {}) as Record<string, unknown>;
  const ddm = (report.ddm?.detail || {}) as Record<string, unknown>;
  const set = (k: keyof Assumptions, v: unknown) => {
    if (fin(v) !== undefined) out[k] = v;
  };
  set("target_ebit_margin", dcfA.target_ebit_margin);
  set("revenue_growth_y1", Array.isArray(path) ? path[0] : undefined);
  set("tax_rate", dcfA.tax_rate);
  set("terminal_ronic", dcfA.terminal_ronic_used);
  set("cost_of_debt", wacc.pretax_cost_of_debt);
  set("ddm_high_growth_rate", ddm.high_growth);
  return out;
}

// A method choice, lower-cased; an unset one is the engine's default.
const methodOf = (v: unknown, engineDefault: string) =>
  typeof v === "string" && v.trim() ? v.trim().toLowerCase() : engineDefault;

// Inputs only some methods read, with the test for a set of values: the exit
// multiple (an exit-multiple terminal value), the terminal RONIC (a
// value-driver one) and the DDM high-growth rate and years (a two-stage or
// H-model DDM; a Gordon DDM has no high-growth stage). The draft keeps such
// an input when the method is switched away, so switching back restores it,
// and the backend echoes whatever it was sent; neither makes it count.
const USED_ONLY_WHEN: Partial<Record<keyof Assumptions, (v: Values) => boolean>> = {
  exit_ev_ebitda: (v) => methodOf(v.terminal_method, "gordon") === "exit_multiple",
  terminal_ronic: (v) => methodOf(v.terminal_method, "gordon") === "value_driver",
  ddm_high_growth_rate: (v) => methodOf(v.ddm_method, "two_stage") !== "gordon",
  ddm_high_growth_years: (v) => methodOf(v.ddm_method, "two_stage") !== "gordon",
};

// The fields that differ between `base` and `next`, as {key, label, from, to}
// with the values formatted. Year-1 revenue growth takes precedence over the
// growth path: when either side sets it, the path (which the backend builds
// from it) is not compared; otherwise the paths are. With `skipUnsetNext`, a
// field `next` leaves unset is not a change (the draft leaves it to the
// engine). An input `next`'s method ignores (USED_ONLY_WHEN) is not a change
// either; one `base`'s method ignored reads as unset there (the base run did
// not use it), though an unchanged value is still no change, as the method
// change already says what moves.
function compare(
  next: Values,
  base: Values,
  skipUnsetNext: boolean,
  derived: Values = {}
): AssumptionChange[] {
  const y1Set = next.revenue_growth_y1 != null || base.revenue_growth_y1 != null;
  const out: AssumptionChange[] = [];
  for (const f of ASSUMPTION_FIELDS) {
    if (f.key === "revenue_growth" && y1Set) continue;
    const usedBy = USED_ONLY_WHEN[f.key];
    if (usedBy && !usedBy(next)) continue;
    const to = next[f.key];
    const from = base[f.key];
    if (skipUnsetNext && to === undefined) continue;
    if (same(f.kind, to, from)) continue;
    const baseUsed = !usedBy || usedBy(base);
    let fromText = baseUsed ? show(f, from) : f.unset;
    // A year-1 rate set over an explicit path: show the path's first year.
    if (f.key === "revenue_growth_y1" && from == null && Array.isArray(base.revenue_growth)) {
      fromText = `${rate(base.revenue_growth[0])} path`;
    } else if (baseUsed && canonical(f.kind, from) === null
      && canonical(f.kind, derived[f.key]) !== null) {
      // Left to the engine: say what it derived ("derived 31.97%").
      fromText = `${f.unset} ${f.format(derived[f.key])}`;
    }
    out.push({ key: f.key, label: f.label, from: fromText, to: show(f, to) });
  }
  return out;
}

// Edits in the draft that the report on screen does not reflect yet:
// `draft` against assumptionsFromUsed(report.assumptions_used). A field the
// draft leaves unset is not an edit, nor is an input the draft's method
// ignores (an exit multiple kept after switching back to Gordon). Rates
// compare within RATE_TOLERANCE and peers as normalised lists
// (normalizePeers), so the draft a recompute normalises (assumptionsFromUsed)
// always compares equal. `derived` (derivedValues of that report) gives the
// figure the engine derived for an input the report left unset, shown with
// the "from" side; it never decides what counts as a change.
export function diffAssumptions(
  draft: Assumptions | null | undefined,
  usedAsAssumptions: Assumptions | null | undefined,
  derived?: Values | null
): AssumptionChange[] {
  return compare((draft ?? {}) as Values, (usedAsAssumptions ?? {}) as Values, true,
    derived ?? {});
}

// Inputs the report was run with that differ from the engine's defaults
// (report.assumptions_default, the echo of an empty request): each as {key,
// label, from: default, to: used}. A peer set always counts (the default
// has none); an input the method used ignores never does. Empty without
// either echo (an older backend sends no defaults). A default the engine
// derives reads "derived" without a figure: the report on screen was run with
// the custom value, so its echo holds that, not the derived default.
export function customFields(
  used: AssumptionsUsed | null | undefined,
  defaults: AssumptionsUsed | null | undefined
): AssumptionChange[] {
  if (!used || !defaults) return [];
  return compare(used as unknown as Values, defaults as unknown as Values, false);
}

// --- AI suggestions ---------------------------------------------------------- //
// The draft input each AI-suggested field sets (the page's Apply).
export const SUGGESTION_KEYS: Record<SuggestionField, keyof Assumptions> = {
  terminal_growth: "terminal_growth",
  forecast_years: "forecast_years",
  target_ebit_margin: "target_ebit_margin",
  risk_free_rate: "rf",
  equity_risk_premium: "erp",
  tax_rate: "tax_rate",
  exit_ev_ebitda: "exit_ev_ebitda",
  revenue_growth_y1: "revenue_growth_y1",
};

// The value a suggestion would replace in the report on screen: what the
// engine used (assumptions_used), else what it derived (derivedValues), so a
// suggestion from an older digest is compared with the model as it is now,
// not with the value the AI echoed when it wrote it. Null when neither is
// known (an exit multiple under a Gordon terminal value, an older backend).
export function liveSuggestionValue(
  report: Report | null | undefined,
  field: SuggestionField
): number | null {
  const key = SUGGESTION_KEYS[field];
  if (!report || !key) return null;
  const fin = (x: unknown) => (typeof x === "number" && Number.isFinite(x) ? x : null);
  const used = (report.assumptions_used || {}) as Partial<Record<keyof Assumptions, unknown>>;
  if (key === "revenue_growth_y1") {
    const path = used.revenue_growth;
    const first = fin(used.revenue_growth_y1) ?? (Array.isArray(path) ? fin(path[0]) : null);
    if (first !== null) return first;
  } else if (fin(used[key]) !== null) {
    return fin(used[key]);
  }
  return fin(derivedValues(report)[key]);
}
