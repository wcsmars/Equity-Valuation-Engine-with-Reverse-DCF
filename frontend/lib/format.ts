// Formatting helpers shared by every panel. All tolerate null/undefined/NaN.

import type { Digest } from "./types";

const SYMBOLS: Record<string, string> = {
  USD: "$",
  EUR: "€",
  GBP: "£",
  JPY: "¥",
  CAD: "C$",
  AUD: "A$",
  CHF: "CHF ",
  // Yuan and yen both write "¥"; peer rows keep each peer's own currency, so
  // the yuan takes the "CN¥" form to stay distinguishable.
  CNY: "CN¥",
  HKD: "HK$",
  INR: "₹",
};

// Unmapped currencies fall back to their ISO code ("SEK 33.74") so a figure
// never renders without any currency marker.
export function sym(currency?: string | null): string {
  const c = currency || "USD";
  return SYMBOLS[c] ?? `${c} `;
}

function isNum(x: unknown): x is number {
  return typeof x === "number" && Number.isFinite(x);
}

// Snap values that round to zero at the displayed precision to +0, so tiny
// negatives render as "$0.00" / "0.0%" rather than "-$0.00" / "-0.0%".
function snap(x: number, decimals: number): number {
  return Number(x.toFixed(decimals)) === 0 ? 0 : x;
}

// The minus sign goes before the currency symbol: -$36.50, not $-36.50.
export function fmtMoney(
  x: number | null | undefined,
  currency?: string | null,
  decimals = 2
): string {
  if (!isNum(x)) return "—";
  x = snap(x, decimals);
  const sign = x < 0 ? "-" : "";
  return `${sign}${sym(currency)}${Math.abs(x).toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })}`;
}

// Decimals for a share price shown beside a reference price (the current
// price): cents unless the price is in the thousands, where they are noise.
// Whole units would make most cells of a sub-$20 stock's grid read the same.
export function priceDecimals(ref: number | null | undefined): number {
  return isNum(ref) && Math.abs(ref) >= 1000 ? 0 : 2;
}

// Compact magnitude without sign or symbol: 391.0B, 1.2T, 540.0M. The unit is
// picked on the rounded value, so 999.96B renders as 1.0T, not 1000.0B.
function compact(abs: number): string {
  const units: [string, number][] = [
    ["T", 1e12],
    ["B", 1e9],
    ["M", 1e6],
    ["K", 1e3],
  ];
  for (const [u, d] of units) {
    const v = (abs / d).toFixed(1);
    if (Number(v) >= 1) return `${v}${u}`;
  }
  return abs.toFixed(0);
}

// Compact money amounts: $391.0B, -$36.5B, €1.2T.
export function fmtBig(
  x: number | null | undefined,
  currency?: string | null
): string {
  if (!isNum(x)) return "—";
  x = snap(x, 0);
  return `${x < 0 ? "-" : ""}${sym(currency)}${compact(Math.abs(x))}`;
}

// Compact counts with no currency symbol (share counts): 7.4B, 540.0M.
export function fmtCount(x: number | null | undefined): string {
  if (!isNum(x)) return "—";
  x = snap(x, 0);
  return `${x < 0 ? "-" : ""}${compact(Math.abs(x))}`;
}

export function fmtPct(
  x: number | null | undefined,
  opts: { signed?: boolean; decimals?: number } = {}
): string {
  if (!isNum(x)) return "—";
  const { signed = false, decimals = 1 } = opts;
  const p = snap(x * 100, decimals);
  const sign = signed && p > 0 ? "+" : "";
  return `${sign}${p.toFixed(decimals)}%`;
}

export function fmtNum(
  x: number | null | undefined,
  decimals = 2
): string {
  if (!isNum(x)) return "—";
  return snap(x, decimals).toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

// Trading multiple, e.g. 12.3x.
export function fmtMult(x: number | null | undefined, decimals = 1): string {
  if (!isNum(x)) return "—";
  return `${snap(x, decimals).toFixed(decimals)}x`;
}

// File/request size in decimal megabytes, e.g. 41.3 MB.
export function fmtBytes(n: number | null | undefined): string {
  if (!isNum(n)) return "—";
  return `${(n / 1e6).toFixed(1)} MB`;
}

// A message ending in exactly one sentence stop, so text that already ends
// in "." (or "!", "?", "…") is not given a second one.
export function withFullStop(s: string): string {
  const t = s.trimEnd();
  return /[.!?…]$/.test(t) ? t : `${t}.`;
}

export function fmtDate(s: string | null | undefined): string {
  if (!s) return "";
  // A date-only ISO string ("2024-12-31") parses as UTC midnight, which
  // renders as the previous day west of UTC; build it as a local date instead.
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(s);
  const d = m ? new Date(+m[1], +m[2] - 1, +m[3]) : new Date(s);
  if (Number.isNaN(d.getTime())) return s;
  if (m && (d.getMonth() !== +m[2] - 1 || d.getDate() !== +m[3])) return s;
  return d.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

// The research-log entry for a digest. The log is the context every later
// note, chat and digest receives, so it carries the digest's cited facts (each
// with its source), risks and catalysts as well as its summary; the summary
// alone would leave the note nothing to cite.
export function digestLogEntry(source: string, digest: Partial<Digest>): string {
  const text = (x: unknown) => (typeof x === "string" ? x.trim() : "");
  const texts = (x: unknown) => (Array.isArray(x) ? x.map(text).filter(Boolean) : []);
  const lines = [`[${source}] ${text(digest.summary)}`.trimEnd()];
  for (const f of Array.isArray(digest.key_facts) ? digest.key_facts : []) {
    const fact = text(f?.fact);
    if (fact) lines.push(`- ${fact} (source: ${text(f?.source) || source})`);
  }
  for (const r of texts(digest.risks)) lines.push(`- Risk: ${r}`);
  for (const c of texts(digest.catalysts)) lines.push(`- Catalyst: ${c}`);
  return lines.join("\n");
}

// The research log with `entry` appended as its own paragraph.
export function appendLogEntry(log: string, entry: string): string {
  return (log ? log + "\n\n" : "") + entry;
}

// A research log saved before digest entries carried their facts holds one
// "[source] summary" paragraph per digest. Each such paragraph that is still
// exactly as it was written becomes that saved digest's full entry
// (digestLogEntry), so notes for tickers researched earlier get the cited
// facts too. Any other text, including an edited or removed line, is kept as
// it is, and a log already upgraded is returned unchanged.
export function upgradeDigestLog(
  log: string,
  digests: readonly { source?: unknown; digest?: Partial<Digest> | null }[] | null | undefined
): string {
  if (!log || !Array.isArray(digests)) return log;
  const full = new Map<string, string>();
  for (const d of digests) {
    const summary = d?.digest?.summary;
    if (typeof d?.source !== "string" || typeof summary !== "string") continue;
    const old = `[${d.source}] ${summary}`;
    if (!full.has(old)) full.set(old, digestLogEntry(d.source, d.digest ?? {}));
  }
  if (full.size === 0) return log;
  return log.split("\n\n").map((p) => full.get(p) ?? p).join("\n\n");
}

// The verdict bands' tolerance: engine._recommendation counts an upside
// within 1e-12 of +/-15% as on the band (math.isclose), so a target at
// exactly 1.15x the price, which reads 0.15 minus float noise, is
// "Undervalued" there and is coloured as such here.
const BAND_TOLERANCE = 1e-12;

// Map an upside/recommendation to a semantic color class.
export function toneForUpside(upside: number | null | undefined): string {
  if (!isNum(upside)) return "text-ink-dim";
  if (upside >= 0.15 - BAND_TOLERANCE) return "text-up";
  if (upside <= -0.15 + BAND_TOLERANCE) return "text-down";
  return "text-flat";
}

// True when a class list sets a text colour (text-ink, text-ink-dim, text-up,
// text-down, text-flat, text-brand, ...), not an alignment or a size. A table
// cell given one drops its default text-ink: Tailwind emits the colour
// utilities alphabetically, so text-down or text-flat beside text-ink loses to
// it and a negative upside would show in plain ink.
export function setsTextColor(className?: string | null): boolean {
  return /(?:^|\s)text-(?:ink|up|down|flat|brand|white)(?:-[a-z]+)?(?=\s|$)/.test(
    className || ""
  );
}

export function toneForRecommendation(rec: string | null | undefined): string {
  switch (rec) {
    case "Undervalued":
      return "text-up";
    case "Overvalued":
      return "text-down";
    case "Fairly valued":
      return "text-flat";
    default:
      return "text-ink-dim";
  }
}

// --- Blended target -------------------------------------------------------- //
// The engine's blended_target is null when no method gives a usable value, or
// when it gives no target at all (a captive-finance group or a lessor whose
// only method left is the DDM asks for peers instead). blended_upside is also
// null when the verdict is withheld (a blend that rests on the DDM alone, or a
// corporate action after the latest fiscal year; verdict_withheld says which).
// A null is shown as text, never as a figure, and is not coloured. These rules
// copy equity_valuation/report/excel.py, which the Excel and HTML reports, the
// memo and deck and the AI context share; a Python test pins the copy.
export const NO_TARGET_SUPPLY_PEERS = "No target (supply peers)";
// The same, where a table cell has no room for the reason (the header's
// verdict explainer gives it).
export const NO_TARGET_SHORT = "No target";

interface BlendFields {
  methods?: Record<string, number | null> | null;
  blended_target?: number | null;
  blended_upside?: number | null;
  currency?: string | null;
  recommendation?: string | null;
  excluded_from_blend?: Record<string, string> | null;
  financial_institution?: string | null;
  financial_kind?: string | null;
  verdict_withheld?: string | null;
}

// The report's trading comps (null when comps did not run).
type CompsFields = { peers?: unknown[] | null; notes?: string[] | null } | null | undefined;

// No target because the company is flagged (a bank, a captive-finance group,
// a lessor, ...) and every method it has is reference only, and no usable
// peers were supplied: trading comps would give a target (the engine never
// finds peers itself). Not when peers were supplied but comps gave no price,
// nor when an ordinary company's methods gave no valuation: those read "n/a".
export function needsPeers(s: BlendFields, comps?: CompsFields): boolean {
  if (isNum(s.blended_target)) return false;
  const names = Object.keys(s.methods || {});
  if (names.length === 0 || names.some((n) => n.startsWith("Comps"))) return false;
  if (!(s.financial_kind || s.financial_institution)) return false;
  return !(comps && comps.peers && comps.peers.length > 0);
}

// `short`: "No target" without the reason, for a narrow table cell.
export function fmtBlendedTarget(
  s: BlendFields,
  comps?: CompsFields,
  opts: { short?: boolean } = {}
): string {
  if (isNum(s.blended_target)) return fmtMoney(s.blended_target, s.currency);
  if (!needsPeers(s, comps)) return "n/a";
  return opts.short ? NO_TARGET_SHORT : NO_TARGET_SUPPLY_PEERS;
}

// The upside is shown only beside a target, as the engine gives it.
export function fmtBlendedUpside(s: BlendFields): string {
  return isNum(s.blended_target) && isNum(s.blended_upside)
    ? fmtPct(s.blended_upside, { signed: true })
    : "n/a";
}

export function toneForBlendedUpside(s: BlendFields): string {
  return isNum(s.blended_target) && isNum(s.blended_upside)
    ? toneForUpside(s.blended_upside)
    : "text-ink-dim";
}

// Colour for one method's upside (the method summary, and the DCF, DDM and
// FCFE "Implied price" stats). Dim, not green or red, for a method left out of
// the blended target (shown for reference only) and for every method when the
// engine gives no verdict ("N/A": no target, or a verdict withheld), so no
// direction is shown that the engine does not give. `method` is the summary's
// method name: "DCF", "Comps (median)", "DDM", "FCFE".
export function toneForMethodUpside(
  s: BlendFields,
  method: string,
  upside: number | null | undefined
): string {
  const excluded = s.excluded_from_blend || {};
  if (Object.prototype.hasOwnProperty.call(excluded, method)) return "text-ink-dim";
  if (s.recommendation === "N/A") return "text-ink-dim";
  return toneForUpside(upside);
}

// Bar colour for a football-field row: green when its base is 5% or more
// above the price, rose when 5% or more below, amber between. Grey, like the
// method's upside, for a method left out of the blended target (the engine
// labels its row "... (not in blend)") and for every row when the engine gives
// no verdict.
export function footballFieldTone(
  s: BlendFields,
  method: string,
  base: number | null | undefined,
  price: number | null | undefined
): string {
  const name = method.replace(/\s*\(not in blend\)$/, "");
  const excluded = s.excluded_from_blend || {};
  const refOnly =
    name !== method || Object.prototype.hasOwnProperty.call(excluded, name);
  if (refOnly || s.recommendation === "N/A" || !isNum(base) || !isNum(price)) {
    return "bg-ink-faint/30 border-ink-faint";
  }
  if (base >= price * 1.05) return "bg-up/30 border-up";
  if (base <= price * 0.95) return "bg-down/30 border-down";
  return "bg-flat/30 border-flat";
}

// A saved watchlist entry's upside: null without a target or a positive price,
// or when the verdict was withheld ("N/A"), so the card is not coloured.
export function watchlistUpside(w: {
  blended_target: number | null;
  price: number | null;
  recommendation: string | null;
}): number | null {
  if (!isNum(w.blended_target) || !isNum(w.price) || w.price <= 0) return null;
  if (w.recommendation === "N/A") return null;
  return w.blended_target / w.price - 1;
}

// Why the engine left a method out of the blended target (the method is then
// shown for reference only), or null when it is in the blend. Keys are the
// summary's method names: "DCF", "Comps (median)", "DDM", "FCFE".
export function excludedReason(
  s: { excluded_from_blend?: Record<string, string> | null },
  method: string
): string | null {
  const excluded = s.excluded_from_blend || {};
  if (!Object.prototype.hasOwnProperty.call(excluded, method)) return null;
  return excluded[method] || "reference only";
}

// --- Beta ------------------------------------------------------------------ //
// The Blume weights the market data applies to Yahoo's raw beta (copies of
// equity_valuation/data/market.py's; a Python test pins them).
export const BLUME_RAW_WEIGHT = 0.67;
export const BLUME_MARKET_WEIGHT = 0.33;

// How a beta stat is labelled. The market data Blume-adjusts Yahoo's beta
// toward 1 (BLUME_RAW_WEIGHT x raw + BLUME_MARKET_WEIGHT) and keeps the raw
// figure, so an adjusted beta reads "Beta (adj.)" with the raw value in its
// subtitle and tooltip. `source` is the WACC's detail.beta_source when there
// is one ("DEFAULT_BETA" when no usable market beta was found).
export function betaStat(
  beta: number | null | undefined,
  raw: number | null | undefined,
  source?: unknown
): { label: string; value: string; sub?: string; title?: string } {
  const value = fmtNum(beta, 2);
  if (source === "DEFAULT_BETA") {
    return {
      label: "Beta (default)",
      value,
      title: "No usable market beta, so the models use the default beta.",
    };
  }
  const hasRaw = isNum(beta) && isNum(raw) && Math.abs(raw - beta) > 1e-9;
  const adjusted =
    hasRaw || (typeof source === "string" && source.includes("adjusted"));
  if (!adjusted || !isNum(beta)) return { label: "Beta", value };
  if (!hasRaw) {
    return {
      label: "Beta (adj.)",
      value,
      title: "Adjusted toward 1 by the market data (see the data notes).",
    };
  }
  return {
    label: "Beta (adj.)",
    value,
    sub: `raw ${fmtNum(raw, 2)}`,
    title:
      `Yahoo's 5-year monthly beta ${fmtNum(raw, 3)}, Blume-adjusted toward 1 ` +
      `(${BLUME_RAW_WEIGHT} × raw + ${BLUME_MARKET_WEIGHT}) to ${fmtNum(beta, 3)} for CAPM.`,
  };
}

// A rate with one decimal, or two when the second is needed (2.5%, 3.0%,
// 2.25%): assumption steps run in quarter points.
export function fmtRate(x: number | null | undefined): string {
  if (!isNum(x)) return "—";
  const tenths = x * 1000;
  return fmtPct(x, { decimals: Math.abs(tenths - Math.round(tenths)) < 1e-6 ? 1 : 2 });
}

// --- Engine text in the dashboard ------------------------------------------ //
// The engine's warnings and notes are written for the command line. Rewrite
// the phrases that name a CLI flag so they point at the dashboard control
// that does the same thing; any other text is returned unchanged.
const CLI_PHRASES: [RegExp, string][] = [
  [/pass --peers to supply them/g, "add peers in the Trading comps box"],
  [/Supply peers with --peers/g, "Add peers in the Trading comps box"],
  [/supply peers with --peers/g, "add peers in the Trading comps box"],
  [/--comps-basis ttm in the CLI/g, "the TTM toggle in the Trading comps box"],
];

export function uiText(text: string): string {
  return CLI_PHRASES.reduce((t, [re, to]) => t.replace(re, to), text);
}

// Warnings as the dashboard shows them: rewritten (uiText), each once.
export function uiWarnings(warnings: readonly string[] | null | undefined): string[] {
  return [...new Set((warnings || []).filter((w) => typeof w === "string").map(uiText))];
}

// The warnings that explain a DCF with no price, for its empty card: the
// DCF's own ("DCF: ...", "WARNING: DCF: ...", "DCF excluded from ...") and,
// for a flagged company (`flagged`), the engine's note on why its FCFF DCF
// is reference only. A bank's report carries a dozen other notes that would
// bury the one that matters ("EBIT margin unavailable ... set a target EBIT
// margin"). Every warning when none of them is there; rewritten (uiWarnings).
const FLAG_NOTE = /^(Financial institution|Captive finance arm|Debt-funded lessor)\b/;

export function dcfWarnings(
  warnings: readonly string[] | null | undefined,
  flagged: boolean
): string[] {
  const all = uiWarnings(warnings);
  const own = all.filter((w) => /^(WARNING: )?DCF[: ]/.test(w)
    || (flagged && FLAG_NOTE.test(w) && w.includes("DCF")));
  return own.length > 0 ? own : all;
}

// The engine copies some comps notes into the warnings as "Comps: <note>"
// (peers not used, equity multiples only, ...) and adds comps warnings of its
// own (no usable peers, a failed fetch). For the comps card: the comps
// warnings, without the "Comps" label (and capitalised), that do not repeat
// one of its notes, rewritten (uiText) and each once.
export function compsWarnings(
  warnings: readonly string[] | null | undefined,
  notes: readonly string[] | null | undefined
): string[] {
  const seen = new Set((notes || []).map(uiText));
  const out: string[] = [];
  for (const w of warnings || []) {
    if (typeof w !== "string") continue;
    const m = /^(WARNING: )?Comps( failed)?: (.*)$/s.exec(w);
    if (!m) continue;
    const body = m[2] ? `Comps failed: ${m[3]}` : m[3].charAt(0).toUpperCase() + m[3].slice(1);
    const text = uiText(`${m[1] ?? ""}${body}`);
    if (seen.has(text)) continue;
    seen.add(text);
    out.push(text);
  }
  return out;
}

// --- Peer status ------------------------------------------------------------ //
// A tone (ui.Badge's) and a reason for what comps did with each requested
// peer (comps.peer_status).
export function peerStatusTone(status: string): "up" | "down" | "flat" | "neutral" {
  switch (status) {
    case "used":
      return "up";
    case "not found":
    case "not fetched":
      return "down";
    case "no usable multiple":
      return "flat";
    default:
      return "neutral";
  }
}

export function peerStatusReason(status: string): string {
  switch (status) {
    case "used":
      return "Used in the peer multiples.";
    case "not found":
      return "Not used: no market data came back for this ticker. Check the symbol.";
    case "no usable multiple":
      return "Not used: its data has no positive EV/EBITDA, EV/Sales, P/E or P/B.";
    case "is target":
      return "Not used: this is the company being valued.";
    case "not fetched":
      return "Not used: the peer data could not be fetched. Try again later.";
    default:
      return `Status: ${status}.`;
  }
}

// --- Why there is no verdict ------------------------------------------------ //
// Who a flagged company is (summary.financial_kind), with its article.
const KIND_WHOM: Record<string, string> = {
  bank: "a bank",
  insurer: "an insurer",
  reit: "a REIT",
  lender: "a lender",
  captive_finance: "a company with a consolidated captive finance arm",
  lessor: "a debt-funded lessor",
  financial: "a financial institution",
};

export function financialWhom(s: BlendFields): string | null {
  const kind = s.financial_kind;
  if (kind) return KIND_WHOM[kind] ?? "a financial institution";
  return s.financial_institution ? "a financial institution" : null;
}

// Flagged kinds whose trading comps price on P/E and P/B only (a copy of
// equity_valuation/models/comps.py's EQUITY_MULTIPLES_ONLY_KINDS; a test pins
// it). Their debt funds the business (deposits, a loan book, a finance arm's
// receivables, a leased fleet), so EV/EBITDA and EV/Sales mean little: a
// bank's EBIT is zero by design and its EBITDA is D&A alone.
export const EQUITY_MULTIPLES_ONLY_KINDS = [
  "bank", "insurer", "lender", "financial", "captive_finance", "lessor",
];

// Usable peer values a comps multiple needs for an implied price (a copy of
// equity_valuation/config.py's COMPS_MIN_PEER_VALUES; a test pins it).
export const COMPS_MIN_PEER_VALUES = 3;
// Start of the comps note naming the multiples with too few peer values
// (equity_valuation/models/comps.py FEW_PEER_VALUES_NOTE_PREFIX).
export const FEW_PEER_VALUES_NOTE_PREFIX = "Too few peer values";

// True when the engine prices the company on equity multiples only (a flag
// without a kind, from an older backend, counts as "financial").
export function equityMultiplesOnly(s: BlendFields): boolean {
  if (s.financial_kind) return EQUITY_MULTIPLES_ONLY_KINDS.includes(s.financial_kind);
  return Boolean(s.financial_institution);
}

// Column headers for a projection table. The FCFE numbers its years by fiscal
// year (2026, ...), the DCF by forecast index (1..n): an index is mapped onto
// the latest fiscal year in the financials, so both tables label the same
// periods alike (equity_valuation/report/excel.py's _year_labels does the
// same); without one it stays "Y1", "Y2", ...
export function projectionYearLabels(
  years: readonly unknown[] | null | undefined,
  fiscalYears?: readonly unknown[] | null
): string[] {
  const fys = (fiscalYears || []).filter(isNum).map((y) => Math.trunc(y));
  const lastFy = fys.length > 0 ? fys[fys.length - 1] : null;
  return (years || []).map((y) => {
    if (!isNum(y)) return String(y ?? "—");
    if (y >= 1000) return String(Math.trunc(y));
    return lastFy != null ? String(lastFy + Math.trunc(y)) : `Y${Math.trunc(y)}`;
  });
}

// --- DDM ------------------------------------------------------------------- //
// A DDM variant's display name, as the segmented control, the HTML report and
// the workbook write it (excel.DDM_METHOD_LABELS, which a test pins); an
// unknown identifier is shown as it is.
export const DDM_METHOD_LABELS: Record<string, string> = {
  two_stage: "Two-stage",
  h_model: "H-model",
  gordon: "Gordon growth",
};

export function ddmMethodLabel(method: unknown): string {
  if (typeof method !== "string" || !method) return "—";
  return DDM_METHOD_LABELS[method] ?? method;
}

function joinNames(names: string[]): string {
  return names.length <= 1
    ? names.join("")
    : `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}

// A line under the header verdict when there is no verdict ("N/A"), no
// blended target, or methods left out of the blend, built only from the
// summary's structured fields (excluded_from_blend and its reasons,
// financial_kind, verdict_withheld, needsPeers), never from warning text. `addPeers` asks for an
// "Add peers" action: the blend has no comps, and trading comps would give
// (or add to) a target. Null when there is nothing to explain.
export function verdictExplanation(
  s: BlendFields,
  comps?: CompsFields
): { text: string; addPeers: boolean } | null {
  const excluded = s.excluded_from_blend || {};
  const outNames = Object.keys(excluded);
  const noTarget = !isNum(s.blended_target);
  const noVerdict = s.recommendation === "N/A";
  if (!noVerdict && !noTarget && outNames.length === 0) return null;

  const methods = s.methods || {};
  const hasComps = Object.keys(methods).some((n) => n.startsWith("Comps"));
  const inBlend = Object.entries(methods)
    .filter(([n, v]) => isNum(v) && !Object.prototype.hasOwnProperty.call(excluded, n))
    .map(([n]) => n);
  const whom = financialWhom(s);
  // Peers were supplied but gave no comps value.
  const peersGaveNone = !hasComps && Boolean(comps && comps.peers && comps.peers.length > 0);
  // ...because too few peers had each multiple: the comps notes say so (each
  // multiple needs at least COMPS_MIN_PEER_VALUES usable peer values; GM's F,
  // RIVN and TSLA gave one P/E and two P/Bs). Other causes, such as a missing
  // share count, are not fixed by adding peers.
  const tooFewPeers = peersGaveNone
    && Boolean(comps?.notes?.some((n) => n.startsWith(FEW_PEER_VALUES_NOTE_PREFIX)));
  // The engine keeps a failed method as 0.00 with an exclusion reason.
  const failed = (n: string) => /^(no valuation|non-finite)/.test(excluded[n] ?? "");
  const sentences: string[] = [];

  if (needsPeers(s, comps)) {
    sentences.push(`No target: every method left for ${whom ?? "this company"} is reference only.`);
  } else if (noTarget) {
    // "No usable value" only when no method has one: a flagged company's
    // DCF, DDM and FCFE can all have values that are reference only.
    const anyValue = Object.entries(methods).some(([n, v]) => isNum(v) && !failed(n));
    let line = !anyValue
      ? "No target: no method gave a usable value"
      : inBlend.length === 0 && whom != null
        ? `No target: every method left for ${whom} is reference only`
        : "No target";
    if (tooFewPeers) {
      line += `, and the peers supplied give no comps value (each multiple needs at least ${COMPS_MIN_PEER_VALUES} usable peer values)`;
    } else if (peersGaveNone) {
      line += ", and the peers supplied give no usable comps price";
    }
    sentences.push(`${line}.`);
  } else if (noVerdict) {
    // The engine's reason: a corporate action (it withholds any target's
    // verdict), else a DDM-only blend.
    const withheld = typeof s.verdict_withheld === "string" ? s.verdict_withheld : "";
    if (withheld.startsWith("a corporate action")) {
      sentences.push(`No verdict: Yahoo records ${withheld}, so the latest statements may `
        + "still include a business the company no longer owns.");
    } else if (inBlend.length === 1 && inBlend[0] === "DDM") {
      sentences.push(
        "No verdict: the target rests on the DDM alone, which counts dividends but not buybacks.");
    } else {
      sentences.push(withheld
        ? `No verdict: ${withheld}; the target is shown for reference.`
        : "No verdict.");
    }
  }
  if (outNames.length > 0) {
    // One entry per reason the engine gives.
    const byReason = new Map<string, string[]>();
    for (const n of outNames) {
      const why = excluded[n] || "reference only";
      byReason.set(why, [...(byReason.get(why) ?? []), n]);
    }
    const groups = [...byReason].map(([why, names]) => `${joinNames(names)} (${why})`);
    sentences.push(`Left out of the target: ${groups.join("; ")}.`);
  }
  const addPeers = !hasComps;
  if (addPeers && tooFewPeers) {
    sentences.push(whom
      ? `Add more peers: with enough usable peer values, comps give a P/E and P/B value for ${whom}.`
      : "Add more peers: with enough usable peer values, comps add a trading-comps value.");
  } else if (addPeers && !peersGaveNone) {
    sentences.push(whom
      ? `Peers give a P/E and P/B comps value for ${whom}.`
      : "Peers add a trading-comps value.");
  }
  return { text: sentences.join(" "), addPeers };
}
