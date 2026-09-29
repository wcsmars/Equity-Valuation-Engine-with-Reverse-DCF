// Formatting helpers shared by every panel. All tolerate null/undefined/NaN.

const SYMBOLS: Record<string, string> = {
  USD: "$",
  EUR: "€",
  GBP: "£",
  JPY: "¥",
  CAD: "C$",
  AUD: "A$",
  CHF: "CHF ",
  CNY: "¥",
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

// Map an upside/recommendation to a semantic color class.
export function toneForUpside(upside: number | null | undefined): string {
  if (!isNum(upside)) return "text-ink-dim";
  if (upside >= 0.15) return "text-up";
  if (upside <= -0.15) return "text-down";
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
// null when the verdict is withheld (a blend that rests on the DDM alone).
// A null is shown as text, never as a figure, and is not coloured. These rules
// copy equity_valuation/report/excel.py, which the Excel and HTML reports, the
// memo and deck and the AI context share; a Python test pins the copy.
export const NO_TARGET_SUPPLY_PEERS = "No target (supply peers)";

interface BlendFields {
  methods?: Record<string, number | null> | null;
  blended_target?: number | null;
  blended_upside?: number | null;
  currency?: string | null;
  recommendation?: string | null;
  excluded_from_blend?: Record<string, string> | null;
  financial_institution?: string | null;
  financial_kind?: string | null;
}

// The report's trading comps (null when comps did not run).
type CompsFields = { peers?: unknown[] | null } | null | undefined;

// No target because the company is flagged (a bank, a captive-finance group,
// a lessor, ...) and every method it has is reference only, and no usable
// peers were supplied: trading comps would give a target (peers are never
// found automatically). Not when peers were supplied but comps gave no price,
// nor when an ordinary company's methods gave no valuation: those read "n/a".
export function needsPeers(s: BlendFields, comps?: CompsFields): boolean {
  if (isNum(s.blended_target)) return false;
  const names = Object.keys(s.methods || {});
  if (names.length === 0 || names.some((n) => n.startsWith("Comps"))) return false;
  if (!(s.financial_kind || s.financial_institution)) return false;
  return !(comps && comps.peers && comps.peers.length > 0);
}

export function fmtBlendedTarget(s: BlendFields, comps?: CompsFields): string {
  if (isNum(s.blended_target)) return fmtMoney(s.blended_target, s.currency);
  return needsPeers(s, comps) ? NO_TARGET_SUPPLY_PEERS : "n/a";
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
