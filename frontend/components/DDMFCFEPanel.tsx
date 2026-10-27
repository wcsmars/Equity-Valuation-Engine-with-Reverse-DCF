"use client";

// Dividend discount model (DDM) + levered FCFE panel. Follows the structure,
// density, and color usage of ValuationSummary (the reference panel).
//
// The DDM card carries its own drivers (method, high-growth rate and years),
// since for a bank, insurer or other flagged company the DDM is often the
// method that sets the target while the DCF is reference only.

import React from "react";
import type { Assumptions, DDM, DDMMethod, Report } from "@/lib/types";
import { assumptionsFromUsed, diffAssumptions } from "@/lib/assumptions";
import {
  ddmMethodLabel,
  excludedReason,
  fmtBig,
  fmtCount,
  fmtMoney,
  fmtNum,
  fmtPct,
  fmtRate,
  projectionYearLabels,
  toneForMethodUpside,
  uiWarnings,
} from "@/lib/format";
import {
  AssumptionSlider,
  Button,
  Card,
  EmptyState,
  Segmented,
  Spinner,
  Stat,
  Table,
  TD,
  TH,
} from "@/components/ui";

const DDM_METHODS: DDMMethod[] = ["gordon", "two_stage", "h_model"];

// The backend takes a high-growth rate strictly between -50% and 60%; keep a
// slider widened to an engine-derived rate beyond that inside the range.
const HIGH_GROWTH_MIN = -0.4975;
const HIGH_GROWTH_MAX = 0.5975;

const finite = (x: unknown): number | null =>
  typeof x === "number" && Number.isFinite(x) ? x : null;

// The DDM's drivers, defaulting to what the engine used (ddm.detail: the
// derived high growth and its years). Edits go into the shared draft, so the
// DCF card's unapplied-changes bar lists them too; Recompute applies them and
// Defaults hands the drivers back to the engine.
function DDMDrivers({
  report,
  ddm,
  assumptions,
  setAssumptions,
  onRecompute,
  recomputing,
}: {
  report: Report;
  ddm: DDM;
  assumptions: Assumptions;
  setAssumptions: (a: Assumptions) => void;
  onRecompute: (override?: Assumptions) => void;
  recomputing: boolean;
}) {
  const detail = (ddm.detail || {}) as Record<string, unknown>;
  const used = report.assumptions_used;
  const changes = used ? diffAssumptions(assumptions, assumptionsFromUsed(used)) : [];
  const changed = new Set<string>(changes.map((c) => c.key));
  const pending = changes.filter((c) => c.key.startsWith("ddm_")).length;

  const engineMethod = DDM_METHODS.find((m) => m === ddm.method) ?? null;
  const method: DDMMethod =
    assumptions.ddm_method ?? engineMethod ?? used?.ddm_method ?? "two_stage";
  const hHalf = finite(detail.H_half);
  const highGrowth =
    assumptions.ddm_high_growth_rate ??
    finite(detail.high_growth) ??
    used?.ddm_high_growth_rate ??
    used?.terminal_growth ??
    0.025;
  const highGrowthYears =
    assumptions.ddm_high_growth_years ??
    finite(detail.high_growth_years) ??
    (hHalf != null ? hHalf * 2 : null) ??
    used?.ddm_high_growth_years ??
    5;

  return (
    <div className="mt-4 border-t border-line pt-3">
      <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-ink-faint">
        DDM drivers
      </div>
      <div className="space-y-3">
        <div>
          <div className="mb-1.5 text-xs font-medium text-ink-dim">
            Method
            {changed.has("ddm_method") && (
              <span className="ml-1.5 text-[11px] font-normal text-flat">changed</span>
            )}
          </div>
          <Segmented<DDMMethod>
            label="DDM method"
            value={method}
            options={[
              { label: "Gordon", value: "gordon" },
              { label: "Two-stage", value: "two_stage" },
              { label: "H-model", value: "h_model" },
            ]}
            onChange={(v) => setAssumptions({ ...assumptions, ddm_method: v })}
          />
        </div>
        {method === "gordon" ? (
          <p className="text-[11px] text-ink-faint">
            Gordon grows the dividend at the terminal growth rate from the first
            year, so it has no high-growth stage.
          </p>
        ) : (
          <>
            <AssumptionSlider
              label="High-growth rate"
              value={highGrowth}
              min={-0.1}
              max={0.3}
              step={0.0025}
              anchors={[finite(detail.high_growth)]}
              format={(v) => fmtRate(v)}
              hint={
                method === "h_model"
                  ? "Starting dividend growth, fading linearly to terminal growth"
                  : "Dividend growth in each high-growth year (the engine's own: ROE × retention or the dividend CAGR; Defaults restores it)"
              }
              changed={changed.has("ddm_high_growth_rate")}
              onChange={(v) =>
                setAssumptions({
                  ...assumptions,
                  ddm_high_growth_rate: Math.min(HIGH_GROWTH_MAX, Math.max(HIGH_GROWTH_MIN, v)),
                })
              }
            />
            <AssumptionSlider
              label="High-growth years"
              value={highGrowthYears}
              min={1}
              max={15}
              step={1}
              anchors={[finite(detail.high_growth_years)]}
              format={(v) => `${v}y`}
              valueText={(v) => `${v} years`}
              changed={changed.has("ddm_high_growth_years")}
              onChange={(v) =>
                setAssumptions({ ...assumptions, ddm_high_growth_years: Math.round(v) })
              }
            />
          </>
        )}
        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={() => onRecompute()} disabled={recomputing}>
            {recomputing ? (
              <>
                <Spinner /> Recomputing
              </>
            ) : (
              "Recompute"
            )}
          </Button>
          <Button
            variant="ghost"
            disabled={recomputing}
            onClick={() => {
              // Hand the DDM drivers back to the engine; the recompute
              // normalises the draft to what it used.
              const next: Assumptions = {
                ...assumptions,
                ddm_method: undefined,
                ddm_high_growth_rate: undefined,
                ddm_high_growth_years: undefined,
              };
              setAssumptions(next);
              onRecompute(next);
            }}
          >
            Defaults
          </Button>
          {pending > 0 && (
            <span className="text-[11px] text-flat">
              {`${pending} unapplied DDM change${pending === 1 ? "" : "s"}`}
            </span>
          )}
        </div>
        <p className="text-[11px] text-ink-faint">
          Terminal growth and the cost of equity (risk-free rate, equity risk
          premium) come from the DCF assumptions. The FCFE does not use these
          drivers.
        </p>
      </div>
    </div>
  );
}

// Labels for the engine's DDM and FCFE detail keys (the DDM's follow the HTML
// report's). Any other key is shown in sentence case (humanizeKey).
const DETAIL_LABELS: Record<string, string> = {
  cost_of_equity: "Cost of equity",
  D0: "Latest dividend per share (D0)",
  terminal_growth: "Terminal growth (input)",
  terminal_growth_used: "Terminal growth (used)",
  growth: "Perpetual growth",
  high_growth: "High-growth rate",
  high_growth_years: "High-growth years",
  H_half: "Half of the high-growth years (H)",
  dividends: "Stage-1 dividends per share",
  stage1_pvs: "PV of each stage-1 dividend",
  pv_stage1: "PV of stage-1 dividends",
  terminal_value: "Terminal value (per share)",
  pv_terminal: "PV of terminal value",
  implied_price: "Implied price",
  delta_debt_policy: "Debt policy",
  revenue_growth_path: "Revenue growth path",
  revenue: "Revenue",
  net_margin: "Net margin",
  net_margin_path: "Net margin path",
  da_pct_revenue: "D&A (% of revenue)",
  capex_pct_revenue: "Capex (% of revenue)",
  capex_pct_path: "Capex path (% of revenue)",
  nwc_pct_delta_revenue: "Working capital (% of revenue change)",
  equity_value: "Equity value",
};

// A detail key's label: DETAIL_LABELS, else the snake_case key in sentence
// case ("payout_ratio" -> "Payout ratio").
function humanizeKey(key: string): string {
  if (Object.prototype.hasOwnProperty.call(DETAIL_LABELS, key)) return DETAIL_LABELS[key];
  const words = key.replace(/_/g, " ").trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

// Format an untrusted detail value. Numbers that look like rates/growth render
// as percentages, money-like keys (prices, dividends, values, revenue) as
// currency (compact when large), arrays element-wise; strings pass through.
function fmtDetailValue(key: string, value: unknown, cur: string): string {
  if (Array.isArray(value)) {
    return value.map((v) => fmtDetailValue(key, v, cur)).join(", ") || "—";
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "—";
    const k = key.toLowerCase();
    // Year counts (e.g. high_growth_years) are integers, not rates.
    if (k.includes("year")) return String(Math.round(value));
    const rateLike =
      /(growth|rate|margin|yield|return|wacc|cost|premium|payout|retention|pct|equity_w|weight)/.test(
        k
      ) || k === "ke";
    if (rateLike) return fmtPct(value);
    const moneyLike =
      /(price|value|pv|dividend|revenue|equity)/.test(k) || k === "d0";
    if (moneyLike)
      return Math.abs(value) >= 1e5 ? fmtBig(value, cur) : fmtMoney(value, cur);
    return fmtNum(value, 2);
  }
  if (value == null) return "—";
  if (key === "method") return ddmMethodLabel(value);
  return String(value);
}

function DetailList({
  detail,
  cur,
}: {
  detail: Record<string, unknown> | null | undefined;
  cur: string;
}) {
  // Engine notes are free-text sentences: list them under the grid.
  const rawNotes = detail?.notes;
  const notes = Array.isArray(rawNotes)
    ? rawNotes.filter((n): n is string => typeof n === "string" && n !== "")
    : [];
  // The method is the card's Method stat.
  const entries = Object.entries(detail || {}).filter(
    ([k]) => !["notes", "valuation_available", "method"].includes(k)
  );
  if (entries.length === 0 && notes.length === 0) return null;
  return (
    <div className="mt-3 border-t border-line pt-3">
      <div className="grid grid-cols-1 gap-x-4 gap-y-1 sm:grid-cols-2">
        {entries.map(([key, value]: [string, unknown]) => (
          <div
            key={key}
            className="flex min-w-0 items-baseline justify-between gap-3"
          >
            <span className="shrink-0 text-xs text-ink-dim">
              {humanizeKey(key)}
            </span>
            <span className="num min-w-0 break-words text-right text-xs text-ink">
              {fmtDetailValue(key, value, cur)}
            </span>
          </div>
        ))}
      </div>
      {notes.length > 0 && (
        <ul className="mt-2 list-disc space-y-0.5 pl-5 text-xs text-ink-faint">
          {notes.map((n, i) => (
            <li key={i}>{n}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default function DDMFCFEPanel({
  report,
  assumptions = {},
  setAssumptions = () => {},
  onRecompute = () => {},
  recomputing = false,
}: {
  report: Report;
  assumptions?: Assumptions;
  setAssumptions?: (a: Assumptions) => void;
  onRecompute?: (override?: Assumptions) => void;
  recomputing?: boolean;
}) {
  const cur = report.summary.currency;
  const price = report.current_price;
  const ddm = report.ddm;
  const fcfe = report.fcfe;

  const ddmUpside =
    ddm && ddm.implied_price != null && price > 0 ? ddm.implied_price / price - 1 : null;
  const fcfeUpside =
    fcfe && fcfe.implied_price != null && price > 0 ? fcfe.implied_price / price - 1 : null;

  const fcfeYears = fcfe?.years || [];
  // Fiscal years (the DCF's projection maps its 1..n onto the same ones).
  const fcfeYearLabels = projectionYearLabels(fcfeYears, report.company?.financials?.fiscal_years);
  // Methods left out of the blended target are shown for reference only (a
  // flagged company's FCFE, or a low-payout DDM), with the engine's reason and
  // an uncoloured upside (also uncoloured when the engine gives no verdict).
  const ddmOut = excludedReason(report.summary, "DDM");
  const fcfeOut = excludedReason(report.summary, "FCFE");
  const refOnly = (why: string | null, base: string) =>
    why ? `${base} · Reference only, not in the blended target: ${why}` : base;

  return (
    <div className="grid grid-cols-1 gap-4">
      <Card
        title="Dividend discount model (DDM)"
        subtitle={refOnly(
          ddmOut,
          "Implied value from projected dividends discounted at the cost of equity"
        )}
      >
        {ddm == null || ddm.implied_price == null ? (
          <EmptyState
            title="DDM unavailable"
            hint="No valid dividend valuation was produced. Check the report warnings and dividend assumptions."
          />
        ) : (
          <>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
              <Stat
                label={ddmOut ? "Implied price (ref.)" : "Implied price"}
                value={fmtMoney(ddm.implied_price, cur)}
                tone={toneForMethodUpside(report.summary, "DDM", ddmUpside)}
                sub={fmtPct(ddmUpside, { signed: true })}
              />
              <Stat
                label="Method"
                value={
                  <span className="text-sm">{ddmMethodLabel(ddm.method)}</span>
                }
              />
              <Stat
                label="Cost of equity"
                value={fmtPct(ddm.cost_of_equity)}
              />
            </div>
            <DetailList detail={ddm.detail} cur={cur} />
          </>
        )}
        {ddm != null && (
          <DDMDrivers
            report={report}
            ddm={ddm}
            assumptions={assumptions}
            setAssumptions={setAssumptions}
            onRecompute={onRecompute}
            recomputing={recomputing}
          />
        )}
      </Card>

      <Card
        title="Levered FCFE"
        subtitle={refOnly(
          fcfeOut,
          "Free cash flow to equity discounted at the cost of equity"
        )}
      >
        {fcfe == null || fcfe.implied_price == null ? (
          <EmptyState
            title="FCFE unavailable"
            hint={uiWarnings(report.warnings).join(" · ")}
          />
        ) : (
          <>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <Stat
                label={fcfeOut ? "Implied price (ref.)" : "Implied price"}
                value={fmtMoney(fcfe.implied_price, cur)}
                tone={toneForMethodUpside(report.summary, "FCFE", fcfeUpside)}
                sub={fmtPct(fcfeUpside, { signed: true })}
              />
              <Stat
                label="Cost of equity"
                value={fmtPct(fcfe.cost_of_equity)}
              />
              <Stat
                label="Equity value"
                value={fmtBig(fcfe.equity_value, cur)}
              />
              <Stat label="Shares" value={fmtCount(fcfe.shares)} />
            </div>

            {fcfeYears.length > 0 && (
              <div className="mt-4">
                <Table>
                  <thead>
                    <tr>
                      <TH align="left">Year</TH>
                      {fcfeYearLabels.map((y, i) => (
                        <TH key={i}>{y}</TH>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <TD align="left">FCFE</TD>
                      {fcfeYears.map((_y: number, i: number) => (
                        <TD key={i} num>
                          {fmtBig(fcfe.fcfe?.[i], cur)}
                        </TD>
                      ))}
                    </tr>
                    <tr>
                      <TD align="left">PV of FCFE</TD>
                      {fcfeYears.map((_y: number, i: number) => (
                        <TD key={i} num>
                          {fmtBig(fcfe.pv_fcfe?.[i], cur)}
                        </TD>
                      ))}
                    </tr>
                  </tbody>
                </Table>
              </div>
            )}

            <div className="mt-4 grid grid-cols-2 gap-x-4 gap-y-1 border-t border-line pt-3 sm:grid-cols-4">
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-xs text-ink-dim">Terminal value</span>
                <span className="num text-xs text-ink">
                  {fmtBig(fcfe.terminal_value, cur)}
                </span>
              </div>
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-xs text-ink-dim">PV terminal</span>
                <span className="num text-xs text-ink">
                  {fmtBig(fcfe.pv_terminal, cur)}
                </span>
              </div>
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-xs text-ink-dim">Equity value</span>
                <span className="num text-xs text-ink">
                  {fmtBig(fcfe.equity_value, cur)}
                </span>
              </div>
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-xs text-ink-dim">Implied price</span>
                <span className="num text-xs text-ink">
                  {fmtMoney(fcfe.implied_price, cur)}
                </span>
              </div>
            </div>

            <DetailList detail={fcfe.detail} cur={cur} />
          </>
        )}
      </Card>
    </div>
  );
}
