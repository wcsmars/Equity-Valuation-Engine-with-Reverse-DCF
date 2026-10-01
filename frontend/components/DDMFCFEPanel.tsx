"use client";

// Dividend discount model (DDM) + levered FCFE panel. Follows the structure,
// density, and color usage of ValuationSummary (the reference panel).

import React from "react";
import type { Report } from "@/lib/types";
import {
  excludedReason,
  fmtBig,
  fmtCount,
  fmtMoney,
  fmtNum,
  fmtPct,
  toneForMethodUpside,
} from "@/lib/format";
import { Card, EmptyState, Stat, Table, TD, TH } from "@/components/ui";

// Humanize a snake_case detail key, e.g. "terminal_growth" -> "terminal growth".
function humanizeKey(key: string): string {
  return key.replace(/_/g, " ");
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
  const entries = Object.entries(detail || {}).filter(([k]) => k !== "notes");
  if (entries.length === 0 && notes.length === 0) return null;
  return (
    <div className="mt-3 border-t border-line pt-3">
      <div className="grid grid-cols-1 gap-x-4 gap-y-1 sm:grid-cols-2">
        {entries.map(([key, value]: [string, unknown]) => (
          <div
            key={key}
            className="flex min-w-0 items-baseline justify-between gap-3"
          >
            <span className="shrink-0 text-xs capitalize text-ink-dim">
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

export default function DDMFCFEPanel({ report }: { report: Report }) {
  const cur = report.summary.currency;
  const price = report.current_price;
  const ddm = report.ddm;
  const fcfe = report.fcfe;

  const ddmUpside =
    ddm && price ? ddm.implied_price / price - 1 : null;
  const fcfeUpside =
    fcfe && price ? fcfe.implied_price / price - 1 : null;

  const fcfeYears = fcfe?.years || [];
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
        {ddm == null ? (
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
                  <span className="text-sm">{ddm.method || "—"}</span>
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
      </Card>

      <Card
        title="Levered FCFE"
        subtitle={refOnly(
          fcfeOut,
          "Free cash flow to equity discounted at the cost of equity"
        )}
      >
        {fcfe == null ? (
          <EmptyState
            title="FCFE unavailable"
            hint={(report.warnings || []).join(" · ")}
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
                      {fcfeYears.map((y: number, i: number) => (
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
