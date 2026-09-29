"use client";

// Overview panel + the football-field chart. This is the REFERENCE panel —
// other panels follow its structure, primitives, and color usage.

import React from "react";
import type { Report } from "@/lib/types";
import {
  betaStat,
  excludedReason,
  fmtBig,
  fmtBlendedTarget,
  fmtBlendedUpside,
  fmtCount,
  fmtMoney,
  fmtPct,
  footballFieldTone,
  toneForBlendedUpside,
  toneForMethodUpside,
} from "@/lib/format";
import { Badge, Card, Stat, Table, TD, TH } from "@/components/ui";

function FootballField({ report }: { report: Report }) {
  const rows = report.football_field || [];
  const price = report.current_price;
  const cur = report.summary.currency;
  if (rows.length === 0) {
    return <p className="text-sm text-ink-faint">No valuation ranges available.</p>;
  }

  const lows = rows.map((r) => r.low);
  const highs = rows.map((r) => r.high);
  let min = Math.min(...lows, price);
  let max = Math.max(...highs, price);
  const pad = (max - min) * 0.06 || 1;
  min -= pad;
  max += pad;
  const span = max - min || 1;
  const pos = (v: number) => ((v - min) / span) * 100;

  return (
    <div>
      <div className="space-y-3">
        {rows.map((r) => {
          const left = pos(r.low);
          const width = Math.max(pos(r.high) - left, 0.8);
          // Grey for a reference-only method or when there is no verdict.
          const tone = footballFieldTone(report.summary, r.method, r.base, price);
          return (
            <div key={r.method} className="flex items-center gap-3">
              <div className="w-32 shrink-0 truncate text-right text-xs text-ink-dim">
                {r.method}
              </div>
              <div className="relative h-7 flex-1">
                <div
                  className={`absolute top-1/2 -translate-y-1/2 rounded border ${tone}`}
                  style={{ left: `${left}%`, width: `${width}%`, height: 12 }}
                />
                {/* base marker */}
                <div
                  className="absolute top-1/2 h-4 w-[2px] -translate-x-1/2 -translate-y-1/2 bg-ink"
                  style={{ left: `${pos(r.base)}%` }}
                  title={`base ${fmtMoney(r.base, cur)}`}
                />
                <div
                  className="num absolute top-1/2 -translate-y-1/2 text-[10px] text-ink-faint"
                  style={{ left: `calc(${pos(r.high)}% + 6px)` }}
                >
                  {fmtMoney(r.high, cur, 0)}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* current-price reference line spanning the chart */}
      <div className="mt-2 flex items-center gap-3">
        <div className="w-32 shrink-0" />
        <div className="relative h-5 flex-1">
          <div
            className="absolute top-0 h-5 w-[2px] -translate-x-1/2 bg-brand"
            style={{ left: `${pos(price)}%` }}
          />
          <div
            className="num absolute top-0 -translate-x-1/2 whitespace-nowrap text-[10px] text-brand"
            style={{ left: `${pos(price)}%` }}
          >
            price {fmtMoney(price, cur, 0)}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function ValuationSummary({ report }: { report: Report }) {
  const s = report.summary;
  const m = report.company.market;
  const cur = s.currency;
  const methods = Object.entries(s.methods || {});
  const excluded = s.excluded_from_blend || {};
  const beta = betaStat(m.beta, m.raw_beta);

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
      <Card
        title="Valuation football field"
        subtitle="Implied value range by method vs. current price"
        className="lg:col-span-2"
      >
        <FootballField report={report} />
      </Card>

      <Card
        title="Method summary"
        subtitle={
          Object.keys(excluded).length > 0
            ? `Not in the blended target: ${Object.entries(excluded)
                .map(([name, why]) => `${name} (${why})`)
                .join("; ")}`
            : undefined
        }
      >
        <Table>
          <thead>
            <tr>
              <TH align="left">Method</TH>
              <TH>Implied</TH>
              <TH>Upside</TH>
            </tr>
          </thead>
          <tbody>
            {methods.map(([name, price]) => {
              const up =
                price != null && s.current_price
                  ? price / s.current_price - 1
                  : null;
              // Shown for reference only: the blended target leaves it out
              // (a flagged company's DCF and FCFE, whatever its kind). Its
              // upside, and every upside when there is no verdict, is dim.
              const why = excludedReason(s, name);
              return (
                <tr key={name}>
                  <TD align="left">
                    {name}
                    {why && (
                      <span
                        className="ml-1.5 text-[10px] text-ink-faint"
                        title={`Not in the blended target: ${why}`}
                      >
                        (not in blend)
                      </span>
                    )}
                  </TD>
                  <TD num>{fmtMoney(price, cur)}</TD>
                  <TD num className={toneForMethodUpside(s, name, up)}>
                    {fmtPct(up, { signed: true })}
                  </TD>
                </tr>
              );
            })}
            {/* No target -> "n/a" or "No target (supply peers)"; a withheld
                verdict -> the upside reads "n/a", uncoloured. */}
            <tr className="font-semibold">
              <TD align="left">Blended target</TD>
              <TD num>{fmtBlendedTarget(s, report.comps)}</TD>
              <TD num className={toneForBlendedUpside(s)}>
                {fmtBlendedUpside(s)}
              </TD>
            </tr>
          </tbody>
        </Table>
      </Card>

      <Card title="Snapshot" className="lg:col-span-3">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
          <Stat label="Market cap" value={fmtBig(m.market_cap, cur)} />
          <Stat
            label={beta.label}
            value={beta.value}
            sub={beta.sub}
            title={beta.title}
          />
          <Stat
            label="52-week range"
            value={
              <span className="text-sm">
                {fmtMoney(m.fifty_two_week_low, cur, 0)} –{" "}
                {fmtMoney(m.fifty_two_week_high, cur, 0)}
              </span>
            }
          />
          <Stat
            label="Shares out"
            value={fmtCount(m.shares_outstanding)}
          />
          <Stat
            label="Dividend / sh"
            value={fmtMoney(m.dividend_per_share, cur)}
          />
          <Stat
            label="Sector"
            value={<span className="text-sm">{m.sector || "—"}</span>}
            sub={m.industry || undefined}
          />
        </div>
        {report.warnings.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-2 border-t border-line pt-3">
            {report.warnings.map((w, i) => (
              <Badge key={i} tone="neutral">
                {w}
              </Badge>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
