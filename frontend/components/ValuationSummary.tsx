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
  priceDecimals,
  toneForBlendedUpside,
  toneForMethodUpside,
  uiWarnings,
} from "@/lib/format";
import { Badge, Card, Stat, Table, TD, TH } from "@/components/ui";

// Room kept right of the plot for the high labels (mr-16, 4rem): a bar that
// ends at the right edge still has its label inside the card at phone widths.
function FootballField({ report }: { report: Report }) {
  const rows = report.football_field || [];
  const price = report.current_price;
  const hasPrice = typeof price === "number" && Number.isFinite(price) && price > 0;
  const cur = report.summary.currency;
  // Cents unless prices run into the thousands (a sub-$20 stock's labels
  // would otherwise round to the same few whole units).
  const dp = priceDecimals(hasPrice ? price : rows[0]?.base);
  if (rows.length === 0) {
    return <p className="text-sm text-ink-faint">No valuation ranges available.</p>;
  }

  const lows = rows.map((r) => r.low);
  const highs = rows.map((r) => r.high);
  let min = Math.min(...lows, ...(hasPrice ? [price] : []));
  let max = Math.max(...highs, ...(hasPrice ? [price] : []));
  const pad = (max - min) * 0.06 || 1;
  min -= pad;
  max += pad;
  const span = max - min || 1;
  const pos = (v: number) => ((v - min) / span) * 100;

  // The current price: a line through every row (each row draws its own
  // segment; the rows touch, so the segments join up) and, in its own row
  // under the bars, a label beside the line, on the side with more room.
  const pricePos = hasPrice ? pos(price) : null;
  const priceLine = pricePos != null && (
    <div
      aria-hidden="true"
      className="absolute inset-y-0 w-[2px] -translate-x-1/2 bg-brand/70"
      style={{ left: `${pricePos}%` }}
    />
  );

  return (
    <div>
      {rows.map((r) => {
        const left = pos(r.low);
        const width = Math.max(pos(r.high) - left, 0.8);
        // Grey for a reference-only method or when there is no verdict.
        const tone = footballFieldTone(report.summary, r.method, r.base, price);
        return (
          <div key={r.method} className="flex items-center gap-3">
            <div
              className="w-32 shrink-0 truncate text-right text-xs text-ink-dim"
              title={r.method}
            >
              {r.method}
            </div>
            <div className="relative mr-16 h-10 flex-1">
              {priceLine}
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
              {/* On the card's background, so the price line passes behind it. */}
              <div
                className="num absolute top-1/2 -translate-y-1/2 whitespace-nowrap rounded-sm bg-surface px-0.5 text-[10px] text-ink-faint"
                style={{ left: `calc(${pos(r.high)}% + 4px)` }}
              >
                {fmtMoney(r.high, cur, dp)}
              </div>
            </div>
          </div>
        );
      })}

      {pricePos != null && (
        <div className="flex items-center gap-3">
          <div className="w-32 shrink-0" />
          <div className="relative mr-16 h-5 flex-1">
            {priceLine}
            <div
              className="num absolute top-1 whitespace-nowrap text-[10px] text-brand"
              style={
                pricePos > 50
                  ? { right: `calc(${100 - pricePos}% + 5px)` }
                  : { left: `calc(${pricePos}% + 5px)` }
              }
            >
              price {fmtMoney(price, cur, dp)}
            </div>
          </div>
        </div>
      )}
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
  // Engine warnings, with CLI phrasing pointed at the dashboard's controls.
  const warnings = uiWarnings(report.warnings);

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
            {/* No target -> "n/a" or "No target" (the reason, supply peers,
                is in its tooltip and the header's verdict explainer, so the
                cell stays narrow); a withheld verdict -> the upside reads
                "n/a", uncoloured. */}
            <tr className="font-semibold">
              <TD align="left">Blended target</TD>
              <TD num>
                <span title={fmtBlendedTarget(s, report.comps)}>
                  {fmtBlendedTarget(s, report.comps, { short: true })}
                </span>
              </TD>
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
                {fmtMoney(m.fifty_two_week_low, cur, priceDecimals(s.current_price))} –{" "}
                {fmtMoney(m.fifty_two_week_high, cur, priceDecimals(s.current_price))}
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
        {warnings.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-2 border-t border-line pt-3">
            {warnings.map((w, i) => (
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
