"use client";

// Multiples panel: valuation multiples from the engine (report.comps.target,
// always available) plus optional FMP enrichment (TTM ratios, analyst price
// targets, and forward estimates). Mirrors ValuationSummary's structure and
// primitives. Everything from the enrichment side is untrusted — FMP field
// names vary by plan — so every lookup is defensive and renders "—" rather
// than throwing.

import React from "react";
import type { CompRow, Enrichment, Report } from "@/lib/types";
import {
  fmtBig,
  fmtCount,
  fmtMoney,
  fmtMult,
  fmtNum,
  fmtPct,
  toneForUpside,
} from "@/lib/format";
import { Card, EmptyState, Stat, Table, TD, TH } from "@/components/ui";

// Defensive numeric lookup against an untrusted Record. Returns null unless the
// value is a finite number, so callers can hand it straight to the fmt* helpers
// (which already render "—" for null).
function getNum(
  obj: Record<string, unknown> | null | undefined,
  key: string
): number | null {
  const v = obj?.[key];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

// Defensive string lookup (used for estimate row dates).
function getStr(
  obj: Record<string, unknown> | null | undefined,
  key: string
): string | null {
  const v = obj?.[key];
  return typeof v === "string" && v.length > 0 ? v : null;
}

export default function MultiplesPanel({
  report,
  enrichment,
}: {
  report: Report;
  enrichment: Enrichment | null;
}) {
  const cur = report.summary.currency;
  const target: CompRow | undefined = report.comps?.target;
  const price = report.current_price;

  // --- FMP TTM ratios: build only the rows that resolve to a number. ------- //
  const ratios = enrichment?.ratios_ttm;
  const keyMetrics = enrichment?.key_metrics_ttm;

  type FmpRow = {
    label: string;
    obj: Record<string, unknown> | null | undefined;
    key: string;
    format: (v: number) => string;
  };

  const fmpRows: FmpRow[] = [
    { label: "P/E", obj: ratios, key: "peRatioTTM", format: (v) => fmtMult(v) },
    {
      label: "P/S",
      obj: ratios,
      key: "priceToSalesRatioTTM",
      format: (v) => fmtMult(v),
    },
    {
      label: "P/B",
      obj: ratios,
      key: "priceToBookRatioTTM",
      format: (v) => fmtMult(v),
    },
    {
      label: "P/FCF",
      obj: ratios,
      key: "priceToFreeCashFlowsRatioTTM",
      format: (v) => fmtMult(v),
    },
    {
      label: "Dividend yield",
      obj: ratios,
      key: "dividendYieldTTM",
      format: (v) => fmtPct(v),
    },
    {
      label: "Payout ratio",
      obj: ratios,
      key: "payoutRatioTTM",
      format: (v) => fmtPct(v),
    },
    {
      label: "ROE",
      obj: ratios,
      key: "returnOnEquityTTM",
      format: (v) => fmtPct(v),
    },
    {
      label: "ROIC",
      obj: keyMetrics,
      key: "roicTTM",
      format: (v) => fmtPct(v),
    },
    {
      label: "Gross margin",
      obj: ratios,
      key: "grossProfitMarginTTM",
      format: (v) => fmtPct(v),
    },
    {
      label: "Operating margin",
      obj: ratios,
      key: "operatingProfitMarginTTM",
      format: (v) => fmtPct(v),
    },
    {
      label: "Net margin",
      obj: ratios,
      key: "netProfitMarginTTM",
      format: (v) => fmtPct(v),
    },
    {
      label: "Debt/Equity",
      obj: ratios,
      key: "debtEquityRatioTTM",
      format: (v) => fmtNum(v, 2),
    },
    {
      label: "Current ratio",
      obj: ratios,
      key: "currentRatioTTM",
      format: (v) => fmtNum(v, 2),
    },
  ];

  const resolvedFmp = fmpRows
    .map((r) => ({ label: r.label, value: getNum(r.obj, r.key), format: r.format }))
    .filter(
      (r): r is { label: string; value: number; format: (v: number) => string } =>
        r.value !== null
    );

  // --- Analyst view -------------------------------------------------------- //
  const pt = enrichment?.price_target;
  const consensus = getNum(pt, "targetConsensus");
  const consensusUpside =
    consensus !== null && Number.isFinite(price) && price !== 0
      ? consensus / price - 1
      : null;
  const estimates = enrichment?.estimates ?? [];
  const hasEstimates = estimates.length > 0;

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      {/* CARD 1 — engine trailing multiples (always present). */}
      <Card
        title="Trailing multiples (target)"
        subtitle="From the valuation engine's comp set"
      >
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
          <Stat label="P/E" value={fmtMult(target?.pe)} />
          <Stat label="EV/EBITDA" value={fmtMult(target?.ev_ebitda)} />
          <Stat label="EV/Sales" value={fmtMult(target?.ev_sales)} />
          <Stat label="P/B" value={fmtMult(target?.pb)} />
          <Stat label="PEG (reference)" value={fmtMult(target?.peg)}
            title="Informational only; mixed peer earnings-growth horizons make PEG unsuitable for deriving the target price." />
          <Stat label="Market cap" value={target?.currency ? fmtBig(target.market_cap, target.currency) : fmtCount(target?.market_cap)}
            sub={target && !target.currency ? "Currency unavailable" : undefined} />
          <Stat
            label="EV"
            value={target?.currency ? fmtBig(target.enterprise_value, target.currency) : fmtCount(target?.enterprise_value)}
            sub={target && !target.currency ? "Currency unavailable" : undefined}
          />
        </div>
        {!report.comps && (
          <p className="mt-3 border-t border-line pt-3 text-xs text-ink-faint">
            Comparable-company analysis is unavailable for this report.
          </p>
        )}
      </Card>

      {/* CARD 2 — FMP TTM ratios (optional). */}
      <Card title="FMP TTM ratios" subtitle="Trailing-twelve-month fundamentals">
        {!enrichment?.enabled ? (
          <EmptyState
            title="Connect FMP"
            hint="Set FMP_API_KEY to load TTM ratios, price targets, and analyst estimates."
          />
        ) : resolvedFmp.length === 0 ? (
          <p className="text-xs text-ink-faint">
            No TTM ratios were returned — your FMP plan may not cover these
            endpoints.
          </p>
        ) : (
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
            {resolvedFmp.map((r) => (
              <Stat key={r.label} label={r.label} value={r.format(r.value)} />
            ))}
          </div>
        )}
      </Card>

      {/* CARD 3 — Analyst view (optional). */}
      {enrichment?.enabled && (
        <Card
          title="Analyst view (FMP)"
          subtitle="Consensus price targets and forward estimates"
          className="lg:col-span-2"
        >
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Stat
              label="Consensus target"
              value={fmtMoney(consensus, cur)}
              tone={toneForUpside(consensusUpside)}
              sub={
                consensusUpside !== null
                  ? `${fmtPct(consensusUpside, { signed: true })} vs price`
                  : undefined
              }
            />
            <Stat label="High" value={fmtMoney(getNum(pt, "targetHigh"), cur)} />
            <Stat label="Low" value={fmtMoney(getNum(pt, "targetLow"), cur)} />
            <Stat
              label="Median"
              value={fmtMoney(getNum(pt, "targetMedian"), cur)}
            />
          </div>

          {hasEstimates && (
            <div className="mt-4 border-t border-line pt-3">
              <Table>
                <thead>
                  <tr>
                    <TH align="left">Period</TH>
                    <TH>Est. revenue</TH>
                    <TH>Est. EPS</TH>
                  </tr>
                </thead>
                <tbody>
                  {estimates.map((row, i) => {
                    const date = getStr(row, "date");
                    const label = date ? date.slice(0, 4) : "—";
                    return (
                      <tr key={`${date ?? "row"}-${i}`}>
                        <TD align="left">{label}</TD>
                        <TD num>
                          {fmtBig(getNum(row, "estimatedRevenueAvg"), cur)}
                        </TD>
                        <TD num>
                          {fmtMoney(getNum(row, "estimatedEpsAvg"), cur)}
                        </TD>
                      </tr>
                    );
                  })}
                </tbody>
              </Table>
            </div>
          )}
        </Card>
      )}
    </div>
  );
}
