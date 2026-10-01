"use client";

// Trading comps panel: editable peer set, peer multiples table, distribution
// stats, and implied values vs. the target. Built from the shared primitives
// and follows ValuationSummary's structure/density.

import React, { useEffect, useRef, useState } from "react";
import type { Assumptions, CompRow, Report, StatRow } from "@/lib/types";
import { fmtBig, fmtCount, fmtMoney, fmtMult, fmtPct, toneForUpside } from "@/lib/format";
import {
  Badge,
  Button,
  Card,
  EmptyState,
  Stat,
  Table,
  TD,
  TH,
} from "@/components/ui";

// Multiple keys + display labels, in render order.
const MULTIPLES: { key: keyof Pick<CompRow, "ev_ebitda" | "ev_sales" | "pe" | "pb" | "peg">; label: string }[] = [
  { key: "ev_ebitda", label: "EV/EBITDA" },
  { key: "ev_sales", label: "EV/Sales" },
  { key: "pe", label: "P/E" },
  { key: "pb", label: "P/B" },
  { key: "peg", label: "PEG" },
];

function compAmount(value: number | null, currency?: string | null): string {
  return currency ? fmtBig(value, currency) : `${fmtCount(value)} (currency unknown)`;
}

export default function CompsPanel({
  report,
  assumptions,
  setAssumptions,
  onRecompute,
  recomputing,
}: {
  report: Report;
  assumptions: Assumptions;
  setAssumptions: (a: Assumptions) => void;
  onRecompute: (override?: Assumptions) => void;
  recomputing: boolean;
}) {
  const cur = report.summary.currency;
  const comps = report.comps;
  const price = report.current_price;

  const [localPeers, setLocalPeers] = useState<string>(assumptions.peers ?? "");
  const peersInputRef = useRef<HTMLInputElement | null>(null);

  // Keep the local input in sync when the assumptions change upstream
  // (e.g. FMP auto-peering) — but never clobber what the user is typing.
  useEffect(() => {
    if (document.activeElement === peersInputRef.current) return;
    setLocalPeers(assumptions.peers ?? "");
  }, [assumptions.peers]);

  const applyPeers = () => {
    const next: Assumptions = { ...assumptions, peers: localPeers };
    setAssumptions(next);
    onRecompute(next);
  };

  const hasPeers = comps != null && (comps.peers?.length ?? 0) > 0;

  return (
    <Card title="Trading comps" subtitle="Peer multiples vs. the target">
      {/* --- Peer editor --------------------------------------------------- */}
      <div className="flex flex-wrap items-center gap-2">
        <input
          ref={peersInputRef}
          type="text"
          value={localPeers}
          onChange={(e) => setLocalPeers(e.target.value)}
          placeholder="MSFT, GOOGL, AAPL"
          className="min-w-[16rem] flex-1 rounded-lg border border-line bg-surface px-3 py-1.5 text-sm text-ink placeholder:text-ink-faint"
        />
        <Button onClick={applyPeers} disabled={recomputing}>
          Update peers
        </Button>
      </div>
      <p className="mt-1 text-[11px] text-ink-faint">
        Comma-separated tickers (e.g. MSFT, GOOGL). Leave blank to auto-suggest.
      </p>

      {!hasPeers || comps == null ? (
        <div className="mt-4 space-y-3">
          <EmptyState
            title="No peer set yet"
            hint="Add comma-separated peer tickers above. (When FMP is connected, peers are auto-filled.)"
          />
          {(comps?.notes?.length ?? 0) > 0 && (
            <div className="space-y-1">
              {comps?.notes?.map((n, i) => (
                <p key={i} className="text-xs text-ink-faint">
                  {n}
                </p>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div className="mt-4 space-y-5">
          {/* --- Comps table + stat rows ----------------------------------- */}
          <Table>
            <thead>
              <tr>
                <TH align="left">Ticker</TH>
                <TH align="left">Name</TH>
                <TH>Mkt cap</TH>
                <TH>EV</TH>
                <TH>EV/EBITDA</TH>
                <TH>EV/Sales</TH>
                <TH>P/E</TH>
                <TH>P/B</TH>
                <TH><span title="Informational only: peer growth horizons may differ; PEG does not set an implied price.">PEG (ref.)</span></TH>
              </tr>
            </thead>
            <tbody>
              {/* Target row (highlighted) */}
              {comps.target && (
                <tr>
                  <TD align="left" className="bg-surface-hi font-semibold">
                    <span className="inline-flex items-center gap-1.5">
                      {comps.target.ticker || "—"}
                      <Badge tone="brand">you</Badge>
                    </span>
                  </TD>
                  <TD align="left" className="max-w-[12rem] truncate bg-surface-hi">
                    {comps.target.name || "—"}
                  </TD>
                  <TD num className="bg-surface-hi">
                    {compAmount(comps.target.market_cap, comps.target.currency)}
                  </TD>
                  <TD num className="bg-surface-hi">
                    {compAmount(comps.target.enterprise_value, comps.target.currency)}
                  </TD>
                  <TD num className="bg-surface-hi">
                    {fmtMult(comps.target.ev_ebitda)}
                  </TD>
                  <TD num className="bg-surface-hi">
                    {fmtMult(comps.target.ev_sales)}
                  </TD>
                  <TD num className="bg-surface-hi">
                    {fmtMult(comps.target.pe)}
                  </TD>
                  <TD num className="bg-surface-hi">
                    {fmtMult(comps.target.pb)}
                  </TD>
                  <TD num className="bg-surface-hi">
                    {fmtMult(comps.target.peg)}
                  </TD>
                </tr>
              )}

              {/* Peer rows */}
              {(comps.peers ?? []).map((p: CompRow, i: number) => (
                <tr key={p.ticker || i}>
                  <TD align="left">{p.ticker || "—"}</TD>
                  <TD align="left" className="max-w-[12rem] truncate">
                    {p.name || "—"}
                  </TD>
                  <TD num>{compAmount(p.market_cap, p.currency)}</TD>
                  <TD num>{compAmount(p.enterprise_value, p.currency)}</TD>
                  <TD num>{fmtMult(p.ev_ebitda)}</TD>
                  <TD num>{fmtMult(p.ev_sales)}</TD>
                  <TD num>{fmtMult(p.pe)}</TD>
                  <TD num>{fmtMult(p.pb)}</TD>
                  <TD num>{fmtMult(p.peg)}</TD>
                </tr>
              ))}

              {/* Distribution stats */}
              {(["median", "p25", "p75", "min", "max"] as const).map(
                (statKey, rIdx) => {
                  const rowLabel: Record<typeof statKey, string> = {
                    median: "Peer median",
                    p25: "Peer 25th pct",
                    p75: "Peer 75th pct",
                    min: "Peer min",
                    max: "Peer max",
                  };
                  const statOf = (mKey: string): number | null => {
                    const row: StatRow | undefined = comps.stats?.[mKey];
                    const v = row?.[statKey];
                    return v == null ? null : v;
                  };
                  return (
                    <tr
                      key={statKey}
                      className={rIdx === 0 ? "border-t-2 border-line" : undefined}
                    >
                      <TD align="left" className="text-ink-dim">
                        {rowLabel[statKey]}
                      </TD>
                      <TD align="left" className="text-ink-faint">
                        —
                      </TD>
                      <TD num className="text-ink-faint">
                        —
                      </TD>
                      <TD num className="text-ink-faint">
                        —
                      </TD>
                      <TD num className="text-ink-dim">
                        {fmtMult(statOf("ev_ebitda"))}
                      </TD>
                      <TD num className="text-ink-dim">
                        {fmtMult(statOf("ev_sales"))}
                      </TD>
                      <TD num className="text-ink-dim">
                        {fmtMult(statOf("pe"))}
                      </TD>
                      <TD num className="text-ink-dim">
                        {fmtMult(statOf("pb"))}
                      </TD>
                      <TD num className="text-ink-dim">
                        {fmtMult(statOf("peg"))}
                      </TD>
                    </tr>
                  );
                }
              )}
            </tbody>
          </Table>

          {/* --- Implied values -------------------------------------------- */}
          <div>
            <div className="mb-2 text-[11px] uppercase tracking-wider text-ink-faint">
              Implied value
            </div>
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
              {MULTIPLES.map(({ key, label }) => {
                const implied = comps.implied?.[key];
                if (implied == null) return null;
                const up = price ? implied / price - 1 : null;
                return (
                  <Stat
                    key={key}
                    label={label}
                    value={fmtMoney(implied, cur)}
                    tone={toneForUpside(up)}
                    sub={fmtPct(up, { signed: true })}
                  />
                );
              })}
              <Stat
                label="Comps blend (median)"
                value={fmtMoney(comps.implied_price_summary?.median, cur)}
                tone={toneForUpside(
                  price && comps.implied_price_summary?.median != null
                    ? comps.implied_price_summary.median / price - 1
                    : null
                )}
                sub={
                  <>
                    {fmtMoney(comps.implied_price_summary?.low, cur)} –{" "}
                    {fmtMoney(comps.implied_price_summary?.high, cur)}
                  </>
                }
              />
            </div>
          </div>

          {/* --- Notes ----------------------------------------------------- */}
          {(comps.notes?.length ?? 0) > 0 && (
            <div className="space-y-1 border-t border-line pt-3">
              {comps.notes.map((n, i) => (
                <p key={i} className="text-xs text-ink-faint">
                  {n}
                </p>
              ))}
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
