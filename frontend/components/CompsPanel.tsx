"use client";

// Trading comps panel: editable peer set, peer multiples table, distribution
// stats, and implied values vs. the target. Built from the shared primitives
// and follows ValuationSummary's structure/density.
//
// "Suggest peers" fills the peer box from Yahoo's industry list (GET
// /api/peers) for review; nothing is applied until "Update peers". Each
// requested peer is tagged with what comps did with it (comps.peer_status),
// the multiple headers carry their sample size, and the FY/TTM toggle picks
// the target figures the peer multiples apply to (comps_basis).

import React, { useEffect, useRef, useState } from "react";
import type { Assumptions, CompRow, CompsBasis, Report, StatRow } from "@/lib/types";
import { fetchPeerSuggestions } from "@/lib/api";
import { normalizePeers } from "@/lib/assumptions";
import {
  compsWarnings,
  fmtBig,
  fmtCount,
  fmtMoney,
  fmtMult,
  fmtPct,
  peerStatusReason,
  peerStatusTone,
  toneForUpside,
  uiText,
} from "@/lib/format";
import {
  Badge,
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

// Quartiles of fewer peers than this are not shown: with two or three values
// they only interpolate between the extremes.
const MIN_PEERS_FOR_QUARTILES = 4;

// A multiple's column header with its sample size ("P/E · n=2"; the n stays
// lower case in the upper-case header row).
function withN(label: string, row: StatRow | undefined): React.ReactNode {
  if (typeof row?.n !== "number") return label;
  return (
    <>
      {label}
      <span className="font-normal normal-case">{` · n=${row.n}`}</span>
    </>
  );
}

export default function CompsPanel({
  report,
  assumptions,
  setAssumptions,
  onRecompute,
  recomputing,
  inputRef,
}: {
  report: Report;
  assumptions: Assumptions;
  setAssumptions: (a: Assumptions) => void;
  onRecompute: (override?: Assumptions) => void;
  recomputing: boolean;
  // The peer input, for the page's "Add peers" action to focus.
  inputRef?: React.RefObject<HTMLInputElement | null>;
}) {
  const cur = report.summary.currency;
  const comps = report.comps;
  const price = report.current_price;
  const ticker = report.summary.ticker || report.company?.ticker || "";

  const [localPeers, setLocalPeers] = useState<string>(assumptions.peers ?? "");
  const ownInputRef = useRef<HTMLInputElement | null>(null);
  const peersInputRef = inputRef ?? ownInputRef;
  // "Suggest peers": the request in flight, and its notes (or error).
  const [suggesting, setSuggesting] = useState(false);
  const [suggestion, setSuggestion] = useState<{ notes: string[]; count: number } | null>(null);
  const [suggestError, setSuggestError] = useState<string | null>(null);
  const tickerRef = useRef(ticker);
  useEffect(() => {
    tickerRef.current = ticker;
    setSuggestion(null);
    setSuggestError(null);
  }, [ticker]);

  // Keep the local input in sync when the assumptions change upstream
  // (e.g. FMP auto-peering) — but never clobber what the user is typing.
  useEffect(() => {
    if (document.activeElement === peersInputRef.current) return;
    setLocalPeers(assumptions.peers ?? "");
  }, [assumptions.peers]);

  // The suggestion's notes describe the box as Suggest peers filled it; once
  // a peer set is applied they no longer do, so they go.
  const applyPeers = () => {
    const next: Assumptions = { ...assumptions, peers: localPeers };
    setSuggestion(null);
    setSuggestError(null);
    setAssumptions(next);
    onRecompute(next);
  };

  // Fills the peer box with suggestions for review; never applies them.
  const suggestPeers = async () => {
    const t = ticker;
    if (!t || suggesting) return;
    setSuggesting(true);
    setSuggestError(null);
    try {
      const res = await fetchPeerSuggestions(t);
      if (tickerRef.current !== t) return;
      const names = normalizePeers(res.peers.map((p) => p.ticker).join(","));
      if (names.length > 0) setLocalPeers(names.join(", "));
      setSuggestion({ notes: res.notes, count: names.length });
    } catch (e) {
      if (tickerRef.current === t) {
        setSuggestError(`Couldn't suggest peers: ${e instanceof Error ? e.message : String(e)}`);
      }
    } finally {
      if (tickerRef.current === t) setSuggesting(false);
    }
  };

  // The comps basis in the draft (else the one the report used).
  const basis: CompsBasis =
    assumptions.comps_basis ?? report.assumptions_used?.comps_basis ?? "fy";
  const setBasis = (v: CompsBasis) => {
    if (v === basis) return;
    const next: Assumptions = { ...assumptions, comps_basis: v };
    setAssumptions(next);
    onRecompute(next);
  };

  // The peer box differs from the peer set the report used.
  const applied = report.assumptions_used ? report.assumptions_used.peers : assumptions.peers;
  const appliedKey = normalizePeers(applied ?? "").join(",");
  const peersPending = normalizePeers(localPeers).join(",") !== appliedKey;
  // The same when the applied peer set changes some other way (FMP's peers
  // filled in on load, a reset, a restored research state).
  const appliedSeen = useRef(appliedKey);
  useEffect(() => {
    if (appliedSeen.current === appliedKey) return;
    appliedSeen.current = appliedKey;
    setSuggestion(null);
    setSuggestError(null);
  }, [appliedKey]);

  // What comps did with each requested peer, in request order.
  const peerStatus = Object.entries(comps?.peer_status ?? {});
  const requested = peerStatus.length > 0 || normalizePeers(applied ?? "").length > 0;
  // The comps notes, then the comps warnings that do not repeat one (why no
  // requested peer was usable, a failed fetch); without a requested peer the
  // empty state already says there is none, so only a failure is added.
  const messages = [
    ...(comps?.notes ?? []).map(uiText),
    ...compsWarnings(report.warnings, comps?.notes).filter(
      (w) => requested || w.startsWith("Comps failed")
    ),
  ];
  const notesBlock = messages.length > 0 && (
    <div className="space-y-1 border-t border-line pt-3">
      {messages.map((n, i) => (
        <p key={i} className="text-xs text-ink-faint">
          {n}
        </p>
      ))}
    </div>
  );

  const hasPeers = comps != null && (comps.peers?.length ?? 0) > 0;
  // Peers behind each multiple; quartiles need MIN_PEERS_FOR_QUARTILES.
  const sizes = MULTIPLES.map(({ key }) => comps?.stats?.[key]?.n).filter(
    (n): n is number => typeof n === "number"
  );
  const fewPeers = sizes.length > 0 && Math.max(...sizes) < MIN_PEERS_FOR_QUARTILES;
  const statRows = (["median", "p25", "p75", "min", "max"] as const).filter(
    (k) => !(fewPeers && (k === "p25" || k === "p75"))
  );

  return (
    <Card
      title="Trading comps"
      subtitle="Peer multiples vs. the target"
      right={
        <Segmented<CompsBasis>
          label="Comps basis"
          value={basis}
          options={[
            { label: "FY", value: "fy" },
            { label: "TTM", value: "ttm" },
          ]}
          onChange={setBasis}
        />
      }
    >
      {/* --- Peer editor --------------------------------------------------- */}
      <div className="flex flex-wrap items-center gap-2">
        <input
          ref={peersInputRef}
          type="text"
          aria-label="Peer tickers"
          value={localPeers}
          onChange={(e) => setLocalPeers(e.target.value)}
          placeholder="MSFT, GOOGL, AAPL"
          className="min-w-[16rem] flex-1 rounded-lg border border-line bg-surface px-3 py-1.5 text-sm text-ink placeholder:text-ink-faint"
        />
        <Button onClick={applyPeers} disabled={recomputing}>
          Update peers
        </Button>
        <Button variant="ghost" onClick={suggestPeers} disabled={suggesting || !ticker}>
          {suggesting ? (
            <>
              <Spinner className="h-3 w-3" /> Suggesting
            </>
          ) : (
            "Suggest peers"
          )}
        </Button>
      </div>
      {peersPending && (
        <p className="mt-1 text-[11px] text-flat">
          Not applied: press Update peers to value with this peer set.
        </p>
      )}
      <p className="mt-1 text-[11px] text-ink-faint">
        Comma-separated tickers (e.g. MSFT, GOOGL); leave blank for no comps.
        Suggest peers fills the box from Yahoo&apos;s industry list for you to
        review, and nothing is applied until you press Update peers (FMP, when
        connected, fills them once on load). FY applies the peer multiples to
        the target&apos;s latest fiscal year, TTM to its trailing twelve months.
      </p>
      {suggestError && <p className="mt-1 text-xs text-down">{suggestError}</p>}
      {suggestion && (
        <div className="mt-2 space-y-0.5 rounded-lg border border-line bg-surface-raised/60 px-3 py-2">
          {suggestion.count === 0 && (
            <p className="text-xs text-ink-dim">No peers suggested.</p>
          )}
          {suggestion.notes.map((n, i) => (
            <p key={i} className="text-[11px] text-ink-faint">
              {uiText(n)}
            </p>
          ))}
        </div>
      )}

      {/* --- What comps did with each requested peer ----------------------- */}
      {peerStatus.length > 0 && (
        <div className="mt-2 flex flex-wrap items-center gap-1.5" role="list" aria-label="Peer status">
          {peerStatus.map(([t, status]) => (
            <span role="listitem" key={t}>
              <Badge tone={peerStatusTone(status)} title={peerStatusReason(status)}>
                {t}
                {status !== "used" && (
                  <span className="ml-1 font-normal">· {status}</span>
                )}
              </Badge>
            </span>
          ))}
        </div>
      )}

      {!hasPeers || comps == null ? (
        <div className="mt-4 space-y-3">
          <EmptyState
            title={requested ? "No usable peers" : "No peer set yet"}
            hint={
              requested
                ? "None of the requested peers gave a usable multiple; see each peer's status above."
                : "Add comma-separated peer tickers above, or press Suggest peers."
            }
          />
          {notesBlock}
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
                <TH>{withN("EV/EBITDA", comps.stats?.ev_ebitda)}</TH>
                <TH>{withN("EV/Sales", comps.stats?.ev_sales)}</TH>
                <TH>{withN("P/E", comps.stats?.pe)}</TH>
                <TH>{withN("P/B", comps.stats?.pb)}</TH>
                <TH><span title="Informational only: peer growth horizons may differ; PEG does not set an implied price.">{withN("PEG (ref.)", comps.stats?.peg)}</span></TH>
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
              {/* Quartile rows only with enough peers (fewPeers). */}
              {statRows.map(
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
                    // A quartile of too few peers for this multiple.
                    if ((statKey === "p25" || statKey === "p75") &&
                        typeof row?.n === "number" && row.n < MIN_PEERS_FOR_QUARTILES) {
                      return null;
                    }
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
          {notesBlock}
        </div>
      )}
    </Card>
  );
}
