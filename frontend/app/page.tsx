"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import type {
  AssumptionsUsed,
  Assumptions,
  AssumptionSuggestion,
  Digest,
  DigestEntry,
  Enrichment,
  Report,
  ResearchNote,
  WatchlistItem,
} from "@/lib/types";
import {
  apiUrl,
  downloadExport,
  errorDetail,
  fetchEnrichment,
  fetchResearchState,
  fetchValuation,
  fetchWatchlist,
  postResearchNote,
  saveResearchState,
  updateWatchlist,
  type ExportKind,
} from "@/lib/api";
import {
  fmtBlendedTarget,
  fmtBlendedUpside,
  fmtDate,
  fmtMoney,
  fmtPct,
  toneForBlendedUpside,
  toneForRecommendation,
  toneForUpside,
  watchlistUpside,
} from "@/lib/format";
import { NoteOrder, chainSave, settledWithin } from "@/lib/sync";
import { Badge, Button, Spinner, cx } from "@/components/ui";

import ValuationSummary from "@/components/ValuationSummary";
import DCFPanel from "@/components/DCFPanel";
import CompsPanel from "@/components/CompsPanel";
import DDMFCFEPanel from "@/components/DDMFCFEPanel";
import SensitivityPanel from "@/components/SensitivityPanel";
import MultiplesPanel from "@/components/MultiplesPanel";
import FinancialsPanel from "@/components/FinancialsPanel";
import FilingsPanel from "@/components/FilingsPanel";
import NewsPanel from "@/components/NewsPanel";
import AIResearchPanel from "@/components/AIResearchPanel";

const TABS = [
  "Overview",
  "Valuation",
  "Financials",
  "Filings",
  "Multiples",
  "News",
  "AI Research",
] as const;
type Tab = (typeof TABS)[number];

function assumptionsFromUsed(used?: AssumptionsUsed): Assumptions {
  if (!used) return {};
  return {
    rf: used.rf,
    erp: used.erp,
    tax_rate: used.tax_rate ?? undefined,
    forecast_years: used.forecast_years,
    terminal_growth: used.terminal_growth,
    terminal_method: used.terminal_method,
    exit_ev_ebitda: used.exit_ev_ebitda ?? undefined,
    target_ebit_margin: used.target_ebit_margin ?? undefined,
    revenue_growth_y1: used.revenue_growth_y1 ?? undefined,
    peers: used.peers ?? undefined,
  };
}

// Longest a ticker load waits for earlier saves to land before re-reading
// saved research (a local save takes milliseconds; this only bounds a save
// that never answers).
const SAVE_WAIT_MS = 10_000;

const FIELD_TO_KEY: Record<AssumptionSuggestion["field"], keyof Assumptions> = {
  terminal_growth: "terminal_growth",
  forecast_years: "forecast_years",
  target_ebit_margin: "target_ebit_margin",
  risk_free_rate: "rf",
  equity_risk_premium: "erp",
  tax_rate: "tax_rate",
  exit_ev_ebitda: "exit_ev_ebitda",
  revenue_growth_y1: "revenue_growth_y1",
};

const EXPORTS: { kind: ExportKind; label: string }[] = [
  { kind: "excel", label: "Excel model" },
  { kind: "memo", label: "Word memo" },
  { kind: "deck", label: "Deck" },
  { kind: "html", label: "HTML report" },
];

export default function Home() {
  const [input, setInput] = useState("");
  const [ticker, setTicker] = useState("");
  const [report, setReport] = useState<Report | null>(null);
  const [enrichment, setEnrichment] = useState<Enrichment | null>(null);
  const [assumptions, setAssumptions] = useState<Assumptions>({});
  const [tab, setTab] = useState<Tab>("Overview");
  const [loading, setLoading] = useState(false);
  const [recomputing, setRecomputing] = useState(false);
  const [enrichLoading, setEnrichLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [researchNotes, setResearchNotes] = useState("");
  const [digests, setDigests] = useState<DigestEntry[]>([]);
  const [note, setNote] = useState<ResearchNote | null>(null);
  const [noteLoading, setNoteLoading] = useState(false);
  const [watchlist, setWatchlist] = useState<WatchlistItem[]>([]);
  const [exporting, setExporting] = useState<ExportKind | null>(null);
  const [status, setStatus] = useState<{
    fmp_enabled: boolean;
    anthropic_enabled: boolean;
  } | null>(null);
  const [showKeys, setShowKeys] = useState(false);
  const [keyAnthropic, setKeyAnthropic] = useState("");
  const [keyFmp, setKeyFmp] = useState("");
  const [savingKeys, setSavingKeys] = useState(false);
  // Ticker whose saved research couldn't be read (autosave is paused for it).
  const [restoreFailedFor, setRestoreFailedFor] = useState("");

  const autoPeeredFor = useRef<string>("");
  // Autosave only runs for this ticker: set once its saved research has been
  // restored and its valuation has loaded, so a switch can never write the
  // previous ticker's (or a blank) state over the new ticker's saved research.
  const stateLoadedFor = useRef<string>("");
  // Set once the current load has finished reading this ticker's saved
  // research (whether or not the read succeeded).
  const restoreDoneFor = useRef<string>("");
  // A research note that finished for a ticker whose saved state was still
  // being re-read; the restore applies it instead of the older saved note.
  const lateNote = useRef<{ ticker: string; note: ResearchNote } | null>(null);
  // The debounced autosave waiting to run, so a ticker switch can flush it
  // instead of dropping the last edits, and every save sent so far (chained,
  // see chainSave), so a reload reads the saved state only after all of them
  // have landed.
  const pendingSave = useRef<(() => void) | null>(null);
  const lastSave = useRef<Promise<void> | null>(null);
  // Per-ticker order of note requests, so an older note that finishes after
  // a newer one never replaces it.
  const noteOrder = useRef(new NoteOrder());
  const loadSeq = useRef(0); // guards against a stale load finishing late
  const recomputeSeq = useRef(0); // only the newest recompute may land

  useEffect(() => {
    // Retry with backoff — in the desktop app the backend can still be
    // finishing its boot when the page mounts; one failed probe must not
    // permanently disable the AI/FMP features.
    let cancelled = false;
    const probe = (attempt: number) => {
      fetch(apiUrl("/api/health"))
        .then((r) => r.json())
        .then((s) => {
          if (!cancelled) setStatus(s);
        })
        .catch(() => {
          if (!cancelled && attempt < 6)
            setTimeout(() => probe(attempt + 1), 1000 * (attempt + 1));
        });
    };
    probe(0);
    fetchWatchlist().then(setWatchlist).catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  // Fresh watchlist for effects that fire from stale closures.
  const watchlistRef = useRef<WatchlistItem[]>(watchlist);
  useEffect(() => {
    watchlistRef.current = watchlist;
  }, [watchlist]);
  // Set synchronously by loadTicker (not only after render) so async work
  // started for the previous ticker can tell it has been superseded.
  const tickerRef = useRef("");
  useEffect(() => {
    tickerRef.current = ticker;
  }, [ticker]);

  const loadTicker = useCallback(async (sym: string) => {
    const t = sym.trim().toUpperCase();
    if (!t) return;
    const seq = ++loadSeq.current;
    // Write any debounced edits of the ticker on screen before its state is
    // cleared (and before a reload of the same ticker re-reads it).
    pendingSave.current?.();
    const flush = settledWithin(lastSave.current, SAVE_WAIT_MS);
    setLoading(true);
    setError(null);
    setReport(null);
    setEnrichment(null);
    setAssumptions({});
    setResearchNotes("");
    setDigests([]);
    setNote(null);
    setNoteLoading(false);
    setRecomputing(false);
    setRestoreFailedFor("");
    autoPeeredFor.current = "";
    stateLoadedFor.current = "";
    restoreDoneFor.current = "";
    if (lateNote.current?.ticker !== t) lateNote.current = null;
    tickerRef.current = t;
    setTicker(t);
    await flush;
    if (seq !== loadSeq.current) return;

    // Restore persisted research FIRST so saved assumptions drive the first
    // valuation (and saved notes/digests/note come back with it). A never-seen
    // ticker restores as {}; a failed read throws.
    let savedAssumptions: Assumptions = {};
    let restored = false;
    try {
      const st = await fetchResearchState(t);
      if (seq !== loadSeq.current) return; // a newer load superseded us
      setResearchNotes(st.notes ?? "");
      setDigests(st.digests ?? []);
      setNote(st.note ?? null);
      if (st.assumptions && Object.keys(st.assumptions).length > 0) {
        savedAssumptions = st.assumptions;
      }
      setAssumptions(savedAssumptions);
      restored = true;
    } catch {
      if (seq !== loadSeq.current) return;
      // Value the ticker anyway, but keep autosave off so the blank state
      // can't overwrite research we merely failed to read.
      setRestoreFailedFor(t);
    }
    // A note that finished while the saved state was being read is newer
    // than the saved one.
    restoreDoneFor.current = t;
    const late = lateNote.current;
    lateNote.current = null;
    if (late?.ticker === t) setNote(late.note);

    try {
      const rep = await fetchValuation(t, savedAssumptions);
      if (seq !== loadSeq.current) return;
      setReport(rep);
      setAssumptions(assumptionsFromUsed(rep.assumptions_used));
      setTab("Overview");
      if (restored) stateLoadedFor.current = t;
    } catch (e) {
      if (seq !== loadSeq.current) return;
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (seq === loadSeq.current) setLoading(false);
    }
    setEnrichLoading(true);
    fetchEnrichment(t)
      .then((en) => {
        if (seq === loadSeq.current) setEnrichment(en);
      })
      .catch(() => {
        if (seq === loadSeq.current) setEnrichment({ enabled: false });
      })
      .finally(() => {
        if (seq === loadSeq.current) setEnrichLoading(false);
      });
  }, []);

  const recompute = useCallback(
    async (override?: Assumptions) => {
      if (!ticker || ticker !== tickerRef.current) return;
      const a = override ?? assumptions;
      const seq = loadSeq.current;
      const rseq = ++recomputeSeq.current;
      // Drop the result if another ticker was loaded or a newer recompute
      // started meanwhile — it would overwrite the newer report.
      const current = () =>
        seq === loadSeq.current && rseq === recomputeSeq.current;
      setRecomputing(true);
      setError(null);
      try {
        const rep = await fetchValuation(ticker, a);
        if (current()) setReport(rep);
      } catch (e) {
        if (current()) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (current()) setRecomputing(false);
      }
    },
    [ticker, assumptions]
  );

  // Auto-feed FMP peers once when the engine found none.
  useEffect(() => {
    if (!report || !enrichment?.enabled || !ticker) return;
    if (autoPeeredFor.current === ticker) return;
    const hasPeers = (report.comps?.peers?.length ?? 0) > 0;
    const fmpPeers = (enrichment.peers || []).filter((p) => p && p !== ticker);
    if (!hasPeers && fmpPeers.length > 0 && !assumptions.peers) {
      autoPeeredFor.current = ticker;
      const next = { ...assumptions, peers: fmpPeers.slice(0, 8).join(",") };
      setAssumptions(next);
      recompute(next);
    }
  }, [report, enrichment, ticker, assumptions, recompute]);

  // Autosave research state (debounced), only once this ticker's saved state
  // was restored and its valuation loaded (see stateLoadedFor).
  useEffect(() => {
    if (!ticker || stateLoadedFor.current !== ticker) return;
    const save = () => {
      clearTimeout(id);
      if (pendingSave.current === save) pendingSave.current = null;
      lastSave.current = chainSave(
        lastSave.current,
        saveResearchState(ticker, {
          notes: researchNotes,
          digests,
          note,
          assumptions,
        })
      );
    };
    const id = setTimeout(save, 1500);
    pendingSave.current = save;
    return () => {
      clearTimeout(id);
      if (pendingSave.current === save) pendingSave.current = null;
    };
  }, [ticker, researchNotes, digests, note, assumptions]);

  // Keep the watchlist snapshot fresh whenever a watched ticker reloads.
  // Reads through watchlistRef so a just-removed ticker isn't re-added by a
  // recompute that was already in flight.
  useEffect(() => {
    if (!report || !ticker) return;
    if (!watchlistRef.current.some((w) => w.ticker === ticker)) return;
    const s = report.summary;
    updateWatchlist("add", ticker, {
      name: s.name,
      currency: s.currency,
      price: s.current_price,
      blended_target: s.blended_target,
      recommendation: s.recommendation,
    })
      .then((wl) => {
        if (watchlistRef.current.some((w) => w.ticker === ticker))
          setWatchlist(wl);
      })
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [report]);

  const applySuggestion = useCallback(
    (sug: AssumptionSuggestion) => {
      const key = FIELD_TO_KEY[sug.field];
      const next: Assumptions = { ...assumptions, [key]: sug.suggested_value };
      if (sug.field === "exit_ev_ebitda") next.terminal_method = "exit_multiple";
      setAssumptions(next);
      recompute(next);
      setTab("Valuation");
    },
    [assumptions, recompute]
  );

  // Digests can take a minute+; if the user switched to another ticker while
  // one was in flight, drop the result instead of appending it to the wrong
  // company's research. Panels are keyed by ticker, so the closure's `ticker`
  // is the ticker the digest was started for.
  const onDigested = useCallback(
    (source: string, digest: Digest) => {
      if (tickerRef.current !== ticker) return;
      setDigests((prev) => [
        ...prev,
        { source, digest, at: new Date().toISOString() },
      ]);
      setResearchNotes((prev) =>
        (prev ? prev + "\n\n" : "") + `[${source}] ${digest.summary}`
      );
      setTab("AI Research");
    },
    [ticker]
  );

  // A note takes about a minute. If another load started meanwhile, never
  // show it on another ticker's page: save it to its own ticker (the store
  // merges), and if that ticker is on screen again show it too, now if its
  // saved state has been re-read, else once the restore lands (lateNote), so
  // the restore and the autosave that follows can't put the older note back.
  // A note from an older request that finishes after a newer request's note
  // for the same ticker is dropped (noteOrder).
  const generateNote = useCallback(async () => {
    if (!report || !ticker) return;
    const seq = loadSeq.current;
    const t = ticker;
    const order = noteOrder.current.start(t);
    setNoteLoading(true);
    setError(null);
    try {
      const n = await postResearchNote({
        report,
        extra_context: researchNotes,
      });
      if (!noteOrder.current.keep(t, order)) return;
      if (seq === loadSeq.current) {
        setNote(n);
      } else {
        lastSave.current = chainSave(
          lastSave.current,
          saveResearchState(t, { note: n })
        );
        if (tickerRef.current === t) {
          if (restoreDoneFor.current === t) setNote(n);
          else lateNote.current = { ticker: t, note: n };
        }
      }
    } catch (e) {
      if (seq === loadSeq.current)
        setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (seq === loadSeq.current) setNoteLoading(false);
    }
  }, [report, researchNotes, ticker]);

  // Export the inputs of the valuation on screen, not slider moves that
  // haven't been recomputed yet, so the file matches the dashboard.
  const handleExport = useCallback(
    async (kind: ExportKind) => {
      if (!ticker || !report || exporting !== null) return; // no concurrent exports
      setExporting(kind);
      setError(null);
      try {
        await downloadExport(
          kind,
          ticker,
          assumptionsFromUsed(report.assumptions_used),
          note
        );
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setExporting(null);
      }
    },
    [ticker, report, note, exporting]
  );

  const watching = watchlist.some((w) => w.ticker === ticker);
  const toggleWatch = useCallback(async () => {
    if (!ticker) return;
    try {
      if (watching) {
        setWatchlist(await updateWatchlist("remove", ticker));
      } else {
        const s = report?.summary;
        setWatchlist(
          await updateWatchlist("add", ticker, {
            name: s?.name ?? null,
            currency: s?.currency ?? null,
            price: s?.current_price ?? null,
            blended_target: s?.blended_target ?? null,
            recommendation: s?.recommendation ?? null,
          })
        );
      }
    } catch {
      /* watchlist is best-effort */
    }
  }, [ticker, watching, report]);

  const removeFromWatchlist = useCallback(async (t: string) => {
    try {
      setWatchlist(await updateWatchlist("remove", t));
    } catch {
      /* ignore */
    }
  }, []);

  const s = report?.summary;

  return (
    <div className="mx-auto max-w-7xl px-4 pb-16 pt-5">
      {/* Top bar */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <div className="text-base font-semibold tracking-tight text-ink">
            Equity Research <span className="text-brand">Automation</span>
          </div>
          <Badge tone="brand">DCF · Comps · Filings · AI</Badge>
        </div>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            loadTicker(input);
          }}
          className="flex items-center gap-2"
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ticker (e.g. AAPL, MSFT, NVDA)"
            className="w-56 rounded-lg border border-line bg-surface px-3 py-1.5 text-sm text-ink placeholder:text-ink-faint focus:border-brand focus:outline-none"
            autoFocus
          />
          <Button type="submit" disabled={loading}>
            {loading ? <Spinner /> : "Analyze"}
          </Button>
        </form>
      </div>

      {status && (
        <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-ink-faint">
          <span>Data: EDGAR + yfinance (free)</span>
          <span>·</span>
          <span className={status.fmp_enabled ? "text-up" : "text-ink-faint"}>
            FMP {status.fmp_enabled ? "connected" : "off"}
          </span>
          <span>·</span>
          <span
            className={status.anthropic_enabled ? "text-up" : "text-ink-faint"}
          >
            AI {status.anthropic_enabled ? "connected" : "off"}
          </span>
          {(!status.fmp_enabled || !status.anthropic_enabled) && (
            <button
              onClick={() => setShowKeys((v) => !v)}
              className="rounded-md border border-brand/40 bg-brand/10 px-2 py-0.5 font-medium text-brand transition hover:bg-brand/20"
            >
              {showKeys ? "Close" : "Add API keys"}
            </button>
          )}
        </div>
      )}

      {showKeys && status && (
        <form
          onSubmit={async (e) => {
            e.preventDefault();
            setSavingKeys(true);
            setError(null);
            try {
              const res = await fetch(apiUrl("/api/settings"), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  anthropic_api_key: keyAnthropic || null,
                  fmp_api_key: keyFmp || null,
                }),
              });
              if (!res.ok) throw new Error(await errorDetail(res));
              setStatus(await res.json());
              setKeyAnthropic("");
              setKeyFmp("");
              setShowKeys(false);
              // FMP just connected: refresh enrichment for the loaded ticker.
              if (ticker) {
                const seq = loadSeq.current;
                setEnrichLoading(true);
                fetchEnrichment(ticker)
                  .then((en) => {
                    if (seq === loadSeq.current) setEnrichment(en);
                  })
                  .catch(() => {})
                  .finally(() => {
                    if (seq === loadSeq.current) setEnrichLoading(false);
                  });
              }
            } catch (err2) {
              setError(err2 instanceof Error ? err2.message : String(err2));
            } finally {
              setSavingKeys(false);
            }
          }}
          className="mt-2 flex flex-wrap items-end gap-3 rounded-xl border border-line bg-surface px-4 py-3"
        >
          {!status.anthropic_enabled && (
            <label className="flex flex-col gap-1 text-[11px] text-ink-dim">
              Anthropic API key (AI researcher)
              <input
                type="password"
                value={keyAnthropic}
                onChange={(e) => setKeyAnthropic(e.target.value)}
                placeholder="sk-ant-…"
                className="w-72 rounded-lg border border-line bg-surface-raised px-3 py-1.5 text-sm text-ink placeholder:text-ink-faint focus:border-brand focus:outline-none"
              />
            </label>
          )}
          {!status.fmp_enabled && (
            <label className="flex flex-col gap-1 text-[11px] text-ink-dim">
              FMP API key (news, transcripts, estimates)
              <input
                type="password"
                value={keyFmp}
                onChange={(e) => setKeyFmp(e.target.value)}
                placeholder="optional"
                className="w-72 rounded-lg border border-line bg-surface-raised px-3 py-1.5 text-sm text-ink placeholder:text-ink-faint focus:border-brand focus:outline-none"
              />
            </label>
          )}
          <Button type="submit" disabled={savingKeys || (!keyAnthropic && !keyFmp)}>
            {savingKeys ? <Spinner /> : "Save keys"}
          </Button>
          <span className="text-[11px] text-ink-faint">
            Stored locally in the project&apos;s .env — never leaves your Mac.
          </span>
        </form>
      )}

      {error && (
        <div className="mt-4 rounded-lg border border-down/40 bg-down/10 px-4 py-3 text-sm text-down">
          {error}
        </div>
      )}

      {!report && !loading && !error && (
        <div className="mt-16 text-center">
          <h1 className="text-2xl font-semibold text-ink">
            Type a ticker for an instant multi-model valuation.
          </h1>
          <p className="mx-auto mt-3 max-w-xl text-sm text-ink-dim">
            DCF, DDM, FCFE, comps, sensitivity and reverse DCF — every
            assumption editable. The AI analyst reads SEC filings, earnings
            calls and anything you feed it, cites its sources, proposes model
            changes, and writes the research memo and deck.
          </p>

          {watchlist.length > 0 && (
            <div className="mx-auto mt-10 max-w-4xl text-left">
              <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-ink-faint">
                Watchlist
              </h2>
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
                {watchlist.map((w) => {
                  // No upside (and no colour) without a target or when the
                  // verdict was withheld.
                  const up = watchlistUpside(w);
                  const hasTarget = Number.isFinite(w.blended_target as number);
                  return (
                    <div
                      key={w.ticker}
                      onClick={() => loadTicker(w.ticker)}
                      className="group cursor-pointer rounded-xl border border-line bg-surface px-4 py-3 transition hover:border-brand"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-ink">
                            {w.ticker}
                          </span>
                          <span className="truncate text-xs text-ink-faint">
                            {w.name}
                          </span>
                        </div>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            removeFromWatchlist(w.ticker);
                          }}
                          className="text-ink-faint opacity-0 transition hover:text-down group-hover:opacity-100"
                          title="Remove"
                        >
                          ×
                        </button>
                      </div>
                      <div className="num mt-1 flex items-baseline gap-2 text-sm">
                        <span className="text-ink">
                          {fmtMoney(w.price, w.currency)}
                        </span>
                        <span className="text-ink-faint">→</span>
                        <span className={toneForUpside(up)}>
                          {hasTarget
                            ? `${fmtMoney(w.blended_target, w.currency)} (${
                                up == null ? "n/a" : fmtPct(up, { signed: true })
                              })`
                            : "no target"}
                        </span>
                      </div>
                      <div className="mt-1 flex items-center justify-between text-[11px]">
                        <span
                          className={toneForRecommendation(w.recommendation)}
                        >
                          {w.recommendation}
                        </span>
                        <span className="text-ink-faint">
                          {fmtDate(w.updated_at)}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {loading && (
        <div className="mt-24 flex items-center justify-center gap-3 text-ink-dim">
          <Spinner className="h-6 w-6" />
          <span>Pulling fundamentals and running the models…</span>
        </div>
      )}

      {report && s && (
        <>
          {/* Company header */}
          <div className="mt-5 flex flex-wrap items-end justify-between gap-4 rounded-xl border border-line bg-surface px-5 py-4">
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-xl font-semibold text-ink">{s.name}</h2>
                <Badge tone="neutral">{s.ticker}</Badge>
                <button
                  onClick={toggleWatch}
                  className={cx(
                    "text-lg leading-none transition",
                    watching
                      ? "text-flat"
                      : "text-ink-faint hover:text-flat"
                  )}
                  title={watching ? "Remove from watchlist" : "Add to watchlist"}
                >
                  {watching ? "★" : "☆"}
                </button>
                {report.company.market.sector && (
                  <span className="text-xs text-ink-faint">
                    {report.company.market.sector}
                  </span>
                )}
              </div>
              <div className="num mt-1 text-2xl font-semibold text-ink">
                {fmtMoney(s.current_price, s.currency)}
                <span className="ml-2 text-sm font-normal text-ink-faint">
                  {s.currency}
                </span>
              </div>
              <div className="mt-2 flex flex-wrap items-center gap-1.5">
                {EXPORTS.map((e) => (
                  <button
                    key={e.kind}
                    onClick={() => handleExport(e.kind)}
                    disabled={exporting !== null}
                    className="inline-flex items-center gap-1.5 rounded-md border border-line bg-surface-raised px-2 py-1 text-[11px] font-medium text-ink-dim transition hover:border-brand hover:text-ink disabled:opacity-50"
                  >
                    {exporting === e.kind ? (
                      <Spinner className="h-3 w-3" />
                    ) : (
                      <span aria-hidden>⬇</span>
                    )}
                    {e.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="flex items-end gap-6">
              <div className="text-right">
                <div className="text-[11px] uppercase tracking-wider text-ink-faint">
                  Blended fair value
                </div>
                {/* A null target reads "n/a" or "No target (supply peers)";
                    a withheld verdict leaves the upside "n/a", uncoloured. */}
                <div
                  className={cx(
                    "num font-semibold text-ink",
                    Number.isFinite(s.blended_target as number)
                      ? "text-2xl"
                      : "text-base"
                  )}
                >
                  {fmtBlendedTarget(s, report.comps)}
                </div>
                <div
                  className={cx(
                    "num text-sm font-medium",
                    toneForBlendedUpside(s)
                  )}
                >
                  {fmtBlendedUpside(s)} vs price
                </div>
              </div>
              <div className="text-right">
                <div className="text-[11px] uppercase tracking-wider text-ink-faint">
                  Verdict
                </div>
                <div
                  className={cx(
                    "text-lg font-semibold",
                    toneForRecommendation(s.recommendation)
                  )}
                >
                  {s.recommendation}
                </div>
                {recomputing && (
                  <div className="mt-1 flex items-center justify-end gap-1 text-xs text-ink-faint">
                    <Spinner className="h-3 w-3" /> recomputing
                  </div>
                )}
              </div>
            </div>
          </div>

          {restoreFailedFor === ticker && (
            <div className="mt-3 rounded-lg border border-flat/40 bg-flat/10 px-4 py-2 text-xs text-flat">
              Couldn&apos;t load the saved research for {ticker}, so autosave is
              paused to avoid overwriting it. Load the ticker again to retry.
            </div>
          )}

          {/* Tabs */}
          <div className="mt-4 flex flex-wrap gap-1 border-b border-line">
            {TABS.map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={cx(
                  "-mb-px border-b-2 px-3 py-2 text-sm font-medium transition",
                  tab === t
                    ? "border-brand text-ink"
                    : "border-transparent text-ink-faint hover:text-ink-dim"
                )}
              >
                {t}
                {t === "AI Research" && digests.length > 0 && (
                  <span className="num ml-1.5 rounded-full bg-brand/15 px-1.5 text-[10px] text-brand">
                    {digests.length}
                  </span>
                )}
              </button>
            ))}
          </div>

          <div className="mt-4">
            {tab === "Overview" && <ValuationSummary report={report} />}

            {tab === "Valuation" && (
              <div className="grid grid-cols-1 gap-4">
                <DCFPanel
                  report={report}
                  assumptions={assumptions}
                  setAssumptions={setAssumptions}
                  onRecompute={recompute}
                  recomputing={recomputing}
                />
                <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
                  <CompsPanel
                    report={report}
                    assumptions={assumptions}
                    setAssumptions={setAssumptions}
                    onRecompute={recompute}
                    recomputing={recomputing}
                  />
                  <DDMFCFEPanel report={report} />
                </div>
                <SensitivityPanel report={report} />
              </div>
            )}

            {tab === "Financials" && <FinancialsPanel report={report} />}

            {/* Kept mounted (hidden) so a 60s filing digest survives tab
                switches; keyed by ticker so state resets per company. */}
            <div className={tab === "Filings" ? "" : "hidden"}>
              <FilingsPanel
                key={ticker}
                ticker={ticker}
                report={report}
                extraContext={researchNotes}
                aiEnabled={status?.anthropic_enabled ?? false}
                fmpEnabled={status?.fmp_enabled ?? false}
                onDigested={onDigested}
              />
            </div>

            {tab === "Multiples" && (
              <MultiplesPanel report={report} enrichment={enrichment} />
            )}

            {tab === "News" && (
              <NewsPanel enrichment={enrichment} loading={enrichLoading} />
            )}

            {/* Kept mounted (hidden) so chat history and in-progress digests
                survive tab switches; keyed by ticker. */}
            <div className={tab === "AI Research" ? "" : "hidden"}>
              <AIResearchPanel
                key={ticker}
                report={report}
                researchNotes={researchNotes}
                setResearchNotes={setResearchNotes}
                onApplySuggestion={applySuggestion}
                aiEnabled={status?.anthropic_enabled ?? false}
                digests={digests}
                onDigested={onDigested}
                note={note}
                onGenerateNote={generateNote}
                noteLoading={noteLoading}
                onExportMemo={() => handleExport("memo")}
              />
            </div>
          </div>

          <p className="mt-8 text-center text-[11px] text-ink-faint">
            Model estimates driven by your assumptions — not investment advice.
            Sanity-check the drivers (growth, margins, WACC, peer set) before
            acting.
          </p>
        </>
      )}
    </div>
  );
}
