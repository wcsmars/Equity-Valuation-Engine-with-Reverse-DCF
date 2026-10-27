"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import type {
  Assumptions,
  AssumptionSuggestion,
  Digest,
  DigestEntry,
  Enrichment,
  Report,
  ResearchNote,
  ResearchState,
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
  appendLogEntry,
  digestLogEntry,
  fmtBlendedTarget,
  fmtBlendedUpside,
  fmtMoney,
  toneForBlendedUpside,
  toneForRecommendation,
  upgradeDigestLog,
  withFullStop,
} from "@/lib/format";
import {
  SUGGESTION_KEYS,
  assumptionsFromUsed,
  customFields,
  isPlausibleTicker,
  peersOnly,
} from "@/lib/assumptions";
import {
  HeldDigests,
  LoadGate,
  NoteOrder,
  chainSave,
  digestLanding,
  waitForSaves,
} from "@/lib/sync";
import { Alert, Badge, Button, Spinner, TabBar, cx } from "@/components/ui";
import { ApiKeysForm, ApiKeysToggle } from "@/components/ApiKeys";
import { WatchlistGrid, WatchToggle } from "@/components/Watchlist";

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
import VerdictExplainer from "@/components/VerdictExplainer";

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

// Longest a ticker load waits for earlier saves to land before re-reading
// saved research (a local save takes milliseconds; this only bounds a save
// that never answers).
const SAVE_WAIT_MS = 10_000;

// Most peers FMP's list fills into an empty peer box when a ticker loads.
const MAX_AUTO_PEERS = 8;

// The HTTP status a failed API call carries (lib/api's ApiError), else null.
function httpStatus(e: unknown): number | null {
  const status = (e as { status?: unknown } | null)?.status;
  return typeof status === "number" ? status : null;
}

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
  // The failed load behind the error banner, for its Retry buttons: shown
  // only while that error is the one on screen.
  const [loadFailure, setLoadFailure] = useState<{
    message: string;
    status: number | null;
  } | null>(null);
  const [researchNotes, setResearchNotes] = useState("");
  const [digests, setDigests] = useState<DigestEntry[]>([]);
  const [note, setNote] = useState<ResearchNote | null>(null);
  const [noteLoading, setNoteLoading] = useState(false);
  const [watchlist, setWatchlist] = useState<WatchlistItem[]>([]);
  const watchlistRef = useRef<WatchlistItem[]>([]);
  const watchlistWrites = useRef<Promise<void> | null>(null);
  const watchlistRevision = useRef(0);
  const watchlistPending = useRef(new Set<string>());
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
  const [saveError, setSaveError] = useState<string | null>(null);
  // The header's "Custom inputs" list is open.
  const [showCustom, setShowCustom] = useState(false);
  // The comps card's peer input, focused by the header's "Add peers" action
  // once the Valuation tab (which holds the card) has rendered: a request
  // bumps peerFocus and sets peerFocusPending, which the effect consumes.
  const peerInputRef = useRef<HTMLInputElement | null>(null);
  const [peerFocus, setPeerFocus] = useState(0);
  const peerFocusPending = useRef(false);

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
  const unsaved = useRef(false);
  // Per-ticker order of note requests, so an older note that finishes after
  // a newer one never replaces it.
  const noteOrder = useRef(new NoteOrder());
  // Ticker loads and the view on screen (see LoadGate): work captures
  // loads.current.view when it starts and lands only while it is unchanged,
  // so it keeps landing on the ticker on screen until a load really replaces
  // it, and a stale load or recompute that finishes late is dropped.
  const loads = useRef(new LoadGate());
  const loadRunning = useRef(false); // a switched-to load is still fetching
  // The newest load asked for, until it lands or is refused: Analyze for a
  // ticker still loading does nothing, while another ticker replaces it
  // (LoadGate drops the older load).
  const requested = useRef<{ ticker: string; load: number } | null>(null);
  const recomputeSeq = useRef(0); // only the newest recompute may land
  // Digests that finished while a load was re-reading saved research, or was
  // waiting while the research on screen was not autosaved (digestLanding).
  const held = useRef(new HeldDigests<DigestEntry>());
  // The research log and digests as last rendered, for a save made outside
  // the autosave (a digest that lands while a load waits, see onDigested).
  const shownResearch = useRef<{ notes: string; digests: DigestEntry[] }>({
    notes: "",
    digests: [],
  });
  useEffect(() => {
    shownResearch.current = { notes: researchNotes, digests };
  }, [researchNotes, digests]);

  const queueResearchSave = useCallback((t: string, state: ResearchState) => {
    const save = chainSave(lastSave.current, () => saveResearchState(t, state));
    lastSave.current = save;
    unsaved.current = true;
    save.then(
      () => {
        if (lastSave.current === save) unsaved.current = false;
        setSaveError(null);
      },
      (e) => setSaveError(withFullStop(
        `Couldn't save research for ${t}: ${e instanceof Error ? e.message : String(e)}`
      ))
    );
  }, []);

  // Adds finished digests to the research on screen: the digest history and
  // the research log, which carries each digest's cited facts, risks and
  // catalysts (the log is the context later notes, chats and digests get).
  const appendDigests = useCallback((entries: DigestEntry[]) => {
    if (entries.length === 0) return;
    setDigests((prev) => [...prev, ...entries]);
    setResearchNotes((prev) =>
      entries.reduce((log, e) => appendLogEntry(log, digestLogEntry(e.source, e.digest)), prev)
    );
  }, []);

  useEffect(() => {
    const beforeUnload = (event: BeforeUnloadEvent) => {
      pendingSave.current?.();
      if (unsaved.current) {
        event.preventDefault();
        event.returnValue = "";
      }
    };
    const visibilityChanged = () => {
      if (document.visibilityState === "hidden") pendingSave.current?.();
    };
    window.addEventListener("beforeunload", beforeUnload);
    document.addEventListener("visibilitychange", visibilityChanged);
    return () => {
      window.removeEventListener("beforeunload", beforeUnload);
      document.removeEventListener("visibilitychange", visibilityChanged);
    };
  }, []);

  useEffect(() => {
    // Retry with backoff — in the desktop app the backend can still be
    // finishing its boot when the page mounts; one failed probe must not
    // permanently disable the AI/FMP features.
    let cancelled = false;
    const probe = (attempt: number) => {
      fetch(apiUrl("/api/health"))
        .then(async (r) => {
          if (!r.ok) throw new Error(await errorDetail(r));
          return r.json();
        })
        .then((s) => {
          if (!cancelled) setStatus(s);
        })
        .catch(() => {
          if (!cancelled && attempt < 6)
            setTimeout(() => probe(attempt + 1), 1000 * (attempt + 1));
        });
    };
    probe(0);
    const revision = watchlistRevision.current;
    fetchWatchlist().then((wl) => {
      if (!cancelled && revision === watchlistRevision.current) {
        watchlistRef.current = wl;
        setWatchlist(wl);
      }
    }).catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  // Server mutations and their full-list responses must stay ordered. A
  // refresh queued behind a removal also rechecks membership when it runs.
  const changeWatchlist = useCallback((action: "add" | "remove", t: string,
    snapshot?: Partial<WatchlistItem>, refreshOnly = false) => {
    watchlistRevision.current += 1;
    const write = chainSave(watchlistWrites.current, async () => {
      if (refreshOnly && !watchlistRef.current.some((w) => w.ticker === t)) return;
      const wl = await updateWatchlist(action, t, snapshot);
      watchlistRef.current = wl;
      setWatchlist(wl);
    });
    watchlistWrites.current = write;
    write.catch((e) => setError(`Couldn't update watchlist: ${e instanceof Error ? e.message : String(e)}`));
    return write;
  }, []);
  // Set synchronously by loadTicker (not only after render) so async work
  // started for the previous ticker can tell it has been superseded.
  const tickerRef = useRef("");
  useEffect(() => {
    tickerRef.current = ticker;
  }, [ticker]);

  const loadTicker = useCallback(async (sym: string, resetAssumptions = false) => {
    const t = sym.trim().toUpperCase();
    if (!t) return;
    // The backend refuses anything else (400 "Invalid ticker."); say so
    // before any request, and keep the ticker on screen.
    if (!isPlausibleTicker(t)) {
      setLoadFailure(null);
      setError(`"${sym.trim()}" is not a ticker: use up to 12 letters, digits, dots or hyphens (e.g. BRK-B, 7203.T).`);
      return;
    }
    // First write the debounced edits of the ticker on screen, and wait until
    // every save sent so far has landed (waitForSaves), before its state is
    // cleared (or a reload of the same ticker re-reads it). Until then the
    // ticker on screen stays live: a recompute, note or digest that finishes
    // meanwhile still lands on it (a note or digest is also saved at once, so
    // the wait covers it). The view switches only after the wait, so a
    // refused or superseded load leaves that work intact, and work started on
    // the old view can never land on the new one.
    const load = loads.current.start();
    requested.current = { ticker: t, load };
    const done = () => {
      if (requested.current?.load === load) requested.current = null;
    };
    // The input shows the ticker being loaded, however it was asked for (the
    // watchlist, Retry, Reset to defaults).
    setInput(t);
    setLoading(true);
    setError(null);
    setLoadFailure(null);
    const wait = await waitForSaves({
      flush: () => pendingSave.current?.(),
      last: () => lastSave.current,
      pending: () => pendingSave.current !== null,
      superseded: () => loads.current.isSuperseded(load),
    }, SAVE_WAIT_MS);
    // The view switches here, after the wait, and only when every save has
    // landed (LoadGate.settle); otherwise seq is null.
    const seq = loads.current.settle(load, wait);
    if (seq === null) {
      if (wait === "superseded") return; // the newer load takes over
      // Preserve the visible edits if saving failed or timed out. Reading old
      // server state and autosaving it would otherwise silently lose them.
      // Digests held while waiting go to the research still on screen.
      const onScreen = tickerRef.current;
      if (onScreen && restoreDoneFor.current === onScreen) {
        appendDigests(held.current.take(onScreen));
      }
      setError("Research has not finished saving. Your current edits are still here; retry saving before loading another ticker.");
      if (!loadRunning.current) setLoading(false);
      done();
      return;
    }
    loadRunning.current = true;
    held.current.keepOnly(t);
    setLoading(true); // an older load may have finished while this one waited
    setReport(null);
    setEnrichment(null);
    setEnrichLoading(false);
    setAssumptions({});
    setResearchNotes("");
    setDigests([]);
    setNote(null);
    setNoteLoading(false);
    setRecomputing(false);
    setRestoreFailedFor("");
    setShowCustom(false);
    peerFocusPending.current = false;
    autoPeeredFor.current = "";
    stateLoadedFor.current = "";
    restoreDoneFor.current = "";
    if (lateNote.current?.ticker !== t) lateNote.current = null;
    tickerRef.current = t;
    setTicker(t);

    // Restore persisted research FIRST so saved assumptions drive the first
    // valuation (and saved notes/digests/note come back with it). A never-seen
    // ticker restores as {}; a failed read throws.
    let savedAssumptions: Assumptions = {};
    let restored = false;
    try {
      const st = await fetchResearchState(t);
      if (seq !== loads.current.view) return; // a newer load superseded us
      // A log saved before digest entries carried their cited facts gets
      // them back from the saved digests (see upgradeDigestLog).
      setResearchNotes(upgradeDigestLog(st.notes ?? "", st.digests));
      setDigests(st.digests ?? []);
      setNote(st.note ?? null);
      if (st.assumptions && Object.keys(st.assumptions).length > 0) {
        // "Retry with default assumptions" drops the saved DCF drivers but
        // keeps the peer set (see peersOnly).
        savedAssumptions = resetAssumptions ? peersOnly(st.assumptions) : st.assumptions;
      }
      setAssumptions(savedAssumptions);
      restored = true;
    } catch {
      if (seq !== loads.current.view) return;
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
    // Digests for this ticker that finished while it was being re-read.
    appendDigests(held.current.take(t));

    try {
      const rep = await fetchValuation(t, savedAssumptions);
      if (seq !== loads.current.view) return;
      setReport(rep);
      setAssumptions(assumptionsFromUsed(rep.assumptions_used));
      setTab("Overview");
      if (restored) stateLoadedFor.current = t;
    } catch (e) {
      if (seq !== loads.current.view) return;
      // The backend's message as it is ("No market data for X; check the
      // symbol…" for a 404); the status decides which retries make sense.
      const message = e instanceof Error ? e.message : String(e);
      setError(message);
      setLoadFailure({ message, status: httpStatus(e) });
    } finally {
      if (seq === loads.current.view) {
        setLoading(false);
        loadRunning.current = false;
      }
      done();
    }
    setEnrichLoading(true);
    fetchEnrichment(t)
      .then((en) => {
        if (seq === loads.current.view) setEnrichment(en);
      })
      .catch(() => {
        if (seq === loads.current.view) setEnrichment({ enabled: false });
      })
      .finally(() => {
        if (seq === loads.current.view) setEnrichLoading(false);
      });
  }, [appendDigests]);

  const recompute = useCallback(
    async (override?: Assumptions) => {
      if (!ticker || ticker !== tickerRef.current) return;
      const a = override ?? assumptions;
      const seq = loads.current.view;
      const rseq = ++recomputeSeq.current;
      // Drop the result if another ticker was loaded or a newer recompute
      // started meanwhile — it would overwrite the newer report.
      const current = () =>
        seq === loads.current.view && rseq === recomputeSeq.current;
      setRecomputing(true);
      setError(null);
      try {
        const rep = await fetchValuation(ticker, a);
        if (current()) {
          setReport(rep);
          // Normalize clamped inputs without overwriting a newer unsent draft.
          setAssumptions((draft) => draft === a ? assumptionsFromUsed(rep.assumptions_used) : draft);
        }
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
      const next = { ...assumptions, peers: fmpPeers.slice(0, MAX_AUTO_PEERS).join(",") };
      setAssumptions(next);
      recompute(next);
    }
  }, [report, enrichment, ticker, assumptions, recompute]);

  // Autosave research state (debounced), only once this ticker's saved state
  // was restored and its valuation loaded (see stateLoadedFor).
  useEffect(() => {
    if (!ticker || stateLoadedFor.current !== ticker) return;
    unsaved.current = true;
    const save = () => {
      clearTimeout(id);
      if (pendingSave.current === save) pendingSave.current = null;
      queueResearchSave(ticker, {
          notes: researchNotes,
          digests,
          note,
          assumptions,
        });
    };
    const id = setTimeout(save, 1500);
    pendingSave.current = save;
    return () => {
      clearTimeout(id);
      if (pendingSave.current === save) pendingSave.current = null;
    };
  }, [ticker, researchNotes, digests, note, assumptions, queueResearchSave]);

  // "Add peers": the Valuation tab holds the comps card; focus its peer input
  // once that tab has rendered (the ref is attached by then).
  const addPeers = useCallback(() => {
    peerFocusPending.current = true;
    setTab("Valuation");
    setPeerFocus((n) => n + 1);
  }, []);
  useEffect(() => {
    if (!peerFocusPending.current || tab !== "Valuation") return;
    peerFocusPending.current = false;
    const el = peerInputRef.current;
    el?.focus();
    el?.scrollIntoView?.({ block: "center", behavior: "smooth" });
  }, [peerFocus, tab]);

  // Keep the watchlist snapshot fresh whenever a watched ticker reloads.
  // Reads through watchlistRef so a just-removed ticker isn't re-added by a
  // recompute that was already in flight.
  useEffect(() => {
    if (!report || !ticker) return;
    if (!watchlistRef.current.some((w) => w.ticker === ticker)) return;
    const s = report.summary;
    changeWatchlist("add", ticker, {
      name: s.name,
      currency: s.currency,
      price: s.current_price,
      blended_target: s.blended_target,
      recommendation: s.recommendation,
    }, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [report]);

  const applySuggestion = useCallback(
    (sug: AssumptionSuggestion) => {
      const key = SUGGESTION_KEYS[sug.field];
      const next: Assumptions = { ...assumptions, [key]: sug.suggested_value };
      if (sug.field === "revenue_growth_y1") next.revenue_growth = undefined;
      if (sug.field === "exit_ev_ebitda") next.terminal_method = "exit_multiple";
      setAssumptions(next);
      recompute(next);
      setTab("Valuation");
    },
    [assumptions, recompute]
  );

  // Digests can take a minute+. Panels are keyed by ticker, so the closure's
  // `ticker` is the ticker the digest was started for. A digest is never added
  // to another company's research. While a load waits for saves it is added
  // and saved at once, so the wait covers it before the view switches; while
  // this ticker's saved research is being re-read it is held until that
  // settles, so re-analyzing the same ticker keeps it (see digestLanding).
  const onDigested = useCallback(
    (source: string, digest: Digest) => {
      const entry: DigestEntry = { source, digest, at: new Date().toISOString() };
      const landing = digestLanding({
        ticker,
        onScreen: tickerRef.current,
        loadWaiting: loads.current.waiting,
        restored: restoreDoneFor.current === ticker,
        autosaved: stateLoadedFor.current === ticker,
      });
      if (landing === "hold") held.current.hold(ticker, entry);
      if (landing === "drop" || landing === "hold") return;
      appendDigests([entry]);
      if (landing === "save") {
        // Send the debounced edits first, then the research with this digest
        // added, so the newest save is the one the waiting load covers.
        pendingSave.current?.();
        const next = {
          notes: appendLogEntry(shownResearch.current.notes, digestLogEntry(source, digest)),
          digests: [...shownResearch.current.digests, entry],
        };
        shownResearch.current = next;
        queueResearchSave(ticker, next);
      }
      setTab("AI Research");
    },
    [ticker, appendDigests, queueResearchSave]
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
    const seq = loads.current.view;
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
      if (seq === loads.current.view) {
        setNote(n);
        // A load waiting for saves could switch away (or re-read this
        // ticker) before the autosave picks the note up; send it now so the
        // wait covers it.
        if (loads.current.waiting) {
          queueResearchSave(t, { note: n });
        }
      } else {
        queueResearchSave(t, { note: n });
        if (tickerRef.current === t) {
          if (restoreDoneFor.current === t) setNote(n);
          else lateNote.current = { ticker: t, note: n };
        }
      }
    } catch (e) {
      if (seq === loads.current.view)
        setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (seq === loads.current.view) setNoteLoading(false);
    }
  }, [report, researchNotes, ticker, queueResearchSave]);

  // Export the saved report snapshot so changing prices or cache expiry do
  // not change the valuation in the file. Unsent slider edits stay separate.
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
          note,
          report.report_id
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
    if (!ticker || watchlistPending.current.has(ticker)) return;
    watchlistPending.current.add(ticker);
    try {
      if (watchlistRef.current.some((w) => w.ticker === ticker)) {
        await changeWatchlist("remove", ticker);
      } else {
        const s = report?.summary;
        await changeWatchlist("add", ticker, {
            name: s?.name ?? null,
            currency: s?.currency ?? null,
            price: s?.current_price ?? null,
            blended_target: s?.blended_target ?? null,
            recommendation: s?.recommendation ?? null,
          });
      }
    } catch {
      /* changeWatchlist displays the error. */
    } finally {
      watchlistPending.current.delete(ticker);
    }
  }, [ticker, report, changeWatchlist]);

  const removeFromWatchlist = useCallback(async (t: string) => {
    try {
      await changeWatchlist("remove", t);
    } catch {
      /* ignore */
    }
  }, [changeWatchlist]);

  const s = report?.summary;
  // Inputs this report was run with that differ from the engine's defaults
  // (saved research restores them silently, so the header says so).
  const custom = report ? customFields(report.assumptions_used, report.assumptions_default) : [];

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
            // Analyze stays enabled while a ticker loads, so another ticker
            // can replace it; the one already loading is not asked for twice.
            if (input.trim().toUpperCase() === requested.current?.ticker) return;
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
          <Button type="submit">
            {loading ? <Spinner /> : "Analyze"}
          </Button>
        </form>
      </div>

      {status && (
        <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-ink-faint">
          <span>Data: EDGAR + yfinance (free)</span>
          <span>·</span>
          <span className={status.fmp_enabled ? "text-up" : "text-ink-faint"}>
            FMP {status.fmp_enabled ? "configured" : "off"}
          </span>
          <span>·</span>
          <span
            className={status.anthropic_enabled ? "text-up" : "text-ink-faint"}
          >
            AI {status.anthropic_enabled ? "configured" : "off"}
          </span>
          {/* Always offered, so a wrong or revoked key can be replaced. */}
          <ApiKeysToggle
            status={status}
            open={showKeys}
            onToggle={() => setShowKeys((v) => !v)}
          />
        </div>
      )}

      {showKeys && status && (
        <ApiKeysForm
          status={status}
          anthropicKey={keyAnthropic}
          fmpKey={keyFmp}
          setAnthropicKey={setKeyAnthropic}
          setFmpKey={setKeyFmp}
          saving={savingKeys}
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
              const activeTicker = tickerRef.current;
              if (activeTicker) {
                const seq = loads.current.view;
                setEnrichLoading(true);
                fetchEnrichment(activeTicker)
                  .then((en) => {
                    if (seq === loads.current.view) setEnrichment(en);
                  })
                  .catch(() => {})
                  .finally(() => {
                    if (seq === loads.current.view) setEnrichLoading(false);
                  });
              }
            } catch (err2) {
              setError(err2 instanceof Error ? err2.message : String(err2));
            } finally {
              setSavingKeys(false);
            }
          }}
        />
      )}

      {error && (
        <Alert className="mt-4">
          {error}
          {/* A plain retry keeps the saved assumptions; the second drops the
              saved DCF drivers (the usual cause of a refused valuation) but
              keeps the peer set. Neither is offered for an error that is not
              a failed load, and no assumption brings back market data that
              is missing (404). */}
          {!report && !loading && ticker && loadFailure?.message === error && (
            <>
              <Button variant="ghost" className="ml-3" onClick={() => loadTicker(ticker)}>
                Retry
              </Button>
              {loadFailure.status !== 404 && (
                <Button variant="ghost" className="ml-2" onClick={() => loadTicker(ticker, true)}>
                  Retry with default assumptions
                </Button>
              )}
            </>
          )}
        </Alert>
      )}

      {saveError && (
        <Alert tone="flat" className="mt-3">
          {saveError} Keep this page open until saving succeeds.
          {ticker && stateLoadedFor.current === ticker && (
            <Button variant="ghost" className="ml-3" onClick={() => queueResearchSave(ticker, {
              notes: researchNotes, digests, note, assumptions,
            })}>Retry save</Button>
          )}
        </Alert>
      )}

      {/* Shown under an error too, so a mistyped ticker does not hide the
          watchlist. */}
      {!report && !loading && (
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
              <WatchlistGrid
                items={watchlist}
                onOpen={(t) => loadTicker(t)}
                onRemove={removeFromWatchlist}
              />
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
          {/* relative: the Custom inputs list is placed against this card,
              not its button, so it stays on screen at every width. */}
          <div className="relative mt-5 flex flex-wrap items-end justify-between gap-4 rounded-xl border border-line bg-surface px-5 py-4">
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-xl font-semibold text-ink">{s.name}</h2>
                <Badge tone="neutral">{s.ticker}</Badge>
                <WatchToggle
                  ticker={s.ticker}
                  watching={watching}
                  onToggle={toggleWatch}
                />
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
              {report.valued_at && (
                <p className="mt-1 text-[11px] text-ink-faint">
                  Calculated {new Date(report.valued_at).toLocaleString(undefined, {
                    dateStyle: "medium", timeStyle: "short",
                  })}. Market prices and filings may be older.
                </p>
              )}
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
            <div className="flex flex-col items-end gap-1.5">
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
                  <div className="flex items-center justify-end gap-2">
                    {/* Saved research restores custom inputs silently; say
                        which ones drive this verdict, with a way back. */}
                    {custom.length > 0 && (
                      <div
                        onKeyDown={(e) => {
                          if (e.key === "Escape") setShowCustom(false);
                        }}
                      >
                        <button
                          type="button"
                          aria-expanded={showCustom}
                          // The list exists only while open.
                          aria-controls={showCustom ? "custom-inputs" : undefined}
                          onClick={() => setShowCustom((v) => !v)}
                          title="This valuation uses inputs that differ from the engine's defaults"
                          className="rounded-md border border-flat/40 bg-flat/10 px-2 py-0.5 text-[11px] font-medium text-flat transition hover:border-flat"
                        >
                          {`Custom inputs (${custom.length})`}
                        </button>
                        {showCustom && (
                          <div
                            id="custom-inputs"
                            role="region"
                            aria-label="Custom inputs"
                            // Placed against the header card: a 320 px panel
                            // aligned to the button ran off the left edge
                            // wherever the card wraps and the button sits
                            // mid-row (phones, and tablets up to about 1000
                            // px). Card-wide there; from lg up, 320 px under
                            // the card's right edge, beside the verdict.
                            className="absolute inset-x-3 top-full z-20 mt-1 rounded-lg border border-line bg-surface p-3 text-left shadow-lg lg:left-auto lg:right-5 lg:w-80"
                          >
                            <p className="text-[11px] text-ink-faint">
                              This valuation uses inputs that differ from the
                              engine&apos;s defaults. They are saved with this
                              ticker&apos;s research and come back when it loads.
                            </p>
                            <ul className="mt-2 space-y-1 text-xs">
                              {custom.map((c) => (
                                <li key={c.key} className="flex items-baseline justify-between gap-3">
                                  <span className="shrink-0 text-ink-dim">{c.label}</span>
                                  <span className="num min-w-0 break-words text-right text-ink">
                                    {c.to}{" "}
                                    <span className="text-ink-faint">(default {c.from})</span>
                                  </span>
                                </li>
                              ))}
                            </ul>
                            <Button
                              variant="ghost"
                              className="mt-3 w-full"
                              disabled={loading}
                              onClick={() => {
                                setShowCustom(false);
                                loadTicker(ticker, true);
                              }}
                            >
                              Reset to defaults
                            </Button>
                            {custom.some((c) => c.key === "peers") && (
                              <p className="mt-1 text-[11px] text-ink-faint">
                                Reset keeps the peer set; edit it in the Trading
                                comps box.
                              </p>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                    <div
                      className={cx(
                        "text-lg font-semibold",
                        toneForRecommendation(s.recommendation)
                      )}
                    >
                      {s.recommendation}
                    </div>
                  </div>
                  {recomputing && (
                    <div className="mt-1 flex items-center justify-end gap-1 text-xs text-ink-faint">
                      <Spinner className="h-3 w-3" /> recomputing
                    </div>
                  )}
                </div>
              </div>
              {/* Why there is no verdict, no target or a smaller blend. */}
              <VerdictExplainer
                summary={s}
                comps={report.comps}
                onAddPeers={addPeers}
              />
            </div>
          </div>

          {restoreFailedFor === ticker && (
            <div className="mt-3 rounded-lg border border-flat/40 bg-flat/10 px-4 py-2 text-xs text-flat">
              Couldn&apos;t load the saved research for {ticker}. Research editing
              and AI actions are paused to protect your saved notes. Load the
              ticker again to retry.
            </div>
          )}

          {/* Tabs */}
          <TabBar
            className="mt-4"
            label="Report sections"
            tabs={TABS}
            value={tab}
            onChange={setTab}
            badge={(t) =>
              t === "AI Research" && digests.length > 0 && (
                <span className="num ml-1.5 rounded-full bg-brand/15 px-1.5 text-[10px] text-brand">
                  {digests.length}
                </span>
              )
            }
          />

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
                    inputRef={peerInputRef}
                  />
                  <DDMFCFEPanel
                    report={report}
                    assumptions={assumptions}
                    setAssumptions={setAssumptions}
                    onRecompute={recompute}
                    recomputing={recomputing}
                  />
                </div>
                <SensitivityPanel report={report} />
              </div>
            )}

            {tab === "Financials" && <FinancialsPanel report={report} />}

            {/* Kept mounted (hidden) so a 60s filing digest survives tab
                switches; keyed by ticker so state resets per company. */}
            <fieldset disabled={restoreFailedFor === ticker} className={tab === "Filings" ? "" : "hidden"}>
              <FilingsPanel
                key={ticker}
                ticker={ticker}
                report={report}
                extraContext={researchNotes}
                aiEnabled={status?.anthropic_enabled ?? false}
                fmpEnabled={status?.fmp_enabled ?? false}
                onDigested={onDigested}
              />
            </fieldset>

            {tab === "Multiples" && (
              <MultiplesPanel report={report} enrichment={enrichment} />
            )}

            {tab === "News" && (
              <NewsPanel enrichment={enrichment} loading={enrichLoading} />
            )}

            {/* Kept mounted (hidden) so chat history and in-progress digests
                survive tab switches; keyed by ticker. */}
            <fieldset disabled={restoreFailedFor === ticker} className={tab === "AI Research" ? "" : "hidden"}>
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
            </fieldset>
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
