// Ordering helpers for saving research state and for research notes and
// digests that finish late (used by app/page.tsx and its tests).
//
// A note or a digest takes about a minute, so the page must
// (a) wait for every save still in flight before it re-reads a ticker's saved
// research or switches to another ticker, not only the latest one, and switch
// the view only after that wait (LoadGate), (b) never let an older note
// replace a newer one for the same ticker, and (c) never add a digest to
// another company's research or lose one to a reload or a switch.

// Dispatch writes in the same order as edits. Passing a thunk is essential:
// accepting an already-started promise would let an older request land last
// and overwrite newer research. A failed write remains visible to its caller
// but does not prevent the next write from trying again.
export function chainSave(
  prev: Promise<void> | null,
  save: () => Promise<unknown>
): Promise<void> {
  return (prev ?? Promise.resolve()).catch(() => undefined).then(save).then(() => undefined);
}

// Resolves once `p` settles or `ms` milliseconds have passed, whichever comes
// first, so one request that never answers cannot hold up every later load.
export function settledWithin(
  p: Promise<unknown> | null,
  ms: number
): Promise<boolean> {
  if (!p) return Promise.resolve(true);
  return new Promise<boolean>((resolve) => {
    const timer = setTimeout(() => resolve(false), ms);
    const done = (ok: boolean) => {
      clearTimeout(timer);
      resolve(ok);
    };
    p.then(() => done(true), () => done(false));
  });
}

// Per-ticker order of research-note requests. Each request takes a number
// from start(); when it finishes, keep() says whether to use its note. A note
// is dropped only when a note from a later request for the same ticker was
// already kept, so an older note (built on older context) that finishes last
// never replaces a newer one, while a failed newer request still leaves the
// older note usable.
export class NoteOrder {
  private started = new Map<string, number>();
  private kept = new Map<string, number>();

  start(ticker: string): number {
    const n = (this.started.get(ticker) ?? 0) + 1;
    this.started.set(ticker, n);
    return n;
  }

  keep(ticker: string, n: number): boolean {
    if (n < (this.kept.get(ticker) ?? 0)) return false;
    this.kept.set(ticker, n);
    return true;
  }
}

// What a ticker load got from waitForSaves.
export type SaveWait = "saved" | "failed" | "superseded";

// Waits, before a load clears the ticker on screen or re-reads saved research,
// until every research save sent so far has landed. Debounced edits are sent
// first, and so is anything that lands while waiting (a note, a digest or a
// recompute for the ticker still on screen), so the wait repeats until a pass
// sends nothing new. "failed": a save failed or took longer than `ms`;
// "superseded": a newer load was asked for meanwhile. Nothing on screen may
// be switched before this returns "saved".
export async function waitForSaves(
  io: {
    flush: () => void; // send the debounced save now, if one is waiting
    last: () => Promise<unknown> | null; // the newest save sent
    pending: () => boolean; // a debounced save is waiting
    superseded: () => boolean;
  },
  ms: number,
  maxPasses = 50
): Promise<SaveWait> {
  for (let pass = 0; pass < maxPasses; pass++) {
    io.flush();
    const waitedOn = io.last();
    const ok = await settledWithin(waitedOn, ms);
    if (io.superseded()) return "superseded";
    if (!ok) return "failed";
    if (!io.pending() && io.last() === waitedOn) return "saved";
  }
  return "failed"; // edits kept arriving; keep the ticker on screen
}

// Ticker loads and the view on screen. Each load takes a number from start()
// and waits for earlier saves (waitForSaves); settle() then says whether it
// may switch the view. Only "saved" switches it, returning the new view
// number; a refused ("failed") or superseded load returns null and leaves the
// view as it is. Work for the ticker on screen (a recompute, a note) captures
// `view` when it starts and lands only while the view is unchanged, so it
// keeps landing until a load really replaces that ticker, and work started on
// an old view never lands on a new one. `waiting` is true from a load's
// start() until its settle(), while it may still switch the view.
export class LoadGate {
  private asked = 0; // loads asked for
  private settled = 0; // the newest load whose wait for saves has ended
  private shown = 0; // views switched to

  start(): number {
    return ++this.asked;
  }

  isSuperseded(load: number): boolean {
    return load !== this.asked;
  }

  get waiting(): boolean {
    return this.asked !== this.settled;
  }

  get view(): number {
    return this.shown;
  }

  settle(load: number, wait: SaveWait): number | null {
    if (wait === "superseded" || this.isSuperseded(load)) return null;
    this.settled = load;
    return wait === "saved" ? ++this.shown : null;
  }
}

// What to do with a digest that finished for `ticker` (the ticker its panel
// was showing; a digest can take a minute):
//  "drop":   another ticker is on screen; never add it to the wrong company.
//  "save":   the ticker is on screen with its research loaded and autosaved,
//            but a load is waiting for saves and may then replace it (another
//            ticker, or a re-read of this one); add it now and save it at
//            once, so the load's wait covers it, as it does a note.
//  "hold":   the ticker is on screen but a load is waiting while its research
//            is not autosaved, or the load is re-reading its saved research,
//            and would replace what is shown; keep it until that settles
//            (HeldDigests).
//  "append": add it to the research on screen now (the autosave saves it).
export function digestLanding(s: {
  ticker: string;
  onScreen: string;
  loadWaiting: boolean;
  restored: boolean; // the read of the ticker's saved research has finished
  autosaved: boolean; // that read succeeded and its valuation loaded
}): "drop" | "save" | "hold" | "append" {
  if (s.onScreen !== s.ticker) return "drop";
  if (s.loadWaiting) return s.autosaved ? "save" : "hold";
  if (!s.restored) return "hold";
  return "append";
}

// Digests held back by digestLanding until the load in progress settles: the
// ticker's restore applies them on top of its saved research, or, when the
// load is refused, they go to the research still on screen. A switch to
// another ticker drops them: they finished while their ticker's research was
// not autosaved (its saved research could not be read, or its valuation had
// not loaded), so there is no saved research to add them to.
export class HeldDigests<E> {
  private items: { ticker: string; entry: E }[] = [];

  hold(ticker: string, entry: E): void {
    this.items.push({ ticker, entry });
  }

  // A load switched to `ticker`: digests held for any other ticker are
  // dropped (that research is no longer on screen).
  keepOnly(ticker: string): void {
    this.items = this.items.filter((d) => d.ticker === ticker);
  }

  // The digests held for `ticker`, in the order they finished; the rest are
  // dropped.
  take(ticker: string): E[] {
    const out = this.items.filter((d) => d.ticker === ticker).map((d) => d.entry);
    this.items = [];
    return out;
  }
}
