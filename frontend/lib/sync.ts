// Ordering helpers for saving research state and for research notes that
// finish late (used by app/page.tsx and its tests).
//
// A note takes about a minute, so the page must
// (a) wait for every save still in flight before it re-reads a ticker's saved
// research, not only the latest one, and (b) never let an older note replace
// a newer one for the same ticker.

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
