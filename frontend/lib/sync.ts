// Ordering helpers for saving research state and for research notes that
// finish late (used by app/page.tsx and its tests).
//
// Saves run concurrently, and a note takes about a minute, so the page must
// (a) wait for every save still in flight before it re-reads a ticker's saved
// research, not only the latest one, and (b) never let an older note replace
// a newer one for the same ticker.

// The promise of every save so far plus `save`. Waiting on it waits for all of
// them: a newer save does not make an older one still in flight forgotten. A
// failed save counts as settled.
export function chainSave(
  prev: Promise<void> | null,
  save: Promise<unknown>
): Promise<void> {
  const settled = save.then(
    () => undefined,
    () => undefined
  );
  return prev ? Promise.all([prev, settled]).then(() => undefined) : settled;
}

// Resolves once `p` settles or `ms` milliseconds have passed, whichever comes
// first, so one request that never answers cannot hold up every later load.
export function settledWithin(
  p: Promise<unknown> | null,
  ms: number
): Promise<void> {
  if (!p) return Promise.resolve();
  return new Promise<void>((resolve) => {
    const timer = setTimeout(resolve, ms);
    const done = () => {
      clearTimeout(timer);
      resolve();
    };
    p.then(done, done);
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
