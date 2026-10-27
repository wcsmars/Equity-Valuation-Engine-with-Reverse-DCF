// Save chaining and research-note ordering (lib/sync.ts), the helpers the
// dashboard page uses so a reload never reads stale research and an older
// note never replaces a newer one. Run with: npm test

import assert from "node:assert/strict";
import { test } from "node:test";
import { loadLib } from "./load-lib.mjs";

const {
  HeldDigests,
  LoadGate,
  NoteOrder,
  chainSave,
  digestLanding,
  settledWithin,
  waitForSaves,
} = loadLib("sync");

// A promise with its resolve/reject exposed.
function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

// Whether `p` has settled once pending callbacks have run.
async function isSettled(p) {
  let done = false;
  p.then(
    () => (done = true),
    () => (done = true)
  );
  await new Promise((r) => setTimeout(r, 0));
  return done;
}

test("writes are dispatched in edit order so a slow older save cannot overwrite newer research", async () => {
  const older = deferred();
  const newer = deferred();
  const calls = [];
  let persisted = "";
  let chain = chainSave(null, async () => {
    calls.push("older");
    await older.promise;
    persisted = "older";
  });
  chain = chainSave(chain, async () => {
    calls.push("newer");
    await newer.promise;
    persisted = "newer";
  });
  assert.equal(await isSettled(chain), false);
  assert.deepEqual(calls, ["older"]);
  older.resolve();
  assert.equal(await isSettled(chain), false);
  assert.deepEqual(calls, ["older", "newer"]);
  newer.resolve();
  assert.equal(await isSettled(chain), true);
  assert.equal(persisted, "newer");
});

test("failed saves are visible but a retry can still succeed", async () => {
  const failed = chainSave(null, async () => { throw new Error("413"); });
  await assert.rejects(failed, /413/);
  assert.equal(await settledWithin(failed, 1000), false);
  const chain = chainSave(failed, async () => {});
  await chain;
  assert.equal(await isSettled(chain), true);
  assert.equal(await settledWithin(chain, 1000), true);
});

test("a load waits for the saves, but not forever", async () => {
  assert.equal(await isSettled(settledWithin(null, 1000)), true);
  const quick = deferred();
  const waited = settledWithin(quick.promise, 60_000);
  assert.equal(await isSettled(waited), false);
  quick.resolve();
  assert.equal(await isSettled(waited), true);
  // A save that never answers stops holding up the load after the bound.
  const hung = new Promise(() => {});
  const t0 = Date.now();
  assert.equal(await settledWithin(hung, 30), false);
  assert.ok(Date.now() - t0 >= 25);
});

test("an older note that finishes after a newer one is dropped", () => {
  const order = new NoteOrder();
  const n1 = order.start("SYNT"); // before a reload of SYNT
  const n2 = order.start("SYNT"); // after it
  assert.equal(order.keep("SYNT", n2), true);
  assert.equal(order.keep("SYNT", n1), false);
});

test("notes that finish in order are all kept", () => {
  const order = new NoteOrder();
  const n1 = order.start("SYNT");
  const n2 = order.start("SYNT");
  assert.equal(order.keep("SYNT", n1), true);
  assert.equal(order.keep("SYNT", n2), true);
});

test("an older note is kept when the newer request fails", () => {
  // The newer request never calls keep(); the older note is still the
  // newest one there is.
  const order = new NoteOrder();
  const n1 = order.start("SYNT");
  order.start("SYNT");
  assert.equal(order.keep("SYNT", n1), true);
});

test("note order is tracked per ticker", () => {
  const order = new NoteOrder();
  const a = order.start("ABC");
  const s = order.start("SYNT");
  const a2 = order.start("ABC");
  assert.equal(order.keep("ABC", a2), true);
  assert.equal(order.keep("SYNT", s), true);
  assert.equal(order.keep("ABC", a), false);
});

// The page's saving as waitForSaves sees it: a debounced edit waiting to be
// sent (pending), and the chain of saves sent so far (last). Each save that
// is sent waits on a deferred the test settles.
function savingModel() {
  const m = {
    pending: false,
    last: null,
    sent: [],
    superseded: false,
    edit() { m.pending = true; }, // an edit lands; the debounce timer runs
    send() { // a save sent directly (a note landing while a load waits)
      const d = deferred();
      m.sent.push(d);
      m.last = chainSave(m.last, () => d.promise);
    },
    io: {
      flush: () => { if (m.pending) { m.pending = false; m.send(); } },
      last: () => m.last,
      pending: () => m.pending,
      superseded: () => m.superseded,
    },
  };
  return m;
}

test("a load switches only after every save, including edits that land while it waits", async () => {
  const m = savingModel();
  assert.equal(await waitForSaves(m.io, 1000), "saved"); // nothing to wait for
  m.edit(); // a slider edit still debounced
  const wait = waitForSaves(m.io, 1000);
  assert.equal(m.sent.length, 1); // flushed at once
  // While the first save is slow, a digest lands on the ticker on screen.
  m.edit();
  m.sent[0].resolve();
  assert.equal(await isSettled(wait), false);
  assert.equal(m.sent.length, 2); // the new edit was sent too
  // A note lands and is saved directly.
  m.send();
  m.sent[1].resolve();
  assert.equal(await isSettled(wait), false);
  m.sent[2].resolve();
  assert.equal(await wait, "saved");
});

test("a failed or slow save refuses the load, and a newer load supersedes a waiting one", async () => {
  const failing = savingModel();
  failing.edit();
  const refused = waitForSaves(failing.io, 1000);
  failing.sent[0].reject(new Error("503 store locked"));
  assert.equal(await refused, "failed");

  const hung = savingModel();
  hung.send();
  assert.equal(await waitForSaves(hung.io, 20), "failed");

  const older = savingModel();
  older.send();
  const wait = waitForSaves(older.io, 1000);
  older.superseded = true; // another ticker was asked for meanwhile
  older.sent[0].reject(new Error("503"));
  assert.equal(await wait, "superseded");
});

test("a load gives up when edits never stop arriving", async () => {
  const io = { flush() {}, last: () => null, pending: () => true, superseded: () => false };
  assert.equal(await waitForSaves(io, 1000, 3), "failed");
});

test("a finished digest is never added to another ticker, and is saved or held while a load is pending", () => {
  const at = (o) => digestLanding({ ticker: "AAPL", onScreen: "AAPL", loadWaiting: false,
    restored: true, autosaved: true, ...o });
  assert.equal(at({}), "append");
  assert.equal(at({ onScreen: "NOK" }), "drop");
  assert.equal(at({ onScreen: "NOK", loadWaiting: true }), "drop");
  // A load waits for saves before switching away from (or re-reading) AAPL:
  // the digest is saved at once so the wait covers it.
  assert.equal(at({ loadWaiting: true }), "save");
  // ...unless AAPL's research is not autosaved (it could not be read, or its
  // valuation has not loaded): then it is held.
  assert.equal(at({ loadWaiting: true, autosaved: false }), "hold");
  // Re-reading AAPL's saved research after Analyze was pressed on AAPL again.
  assert.equal(at({ restored: false, autosaved: false }), "hold");
  // Research that could not be read is shown, not autosaved: added on screen.
  assert.equal(at({ autosaved: false }), "append");
});

test("held digests go to their own ticker, in order, once", () => {
  const held = new HeldDigests();
  held.hold("AAPL", "d1");
  held.hold("NOK", "n1");
  held.hold("AAPL", "d2");
  held.keepOnly("AAPL"); // the load switched to AAPL (a reload)
  assert.deepEqual(held.take("AAPL"), ["d1", "d2"]);
  assert.deepEqual(held.take("AAPL"), []);
  held.hold("AAPL", "d3");
  held.keepOnly("NOK"); // switched to another ticker: AAPL's digest goes
  assert.deepEqual(held.take("NOK"), []);
});

test("a load switches the view only once its saves have landed", () => {
  const gate = new LoadGate();
  assert.equal(gate.view, 0);
  assert.equal(gate.waiting, false);
  // AAPL loads at once.
  const aapl = gate.start();
  assert.equal(gate.waiting, true);
  assert.equal(gate.settle(aapl, "saved"), 1);
  assert.equal(gate.view, 1);
  assert.equal(gate.waiting, false);
  // NOK is asked for; while it waits for AAPL's saves, a recompute (or a
  // note) starts on AAPL's view.
  const nok = gate.start();
  assert.equal(gate.waiting, true);
  const recompute = gate.view;
  assert.equal(recompute, 1); // still AAPL's view: the switch has not happened
  assert.equal(gate.settle(nok, "saved"), 2);
  // The recompute finishes after the switch: it belongs to AAPL and is
  // dropped rather than landing on NOK's page.
  assert.notEqual(recompute, gate.view);
});

test("a refused load leaves the view, and work for the ticker on screen still lands", () => {
  const gate = new LoadGate();
  gate.settle(gate.start(), "saved");
  const note = gate.view; // a note starts for the ticker on screen
  const nok = gate.start();
  assert.equal(gate.settle(nok, "failed"), null); // a save failed: no switch
  assert.equal(gate.waiting, false);
  assert.equal(note, gate.view); // the note still lands
});

test("a newer load supersedes a waiting one, and only the newest switches", () => {
  const gate = new LoadGate();
  const first = gate.start();
  const second = gate.start();
  assert.equal(gate.isSuperseded(first), true);
  assert.equal(gate.isSuperseded(second), false);
  assert.equal(gate.settle(first, "superseded"), null);
  // A superseded load cannot switch the view whatever its wait returned,
  // and leaves the newer one waiting.
  assert.equal(gate.settle(first, "saved"), null);
  assert.equal(gate.waiting, true);
  assert.equal(gate.view, 0);
  assert.equal(gate.settle(second, "saved"), 1);
  assert.equal(gate.waiting, false);
});
