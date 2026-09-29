// Save chaining and research-note ordering (lib/sync.ts), the helpers the
// dashboard page uses so a reload never reads stale research and an older
// note never replaces a newer one. Run with: npm test

import assert from "node:assert/strict";
import { test } from "node:test";
import { loadLib } from "./load-lib.mjs";

const { NoteOrder, chainSave, settledWithin } = loadLib("sync");

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

test("the chain waits for an older save that is still in flight", async () => {
  // A note merge-save (older) is still running when an autosave (newer)
  // finishes: a load must keep waiting for the older one.
  const older = deferred();
  const newer = deferred();
  let chain = chainSave(null, older.promise);
  chain = chainSave(chain, newer.promise);
  newer.resolve();
  assert.equal(await isSettled(chain), false);
  older.resolve();
  assert.equal(await isSettled(chain), true);
});

test("a failed save settles the chain instead of rejecting it", async () => {
  const failed = deferred();
  const ok = deferred();
  const chain = chainSave(chainSave(null, failed.promise), ok.promise);
  failed.reject(new Error("413"));
  ok.resolve();
  await chain; // must not throw
  assert.equal(await isSettled(chain), true);
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
  await settledWithin(hung, 30);
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
