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
