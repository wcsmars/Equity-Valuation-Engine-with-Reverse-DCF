import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import { loadLib } from "./load-lib.mjs";
const { apiBase, errorDetail, downloadExport, fetchPeerSuggestions, fetchValuation } =
  loadLib("api");
const originalWindow = globalThis.window;
const originalFetch = globalThis.fetch;
const originalDocument = globalThis.document;
afterEach(() => {
  globalThis.window = originalWindow;
  globalThis.fetch = originalFetch;
  globalThis.document = originalDocument;
});

test("desktop API override accepts only valid loopback ports", () => {
  for (const [hostname, search, expected] of [
    ["127.0.0.1", "?api=18080", "http://127.0.0.1:18080"],
    ["localhost", "?api=00080", "http://127.0.0.1:80"],
    ["localhost", "?api=0", ""], ["localhost", "?api=65536", ""],
    ["localhost", "?api=http://bad.example", ""],
    ["example.com", "?api=18080", ""],
  ]) {
    globalThis.window = { location: { hostname, search } };
    assert.equal(apiBase(), expected);
  }
});

test("API validation errors remain readable and proxy HTML gets a status fallback", async () => {
  const response = new Response(JSON.stringify({ detail: [{ loc: ["body", "ticker"], msg: "Required" }] }), { status: 422 });
  assert.equal(await errorDetail(response), "ticker: Required");
  assert.equal(await errorDetail(new Response("<html>proxy down</html>", { status: 502, statusText: "Bad Gateway" })), "502 Bad Gateway");
});

test("exports identify the displayed report and surface expired-snapshot errors", async () => {
  let sent;
  globalThis.fetch = async (_url, options) => {
    sent = JSON.parse(options.body);
    return new Response(JSON.stringify({ detail: "Report snapshot expired; recompute the valuation." }), { status: 409 });
  };
  await assert.rejects(downloadExport("excel", "SYNT", { terminal_growth: 0.025 }, null, "saved-report"), /snapshot expired/);
  assert.equal(sent.report_id, "saved-report");
  assert.equal(sent.ticker, "SYNT");
  assert.equal(sent.terminal_growth, 0.025);
});

test("peer suggestions are fetched for the ticker and kept to well-formed rows", async () => {
  let url;
  globalThis.window = undefined;
  globalThis.fetch = async (u) => {
    url = u;
    return new Response(JSON.stringify({ ticker: "BRK.B", industry: "Insurance",
      peers: [{ ticker: "CB", name: "Chubb", size_ratio: 0.2 }, { name: "no ticker" }, null,
        { ticker: " " }, { ticker: "TRV" }],
      notes: ["Peers suggested from Yahoo's Insurance list; review them.", 3],
      source: "Yahoo Finance industry list" }), { status: 200 });
  };
  const res = await fetchPeerSuggestions("BRK.B");
  assert.equal(url, "/api/peers/BRK.B");
  assert.deepEqual(res.peers.map((p) => p.ticker), ["CB", "TRV"]);
  assert.deepEqual(res.notes, ["Peers suggested from Yahoo's Insurance list; review them."]);
  assert.equal(res.industry, "Insurance");
  // A refused ticker surfaces the backend's message.
  globalThis.fetch = async () => new Response(JSON.stringify({ detail: "Invalid ticker." }), { status: 400 });
  await assert.rejects(fetchPeerSuggestions("BAD TICKER!"), /Invalid ticker/);
});

test("a refused valuation carries the backend's message and its HTTP status", async () => {
  globalThis.window = undefined;
  const answer = (status, detail) => async () =>
    new Response(JSON.stringify({ detail }), { status });
  globalThis.fetch = answer(404,
    "No market data for ZZZZQX; check the symbol, or try again shortly if the "
    + "market-data service is unreachable.");
  const missing = await fetchValuation("ZZZZQX").then(() => null, (e) => e);
  assert.ok(missing instanceof Error);
  assert.equal(missing.status, 404);
  assert.equal(missing.message, "No market data for ZZZZQX; check the symbol, or try again "
    + "shortly if the market-data service is unreachable.");
  globalThis.fetch = answer(400, "Invalid ticker.");
  const invalid = await fetchValuation("AAPL$").then(() => null, (e) => e);
  assert.equal(invalid.status, 400);
  assert.equal(invalid.message, "Invalid ticker.");
});
