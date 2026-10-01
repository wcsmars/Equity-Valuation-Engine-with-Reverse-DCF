import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import { loadLib } from "./load-lib.mjs";
const { apiBase, errorDetail } = loadLib("api");
const originalWindow = globalThis.window;
afterEach(() => { globalThis.window = originalWindow; });

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
