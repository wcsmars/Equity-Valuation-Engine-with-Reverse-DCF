// Client-side request-size checks: refused before anything is uploaded.
// Run with: npm test

import assert from "node:assert/strict";
import { afterEach, beforeEach, test } from "node:test";
import { loadLib } from "./load-lib.mjs";

const api = loadLib("api");
const { MAX_UPLOAD_BYTES, base64Bytes, planPdfAttachments } = api;

const MB = 1_000_000;
const pdf = (name, size) => ({ name, size, type: "application/pdf" });

let calls;
let realFetch;
beforeEach(() => {
  calls = [];
  realFetch = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    calls.push({ url, bytes: init?.body?.length ?? 0 });
    return new Response(JSON.stringify({ reply: "ok" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };
});
afterEach(() => {
  globalThis.fetch = realFetch;
});

test("the upload cap sits below the 50 MiB proxy and backend caps", () => {
  assert.equal(MAX_UPLOAD_BYTES, 50 * MB);
  assert.ok(MAX_UPLOAD_BYTES < 50 * 1024 * 1024);
  assert.equal(base64Bytes(3), 4);
  assert.equal(base64Bytes(4), 8);
});

test("picked PDFs that would push the request over the cap are refused", () => {
  const used = 1 * MB; // the model, research log and pasted text
  const { accepted, refused } = planPdfAttachments(
    [pdf("a.pdf", 20 * MB), pdf("big.pdf", 20 * MB), pdf("b.pdf", 5 * MB)],
    used
  );
  // 20 MB -> 26.7 MB encoded fits; a second 20 MB would not; 5 MB still fits.
  assert.deepEqual(accepted.map((f) => f.name), ["a.pdf", "b.pdf"]);
  assert.equal(refused.length, 1);
  assert.match(refused[0], /"big\.pdf" \(20\.0 MB\) was not attached/);
  assert.match(refused[0], /limited to 50\.0 MB/);
  assert.match(refused[0], /about 16\.\d MB of PDF can still be attached/);
});

test("non-PDF files are refused; a .pdf name without a MIME type is accepted", () => {
  const { accepted, refused } = planPdfAttachments(
    [
      { name: "notes.txt", size: 10, type: "text/plain" },
      { name: "Report.PDF", size: 10, type: "" },
    ],
    0
  );
  assert.deepEqual(accepted.map((f) => f.name), ["Report.PDF"]);
  assert.deepEqual(refused, ['"notes.txt" is not a PDF.']);
});

test("an oversized AI request fails before it is sent", async () => {
  const pdfs = [{ name: "x.pdf", data_base64: "A".repeat(MAX_UPLOAD_BYTES) }];
  await assert.rejects(
    api.postDigest({ report: null, pdfs }),
    /over the 50\.0 MB limit\. Remove or shrink attached PDFs/
  );
  assert.equal(calls.length, 0);
  // Just under the cap is sent.
  const small = [{ name: "x.pdf", data_base64: "A".repeat(MAX_UPLOAD_BYTES - 1000) }];
  await api.postChat({ report: null, turns: [], pdfs: small });
  assert.equal(calls.length, 1);
  assert.ok(calls[0].bytes <= MAX_UPLOAD_BYTES);
});

test("other routes keep the smaller JSON cap", async () => {
  const notes = "n".repeat(8 * MB);
  await assert.rejects(
    api.saveResearchState("SYNT", { notes }),
    /over the 8\.0 MB limit\. Trim the research log/
  );
  assert.equal(calls.length, 0);
});
