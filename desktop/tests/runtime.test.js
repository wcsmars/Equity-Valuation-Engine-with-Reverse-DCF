const assert = require("node:assert/strict");
const { test } = require("node:test");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const http = require("node:http");
const { EventEmitter } = require("node:events");
const { loadDotenv, uniquePath, waitForHttp, watchChildExit } = require("../runtime");

test("dotenv accepts shell-style literal keys without executing expressions", (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "equity-desktop-env-"));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const file = path.join(dir, ".env");
  fs.writeFileSync(file, `# ignored\nexport A='hello # world'\nB="quote\\\"d"\nC=plain # comment\nINVALID KEY=no\nD=$(echo no-execution)\n`);
  assert.deepEqual(loadDotenv(file), { A: "hello # world", B: 'quote"d', C: "plain", D: "$(echo no-execution)" });
  assert.deepEqual(loadDotenv(path.join(dir, "missing")), {});
});

test("exports stay unique beyond 99 copies and while another download is pending", (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "equity-desktop-download-"));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const first = path.join(dir, "valuation.xlsx");
  fs.writeFileSync(first, "original");
  for (let n = 2; n <= 100; n++) fs.writeFileSync(path.join(dir, `valuation (${n}).xlsx`), "");
  const next = uniquePath(first);
  assert.equal(next, path.join(dir, "valuation (101).xlsx"));
  assert.equal(uniquePath(first, new Set([next])), path.join(dir, "valuation (102).xlsx"));
  assert.equal(fs.readFileSync(first, "utf8"), "original");
  assert.equal(uniquePath(path.join(dir, "extensionless"), new Set([path.join(dir, "extensionless")])), path.join(dir, "extensionless (2)"));
});

async function serverFor(t, handler) {
  const server = http.createServer(handler);
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => { server.closeAllConnections(); server.close(); });
  return `http://127.0.0.1:${server.address().port}/`;
}

test("startup waits through error responses before declaring readiness", async (t) => {
  let calls = 0;
  const url = await serverFor(t, (_req, res) => {
    res.statusCode = ++calls === 1 ? 503 : 200;
    res.end();
  });
  assert.equal(await waitForHttp(url, 2000), true);
  assert.equal(calls, 2);
});

test("startup deadline aborts a server that accepts a connection but never replies", async (t) => {
  const url = await serverFor(t, () => {});
  const start = Date.now();
  await assert.rejects(waitForHttp(url, 80), /Timed out waiting/);
  assert.ok(Date.now() - start < 1500);
});

test("spawn failures reject readiness immediately instead of becoming uncaught process errors", async () => {
  const proc = new EventEmitter();
  const failed = watchChildExit("Backend", proc, () => "last log line");
  proc.emit("error", new Error("spawn EACCES"));
  await assert.rejects(failed, /Backend couldn't start: spawn EACCES/);
});

test("early child exit includes its startup diagnostics", async () => {
  const proc = new EventEmitter();
  const failed = watchChildExit("Frontend", proc, () => "build missing");
  proc.emit("exit", 1);
  await assert.rejects(failed, /Frontend exited during startup.*1[\s\S]*build missing/);
});
