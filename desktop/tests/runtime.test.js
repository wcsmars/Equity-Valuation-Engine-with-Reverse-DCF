const assert = require("node:assert/strict");
const { test } = require("node:test");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const http = require("node:http");
const { EventEmitter } = require("node:events");
const { spawn } = require("node:child_process");
const { loadDotenv, uniquePath, waitForHttp, withLogTail, watchChildExit, startupFailure, productionBuildIsCurrent,
  startReaper } = require("../runtime");

test("desktop rebuilds stale source and dependencies even while BUILD_ID exists", (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "equity-desktop-build-"));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  assert.equal(productionBuildIsCurrent(dir), false);
  fs.mkdirSync(path.join(dir, ".next"));
  fs.mkdirSync(path.join(dir, "app"));
  const marker = path.join(dir, ".next", "BUILD_ID");
  const source = path.join(dir, "app", "page.tsx");
  fs.writeFileSync(source, "old source");
  fs.utimesSync(source, 1000, 1000);
  fs.utimesSync(path.join(dir, "app"), 1000, 1000);
  fs.writeFileSync(path.join(dir, ".next", "prerender-manifest.json"), "{}");
  fs.writeFileSync(path.join(dir, ".next", "export-marker.json"), "{}");
  fs.writeFileSync(marker, "production");
  fs.utimesSync(marker, 2000, 2000);
  assert.equal(productionBuildIsCurrent(dir), true);
  fs.utimesSync(source, 3000, 3000);
  assert.equal(productionBuildIsCurrent(dir), false);
  fs.utimesSync(source, 1000, 1000);
  fs.unlinkSync(source);
  assert.equal(productionBuildIsCurrent(dir), false);
  fs.utimesSync(path.join(dir, "app"), 1000, 1000);
  fs.writeFileSync(path.join(dir, "package-lock.json"), "{}");
  assert.equal(productionBuildIsCurrent(dir), false);
});

test("a failed or interrupted build that left only BUILD_ID is rebuilt", (t) => {
  // `next build` writes BUILD_ID before static generation and the other two
  // files only at the end, so BUILD_ID alone means the build never finished.
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "equity-desktop-partial-"));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const dist = path.join(dir, ".next");
  fs.mkdirSync(dist);
  fs.writeFileSync(path.join(dist, "BUILD_ID"), "production");
  assert.equal(productionBuildIsCurrent(dir), false);
  fs.writeFileSync(path.join(dist, "prerender-manifest.json"), "{}");
  assert.equal(productionBuildIsCurrent(dir), false);
  fs.writeFileSync(path.join(dist, "export-marker.json"), "{}");
  assert.equal(productionBuildIsCurrent(dir), true);
  fs.unlinkSync(path.join(dist, "prerender-manifest.json"));
  assert.equal(productionBuildIsCurrent(dir), false);
});

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
  await assert.rejects(waitForHttp(url, 80), /Timed out waiting.*\(last result: no response yet\)/);
  assert.ok(Date.now() - start < 1500);
});

test("a server that stays up but never turns healthy times out with its status and recent output", async (t) => {
  const url = await serverFor(t, (_req, res) => {
    res.statusCode = 503;
    res.end();
  });
  const tail = () => "[backend] RuntimeError: store file is locked by another process";
  const error = await withLogTail(waitForHttp(url, 300), tail).then(() => null, (e) => e);
  assert.match(error.message, /^Timed out waiting for http:\/\/127\.0\.0\.1:\d+\/ \(last result: HTTP 503\)/);
  assert.match(error.message, /\n\nRecent output:\n\[backend\] RuntimeError: store file is locked/);
  // Nothing captured: keep the bare message rather than an empty section.
  await assert.rejects(withLogTail(Promise.reject(new Error("Timed out")), () => " \n"), (e) => e.message === "Timed out");
  assert.equal(await withLogTail(Promise.resolve(true), tail), true);
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

test("a startup failure is reported, but quitting during startup is not", () => {
  const timeout = new Error("Timed out waiting for http://127.0.0.1:1/api/health");
  assert.equal(startupFailure(timeout, false), timeout.message);
  assert.equal(startupFailure("spawn EACCES", false), "spawn EACCES");
  assert.equal(startupFailure(timeout, true), null);
});

const isAlive = (pid) => {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return error.code === "EPERM";
  }
};

async function waitFor(condition, timeoutMs) {
  const end = Date.now() + timeoutMs;
  while (!condition() && Date.now() < end) await new Promise((resolve) => setTimeout(resolve, 50));
  return condition();
}

test("servers are stopped when the app process is killed outright", { skip: process.platform === "win32" }, async (t) => {
  // A stand-in for the app: it starts detached "servers" the way main.js does,
  // registers them with reapers, then is SIGKILLed so no quit handler runs.
  const app = `
    const { spawn } = require("child_process");
    const { startReaper } = require(${JSON.stringify(path.join(__dirname, "..", "runtime.js"))});
    const server = () => spawn(process.execPath, ["-e", "setInterval(() => {}, 1000)"], { detached: true });
    const reaper = startReaper(process.execPath, process.env);
    const stopped = startReaper(process.execPath, process.env);
    const killed = server(), released = server(), spared = server();
    reaper.guard(killed.pid);
    reaper.guard(released.pid);
    reaper.release(released.pid);   // e.g. a finished build
    stopped.guard(spared.pid);
    stopped.stop();                 // a normal quit that already stopped it
    console.log(JSON.stringify({ reapers: [reaper.pid, stopped.pid],
      killed: killed.pid, released: released.pid, spared: spared.pid }));
    setInterval(() => {}, 1000);
  `;
  const parent = spawn(process.execPath, ["-e", app], { stdio: ["ignore", "pipe", "inherit"] });
  let pids = null;
  t.after(() => {
    for (const pid of [parent.pid, ...(pids ? [pids.killed, pids.released, pids.spared, ...pids.reapers] : [])]) {
      try {
        process.kill(pid, "SIGKILL");
      } catch (_) {
        /* already gone */
      }
    }
  });
  pids = await new Promise((resolve, reject) => {
    let out = "";
    parent.stdout.on("data", (chunk) => {
      out += chunk;
      if (out.includes("\n")) resolve(JSON.parse(out.split("\n")[0]));
    });
    parent.once("exit", (code) => reject(new Error(`stand-in app exited early (code ${code})`)));
  });
  assert.ok(isAlive(pids.killed) && isAlive(pids.released) && isAlive(pids.spared));

  parent.kill("SIGKILL");
  assert.ok(await waitFor(() => !isAlive(pids.killed), 10000), "guarded server outlived the killed app");
  assert.ok(await waitFor(() => pids.reapers.every((pid) => !isAlive(pid)), 10000), "reaper did not exit");
  await new Promise((resolve) => setTimeout(resolve, 300)); // let any stray SIGTERM land
  assert.ok(isAlive(pids.released), "a released process group was signalled");
  assert.ok(isAlive(pids.spared), "a stopped reaper still signalled its group");
});
