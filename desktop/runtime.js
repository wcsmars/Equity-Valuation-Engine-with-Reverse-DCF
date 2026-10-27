const fs = require("fs");
const path = require("path");
const http = require("http");
const { spawn } = require("child_process");

// Minimal .env parser so the AI/FMP keys reach the backend even when launched
// from Finder (no shell environment). It reads typical lines the way
// run_dev.sh's `source .env` does: an optional `export ` prefix, single- or
// double-quoted values, and trailing ` # comments` on unquoted values. It does
// not expand variables or read multi-line values.
function loadDotenv(file) {
  const out = {};
  try {
    for (const line of fs.readFileSync(file, "utf8").split("\n")) {
      const t = line.trim().replace(/^export\s+/, "");
      if (!t || t.startsWith("#")) continue;
      const eq = t.indexOf("=");
      if (eq === -1) continue;
      const k = t.slice(0, eq).trim();
      const raw = t.slice(eq + 1);
      const dq = raw.trim().match(/^"((?:[^"\\]|\\.)*)"/);
      const sq = raw.trim().match(/^'([^']*)'/);
      let v;
      if (dq) v = dq[1].replace(/\\(["\\$`])/g, "$1");
      else if (sq) v = sq[1];
      else v = raw.replace(/\s+#.*$/, "").trim();
      if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(k)) out[k] = v;
    }
  } catch (_) {
    /* no .env is fine */
  }
  return out;
}

// Resolves only on a real 2xx/3xx response; keeps retrying on connection
// refused or 5xx until the timeout. Used to confirm a server is actually
// READY (not merely that the port eventually answered with an error). A
// timeout names the last probe result (an HTTP status or a connection error),
// which separates "never listening" from "listening but failing".
function waitForHttp(url, timeoutMs) {
  return new Promise((resolve, reject) => {
    let request;
    let timer;
    let finished = false;
    let last = "no response yet";
    const finish = (error) => {
      if (finished) return;
      finished = true;
      clearTimeout(deadline);
      clearTimeout(timer);
      request?.destroy();
      if (error) reject(error);
      else resolve(true);
    };
    const deadline = setTimeout(
      () => finish(new Error(`Timed out waiting for ${url} (last result: ${last})`)),
      timeoutMs
    );
    const retry = () => {
      if (finished) return;
      clearTimeout(timer);
      timer = setTimeout(tryOnce, 400);
    };
    const tryOnce = () => {
      if (finished) return;
      request = http.get(url, (res) => {
        const ok = res.statusCode >= 200 && res.statusCode < 400;
        res.resume();
        if (ok) finish();
        else {
          last = `HTTP ${res.statusCode}`;
          retry();
        }
      });
      request.on("error", (error) => {
        if (!finished) last = error.code || error.message;
        retry();
      });
      request.setTimeout(2500, () => request.destroy(new Error("no reply within 2.5 s")));
    };
    tryOnce();
  });
}

// A server that stays up but never turns healthy fails only by timeout. Attach
// the recent server output to that error, as an early exit already does, so the
// dialog shows what the server printed instead of just the URL.
function withLogTail(promise, logTail) {
  return promise.catch((error) => {
    const tail = String(logTail() || "").trim();
    const message = String(error && error.message ? error.message : error);
    throw new Error(tail ? `${message}\n\nRecent output:\n${tail.slice(-1200)}` : message);
  });
}

// foo.docx -> foo (2).docx if needed, so repeated exports never overwrite.
function uniquePath(p, reserved = new Set()) {
  const taken = (candidate) => fs.existsSync(candidate) || reserved.has(candidate);
  if (!taken(p)) return p;
  const ext = path.extname(p);
  const base = p.slice(0, -ext.length || undefined);
  for (let i = 2; ; i++) {
    const cand = `${base} (${i})${ext}`;
    if (!taken(cand)) return cand;
  }
}

// Rejects as soon as a child process dies, so a crashed server surfaces its
// log tail immediately instead of a 3-minute spinner + timeout.
function watchChildExit(name, proc, logTail) {
  return new Promise((_resolve, reject) => {
    proc.once("error", (error) => reject(new Error(`${name} couldn't start: ${error.message}`)));
    proc.on("exit", (code) =>
      reject(
        new Error(
          `${name} exited during startup (code ${code}).\n\nRecent output:\n` +
            logTail().slice(-1200)
        )
      )
    );
  });
}

// The message a failed startup should report, or null when there is nothing to
// report: quitting while the servers are still starting kills them and rejects
// the startup promise too, and that is not a failure.
function startupFailure(error, quitting) {
  if (quitting) return null;
  return String(error && error.message ? error.message : error);
}

// `next build` writes BUILD_ID before static generation and writes these two
// files only when it finishes. A failed or interrupted build (for example one
// stopped by quitting the app) therefore leaves BUILD_ID without them, and
// `next start` cannot serve that. `next dev` removes BUILD_ID too.
const COMPLETE_BUILD_FILES = ["BUILD_ID", "prerender-manifest.json", "export-marker.json"];

// A production bundle can remain present after code or dependencies change.
// Rebuild when the build is incomplete or any input is newer, not just after
// `next dev` removes BUILD_ID.
function productionBuildIsCurrent(frontendDir) {
  const distDir = path.join(frontendDir, ".next");
  if (!COMPLETE_BUILD_FILES.every((name) => fs.existsSync(path.join(distDir, name)))) return false;
  const builtAt = fs.statSync(path.join(distDir, "BUILD_ID")).mtimeMs;
  const changed = (file) => {
    if (!fs.existsSync(file)) return false;
    const stat = fs.statSync(file);
    if (stat.isDirectory()) return stat.mtimeMs > builtAt ||
      fs.readdirSync(file).some((name) => changed(path.join(file, name)));
    return stat.mtimeMs > builtAt;
  };
  return !["app", "components", "lib", "public", "package.json", "package-lock.json",
    "next.config.mjs", "postcss.config.mjs", "tailwind.config.ts", "tsconfig.json"]
    .some((name) => changed(path.join(frontendDir, name)));
}

// The servers run in their own process groups and are stopped by the app's quit
// handlers, which never run when the app is force-quit, killed with SIGKILL or
// crashes. The reaper covers that case: a small Node process in its own session
// that reads "+<pgid>" (watch) and "-<pgid>" (forget) lines on stdin. The
// kernel closes that pipe however the app exits, and on end-of-file the reaper
// sends SIGTERM to every group it still watches. Inline code (`-e`) runs under
// a system Node and under Electron's Node alike; a script file would sit inside
// app.asar in a packaged app, where a system Node cannot read it.
const REAPER_SOURCE = `
const groups = new Set();
let pending = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  const lines = (pending + chunk).split("\\n");
  pending = lines.pop();
  for (const line of lines) {
    const id = Number(line.slice(1));
    if (!Number.isInteger(id) || id <= 1) continue;
    if (line[0] === "+") groups.add(id);
    else if (line[0] === "-") groups.delete(id);
  }
});
const reap = () => {
  for (const id of groups) {
    try { process.kill(-id, "SIGTERM"); } catch (_) {}
  }
  process.exit(0);
};
process.stdin.on("end", reap);
process.stdin.on("error", reap);
`;

// Starts the reaper with `nodeBin` (env must make it run as Node). guard(pid)
// watches the process group a detached child leads; release(pid) forgets it
// once that child has exited, so a reused ID is never signalled; stop() ends
// the reaper without signalling anything, for a normal quit that has already
// stopped the servers. Process groups do not exist on Windows, so there the
// returned functions do nothing.
function startReaper(nodeBin, env) {
  const inert = { guard() {}, release() {}, stop() {} };
  if (process.platform === "win32") return inert;
  let proc;
  try {
    proc = spawn(nodeBin, ["-e", REAPER_SOURCE], {
      env,
      detached: true,
      stdio: ["pipe", "ignore", "ignore"],
    });
  } catch (_) {
    return inert;
  }
  proc.on("error", () => {});
  proc.stdin.on("error", () => {});
  const send = (line) => {
    if (proc.exitCode === null && proc.signalCode === null && proc.stdin.writable) {
      proc.stdin.write(line + "\n");
    }
  };
  return {
    pid: proc.pid,
    guard: (pid) => pid && send("+" + pid),
    release: (pid) => pid && send("-" + pid),
    stop: () => {
      try {
        proc.kill("SIGKILL");
      } catch (_) {
        /* already gone */
      }
    },
  };
}

module.exports = { loadDotenv, waitForHttp, withLogTail, uniquePath, watchChildExit,
  startupFailure, productionBuildIsCurrent, startReaper };
