const fs = require("fs");
const path = require("path");
const http = require("http");

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
// READY (not merely that the port eventually answered with an error).
function waitForHttp(url, timeoutMs) {
  return new Promise((resolve, reject) => {
    let request;
    let timer;
    let finished = false;
    const finish = (error) => {
      if (finished) return;
      finished = true;
      clearTimeout(deadline);
      clearTimeout(timer);
      request?.destroy();
      if (error) reject(error);
      else resolve(true);
    };
    const deadline = setTimeout(() => finish(new Error("Timed out waiting for " + url)), timeoutMs);
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
        else retry();
      });
      request.on("error", retry);
      request.setTimeout(2500, () => request.destroy());
    };
    tryOnce();
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

module.exports = { loadDotenv, waitForHttp, uniquePath, watchChildExit };
