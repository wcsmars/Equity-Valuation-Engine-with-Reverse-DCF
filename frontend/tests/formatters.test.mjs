// Primitive formatters (lib/format.ts): signs, rounding to zero, compact
// units, currency symbols, dates, tone bands, price decimals and the research
// log entry for a digest (and the upgrade of logs saved before it carried the
// digest's facts). Run with: npm test

// West of UTC, so a date-only string parsed as UTC midnight would show the
// previous day. Each test file runs in its own process, and nothing below
// reads a date before this line.
process.env.TZ = "America/Los_Angeles";

import assert from "node:assert/strict";
import { test } from "node:test";
import { loadLib } from "./load-lib.mjs";

const {
  appendLogEntry,
  digestLogEntry,
  fmtBig,
  fmtCount,
  fmtDate,
  fmtMoney,
  fmtMult,
  fmtNum,
  fmtPct,
  priceDecimals,
  toneForUpside,
  upgradeDigestLog,
  withFullStop,
} = loadLib("format");

test("money puts the minus sign before the symbol and snaps tiny negatives to zero", () => {
  assert.equal(fmtMoney(-36.5), "-$36.50");
  assert.equal(fmtMoney(1234.5, "EUR"), "€1,234.50");
  assert.equal(fmtMoney(-0.001), "$0.00");
  assert.equal(fmtMoney(-0.4, "USD", 0), "$0");
  assert.equal(fmtMoney(33.74, "SEK"), "SEK 33.74");
  assert.equal(fmtMoney(null), "—");
  assert.equal(fmtMoney(Number.NaN), "—");
});

test("yuan and yen amounts stay distinguishable", () => {
  assert.equal(fmtMoney(1234.5, "JPY"), "¥1,234.50");
  assert.equal(fmtMoney(1234.5, "CNY"), "CN¥1,234.50");
  assert.notEqual(fmtBig(2e9, "CNY"), fmtBig(2e9, "JPY"));
});

test("compact amounts pick the unit after rounding", () => {
  assert.equal(fmtBig(999.96e9), "$1.0T");
  assert.equal(fmtBig(391e9), "$391.0B");
  assert.equal(fmtBig(-36.5e9), "-$36.5B");
  assert.equal(fmtBig(540e6, "HKD"), "HK$540.0M");
  assert.equal(fmtBig(-0.4), "$0");
  assert.equal(fmtCount(7.4e9), "7.4B");
  assert.equal(fmtCount(undefined), "—");
});

test("percentages, numbers and multiples never show a signed zero", () => {
  assert.equal(fmtPct(0.123, { signed: true }), "+12.3%");
  assert.equal(fmtPct(-0.207), "-20.7%");
  assert.equal(fmtPct(-0.0001, { signed: true }), "0.0%");
  assert.equal(fmtPct(0.0001, { signed: true }), "0.0%");
  assert.equal(fmtPct(0.0254, { decimals: 2 }), "2.54%");
  assert.equal(fmtNum(1234.567), "1,234.57");
  assert.equal(fmtNum(-0.001), "0.00");
  assert.equal(fmtMult(12.34), "12.3x");
  assert.equal(fmtMult(-0.04), "0.0x");
});

test("a date-only string keeps its calendar day west of UTC", () => {
  // The zone above is in effect: UTC midnight on Dec 31 is Dec 30 here.
  assert.equal(new Date(Date.UTC(2024, 11, 31)).getDate(), 30);
  assert.equal(fmtDate("2024-12-31"), "Dec 31, 2024");
  assert.equal(fmtDate("2024-01-01"), "Jan 1, 2024");
  // Impossible dates and unparseable text come back as given.
  assert.equal(fmtDate("2024-02-30"), "2024-02-30");
  assert.equal(fmtDate("not a date"), "not a date");
  assert.equal(fmtDate(""), "");
  assert.equal(fmtDate(null), "");
});

test("upside tone bands switch at plus or minus 15 percent", () => {
  assert.equal(toneForUpside(0.15), "text-up");
  assert.equal(toneForUpside(0.149), "text-flat");
  assert.equal(toneForUpside(-0.149), "text-flat");
  assert.equal(toneForUpside(-0.15), "text-down");
  assert.equal(toneForUpside(null), "text-ink-dim");
  assert.equal(toneForUpside(0.16), "text-up");
  assert.equal(toneForUpside(0.14), "text-flat");
  assert.equal(toneForUpside(-0.16), "text-down");
  assert.equal(toneForUpside(-0.14), "text-flat");
});

test("the bands take float noise as the engine's verdict does (1e-12)", () => {
  // A target at 1.15x or 0.85x the price: 0.15 and -0.15 up to float noise,
  // which engine._recommendation reads as on the band (math.isclose).
  assert.notEqual(115 / 100 - 1, 0.15);
  assert.equal(toneForUpside(115 / 100 - 1), "text-up");
  assert.equal(toneForUpside(85 / 100 - 1), "text-down");
  assert.equal(toneForUpside(0.15 - 1e-13), "text-up");
  assert.equal(toneForUpside(-(0.15 - 1e-13)), "text-down");
  // Beyond the tolerance it is fairly valued.
  assert.equal(toneForUpside(0.15 - 1e-9), "text-flat");
  assert.equal(toneForUpside(-0.15 + 1e-9), "text-flat");
});

test("share prices keep their cents unless they run into the thousands", () => {
  assert.equal(priceDecimals(9.25), 2);
  assert.equal(priceDecimals(343.99), 2);
  assert.equal(priceDecimals(1500), 0);
  assert.equal(priceDecimals(-2500), 0);
  assert.equal(priceDecimals(null), 2);
  // A sub-$10 stock's neighbouring grid values stay apart.
  const cells = [4.13, 4.51, 4.98].map((p) => fmtMoney(p, "USD", priceDecimals(9.25)));
  assert.deepEqual(cells, ["$4.13", "$4.51", "$4.98"]);
});

test("a message gets exactly one closing full stop", () => {
  assert.equal(withFullStop("Trim the research log and try again."), "Trim the research log and try again.");
  assert.equal(withFullStop("Failed to fetch"), "Failed to fetch.");
  assert.equal(withFullStop("Failed to fetch  "), "Failed to fetch.");
  assert.equal(withFullStop("Really?"), "Really?");
  assert.equal(withFullStop("Waiting…"), "Waiting…");
});

test("a digest's research-log entry keeps its cited facts, risks and catalysts", () => {
  const digest = {
    summary: "Margins expanded on mix.",
    sentiment: "bullish",
    key_facts: [
      { fact: "Q3 gross margin 46.2%, +180bp y/y", source: "Q3 FY2026 earnings call" },
      { fact: "Services revenue +14% y/y", source: "" },
      { fact: "", source: "ignored" },
    ],
    risks: ["China demand", " "],
    catalysts: ["New product cycle"],
    suggested_assumptions: [],
  };
  assert.equal(
    digestLogEntry("Q3 call", digest),
    [
      "[Q3 call] Margins expanded on mix.",
      "- Q3 gross margin 46.2%, +180bp y/y (source: Q3 FY2026 earnings call)",
      "- Services revenue +14% y/y (source: Q3 call)",
      "- Risk: China demand",
      "- Catalyst: New product cycle",
    ].join("\n")
  );
  // A digest without facts (or with malformed fields) still gives its line.
  assert.equal(digestLogEntry("Pasted material", { summary: "Short." }), "[Pasted material] Short.");
  assert.equal(
    digestLogEntry("x", { summary: "S", key_facts: null, risks: "r", catalysts: [1] }),
    "[x] S"
  );
  // Entries are separate paragraphs of the log.
  assert.equal(appendLogEntry("", "a"), "a");
  assert.equal(appendLogEntry("old", "a"), "old\n\na");
});

test("a log saved with summary-only digest lines gets the saved digests' facts back", () => {
  const q3 = { source: "Q3 call", at: "2026-10-01T00:00:00Z", digest: {
    summary: "Margins expanded on mix.",
    key_facts: [{ fact: "Q3 gross margin 46.2%", source: "Q3 FY2026 earnings call" }],
    risks: ["China demand"], catalysts: [],
  } };
  const tenK = { source: "10-K 2025", at: "2026-10-02T00:00:00Z", digest: {
    summary: "Debt maturities are spread out.", key_facts: [{ fact: "No maturity before 2028" }],
  } };
  const pasted = { source: "Pasted material", at: "2026-10-03T00:00:00Z", digest: { summary: "Short." } };
  // The old entries were "[source] summary" paragraphs; the user also wrote
  // notes and edited the 10-K line.
  const old = [
    "My view: services mix.",
    "[Q3 call] Margins expanded on mix.",
    "[10-K 2025] Debt maturities are spread out. (checked)",
    "[Pasted material] Short.",
  ].join("\n\n");
  const upgraded = upgradeDigestLog(old, [q3, tenK, pasted]);
  assert.equal(upgraded, [
    "My view: services mix.",
    "[Q3 call] Margins expanded on mix.\n- Q3 gross margin 46.2% (source: Q3 FY2026 earnings call)\n- Risk: China demand",
    "[10-K 2025] Debt maturities are spread out. (checked)",
    "[Pasted material] Short.",
  ].join("\n\n"));
  // Upgrading again, or a log written with the full entries, changes nothing.
  assert.equal(upgradeDigestLog(upgraded, [q3, tenK, pasted]), upgraded);
  // A removed line is not put back; no or malformed digests leave the log.
  assert.equal(upgradeDigestLog("My view.", [q3]), "My view.");
  assert.equal(upgradeDigestLog(old, undefined), old);
  assert.equal(upgradeDigestLog(old, [{ source: 3, digest: null }, null]), old);
  assert.equal(upgradeDigestLog("", [q3]), "");
});
