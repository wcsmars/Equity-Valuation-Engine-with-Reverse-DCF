// Summary formatting (lib/format.ts): a missing blended target or a withheld
// upside, methods left out of the blend, and the adjusted beta.
// Run with: npm test

import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { loadLib } from "./load-lib.mjs";

const {
  BLUME_MARKET_WEIGHT,
  BLUME_RAW_WEIGHT,
  COMPS_MIN_PEER_VALUES,
  DDM_METHOD_LABELS,
  EQUITY_MULTIPLES_ONLY_KINDS,
  FEW_PEER_VALUES_NOTE_PREFIX,
  NO_TARGET_SHORT,
  NO_TARGET_SUPPLY_PEERS,
  betaStat,
  compsWarnings,
  ddmMethodLabel,
  equityMultiplesOnly,
  excludedReason,
  fmtBlendedTarget,
  fmtBlendedUpside,
  fmtRate,
  footballFieldTone,
  needsPeers,
  peerStatusReason,
  peerStatusTone,
  projectionYearLabels,
  setsTextColor,
  toneForBlendedUpside,
  toneForMethodUpside,
  uiText,
  uiWarnings,
  verdictExplanation,
  watchlistUpside,
} = loadLib("format");

const demo = {
  currency: "USD",
  current_price: 40.84,
  methods: { DCF: 33.38, "Comps (median)": 42.88, DDM: 10.99, FCFE: 31.41 },
  blended_target: 32.4,
  blended_upside: -0.207,
  recommendation: "Overvalued",
  excluded_from_blend: {},
  financial_institution: null,
  financial_kind: null,
};

// The engine's summaries (equity_valuation.value_company on the synthetic
// company) for the kinds that change the blend, with its own reason strings.
const LESSOR = "not meaningful for a debt-funded lessor";
const LOW_PAYOUT = "dividends only (30% of net income); buybacks ignored";
// A lessor without peers (as AER): every method is reference only, no target.
const lessor = {
  ...demo,
  methods: { DCF: 33.38, DDM: 10.99, FCFE: 31.41 },
  blended_target: null,
  blended_upside: null,
  recommendation: "N/A",
  excluded_from_blend: { DCF: LESSOR, DDM: LOW_PAYOUT, FCFE: LESSOR },
  financial_institution: "operating lessor per its filings",
  financial_kind: "lessor",
};
// A bank without peers (as BAC): the blend is the DDM alone, so the target is
// kept but the verdict and upside are withheld.
const BANK = "not meaningful for a financial institution";
const bank = {
  ...demo,
  methods: { DCF: 33.38, DDM: 10.99, FCFE: 31.41 },
  blended_target: 10.99,
  blended_upside: null,
  recommendation: "N/A",
  excluded_from_blend: { DCF: BANK, FCFE: BANK },
  financial_institution: "bank",
  financial_kind: "bank",
};

test("a blended target and its upside are shown and coloured", () => {
  assert.equal(fmtBlendedTarget(demo), "$32.40");
  assert.equal(fmtBlendedUpside(demo), "-20.7%");
  assert.equal(toneForBlendedUpside(demo), "text-down");
  assert.equal(needsPeers(demo), false);
});

test("no target without peers asks for them, uncoloured", () => {
  // Comps did not run (null) or found no usable peer (an empty list).
  for (const comps of [null, undefined, { peers: [] }]) {
    assert.equal(needsPeers(lessor, comps), true);
    assert.equal(fmtBlendedTarget(lessor, comps), NO_TARGET_SUPPLY_PEERS);
  }
  // The same text as the Python outputs (tests/test_reports_cli.py pins both).
  assert.equal(NO_TARGET_SUPPLY_PEERS, "No target (supply peers)");
  assert.equal(fmtBlendedUpside(lessor), "n/a");
  assert.equal(toneForBlendedUpside(lessor), "text-ink-dim");
  // A captive-finance group is flagged the same way.
  const captive = { ...lessor, financial_kind: "captive_finance" };
  assert.equal(fmtBlendedTarget(captive, null), NO_TARGET_SUPPLY_PEERS);
  // A narrow table cell says only "No target" (the explainer gives why).
  assert.equal(fmtBlendedTarget(captive, null, { short: true }), NO_TARGET_SHORT);
  assert.equal(NO_TARGET_SHORT, "No target");
  assert.equal(fmtBlendedTarget(demo, null, { short: true }), "$32.40");
  assert.equal(fmtBlendedTarget({ ...demo, methods: {}, blended_target: null }, null,
    { short: true }), "n/a");
});

test("no target reads n/a when peers would not help", () => {
  // Peers were supplied but comps gave no price (a lessor's peers without
  // P/E or P/B).
  assert.equal(needsPeers(lessor, { peers: [{ ticker: "P1" }] }), false);
  assert.equal(fmtBlendedTarget(lessor, { peers: [{ ticker: "P1" }] }), "n/a");
  // An ordinary company whose methods gave no valuation (0.00).
  const none = "no valuation (0.00)";
  const placeholder = {
    ...demo,
    methods: { DCF: 0, DDM: 0, FCFE: 0 },
    blended_target: null,
    blended_upside: null,
    recommendation: "N/A",
    excluded_from_blend: { DCF: none, DDM: none, FCFE: none },
  };
  assert.equal(needsPeers(placeholder, null), false);
  assert.equal(fmtBlendedTarget(placeholder, null), "n/a");
  // Comps among the methods, or no method at all.
  const withComps = { ...demo, blended_target: null, blended_upside: null };
  assert.equal(fmtBlendedTarget(withComps), "n/a");
  const nothing = { ...demo, methods: {}, blended_target: null, blended_upside: null };
  assert.equal(fmtBlendedTarget(nothing), "n/a");
  assert.equal(fmtBlendedUpside(nothing), "n/a");
});

test("a withheld verdict keeps the target but not the upside", () => {
  // A bank whose blend rests on the DDM alone: target shown, upside n/a.
  assert.equal(fmtBlendedTarget(bank, null), "$10.99");
  assert.equal(fmtBlendedUpside(bank), "n/a");
  assert.equal(toneForBlendedUpside(bank), "text-ink-dim");
});

test("method upsides are uncoloured when out of the blend or with no verdict", () => {
  const up = (s, name) => s.methods[name] / s.current_price - 1;
  // The bank's DDM is the whole blend, but its -73% is the very figure the
  // engine withholds; its reference-only DCF and FCFE are dim too.
  for (const name of ["DCF", "DDM", "FCFE"]) {
    assert.equal(toneForMethodUpside(bank, name, up(bank, name)), "text-ink-dim", name);
  }
  // The lessor's reference-only methods (as AER's +128% DCF).
  for (const name of ["DCF", "DDM", "FCFE"]) {
    assert.equal(toneForMethodUpside(lessor, name, 1.28), "text-ink-dim", name);
  }
  // A lessor with peers: the blend rests on comps and gives a verdict, so the
  // comps upside keeps its colour while the reference-only rows stay dim.
  const withPeers = {
    ...lessor,
    methods: { ...lessor.methods, "Comps (median)": 38.84 },
    blended_target: 38.84,
    blended_upside: -0.049,
    recommendation: "Fairly valued",
  };
  assert.equal(toneForMethodUpside(withPeers, "Comps (median)", -0.049), "text-flat");
  assert.equal(toneForMethodUpside(withPeers, "DCF", -0.18), "text-ink-dim");
  // The demo: every method is in the blend and coloured by its sign.
  assert.equal(toneForMethodUpside(demo, "DCF", up(demo, "DCF")), "text-down");
  assert.equal(toneForMethodUpside(demo, "Comps (median)", 0.05), "text-flat");
  assert.equal(toneForMethodUpside(demo, "DDM", 0.2), "text-up");
  assert.equal(toneForMethodUpside(demo, "FCFE", null), "text-ink-dim");
});

test("a watchlist card is uncoloured without a target or a verdict", () => {
  const w = { blended_target: 32.4, price: 40.84, recommendation: "Overvalued" };
  assert.ok(Math.abs(watchlistUpside(w) - (32.4 / 40.84 - 1)) < 1e-12);
  // A 0.00 target is a real -100% target.
  assert.equal(watchlistUpside({ ...w, blended_target: 0 }), -1);
  assert.equal(watchlistUpside({ ...w, blended_target: null }), null);
  assert.equal(watchlistUpside({ ...w, price: 0 }), null);
  assert.equal(watchlistUpside({ ...w, recommendation: "N/A" }), null);
});

test("methods left out of the blend are found for every flag kind", () => {
  assert.equal(excludedReason(lessor, "DCF"), LESSOR);
  assert.equal(excludedReason(lessor, "DDM"), LOW_PAYOUT);
  assert.equal(excludedReason(lessor, "FCFE"), LESSOR);
  assert.equal(excludedReason(bank, "DDM"), null);
  assert.equal(excludedReason({}, "DCF"), null);
  assert.equal(excludedReason({ excluded_from_blend: { DDM: "" } }, "DDM"), "reference only");
});

test("an adjusted beta is labelled with its raw value", () => {
  const adj = betaStat(1.815, 2.217);
  assert.equal(adj.label, "Beta (adj.)");
  assert.equal(adj.value, "1.82");
  assert.equal(adj.sub, "raw 2.22");
  assert.match(adj.title, /2\.217, Blume-adjusted toward 1 \(0\.67 × raw \+ 0\.33\) to 1\.815/);
  // The weights equity_valuation/data/market.py applies (pinned there too).
  assert.equal(BLUME_RAW_WEIGHT, 0.67);
  assert.equal(BLUME_MARKET_WEIGHT, 0.33);
  // The WACC's own record of an adjusted beta whose raw value is unknown.
  assert.equal(betaStat(1.2, null, "market data (adjusted)").label, "Beta (adj.)");
  // Not adjusted: the synthetic company, a beta given by hand, or none.
  assert.deepEqual(betaStat(1.1, null), { label: "Beta", value: "1.10" });
  assert.deepEqual(betaStat(1.1, undefined, "market data"), { label: "Beta", value: "1.10" });
  assert.deepEqual(betaStat(null, null), { label: "Beta", value: "—" });
  assert.equal(betaStat(1.0, null, "DEFAULT_BETA").label, "Beta (default)");
});

test("a table cell given a tone drops its default ink", () => {
  // Tailwind emits text-down and text-flat before text-ink, so the cell's
  // default ink must go for a negative or fair upside to show its colour.
  for (const c of ["text-down", "text-flat", "text-up", "text-ink-dim", "x text-ink-faint"]) {
    assert.equal(setsTextColor(c), true, c);
  }
  for (const c of ["", null, undefined, "text-left", "text-right", "text-[10px]",
                   "bg-surface-hi font-semibold", "max-w-[12rem] truncate"]) {
    assert.equal(setsTextColor(c), false, String(c));
  }
});

test("football-field bars are grey out of the blend or with no verdict", () => {
  const grey = "bg-ink-faint/30 border-ink-faint";
  // The demo: coloured by where the base sits against the price.
  assert.equal(footballFieldTone(demo, "DCF", 33.38, 40.84), "bg-down/30 border-down");
  assert.equal(footballFieldTone(demo, "P/E comps", 42.0, 40.84), "bg-flat/30 border-flat");
  assert.equal(footballFieldTone(demo, "EV/EBITDA comps", 45.0, 40.84), "bg-up/30 border-up");
  // A lessor with peers: the engine labels the reference-only rows.
  const withPeers = { ...lessor, blended_target: 38.84, recommendation: "Fairly valued" };
  assert.equal(footballFieldTone(withPeers, "DCF (not in blend)", 93.0, 40.84), grey);
  assert.equal(footballFieldTone(withPeers, "DDM", 10.99, 40.84), grey); // excluded by name
  assert.equal(footballFieldTone(withPeers, "P/E comps", 38.84, 40.84), "bg-flat/30 border-flat");
  // No verdict (the bank's withheld one, the lessor's missing target): all grey.
  assert.equal(footballFieldTone(bank, "DDM", 10.99, 40.84), grey);
  assert.equal(footballFieldTone(lessor, "52-week range", 40.84, 40.84), grey);
});

// --- Engine text in the dashboard (uiText, uiWarnings, compsWarnings) ------ //
// The engine's own wording (equity_valuation/engine.py and models/comps.py).
const NO_PEERS = "Comps: no usable peers found (pass --peers to supply them).";
const NO_TARGET = "No blended target: the DCF and FCFE are left out for a financial institution; "
  + "the DDM is shown for reference only, and there are no comps. Supply peers with --peers to get a target.";
const DISCOVERED = "Comps: none of the peers discovered (A, B) returned usable multiples; supply peers with --peers.";
const GAP = "Fiscal-year vs trailing gap: trailing-twelve-month net income is +60.6% vs fiscal year "
  + "2026 (an opt-in trailing basis, --comps-basis ttm in the CLI, uses Yahoo's trailing figures).";

test("uiText points the engine's CLI phrasing at the dashboard's controls", () => {
  assert.equal(uiText(NO_PEERS), "Comps: no usable peers found (add peers in the Trading comps box).");
  assert.match(uiText(NO_TARGET), /there are no comps\. Add peers in the Trading comps box to get a target\.$/);
  assert.match(uiText(DISCOVERED), /usable multiples; add peers in the Trading comps box\.$/);
  assert.match(uiText(GAP), /an opt-in trailing basis, the TTM toggle in the Trading comps box, uses/);
  for (const t of [NO_PEERS, NO_TARGET, DISCOVERED, GAP]) assert.doesNotMatch(uiText(t), /--/);
  // Anything else is unchanged, including other double dashes.
  for (const t of ["DCF: terminal growth clamped to 2.4%", "a -- b", "", "Supply peers."]) {
    assert.equal(uiText(t), t);
  }
});

test("warnings are rewritten and shown once; comps warnings skip what the comps notes say", () => {
  assert.deepEqual(uiWarnings([NO_PEERS, "DCF: x", NO_PEERS, 7]),
    ["Comps: no usable peers found (add peers in the Trading comps box).", "DCF: x"]);
  assert.deepEqual(uiWarnings(undefined), []);
  const notes = ["Peers not used: XYZNOTREAL (not found).", "PEG is shown for reference only."];
  const warnings = ["DCF: x", "Comps: Peers not used: XYZNOTREAL (not found).",
    "Comps: none of the peers supplied (XYZNOTREAL) returned usable multiples; check the tickers or supply other peers.",
    "WARNING: Comps: thin peer set", "Comps failed: timeout", NO_PEERS, NO_PEERS];
  assert.deepEqual(compsWarnings(warnings, notes), [
    "None of the peers supplied (XYZNOTREAL) returned usable multiples; check the tickers or supply other peers.",
    "WARNING: Thin peer set",
    "Comps failed: timeout",
    "No usable peers found (add peers in the Trading comps box).",
  ]);
  assert.deepEqual(compsWarnings(null, null), []);
});

test("each peer status has a tone and a reason", () => {
  assert.equal(peerStatusTone("used"), "up");
  assert.equal(peerStatusTone("not found"), "down");
  assert.equal(peerStatusTone("not fetched"), "down");
  assert.equal(peerStatusTone("no usable multiple"), "flat");
  assert.equal(peerStatusTone("is target"), "neutral");
  assert.equal(peerStatusTone("something new"), "neutral");
  assert.match(peerStatusReason("not found"), /no market data came back/);
  assert.match(peerStatusReason("no usable multiple"), /no positive EV\/EBITDA, EV\/Sales, P\/E or P\/B/);
  assert.match(peerStatusReason("is target"), /company being valued/);
  assert.equal(peerStatusReason("something new"), "Status: something new.");
});

test("rates keep a second decimal only when they need it", () => {
  assert.equal(fmtRate(0.025), "2.5%");
  assert.equal(fmtRate(0.03), "3.0%");
  assert.equal(fmtRate(0.0225), "2.25%");
  assert.equal(fmtRate(-0.0675), "-6.75%");
  assert.equal(fmtRate(null), "—");
});

// --- Why there is no verdict ---------------------------------------------- //
test("the verdict explainer is silent for an ordinary blend", () => {
  assert.equal(verdictExplanation(demo, { peers: [{}] }), null);
});

test("a bank's DDM-only blend explains the withheld verdict and asks for peers", () => {
  const why = verdictExplanation(bank, { peers: [] });
  assert.equal(why.text,
    "No verdict: the target rests on the DDM alone, which counts dividends but not buybacks. "
    + "Left out of the target: DCF and FCFE (not meaningful for a financial institution). "
    + "Peers give a P/E and P/B comps value for a bank.");
  assert.equal(why.addPeers, true);
});

test("a verdict withheld after a corporate action names the engine's reason", () => {
  const spunOff = { ...demo, blended_target: 285.21, blended_upside: null,
    recommendation: "N/A", methods: { DCF: 300, "Comps (median)": 270 },
    excluded_from_blend: {},
    verdict_withheld: "a corporate action after FY2025 (likely a spin-off)" };
  assert.deepEqual(verdictExplanation(spunOff, { peers: [{}] }), {
    text: "No verdict: Yahoo records a corporate action after FY2025 (likely a spin-off), so "
      + "the latest statements may still include a business the company no longer owns.",
    addPeers: false,
  });
  // It outranks the DDM-only sentence: the engine reports the corporate action.
  const ddmOnly = { ...bank, verdict_withheld: spunOff.verdict_withheld };
  assert.match(verdictExplanation(ddmOnly, { peers: [] }).text,
    /^No verdict: Yahoo records a corporate action after FY2025/);
  // Another engine reason is quoted; with none the line stays bare.
  assert.match(verdictExplanation({ ...spunOff, verdict_withheld: "some new reason" },
    { peers: [{}] }).text, /^No verdict: some new reason; the target is shown for reference\.$/);
  assert.equal(verdictExplanation({ ...spunOff, verdict_withheld: null },
    { peers: [{}] }).text, "No verdict.");
  // The bank's DDM-only blend keeps its own sentence with the engine's reason.
  const bankWhy = { ...bank, verdict_withheld: "the target rests on the DDM alone" };
  assert.match(verdictExplanation(bankWhy, { peers: [] }).text,
    /^No verdict: the target rests on the DDM alone, which counts dividends but not buybacks\. /);
});

test("a lessor without peers has no target and groups the reasons", () => {
  const why = verdictExplanation(lessor, null);
  assert.equal(why.text,
    "No target: every method left for a debt-funded lessor is reference only. "
    + `Left out of the target: DCF and FCFE (${LESSOR}); DDM (${LOW_PAYOUT}). `
    + "Peers give a P/E and P/B comps value for a debt-funded lessor.");
  assert.equal(why.addPeers, true);
  // With comps in the blend there is a target and nothing to add.
  const withPeers = { ...lessor, methods: { ...lessor.methods, "Comps (median)": 38.84 },
    blended_target: 38.84, blended_upside: -0.05, recommendation: "Fairly valued" };
  const ok = verdictExplanation(withPeers, { peers: [{}] });
  assert.equal(ok.addPeers, false);
  assert.match(ok.text, /^Left out of the target: DCF and FCFE/);
});

test("an ordinary company's left-out DDM is explained from the engine's reason", () => {
  const lowPayout = { ...demo, methods: { DCF: 33.38, DDM: 10.99, FCFE: 31.41 },
    excluded_from_blend: { DDM: LOW_PAYOUT } };
  assert.deepEqual(verdictExplanation(lowPayout, null), {
    text: `Left out of the target: DDM (${LOW_PAYOUT}). Peers add a trading-comps value.`,
    addPeers: true,
  });
  const none = { ...demo, methods: {}, blended_target: null, blended_upside: null,
    recommendation: "N/A" };
  assert.match(verdictExplanation(none, null).text, /^No target: no method gave a usable value\./);
});

test("a captive-finance group whose peers give no comps value is not told no method had one", () => {
  // GM with F, RIVN and TSLA: one P/E and two P/Bs, so no comps price; the
  // DCF, DDM and FCFE all have values, each reference only.
  const CAPTIVE = "not meaningful with a consolidated captive finance arm";
  const gm = { ...demo, current_price: 45.12, methods: { DCF: 264.93, DDM: 9.81, FCFE: 244.03 },
    blended_target: null, blended_upside: null, recommendation: "N/A",
    excluded_from_blend: { DCF: CAPTIVE, DDM: LOW_PAYOUT, FCFE: CAPTIVE },
    financial_institution: "consolidated captive finance arm", financial_kind: "captive_finance" };
  const peers = { peers: [{ ticker: "F" }, { ticker: "RIVN" }, { ticker: "TSLA" }],
    notes: ["Too few peer values (at least 3 needed): P/E has 1 usable value and P/B has 2 "
      + "usable values, so they give no implied price: a multiple from fewer than 3 peers is "
      + "their valuation, not a peer median."] };
  const why = verdictExplanation(gm, peers);
  assert.doesNotMatch(why.text, /no method gave a usable value/);
  assert.match(why.text, /reference only/);
  assert.equal(why.text,
    "No target: every method left for a company with a consolidated captive finance arm is "
    + "reference only, and the peers supplied give no comps value (each multiple needs at "
    + "least 3 usable peer values). "
    + `Left out of the target: DCF and FCFE (${CAPTIVE}); DDM (${LOW_PAYOUT}). `
    + "Add more peers: with enough usable peer values, comps give a P/E and P/B value for a "
    + "company with a consolidated captive finance arm.");
  assert.equal(why.addPeers, true);
  // Without peers it still asks for them, as before.
  assert.match(verdictExplanation(gm, { peers: [] }).text,
    /^No target: every method left for .* is reference only\. .*Peers give a P\/E and P\/B/);
  // No method with a value keeps the plain line, with the peers' part when they ran.
  const none = { ...demo, methods: {}, blended_target: null, blended_upside: null,
    recommendation: "N/A" };
  assert.equal(verdictExplanation(none, peers).text,
    "No target: no method gave a usable value, and the peers supplied give no comps value "
    + "(each multiple needs at least 3 usable peer values). Add more peers: with enough usable "
    + "peer values, comps add a trading-comps value.");
  // Peers that ran but gave no price for another reason name no cause and ask for nothing.
  assert.equal(verdictExplanation(none, { peers: peers.peers, notes: [] }).text,
    "No target: no method gave a usable value, and the peers supplied give no usable comps "
    + "price.");
});

test("failed methods kept as 0.00 are not values, and an ordinary company is not reference only", () => {
  // No share count: the engine keeps the DCF and FCFE as 0.00 failures, and the
  // peers' multiples cannot be turned into a price either.
  const FAILED = "no valuation (0.00)";
  const synt = { ...demo, methods: { DCF: 0, FCFE: 0 }, blended_target: null,
    blended_upside: null, recommendation: "N/A",
    excluded_from_blend: { DCF: FAILED, FCFE: FAILED } };
  const why = verdictExplanation(synt,
    { peers: [{ ticker: "PEER1" }, { ticker: "PEER2" }, { ticker: "PEER3" }], notes: [] });
  assert.match(why.text, /^No target: no method gave a usable value/);
  assert.doesNotMatch(why.text, /reference only|at least 3 usable peer values|Add more peers/);
  assert.equal(why.text, "No target: no method gave a usable value, and the peers supplied "
    + `give no usable comps price. Left out of the target: DCF and FCFE (${FAILED}).`);
  // An unflagged company whose remaining value is excluded reads plain "No target".
  const lowPayoutOnly = { ...synt, methods: { DCF: 0, DDM: 9.5 },
    excluded_from_blend: { DCF: FAILED, DDM: LOW_PAYOUT } };
  assert.match(verdictExplanation(lowPayoutOnly, null).text, /^No target\. /);
});

test("the few-peer-values note prefix the explainer matches is the engine's", (t) => {
  const src = python("equity_valuation/models/comps.py");
  if (src === null) {
    t.skip("no engine in this tree");
    return;
  }
  const m = /^FEW_PEER_VALUES_NOTE_PREFIX = "([^"]+)"/m.exec(src);
  assert.ok(m, "FEW_PEER_VALUES_NOTE_PREFIX not found in comps.py");
  assert.equal(m[1], FEW_PEER_VALUES_NOTE_PREFIX);
});

// The Python sources the dashboard copies (skipped when this tree has none).
const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const python = (rel) => {
  const file = join(ROOT, rel);
  return existsSync(file) ? readFileSync(file, "utf8") : null;
};

test("EV multiples are not meaningful for the kinds comps prices on equity multiples only", (t) => {
  for (const kind of ["bank", "insurer", "lender", "financial", "captive_finance", "lessor"]) {
    assert.equal(equityMultiplesOnly({ financial_kind: kind }), true, kind);
  }
  // A REIT keeps its EV multiples; an ordinary company too.
  assert.equal(equityMultiplesOnly({ financial_kind: "reit" }), false);
  assert.equal(equityMultiplesOnly({ financial_kind: null, financial_institution: null }), false);
  assert.equal(equityMultiplesOnly({}), false);
  // A flag without a kind (an older backend) counts as a financial.
  assert.equal(equityMultiplesOnly({ financial_institution: "bank (SIC 6021)" }), true);
  const src = python("equity_valuation/models/comps.py");
  if (src === null) {
    t.skip("no engine in this tree");
    return;
  }
  const m = /^EQUITY_MULTIPLES_ONLY_KINDS = \(([^)]*)\)/m.exec(src);
  assert.ok(m, "EQUITY_MULTIPLES_ONLY_KINDS not found in comps.py");
  const kinds = [...m[1].matchAll(/"([a-z_]+)"/g)].map((x) => x[1]);
  assert.deepEqual([...kinds].sort(), [...EQUITY_MULTIPLES_ONLY_KINDS].sort());
});

test("the comps peer minimum the verdict explainer quotes is the engine's", (t) => {
  const src = python("equity_valuation/config.py");
  if (src === null) {
    t.skip("no engine in this tree");
    return;
  }
  const m = /^COMPS_MIN_PEER_VALUES = (\d+)/m.exec(src);
  assert.ok(m, "COMPS_MIN_PEER_VALUES not found in config.py");
  assert.equal(Number(m[1]), COMPS_MIN_PEER_VALUES);
});

test("DCF forecast indices read as the fiscal years the FCFE table shows", () => {
  // The DCF's 1..n after FY2025; the FCFE's own fiscal years pass through.
  assert.deepEqual(projectionYearLabels([1, 2, 3], [2023, 2024, 2025]),
    ["2026", "2027", "2028"]);
  assert.deepEqual(projectionYearLabels([2026, 2027], [2023, 2024, 2025]), ["2026", "2027"]);
  // The latest usable fiscal year counts; without one the index stays.
  assert.deepEqual(projectionYearLabels([1, 2], [2024, null]), ["2025", "2026"]);
  assert.deepEqual(projectionYearLabels([1, 2], []), ["Y1", "Y2"]);
  assert.deepEqual(projectionYearLabels([1], undefined), ["Y1"]);
  assert.deepEqual(projectionYearLabels(undefined, [2025]), []);
});

test("DDM methods read as the segmented control and the reports name them", (t) => {
  assert.equal(ddmMethodLabel("two_stage"), "Two-stage");
  assert.equal(ddmMethodLabel("h_model"), "H-model");
  assert.equal(ddmMethodLabel("gordon"), "Gordon growth");
  assert.equal(ddmMethodLabel("something_new"), "something_new");
  assert.equal(ddmMethodLabel(null), "—");
  const src = python("equity_valuation/report/excel.py");
  if (src === null) {
    t.skip("no engine in this tree");
    return;
  }
  const m = /^DDM_METHOD_LABELS = \{([^}]*)\}/m.exec(src);
  assert.ok(m, "DDM_METHOD_LABELS not found in excel.py");
  const pairs = Object.fromEntries([...m[1].matchAll(/"(\w+)": "([^"]+)"/g)].map((x) => [x[1], x[2]]));
  assert.deepEqual(pairs, DDM_METHOD_LABELS);
});
