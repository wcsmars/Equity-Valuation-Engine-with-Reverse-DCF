// Summary formatting (lib/format.ts): a missing blended target or a withheld
// upside, methods left out of the blend, and the adjusted beta.
// Run with: npm test

import assert from "node:assert/strict";
import { test } from "node:test";
import { loadLib } from "./load-lib.mjs";

const {
  BLUME_MARKET_WEIGHT,
  BLUME_RAW_WEIGHT,
  NO_TARGET_SUPPLY_PEERS,
  betaStat,
  excludedReason,
  fmtBlendedTarget,
  fmtBlendedUpside,
  footballFieldTone,
  needsPeers,
  setsTextColor,
  toneForBlendedUpside,
  toneForMethodUpside,
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
