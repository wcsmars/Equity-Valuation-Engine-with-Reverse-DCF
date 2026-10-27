// The assumption comparisons (lib/assumptions.ts) behind the DCF card's
// unapplied-changes bar and the header's "Custom inputs" badge, and the
// draft built from the engine's echo.
// Run with: npm test

import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { loadLib } from "./load-lib.mjs";

const {
  ASSUMPTION_FIELDS,
  TICKER_PATTERN,
  assumptionsFromUsed,
  customFields,
  derivedValues,
  diffAssumptions,
  isPlausibleTicker,
  keepOnDcfReset,
  liveSuggestionValue,
  normalizePeers,
  SUGGESTION_KEYS,
} = loadLib("assumptions");

// The backend's echo of an empty request (parse_assumptions({})[5]).
const DEFAULTS = {
  rf: 0.042, erp: 0.05, tax_rate: null, cost_of_debt: null, forecast_years: 5,
  terminal_growth: 0.025, terminal_method: "gordon", terminal_ronic: null,
  exit_ev_ebitda: null, target_ebit_margin: null, revenue_growth_y1: null,
  revenue_growth: null, peers: null, ddm_method: "two_stage",
  ddm_high_growth_rate: null, ddm_high_growth_years: 5, comps_basis: "fy",
};
const keys = (changes) => changes.map((c) => c.key);

test("rates compare within a tolerance, so float noise is not an edit", () => {
  const used = assumptionsFromUsed(DEFAULTS);
  assert.deepEqual(diffAssumptions({ ...used, terminal_growth: 0.025 + 1e-12 }, used), []);
  assert.deepEqual(diffAssumptions({ ...used, rf: 0.042 - 1e-12 }, used), []);
  const changes = diffAssumptions({ ...used, terminal_growth: 0.035 }, used);
  assert.deepEqual(changes, [
    { key: "terminal_growth", label: "Terminal growth", from: "2.5%", to: "3.5%" },
  ]);
  // A 0.01 move is an edit; quarter points keep their second decimal.
  assert.deepEqual(keys(diffAssumptions({ ...used, erp: 0.06 }, used)), ["erp"]);
  assert.equal(diffAssumptions({ ...used, terminal_growth: 0.0225 }, used)[0].to, "2.25%");
});

test("a field the draft leaves unset is not an edit; one it sets over a derived value is", () => {
  const used = assumptionsFromUsed(DEFAULTS);
  assert.deepEqual(diffAssumptions({}, used), []);
  const [margin] = diffAssumptions({ ...used, target_ebit_margin: 0.3 }, used);
  assert.deepEqual(margin, { key: "target_ebit_margin", label: "Target EBIT margin",
    from: "derived", to: "30.0%" });
});

test("year-1 revenue growth takes precedence over the growth path", () => {
  const usedPath = { ...DEFAULTS, revenue_growth: [0.08, 0.07, 0.06] };
  // A year-1 rate over an explicit path: one change, the path is not compared.
  const changes = diffAssumptions(
    { ...assumptionsFromUsed(usedPath), revenue_growth_y1: 0.1, revenue_growth: undefined },
    assumptionsFromUsed(usedPath));
  assert.deepEqual(keys(changes), ["revenue_growth_y1"]);
  assert.equal(changes[0].from, "8.0% path");
  assert.equal(changes[0].to, "10.0%");
  // The backend echoes the path it built from a year-1 rate; the draft drops
  // it, and the two compare equal.
  const usedY1 = { ...DEFAULTS, revenue_growth_y1: 0.1, revenue_growth: [0.1, 0.08, 0.06] };
  assert.equal(assumptionsFromUsed(usedY1).revenue_growth, undefined);
  assert.deepEqual(diffAssumptions(assumptionsFromUsed(usedY1), assumptionsFromUsed(usedY1)), []);
  assert.deepEqual(customFields(usedY1, DEFAULTS).map((c) => c.key), ["revenue_growth_y1"]);
  // Without a year-1 rate on either side the paths are compared.
  const other = { ...assumptionsFromUsed(usedPath), revenue_growth: [0.08, 0.07, 0.05] };
  assert.deepEqual(keys(diffAssumptions(other, assumptionsFromUsed(usedPath))), ["revenue_growth"]);
  assert.deepEqual(diffAssumptions({ ...other, revenue_growth: [0.08, 0.07, 0.06 + 1e-12] },
    assumptionsFromUsed(usedPath)), []);
});

test("peers compare as trimmed, upper-cased tickers without repeats", () => {
  const used = assumptionsFromUsed({ ...DEFAULTS, peers: "BAC,WFC" });
  assert.deepEqual(diffAssumptions({ ...used, peers: "bac, wfc ,BAC" }, used), []);
  assert.deepEqual(normalizePeers("bac, wfc ,BAC,,"), ["BAC", "WFC"]);
  const [change] = diffAssumptions({ ...used, peers: "bac, c" }, used);
  assert.deepEqual(change, { key: "peers", label: "Peers", from: "BAC, WFC", to: "BAC, C" });
  // A blank peer box is no peer set.
  assert.deepEqual(diffAssumptions({ peers: "  " }, assumptionsFromUsed(DEFAULTS)), []);
});

test("the new engine inputs are compared too", () => {
  const fields = ASSUMPTION_FIELDS.map((f) => f.key);
  for (const k of ["terminal_ronic", "ddm_method", "ddm_high_growth_rate",
    "ddm_high_growth_years", "comps_basis"]) {
    assert.ok(fields.includes(k), k);
  }
  const used = assumptionsFromUsed(DEFAULTS);
  const changes = diffAssumptions({ ...used, ddm_method: "h_model", ddm_high_growth_rate: 0.09,
    ddm_high_growth_years: 8, comps_basis: "ttm", terminal_method: "value_driver",
    terminal_ronic: 0.15 }, used);
  assert.deepEqual(changes.map((c) => `${c.label}: ${c.from} → ${c.to}`), [
    "Terminal method: Gordon → Value driver",
    "Terminal RONIC: WACC → 15.0%",
    "DDM method: Two-stage → H-model",
    "DDM high growth: derived → 9.0%",
    "DDM high-growth years: 5y → 8y",
    "Comps basis: Fiscal year → Trailing 12 months",
  ]);
  // Choices are compared without regard to case.
  assert.deepEqual(diffAssumptions({ ...used, comps_basis: "FY" }, used), []);
});

test("custom inputs: a set tax rate over a derived default and any peer set", () => {
  assert.deepEqual(customFields(DEFAULTS, DEFAULTS), []);
  const used = { ...DEFAULTS, tax_rate: 0.21, peers: "BAC,WFC" };
  assert.deepEqual(customFields(used, DEFAULTS), [
    { key: "tax_rate", label: "Tax rate", from: "derived", to: "21.0%" },
    { key: "peers", label: "Peers", from: "none", to: "BAC, WFC" },
  ]);
  // An older backend sends no defaults: nothing is flagged.
  assert.deepEqual(customFields(used, undefined), []);
  assert.deepEqual(customFields(undefined, DEFAULTS), []);
});

test("the draft built from the echo round-trips every echo key", () => {
  const echo = {
    rf: 0.04, erp: 0.055, tax_rate: 0.2, cost_of_debt: 0.06, forecast_years: 7,
    terminal_growth: 0.03, terminal_method: "value_driver", terminal_ronic: 0.14,
    exit_ev_ebitda: 11, target_ebit_margin: 0.22, revenue_growth_y1: null,
    revenue_growth: [0.09, 0.07, 0.05], peers: "KO,PEP", ddm_method: "h_model",
    ddm_high_growth_rate: 0.07, ddm_high_growth_years: 6, comps_basis: "ttm",
  };
  const draft = assumptionsFromUsed(echo);
  for (const [k, v] of Object.entries(echo)) assert.deepEqual(draft[k], v ?? undefined, k);
  assert.deepEqual(diffAssumptions(draft, assumptionsFromUsed(echo)), []);
  // Every input differs from the defaults here (the path stands in for year
  // 1), but the exit multiple is not custom: the value-driver terminal value
  // ignores it.
  assert.deepEqual(customFields(echo, DEFAULTS).map((c) => c.key).sort(),
    Object.keys(echo).filter((k) => k !== "revenue_growth_y1" && k !== "exit_ev_ebitda").sort());
  // Unset optional inputs stay unset.
  const plain = assumptionsFromUsed(DEFAULTS);
  for (const k of ["tax_rate", "cost_of_debt", "terminal_ronic", "exit_ev_ebitda",
    "target_ebit_margin", "revenue_growth_y1", "revenue_growth", "peers", "ddm_high_growth_rate"]) {
    assert.equal(plain[k], undefined, k);
  }
  assert.deepEqual(assumptionsFromUsed(undefined), {});
});

test("every input the backend echoes is in the draft and compared", (t) => {
  const service = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "backend",
    "valuation_service.py");
  if (!existsSync(service)) return t.skip("no backend in this tree");
  const block = /\n    echo = \{\n([\s\S]*?)\n    \}\n/.exec(readFileSync(service, "utf8"));
  assert.ok(block, "echo dict not found in valuation_service.py");
  const echoed = [...block[1].matchAll(/^\s*"(\w+)":/gm)].map((m) => m[1]);
  assert.ok(echoed.length >= 17, echoed.join(","));
  const fields = ASSUMPTION_FIELDS.map((f) => f.key);
  const full = Object.fromEntries(echoed.map((k) => [k, k === "peers" ? "A" : 1]));
  full.revenue_growth_y1 = null; // so the path is kept
  full.revenue_growth = [0.05];
  const draft = assumptionsFromUsed(full);
  for (const k of echoed) {
    assert.ok(fields.includes(k), `${k} is not compared`);
    assert.ok(k === "revenue_growth_y1" || draft[k] !== undefined, `${k} is dropped from the draft`);
  }
});

test("an input the chosen method ignores is neither an unapplied edit nor custom", () => {
  const used = assumptionsFromUsed(DEFAULTS);
  // Exit multiple and back to Gordon: the draft keeps the 12x (so a second
  // switch restores it), and the backend echoes it, but Gordon ignores it.
  const toggledBack = { ...used, terminal_method: "gordon", exit_ev_ebitda: 12 };
  assert.deepEqual(diffAssumptions(toggledBack, used), []);
  assert.deepEqual(customFields({ ...DEFAULTS, exit_ev_ebitda: 12 }, DEFAULTS), []);
  // Under the exit-multiple method it counts.
  assert.deepEqual(diffAssumptions({ ...toggledBack, terminal_method: "exit_multiple" }, used)
    .map((c) => `${c.label}: ${c.from} → ${c.to}`),
  ["Terminal method: Gordon → Exit multiple", "Exit EV / EBITDA: none → 12.0x"]);
  assert.deepEqual(keys(customFields({ ...DEFAULTS, terminal_method: "exit_multiple",
    exit_ev_ebitda: 12 }, DEFAULTS)), ["terminal_method", "exit_ev_ebitda"]);
  // The terminal RONIC counts only under the value-driver method.
  assert.deepEqual(diffAssumptions({ ...used, terminal_ronic: 0.15 }, used), []);
  assert.deepEqual(diffAssumptions({ ...used, terminal_method: "exit_multiple", exit_ev_ebitda: 12,
    terminal_ronic: 0.15 }, used).map((c) => c.key), ["terminal_method", "exit_ev_ebitda"]);
  assert.deepEqual(customFields({ ...DEFAULTS, terminal_ronic: 0.15 }, DEFAULTS), []);
  assert.deepEqual(keys(customFields({ ...DEFAULTS, terminal_method: "value_driver",
    terminal_ronic: 0.15 }, DEFAULTS)), ["terminal_method", "terminal_ronic"]);
  // A Gordon DDM has no high-growth stage.
  const gordonDdm = { ...used, ddm_method: "gordon", ddm_high_growth_rate: 0.09,
    ddm_high_growth_years: 8 };
  assert.deepEqual(keys(diffAssumptions(gordonDdm, used)), ["ddm_method"]);
  assert.deepEqual(keys(customFields({ ...DEFAULTS, ddm_method: "gordon",
    ddm_high_growth_rate: 0.09, ddm_high_growth_years: 8 }, DEFAULTS)), ["ddm_method"]);
  assert.deepEqual(keys(diffAssumptions({ ...gordonDdm, ddm_method: "h_model" }, used)),
    ["ddm_method", "ddm_high_growth_rate", "ddm_high_growth_years"]);
});

test("an input the base run ignored reads as unset there, and an unchanged one is no edit", () => {
  // The report ran Gordon with a stale 12x echoed; the draft switches to the
  // exit multiple. Keeping 12x adds nothing past the method change; 15x is
  // set from none (the Gordon run used no multiple).
  const used = assumptionsFromUsed({ ...DEFAULTS, exit_ev_ebitda: 12 });
  assert.deepEqual(keys(diffAssumptions({ ...used, terminal_method: "exit_multiple" }, used)),
    ["terminal_method"]);
  const [, multiple] = diffAssumptions({ ...used, terminal_method: "exit_multiple",
    exit_ev_ebitda: 15 }, used);
  assert.deepEqual(multiple, { key: "exit_ev_ebitda", label: "Exit EV / EBITDA",
    from: "none", to: "15.0x" });
  // A Gordon DDM back to two-stage with the echoed 5 years: only the method.
  const ddmUsed = assumptionsFromUsed({ ...DEFAULTS, ddm_method: "gordon" });
  assert.deepEqual(keys(diffAssumptions({ ...ddmUsed, ddm_method: "two_stage" }, ddmUsed)),
    ["ddm_method"]);
  // An unset method is the engine's default (Gordon terminal value,
  // two-stage DDM).
  assert.deepEqual(diffAssumptions({ exit_ev_ebitda: 12, terminal_ronic: 0.15 },
    assumptionsFromUsed(DEFAULTS)), []);
  assert.deepEqual(keys(diffAssumptions({ ddm_high_growth_years: 8 },
    assumptionsFromUsed(DEFAULTS))), ["ddm_high_growth_years"]);
});

test("the DCF card's reset keeps the inputs other cards own", () => {
  assert.deepEqual(keepOnDcfReset({ peers: "BAC", forecast_years: 7, comps_basis: "ttm",
    ddm_method: "gordon", ddm_high_growth_rate: 0.05, ddm_high_growth_years: 3, rf: 0.05 }),
  { peers: "BAC", comps_basis: "ttm", ddm_method: "gordon", ddm_high_growth_rate: 0.05,
    ddm_high_growth_years: 3 });
  assert.deepEqual(keepOnDcfReset({ forecast_years: 7, peers: " " }), {});
  assert.deepEqual(keepOnDcfReset(undefined), {});
});

// What a report says the engine derived for the inputs it left unset.
const DERIVED_REPORT = {
  dcf: { assumptions: { target_ebit_margin: 0.3197, revenue_growth_path: [0.0815, 0.07],
    tax_rate: 0.185, terminal_ronic_used: 0.0852 },
  wacc: { detail: { pretax_cost_of_debt: 0.052 } } },
  ddm: { detail: { high_growth: 0.1079, high_growth_years: 5 } },
};

test("the unapplied bar shows what the engine derived beside 'derived'", () => {
  const derived = derivedValues(DERIVED_REPORT);
  assert.deepEqual(derived, { target_ebit_margin: 0.3197, revenue_growth_y1: 0.0815,
    tax_rate: 0.185, terminal_ronic: 0.0852, cost_of_debt: 0.052,
    ddm_high_growth_rate: 0.1079 });
  const used = assumptionsFromUsed(DEFAULTS);
  const line = (draft, base = used) => diffAssumptions(draft, base, derived)
    .map((c) => `${c.label} ${c.from} → ${c.to}`);
  assert.deepEqual(line({ ...used, target_ebit_margin: 0.3 }),
    ["Target EBIT margin derived 31.97% → 30.0%"]);
  assert.deepEqual(line({ ...used, ddm_high_growth_rate: 0.1175 }),
    ["DDM high growth derived 10.79% → 11.75%"]);
  assert.deepEqual(line({ ...used, revenue_growth_y1: 0.1, tax_rate: 0.21, cost_of_debt: 0.06 }), [
    "Near-term revenue growth derived 8.15% → 10.0%",
    "Tax rate derived 18.5% → 21.0%",
    "Cost of debt derived 5.2% → 6.0%",
  ]);
  // A value-driver run's RONIC defaults to WACC.
  const vd = assumptionsFromUsed({ ...DEFAULTS, terminal_method: "value_driver" });
  assert.deepEqual(line({ ...vd, terminal_ronic: 0.15 }, vd),
    ["Terminal RONIC WACC 8.52% → 15.0%"]);
  // A set value, an input the base run ignored and an explicit path keep
  // their own text; the figure never makes a change.
  assert.deepEqual(line({ ...used, terminal_growth: 0.035 }), ["Terminal growth 2.5% → 3.5%"]);
  assert.deepEqual(line({ ...used, terminal_method: "value_driver", terminal_ronic: 0.15 }),
    ["Terminal method Gordon → Value driver", "Terminal RONIC WACC → 15.0%"]);
  const path = assumptionsFromUsed({ ...DEFAULTS, revenue_growth: [0.08, 0.07] });
  assert.deepEqual(line({ ...path, revenue_growth_y1: 0.1, revenue_growth: undefined }, path),
    ["Near-term revenue growth 8.0% path → 10.0%"]);
  assert.deepEqual(diffAssumptions({ ...used, target_ebit_margin: 0.3197 }, used, derived)
    .map((c) => c.key), ["target_ebit_margin"]);
  assert.deepEqual(diffAssumptions(used, used, derived), []);
  // Without the echo (or a figure) it stays "derived".
  assert.equal(diffAssumptions({ ...used, target_ebit_margin: 0.3 }, used)[0].from, "derived");
  assert.deepEqual(derivedValues(null), {});
  assert.deepEqual(derivedValues({ dcf: { assumptions: { target_ebit_margin: null } } }), {});
});

test("a ticker is checked as the backend checks it, before any request", (t) => {
  for (const ok of ["AAPL", "brk.b", "BRK-B", "7203.T", " msft ", "A", "ABCDEFGHIJKL"]) {
    assert.equal(isPlausibleTicker(ok), true, ok);
  }
  for (const bad of ["AAPL$", "", "  ", "ABCDEFGHIJKLM", "BAD TICKER", "../etc", null, 7]) {
    assert.equal(isPlausibleTicker(bad), false, String(bad));
  }
  const app = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "backend", "app.py");
  if (!existsSync(app)) return t.skip("no backend in this tree");
  const m = /^_TICKER_RE = re\.compile\(r"([^"]+)"\)$/m.exec(readFileSync(app, "utf8"));
  assert.ok(m, "_TICKER_RE not found in backend/app.py");
  assert.equal(TICKER_PATTERN.source, `^${m[1]}$`);
});

test("an AI suggestion is compared with the model on screen, not the value it echoed", () => {
  const report = { ...DERIVED_REPORT, assumptions_used: { ...DEFAULTS, tax_rate: 0.21 } };
  // Set inputs come from the echo, unset ones from what the engine derived.
  assert.equal(liveSuggestionValue(report, "tax_rate"), 0.21);
  assert.equal(liveSuggestionValue(report, "risk_free_rate"), 0.042);
  assert.equal(liveSuggestionValue(report, "equity_risk_premium"), 0.05);
  assert.equal(liveSuggestionValue(report, "forecast_years"), 5);
  assert.equal(liveSuggestionValue(report, "terminal_growth"), 0.025);
  assert.equal(liveSuggestionValue(report, "target_ebit_margin"), 0.3197);
  assert.equal(liveSuggestionValue(report, "revenue_growth_y1"), 0.0815);
  // An explicit growth path's first year; no exit multiple under Gordon.
  const path = { ...report, assumptions_used: { ...DEFAULTS, revenue_growth: [0.06, 0.05] } };
  assert.equal(liveSuggestionValue(path, "revenue_growth_y1"), 0.06);
  assert.equal(liveSuggestionValue(report, "exit_ev_ebitda"), null);
  assert.equal(liveSuggestionValue(null, "tax_rate"), null);
  // Every suggestion field maps to an input the draft has.
  const fields = ASSUMPTION_FIELDS.map((f) => f.key);
  for (const key of Object.values(SUGGESTION_KEYS)) assert.ok(fields.includes(key), key);
});
