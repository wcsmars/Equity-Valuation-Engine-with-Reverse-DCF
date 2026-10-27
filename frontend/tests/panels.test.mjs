import assert from "node:assert/strict";
import { test } from "node:test";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { loadComponent } from "./load-component.mjs";

const DCFPanel = loadComponent("components/DCFPanel.tsx").default;
const DDMFCFEPanel = loadComponent("components/DDMFCFEPanel.tsx").default;
const CompsPanel = loadComponent("components/CompsPanel.tsx").default;
const MultiplesPanel = loadComponent("components/MultiplesPanel.tsx").default;
const ValuationSummary = loadComponent("components/ValuationSummary.tsx").default;
const NewsPanel = loadComponent("components/NewsPanel.tsx").default;
const baseReport = {
  summary: { currency: "USD" }, current_price: 100,
  company: { financials: { revenue: [100], ebit: [20] }, balance_sheet: {} },
  macro: { risk_free_rate: 0.04, equity_risk_premium: 0.05 }, warnings: [],
  dcf: null, ddm: null, fcfe: null,
};
const renderDCF = (report, assumptions = {}) => renderToStaticMarkup(React.createElement(DCFPanel,
  { report, assumptions, setAssumptions() {}, onRecompute() {}, recomputing: false }));

test("a failed DCF keeps the assumption controls available for recovery", () => {
  const html = renderDCF(baseReport);
  assert.match(html, /No valid DCF at the current inputs/);
  assert.match(html, /Recompute/);
  assert.equal((html.match(/type="range"/g) || []).length, 7);
});

test("reverse DCF compares the computed growth rather than unsaved slider changes", () => {
  const report = { ...baseReport,
    dcf: { implied_price: 100, assumptions: { revenue_growth_path: [0.08] }, wacc: {} },
    reverse_dcf: { converged: true, implied_growth_y1: 0.12, current_assumption_y1: 0.08 },
  };
  const html = renderDCF(report, { revenue_growth_y1: 0.30 });
  const result = html.slice(html.indexOf("Reverse DCF"));
  assert.match(result, /8\.0%/);
  assert.doesNotMatch(result, /30\.0%/);
});

test("DDM detail units remain percentages above 150 percent", () => {
  const report = { ...baseReport, ddm: { implied_price: 100, cost_of_equity: 0.1,
    method: "test", detail: { payout_ratio: 1.8 } } };
  const html = renderToStaticMarkup(React.createElement(DDMFCFEPanel, { report }));
  assert.match(html, /180\.0%/);
  assert.doesNotMatch(html, /\$1\.80/);
});

test("peer monetary values carry their own currency and never inherit the target currency", () => {
  const report = { ...baseReport, comps: {
    peers: [{ ticker: "HK", name: "Hong Kong peer", currency: "HKD", market_cap: 1e9, enterprise_value: 2e9 },
      { ticker: "UNKNOWN", name: "Unknown currency", market_cap: 3e9, enterprise_value: 4e9 }],
    notes: [], stats: {}, implied: {},
  } };
  const html = renderToStaticMarkup(React.createElement(CompsPanel, {
    report, assumptions: {}, setAssumptions() {}, onRecompute() {}, recomputing: false,
  }));
  assert.match(html, /HK\$1\.0B/);
  assert.match(html, /3\.0B \(currency unknown\)/);
  assert.doesNotMatch(html, /\$3\.0B/);
});

test("FMP estimates use reporting currency and leave unknown currency unmarked", () => {
  const html = renderToStaticMarkup(React.createElement(MultiplesPanel, {
    report: baseReport,
    enrichment: { enabled: true, estimates: [
      { date: "2027-12-31", estimatedRevenueAvg: 2e9, estimatedEpsAvg: 10, reportedCurrency: "TWD" },
      { date: "2028-12-31", estimatedRevenueAvg: 3e9, estimatedEpsAvg: 12 },
    ] },
  }));
  assert.match(html, /TWD 2\.0B/);
  assert.match(html, /TWD 10\.00/);
  assert.match(html, /3\.0B/);
  assert.doesNotMatch(html, /\$3\.0B|\$12\.00/);
  assert.match(html, /reporting currency/);
});

test("unavailable FCFE prices stay missing while a genuine zero retains minus 100 percent", () => {
  const render = (price) => renderToStaticMarkup(React.createElement(DDMFCFEPanel, {
    report: { ...baseReport, fcfe: { implied_price: price, cost_of_equity: 0.1, detail: {} } },
  }));
  assert.doesNotMatch(render(null), /-100\.0%/);
  assert.match(render(null), /FCFE unavailable/);
  assert.match(render(0), /-100\.0%/);
});

test("an unavailable DCF retains controls while a genuine zero displays its valuation", () => {
  assert.match(renderDCF({ ...baseReport, dcf: { implied_price: null, wacc: {} } }), /No valid DCF at the current inputs/);
  assert.match(renderDCF({ ...baseReport, dcf: { implied_price: 0, upside: -1, wacc: {} } }), /-100\.0%/);
});

test("football field does not invent a zero-price reference when market price is missing", () => {
  const html = renderToStaticMarkup(React.createElement(ValuationSummary, { report: {
    ...baseReport, current_price: null,
    company: { ...baseReport.company, market: {} },
    football_field: [{ method: "DCF", low: 80, base: 100, high: 120 }],
  } }));
  assert.doesNotMatch(html, /price —|NaN%|Infinity%/);
  assert.match(html, /base \$100\.00/);
});

test("malformed news rows stay readable and unsafe URLs do not become links", () => {
  const html = renderToStaticMarkup(React.createElement(NewsPanel, {
    enrichment: { enabled: true, news: [{ title: { bad: true }, text: 123,
      site: {}, publishedDate: [], url: "javascript:alert(1)", image: "file:///private/image" }] },
    loading: false,
  }));
  assert.match(html, /Untitled/);
  assert.doesNotMatch(html, /javascript:|file:|<img|<a /);
});

const SensitivityPanel = loadComponent("components/SensitivityPanel.tsx").default;
const { peersOnly, sliderBounds } = loadComponent("lib/assumptions.ts");

// Each range input's label (from its <label for>) with its min, max and value.
function sliders(markup) {
  const names = Object.fromEntries([...markup.matchAll(/<label for="([^"]+)"[^>]*>([^<]+)<\/label>/g)]
    .map((m) => [m[1], m[2]]));
  return Object.fromEntries([...markup.matchAll(/<input ([^>]*type="range"[^>]*)>/g)].map((m) => {
    const attr = (k) => new RegExp(`(?:^| )${k}="([^"]*)"`).exec(m[1])?.[1];
    return [names[attr("id")], { min: +attr("min"), max: +attr("max"), value: +attr("value") }];
  }));
}

test("slider ranges take in the engine's own values so a nudge cannot jump to a bound", () => {
  // A loss-maker keeps its negative margin (as RIVN's -66.5%).
  const lossMaker = { ...baseReport,
    company: { ...baseReport.company, financials: { revenue: [100], ebit: [-66.55] } },
    dcf: { implied_price: 10, wacc: {}, assumptions: { target_ebit_margin: -0.6655, forecast_years: 5 } },
  };
  const m = sliders(renderDCF(lossMaker))["Target EBIT margin"];
  assert.equal(m.value, -0.6655);
  assert.ok(m.min <= -0.6655 && m.max >= 0.6, JSON.stringify(m));
  // Above the usual 60% top (as NVDA's 60.4%), and the full 1-15 year span.
  const highMargin = { ...baseReport,
    dcf: { implied_price: 10, wacc: {}, assumptions: { target_ebit_margin: 0.6038 } } };
  const all = sliders(renderDCF(highMargin));
  assert.ok(all["Target EBIT margin"].max >= 0.6038);
  assert.deepEqual([all["Forecast years"].min, all["Forecast years"].max], [1, 15]);
  // A negative target margin can be chosen for a profitable company too.
  assert.ok(all["Target EBIT margin"].min < 0);
  // A value from an AI suggestion beyond every bound is still in range.
  const years = sliders(renderDCF(baseReport, { forecast_years: 20, exit_ev_ebitda: 45,
    terminal_method: "exit_multiple" }));
  assert.equal(years["Forecast years"].max, 20);
  assert.equal(years["Exit EV / EBITDA"].max, 45);
});

test("slider bounds widen outward onto the step grid and ignore missing values", () => {
  assert.deepEqual(sliderBounds(-0.3, 0.6, 0.005, [-0.6655]), [-0.67, 0.6]);
  assert.deepEqual(sliderBounds(-0.3, 0.6, 0.005, [-0.665]), [-0.665, 0.6]);
  assert.deepEqual(sliderBounds(-0.3, 0.6, 0.005, [0.6038]), [-0.3, 0.605]);
  assert.deepEqual(sliderBounds(1, 15, 1, [20, null, undefined, Number.NaN]), [1, 20]);
  assert.deepEqual(sliderBounds(0, 0.05, 0.0025, [0.025]), [0, 0.05]);
});

// The props of the first element in `node`'s tree whose children are `text`.
function findByText(node, text) {
  if (!node || typeof node !== "object") return null;
  if (Array.isArray(node)) {
    for (const n of node) {
      const hit = findByText(n, text);
      if (hit) return hit;
    }
    return null;
  }
  if (node.props?.children === text) return node.props;
  return findByText(node.props?.children, text);
}

test("Reset defaults resets the DCF drivers but keeps the peer set", () => {
  const press = (assumptions) => {
    const calls = [];
    const tree = DCFPanel({ report: baseReport, assumptions, recomputing: false,
      setAssumptions: (a) => calls.push(["set", a]), onRecompute: (a) => calls.push(["recompute", a]) });
    findByText(tree, "Reset defaults").onClick();
    return calls;
  };
  const calls = press({ peers: "BAC,C,WFC", forecast_years: 7, target_ebit_margin: 0.1 });
  assert.deepEqual(calls, [["set", { peers: "BAC,C,WFC" }], ["recompute", { peers: "BAC,C,WFC" }]]);
  // The same object goes to both, so the recompute can normalise it.
  assert.equal(calls[0][1], calls[1][1]);
  assert.deepEqual(press({ forecast_years: 7, peers: "  " }), [["set", {}], ["recompute", {}]]);
  assert.deepEqual(peersOnly(undefined), {});
  assert.deepEqual(peersOnly({ peers: "F,TM", rf: 0.05 }), { peers: "F,TM" });
});

test("reverse DCF shows the engine's caveat when several growth rates fit", () => {
  const report = { ...baseReport,
    dcf: { implied_price: 100, assumptions: { revenue_growth_path: [0.066] }, wacc: {} },
    reverse_dcf: { converged: true, implied_growth_y1: 0.05, current_assumption_y1: 0.066,
      note: "More than one growth rate reproduces the market price with these assumptions; showing the lowest." },
  };
  const result = renderDCF(report).split("Reverse DCF")[1];
  assert.match(result, /5\.0%/);
  assert.match(result, /More than one growth rate reproduces the market price/);
  assert.match(result, /have to believe/);
});

test("reverse DCF of a reference-only DCF is labelled and not read as what is priced in", () => {
  const report = { ...baseReport,
    summary: { currency: "USD", excluded_from_blend: { DCF: "not meaningful with a consolidated captive finance arm" } },
    dcf: { implied_price: 247.17, assumptions: { revenue_growth_path: [0.033] }, wacc: {} },
    reverse_dcf: { converged: true, implied_growth_y1: -0.207, current_assumption_y1: 0.033 },
  };
  const result = renderDCF(report).split("Reverse DCF")[1];
  assert.match(result, /^ \(reference only\)</);
  assert.match(result, /-20\.7%/);
  assert.match(result, /year-1 revenue growth at which the reference DCF equals the price/);
  assert.match(result, /inverts the reference-only DCF above \(not meaningful with a consolidated captive finance arm\), so it is not a market-implied growth rate/);
  // Nothing in the box reads it as what the market price implies.
  assert.doesNotMatch(result, /priced in|implied by the market price|have to believe/);
  // A DCF in the blend keeps the market reading.
  const inBlend = renderDCF({ ...report, summary: { currency: "USD" } }).split("Reverse DCF")[1];
  assert.match(inBlend, /^ — what&#x27;s priced in</);
  assert.match(inBlend, /year-1 revenue growth implied by the market price/);
  assert.doesNotMatch(inBlend, /reference only|reference DCF/);
});

test("sensitivity cells and football-field labels keep cents for a low-priced stock", () => {
  const grid = { title: "WACC x terminal growth", row_label: "WACC", col_label: "Terminal growth",
    row_values: [0.065, 0.072], col_values: [0.02, 0.025, 0.03],
    grid: [[4.13, 4.51, 4.98], [3.70, 3.99, 4.33]] };
  const sens = renderToStaticMarkup(React.createElement(SensitivityPanel, { report: {
    ...baseReport, current_price: 9.25, sensitivities: [grid] } }));
  for (const v of ["$4.13", "$4.51", "$4.98", "$3.70", "$3.99", "$4.33"]) assert.ok(sens.includes(`>${v}<`), v);
  assert.match(sens, /shaded vs\. current \$9\.25/);
  const ff = renderToStaticMarkup(React.createElement(ValuationSummary, { report: {
    ...baseReport, current_price: 9.25,
    company: { ...baseReport.company, market: { fifty_two_week_low: 6.12, fifty_two_week_high: 11.87 } },
    summary: { currency: "USD", current_price: 9.25 },
    football_field: [{ method: "DCF", low: 4.13, base: 5.58, high: 6.38 }],
  } }));
  assert.match(ff, />\$6\.38</);
  assert.match(ff, /price \$9\.25/);
  assert.match(ff, /\$6\.12 –/);
  // Prices in the thousands stay whole.
  const big = renderToStaticMarkup(React.createElement(SensitivityPanel, { report: {
    ...baseReport, current_price: 2500, sensitivities: [{ ...grid, grid: [[2400.4, 2500.6, 2600], [1, 2, 3]] }] } }));
  assert.match(big, />\$2,400</);
});

test("the comps hint says suggested peers are only applied by Update peers", () => {
  const html = renderToStaticMarkup(React.createElement(CompsPanel, {
    report: baseReport, assumptions: {}, setAssumptions() {}, onRecompute() {}, recomputing: false,
  }));
  assert.match(html, /leave blank for no comps\.\s+Suggest peers fills the box from Yahoo&#x27;s industry list for you to\s+review, and nothing is applied until you press Update peers \(FMP, when\s+connected, fills them once on load\)\./);
  assert.match(html, /<button[^>]*>Suggest peers<\/button>/);
  // One statement of the FMP fill: the empty state does not repeat it.
  assert.equal((html.match(/FMP/g) || []).length, 1);
  assert.doesNotMatch(html, /auto-suggest|auto-filled|skip comps/);
});

// --- Unapplied edits, reference-only drivers and the new engine outputs ---- //
const VerdictExplainer = loadComponent("components/VerdictExplainer.tsx").default;
const { assumptionsFromUsed } = loadComponent("lib/assumptions.ts");
// The backend's echo of an empty request (parse_assumptions({})[5]).
const USED = {
  rf: 0.042, erp: 0.05, tax_rate: null, cost_of_debt: null, forecast_years: 5,
  terminal_growth: 0.025, terminal_method: "gordon", terminal_ronic: null,
  exit_ev_ebitda: null, target_ebit_margin: null, revenue_growth_y1: null,
  revenue_growth: null, peers: null, ddm_method: "two_stage",
  ddm_high_growth_rate: null, ddm_high_growth_years: 5, comps_basis: "fy",
};
// The synthetic company's DCF echo (terminal-value check included).
const DCF_ECHO = { forecast_years: 5, terminal_growth: 0.025, terminal_growth_used: 0.025,
  revenue_growth_path: [0.08], target_ebit_margin: 0.25, terminal_method: "gordon",
  terminal_reinvestment_rate: 0.0673, implied_terminal_ronic: 0.3714, latest_roic: 0.2986,
  terminal_ronic_used: null };
const valued = { ...baseReport, assumptions_used: USED,
  dcf: { implied_price: 33.38, upside: -0.18, wacc: { wacc: 0.095 }, assumptions: DCF_ECHO } };
const renderComps = (report, assumptions = {}) => renderToStaticMarkup(React.createElement(CompsPanel,
  { report, assumptions, setAssumptions() {}, onRecompute() {}, recomputing: false }));
// The first element in `node`'s tree whose component is named `name`.
function findType(node, name) {
  if (!node || typeof node !== "object") return null;
  if (Array.isArray(node)) {
    for (const n of node) {
      const hit = findType(n, name);
      if (hit) return hit;
    }
    return null;
  }
  if (node.type?.name === name) return node;
  return findType(node.props?.children, name);
}

test("unapplied edits show in an amber bar with Recompute and Discard", () => {
  const draft = { ...assumptionsFromUsed(USED), terminal_growth: 0.035 };
  const html = renderDCF(valued, draft);
  assert.match(html, /<p[^>]*><span class="font-semibold" aria-hidden="true">1 unapplied change:<\/span> Terminal growth 2\.5% → 3\.5%<\/p>/);
  // Only the count is a live region, so a slider drag that rewrites the list
  // is not re-announced on every tick.
  assert.deepEqual([...html.matchAll(/role="status"[^>]*>([^<]*)</g)].map((m) => m[1]),
    ["1 unapplied change"]);
  // The changed slider's value is amber and read out as not applied.
  assert.match(html, />Terminal growth<\/label><span class="num text-sm font-semibold text-flat" title="Changed: recompute to apply">3\.5%</);
  assert.match(html, /aria-valuetext="3\.5%, not applied"/);
  assert.match(html, />Forecast years<\/label><span class="num text-sm font-semibold text-ink">/);
  // No bar once the draft is what the engine used; the (empty) live region
  // stays, so the next first change is announced.
  const clean = renderDCF(valued, assumptionsFromUsed(USED));
  assert.doesNotMatch(clean, /unapplied/);
  assert.match(clean, /<span role="status" class="sr-only"><\/span>/);
  // Past three changes the list ends in an ellipsis.
  const many = renderDCF(valued, { ...draft, rf: 0.05, erp: 0.06, tax_rate: 0.3 });
  assert.match(many, /4 unapplied changes:<\/span> Terminal growth 2\.5% → 3\.5%; Risk-free rate 4\.2% → 5\.0%; Equity risk premium 5\.0% → 6\.0% …/);
  // Discard returns to what the engine used; Recompute sends the draft.
  const calls = [];
  const bar = findType(DCFPanel({ report: valued, assumptions: draft, recomputing: false,
    setAssumptions: (a) => calls.push(["set", a]), onRecompute: (a) => calls.push(["recompute", a]) }),
  "UnappliedBar").props;
  bar.onDiscard();
  bar.onRecompute();
  assert.deepEqual(calls, [["set", assumptionsFromUsed(USED)], ["recompute", undefined]]);
});

test("a reference-only DCF collapses its drivers behind what moves the target", () => {
  const bank = { ...valued, summary: { currency: "USD", ticker: "JPM",
    excluded_from_blend: { DCF: "not meaningful for a financial institution" },
    financial_kind: "bank", financial_institution: "bank" } };
  const html = renderDCF(bank);
  assert.match(html, /Reference-only for a bank: these\s+drivers do not move the blended target\./);
  assert.match(html, /left out of the target \(not meaningful for a financial institution\)/);
  // No DDM (a flagged non-payer): the peer set alone moves the target.
  assert.match(html, /What moves it: the peer set in the Trading comps box, below\./);
  assert.doesNotMatch(html, /DDM drivers in the dividend discount card/);
  // A DDM in the blend is named too.
  const ddm = { implied_price: 40, cost_of_equity: 0.1, method: "two_stage", detail: {} };
  assert.match(renderDCF({ ...bank, ddm }),
    /What moves it: the DDM drivers in the dividend discount card and the peer set in the Trading comps box, below\./);
  // A DDM itself left out of the blend (a low-payout lessor) moves nothing.
  const lessor = { ...bank, ddm, summary: { ...bank.summary, financial_kind: "lessor",
    excluded_from_blend: { DCF: "not meaningful for a debt-funded lessor",
      DDM: "dividends only (12% of net income); buybacks ignored",
      FCFE: "not meaningful for a debt-funded lessor" } } };
  const lessorHtml = renderDCF(lessor);
  assert.match(lessorHtml, /Reference-only for a debt-funded lessor/);
  assert.match(lessorHtml, /What moves it: the peer set in the Trading comps box, below\./);
  assert.doesNotMatch(lessorHtml, /DDM drivers in the dividend discount card/);
  assert.match(html, /aria-expanded="false"[^>]*>Show the DCF drivers</);
  assert.equal((html.match(/type="range"/g) || []).length, 0);
  // The terminal-value check is not shown for a reference-only DCF.
  assert.doesNotMatch(html, /Terminal reinvestment|Implied RONIC/);
  // Recompute and Reset stay in reach; an ordinary DCF shows its drivers.
  assert.match(html, />Recompute</);
  assert.doesNotMatch(renderDCF(valued), /Reference-only|Show the DCF drivers/);
});

test("a DCF left out only because it failed keeps its drivers open for recovery", () => {
  // An ordinary company whose DCF has no price at these inputs: the engine
  // leaves it out of the blend, but a recompute at other inputs can bring it
  // back, so it is not reference-only.
  const failed = { ...valued, summary: { currency: "USD", ticker: "XYZ",
    excluded_from_blend: { DCF: "non-finite implied price" } },
  dcf: { ...valued.dcf, implied_price: null } };
  const html = renderDCF(failed);
  assert.doesNotMatch(html, /Reference-only|Show the DCF drivers/);
  assert.equal((html.match(/type="range"/g) || []).length, 7);
  assert.match(html, /No valid DCF at the current inputs/);
});

test("a bank with no operating income gets a reference-DCF card with the DCF's own notes", () => {
  // JPM-like: no EBIT line, so the reference DCF has no price; the report's
  // other notes (restricted cash, comps, the DDM) do not belong on this card.
  const BANK = "not meaningful for a financial institution";
  const noEbit = "DCF: EBIT margin unavailable: no operating income is reported, and a flagged "
    + "company's EBIT is not rebuilt from pretax income, so the DCF gives no reference price "
    + "instead of projecting a 0% margin -- set a target EBIT margin to project one";
  const fi = "Financial institution (Banks - Diversified): banks fund operations with deposits, "
    + "so an FCFF DCF and an FCFE do not describe the business. The FCFE is shown for reference "
    + "only and left out of the blended target; the DCF has no reference price (no operating "
    + "income is reported).";
  const jpm = { ...valued,
    summary: { currency: "USD", ticker: "JPM", financial_kind: "bank",
      financial_institution: "Banks - Diversified", excluded_from_blend: { DCF: BANK, FCFE: BANK } },
    dcf: { ...valued.dcf, implied_price: null },
    warnings: [
      "Cash and short-term investments exclude 25.00B of restricted cash held at central banks",
      fi,
      noEbit,
      "Comps: Equity multiples only (Banks - Diversified): EV/EBITDA and EV/Sales give no implied price",
      "Financial institution: the DDM is shown for reference only and left out of the blended target.",
    ] };
  const html = renderDCF(jpm);
  assert.match(html, /No reference DCF/);
  assert.doesNotMatch(html, /No valid DCF at the current inputs/);
  const card = html.slice(html.indexOf("No reference DCF"));
  assert.match(card, /EBIT margin unavailable/);
  assert.match(card, /the DCF has no reference price/);
  assert.doesNotMatch(card, /restricted cash|Equity multiples only|the DDM is shown/);
  // With no DCF note among them, every warning is shown, as before.
  const other = renderDCF({ ...jpm, warnings: ["Something about the data"] });
  assert.match(other.slice(other.indexOf("No reference DCF")), /Something about the data/);
});

test("the value-driver terminal method offers a RONIC slider that defaults to WACC", () => {
  const html = renderDCF(valued, { ...assumptionsFromUsed(USED), terminal_method: "value_driver" });
  assert.match(html, /aria-pressed="true"[^>]*>Value driver</);
  assert.equal(sliders(html)["Terminal RONIC"].value, 0.095);
  assert.match(html, /At WACC \(the default\) growth adds no value/);
  assert.equal(sliders(renderDCF(valued))["Terminal RONIC"], undefined);
  const set = sliders(renderDCF({ ...valued, dcf: { ...valued.dcf, assumptions: {
    ...DCF_ECHO, terminal_method: "value_driver", terminal_ronic_used: 0.15 } } },
  { terminal_method: "value_driver" }));
  assert.equal(set["Terminal RONIC"].value, 0.15);
});

test("the DCF output shows the terminal reinvestment and the RONIC it implies", () => {
  const html = renderDCF(valued);
  assert.match(html, />Terminal reinvestment<\/div><div class="num mt-0\.5 text-lg font-semibold text-ink">6\.7%<\/div><div[^>]*>of NOPAT</);
  assert.match(html, />Implied RONIC<\/div><div class="num mt-0\.5 text-lg font-semibold text-ink">37\.1%<\/div><div[^>]*>latest ROIC 29\.9%/);
  // Growth with no net reinvestment reads amber.
  const none = renderDCF({ ...valued, dcf: { ...valued.dcf, assumptions: {
    ...DCF_ECHO, terminal_reinvestment_rate: -0.01, implied_terminal_ronic: null } } });
  assert.match(none, />Implied RONIC<\/div><div class="num mt-0\.5 text-lg font-semibold text-flat">n\/a<\/div><div[^>]*>growth without reinvestment/);
  // Not for an exit multiple, whose terminal value projects no stable year.
  const exit = renderDCF({ ...valued, dcf: { ...valued.dcf, assumptions: {
    ...DCF_ECHO, terminal_method: "exit_multiple", terminal_reinvestment_rate: null } } });
  assert.doesNotMatch(exit, /Terminal reinvestment|Implied RONIC/);
  // An older backend that does not echo the method is read as Gordon.
  const { terminal_method: _m, ...noMethod } = DCF_ECHO;
  const old = renderDCF({ ...valued, dcf: { ...valued.dcf, assumptions: {
    ...noMethod, terminal_reinvestment_rate: -0.01, implied_terminal_ronic: null } } });
  assert.match(old, /text-flat">n\/a<\/div><div[^>]*>growth without reinvestment/);
});

test("a value-driver terminal value shows the reinvestment it uses and is never flagged", () => {
  // The drivers carry no net reinvestment, but the value-driver terminal
  // value reinvests g / RONIC = 2.5% / 15% of NOPAT by construction, so the
  // engine runs no check and nothing reads amber.
  const vd = (extra) => renderDCF({ ...valued, dcf: { ...valued.dcf, assumptions: {
    ...DCF_ECHO, terminal_method: "value_driver", terminal_ronic_used: 0.15, ...extra } } },
  { ...assumptionsFromUsed(USED), terminal_method: "value_driver" });
  const html = vd({ terminal_reinvestment_rate: -0.01, implied_terminal_ronic: null });
  const stats = html.slice(html.indexOf(">Terminal reinvestment<"), html.indexOf("Projection"));
  assert.match(stats, /^>Terminal reinvestment<\/div><div class="num mt-0\.5 text-lg font-semibold text-ink">16\.7%<\/div><div[^>]*>of NOPAT · drivers -1\.0%</);
  assert.match(stats, />RONIC used<\/div><div class="num mt-0\.5 text-lg font-semibold text-ink">15\.0%<\/div><div[^>]*>latest ROIC 29\.9%</);
  assert.doesNotMatch(stats, /text-flat|growth without reinvestment|Implied RONIC/);
  // Nor when the drivers' implied RONIC is far above the latest ROIC.
  const high = vd({ terminal_reinvestment_rate: 0.005, implied_terminal_ronic: 5 });
  const highStats = high.slice(high.indexOf(">Terminal reinvestment<"), high.indexOf("Projection"));
  assert.match(highStats, /of NOPAT · drivers 0\.5%/);
  assert.doesNotMatch(highStats, /text-flat|Implied RONIC/);
  // Under Gordon the same drivers still read amber.
  const gordon = renderDCF({ ...valued, dcf: { ...valued.dcf, assumptions: {
    ...DCF_ECHO, terminal_reinvestment_rate: 0.005, implied_terminal_ronic: 5 } } });
  assert.match(gordon, />Implied RONIC<\/div><div class="num mt-0\.5 text-lg font-semibold text-flat">500\.0%</);
});

test("the reverse DCF lists the implied margin and discount rate", () => {
  const report = { ...valued, reverse_dcf: { converged: true, implied_growth_y1: 0.1747,
    current_assumption_y1: 0.08,
    margin: { converged: true, implied_target_ebit_margin: 0.3084, current_target_ebit_margin: 0.25,
      note: null },
    wacc: { converged: false, implied_wacc: null, current_wacc: 0.095,
      note: "Market price is outside the solvable discount rate range." } } };
  const result = renderDCF(report).split("Reverse DCF")[1];
  assert.match(result, /Target EBIT margin<\/span><span[^>]*><span class="num font-semibold text-ink">30\.8%<\/span> vs <span class="num text-ink">25\.0%<\/span> in the model/);
  assert.match(result, /Discount rate \(WACC\)<\/span><span[^>]*>Market price is outside the solvable discount rate range\.</);
  assert.match(result, /Each driver is solved with the others held at the model&#x27;s values\./);
  // Without the two legs (an older backend) the box is as before.
  assert.doesNotMatch(renderDCF({ ...report, reverse_dcf: { converged: true, implied_growth_y1: 0.1,
    current_assumption_y1: 0.08 } }), /Discount rate \(WACC\)|Each driver is solved/);
});

test("an all-n/a sensitivity grid is one empty state with the reasons", () => {
  const grid = { title: "WACC x terminal growth", row_label: "WACC", col_label: "Terminal growth",
    row_values: [0.08, 0.09], col_values: [0.02, 0.025, 0.03],
    grid: [[null, null, null], [null, null, null]] };
  const html = renderToStaticMarkup(React.createElement(SensitivityPanel, {
    report: { ...baseReport, sensitivities: [grid] } }));
  assert.match(html, /No valid price anywhere in this grid/);
  assert.match(html, /terminal growth too close to WACC/);
  assert.doesNotMatch(html, /<table|>n\/a</);
  // One valid cell keeps the grid, with "n/a" in the others.
  const one = renderToStaticMarkup(React.createElement(SensitivityPanel, {
    report: { ...baseReport, sensitivities: [{ ...grid, grid: [[null, 12.5, null], [null, null, null]] }] } }));
  assert.match(one, /<table/);
  assert.equal((one.match(/>n\/a</g) || []).length, 5);
});

const STATS = (n) => Object.fromEntries(["ev_ebitda", "ev_sales", "pe", "pb", "peg"].map((k) =>
  [k, { n, median: 10, mean: 10, min: 8, max: 12, p25: 9, p75: 11 }]));
const COMPS = { target: { ticker: "SYNT", name: "Synthetic", currency: "USD" },
  peers: [{ ticker: "PEERA", name: "A", currency: "USD" }, { ticker: "PEERB", name: "B", currency: "USD" }],
  implied: {}, implied_price_summary: {}, notes: [], peer_status: {} };

test("comps headers carry each multiple's sample size and quartiles need four peers", () => {
  const two = renderComps({ ...baseReport, comps: { ...COMPS, stats: STATS(2) } });
  const text = (html) => html.replace(/<[^>]+>/g, "");
  for (const h of ["EV/EBITDA · n=2", "EV/Sales · n=2", "P/E · n=2", "P/B · n=2", "PEG (ref.) · n=2"]) {
    assert.ok(text(two).includes(h), h);
  }
  // The n stays lower case in the upper-case header row.
  assert.match(two, /P\/E<span class="font-normal normal-case"> · n=2<\/span>/);
  assert.doesNotMatch(two, /Peer 25th pct|Peer 75th pct/);
  assert.match(two, /Peer median/);
  const five = renderComps({ ...baseReport, comps: { ...COMPS, stats: STATS(5) } });
  assert.match(text(five), /P\/E · n=5/);
  assert.match(five, /Peer 25th pct/);
  assert.match(five, /Peer 75th pct/);
  // Mixed: the rows show, but a multiple with too few peers has no quartile.
  const mixed = renderComps({ ...baseReport, comps: { ...COMPS, stats: { ...STATS(2), pe: STATS(5).pe } } });
  const p25 = mixed.slice(mixed.indexOf("Peer 25th pct"), mixed.indexOf("Peer 75th pct"));
  assert.equal((p25.match(/>9\.0x</g) || []).length, 1);
  // An older backend without sample sizes keeps the plain headers and rows.
  const plain = renderComps({ ...baseReport, comps: { ...COMPS, stats: {} } });
  assert.match(plain, />P\/E</);
  assert.match(plain, /Peer 25th pct/);
});

test("each requested peer is tagged with what comps did with it, with the reason", () => {
  const comps = { ...COMPS, peers: [], stats: {},
    notes: ["Peers not used: XYZNOTREAL (not found), ABC (no usable multiple)."],
    peer_status: { BAC: "used", XYZNOTREAL: "not found", ABC: "no usable multiple", JPM: "is target" } };
  const report = { ...baseReport, comps, assumptions_used: { ...USED, peers: "BAC,XYZNOTREAL,ABC,JPM" },
    warnings: ["Comps: Peers not used: XYZNOTREAL (not found), ABC (no usable multiple).",
      "Comps: none of the peers supplied (XYZNOTREAL, ABC) returned usable multiples; check the tickers or supply other peers."] };
  const html = renderComps(report, { peers: "BAC,XYZNOTREAL,ABC,JPM" });
  assert.match(html, /<span title="Used in the peer multiples\." class="[^"]*text-up[^"]*">BAC<\/span>/);
  assert.match(html, /<span title="Not used: no market data came back[^"]*" class="[^"]*text-down[^"]*">XYZNOTREAL<span class="ml-1 font-normal">· not found<\/span>/);
  assert.match(html, /<span title="Not used: its data has no positive[^"]*" class="[^"]*text-flat[^"]*">ABC</);
  assert.match(html, /<span title="Not used: this is the company being valued\." class="[^"]*text-ink-dim[^"]*">JPM</);
  // No usable peer: the empty state says so, the note shows once and the
  // engine's reason is added.
  assert.match(html, /No usable peers/);
  assert.equal((html.match(/Peers not used: XYZNOTREAL/g) || []).length, 1);
  assert.match(html, />None of the peers supplied \(XYZNOTREAL, ABC\) returned usable multiples/);
  // Without a requested peer, only the empty state speaks.
  const none = renderComps({ ...baseReport, comps: { ...COMPS, peers: [], stats: {}, peer_status: {} },
    assumptions_used: USED, warnings: ["Comps: no usable peers found (pass --peers to supply them)."] });
  assert.match(none, /No peer set yet/);
  assert.doesNotMatch(none, /no usable peers found|--peers/i);
});

test("comps notes point at the dashboard, and the basis toggle shows the report's basis", () => {
  const report = { ...baseReport, assumptions_used: { ...USED, comps_basis: "ttm" },
    comps: { ...COMPS, stats: STATS(2), notes: ["Fiscal-year vs trailing gap: x (an opt-in trailing "
      + "basis, --comps-basis ttm in the CLI, uses Yahoo's trailing figures)."] } };
  const html = renderComps(report);
  assert.match(html, /the TTM toggle in the Trading comps box/);
  assert.doesNotMatch(html, /--comps-basis/);
  assert.match(html, /role="group" aria-label="Comps basis"/);
  assert.match(html, /aria-pressed="true"[^>]*>TTM</);
  assert.match(renderComps(report, { comps_basis: "fy" }), /aria-pressed="true"[^>]*>FY</);
  // A peer box that differs from the peers used is marked as not applied.
  assert.match(renderComps(report, { peers: "KO" }), /Not applied: press Update peers/);
  assert.doesNotMatch(html, /Not applied/);
});

const DDM_REPORT = { ...baseReport, assumptions_used: USED,
  ddm: { implied_price: 10.99, method: "two_stage", cost_of_equity: 0.097,
    detail: { high_growth: 0.08, high_growth_years: 5 } } };
const renderDDM = (report, assumptions = {}, extra = {}) => renderToStaticMarkup(React.createElement(
  DDMFCFEPanel, { report, assumptions, setAssumptions() {}, onRecompute() {}, recomputing: false, ...extra }));

test("the DDM card carries its drivers, defaulting to what the engine used", () => {
  const html = renderDDM(DDM_REPORT, assumptionsFromUsed(USED));
  assert.match(html, /DDM drivers/);
  assert.match(html, /role="group" aria-label="DDM method"/);
  assert.match(html, /aria-pressed="true"[^>]*>Two-stage</);
  const s = sliders(html);
  assert.equal(s["High-growth rate"].value, 0.08);
  assert.equal(s["High-growth years"].value, 5);
  assert.doesNotMatch(html, /unapplied DDM change/);
  // No DDM, no drivers.
  assert.doesNotMatch(renderDDM(baseReport), /DDM drivers/);
  // Gordon has no high-growth stage.
  const gordon = renderDDM(DDM_REPORT, { ddm_method: "gordon" });
  assert.equal(sliders(gordon)["High-growth rate"], undefined);
  assert.match(gordon, /no high-growth stage/);
  // An H-model reports its half-life; the years are twice it.
  const h = renderDDM({ ...DDM_REPORT, ddm: { ...DDM_REPORT.ddm, method: "h_model",
    detail: { high_growth: 0.06, H_half: 3 } } });
  assert.equal(sliders(h)["High-growth years"].value, 6);
  // An edit is flagged until recomputed.
  const edited = renderDDM(DDM_REPORT, { ...assumptionsFromUsed(USED), ddm_high_growth_rate: 0.12 });
  assert.match(edited, /1 unapplied DDM change/);
  assert.match(edited, />High-growth rate<\/label><span class="num text-sm font-semibold text-flat"/);
});

test("DDM driver edits go to the shared draft, inside the backend's bounds", () => {
  const calls = [];
  const props = { report: DDM_REPORT, assumptions: { rf: 0.04 }, recomputing: false,
    setAssumptions: (a) => calls.push(["set", a]), onRecompute: (a) => calls.push(["recompute", a]) };
  const drivers = findType(DDMFCFEPanel(props), "DDMDrivers");
  const tree = drivers.type(drivers.props);
  findType(tree, "AssumptionSlider").props.onChange(0.75); // the high-growth rate
  findType(tree, "Segmented").props.onChange("h_model");
  findByText(tree, "Defaults").onClick();
  assert.deepEqual(calls, [
    ["set", { rf: 0.04, ddm_high_growth_rate: 0.5975 }],
    ["set", { rf: 0.04, ddm_method: "h_model" }],
    ["set", { rf: 0.04, ddm_method: undefined, ddm_high_growth_rate: undefined,
      ddm_high_growth_years: undefined }],
    ["recompute", { rf: 0.04, ddm_method: undefined, ddm_high_growth_rate: undefined,
      ddm_high_growth_years: undefined }],
  ]);
});

test("the verdict explainer says why and offers to add peers", () => {
  const bank = { currency: "USD", methods: { DCF: 33.38, DDM: 10.99, FCFE: 31.41 },
    blended_target: 10.99, blended_upside: null, recommendation: "N/A",
    excluded_from_blend: { DCF: "not meaningful for a financial institution",
      FCFE: "not meaningful for a financial institution" },
    financial_kind: "bank", financial_institution: "bank" };
  const html = renderToStaticMarkup(React.createElement(VerdictExplainer,
    { summary: bank, comps: { peers: [] }, onAddPeers() {} }));
  assert.match(html, /No verdict: the target rests on the DDM alone/);
  assert.match(html, /Left out of the target: DCF and FCFE \(not meaningful for a financial institution\)/);
  assert.match(html, /<button type="button"[^>]*>Add peers<\/button>/);
  // Nothing to explain for an ordinary blend.
  assert.equal(renderToStaticMarkup(React.createElement(VerdictExplainer, {
    summary: { ...bank, methods: { DCF: 33, "Comps (median)": 47 }, blended_target: 40,
      blended_upside: -0.02, recommendation: "Fairly valued", excluded_from_blend: {},
      financial_kind: null, financial_institution: null },
    comps: { peers: [{}] }, onAddPeers() {} })), "");
});

test("the DCF projection is headed by fiscal years, as the FCFE table is", () => {
  const fy = { ...baseReport.company, financials: { ...baseReport.company.financials,
    fiscal_years: [2023, 2024, 2025] } };
  const dcf = { ...valued.dcf, years: [1, 2, 3], revenue: [1, 2, 3], ebit: [1, 1, 1] };
  const html = renderDCF({ ...valued, company: fy, dcf });
  const head = html.slice(html.indexOf(">Driver<"), html.indexOf("</thead>", html.indexOf(">Driver<")));
  assert.deepEqual([...head.matchAll(/>(\w+)<\/th>/g)].map((m) => m[1]),
    ["Driver", "2026", "2027", "2028"]);
  const fcfe = renderDDM({ ...baseReport, company: fy, fcfe: { implied_price: 30,
    cost_of_equity: 0.1, years: [2026, 2027, 2028], fcfe: [1, 2, 3], pv_fcfe: [1, 2, 3],
    detail: {} } });
  assert.match(fcfe, />Year<\/th><th[^>]*>2026<\/th><th[^>]*>2027<\/th><th[^>]*>2028</);
  // Without fiscal years the index stays, marked as one.
  assert.match(renderDCF({ ...valued, dcf }), />Y1<\/th>/);
});

test("the unapplied bar shows the derived figure an edit replaces", () => {
  const report = { ...valued, ddm: { implied_price: 10.99, method: "two_stage",
    cost_of_equity: 0.097, detail: { high_growth: 0.1079, high_growth_years: 5 } },
  dcf: { ...valued.dcf, assumptions: { ...DCF_ECHO, target_ebit_margin: 0.3197 } } };
  const html = renderDCF(report, { ...assumptionsFromUsed(USED), target_ebit_margin: 0.3,
    ddm_high_growth_rate: 0.1175 });
  assert.match(html, /2 unapplied changes:<\/span> Target EBIT margin derived 31\.97% → 30\.0%; DDM high growth derived 10\.79% → 11\.75%</);
});

test("the DDM card names its method and detail rows in words", () => {
  const html = renderDDM({ ...DDM_REPORT, ddm: { ...DDM_REPORT.ddm, detail: {
    method: "two_stage", cost_of_equity: 0.097, D0: 1.2, terminal_growth: 0.025,
    high_growth: 0.08, high_growth_years: 5, terminal_growth_used: 0.025,
    dividends: [1.3, 1.4], stage1_pvs: [1.2, 1.1], pv_stage1: 2.3, terminal_value: 30,
    pv_terminal: 8, implied_price: 10.99, payout_ratio: 0.4, notes: [] } } });
  const text = html.replace(/<[^>]+>/g, "|");
  assert.match(text, /\|Method\|+Two-stage\|/);
  for (const label of ["Latest dividend per share (D0)", "PV of stage-1 dividends",
    "PV of each stage-1 dividend", "Terminal growth (used)", "High-growth years",
    "Cost of equity", "Payout ratio"]) {
    assert.ok(text.includes(`|${label}|`), label);
  }
  assert.doesNotMatch(html, /two_stage|Pv Stage1|Stage1 Pvs|Cost Of Equity|capitalize/);
  // The method is the Method stat only, not a detail row too.
  assert.equal((text.match(/Two-stage/g) || []).length, 2); // the stat and the control
  const h = renderDDM({ ...DDM_REPORT, ddm: { ...DDM_REPORT.ddm, method: "h_model",
    detail: { method: "h_model", H_half: 2.5 } } });
  assert.match(h.replace(/<[^>]+>/g, "|"), /\|Method\|+H-model\|/);
  assert.match(h, /Half of the high-growth years \(H\)/);
  assert.doesNotMatch(h, /h_model/);
  // The FCFE's driver keys read in words too.
  const fcfe = renderDDM({ ...baseReport, fcfe: { implied_price: 30, cost_of_equity: 0.1,
    detail: { nwc_pct_delta_revenue: 0.05, da_pct_revenue: 0.04, delta_debt_policy: "constant" } } });
  assert.match(fcfe, /Working capital \(% of revenue change\)/);
  assert.match(fcfe, /D&amp;A \(% of revenue\)/);
  assert.doesNotMatch(fcfe, /Nwc Pct|nwc pct/);
});

test("a bank's EV multiples are marked not meaningful on the Multiples tab", () => {
  const target = { ticker: "JPM", name: "JPMorgan", currency: "USD", pe: 13.2, pb: 2.1,
    ev_ebitda: 130.9, ev_sales: 6.3 };
  const bank = { ...baseReport, comps: { ...COMPS, target },
    summary: { currency: "USD", financial_kind: "bank", financial_institution: "bank" } };
  const render = (report) => renderToStaticMarkup(React.createElement(MultiplesPanel,
    { report, enrichment: null }));
  const html = render(bank);
  const stat = (h, label) => h.slice(h.indexOf(`>${label}<`), h.indexOf("</div></div>", h.indexOf(`>${label}<`)));
  for (const label of ["EV/EBITDA", "EV/Sales"]) {
    assert.match(stat(html, label), /text-ink-faint">\d+\.\dx<\/div><div[^>]*>not meaningful \(equity multiples only\)/, label);
  }
  assert.match(html, /title="Not meaningful for a bank: most of its debt funds the business/);
  assert.doesNotMatch(stat(html, "P/E"), /not meaningful/);
  // An ordinary company and a REIT keep them plain.
  for (const summary of [{ currency: "USD" }, { currency: "USD", financial_kind: "reit" }]) {
    assert.doesNotMatch(render({ ...bank, summary }), /not meaningful/);
  }
});

test("the method summary keeps a no-target cell short; the reason is its tooltip", () => {
  const gm = { ...baseReport, football_field: [],
    company: { ...baseReport.company, market: {} },
    summary: { currency: "USD", current_price: 50, methods: { DCF: 80, DDM: 12, FCFE: 70 },
      blended_target: null, blended_upside: null, recommendation: "N/A",
      excluded_from_blend: { DCF: "not meaningful", FCFE: "not meaningful",
        DDM: "dividends only" }, financial_kind: "captive_finance" },
    comps: null };
  const html = renderToStaticMarkup(React.createElement(ValuationSummary, { report: gm }));
  assert.match(html, /Blended target<\/td><td[^>]*><span title="No target \(supply peers\)">No target<\/span>/);
});

test("the football field's price line runs through every row and its label sits beside it", () => {
  const rows = [{ method: "DCF", low: 300, base: 340, high: 388.06 },
    { method: "Comps (median)", low: 280, base: 320, high: 360 }];
  const render = (price) => renderToStaticMarkup(React.createElement(ValuationSummary, {
    report: { ...baseReport, current_price: price, football_field: rows,
      company: { ...baseReport.company, market: {} },
      summary: { currency: "USD", current_price: price, methods: {} } } }));
  const html = render(343.91);
  // One line segment per bar row and one in the label row; rows touch (no
  // gap between them), so the segments join up.
  assert.equal((html.match(/absolute inset-y-0 w-\[2px\] -translate-x-1\/2 bg-brand/g) || []).length, 3);
  assert.doesNotMatch(html, /space-y-3/);
  // The label is beside the line, not centred on it: to its left when the
  // price sits right of centre, else to its right.
  const label = /<div class="num absolute top-1 whitespace-nowrap text-\[10px\] text-brand" style="([^"]*)">price \$343\.91<\/div>/.exec(html);
  assert.ok(label, "no price label");
  assert.match(label[1], /^right:calc\([\d.]+% \+ 5px\)$/);
  assert.doesNotMatch(html, /-translate-x-1\/2 whitespace-nowrap text-\[10px\] text-brand/);
  const low = /text-brand" style="([^"]*)">price \$290\.00</.exec(render(290));
  assert.match(low[1], /^left:calc\([\d.]+% \+ 5px\)$/);
  // Each plot keeps a right gutter for the high labels, which never wrap,
  // so a bar ending at the right edge keeps its label inside the card.
  assert.equal((html.match(/class="relative mr-16 h-10 flex-1"/g) || []).length, 2);
  // On the card's background, so the price line passes behind it.
  assert.match(html, /whitespace-nowrap rounded-sm bg-surface px-0\.5 text-\[10px\] text-ink-faint" style="left:calc\([\d.]+% \+ 4px\)">\$388\.06</);
});
