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
