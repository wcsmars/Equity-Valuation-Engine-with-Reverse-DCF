import assert from "node:assert/strict";
import { test } from "node:test";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { loadComponent } from "./load-component.mjs";

const DCFPanel = loadComponent("components/DCFPanel.tsx").default;
const DDMFCFEPanel = loadComponent("components/DDMFCFEPanel.tsx").default;
const CompsPanel = loadComponent("components/CompsPanel.tsx").default;
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
    dcf: { assumptions: { revenue_growth_path: [0.08] }, wacc: {} },
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
