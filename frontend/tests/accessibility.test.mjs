// Accessibility of the shared controls: text contrast against the design
// tokens, named sliders, announced selection state, keyboard-reachable
// watchlist cards and alerts. Rendered with the project's own React.
// Run with: npm test

import assert from "node:assert/strict";
import { test } from "node:test";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { loadComponent } from "./load-component.mjs";

const ui = loadComponent("components/ui.tsx");
const { WatchlistGrid, WatchToggle } = loadComponent("components/Watchlist.tsx");
const { ApiKeysForm, ApiKeysToggle } = loadComponent("components/ApiKeys.tsx");
const colors = loadComponent("tailwind.config.ts").default.theme.extend.colors;
const h = React.createElement;
const html = (el) => renderToStaticMarkup(el);

// WCAG 2.x contrast ratio of two #rrggbb colours.
function contrast(a, b) {
  const lum = (hex) => {
    const [r, g, b2] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
      .map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
    return 0.2126 * r + 0.7152 * g + 0.0722 * b2;
  };
  const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

test("text tokens meet WCAG AA contrast on every surface", () => {
  const surfaces = [colors.bg, colors.surface.DEFAULT, colors.surface.raised, colors.surface.hi];
  for (const [name, fg] of Object.entries(colors.ink)) {
    for (const bg of surfaces) {
      assert.ok(contrast(fg, bg) >= 4.5, `ink-${name} ${fg} on ${bg}: ${contrast(fg, bg).toFixed(2)}`);
    }
  }
  // White text sits on brand-dim and brand-deep; the accent brand fails it.
  assert.ok(contrast("#ffffff", colors.brand.dim) >= 4.5);
  assert.ok(contrast("#ffffff", colors.brand.deep) >= 4.5);
  assert.ok(contrast("#ffffff", colors.brand.DEFAULT) < 4.5);
});

test("filled buttons and the selected option use the darker brand fill", () => {
  const button = html(h(ui.Button, null, "Analyze"));
  assert.match(button, /bg-brand-dim text-white hover:bg-brand-deep/);
  assert.doesNotMatch(button, /\bbg-brand text-white/);
  const seg = html(h(ui.Segmented, {
    label: "Terminal method", value: "gordon", onChange() {},
    options: [{ label: "Gordon", value: "gordon" }, { label: "Exit multiple", value: "exit_multiple" }],
  }));
  assert.match(seg, /role="group" aria-label="Terminal method"/);
  assert.match(seg, /type="button" aria-pressed="true" class="[^"]*bg-brand-dim text-white[^"]*">Gordon/);
  assert.match(seg, /aria-pressed="false"[^>]*>Exit multiple/);
});

test("each assumption slider is named by its label and reads its value as shown", () => {
  const markup = html(h("div", null,
    h(ui.AssumptionSlider, { label: "Terminal growth", value: 0.025, min: 0, max: 0.05,
      step: 0.0025, format: (v) => `${(v * 100).toFixed(1)}%`, onChange() {} }),
    h(ui.AssumptionSlider, { label: "Forecast years", value: 5, min: 1, max: 15, step: 1,
      format: (v) => `${v}y`, valueText: (v) => `${v} years`, hint: "Explicit years", onChange() {} })));
  const labels = [...markup.matchAll(/<label for="([^"]+)"[^>]*>([^<]+)<\/label>/g)];
  assert.deepEqual(labels.map((m) => m[2]), ["Terminal growth", "Forecast years"]);
  const ids = [...markup.matchAll(/<input id="([^"]+)" type="range"/g)].map((m) => m[1]);
  assert.deepEqual(ids, labels.map((m) => m[1]));
  assert.notEqual(ids[0], ids[1]);
  assert.match(markup, /aria-valuetext="2\.5%"/);
  assert.match(markup, /aria-valuetext="5 years"/);
  // The hint describes its slider.
  assert.match(markup, new RegExp(`aria-describedby="${ids[1]}-hint"`));
  assert.match(markup, new RegExp(`<p id="${ids[1]}-hint"[^>]*>Explicit years`));
});

test("the open tab is marked current, not only coloured", () => {
  const markup = html(h(ui.TabBar, {
    label: "Report sections", tabs: ["Overview", "Valuation", "News"], value: "Valuation",
    onChange() {}, badge: (t) => (t === "News" ? h("span", null, "3") : null),
  }));
  assert.match(markup, /role="group" aria-label="Report sections"/);
  assert.equal((markup.match(/aria-current="true"/g) || []).length, 1);
  assert.match(markup, /aria-current="true"[^>]*>Valuation</);
  assert.match(markup, />News<span>3<\/span>/);
});

test("error banners are announced", () => {
  assert.match(html(h(ui.Alert, null, "Could not load APPL")), /^<div role="alert"[^>]*text-down[^>]*>Could not load APPL/);
  assert.match(html(h(ui.Alert, { tone: "flat" }, "Saving failed")), /role="alert"[^>]*text-flat/);
});

test("watchlist cards open from the keyboard and their remove button shows on focus", () => {
  const markup = html(h(WatchlistGrid, {
    items: [{ ticker: "AAPL", name: "Apple Inc.", currency: "USD", price: 200,
      blended_target: 150, recommendation: "Overvalued", updated_at: "2026-10-18" }],
    onOpen() {}, onRemove() {},
  }));
  // The opener is a real button holding the card's content.
  assert.match(markup, /<button type="button" class="block w-full[^"]*">.*AAPL.*Apple Inc\..*Overvalued/);
  // The remove button is a sibling (not nested), named, and visible on focus.
  const remove = /<button type="button" aria-label="Remove AAPL from watchlist" title="Remove" class="([^"]*)"/.exec(markup);
  assert.ok(remove, markup);
  assert.match(remove[1], /focus-visible:opacity-100/);
  assert.match(remove[1], /group-focus-within:opacity-100/);
  assert.equal((markup.match(/<button/g) || []).length, 2);
  assert.doesNotMatch(markup, /<button[^>]*>(?:(?!<\/button>).)*<button/);
});

test("the watch star announces its state and purpose", () => {
  const on = html(h(WatchToggle, { ticker: "NOK", watching: true, onToggle() {} }));
  assert.match(on, /aria-pressed="true" aria-label="Watch NOK"/);
  assert.match(on, /<span aria-hidden="true">★<\/span>/);
  const off = html(h(WatchToggle, { ticker: "NOK", watching: false, onToggle() {} }));
  assert.match(off, /aria-pressed="false"/);
});

test("API keys can be replaced once both providers are configured", () => {
  const both = { fmp_enabled: true, anthropic_enabled: true };
  assert.match(html(h(ApiKeysToggle, { status: both, open: false, onToggle() {} })), />API keys<\/button>/);
  assert.match(html(h(ApiKeysToggle, { status: { ...both, fmp_enabled: false }, open: false,
    onToggle() {} })), />Add API keys<\/button>/);
  const form = html(h(ApiKeysForm, { status: both, anthropicKey: "", fmpKey: "",
    setAnthropicKey() {}, setFmpKey() {}, saving: false, onSubmit() {} }));
  assert.equal((form.match(/type="password"/g) || []).length, 2);
  assert.equal((form.match(/placeholder="configured — paste to replace"/g) || []).length, 2);
  assert.match(form, /A blank field keeps the stored key/);
  // Unconfigured providers keep their usual hints.
  const none = html(h(ApiKeysForm, { status: { fmp_enabled: false, anthropic_enabled: false },
    anthropicKey: "", fmpKey: "", setAnthropicKey() {}, setFmpKey() {}, saving: false, onSubmit() {} }));
  assert.match(none, /placeholder="sk-ant-…"/);
  assert.match(none, /placeholder="optional"/);
});
