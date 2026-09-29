// Sensitivity grid formatting: both grid kinds (Gordon and exit multiple).
// Run with: npm test

import assert from "node:assert/strict";
import { test } from "node:test";
import { loadLib } from "./load-lib.mjs";

const { baseCaseCell, fmtAxis, isMultipleAxis } = loadLib("sensitivity");

const HEADLINE = 33.38483239072079;

function grid(centre) {
  const g = Array.from({ length: 5 }, (_, i) =>
    Array.from({ length: 5 }, (_, j) => 20 + i + j)
  );
  g[1][1] = null; // invalid combination
  g[2][2] = centre;
  return g;
}

const exitGrid = (centre) => ({
  title: "DCF implied price: WACC vs exit EV/EBITDA",
  row_label: "WACC",
  col_label: "Exit EV/EBITDA",
  row_values: [0.08, 0.087, 0.095, 0.102, 0.109],
  col_values: [10, 11, 12, 13, 14],
  grid: grid(centre),
});

test("axis values are formatted by the axis label", () => {
  assert.equal(isMultipleAxis("Exit EV/EBITDA"), true);
  assert.equal(isMultipleAxis("Exit multiple"), true);
  for (const label of ["WACC", "Terminal growth", "EBIT margin", "EBITDA margin"]) {
    assert.equal(isMultipleAxis(label), false, label);
  }
  const s = exitGrid(HEADLINE);
  assert.deepEqual(
    s.col_values.map((v) => fmtAxis(v, s.col_label)),
    ["10.0x", "11.0x", "12.0x", "13.0x", "14.0x"]
  );
  assert.deepEqual(
    s.row_values.map((v) => fmtAxis(v, s.row_label)),
    ["8.0%", "8.7%", "9.5%", "10.2%", "10.9%"]
  );
  // A rate axis stays a percentage at any magnitude; missing levels are n/a.
  assert.equal(fmtAxis(-1.52, "EBIT margin"), "-152.0%");
  assert.equal(fmtAxis(null, "EBIT margin"), "n/a");
  assert.equal(fmtAxis(Number.NaN, "Exit EV/EBITDA"), "n/a");
});

test("the centre is the base case only when it is the headline DCF price", () => {
  assert.deepEqual(baseCaseCell(exitGrid(HEADLINE), HEADLINE), [2, 2]);
  // A Gordon grid shown beside an exit-multiple headline: centre differs.
  assert.equal(baseCaseCell(exitGrid(HEADLINE * 0.8), HEADLINE), null);
  // An invalid centre, or no DCF at all, marks nothing.
  assert.equal(baseCaseCell(exitGrid(null), HEADLINE), null);
  assert.equal(baseCaseCell(exitGrid(HEADLINE), null), null);
  // An even-sized axis has no centre cell.
  const even = { ...exitGrid(HEADLINE), col_values: [10, 11, 12, 13] };
  assert.equal(baseCaseCell(even, HEADLINE), null);
});
