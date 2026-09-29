// Sensitivity-grid helpers shared by SensitivityPanel (and its tests).
//
// The engine builds two kinds of grid: WACC x terminal growth for a Gordon
// DCF, and WACC x exit EV/EBITDA for an exit-multiple DCF. Axis values are
// formatted by the axis label, the same rule the Excel and HTML reports use,
// and the centre cell counts as the base case only when it reproduces the
// headline DCF price.

import type { Sensitivity } from "./types";
import { fmtMult, fmtPct } from "./format";

// An axis whose label mentions "EV/EBITDA" or "multiple" holds multiples
// (12.0 -> "12.0x"); every other axis the engine builds (WACC, terminal
// growth, EBIT margin) holds decimal rates.
export function isMultipleAxis(label: string | null | undefined): boolean {
  const l = (label || "").toLowerCase();
  return l.includes("ev/ebitda") || l.includes("multiple");
}

// One axis level, formatted by its axis label; "n/a" when missing.
export function fmtAxis(v: number | null | undefined, label: string): string {
  if (typeof v !== "number" || !Number.isFinite(v)) return "n/a";
  return isMultipleAxis(label) ? fmtMult(v) : fmtPct(v);
}

// [row, col] of the centre cell when it is the headline DCF case, else null.
// The centre of an odd-sized grid holds the unshifted inputs, so it should
// reproduce the headline DCF price; when it does not (e.g. a Gordon grid next
// to an exit-multiple headline, or an invalid centre) it is not the base case.
export function baseCaseCell(
  s: Sensitivity,
  headline: number | null | undefined
): [number, number] | null {
  const rows = s.row_values || [];
  const cols = s.col_values || [];
  if (
    typeof headline !== "number" ||
    !Number.isFinite(headline) ||
    rows.length % 2 === 0 ||
    cols.length % 2 === 0
  ) {
    return null;
  }
  const i = Math.floor(rows.length / 2);
  const j = Math.floor(cols.length / 2);
  const cell = (s.grid || [])[i]?.[j];
  if (typeof cell !== "number" || !Number.isFinite(cell)) return null;
  const tol = Math.max(1e-6, 1e-6 * Math.abs(headline));
  return Math.abs(cell - headline) <= tol ? [i, j] : null;
}
