"use client";

// DCF workbench: editable assumptions on the left, model output + projection
// on the right. Follows ValuationSummary's structure, density, and tokens.

import React from "react";
import type { Assumptions, Report } from "@/lib/types";
import {
  betaStat,
  excludedReason,
  fmtBig,
  fmtCount,
  fmtMoney,
  fmtMult,
  fmtNum,
  fmtPct,
  toneForMethodUpside,
} from "@/lib/format";
import {
  AssumptionSlider,
  Button,
  Card,
  EmptyState,
  Segmented,
  Spinner,
  Stat,
  Table,
  TD,
  TH,
} from "@/components/ui";

export default function DCFPanel({
  report,
  assumptions,
  setAssumptions,
  onRecompute,
  recomputing,
}: {
  report: Report;
  assumptions: Assumptions;
  setAssumptions: (a: Assumptions) => void;
  onRecompute: (override?: Assumptions) => void;
  recomputing: boolean;
}) {
  const dcf = report.dcf;
  const cur = report.summary.currency;

  // --- Display defaults derived from the report --------------------------- //
  const fin = report.company.financials;
  const lastRev = fin.revenue?.at(-1) ?? 0;
  const lastEbit = fin.ebit?.at(-1) ?? 0;
  const dcfA = (dcf?.assumptions || {}) as Record<string, unknown>;
  // Left out of the blended target (e.g. a bank, a captive-finance group or a
  // lessor): the DCF is then shown for reference only.
  const notInBlend = excludedReason(report.summary, "DCF");
  const waccDetail = (dcf?.wacc?.detail || {}) as Record<string, unknown>;
  const beta = betaStat(
    dcf?.wacc?.beta,
    waccDetail.beta_raw as number | null | undefined,
    waccDetail.beta_source
  );

  // Display defaults come from the engine's OWN assumption echo
  // (dcf.assumptions) so the sliders always show what the model actually used
  // — the engine derives growth from historical CAGR and tax from the
  // effective historical rate when no override is given.
  const forecastYears =
    assumptions.forecast_years ?? (dcfA.forecast_years as number) ?? 5;
  const terminalGrowth =
    assumptions.terminal_growth ?? (dcfA.terminal_growth as number) ?? 0.025;
  const revenueGrowthY1 =
    assumptions.revenue_growth_y1 ??
    (dcfA.revenue_growth_path as number[] | undefined)?.[0] ??
    0.08;
  const targetEbitMargin =
    assumptions.target_ebit_margin ??
    (dcfA.target_ebit_margin as number | undefined) ??
    (lastRev ? lastEbit / lastRev : 0.2);
  const rf = assumptions.rf ?? report.macro.risk_free_rate;
  const erp = assumptions.erp ?? report.macro.equity_risk_premium;
  const taxRate =
    assumptions.tax_rate ??
    (dcfA.tax_rate as number | undefined) ??
    report.macro.tax_rate ??
    0.21;
  const terminalMethod = assumptions.terminal_method ?? "gordon";
  const exitEvEbitda = assumptions.exit_ev_ebitda ?? 12;

  // --- Projection series (all aligned to dcf?.years) ----------------------- //
  const years = dcf?.years || [];
  const rev = dcf?.revenue || [];
  const ebit = dcf?.ebit || [];
  const nopat = dcf?.nopat || [];
  const fcff = dcf?.fcff || [];
  const dfac = dcf?.discount_factors || [];
  const pvf = dcf?.pv_fcff || [];
  const sumPvFcff = pvf.reduce<number>(
    (acc, v) => acc + (typeof v === "number" && Number.isFinite(v) ? v : 0),
    0
  );

  const projectionRows: { label: string; cells: (i: number) => React.ReactNode }[] = [
    { label: "Revenue", cells: (i) => fmtBig(rev[i], cur) },
    { label: "EBIT", cells: (i) => fmtBig(ebit[i], cur) },
    {
      label: "EBIT margin",
      cells: (i) =>
        fmtPct(rev[i] ? (ebit[i] ?? 0) / rev[i] : null),
    },
    { label: "NOPAT", cells: (i) => fmtBig(nopat[i], cur) },
    { label: "FCFF", cells: (i) => fmtBig(fcff[i], cur) },
    { label: "Discount factor", cells: (i) => fmtNum(dfac[i], 3) },
    { label: "PV of FCFF", cells: (i) => fmtBig(pvf[i], cur) },
  ];

  // The engine deducts minority interest and preferred as well as net debt
  // (EV − net debt − minority − preferred = equity), so show them when present
  // or the bridge won't add up.
  const bs = report.company.balance_sheet;
  const claim = (v: number | null | undefined) =>
    typeof v === "number" && Number.isFinite(v) && v !== 0 ? v : null;
  const minority = claim(bs?.minority_interest);
  const preferred = claim(bs?.preferred_equity);

  const bridge: { label: string; value: React.ReactNode; strong?: boolean }[] = [
    { label: "Sum PV(FCFF)", value: fmtBig(sumPvFcff, cur) },
    { label: "Terminal value", value: fmtBig(dcf?.terminal_value, cur) },
    { label: "PV of terminal", value: fmtBig(dcf?.pv_terminal, cur) },
    { label: "Enterprise value", value: fmtBig(dcf?.enterprise_value, cur), strong: true },
    { label: "(−) Net debt", value: fmtBig(dcf?.net_debt, cur) },
    ...(minority != null
      ? [{ label: "(−) Minority interest", value: fmtBig(minority, cur) }]
      : []),
    ...(preferred != null
      ? [{ label: "(−) Preferred equity", value: fmtBig(preferred, cur) }]
      : []),
    { label: "Equity value", value: fmtBig(dcf?.equity_value, cur), strong: true },
    { label: "Shares", value: fmtCount(dcf?.shares) },
    { label: "Implied price", value: fmtMoney(dcf?.implied_price, cur), strong: true },
  ];

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
      {/* --- Controls -------------------------------------------------- */}
      <Card title="Assumptions" subtitle="Drivers feeding the DCF model">
        <div className="space-y-4">
          <AssumptionSlider
            label="Forecast years"
            value={forecastYears}
            min={3}
            max={10}
            step={1}
            format={(v) => `${v}y`}
            onChange={(v) =>
              setAssumptions({ ...assumptions, forecast_years: Math.round(v) })
            }
          />
          <AssumptionSlider
            label="Terminal growth"
            value={terminalGrowth}
            min={0}
            max={0.05}
            step={0.0025}
            format={(v) => fmtPct(v)}
            hint={
              terminalMethod === "exit_multiple"
                ? "Not in the terminal value (the exit multiple sets it); still ends the revenue-growth fade and drives DDM/FCFE"
                : undefined
            }
            onChange={(v) =>
              setAssumptions({ ...assumptions, terminal_growth: v })
            }
          />
          <AssumptionSlider
            label="Near-term revenue growth"
            value={revenueGrowthY1}
            min={-0.1}
            max={0.4}
            step={0.005}
            format={(v) => fmtPct(v)}
            hint="Year-1 revenue growth; fades to terminal"
            onChange={(v) =>
              setAssumptions({ ...assumptions, revenue_growth_y1: v, revenue_growth: undefined })
            }
          />
          <AssumptionSlider
            label="Target EBIT margin"
            value={targetEbitMargin}
            min={0}
            max={0.6}
            step={0.005}
            format={(v) => fmtPct(v)}
            hint="Terminal EBIT margin"
            onChange={(v) =>
              setAssumptions({ ...assumptions, target_ebit_margin: v })
            }
          />
          <AssumptionSlider
            label="Risk-free rate"
            value={rf}
            min={0}
            max={0.08}
            step={0.001}
            format={(v) => fmtPct(v)}
            onChange={(v) => setAssumptions({ ...assumptions, rf: v })}
          />
          <AssumptionSlider
            label="Equity risk premium"
            value={erp}
            min={0.02}
            max={0.09}
            step={0.0025}
            format={(v) => fmtPct(v)}
            onChange={(v) => setAssumptions({ ...assumptions, erp: v })}
          />
          <AssumptionSlider
            label="Tax rate"
            value={taxRate}
            min={0}
            max={0.4}
            step={0.005}
            format={(v) => fmtPct(v)}
            onChange={(v) => setAssumptions({ ...assumptions, tax_rate: v })}
          />

          <div>
            <div className="mb-1.5 text-xs font-medium text-ink-dim">
              Terminal method
            </div>
            <Segmented<"gordon" | "exit_multiple">
              value={terminalMethod}
              options={[
                { label: "Gordon", value: "gordon" },
                { label: "Exit multiple", value: "exit_multiple" },
              ]}
              onChange={(v) =>
                // Write the displayed multiple into the assumptions too —
                // otherwise the engine receives exit_multiple with no multiple
                // and silently falls back to Gordon while the UI shows 12x.
                setAssumptions({
                  ...assumptions,
                  terminal_method: v,
                  ...(v === "exit_multiple"
                    ? { exit_ev_ebitda: assumptions.exit_ev_ebitda ?? 12 }
                    : {}),
                })
              }
            />
          </div>

          {terminalMethod === "exit_multiple" && (
            <AssumptionSlider
              label="Exit EV / EBITDA"
              value={exitEvEbitda}
              min={4}
              max={30}
              step={0.5}
              format={(v) => fmtMult(v)}
              hint="Terminal EV/EBITDA multiple"
              onChange={(v) =>
                setAssumptions({ ...assumptions, exit_ev_ebitda: v })
              }
            />
          )}

          <div className="border-t border-line pt-3">
            <Button onClick={() => onRecompute()} disabled={recomputing}>
              {recomputing ? (
                <>
                  <Spinner /> Recomputing
                </>
              ) : (
                "Recompute"
              )}
            </Button>
            <Button variant="ghost" className="ml-2" disabled={recomputing} onClick={() => {
              setAssumptions({});
              onRecompute({});
            }}>Reset defaults</Button>
            <p className="mt-2 text-[11px] text-ink-faint">
              Adjust drivers, then recompute (re-uses cached data — instant).
            </p>
          </div>
        </div>
      </Card>

      {/* --- Results --------------------------------------------------- */}
      {dcf ? <Card
        title="DCF output"
        subtitle={
          notInBlend
            ? `Reference only, not in the blended target: ${notInBlend}`
            : undefined
        }
        className="lg:col-span-2"
      >
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
          <Stat
            label={notInBlend ? "Implied price (ref.)" : "Implied price"}
            value={fmtMoney(dcf?.implied_price, cur)}
            tone={toneForMethodUpside(report.summary, "DCF", dcf?.upside)}
            sub={fmtPct(dcf?.upside, { signed: true })}
          />
          <Stat label="WACC" value={fmtPct(dcf?.wacc?.wacc)} />
          <Stat
            label="Cost of equity"
            value={fmtPct(dcf?.wacc?.cost_of_equity)}
          />
          <Stat
            label="After-tax cost of debt"
            value={fmtPct(dcf?.wacc?.after_tax_cost_of_debt)}
          />
          <Stat
            label={beta.label}
            value={beta.value}
            sub={beta.sub}
            title={beta.title}
          />
          <Stat
            label="Equity weight"
            value={fmtPct(dcf?.wacc?.weight_equity)}
          />
        </div>

        {/* Reverse DCF — what the market price implies */}
        {report.reverse_dcf && (
          <div className="mt-4 rounded-lg border border-brand/30 bg-brand/5 px-4 py-3">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-brand">
              Reverse DCF — what&apos;s priced in
            </div>
            {report.reverse_dcf?.converged &&
            report.reverse_dcf?.implied_growth_y1 != null ? (
              <div className="mt-1 flex flex-wrap items-baseline gap-x-6 gap-y-1">
                <div>
                  <span className="num text-lg font-semibold text-ink">
                    {fmtPct(report.reverse_dcf?.implied_growth_y1)}
                  </span>
                  <span className="ml-2 text-xs text-ink-dim">
                    year-1 revenue growth implied by the market price
                  </span>
                </div>
                <div className="text-xs text-ink-dim">
                  vs{" "}
                  <span className="num font-medium text-ink">
                    {fmtPct(report.reverse_dcf.current_assumption_y1 ??
                      (dcfA.revenue_growth_path as number[] | undefined)?.[0])}
                  </span>{" "}
                  in the computed model — the gap is what you&apos;d have to believe to
                  own it at this price.
                </div>
              </div>
            ) : (
              <p className="mt-1 text-xs text-ink-dim">
                {report.reverse_dcf?.note ||
                  "No growth rate in a plausible range reproduces the market price with the current assumptions."}
              </p>
            )}
            <p className="mt-2 text-[11px] text-ink-faint">
              The near-term growth shift fades to zero by the final forecast year.
            </p>
          </div>
        )}

        <div className="mt-4 border-t border-line pt-3">
          <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-ink-faint">
            Projection
          </div>
          <Table>
            <thead>
              <tr>
                <TH align="left">Driver</TH>
                {years.map((y, i) => (
                  <TH key={`${y}-${i}`}>{y}</TH>
                ))}
              </tr>
            </thead>
            <tbody>
              {projectionRows.map((row) => (
                <tr key={row.label}>
                  <TH align="left">{row.label}</TH>
                  {years.map((y, i) => (
                    <TD key={`${row.label}-${y}-${i}`} num>
                      {row.cells(i)}
                    </TD>
                  ))}
                </tr>
              ))}
            </tbody>
          </Table>
        </div>

        <div className="mt-4 border-t border-line pt-3">
          <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-ink-faint">
            Bridge to equity value
          </div>
          <div className="divide-y divide-line/60">
            {bridge.map((b) => (
              <div
                key={b.label}
                className="flex items-baseline justify-between py-1.5"
              >
                <span
                  className={
                    b.strong
                      ? "text-sm font-semibold text-ink"
                      : "text-sm text-ink-dim"
                  }
                >
                  {b.label}
                </span>
                <span
                  className={
                    b.strong
                      ? "num text-sm font-semibold text-ink"
                      : "num text-sm text-ink"
                  }
                >
                  {b.value}
                </span>
              </div>
            ))}
          </div>
        </div>
      </Card> : (
        <Card title="DCF unavailable" className="lg:col-span-2">
          <EmptyState title="No valid DCF at the current inputs"
            hint={(report.warnings || []).join(" · ") || "Adjust the assumptions and recompute."} />
        </Card>
      )}
    </div>
  );
}
