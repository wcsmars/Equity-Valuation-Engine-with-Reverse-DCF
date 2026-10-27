"use client";

// DCF workbench: editable assumptions on the left, model output + projection
// on the right. Follows ValuationSummary's structure, density, and tokens.
//
// Edits are a draft until recomputed: an amber bar lists the edits the
// report on screen does not reflect yet (with Recompute and Discard), and a
// changed slider shows its value in amber. When the engine leaves the DCF
// out of the blended target because of the issuer (a bank, insurer, REIT,
// lender, captive-finance group or lessor), the drivers are collapsed behind
// a note saying they do not move the target, and point to what does: the
// DDM drivers (when the DDM is in the blend) and the peer set. A DCF left out
// only because it failed at these inputs keeps its drivers open, as a
// recompute at other inputs can bring it back.

import React from "react";
import type { Assumptions, Report, TerminalMethod } from "@/lib/types";
import {
  assumptionsFromUsed,
  derivedValues,
  diffAssumptions,
  keepOnDcfReset,
} from "@/lib/assumptions";
import {
  betaStat,
  dcfWarnings,
  excludedReason,
  financialWhom,
  fmtBig,
  fmtCount,
  fmtMoney,
  fmtMult,
  fmtNum,
  fmtPct,
  fmtRate,
  projectionYearLabels,
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
  UnappliedBar,
} from "@/components/ui";

// The DCF drivers of a company whose DCF is reference only: collapsed behind
// a note (the drivers do not move the blended target) with pointers to what
// does, and dimmed when shown. The DDM drivers are named only when `ddmMoves`
// (there is a DDM and it is in the blend); otherwise the peer set alone moves
// the target. Its own component so the panel keeps no state (keyed by ticker,
// it starts collapsed for each company).
function ReferenceOnlyDrivers({
  whom,
  why,
  ddmMoves,
  children,
}: {
  whom: string | null;
  why: string;
  ddmMoves: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = React.useState(false);
  const id = React.useId();
  return (
    <div>
      <div className="rounded-lg border border-line bg-surface-raised/60 px-3 py-2.5">
        <p className="text-xs font-medium text-ink-dim">
          {whom ? `Reference-only for ${whom}` : "Reference-only"}: these
          drivers do not move the blended target.
        </p>
        <p className="mt-1 text-[11px] text-ink-faint">
          The DCF is left out of the target ({why}). What moves it:{" "}
          {ddmMoves
            ? "the DDM drivers in the dividend discount card and the peer set in the Trading comps box, below."
            : "the peer set in the Trading comps box, below."}
        </p>
        <button
          type="button"
          aria-expanded={open}
          aria-controls={id}
          onClick={() => setOpen((v) => !v)}
          className="mt-2 text-[11px] font-medium text-brand hover:underline"
        >
          {open ? "Hide the DCF drivers" : "Show the DCF drivers"}
        </button>
      </div>
      {open && (
        <div id={id} className="mt-4 opacity-75">
          {children}
        </div>
      )}
    </div>
  );
}

// One solved driver of the reverse DCF: the value at which the DCF equals the
// price, against the model's; else the solver's note (or "n/a").
function ReverseRow({
  label,
  implied,
  current,
  converged,
  note,
}: {
  label: string;
  implied: number | null | undefined;
  current: number | null | undefined;
  converged: boolean;
  note?: string | null;
}) {
  const solved = converged && typeof implied === "number" && Number.isFinite(implied);
  return (
    <div className="flex flex-wrap items-baseline justify-between gap-x-3">
      <span className="text-ink-dim">{label}</span>
      {solved ? (
        <span className="text-ink-dim" title={note || undefined}>
          <span className="num font-semibold text-ink">{fmtPct(implied)}</span> vs{" "}
          <span className="num text-ink">{fmtPct(current)}</span> in the model
          {note ? " *" : ""}
        </span>
      ) : (
        <span className="min-w-0 text-right text-ink-faint">{note || "n/a"}</span>
      )}
    </div>
  );
}

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
  // Left out because of the issuer (the engine flags it as a financial of
  // some kind), so no DCF driver moves the target. A DCF left out only
  // because it failed at these inputs ("non-finite implied price") is not:
  // its drivers are how to bring it back.
  const whom = financialWhom(report.summary);
  const referenceOnly = notInBlend != null && whom != null;
  // The DDM moves the target only when there is one and it is in the blend
  // (not a flagged non-payer, nor a low-payout or lone DDM left out).
  const ddmMoves = report.ddm != null && !excludedReason(report.summary, "DDM");
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
  const terminalMethod: TerminalMethod = assumptions.terminal_method ?? "gordon";
  const exitEvEbitda = assumptions.exit_ev_ebitda ?? 12;
  // Value-driver RONIC: the engine's default is WACC.
  const ronic =
    assumptions.terminal_ronic ??
    (dcfA.terminal_ronic_used as number | null | undefined) ??
    dcf?.wacc?.wacc ??
    0.1;
  // The engine's own figures for each driver. A slider's range always takes
  // them in (see AssumptionSlider), so a loss-maker's negative margin, a
  // margin above 60% or a market-implied growth rate can be shown and reached.
  const num = (x: unknown) => (typeof x === "number" ? x : null);
  const histMargin = lastRev ? lastEbit / lastRev : null;

  // Edits not yet reflected in the report on screen (none to compare
  // against when the report carries no echo). An input the report left to
  // the engine shows the figure it derived ("derived 31.97% → 30.0%").
  const used = report.assumptions_used;
  const changes = used
    ? diffAssumptions(assumptions, assumptionsFromUsed(used), derivedValues(report))
    : [];
  const changed = new Set<string>(changes.map((c) => c.key));
  const reset = () => {
    // Reset the DCF drivers only: the peer set and comps basis belong to the
    // comps card and the DDM drivers to the DDM card. The same object goes to
    // both calls, so the recompute replaces it with the defaults the engine
    // used.
    const keep = keepOnDcfReset(assumptions);
    setAssumptions(keep);
    onRecompute(keep);
  };

  // Terminal-value check (the engine's; all null for an exit multiple): the
  // net reinvestment the capex, D&A and working-capital drivers carry into
  // the first stable year, as a share of its NOPAT, and the return on new
  // capital the terminal growth then implies.
  const usedMethod = dcfA.terminal_method as TerminalMethod | undefined;
  const reinvestRate = num(dcfA.terminal_reinvestment_rate);
  const impliedRonic = num(dcfA.implied_terminal_ronic);
  const latestRoic = num(dcfA.latest_roic);
  const ronicUsed = num(dcfA.terminal_ronic_used);
  const gUsed = num(dcfA.terminal_growth_used) ?? num(dcfA.terminal_growth);
  // Not for a reference-only DCF (the engine skips its notes there too).
  const showTerminalCheck =
    !referenceOnly && usedMethod !== "exit_multiple" && "terminal_reinvestment_rate" in dcfA;
  // A value-driver terminal value reinvests g / RONIC of NOPAT by
  // construction (FCFF = NOPAT x (1 - g / RONIC)), whatever the drivers
  // carry, so it shows that rate and the RONIC used, and is never flagged
  // (the engine checks only a Gordon terminal value).
  const valueDriver = usedMethod === "value_driver";
  const tvReinvestRate =
    gUsed != null && ronicUsed != null && ronicUsed > 0 ? gUsed / ronicUsed : null;
  // A Gordon terminal value (or one from an older backend that does not echo
  // the method) reads amber on the engine's own checks: growth with no net
  // reinvestment, or on far less than the business has needed (implied
  // RONIC above 100% and 3x the latest ROIC). Used only when !valueDriver.
  const noReinvestment = gUsed != null && gUsed > 0 && reinvestRate != null && reinvestRate <= 0;
  const ronicHigh =
    impliedRonic != null && impliedRonic > Math.max(latestRoic != null ? 3 * latestRoic : 0, 1);
  const rdcf = report.reverse_dcf;

  // --- Projection series (all aligned to dcf?.years) ----------------------- //
  const years = dcf?.years || [];
  // The engine numbers the forecast years 1..n; the headers show them as
  // fiscal years after the latest one, as the FCFE table does.
  const yearLabels = projectionYearLabels(years, fin?.fiscal_years);
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

  // A reference-only DCF's drivers sit behind ReferenceOnlyDrivers.
  const gate = (controls: React.ReactNode) =>
    referenceOnly ? (
      <ReferenceOnlyDrivers
        key={report.summary.ticker ?? ""}
        whom={whom}
        why={notInBlend}
        ddmMoves={ddmMoves}
      >
        {controls}
      </ReferenceOnlyDrivers>
    ) : (
      controls
    );

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
      {/* --- Controls -------------------------------------------------- */}
      <Card title="Assumptions" subtitle="Drivers feeding the DCF model">
        <div className="space-y-4">
          {gate(
            <div className="space-y-4">
              <AssumptionSlider
                label="Forecast years"
                value={forecastYears}
                min={1}
                max={15}
                step={1}
                anchors={[num(dcfA.forecast_years)]}
                format={(v) => `${v}y`}
                valueText={(v) => `${v} years`}
                changed={changed.has("forecast_years")}
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
                anchors={[num(dcfA.terminal_growth)]}
                format={(v) => fmtRate(v)}
                changed={changed.has("terminal_growth")}
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
                anchors={[
                  (dcfA.revenue_growth_path as number[] | undefined)?.[0],
                  report.reverse_dcf?.implied_growth_y1,
                ]}
                format={(v) => fmtRate(v)}
                hint="Year-1 revenue growth; fades to terminal"
                changed={changed.has("revenue_growth_y1") || changed.has("revenue_growth")}
                onChange={(v) =>
                  setAssumptions({ ...assumptions, revenue_growth_y1: v, revenue_growth: undefined })
                }
              />
              <AssumptionSlider
                label="Target EBIT margin"
                value={targetEbitMargin}
                min={-0.3}
                max={0.6}
                step={0.005}
                anchors={[num(dcfA.target_ebit_margin), histMargin]}
                format={(v) => fmtRate(v)}
                hint="Terminal EBIT margin"
                changed={changed.has("target_ebit_margin")}
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
                anchors={[report.macro.risk_free_rate]}
                format={(v) => fmtRate(v)}
                changed={changed.has("rf")}
                onChange={(v) => setAssumptions({ ...assumptions, rf: v })}
              />
              <AssumptionSlider
                label="Equity risk premium"
                value={erp}
                min={0.02}
                max={0.09}
                step={0.0025}
                anchors={[report.macro.equity_risk_premium]}
                format={(v) => fmtRate(v)}
                changed={changed.has("erp")}
                onChange={(v) => setAssumptions({ ...assumptions, erp: v })}
              />
              <AssumptionSlider
                label="Tax rate"
                value={taxRate}
                min={0}
                max={0.4}
                step={0.005}
                anchors={[num(dcfA.tax_rate), report.macro.tax_rate]}
                format={(v) => fmtRate(v)}
                changed={changed.has("tax_rate")}
                onChange={(v) => setAssumptions({ ...assumptions, tax_rate: v })}
              />

              <div>
                <div className="mb-1.5 text-xs font-medium text-ink-dim">
                  Terminal method
                  {changed.has("terminal_method") && (
                    <span className="ml-1.5 text-[11px] font-normal text-flat">changed</span>
                  )}
                </div>
                <Segmented<TerminalMethod>
                  label="Terminal method"
                  value={terminalMethod}
                  options={[
                    { label: "Gordon", value: "gordon" },
                    { label: "Exit multiple", value: "exit_multiple" },
                    { label: "Value driver", value: "value_driver" },
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
                  anchors={[num(dcfA.exit_ev_ebitda)]}
                  format={(v) => fmtMult(v)}
                  hint="Terminal EV/EBITDA multiple"
                  changed={changed.has("exit_ev_ebitda")}
                  onChange={(v) =>
                    setAssumptions({ ...assumptions, exit_ev_ebitda: v })
                  }
                />
              )}

              {terminalMethod === "value_driver" && (
                <AssumptionSlider
                  label="Terminal RONIC"
                  value={ronic}
                  min={0.02}
                  max={0.4}
                  step={0.0025}
                  anchors={[num(dcfA.terminal_ronic_used), dcf?.wacc?.wacc]}
                  format={(v) => fmtRate(v)}
                  hint="Return on new invested capital after the forecast: terminal FCFF = NOPAT × (1 − g / RONIC). At WACC (the default) growth adds no value; it must exceed terminal growth."
                  changed={changed.has("terminal_ronic")}
                  onChange={(v) =>
                    setAssumptions({ ...assumptions, terminal_ronic: v })
                  }
                />
              )}
            </div>
          )}

          <UnappliedBar
            changes={changes}
            recomputing={recomputing}
            onRecompute={() => onRecompute()}
            onDiscard={() => used && setAssumptions(assumptionsFromUsed(used))}
          />

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
            <Button variant="ghost" className="ml-2" disabled={recomputing} onClick={reset}>
              Reset defaults
            </Button>
            <p className="mt-2 text-[11px] text-ink-faint">
              Adjust drivers, then recompute (re-uses cached data — instant).
              Reset keeps the peer set, the comps basis and the DDM drivers.
            </p>
          </div>
        </div>
      </Card>

      {/* --- Results --------------------------------------------------- */}
      {dcf && dcf.implied_price != null ? <Card
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
          {showTerminalCheck && valueDriver && (
            <>
              <Stat
                label="Terminal reinvestment"
                value={fmtPct(tvReinvestRate)}
                sub={reinvestRate != null ? `of NOPAT · drivers ${fmtPct(reinvestRate)}` : "of NOPAT"}
                title="The value-driver terminal value reinvests terminal growth ÷ RONIC of the first stable year's NOPAT by construction (FCFF = NOPAT × (1 − g / RONIC)). The drivers' rate is the net reinvestment (capex − D&A + change in working capital) the forecast drivers would carry into that year, for comparison; it does not enter this terminal value."
              />
              <Stat
                label="RONIC used"
                value={fmtPct(ronicUsed)}
                sub={`latest ROIC ${fmtPct(latestRoic)}`}
                title="The return on new invested capital the value-driver terminal value assumes (the engine's default is WACC, at which growth adds no value)."
              />
            </>
          )}
          {showTerminalCheck && !valueDriver && (
            <>
              <Stat
                label="Terminal reinvestment"
                value={fmtPct(reinvestRate)}
                sub="of NOPAT"
                tone={noReinvestment ? "text-flat" : undefined}
                title="Net reinvestment (capex − D&A + change in working capital) in the first year after the forecast, as a share of that year's NOPAT."
              />
              <Stat
                label="Implied RONIC"
                value={noReinvestment ? "n/a" : fmtPct(impliedRonic)}
                sub={noReinvestment ? "growth without reinvestment" : `latest ROIC ${fmtPct(latestRoic)}`}
                tone={noReinvestment || ronicHigh ? "text-flat" : undefined}
                title="The return on new invested capital the terminal growth implies: terminal growth ÷ reinvestment rate. Far above the latest ROIC, the terminal growth rests on too little reinvestment."
              />
            </>
          )}
        </div>

        {/* Reverse DCF — what the market price implies. When the DCF is
            reference only, the solved rate only inverts that DCF, so the box
            is labelled reference only and never called what is priced in. */}
        {report.reverse_dcf && (
          <div className="mt-4 rounded-lg border border-brand/30 bg-brand/5 px-4 py-3">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-brand">
              {notInBlend ? "Reverse DCF (reference only)" : <>Reverse DCF — what&apos;s priced in</>}
            </div>
            {report.reverse_dcf?.converged &&
            report.reverse_dcf?.implied_growth_y1 != null ? (
              <div className="mt-1 flex flex-wrap items-baseline gap-x-6 gap-y-1">
                <div>
                  <span className="num text-lg font-semibold text-ink">
                    {fmtPct(report.reverse_dcf?.implied_growth_y1)}
                  </span>
                  <span className="ml-2 text-xs text-ink-dim">
                    {notInBlend
                      ? "year-1 revenue growth at which the reference DCF equals the price"
                      : "year-1 revenue growth implied by the market price"}
                  </span>
                </div>
                <div className="text-xs text-ink-dim">
                  vs{" "}
                  <span className="num font-medium text-ink">
                    {fmtPct(report.reverse_dcf.current_assumption_y1 ??
                      (dcfA.revenue_growth_path as number[] | undefined)?.[0])}
                  </span>{" "}
                  {notInBlend
                    ? `in the computed model — this inverts the reference-only DCF above (${notInBlend}), so it is not a market-implied growth rate.`
                    : <>in the computed model — the gap is what you&apos;d have to believe to own it at this price.</>}
                </div>
              </div>
            ) : (
              <p className="mt-1 text-xs text-ink-dim">
                {report.reverse_dcf?.note ||
                  "No growth rate in a plausible range reproduces the market price with the current assumptions."}
              </p>
            )}
            {/* A caveat that comes with a solved rate (more than one growth
                rate reproduces the price). */}
            {report.reverse_dcf.converged &&
              report.reverse_dcf.implied_growth_y1 != null &&
              report.reverse_dcf.note && (
                <p className="mt-1 text-xs text-ink-dim">
                  {report.reverse_dcf.note}
                </p>
              )}
            {/* The other two drivers, each solved on its own. */}
            {(rdcf?.margin || rdcf?.wacc) && (
              <div className="mt-2 space-y-1 border-t border-brand/20 pt-2 text-xs">
                {rdcf?.margin && (
                  <ReverseRow
                    label="Target EBIT margin"
                    implied={rdcf.margin.implied_target_ebit_margin}
                    current={rdcf.margin.current_target_ebit_margin}
                    converged={rdcf.margin.converged}
                    note={rdcf.margin.note}
                  />
                )}
                {rdcf?.wacc && (
                  <ReverseRow
                    label="Discount rate (WACC)"
                    implied={rdcf.wacc.implied_wacc}
                    current={rdcf.wacc.current_wacc}
                    converged={rdcf.wacc.converged}
                    note={rdcf.wacc.note}
                  />
                )}
              </div>
            )}
            <p className="mt-2 text-[11px] text-ink-faint">
              The near-term growth shift fades to zero by the final forecast year.
              {(rdcf?.margin || rdcf?.wacc) &&
                " Each driver is solved with the others held at the model's values" +
                ((rdcf?.margin?.note && rdcf.margin.converged) ||
                (rdcf?.wacc?.note && rdcf.wacc.converged)
                  ? "; * more than one value fits (hover for the note)."
                  : ".")}
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
                {yearLabels.map((y, i) => (
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
          {/* A flagged company's DCF is reference only, and one with no
              operating income has no price at any default input: say so
              rather than blame the inputs, with the DCF's own notes. */}
          <EmptyState
            title={whom != null ? "No reference DCF" : "No valid DCF at the current inputs"}
            hint={dcfWarnings(report.warnings, whom != null).join(" · ")
              || "Adjust the assumptions and recompute."} />
        </Card>
      )}
    </div>
  );
}
