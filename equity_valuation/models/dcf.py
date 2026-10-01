"""Unlevered discounted-cash-flow (FCFF) valuation.

Projects free cash flow to the firm:

    FCFF_t = NOPAT_t + D&A_t - Capex_t - dNWC_t,   NOPAT_t = EBIT_t * (1 - tax)

Cash flows are projected from the company's latest fundamentals using either
explicit per-year drivers (from ``DCFAssumptions``) or history-derived defaults,
discounted at WACC (optionally on a mid-year convention), and a terminal value
(Gordon growth or an exit EV/EBITDA multiple) is added to obtain enterprise
value, then equity value, then an implied per-share price.

History-derived defaults, each noted in ``assumptions['notes']`` when a rule
changes the plain reading:
  * EBIT margin: the latest year's, or the recent median when the latest year
    is a one-off spike (``utils.robust_latest_margin``). A first profitable
    year after losses is kept but noted (``utils.first_positive_margin``). A
    loss year after at least three profitable ones on a modest revenue move (a
    charge year, ``utils.charge_year_margin``), or a positive margin below a
    third of the prior median after such a run on stable revenue (a collapse
    year, ``utils.collapsed_margin``), is kept as the start but faded to the prior
    median (capped at the previous year's margin) as the target margin, with
    a WARNING note (``margin_fade_target``).
  * Tax: median effective rate over at least 3 clean profitable years, else the
    marginal rate (``wacc.effective_tax_rate_detail``).
  * D&A and capex: revenue-weighted history (``utils.pooled_ratio``); capex well
    above D&A fades with revenue growth (``utils.growth_capex_path``), noted
    when the final year ends at least 0.5pp of revenue below history. With no
    usable capex year at all but D&A above 0, capex is set equal to D&A
    (maintenance only), with a WARNING note, since D&A is still added back;
    not for a bank, insurer, REIT or lender (``utils.MAINTENANCE_CAPEX_KINDS``),
    whose reference-only DCF keeps capex at 0.
  * dNWC: pooled per unit of revenue change, one-off years left out
    (``utils.screened_incremental_ratio``), 0 outside [0, 1].

Pure-Python: stdlib + the package's own helpers only. All money is absolute
units; all rates are decimals; annual series run oldest -> newest.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from ..schemas import (
    CompanyData,
    DCFAssumptions,
    DCFResult,
    MacroAssumptions,
)
from ..utils import (
    EBIT_DERIVED_KINDS,
    MAINTENANCE_CAPEX_KINDS,
    NWC_ONE_OFF_REVENUE_SHARE,
    charge_year_margin,
    collapsed_margin,
    fade_path,
    financial_kind,
    first_positive_margin,
    fiscal_year_labels,
    growth_capex_path,
    is_num,
    material_capex_fade,
    mean,
    net_debt_parts,
    pooled_ratio,
    robust_latest_margin,
    safe_div,
    screened_incremental_ratio,
    series_cagr,
)
from .. import config
from .wacc import compute_wacc, effective_tax_rate_detail

TERMINAL_METHODS = ("gordon", "exit_multiple")


# --------------------------------------------------------------------------- #
#  Small internal helpers
# --------------------------------------------------------------------------- #
def _latest(series, default=None):
    """Last finite element of an oldest->newest series, else ``default``."""
    if not series:
        return default
    val = series[-1]
    return val if is_num(val) else default


def _hist_ratio_mean(numerators, denominators):
    """Mean of numerator_i / denominator_i over years where both are finite and
    the denominator is positive. Returns None if no usable pair exists."""
    ratios = []
    for num, den in zip(numerators or [], denominators or []):
        if is_num(num) and is_num(den) and den > 0:
            r = safe_div(num, den)
            if r is not None:
                ratios.append(r)
    return mean(ratios)


def _all_zero(series) -> bool:
    """True if a series has entries and every finite one is 0 (a zero-filled gap)."""
    vals = [v for v in (series or []) if is_num(v)]
    return bool(vals) and all(v == 0 for v in vals)


def _ebit_history(fin, derive: bool = True) -> tuple[list, Optional[str]]:
    """Historical EBIT with zero-filled (unreported) years rebuilt from pretax.

    Providers write 0.0 when a filer has no operating-income tag (e.g. single-step
    income statements). A 0 there is a gap, not a reading, so where pretax income
    exists the year is approximated as pretax income + interest expense. With
    ``derive=False`` (a flagged company other than a lessor,
    ``utils.EBIT_DERIVED_KINDS``: a bank's or lender's interest is an operating
    cost, so adding it back would count it twice) the gaps are left alone.
    """
    ebit = list(getattr(fin, "ebit", None) or []) if fin is not None else []
    if not derive:
        return ebit, None
    pretax = list(getattr(fin, "pretax_income", None) or []) if fin is not None else []
    interest = list(getattr(fin, "interest_expense", None) or []) if fin is not None else []
    out, rebuilt = [], False
    for i, e in enumerate(ebit):
        p = pretax[i] if i < len(pretax) else None
        if (not is_num(e) or e == 0) and is_num(p) and p != 0:
            it = interest[i] if i < len(interest) and is_num(interest[i]) else 0.0
            out.append(p + abs(it))
            rebuilt = True
        else:
            out.append(e)
    note = ("EBIT not reported for some years; approximated as pretax income + "
            "interest expense") if rebuilt else None
    return out, note


def start_ebit_margin(fin, hist_revenue, base_revenue,
                      derive_ebit: bool = True) -> tuple[Optional[float], list[str]]:
    """(starting EBIT margin, notes): latest EBIT / base revenue, else the trailing
    mean margin, ignoring zero-filled EBIT years. None if nothing is usable.

    A latest margin that is a one-off spike against the recent years (e.g. a
    divestiture gain booked in operating income) is replaced by the recent
    median margin, with a note; see ``utils.robust_latest_margin`` for the test.
    A first positive margin after losses is kept (it may be a real turnaround)
    but noted, since a one-off gain can produce it too
    (``utils.first_positive_margin``). ``derive_ebit`` is passed to
    ``_ebit_history`` (False for a flagged company other than a lessor).
    Shared with the sensitivity grid so its margin axis centres on the same
    start.
    """
    hist_ebit, note = _ebit_history(fin, derive=derive_ebit)
    notes = [note] if note else []
    reported = [e if (is_num(e) and e != 0) else None for e in hist_ebit]
    latest = reported[-1] if reported else None
    margin = safe_div(latest, base_revenue) if (is_num(base_revenue) and base_revenue > 0) else None
    if margin is None:
        # Fall back to the trailing average EBIT margin.
        margin = _hist_ratio_mean(reported, hist_revenue)
    else:
        robust, spike = robust_latest_margin(reported, hist_revenue)
        if spike is not None:
            margin = robust
            notes.append(
                f"latest EBIT margin {spike['latest']:.1%} is out of line with the prior "
                f"median {spike['prior_median']:.1%} (likely a one-off gain or charge); "
                f"starting from the {spike['years']}-year median {spike['median']:.1%} "
                "-- set a target EBIT margin if the latest level should persist")
        else:
            turn = first_positive_margin(reported, hist_revenue)
            if turn is not None:
                notes.append(
                    f"latest EBIT margin {turn['latest']:.1%} is the first positive year after "
                    f"losses (prior median {turn['prior_median']:.1%}); kept as the start "
                    "margin, but check the year for one-off gains such as a disposal "
                    "-- set a target EBIT margin if it should not persist")
    return margin, notes


def margin_fade_target(fin, hist_revenue, derive_ebit: bool = True) -> Optional[dict]:
    """Detail of a latest EBIT margin the DCF starts from but does not hold,
    or None.

    Either rule keeps the latest margin as the start and fades to the
    detail's ``target`` (the prior median, capped at the margin of the year
    before) instead of holding it for the whole projection, unless a target
    EBIT margin is set:
      * a charge year (``utils.charge_year_margin``, ``rule`` "charge"): a
        loss after at least three profitable years on a modest revenue move
        (impairments, restructuring; F FY2025);
      * a collapse year (``utils.collapsed_margin``, ``rule`` "collapse"): a
        margin still positive but below a third of the prior median after at
        least three profitable years, on stable revenue, that the one-off
        spike test does not catch because the move is under five points (JD
        FY2025: 0.28% after 1.75-3.41%).
    Uses the same EBIT history as ``start_ebit_margin`` (shared with the
    sensitivity grid, whose margin axis centres on the target the DCF fades
    to).
    """
    hist_ebit, _ = _ebit_history(fin, derive=derive_ebit)
    reported = [e if (is_num(e) and e != 0) else None for e in hist_ebit]
    if not reported or reported[-1] is None:
        return None  # the start margin is then a trailing mean, not a latest year
    charge = charge_year_margin(reported, hist_revenue)
    if charge is not None:
        return {**charge, "rule": "charge"}
    collapse = collapsed_margin(reported, hist_revenue)
    if collapse is not None:
        return {**collapse, "rule": "collapse"}
    return None


def resolve_terminal_method(assumptions) -> tuple[str, Optional[str]]:
    """(terminal method actually used, note): normalise case/whitespace and fall
    back to Gordon for an unknown method or an exit multiple without a multiple."""
    raw = assumptions.terminal_method if assumptions else None
    method = str(raw or "gordon").strip().lower()
    if method not in TERMINAL_METHODS:
        return "gordon", f"unknown terminal_method {raw!r}; falling back to Gordon"
    if method == "exit_multiple" and not is_num(getattr(assumptions, "exit_ev_ebitda", None)):
        return "gordon", "exit_ev_ebitda missing for exit_multiple method; falling back to Gordon"
    return method, None


# --------------------------------------------------------------------------- #
#  Main entry point
# --------------------------------------------------------------------------- #
def run_dcf(
    company: CompanyData,
    macro: MacroAssumptions,
    assumptions: DCFAssumptions,
    current_price: float,
) -> DCFResult:
    """Run the FCFF DCF and return a fully-populated ``DCFResult``.

    Degrades gracefully on missing data: any unavailable driver falls back to a
    documented default and the choice is recorded in ``DCFResult.assumptions``
    (which carries a human-readable ``notes`` list). Raises ``ValueError`` only
    when there is no positive base revenue to project from (an FCFF DCF is not
    meaningful then; the engine records the failure as a warning).
    """
    notes: list[str] = []

    # ----- 1) discount rate ------------------------------------------------ #
    wacc_result = compute_wacc(company, macro)
    w = wacc_result.wacc
    # Guard a degenerate / non-positive WACC so discounting stays well-defined.
    if not is_num(w) or w <= 0:
        computed = w
        rf = wacc_result.detail.get("risk_free_rate")
        erp = wacc_result.detail.get("equity_risk_premium")
        # Beta-1 cost of equity on the caller's macro inputs, else config defaults.
        w = rf + erp if (is_num(rf) and is_num(erp) and rf + erp > 0) else (
            config.DEFAULT_RISK_FREE_RATE + config.DEFAULT_EQUITY_RISK_PREMIUM)
        notes.append(f"computed WACC {computed:.4f} non-positive/invalid; "
                     f"discounting at fallback rf+ERP {w:.4f}")
        # Report the rate actually used (UI, exports and sensitivity labels read
        # DCFResult.wacc.wacc); keep the rejected value for reference.
        wacc_result = replace(wacc_result, wacc=w,
                              detail={**wacc_result.detail, "wacc": w, "wacc_computed": computed})

    fin = getattr(company, "financials", None)
    bs = getattr(company, "balance_sheet", None)
    market = getattr(company, "market", None)

    n = assumptions.forecast_years if (assumptions and is_num(assumptions.forecast_years)
                                       and assumptions.forecast_years > 0) \
        else config.DEFAULT_FORECAST_YEARS
    n = max(1, int(n))

    terminal_growth = assumptions.terminal_growth if (assumptions and is_num(assumptions.terminal_growth)) \
        else config.DEFAULT_TERMINAL_GROWTH
    if terminal_growth <= -1.0:
        raise ValueError("terminal_growth must be greater than -1")

    terminal_method, method_note = resolve_terminal_method(assumptions)
    if method_note:
        notes.append(method_note)
    if terminal_method == "exit_multiple" and assumptions.exit_ev_ebitda <= 0:
        raise ValueError("exit_ev_ebitda must be positive")
    # Resolve the sustainable growth rate before projecting. Otherwise the
    # operating forecast can fade to a rate the terminal denominator rejects.
    g_used = terminal_growth
    if terminal_method == "gordon" and w - g_used < config.MAX_TERMINAL_GROWTH_VS_WACC:
        g_used = w - config.MAX_TERMINAL_GROWTH_VS_WACC
        notes.append(
            f"terminal growth {terminal_growth:.4f} too close to WACC {w:.4f}; "
            f"clamped to {g_used:.4f} (used in both fade path and terminal value)"
        )

    # ----- 2) revenue growth path ----------------------------------------- #
    hist_revenue = list(getattr(fin, "revenue", None) or []) if fin is not None else []
    base_revenue = _latest(hist_revenue)
    if not is_num(base_revenue) or base_revenue <= 0:
        # No anchor for projections: EV would be 0 and the "price" just
        # -net_debt/shares, which is not a valuation. Fail loudly instead.
        raise ValueError("no positive latest revenue to project from; an FCFF DCF is not meaningful")

    growth_path = None
    if assumptions and assumptions.revenue_growth:
        gp = list(assumptions.revenue_growth)
        if any(not is_num(g) or g <= -1.0 for g in gp):
            raise ValueError("revenue_growth entries must be finite and greater than -1")
        if len(gp) >= n:
            growth_path = gp[:n]
        elif gp:
            # Pad a short explicit path by holding the last given growth flat.
            growth_path = gp + [gp[-1]] * (n - len(gp))
            notes.append("revenue_growth shorter than forecast_years; padded with last value")
    if growth_path is None:
        # Derive base growth from historical revenue CAGR over the available span
        # (zero-filled years are skipped as endpoints but still count as periods).
        base_growth = series_cagr(hist_revenue, getattr(fin, "fiscal_years", None))
        if base_growth is None:
            base_growth = g_used
            notes.append("historical revenue CAGR unavailable; starting growth at terminal_growth")
        # Clamp the near-term growth into a sane band before fading.
        base_growth = max(config.DEFAULT_REVENUE_GROWTH_FLOOR,
                          min(config.DEFAULT_REVENUE_GROWTH_CAP, base_growth))
        growth_path = fade_path(base_growth, g_used, n)

    # Project the revenue series (length n).
    revenue: list[float] = []
    prev = base_revenue
    for g in growth_path:
        cur = prev * (1.0 + g)
        revenue.append(cur)
        prev = cur

    # ----- 3) EBIT margin path -------------------------------------------- #
    # A flagged company's missing EBIT is not rebuilt from pretax income +
    # interest (a lender's interest is an operating cost), except a lessor's,
    # as the data layer does (utils.EBIT_DERIVED_KINDS); the engine leaves a
    # flagged company's DCF out of the blended target either way.
    kind = financial_kind(company)
    derive_ebit = kind in EBIT_DERIVED_KINDS
    start_margin, margin_notes = start_ebit_margin(fin, hist_revenue, base_revenue,
                                                   derive_ebit=derive_ebit)
    notes.extend(margin_notes)
    if start_margin is None:
        start_margin = 0.0
        notes.append("EBIT margin unavailable; defaulting to 0")

    if assumptions and is_num(assumptions.target_ebit_margin):
        target_margin = assumptions.target_ebit_margin
    else:
        target_margin = start_margin
        # A loss year after a profitable run (a charge year), or a margin that
        # collapsed to under a third of the prior median (a collapse year), is
        # not held for the whole projection: start from it, fade to the prior
        # median (capped at the previous year's margin).
        fade = margin_fade_target(fin, hist_revenue, derive_ebit=derive_ebit)
        if fade is not None:
            target_margin = fade["target"]
            # A collapse year is a thin margin: show it to two decimals.
            fmt = ".1%" if fade["rule"] == "charge" else ".2%"
            if fade["target"] < fade["prior_median"]:
                year = "last profitable" if fade["rule"] == "charge" else "previous"
                to = (f"{fade['target']:{fmt}}, the {year} year's margin (below the "
                      "prior median)")
            else:
                to = f"the prior median {fade['prior_median']:{fmt}}"
            if fade["rule"] == "charge":
                notes.append(
                    f"WARNING: latest EBIT margin {fade['latest']:.1%} is a loss after "
                    f"{fade['profitable_years']} profitable years (prior median "
                    f"{fade['prior_median']:.1%}) on a {fade['revenue_move']:+.1%} revenue "
                    "change, likely a charge year (impairments, restructuring); the projection "
                    f"starts from it and fades to {to} by year {n}. Impairments are non-cash, "
                    "so the early years understate cash flow when the loss is mostly "
                    "write-downs -- set a target EBIT margin if the loss should persist")
            else:
                notes.append(
                    f"WARNING: latest EBIT margin {fade['latest']:.2%} is below a third of the "
                    f"prior median {fade['prior_median']:.2%} after {fade['profitable_years']} "
                    f"profitable years, on a {fade['revenue_move']:+.1%} revenue change "
                    "(a trough year: heavy investment, price cuts or charges within operating "
                    f"profit); the projection starts from it and fades to {to} by year {n} "
                    "-- set a target EBIT margin if the trough should persist")
    margin_path = fade_path(start_margin, target_margin, n)
    ebit = [rev * m for rev, m in zip(revenue, margin_path)]

    # ----- 4) tax & NOPAT -------------------------------------------------- #
    if assumptions and is_num(assumptions.tax_rate):
        tax = assumptions.tax_rate
        tax_source = "assumptions.tax_rate"
    elif macro is not None and is_num(macro.tax_rate):
        tax = macro.tax_rate
        tax_source = "macro.tax_rate"
    else:
        # Fewer than 3 clean profitable years (a young issuer's NOL-shielded or
        # allowance-release years) fall back to the marginal rate, with a note.
        tax, tax_source, tax_note = effective_tax_rate_detail(fin, config.DEFAULT_MARGINAL_TAX_RATE)
        if tax_note:
            notes.append(tax_note)
        if tax <= 0:
            # Kept as documented (pass-throughs genuinely pay ~0%), but flag it:
            # providers also zero-fill an unreported tax line.
            notes.append("historical effective tax rate is 0% (tax expense zero or unreported); "
                         "NOPAT is untaxed -- set a tax rate if that is not intended")
    nopat = [e * (1.0 - tax) for e in ebit]

    # ----- 5) D&A, Capex, dNWC -------------------------------------------- #
    hist_da = list(getattr(fin, "dep_amort", None) or []) if fin is not None else []
    hist_capex = list(getattr(fin, "capex", None) or []) if fin is not None else []

    # History-derived D&A and capex are revenue-weighted (sum / sum revenue), so
    # a tiny-revenue ramp year cannot dominate; zero-filled years are gaps, and
    # an all-zero history is the providers' gap filler, not a real 0%.
    da_derived = not (assumptions and is_num(assumptions.da_pct_revenue))
    if not da_derived:
        da_pct = assumptions.da_pct_revenue
    else:
        da_pct = None if _all_zero(hist_da) else pooled_ratio(hist_da, hist_revenue)
        if da_pct is None:
            da_pct = 0.0
            notes.append("D&A %revenue unavailable; defaulting to 0")

    capex_path = None
    if assumptions and is_num(assumptions.capex_pct_revenue):
        capex_pct = assumptions.capex_pct_revenue
    else:
        capex_pct = None if _all_zero(hist_capex) else pooled_ratio(hist_capex, hist_revenue)
        if capex_pct is None and da_pct > 0 and kind in MAINTENANCE_CAPEX_KINDS:
            # D&A is still added back, so a capex of 0 would leave the business
            # with no reinvestment (PSX: DCF 339 vs 233 with capex = D&A; NEE,
            # AER): assume maintenance capex instead. Not for a bank, insurer,
            # REIT or lender, whose D&A is no capex proxy (see
            # utils.MAINTENANCE_CAPEX_KINDS).
            capex_pct = da_pct
            notes.append(
                f"WARNING: no usable capex history (zero or unreported in every year) while D&A "
                f"is {da_pct:.1%} of revenue; capex set equal to D&A (maintenance only) rather "
                "than 0, which would add D&A back with no reinvestment and overstate free cash "
                "flow -- set a capex % of revenue if the filer reports capex under another line")
        elif capex_pct is None:
            capex_pct = 0.0
            notes.append("capex %revenue unavailable; defaulting to 0")
        elif da_derived:
            # Growth-phase capex (well above D&A) is not carried into the
            # terminal year: net capex scales with revenue growth at the
            # historical sales-to-capital ratio (utils.growth_capex_path).
            ref_growth = series_cagr(hist_revenue, getattr(fin, "fiscal_years", None))
            if is_num(ref_growth):
                ref_growth = max(config.DEFAULT_REVENUE_GROWTH_FLOOR,
                                 min(config.DEFAULT_REVENUE_GROWTH_CAP, ref_growth))
            capex_path = growth_capex_path(capex_pct, da_pct, ref_growth, growth_path)
            # A negligible fade (capex just past 1.5x D&A) is applied without a
            # note; capex_pct_path still records it.
            if material_capex_fade(capex_pct, capex_path):
                notes.append(
                    f"capex {capex_pct:.1%} of revenue is growth-phase (D&A {da_pct:.1%}); "
                    f"net capex scaled with revenue growth, reaching {capex_path[-1]:.1%} "
                    f"of revenue in year {n}")

    # Incremental NWC as a % of the revenue *change*.
    if assumptions and is_num(assumptions.nwc_pct_revenue):
        nwc_pct = assumptions.nwc_pct_revenue
    else:
        # Derive from history: pooled sum(dNWC_i) / sum(dRevenue_i), so a single
        # near-flat revenue year cannot dominate. change_in_nwc[i] aligns with
        # revenue[i]. Outside [0, 1] -> treat as no usable signal -> 0.
        # A year whose dNWC is a one-off (> 10% of revenue and larger than the
        # year's revenue change) is left out of the pool, with a note.
        hist_dnwc = list(getattr(fin, "change_in_nwc", None) or []) if fin is not None else []
        nwc_pct, one_off = screened_incremental_ratio(hist_dnwc, hist_revenue)
        if one_off:
            notes.append(
                f"dNWC in {fiscal_year_labels(getattr(fin, 'fiscal_years', None), one_off)} "
                f"exceeds {NWC_ONE_OFF_REVENUE_SHARE:.0%} of revenue and that year's revenue "
                "change; left out of dNWC/dRevenue as a one-off")
        if nwc_pct is None or nwc_pct < 0 or nwc_pct > 1:
            # Unstable/implausible incremental ratio -> assume zero working-capital drag.
            if nwc_pct is not None:
                notes.append(f"derived dNWC/dRevenue {nwc_pct:.3f} implausible; using 0")
            nwc_pct = 0.0

    da = [rev * da_pct for rev in revenue]
    capex_pcts = capex_path if capex_path is not None else [capex_pct] * n
    capex = [rev * c for rev, c in zip(revenue, capex_pcts)]

    # dNWC_t = (revenue_t - revenue_{t-1}) * nwc_pct; t=0 uses base_revenue.
    dnwc: list[float] = []
    prev_rev = base_revenue
    for rev in revenue:
        dnwc.append((rev - prev_rev) * nwc_pct)
        prev_rev = rev

    # FCFF_t = NOPAT_t + D&A_t - Capex_t - dNWC_t
    fcff = [nopat[i] + da[i] - capex[i] - dnwc[i] for i in range(n)]

    # ----- 6) discounting -------------------------------------------------- #
    mid_year = bool(assumptions.mid_year_convention) if assumptions else True
    # Exponent for explicit year t (1-indexed): t-0.5 mid-year, else t.
    exponents = [(t - 0.5) if mid_year else float(t) for t in range(1, n + 1)]
    discount_factors = [1.0 / (1.0 + w) ** e for e in exponents]
    pv_fcff = [fcff[i] * discount_factors[i] for i in range(n)]

    # ----- 7) terminal value ---------------------------------------------- #
    # An unknown method, or exit_multiple without a multiple, falls back to Gordon
    # (with a note) so we still produce a number.
    ebitda_n = (ebit[-1] + da[-1]) if (ebit and da) else 0.0

    terminal_fcff = None
    terminal_capex_pct = capex_pcts[-1]
    if terminal_method == "exit_multiple":
        terminal_value = ebitda_n * assumptions.exit_ev_ebitda
    else:
        # Project the first stable year from its own growth. Scaling FCFF_N
        # carries N's working-capital investment forever when an explicit
        # growth path ends above/below g. Apply the same growth-capex rule at g.
        if capex_path is not None:
            terminal_capex_pct = growth_capex_path(capex_pct, da_pct, ref_growth, [g_used])[0]
        terminal_revenue = revenue[-1] * (1.0 + g_used)
        terminal_fcff = (
            terminal_revenue * (margin_path[-1] * (1.0 - tax) + da_pct - terminal_capex_pct)
            - revenue[-1] * g_used * nwc_pct
        )
        denom = w - g_used
        if denom <= 0:
            # Should not happen after clamping, but guard divide-by-zero anyway.
            terminal_value = 0.0
            notes.append("WACC-g non-positive after clamp; terminal value set to 0")
        else:
            terminal_value = terminal_fcff / denom
            if terminal_fcff < 0:
                notes.append("first stable-year FCFF is negative, so the Gordon terminal value "
                             "capitalises a perpetual cash outflow; review the EBIT margin "
                             "and reinvestment drivers")

    # Discount the TV. A Gordon TV is a perpetuity valued as of year N and shares
    # the final explicit flow's timing (N-0.5 under mid-year, else N). An
    # exit-multiple TV is a point-in-time, year-end sale value (EBITDA_N * mult),
    # so it is always discounted at the FULL period N regardless of mid-year.
    if terminal_method == "exit_multiple":
        tv_exponent = float(n)
        tv_exponent_rationale = (
            "exit-multiple TV is a year-end sale value; discounted at full period N"
        )
    else:
        tv_exponent = (n - 0.5) if mid_year else float(n)
        tv_exponent_rationale = (
            "Gordon TV shares the final explicit flow's timing "
            "(N-0.5 under mid-year, else N)"
        )
    tv_discount_factor = 1.0 / (1.0 + w) ** tv_exponent
    pv_terminal = terminal_value * tv_discount_factor

    # ----- 8) bridge to equity & implied price ---------------------------- #
    enterprise_value = sum(pv_fcff) + pv_terminal

    # net_debt = total_debt - cash (each missing component assumed 0 on its own);
    # add minority interest & preferred to bridge from enterprise to common equity.
    net_debt, bridge_notes = net_debt_parts(bs)
    notes.extend(bridge_notes)
    minority = getattr(bs, "minority_interest", 0.0) if bs is not None else 0.0
    preferred = getattr(bs, "preferred_equity", 0.0) if bs is not None else 0.0
    minority = minority if is_num(minority) else 0.0
    preferred = preferred if is_num(preferred) else 0.0

    total_claims = net_debt + minority + preferred
    equity_value = enterprise_value - total_claims

    # Shares: prefer live market shares outstanding, else latest diluted shares.
    shares = getattr(market, "shares_outstanding", None) if market is not None else None
    if not is_num(shares) or shares <= 0:
        diluted = getattr(fin, "diluted_shares", None) or [] if fin is not None else []
        shares = next((s for s in reversed(diluted) if is_num(s) and s > 0), None)
        if is_num(shares) and shares > 0:
            notes.append("market shares_outstanding unavailable; using latest diluted_shares")
    implied_price = safe_div(equity_value, shares)
    valuation_available = is_num(implied_price)
    if not valuation_available:
        implied_price = 0.0
        notes.append("share count unavailable; implied price set to 0")
        shares = shares if is_num(shares) else 0.0

    cur_price = current_price if is_num(current_price) else (
        getattr(market, "price", None) if market is not None else None)
    upside = safe_div(implied_price, cur_price)
    upside = (upside - 1.0) if upside is not None else 0.0

    assumptions_dict = {
        "forecast_years": n,
        "revenue_growth_path": list(growth_path),
        "base_revenue": base_revenue,
        "ebit_margin_path": list(margin_path),
        "start_ebit_margin": start_margin,
        "target_ebit_margin": target_margin,
        "tax_rate": tax,
        "tax_source": tax_source,
        "da_pct_revenue": da_pct,
        "capex_pct_revenue": capex_pct,
        # Per-year capex % actually applied (differs from capex_pct_revenue only
        # when growth-phase capex is faded).
        "capex_pct_path": list(capex_pcts),
        "nwc_pct_revenue": nwc_pct,
        "terminal_method": terminal_method,
        "terminal_growth": terminal_growth,
        "terminal_growth_used": g_used,
        "terminal_fcff": terminal_fcff,
        "terminal_capex_pct_revenue": terminal_capex_pct,
        "exit_ev_ebitda": (assumptions.exit_ev_ebitda if assumptions else None),
        "ebitda_terminal": ebitda_n,
        "mid_year_convention": mid_year,
        # Document the discounting choice explicitly for downstream exporters.
        "discount_exponents": list(exponents),
        "terminal_discount_exponent": tv_exponent,
        "terminal_discount_exponent_rationale": tv_exponent_rationale,
        "wacc": w,
        "valuation_available": valuation_available,
        "notes": notes,
    }

    return DCFResult(
        wacc=wacc_result,
        years=list(range(1, n + 1)),
        revenue=revenue,
        ebit=ebit,
        nopat=nopat,
        fcff=fcff,
        discount_factors=discount_factors,
        pv_fcff=pv_fcff,
        terminal_value=terminal_value,
        pv_terminal=pv_terminal,
        enterprise_value=enterprise_value,
        net_debt=net_debt,
        equity_value=equity_value,
        shares=shares if is_num(shares) else 0.0,
        implied_price=implied_price,
        current_price=cur_price if is_num(cur_price) else 0.0,
        upside=upside,
        assumptions=assumptions_dict,
    )
