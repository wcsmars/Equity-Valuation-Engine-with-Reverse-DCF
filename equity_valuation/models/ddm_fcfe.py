"""Dividend-Discount (DDM) and Free-Cash-Flow-to-Equity (FCFE) valuation models.

These are *levered* / equity-side counterparts to the unlevered FCFF DCF. They
discount cash flows that accrue directly to equity holders at the cost of equity
(CAPM), so there is no WACC and no enterprise-value bridge -- the output is an
implied price per share directly.

Design notes:
  * This module is deliberately INDEPENDENT of ``models/dcf.py``. The few small
    helpers it needs (revenue-growth path, ratio-of-revenue projections) are
    re-derived locally on top of ``utils`` (and the shared tax-rate rule in
    ``models/wacc.py``) so the two model families can evolve without coupling.
  * Every access to a financial field is guarded -- the providers frequently
    leave series sparse, zero-filled, or None, and the models must degrade
    gracefully and record a human-readable note rather than crash.
  * Money is in absolute units; rates are decimals; annual series run
    OLDEST -> NEWEST (``[-1]`` is the most recent fiscal year).
"""

from __future__ import annotations

from typing import Optional

from .. import config
from ..schemas import (
    AnnualFinancials,
    CompanyData,
    DDMAssumptions,
    DDMResult,
    FCFEResult,
    MacroAssumptions,
)
from ..utils import (
    MAINTENANCE_CAPEX_KINDS,
    MARGIN_SPIKE_WINDOW,
    NWC_ONE_OFF_REVENUE_SHARE,
    charge_year_margin,
    collapsed_margin,
    fade_path,
    financial_kind,
    fiscal_year_labels,
    growth_capex_path,
    is_num,
    material_capex_fade,
    median,
    pooled_ratio,
    robust_latest_margin,
    safe_div,
    screened_incremental_ratio,
    series_cagr,
)
from .wacc import effective_tax_rate_detail

# Minimum spread required between the cost of equity and a perpetual growth rate
# for a Gordon-style terminal/perpetuity to be finite and well-behaved. The
# interface mandates ke - g >= 0.005; if an input growth rate violates it we clamp
# the growth rate down so the spread is restored (recording a note).
_MIN_KE_G_SPREAD = 0.005

# Above this ROE the book equity is too thin (buybacks, write-downs) for
# ROE x retention to say anything about reinvestment-driven growth.
_MAX_MEANINGFUL_ROE = 1.0

# The latest net margin has collapsed when it is below this share of the net
# margin its EBIT implies after interest and tax, while the EBIT margin itself
# is in line (BP FY2025: 0.03% against 3.7%, after impairments below operating
# profit and an 83% tax charge). See ``_latest_net_margin``.
NET_MARGIN_COLLAPSE_SHARE = 0.25

# Total dividends paid that fall by more than half, or more than double, from
# one year to the next mark a break in the paying history: a cut or suspension
# (GM 2020-21), a restart year or a step change (F FY2023, a special on top of a
# restarted dividend). A CAGR across a break measures the break, not dividend
# growth. A single year more than double both neighbours, which are in line
# with each other (COST's FY2024 special), is skipped instead. See
# ``_dividend_cagr``.
DIVIDEND_BREAK_FACTOR = 2.0
# After a break, a dividend CAGR needs at least this many paying years (three
# periods), so a restart ramp alone does not set it.
MIN_DIVIDEND_RUN_YEARS = 4


# --------------------------------------------------------------------------- #
#  Cost of equity (CAPM)
# --------------------------------------------------------------------------- #
def cost_of_equity(company: CompanyData, macro: MacroAssumptions) -> float:
    """CAPM cost of equity: ``ke = rf + beta * ERP``.

    Beta is taken from live market data when available and finite, otherwise it
    falls back to ``config.DEFAULT_BETA``. Risk-free rate and equity-risk-premium
    come from the macro assumptions (both decimals).
    """
    rf = macro.risk_free_rate if (macro is not None and is_num(macro.risk_free_rate)) else config.DEFAULT_RISK_FREE_RATE
    erp = (
        macro.equity_risk_premium
        if macro is not None and is_num(macro.equity_risk_premium)
        else config.DEFAULT_EQUITY_RISK_PREMIUM
    )

    beta = None
    market = getattr(company, "market", None)
    if market is not None:
        beta = getattr(market, "beta", None)
    if not is_num(beta):
        beta = config.DEFAULT_BETA

    return rf + beta * erp


# --------------------------------------------------------------------------- #
#  Small shared helpers (kept local -- no dependency on models/dcf.py)
# --------------------------------------------------------------------------- #
def _shares(company: CompanyData) -> Optional[float]:
    """Best-effort share count: market shares outstanding, else latest diluted."""
    market = getattr(company, "market", None)
    if market is not None:
        so = getattr(market, "shares_outstanding", None)
        if is_num(so) and so > 0:
            return float(so)
    fin = getattr(company, "financials", None)
    if fin is not None:
        diluted = getattr(fin, "diluted_shares", None) or []
        for v in reversed(diluted):  # newest first; take the latest sane value
            if is_num(v) and v > 0:
                return float(v)
    return None


def _hist_revenue_cagr(fin: AnnualFinancials) -> Optional[float]:
    """Historical revenue CAGR over the available (positive) annual series."""
    return series_cagr(getattr(fin, "revenue", None), getattr(fin, "fiscal_years", None))


def _revenue_growth_path(fin: AnnualFinancials, terminal_growth: float, n: int) -> list[float]:
    """Per-year revenue growth, fading the historical CAGR toward terminal growth.

    Mirrors the DCF's growth derivation (but re-implemented locally): start from
    the historical revenue CAGR, clamp to the configured near-term band, then
    linearly fade to ``terminal_growth`` over ``n`` forecast years. If no usable
    history exists, the whole path is the terminal growth (a conservative flat
    assumption).
    """
    if n <= 0:
        return []
    base = _hist_revenue_cagr(fin)
    if not is_num(base):
        base = terminal_growth
    # Clamp the near-term growth into the configured sane band.
    base = max(config.DEFAULT_REVENUE_GROWTH_FLOOR, min(config.DEFAULT_REVENUE_GROWTH_CAP, base))
    return fade_path(base, terminal_growth, n)


def _ratio_of_revenue(series: Optional[list], revenue: Optional[list]) -> Optional[float]:
    """Revenue-weighted ratio of a flow series to revenue (sum / sum revenue).

    Used to turn D&A / capex into a forward % of revenue from history, as the
    DCF does: a tiny-revenue ramp year cannot dominate the way it would in a
    mean of per-year ratios. Zero-filled years are gaps; None if nothing is
    usable (including an all-zero series, the providers' gap filler).
    """
    return pooled_ratio(series, revenue)


def _latest_net_margin(
    fin: AnnualFinancials,
    notes: Optional[list[str]] = None,
    tax_rate: Optional[float] = None,
) -> Optional[float]:
    """Latest net income / revenue, guarding None/zero revenue.

    Adjustments, each with a note, keep one unusual year from being held for
    the whole projection. Both taxes use the rate the DCF applies to NOPAT: an
    explicit ``tax_rate`` (``macro.tax_rate``, the CLI's ``--tax``) when set,
    else ``wacc.effective_tax_rate_detail``'s rate.
      * When the tax history is too thin or distorted for an effective rate
        (``effective_tax_rate_detail`` falls back to the marginal rate:
        NOL-shielded years, an allowance release) and the latest year is
        profitable, net income is rebuilt as pretax income x (1 - that rate).
        With clean tax history reported net income is kept, even when a
        ``tax_rate`` is set.
      * Otherwise a latest net margin that is a one-off spike against the
        recent years is replaced by the recent median net margin, by the DCF's
        start-margin test (``utils.robust_latest_margin``), and returned.
      * Whenever that median was not used (including after the tax rebuild,
        since pretax income carries the same one-off) and the latest *EBIT*
        margin is such a spike (the DCF then starts from the median EBIT
        margin), the after-tax excess of the latest EBIT margin over that
        median is taken off. (SOLV FY2025: a divestiture gain in operating
        income, while the net margin looks in line with carve-out years.)
      * Otherwise, when neither applied, the EBIT margin is positive and in
        line, but the reported net margin is below
        ``NET_MARGIN_COLLAPSE_SHARE`` of the margin that EBIT
        implies, (EBIT - interest) x (1 - tax) / revenue, the net margin has
        collapsed below operating profit (impairments of investments, an
        abnormal tax charge; ``net_margin_collapse``). It is replaced by the
        median net margin of the prior years in the window when that median
        lies between the collapse line and the implied margin, else by the
        implied margin: a prior median above it is capped there (GIS FY2026:
        a median near 125% from a mis-tagged revenue line, so the implied 1.4%
        is used), and one below the line is no better (BP FY2025: 0.03%, with
        0.2% the prior median, -> 3.7%). The WARNING-prefixed note names the
        margin actually used.
    A loss year that the EBIT margin shares (a charge year in operating
    profit) is kept here; ``run_fcfe`` fades it (``_charge_year_net_target``).
    When the provider marks a common-earnings attribution adjustment, only
    like-for-like historical net margins are compared: consolidated EBIT and
    pretax income cannot reconstruct cash flow belonging to common holders.
    """
    ni = getattr(fin, "net_income", None) or []
    rev = getattr(fin, "revenue", None) or []
    if not ni or not rev:
        return None
    latest = safe_div(ni[-1], rev[-1])
    if not is_num(latest) or ni[-1] == 0 or rev[-1] <= 0:
        return latest
    pretax = (getattr(fin, "pretax_income", None) or [None])[-1]
    tax, tax_source, _ = effective_tax_rate_detail(fin, config.DEFAULT_MARGINAL_TAX_RATE)
    thin_history = tax_source != "effective (historical)"
    if is_num(tax_rate):
        tax, rate_label = float(tax_rate), "the set tax rate"
    else:
        rate_label = "the marginal rate"
    notes = notes if notes is not None else []
    margin = latest
    attributed = bool(getattr(fin, "_income_attribution_adjusted", False))
    if thin_history and not attributed and is_num(pretax) and pretax > 0:
        margin = pretax * (1.0 - tax) / rev[-1]
        if abs(margin - latest) > 1e-9:
            notes.append(
                f"Tax history too thin for an effective rate; net margin rebuilt from pretax "
                f"income at {rate_label} {tax:.1%} ({margin:.1%}, reported {latest:.1%}).")
    else:
        robust, spike = robust_latest_margin(ni, rev)
        if spike is not None:
            notes.append(
                f"Latest net margin {spike['latest']:.1%} is out of line with the prior "
                f"median {spike['prior_median']:.1%} (likely a one-off gain or charge); "
                f"projecting from the {spike['years']}-year median {spike['median']:.1%}.")
            return robust
    if attributed:
        # The provider has established (or warned about) a distinct common-
        # earnings basis. Consolidated EBIT/pretax would put preferred holders'
        # or noncontrolling shareholders' income back into common FCFE.
        notes.append("Common earnings attribution retained; consolidated pretax income "
                     "and EBIT are not used to rebuild the net margin.")
        return margin
    ebit = getattr(fin, "ebit", None) or []
    op_spike = None
    if ebit and is_num(ebit[-1]) and ebit[-1] != 0:
        _, op_spike = robust_latest_margin(ebit, rev)
        if op_spike is not None:
            excess = (op_spike["latest"] - op_spike["median"]) * (1.0 - tax)
            notes.append(
                f"Latest EBIT margin {op_spike['latest']:.1%} is out of line with the prior "
                f"median {op_spike['prior_median']:.1%} (likely a one-off gain or charge); "
                f"net margin adjusted by its after-tax excess over the {op_spike['years']}-year "
                f"median (taxed at {tax:.1%}; {margin:.1%} -> {margin - excess:.1%}).")
            margin -= excess
    # Only the reported margin is tested (a rebuilt one already has a normal
    # tax charge, and the rules do not stack).
    collapse = (net_margin_collapse(fin, tax)
                if margin == latest and op_spike is None else None)
    if collapse is not None:
        implied, prior_med = collapse["implied"], collapse["prior_median"]
        source = "the margin EBIT implies after interest and tax"
        if is_num(prior_med) and NET_MARGIN_COLLAPSE_SHARE * implied <= prior_med <= implied:
            new, source = prior_med, "the prior median net margin"
        elif is_num(prior_med) and prior_med > implied:
            # Capped: a prior median above what EBIT now supports (or an
            # implausible one, e.g. from a mis-tagged revenue line) is not used.
            new = implied
            source += f" (below the prior median {prior_med:.1%})"
        else:
            new = implied
            if is_num(prior_med):
                source += f" (prior median {prior_med:.1%} is no better)"
        # WARNING-prefixed like the charge-year notes: it overrides the
        # reported figure and can move the verdict (BP.L, GIS).
        notes.append(
            f"WARNING: latest net margin {margin:.1%} is far below the {implied:.1%} its EBIT "
            f"margin {collapse['ebit_margin']:.1%} implies after interest and tax at {tax:.1%} "
            "(items below operating profit such as impairments, or an abnormal tax "
            f"charge); projecting from {source}, {new:.1%}.")
        margin = new
    return margin


def _prior_median_margin(numerators, denominators) -> Optional[float]:
    """Median margin of the years before the latest, over the same window and
    usable years as ``utils.robust_latest_margin`` (None if there are none)."""
    pts = [n / d for n, d in zip(numerators or [], denominators or [])
           if is_num(n) and is_num(d) and d > 0 and n != 0][-MARGIN_SPIKE_WINDOW:]
    return median(pts[:-1]) if len(pts) >= 2 else None


def net_margin_collapse(
    fin: Optional[AnnualFinancials], tax_rate: Optional[float] = None
) -> Optional[dict]:
    """Detail dict when the latest reported net margin has collapsed below
    operating profit, else None.

    The screen ``_latest_net_margin`` applies (the comps also use it before a
    P/E is applied to the latest earnings): the latest EBIT margin is positive
    and not a one-off spike (``utils.robust_latest_margin``), but the latest
    net margin is below ``NET_MARGIN_COLLAPSE_SHARE`` of the margin that EBIT
    implies, (EBIT - interest) x (1 - tax) / revenue: impairments of
    investments or an abnormal tax charge below operating profit (BP FY2025:
    0.03% against 3.7%). ``tax_rate`` is the rate to apply (the CLI's
    ``--tax``), else ``wacc.effective_tax_rate_detail``'s rate. The dict has
    ``latest``, ``implied``, ``ebit_margin``, ``tax`` and ``prior_median``
    (the median net margin of the prior years in the window, or None). A year
    whose EBIT margin is itself out of line is screened for the comps by
    ``ebit_charge_collapse``.
    A provider-marked common-income attribution difference disables this
    consolidated-profit comparison.
    """
    if getattr(fin, "_income_attribution_adjusted", False):
        return None
    ni = getattr(fin, "net_income", None) or []
    rev = getattr(fin, "revenue", None) or []
    ebit = getattr(fin, "ebit", None) or []
    if not ni or not rev or not ebit:
        return None
    latest = safe_div(ni[-1], rev[-1])
    if not is_num(latest) or ni[-1] == 0 or rev[-1] <= 0:
        return None
    if not (is_num(ebit[-1]) and ebit[-1] > 0) or robust_latest_margin(ebit, rev)[1] is not None:
        return None
    if is_num(tax_rate):
        tax = float(tax_rate)
    else:
        tax, _, _ = effective_tax_rate_detail(fin, config.DEFAULT_MARGINAL_TAX_RATE)
    interest = (getattr(fin, "interest_expense", None) or [None])[-1]
    interest = abs(interest) if is_num(interest) else 0.0
    implied = (ebit[-1] - interest) * (1.0 - tax) / rev[-1]
    if implied <= 0 or latest >= NET_MARGIN_COLLAPSE_SHARE * implied:
        return None
    return {"latest": latest, "implied": implied, "ebit_margin": ebit[-1] / rev[-1],
            "tax": tax, "prior_median": _prior_median_margin(ni, rev)}


def ebit_charge_collapse(fin: Optional[AnnualFinancials]) -> Optional[dict]:
    """Detail dict when the latest net margin has collapsed in a year whose
    EBIT margin is itself out of line, else None.

    ``net_margin_collapse`` skips a year whose EBIT margin is a one-off
    (``utils.robust_latest_margin``), since the DCF and FCFE already start from
    the median then. The comps apply a peer P/E to the reported EPS, so they
    also need this case: the latest net margin, still positive, is below
    ``NET_MARGIN_COLLAPSE_SHARE`` of the median net margin of the prior years
    in the window, and the latest EBIT margin
      * is a one-off drop below its prior median (``robust_latest_margin``),
        or has collapsed to under a third of it (``utils.collapsed_margin``, a
        drop too small in points for the spike screen, which the DCF also
        fades from): ``ebit_move`` "drop" or "collapse", a charge inside
        operating profit (GM FY2025: EBIT margin 1.6% against 6.7%, net margin
        1.46% against 6.1%; at an EBIT margin of 1.8% only the collapse rule
        would catch it); or
      * is a one-off rise above it: ``ebit_move`` "gain", so a charge below
        operating profit took the net margin down next to a one-off gain
        within it.
    The dict has ``latest`` and ``prior_median`` (net margins), ``ebit_margin``,
    ``ebit_prior_median`` and ``ebit_move``.
    """
    ni = getattr(fin, "net_income", None) or []
    rev = getattr(fin, "revenue", None) or []
    ebit = getattr(fin, "ebit", None) or []
    if not ni or not rev or not ebit:
        return None
    latest = safe_div(ni[-1], rev[-1])
    if not is_num(latest) or latest <= 0 or rev[-1] <= 0:
        return None
    if not (is_num(ebit[-1]) and ebit[-1] != 0):
        return None
    spike = robust_latest_margin(ebit, rev)[1]
    if spike is not None:
        move = "drop" if spike["latest"] < spike["prior_median"] else "gain"
    else:
        spike = collapsed_margin(ebit, rev)
        move = "collapse"
    if spike is None:
        return None  # EBIT in line: net_margin_collapse's case
    prior_med = _prior_median_margin(ni, rev)
    if not is_num(prior_med) or prior_med <= 0 or latest >= NET_MARGIN_COLLAPSE_SHARE * prior_med:
        return None
    return {"latest": latest, "prior_median": prior_med, "ebit_margin": spike["latest"],
            "ebit_prior_median": spike["prior_median"], "ebit_move": move}


def _charge_year_net_target(
    fin: AnnualFinancials, start: Optional[float], notes: list[str]
) -> Optional[float]:
    """Net margin to fade to after a charge year, else None.

    Mirrors the DCF (``utils.charge_year_margin``): when the start margin is
    the reported latest net margin, it is a loss, and either the net margin or
    the EBIT margin is a loss after at least three profitable years on a
    modest revenue move, the projection starts from the loss and fades to the
    median net margin of the prior years (positive), capped at the last
    profitable year's net margin as the DCF's target is, with a WARNING note
    that names the series that showed the charge year (F FY2025: EBIT -4.9%
    after four years of operating profit; its net margin, with a loss in
    FY2022, does not qualify on its own).
    """
    ni = getattr(fin, "net_income", None) or []
    rev = getattr(fin, "revenue", None) or []
    if not ni or not rev or not is_num(start):
        return None
    latest = safe_div(ni[-1], rev[-1])
    if not is_num(latest) or latest >= 0 or abs(start - latest) > 1e-12:
        return None  # profitable, or already replaced by an adjusted margin
    ebit = getattr(fin, "ebit", None) or []
    charge = charge_year_margin(ni, rev)
    if charge is not None:
        why = f"is a loss after {charge['profitable_years']} profitable years"
    elif ebit and is_num(ebit[-1]) and ebit[-1] != 0:
        charge = charge_year_margin(ebit, rev)
        if charge is not None:
            why = (f"is a loss, as is the EBIT margin {charge['latest']:.1%} after "
                   f"{charge['profitable_years']} years of operating profit")
    if charge is None:
        return None
    prior_med = _prior_median_margin(ni, rev)
    if not is_num(prior_med) or prior_med <= 0:
        return None
    pts = [n / d for n, d in zip(ni, rev)
           if is_num(n) and is_num(d) and d > 0 and n != 0][-MARGIN_SPIKE_WINDOW:-1]
    last_profitable = next((p for p in reversed(pts) if p > 0), prior_med)
    target = min(prior_med, last_profitable)
    if target < prior_med:
        to = (f"{target:.1%}, the last profitable year's net margin (below the prior median "
              f"{prior_med:.1%})")
    else:
        to = f"the prior median net margin {target:.1%}"
    notes.append(
        f"WARNING: latest net margin {latest:.1%} {why}, likely a charge year "
        f"(impairments, restructuring); the projection starts from it and fades to {to}.")
    return target


# --------------------------------------------------------------------------- #
#  Dividend Discount Model
# --------------------------------------------------------------------------- #
def _sustainable_growth(
    fin: AnnualFinancials, company: CompanyData, notes: Optional[list[str]] = None
) -> Optional[float]:
    """Sustainable growth = ROE * retention ratio.

    ROE = latest net income / book equity. Retention = 1 - payout, where payout =
    dividends paid / net income (clamped to [0, 1]); if the dividends-paid line is
    missing/zero-filled, dividends are estimated as DPS x shares. Returns None if
    inputs are unusable: non-positive net income, non-positive book equity (common
    for buyback-heavy dividend payers) or an ROE above ``_MAX_MEANINGFUL_ROE``.
    The caller then falls back to the dividend CAGR or terminal growth.
    """
    notes = notes if notes is not None else []
    ni = (getattr(fin, "net_income", None) or [None])[-1]
    equity = getattr(getattr(company, "balance_sheet", None), "total_equity", None)
    if not is_num(ni) or ni <= 0:
        return None
    if not is_num(equity) or equity <= 0:
        notes.append("Book equity unavailable or non-positive; ROE x retention skipped "
                     "for high growth.")
        return None
    roe = ni / equity
    if roe > _MAX_MEANINGFUL_ROE:
        notes.append(f"ROE {roe:.0%} not meaningful (thin book equity); ROE x retention "
                     "skipped for high growth.")
        return None

    div_paid = (getattr(fin, "dividends_paid", None) or [None])[-1]
    if not is_num(div_paid) or div_paid <= 0:
        # The DDM only runs for payers (DPS > 0), so a 0 here is a data gap, not
        # 100% retention.
        dps = getattr(getattr(company, "market", None), "dividend_per_share", None)
        shares = _shares(company)
        if is_num(dps) and dps > 0 and shares:
            div_paid = dps * shares
            notes.append("Dividends paid unreported; payout estimated from DPS x shares.")
    payout = safe_div(div_paid, ni)
    if not is_num(payout):
        payout = 0.0
    payout = max(0.0, min(1.0, payout))
    retention = 1.0 - payout
    return roe * retention


def _dividend_cagr(fin: AnnualFinancials, notes: Optional[list[str]] = None) -> Optional[float]:
    """Per-share dividend CAGR over the continuous paying run ending in the
    latest year, else None. Annual DPS is approximated as common dividends paid
    divided by diluted weighted-average shares; when aligned share history is
    unavailable, total dividends are a disclosed growth proxy.

    The run goes back from the latest year and stops at a break: a zero or
    missing year between paying years (a suspension, or a year the provider
    zero-filled), or a year whose dividends fell by more than half or more
    than doubled (``DIVIDEND_BREAK_FACTOR``: a cut, a restart or a step
    change). The break year itself is left out, since it is usually a
    partial, trough or one-off year. GM paid 2.24B in FY2018, cut in 2020,
    suspended in 2021 and restarted in 2022: a CAGR across that (-16.1%/yr)
    measures the suspension, and one from the restart year measures the ramp
    back.

    A single-year spike is skipped, not read as a break: a year more than
    double both neighbours (a special dividend), where the two neighbours are
    in line with each other. The run then carries on across it, and a note
    names the year (COST: 1.25B, 9.04B with the FY2024 special, then 2.18B;
    read as a break, the normal FY2025 would end the run and drop the whole
    regular history). A special in the latest year, or specials over several
    years in a row (AGCO FY2021-24), still break the run, since one neighbour
    cannot show it is a one-off. A one-year dip is not skipped: the trough of
    a suspension sits between a cut year and a restart year that can look in
    line (GM: 669M, 186M, 397M), and skipping it would join the two.

    With no break (leading zero years before a first dividend are not one)
    the CAGR spans the paying history as before. After a break it needs
    ``MIN_DIVIDEND_RUN_YEARS`` paying years since, else it is None and the
    DDM relies on sustainable growth (ROE x retention) or terminal growth; a
    note says which. Periods are fiscal-year differences.
    """
    divs = list(getattr(fin, "dividends_paid", None) or [])
    shares = list(getattr(fin, "diluted_shares", None) or [])
    per_share = len(shares) == len(divs) and all(is_num(s) and s > 0 for s in shares)
    if per_share:
        divs = [d / s if is_num(d) else d for d, s in zip(divs, shares)]
    elif notes is not None and any(is_num(d) and d > 0 for d in divs):
        notes.append("Aligned diluted-share history unavailable; dividend growth uses "
                     "total dividends paid as a proxy for per-share growth.")
    label = "Dividends paid per share" if per_share else "Dividends paid"
    years = list(getattr(fin, "fiscal_years", None) or [])
    if len(years) != len(divs):
        years = []

    def paid(v) -> bool:
        return is_num(v) and v > 0

    def apart(a, b) -> bool:
        """More than the break factor apart (both paid)."""
        return a * DIVIDEND_BREAK_FACTOR < b or a > DIVIDEND_BREAK_FACTOR * b

    if not divs or not paid(divs[-1]):
        return None
    last = len(divs) - 1
    kept, skipped, why = [last], [], None  # positions, newest first
    i = last  # the earliest year kept so far
    while i > 0:
        j = i - 1
        prev, cur = divs[j], divs[i]
        if not paid(prev):
            if any(paid(v) for v in divs[:j]):
                why = f"were zero or unreported in {fiscal_year_labels(years, [j])}"
                kept.pop()  # the first year back may be partial
            break
        if apart(cur, prev):
            before = divs[j - 1] if j > 0 else None
            if (paid(before) and prev > DIVIDEND_BREAK_FACTOR * max(cur, before)
                    and not apart(cur, before)):
                # A one-year spike (a special) between two years in line.
                skipped.append(j)
                kept.append(j - 1)
                i = j - 1
                continue
            move = "fell by more than half" if cur < prev else "more than doubled"
            why = f"{move} in {fiscal_year_labels(years, [i])}"
            kept.pop()
            break
        kept.append(j)
        i = j
    pos = sorted(kept)
    run = [divs[k] for k in pos]
    run_years = [years[k] for k in pos] if years else None
    if skipped and notes is not None and (why is None or len(run) >= MIN_DIVIDEND_RUN_YEARS):
        notes.append(
            f"{label} in {fiscal_year_labels(years, sorted(skipped))} were more than "
            "double both neighbouring years (a special dividend); left out of the dividend "
            "CAGR as one-offs, not read as breaks in the paying run.")
    if why is None:
        return series_cagr(run, run_years)
    enough = len(run) >= MIN_DIVIDEND_RUN_YEARS
    if notes is not None:
        text = (f"{label} {why} (a cut, suspension, restart, special dividend or step "
                "change), so a dividend CAGR across it would measure that break; ")
        if enough:
            span = fiscal_year_labels(years, [pos[0], last]).replace(", ", "-")
            text += f"the CAGR covers the {len(run)} paying years since ({span}) only."
        else:
            text += (f"fewer than {MIN_DIVIDEND_RUN_YEARS} paying years since, so no dividend "
                     "CAGR is used (sustainable growth, else terminal growth).")
        notes.append(text)
    return series_cagr(run, run_years) if enough else None


def run_ddm(
    company: CompanyData,
    macro: MacroAssumptions,
    assumptions: DDMAssumptions,
    current_price: float,
) -> Optional[DDMResult]:
    """Dividend Discount Model -> implied price per share.

    Returns ``None`` (no crash) for non-dividend-payers, i.e. when
    ``market.dividend_per_share`` is None or 0.

    Supported methods (``assumptions.method``):
      * ``gordon``    -- single-stage Gordon growth at the terminal rate.
      * ``two_stage`` -- ``high_growth_years`` of high growth, then a Gordon
                         perpetuity at the terminal rate.
      * ``h_model``   -- linearly-declining growth from an initial high rate to
                         the terminal rate (closed-form H-model).

    In every case the cost of equity is CAPM. Where a perpetuity is taken we
    require ``ke - g >= 0.005`` and clamp ``g`` down if necessary, recording the
    adjustment in ``detail['notes']``.
    """
    notes: list[str] = []
    market = getattr(company, "market", None)
    d0 = getattr(market, "dividend_per_share", None) if market is not None else None

    # Non-dividend payer -> DDM is inapplicable. Return None per the contract.
    if not is_num(d0) or d0 <= 0:
        return None
    d0 = float(d0)

    ke = cost_of_equity(company, macro)
    if not is_num(ke) or ke <= 0:
        raise ValueError("DDM requires a positive finite cost of equity")
    g_terminal = assumptions.terminal_growth if is_num(assumptions.terminal_growth) else 0.0
    if g_terminal <= -1.0:
        raise ValueError("terminal_growth must be greater than -1")
    method = str(assumptions.method or "two_stage").strip().lower()
    fin = getattr(company, "financials", None)

    detail: dict = {
        "method": method,
        "cost_of_equity": ke,
        "D0": d0,
        "terminal_growth": g_terminal,
        "notes": notes,
    }

    # --- helper: clamp a perpetual growth rate so ke - g >= the minimum spread -- #
    def _clamp_terminal_g(g: float, label: str) -> float:
        if ke - g < _MIN_KE_G_SPREAD:
            new_g = ke - _MIN_KE_G_SPREAD
            notes.append(
                f"{label} growth {g:.4f} too close to ke {ke:.4f}; clamped to {new_g:.4f}."
            )
            return new_g
        return g

    # ------------------------------------------------------------------ gordon
    if method == "gordon":
        g = _clamp_terminal_g(g_terminal, "Gordon")
        denom = ke - g
        price = safe_div(d0 * (1.0 + g), denom)
        detail["growth"] = g
        detail["implied_price"] = price
        if not is_num(price) or price < 0:
            notes.append("Gordon DDM produced a non-positive/undefined price.")
            price = 0.0
        return DDMResult(method="gordon", implied_price=float(price), cost_of_equity=ke, detail=detail)

    # --------------------------------------------------------------- h_model
    if method == "h_model":
        # Closed-form H-model: P = [D0*(1+g) + D0*H_half*(gh - g)] / (ke - g)
        # where H_half = high_growth_years / 2 is the half-life of the linear fade.
        gh = _initial_high_growth(assumptions, fin, company, notes)
        g = _clamp_terminal_g(g_terminal, "H-model terminal")
        if is_num(assumptions.high_growth_years) and assumptions.high_growth_years > 0:
            h_years = assumptions.high_growth_years
        else:
            h_years = 5
            notes.append("high_growth_years missing/0; defaulted H-model horizon to 5 years.")
        h_half = h_years / 2.0
        denom = ke - g
        numer = d0 * (1.0 + g) + d0 * h_half * (gh - g)
        price = safe_div(numer, denom)
        detail.update({"high_growth": gh, "terminal_growth_used": g, "H_half": h_half})
        detail["implied_price"] = price
        if not is_num(price) or price < 0:
            notes.append("H-model DDM produced a non-positive/undefined price.")
            price = 0.0
        return DDMResult(method="h_model", implied_price=float(price), cost_of_equity=ke, detail=detail)

    # -------------------------------------------------------------- two_stage
    # (default; also catches any unrecognised method string)
    if method != "two_stage":
        notes.append(f"Unknown DDM method '{method}'; defaulting to two_stage.")
        method = "two_stage"
        detail["method"] = method

    gh = _initial_high_growth(assumptions, fin, company, notes)
    hgy = assumptions.high_growth_years
    h_years = max(1, int(hgy)) if (is_num(hgy) and hgy > 0) else 5
    g = _clamp_terminal_g(g_terminal, "Two-stage terminal")

    # Stage 1: explicit dividends grown at gh, discounted at ke.
    stage_pvs: list[float] = []
    dividends: list[float] = []
    d_prev = d0
    pv_stage1 = 0.0
    for t in range(1, h_years + 1):
        d_t = d_prev * (1.0 + gh)
        df = 1.0 / ((1.0 + ke) ** t)
        pv = d_t * df
        dividends.append(d_t)
        stage_pvs.append(pv)
        pv_stage1 += pv
        d_prev = d_t

    # Stage 2: Gordon perpetuity on the dividend at the end of the high-growth
    # phase, valued at year H then discounted back H years.
    d_h = d_prev  # dividend at the end of year H (the last grown dividend)
    tv = safe_div(d_h * (1.0 + g), ke - g)
    if not is_num(tv):
        tv = 0.0
        notes.append("Two-stage terminal value undefined; set to 0.")
    pv_terminal = tv / ((1.0 + ke) ** h_years)

    price = pv_stage1 + pv_terminal
    if not is_num(price) or price < 0:
        notes.append("Two-stage DDM produced a non-positive/undefined price.")
        price = 0.0

    detail.update(
        {
            "high_growth": gh,
            "high_growth_years": h_years,
            "terminal_growth_used": g,
            "dividends": dividends,
            "stage1_pvs": stage_pvs,
            "pv_stage1": pv_stage1,
            "terminal_value": tv,
            "pv_terminal": pv_terminal,
            "implied_price": price,
        }
    )
    return DDMResult(method="two_stage", implied_price=float(price), cost_of_equity=ke, detail=detail)


def _initial_high_growth(
    assumptions: DDMAssumptions,
    fin: Optional[AnnualFinancials],
    company: CompanyData,
    notes: list[str],
) -> float:
    """Resolve the stage-1 (high) growth rate for two-stage / H-model DDM.

    Precedence:
      1. ``assumptions.high_growth_rate`` if explicitly supplied.
      2. else ``min(sustainable growth = ROE*retention, dividend CAGR)`` over the
         candidates that are actually computable (the dividend CAGR covers only
         the paying run since the last cut, suspension, restart or step
         change, with a one-year special dividend skipped; ``_dividend_cagr``).
      3. else fall back to the terminal growth (a conservative flat assumption).

    Finite-stage growth may exceed the cost of equity: these dividends are an
    explicitly summed finite stream. Only the perpetual terminal rate must be
    below ke. Rates at or below -100% cannot describe a positive dividend.
    """
    gh = assumptions.high_growth_rate
    if not is_num(gh):
        candidates: list[float] = []
        if fin is not None:
            sg = _sustainable_growth(fin, company, notes)
            if is_num(sg):
                candidates.append(sg)
            dcg = _dividend_cagr(fin, notes)
            if is_num(dcg):
                candidates.append(dcg)
        if candidates:
            gh = min(candidates)
        else:
            gh = assumptions.terminal_growth if is_num(assumptions.terminal_growth) else 0.0
            notes.append(
                "No ROE/retention or dividend-CAGR signal; high growth set to terminal growth."
            )

    if not is_num(gh):
        gh = 0.0

    if gh <= -1.0:
        raise ValueError("high_growth_rate must be greater than -1")
    return gh


# --------------------------------------------------------------------------- #
#  Free Cash Flow to Equity model
# --------------------------------------------------------------------------- #
def run_fcfe(
    company: CompanyData,
    macro: MacroAssumptions,
    assumptions: DDMAssumptions,
    current_price: float,
) -> FCFEResult:
    """Levered FCFE DCF -> implied price per share.

    FCFE definition (levered free cash flow available to equity holders):

        FCFE_t = NetIncome_t + D&A_t - Capex_t - ΔNWC_t + ΔDebt_t

    ΔDebt CHOICE (documented): we assume the firm maintains a constant
    debt-to-revenue ratio, so net new borrowing grows the debt balance in line
    with revenue: ΔDebt_t = (total_debt / revenue_0) * (revenue_t - revenue_{t-1}),
    i.e. D_{t-1} * g_t. This is a standard "constant capital structure"
    simplification when no explicit debt schedule is available. If the current
    debt balance is unavailable, ΔDebt falls back to 0 (a note is recorded). This
    keeps leverage neutral rather than assuming aggressive re-levering.

    Projection mechanics:
      * Revenue grows along the historical-CAGR path faded to terminal growth
        (same derivation philosophy as the DCF, re-implemented locally).
      * NetIncome_t = latest net margin * projected revenue_t. The margin is
        rebuilt at the DCF's tax rate (``macro.tax_rate`` if set, else the
        marginal rate) when the tax history is too thin for an effective rate,
        a one-off spike in the latest net or EBIT margin is taken out, and a
        net margin that collapsed below what EBIT implies is replaced (all
        noted; see ``_latest_net_margin``). After a charge year (a loss after
        three or more profitable years) the margin fades from the loss to the
        prior median (capped at the last profitable year's) instead, as the
        DCF's EBIT margin does
        (``_charge_year_net_target``; ``detail['net_margin_path']``).
      * D&A_t / Capex_t = (historical revenue-weighted % of revenue) * revenue_t;
        growth-phase capex (above 1.5x D&A, fully from 2x, never lower for
        more historical capex, and never below 1.5x D&A) fades with revenue
        growth, as in the DCF (``utils.growth_capex_path``, against the same
        all-years D&A % that is added back; noted when the final year ends at
        least 0.5pp below history). With no usable capex
        year at all, capex is set equal to D&A (maintenance only, with a
        WARNING note) rather than 0, as in the DCF; not for a bank, insurer,
        REIT or lender (``utils.MAINTENANCE_CAPEX_KINDS``), whose
        reference-only FCFE keeps capex at 0.
      * ΔNWC_t = (historical pooled ΔNWC / Δrevenue) * Δrevenue_t, one-off years
        left out and set to 0 when the ratio falls outside [0, 1], as in the DCF
        (the ΔNWC series is often zero-filled by the provider).
      * Discount each FCFE_t at the cost of equity (CAPM).
      * Terminal value: Gordon growth on FCFE_N at terminal_growth, discounted N
        years (require ke - g >= 0.005, clamp g if needed).
      * equity_value = sum(PV_FCFE) + PV_terminal; implied_price = equity/shares.
    """
    notes: list[str] = []
    ke = cost_of_equity(company, macro)
    if not is_num(ke) or ke <= 0:
        raise ValueError("FCFE requires a positive finite cost of equity")
    fin = getattr(company, "financials", None)
    bs = getattr(company, "balance_sheet", None)

    fy = assumptions.forecast_years
    years_out = max(1, int(fy)) if (is_num(fy) and fy > 0) else 5
    g_terminal = assumptions.terminal_growth if is_num(assumptions.terminal_growth) else 0.0
    if g_terminal <= -1.0:
        raise ValueError("terminal_growth must be greater than -1")
    shares = _shares(company)

    detail: dict = {
        "cost_of_equity": ke,
        "terminal_growth": g_terminal,
        "delta_debt_policy": "constant debt/revenue ratio (debt grows with revenue)",
        "valuation_available": False,
        "notes": notes,
    }

    # --- guard: we need at least a revenue base and a net margin to project. --- #
    base_revenue = None
    if fin is not None:
        rev_hist = [v for v in (getattr(fin, "revenue", None) or []) if is_num(v)]
        if rev_hist:
            base_revenue = rev_hist[-1]
    # An explicit tax rate is the one the DCF applies to NOPAT: use it for the
    # net-margin adjustments too, so both models tax the same profit alike.
    tax_override = macro.tax_rate if (macro is not None and is_num(macro.tax_rate)) else None
    net_margin = _latest_net_margin(fin, notes, tax_override) if fin is not None else None
    # After a charge year the margin fades from the loss to the prior median,
    # as the DCF's EBIT margin does; otherwise it is held flat.
    net_target = _charge_year_net_target(fin, net_margin, notes) if fin is not None else None

    # Forward fiscal-year labels for the projection horizon.
    last_fy = None
    if fin is not None:
        fys = [int(y) for y in (getattr(fin, "fiscal_years", None) or []) if is_num(y)]
        if fys:
            last_fy = fys[-1]
    proj_years = (
        [last_fy + i for i in range(1, years_out + 1)]
        if last_fy is not None
        else list(range(1, years_out + 1))
    )

    # If we cannot even establish a revenue base or margin, return a graceful,
    # zero-valued result rather than crashing.
    if not is_num(base_revenue) or base_revenue <= 0 or not is_num(net_margin):
        notes.append("Insufficient revenue/margin history to project FCFE; returning zeros.")
        zeros = [0.0] * years_out
        return FCFEResult(
            years=proj_years,
            fcfe=zeros,
            pv_fcfe=zeros,
            terminal_value=0.0,
            pv_terminal=0.0,
            equity_value=0.0,
            shares=shares if is_num(shares) else 0.0,
            implied_price=0.0,
            current_price=current_price,
            cost_of_equity=ke,
            detail=detail,
        )

    # --- per-revenue ratios from history (held flat over the horizon) --------- #
    rev_series = getattr(fin, "revenue", None)
    da_pct = _ratio_of_revenue(getattr(fin, "dep_amort", None), rev_series)
    capex_pct = _ratio_of_revenue(getattr(fin, "capex", None), rev_series)
    if not is_num(da_pct):
        da_pct = 0.0
        notes.append("No usable D&A history; D&A set to 0% of revenue.")
    if not is_num(capex_pct) and da_pct > 0 and financial_kind(company) in MAINTENANCE_CAPEX_KINDS:
        # With D&A added back, a capex of 0 would leave the business with no
        # reinvestment at all (PSX, NEE, AER): assume maintenance capex, as
        # the DCF does (not for a bank, insurer, REIT or lender).
        capex_pct = da_pct
        notes.append(
            f"WARNING: No usable capex history (zero or unreported in every year) while D&A is "
            f"{da_pct:.1%} of revenue; capex set equal to D&A (maintenance only) rather than 0, "
            "which would add D&A back with no reinvestment and overstate FCFE.")
    elif not is_num(capex_pct):
        capex_pct = 0.0
        notes.append("No usable capex history; capex set to 0% of revenue.")

    # Incremental NWC as a fraction of the change in revenue (pooled over history,
    # one-off years left out as in the DCF).
    nwc_pct_delta, one_off = _nwc_per_revenue_change(fin)
    if one_off:
        notes.append(
            f"ΔNWC in {fiscal_year_labels(getattr(fin, 'fiscal_years', None), one_off)} exceeds "
            f"{NWC_ONE_OFF_REVENUE_SHARE:.0%} of revenue and that year's revenue change; left "
            "out of ΔNWC/Δrevenue as a one-off.")
    if not is_num(nwc_pct_delta):
        nwc_pct_delta = 0.0
        notes.append("Unstable/absent ΔNWC history; incremental NWC set to 0% of Δrevenue.")
    elif not 0.0 <= nwc_pct_delta <= 1.0:
        # Same plausibility band as the DCF: outside [0, 1] is noise, not intensity.
        notes.append(f"Derived ΔNWC/Δrevenue {nwc_pct_delta:.3f} implausible; "
                     "incremental NWC set to 0% of Δrevenue.")
        nwc_pct_delta = 0.0

    # Current debt balance for the ΔDebt (debt grows with revenue) policy.
    total_debt = getattr(bs, "total_debt", None) if bs is not None else None
    if not is_num(total_debt) or total_debt < 0:
        total_debt = 0.0
        notes.append("Total debt unavailable or negative; ΔDebt set to 0 (no re-levering).")

    # Terminal growth used consistently as BOTH the fade endpoint of the growth
    # path AND the perpetuity growth, so FCFE_N and the Gordon terminal share one g.
    g_term = min(g_terminal, ke - _MIN_KE_G_SPREAD)
    if g_term < g_terminal:
        notes.append(
            f"Terminal growth {g_terminal:.4f} too close to/above ke {ke:.4f}; "
            f"clamped to {g_term:.4f} (used in both fade path and terminal value)."
        )

    growth_path = _revenue_growth_path(fin, g_term, years_out)
    hist_cagr = _hist_revenue_cagr(fin)
    if hist_cagr is None:
        notes.append("Historical revenue CAGR unavailable; revenue growth held at terminal growth.")
    # Growth-phase capex (well above D&A) fades with revenue growth, as in the DCF.
    capex_pcts = [capex_pct] * years_out
    if is_num(hist_cagr):
        ref_growth = max(config.DEFAULT_REVENUE_GROWTH_FLOOR,
                         min(config.DEFAULT_REVENUE_GROWTH_CAP, hist_cagr))
        faded = growth_capex_path(capex_pct, da_pct, ref_growth, growth_path)
        if faded is not None:
            capex_pcts = faded
        if material_capex_fade(capex_pct, faded):
            # (A negligible fade is applied silently; capex_pct_path records it.)
            notes.append(
                f"Capex {capex_pct:.1%} of revenue is growth-phase (D&A {da_pct:.1%}); net "
                f"capex scaled with revenue growth, reaching {faded[-1]:.1%} of revenue in "
                f"year {years_out}.")
    margin_path = (fade_path(net_margin, net_target, years_out) if is_num(net_target)
                   else [net_margin] * years_out)
    detail["revenue_growth_path"] = growth_path
    detail["net_margin"] = net_margin
    if is_num(net_target):
        # Per-year net margin actually applied (only differs from net_margin
        # after a charge year).
        detail["net_margin_path"] = list(margin_path)
    detail["da_pct_revenue"] = da_pct
    detail["capex_pct_revenue"] = capex_pct
    detail["capex_pct_path"] = list(capex_pcts)
    detail["nwc_pct_delta_revenue"] = nwc_pct_delta

    # --- project the FCFE series --------------------------------------------- #
    # base_revenue > 0 is guaranteed by the guard above.
    debt_to_revenue = total_debt / float(base_revenue)
    revenues: list[float] = []
    fcfe: list[float] = []
    pv_fcfe: list[float] = []
    prev_rev = float(base_revenue)
    for t in range(1, years_out + 1):
        g = growth_path[t - 1] if (t - 1) < len(growth_path) else g_term
        rev_t = prev_rev * (1.0 + g)
        d_rev = rev_t - prev_rev

        ni_t = (margin_path[t - 1] if (t - 1) < len(margin_path) else net_margin) * rev_t
        da_t = da_pct * rev_t
        capex_t = (capex_pcts[t - 1] if (t - 1) < len(capex_pcts) else capex_pct) * rev_t
        dnwc_t = nwc_pct_delta * d_rev
        # ΔDebt: keep debt/revenue constant -> borrow (D0/R0) per unit of revenue
        # growth, which equals D_{t-1} * g_t on the rolled-forward balance.
        ddebt_t = debt_to_revenue * d_rev

        fcfe_t = ni_t + da_t - capex_t - dnwc_t + ddebt_t
        df = 1.0 / ((1.0 + ke) ** t)
        pv_t = fcfe_t * df

        revenues.append(rev_t)
        fcfe.append(fcfe_t)
        pv_fcfe.append(pv_t)
        prev_rev = rev_t

    detail["revenue"] = revenues

    # --- terminal value: Gordon on FCFE_N (shares g_term with the fade path) -- #
    fcfe_n = fcfe[-1] if fcfe else 0.0
    tv = safe_div(fcfe_n * (1.0 + g_term), ke - g_term)
    if not is_num(tv):
        tv = 0.0
        notes.append("FCFE terminal value undefined; set to 0.")
    elif fcfe_n < 0:
        notes.append("Final-year FCFE is negative, so the terminal value capitalises a "
                     "perpetual cash outflow; review the margin and reinvestment drivers.")
    pv_terminal = tv / ((1.0 + ke) ** years_out)

    equity_value = sum(pv_fcfe) + pv_terminal
    implied_price = safe_div(equity_value, shares)
    detail["valuation_available"] = is_num(implied_price)
    if not is_num(implied_price):
        implied_price = 0.0
        notes.append("Share count unavailable; implied price set to 0.")

    detail["terminal_growth_used"] = g_term
    detail["equity_value"] = equity_value

    return FCFEResult(
        years=proj_years,
        fcfe=fcfe,
        pv_fcfe=pv_fcfe,
        terminal_value=float(tv),
        pv_terminal=float(pv_terminal),
        equity_value=float(equity_value),
        shares=float(shares) if is_num(shares) else 0.0,
        implied_price=float(implied_price),
        current_price=current_price,
        cost_of_equity=ke,
        detail=detail,
    )


def _nwc_per_revenue_change(fin: AnnualFinancials) -> tuple[Optional[float], list[int]]:
    """(pooled ΔNWC / Δrevenue, positions of years left out as one-offs).

    The provider stores ``change_in_nwc`` already as the per-year increase in net
    working capital (positive = cash use). Each year's ΔNWC pairs with that year's
    revenue change, pooled as sum(ΔNWC) / sum(Δrevenue) so a near-flat year cannot
    dominate, and a year whose ΔNWC exceeds 10% of revenue and its revenue change
    is left out as a one-off (the DCF uses the same estimator,
    ``utils.screened_incremental_ratio``). The ratio is None if there is no usable
    history; the caller applies the DCF's [0, 1] plausibility band.
    """
    return screened_incremental_ratio(getattr(fin, "change_in_nwc", None),
                                      getattr(fin, "revenue", None))
