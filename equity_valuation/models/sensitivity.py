"""Sensitivity analysis and football-field assembly.

Two public entry points:

  * ``dcf_sensitivity`` re-runs the unlevered FCFF DCF (``models.dcf.run_dcf``)
    across two 2-D grids, capturing the implied price per cell. The column axis
    follows the headline DCF's terminal method, so the centre cell of each grid
    is the headline price:
        Gordon:        Grid 1 — WACC x terminal growth
                       Grid 2 — EBIT margin x terminal growth
        Exit multiple: Grid 1 — WACC x exit EV/EBITDA
                       Grid 2 — EBIT margin x exit EV/EBITDA
    Each cell is an independent DCF run on a *cloned* macro/assumptions pair
    (via ``dataclasses.replace``) so the base inputs are never mutated.

  * ``build_football_field`` collapses the populated ``ValuationReport`` into a
    list of ``FootballFieldRow`` low/base/high bars — one per valuation method
    that has usable inputs (52-week range, DCF, comps, DDM, FCFE).

Design notes:
  * Money is in absolute units; growth/margins/WACC are decimals (see schemas).
  * Nothing here ever raises on a per-cell modelling failure: a failed DCF cell
    stores ``float('nan')`` so the grid stays rectangular and the exporters can
    render a blank cell.
  * Row/column *values* are the ACTUAL resulting levels (e.g. the realized WACC
    read back off the returned ``DCFResult``), not the raw deltas, so the labels
    on the grid are economically meaningful. A cell the DCF could only price by
    changing its inputs (terminal g clamped below WACC, or a non-positive WACC
    replaced by the fallback rate), or an invalid one (an exit multiple <= 0),
    is stored as NaN rather than shown under a label it does not represent.
  * Axis labels tell renderers how to format values: a label containing
    "EV/EBITDA" or "multiple" holds multiples (12.0 -> "12.0x"); "WACC",
    "Terminal growth" and "EBIT margin" hold decimal rates.

"""

from __future__ import annotations

from dataclasses import replace

from .. import config
from ..schemas import (
    CompanyData,
    DCFAssumptions,
    FootballFieldRow,
    MacroAssumptions,
    SensitivityResult,
)
from ..utils import (
    EBIT_DERIVED_KINDS,
    NOT_IN_BLEND,
    ddm_reference_only,
    financial_institution,
    financial_kind,
    is_num,
    median,
    net_debt_parts,
)
from .dcf import margin_fade_target, resolve_terminal_method, run_dcf, start_ebit_margin


# --------------------------------------------------------------------------- #
#  Small internal helpers
# --------------------------------------------------------------------------- #
def _safe_implied_price(
    company: CompanyData,
    macro: MacroAssumptions,
    assumptions: DCFAssumptions,
    current_price: float,
):
    """Run one DCF cell, returning (implied_price, realized_wacc).

    Never raises: on any failure both elements degrade to ``float('nan')`` so the
    caller can keep building a rectangular grid. The price is also NaN when the
    DCF had to clamp the cell's terminal growth or replace a non-positive WACC,
    since the cell would then not be priced at its row/column labels.
    """
    try:
        result = run_dcf(company, macro, assumptions, current_price)
    except Exception:  # defensive: e.g. no positive revenue base
        return float("nan"), float("nan")

    price = getattr(result, "implied_price", None)
    price = float(price) if is_num(price) else float("nan")

    # Read the realized WACC back off the result so the row label reflects the
    # level the model actually used (rf bump propagates through CAPM + weights).
    realized_wacc = float("nan")
    wacc_obj = getattr(result, "wacc", None)
    detail = getattr(wacc_obj, "detail", None) or {}
    if wacc_obj is not None:
        w = getattr(wacc_obj, "wacc", None)
        if "wacc_computed" in detail:
            # Discounted at the fallback rate: keep the CAPM WACC as the (ordered)
            # row label and blank the cell.
            w = detail["wacc_computed"]
            price = float("nan")
        if is_num(w):
            realized_wacc = float(w)

    a = getattr(result, "assumptions", None) or {}
    g_req, g_used = a.get("terminal_growth"), a.get("terminal_growth_used")
    if is_num(g_req) and is_num(g_used) and g_used != g_req:
        price = float("nan")  # g clamped to WACC - gap: not the column's growth
    return price, realized_wacc


def _base_latest_ebit_margin(company: CompanyData) -> float:
    """The target EBIT margin the headline DCF fades to without an explicit
    one, or NaN if it can't be computed.

    Uses the DCF's own derivation: the start margin (latest EBIT / revenue, or
    the recent median after a one-off spike), or, after a charge year or a
    collapse year, the target the DCF fades to (``dcf.margin_fade_target``).
    The margin rows set the target margin, so the centre row is the headline.
    """
    fin = getattr(company, "financials", None)
    if fin is None:
        return float("nan")
    revenue = list(getattr(fin, "revenue", None) or [])
    rev_latest = revenue[-1] if revenue else None
    if not is_num(rev_latest) or rev_latest <= 0:
        return float("nan")
    derive = financial_kind(company) in EBIT_DERIVED_KINDS  # as the DCF
    margin, _ = start_ebit_margin(fin, revenue, rev_latest, derive_ebit=derive)
    fade = margin_fade_target(fin, revenue, derive_ebit=derive) if is_num(margin) else None
    if fade is not None:
        margin = fade["target"]
    return float(margin) if is_num(margin) else float("nan")


# --------------------------------------------------------------------------- #
#  Public API — sensitivity grids
# --------------------------------------------------------------------------- #
def dcf_sensitivity(
    company: CompanyData,
    macro: MacroAssumptions,
    assumptions: DCFAssumptions,
    current_price: float,
) -> list[SensitivityResult]:
    """Build the WACC and EBIT-margin implied-price sensitivity grids.

    The column axis matches the headline DCF's terminal method, so each grid's
    centre cell equals the headline implied price:
      * Gordon (or a fallback to it): columns are terminal growth levels, label
        ``"Terminal growth"``, and every cell runs the Gordon method.
      * Exit multiple: columns are exit EV/EBITDA multiples (the headline
        multiple +/- ``config.SENSITIVITY_EXIT_MULTIPLE_DELTAS``, e.g. 10.0 to
        14.0 around 12.0; the steps shrink for a multiple no larger than the
        widest step so every column stays positive), label
        ``"Exit EV/EBITDA"``, and every cell runs the exit-multiple method.

    Returns a list of up to two ``SensitivityResult`` objects. A grid that can't
    be built at all (e.g. base margin unknown) is still returned, populated with
    ``float('nan')`` cells, so downstream consumers see consistent structure.
    """
    results: list[SensitivityResult] = []

    growth_deltas = list(config.SENSITIVITY_GROWTH_DELTAS)
    wacc_deltas = list(config.SENSITIVITY_WACC_DELTAS)
    margin_deltas = list(config.SENSITIVITY_MARGIN_DELTAS)

    base_growth = getattr(assumptions, "terminal_growth", None)
    if not is_num(base_growth):
        base_growth = config.DEFAULT_TERMINAL_GROWTH
    base_growth = float(base_growth)

    base_rf = getattr(macro, "risk_free_rate", None)
    if not is_num(base_rf):
        base_rf = config.DEFAULT_RISK_FREE_RATE
    base_rf = float(base_rf)

    # Column axis, shared by both grids: exit multiples when the headline DCF
    # uses one, else terminal-growth levels (Gordon).
    headline_method, _ = resolve_terminal_method(assumptions)
    if headline_method == "exit_multiple":
        base_multiple = float(assumptions.exit_ev_ebitda)
        steps = list(config.SENSITIVITY_EXIT_MULTIPLE_DELTAS)
        widest = max((abs(d) for d in steps), default=0.0)
        if base_multiple > 0 and widest >= base_multiple:
            # Shrink the steps for a very low multiple so every column stays a
            # positive multiple (1.5x -> 0.75x..2.25x instead of -0.5x..3.5x).
            steps = [d * base_multiple / (2.0 * widest) for d in steps]
        col_levels = [base_multiple + d for d in steps]
        col_label = "Exit EV/EBITDA"
        col_title = "exit EV/EBITDA"

        def cell_assumptions(base: DCFAssumptions, col: float):
            if col <= 0:
                return None  # not a valid multiple -> blank cell
            return replace(base, exit_ev_ebitda=col, terminal_method="exit_multiple")
    else:
        col_levels = [base_growth + d for d in growth_deltas]
        col_label = "Terminal growth"
        col_title = "terminal growth"

        def cell_assumptions(base: DCFAssumptions, col: float):
            return replace(base, terminal_growth=col, terminal_method="gordon")

    # ----------------------------------------------------------------------- #
    # Grid 1 — WACC (rows) x terminal growth or exit multiple (cols)
    #
    # We shift WACC by bumping the risk-free rate by each delta. Because
    # ke = rf + beta*ERP and kd often keys off rf, a +Δ on rf moves the realized
    # WACC by roughly +Δ. We capture the ACTUAL realized WACC for the row label
    # rather than assuming the bump passes through one-for-one.
    # ----------------------------------------------------------------------- #
    grid1: list[list[float]] = []
    row_wacc_levels: list[float] = []
    for wd in wacc_deltas:
        bumped_macro = replace(macro, risk_free_rate=base_rf + wd)
        row_prices: list[float] = []
        realized_for_row = float("nan")
        for col in col_levels:
            cell = cell_assumptions(assumptions, col)
            if cell is None:
                row_prices.append(float("nan"))
                continue
            price, realized_wacc = _safe_implied_price(
                company, bumped_macro, cell, current_price
            )
            row_prices.append(price)
            # Realized WACC is independent of the column input, so the first
            # finite read for this row is representative of the whole row.
            if not is_num(realized_for_row) and is_num(realized_wacc):
                realized_for_row = realized_wacc
        # Fallback label if every cell in the row failed: approximate the WACC as
        # the base WACC shifted by the rf delta (best-effort, keeps labels sane).
        row_wacc_levels.append(realized_for_row)
        grid1.append(row_prices)

    # If some rows never produced a realized WACC, back-fill the label by adding
    # the rf delta to the nearest known realized WACC so the axis stays ordered.
    _backfill_wacc_labels(row_wacc_levels, wacc_deltas)

    results.append(
        SensitivityResult(
            title=f"DCF implied price: WACC vs {col_title}",
            row_label="WACC",
            col_label=col_label,
            row_values=row_wacc_levels,
            col_values=list(col_levels),
            grid=grid1,
        )
    )

    # ----------------------------------------------------------------------- #
    # Grid 2 — EBIT margin (rows) x terminal growth or exit multiple (cols)
    #
    # Margin rows are the base latest EBIT margin shifted by each absolute delta.
    # If the base margin can't be derived we still emit the grid (all NaN) so the
    # structure is predictable.
    # ----------------------------------------------------------------------- #
    base_margin = _base_latest_ebit_margin(company)
    # If the caller already pinned a target margin, center the grid on that
    # instead of the raw historical margin (it's what the DCF would actually use).
    explicit_target = getattr(assumptions, "target_ebit_margin", None)
    if is_num(explicit_target):
        base_margin = float(explicit_target)

    margin_levels = [
        (base_margin + d) if is_num(base_margin) else float("nan")
        for d in margin_deltas
    ]

    grid2: list[list[float]] = []
    for m in margin_levels:
        row_prices = []
        for col in col_levels:
            cell = cell_assumptions(assumptions, col) if is_num(m) else None
            if cell is None:
                row_prices.append(float("nan"))
                continue
            price, _ = _safe_implied_price(
                company, macro, replace(cell, target_ebit_margin=m), current_price
            )
            row_prices.append(price)
        grid2.append(row_prices)

    results.append(
        SensitivityResult(
            title=f"DCF implied price: EBIT margin vs {col_title}",
            row_label="EBIT margin",
            col_label=col_label,
            row_values=list(margin_levels),
            col_values=list(col_levels),
            grid=grid2,
        )
    )

    return results


def _backfill_wacc_labels(levels: list[float], deltas: list[float]) -> None:
    """In-place: fill NaN WACC row labels from a neighbour + the delta spread.

    If at least one row produced a realized WACC, infer the others by assuming
    the rf bump passes through one-for-one (the realized base WACC minus its
    delta gives an implied base WACC; add each delta back). Pure cosmetic — only
    affects axis labels, never the price grid.
    """
    if len(levels) != len(deltas):
        return
    # Find any anchor row that has a finite realized WACC.
    anchor_idx = next((i for i, v in enumerate(levels) if is_num(v)), None)
    if anchor_idx is None:
        return  # nothing to anchor on; leave NaNs as-is
    implied_base = levels[anchor_idx] - deltas[anchor_idx]
    for i, v in enumerate(levels):
        if not is_num(v):
            levels[i] = implied_base + deltas[i]


# --------------------------------------------------------------------------- #
#  Public API — football field
# --------------------------------------------------------------------------- #
def _finite_grid_values(grid) -> list[float]:
    """Flatten a 2-D grid into the list of its finite numeric cells."""
    out: list[float] = []
    if not grid:
        return out
    for row in grid:
        if not row:
            continue
        for cell in row:
            if is_num(cell):
                out.append(float(cell))
    return out


def _find_wacc_growth_grid(sensitivities):
    """Return the WACC-row SensitivityResult (x terminal growth, or x exit
    multiple for an exit-multiple DCF) from a list, or None."""
    if not sensitivities:
        return None
    for s in sensitivities:
        # Match on the row label we set above; fall back to a title substring.
        row_label = (getattr(s, "row_label", "") or "").lower()
        if "wacc" in row_label:
            return s
    return None


def _band(center: float, pct: float):
    """Return (low, base, high) around ``center`` scaled by (1-pct, 1, 1+pct).

    For a negative center, ``center*(1-pct)`` exceeds ``center*(1+pct)``, so we
    order the edges with min/max to keep the low <= base <= high invariant that
    the exporters rely on (the base is always ``center`` itself).
    """
    lo = center * (1.0 - pct)
    hi = center * (1.0 + pct)
    return min(lo, hi), center, max(lo, hi)


def build_football_field(report) -> list[FootballFieldRow]:
    """Assemble football-field low/base/high bars from a populated report.

    Reads ``report.company`` / ``report.market`` (via company.market),
    ``report.dcf`` / ``.comps`` / ``.ddm`` / ``.fcfe`` (any may be None) and
    ``report.sensitivities``. Only methods whose inputs exist produce a row, so
    the returned list length varies with data availability.

    Conventions:
      * '52-week range'  : low/high from market 52wk lo/hi, base = current_price.
      * 'DCF'            : WACC grid min/max (its centre cell is the headline
                            dcf.implied_price, which is the base; the bar is
                            still widened to contain it if that cell is blank),
                            else +/-15% around dcf.implied_price.
      * 'EV/EBITDA comps' / 'P/E comps': spread from comps stats applied to the
                            target metric where available, else the comps implied
                            price summary, else skipped.
      * 'DDM' / 'FCFE'   : +/-10% bands around their implied prices.
    For a bank, insurer, REIT or lender, a captive-finance group or a
    debt-funded lessor (``utils.financial_institution``) the DCF and FCFE bars
    are labelled 'DCF (not in blend)' / 'FCFE (not in blend)': the engine
    leaves both out of the blended target, and an unmarked bar would read as a
    valuation range. The DDM bar is marked the same way when the engine leaves
    a captive-finance group's or a lessor's DDM out (``utils.ddm_reference_only``:
    a low payout, or no comps to blend it with).
    """
    rows: list[FootballFieldRow] = []

    company = getattr(report, "company", None)
    mark = NOT_IN_BLEND if (company is not None and financial_institution(company)) else ""
    market = getattr(company, "market", None) if company is not None else None
    dcf = getattr(report, "dcf", None)
    comps = getattr(report, "comps", None)
    ddm = getattr(report, "ddm", None)
    fcfe = getattr(report, "fcfe", None)
    sensitivities = getattr(report, "sensitivities", None)

    current_price = getattr(report, "current_price", None)
    if not is_num(current_price):
        # Fall back to the live market price if the report didn't carry one.
        mp = getattr(market, "price", None) if market is not None else None
        current_price = float(mp) if is_num(mp) else float("nan")
    else:
        current_price = float(current_price)

    # ----------------------------------------------------------------------- #
    # 52-week range
    # ----------------------------------------------------------------------- #
    if market is not None:
        lo = getattr(market, "fifty_two_week_low", None)
        hi = getattr(market, "fifty_two_week_high", None)
        if is_num(lo) and is_num(hi):
            low, high = float(lo), float(hi)
            if low > high:  # guard against swapped fields
                low, high = high, low
            base = current_price if is_num(current_price) else (low + high) / 2.0
            # Keep the marker inside the bar for a clean render.
            base = min(max(base, low), high)
            rows.append(
                FootballFieldRow(
                    method="52-week range", low=low, base=base, high=high
                )
            )

    # ----------------------------------------------------------------------- #
    # DCF — prefer the WACC x growth sensitivity spread, else +/-15% band
    # ----------------------------------------------------------------------- #
    if dcf is not None:
        dcf_implied = getattr(dcf, "implied_price", None)
        grid_obj = _find_wacc_growth_grid(sensitivities)
        grid_vals = (
            _finite_grid_values(getattr(grid_obj, "grid", None))
            if grid_obj is not None
            else []
        )
        if grid_vals:
            low = min(grid_vals)
            high = max(grid_vals)
            med = median(grid_vals)
            # Center on the median of the grid; if the point estimate is finite,
            # prefer it as the base (it's the engine's headline number). The grid
            # follows the headline's terminal method, so its centre cell is the
            # headline; still widen the bar to contain the headline (e.g. when
            # that cell is blank) rather than moving the marker.
            base = float(dcf_implied) if is_num(dcf_implied) else float(med)
            low, high = min(low, base), max(high, base)
            rows.append(
                FootballFieldRow(method="DCF" + mark, low=low, base=base, high=high)
            )
        elif is_num(dcf_implied):
            low, base, high = _band(float(dcf_implied), 0.15)
            base = min(max(base, low), high)
            rows.append(
                FootballFieldRow(method="DCF" + mark, low=low, base=base, high=high)
            )

    # ----------------------------------------------------------------------- #
    # Comps — EV/EBITDA and P/E rows
    # ----------------------------------------------------------------------- #
    if comps is not None:
        rows.extend(_comps_rows(comps, company))

    # ----------------------------------------------------------------------- #
    # DDM / FCFE — +/-10% bands around the implied price
    # ----------------------------------------------------------------------- #
    if ddm is not None:
        ddm_price = getattr(ddm, "implied_price", None)
        if is_num(ddm_price):
            low, base, high = _band(float(ddm_price), 0.10)
            base = min(max(base, low), high)
            ddm_mark = NOT_IN_BLEND if ddm_reference_only(report) else ""
            rows.append(
                FootballFieldRow(method="DDM" + ddm_mark, low=low, base=base, high=high)
            )

    if fcfe is not None:
        fcfe_price = getattr(fcfe, "implied_price", None)
        if is_num(fcfe_price):
            low, base, high = _band(float(fcfe_price), 0.10)
            base = min(max(base, low), high)
            rows.append(
                FootballFieldRow(method="FCFE" + mark, low=low, base=base, high=high)
            )

    return rows


def _ev_bridge_inputs(company):
    """Pull the EV->equity bridge inputs (ebitda_latest, net_debt, minority,
    preferred, shares) from ``company`` for re-pricing EV multiples.

    Returns a dict with those keys, or ``None`` if the essential inputs
    (positive latest EBITDA and a usable share count) can't be resolved. Net
    debt / minority / preferred default to 0.0 when the balance sheet is absent,
    mirroring ``models.comps``.
    """
    if company is None:
        return None
    fin = getattr(company, "financials", None)
    bs = getattr(company, "balance_sheet", None)
    market = getattr(company, "market", None)

    ebitda = getattr(fin, "ebitda", None) if fin is not None else None
    ebitda_latest = ebitda[-1] if ebitda else None
    if not is_num(ebitda_latest) or float(ebitda_latest) <= 0:
        return None

    # Prefer market shares outstanding, else latest diluted weighted-average.
    shares = getattr(market, "shares_outstanding", None) if market is not None else None
    if not is_num(shares) or float(shares) <= 0:
        diluted = getattr(fin, "diluted_shares", None) if fin is not None else None
        shares = diluted[-1] if diluted else None
    if not is_num(shares) or float(shares) <= 0:
        return None

    net_debt = minority = preferred = 0.0
    if bs is not None:
        net_debt, _ = net_debt_parts(bs)
        mi = getattr(bs, "minority_interest", 0.0)
        minority = float(mi) if is_num(mi) else 0.0
        pe_eq = getattr(bs, "preferred_equity", 0.0)
        preferred = float(pe_eq) if is_num(pe_eq) else 0.0

    return {
        "ebitda_latest": float(ebitda_latest),
        "net_debt": net_debt,
        "minority": minority,
        "preferred": preferred,
        "shares": float(shares),
    }


def _ev_price_at_multiple(mult: float, bridge: dict):
    """Equity price implied by an EV/EBITDA ``mult`` via the EV->equity bridge.

    price = (mult*ebitda - net_debt - minority - preferred) / shares.
    Returns ``None`` if shares are unusable (guards divide-by-zero).
    """
    shares = bridge.get("shares")
    if not is_num(shares) or float(shares) == 0:
        return None
    ev_star = mult * bridge["ebitda_latest"]
    equity_star = ev_star - bridge["net_debt"] - bridge["minority"] - bridge["preferred"]
    return equity_star / float(shares)


def _comps_rows(comps, company=None) -> list[FootballFieldRow]:
    """Build the EV/EBITDA and P/E football-field rows from a CompsResult.

    Strategy per multiple:
      1. Use the implied price from the peer median as the base.
      2. Spread the band by the dispersion of the peer multiple:
           * Equity multiples (P/E, P/B) scale the base price proportionally by
             p25/median and p75/median.
           * EV multiples (EV/EBITDA) re-apply the EV->equity bridge at the p25
             and p75 peer multiples, since proportional scaling of an equity
             price is only valid for equity multiples. If the bridge inputs
             aren't available, this falls back to the symmetric band below.
      3. If the per-multiple stats aren't usable, fall back to the overall
         ``implied_price_summary`` low/median/high.
    """
    out: list[FootballFieldRow] = []

    stats = getattr(comps, "stats", None) or {}
    implied = getattr(comps, "implied", None) or {}
    summary = getattr(comps, "implied_price_summary", None) or {}

    # EV multiples need the EV->equity bridge to re-price at p25/p75; equity
    # multiples (P/E, P/B) use proportional scaling.
    ev_keys = {"ev_ebitda", "ev_sales"}
    bridge = _ev_bridge_inputs(company)

    mapping = [
        ("ev_ebitda", "EV/EBITDA comps"),
        ("pe", "P/E comps"),
    ]

    used_per_multiple = False
    for key, label in mapping:
        base_price = implied.get(key) if isinstance(implied, dict) else None
        if not is_num(base_price):
            continue
        base_price = float(base_price)

        mult_stats = stats.get(key) if isinstance(stats, dict) else None
        low_price = high_price = None
        if isinstance(mult_stats, dict):
            med = mult_stats.get("median")
            p25 = mult_stats.get("p25")
            p75 = mult_stats.get("p75")
            if key in ev_keys and bridge is not None:
                # Re-apply the EV->equity bridge at the p25/p75 peer multiples:
                # price(mult) = (mult*ebitda - net_debt - minority - preferred)/shares.
                if is_num(p25):
                    low_price = _ev_price_at_multiple(float(p25), bridge)
                if is_num(p75):
                    high_price = _ev_price_at_multiple(float(p75), bridge)
            elif is_num(med) and med != 0:
                # Scale the implied price by the dispersion of the peer multiple.
                if is_num(p25):
                    low_price = base_price * (float(p25) / float(med))
                if is_num(p75):
                    high_price = base_price * (float(p75) / float(med))

        if not is_num(low_price) or not is_num(high_price):
            # Fall back to a symmetric +/-10% band around the implied price.
            lo, _, hi = _band(base_price, 0.10)
            low_price = lo if not is_num(low_price) else low_price
            high_price = hi if not is_num(high_price) else high_price

        low_price, high_price = float(low_price), float(high_price)
        if low_price > high_price:
            low_price, high_price = high_price, low_price
        base = min(max(base_price, low_price), high_price)
        out.append(
            FootballFieldRow(
                method=label, low=low_price, base=base, high=high_price
            )
        )
        used_per_multiple = True

    # Fallback: if neither per-multiple implied price was usable but the engine
    # produced an overall summary, emit a single generic "Comps" bar.
    if not used_per_multiple and isinstance(summary, dict):
        lo = summary.get("low")
        med = summary.get("median")
        hi = summary.get("high")
        if is_num(lo) and is_num(hi):
            low_price, high_price = float(lo), float(hi)
            if low_price > high_price:
                low_price, high_price = high_price, low_price
            base = float(med) if is_num(med) else (low_price + high_price) / 2.0
            base = min(max(base, low_price), high_price)
            out.append(
                FootballFieldRow(
                    method="Comps", low=low_price, base=base, high=high_price
                )
            )

    return out
