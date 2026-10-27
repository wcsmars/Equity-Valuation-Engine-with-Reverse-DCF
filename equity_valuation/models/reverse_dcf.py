"""Reverse DCF: what the market price implies about one DCF driver at a time.

The forward DCF (``models.dcf.run_dcf``) turns drivers into a price. The
reverse DCF asks which value of a single driver reproduces the market price
when every other input is held at the value the forward run resolved
(``DCFResult.assumptions``). Three drivers are solved independently:

  * Year-1 revenue growth (the top-level keys). The resolved growth path is
    shifted by a first-year delta that fades linearly to zero in the final
    explicit year, so a linear fade stays linear and a custom schedule keeps
    its shape and its final-year endpoint. Searched over -40% to +80% in 5%
    steps.
  * The target (steady-state) EBIT margin (``margin``): the margin the
    projection fades to from the latest year's, with the growth path held at
    the resolved one. Searched over -20% to +80% in 2% steps.
  * The discount rate (``wacc``): the WACC at which the resolved cash flows
    (growth, target margin and the terminal growth the forward run used, after
    any clamp, held) are worth the price, applied through
    ``run_dcf(..., wacc_override=w)``. Searched in 0.25% steps from that
    terminal growth + ``config.MAX_TERMINAL_GROWTH_VS_WACC`` + 0.01% (so the
    Gordon clamp never moves the terminal growth; 2% for an exit multiple) to
    25%.

Method (``_solve``). The implied price need not be monotone in a driver (a
negative margin makes extra revenue destroy value, for instance), so each
range is scanned on its coarse grid first. A curve that matches the price at
every grid point identifies nothing. An exact grid hit (within 1e-9 x price,
a tangency included) is a solution. Otherwise the lowest sign change is
bisected, at most 60 halvings, stopping within 1e-9 x price, and is reported
as converged only when the repriced value is within 1e-6 x price. The
tolerances are relative, so a penny stock solves as precisely as a $500 one,
and a jump across the price is not mistaken for a root. When several values
fit, the lowest is shown with a note; when none does, the note gives the
model's price span over the range.

The result is a diagnostic. This module never raises, and each leg degrades
to None on its own. For a bank, insurer, REIT, lender/BDC, captive-finance
group or debt-funded lessor (``utils.financial_institution_detail``) the FCFF
DCF is reference-only, so nothing is solved and a note says why.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional

from .. import config
from ..schemas import DCFAssumptions
from ..utils import fade_path, financial_institution_detail, is_num
from . import dcf as _dcf

# Search ranges and coarse-grid steps per driver (decimals).
GROWTH_RANGE = (-0.40, 0.80, 0.05)
MARGIN_RANGE = (-0.20, 0.80, 0.02)
WACC_MAX, WACC_STEP = 0.25, 0.0025
EXIT_MULTIPLE_WACC_MIN = 0.02
# Margin kept above terminal growth + MAX_TERMINAL_GROWTH_VS_WACC at the
# bottom of the discount-rate range, so no tested rate triggers the clamp.
WACC_CLAMP_MARGIN = 1e-4

BISECTION_STEPS = 60
STOP_TOL = 1e-9     # x price: stop bisecting once this close
ACCEPT_TOL = 1e-6   # x price: report converged only within 0.0001%

# The engine's wording for a flagged company (``engine._build_summary``).
_WHOM = {"captive_finance": "a company with a consolidated captive finance arm",
         "lessor": "a debt-funded lessor"}


@dataclass(frozen=True)
class _Driver:
    """How the notes name a driver."""

    noun: str        # "More than one <noun> reproduces ..."
    range_name: str  # "... the solvable <range_name> range ..."
    subject: str     # "...; <subject> cannot be uniquely inferred."
    fmt: str         # format of the range bounds in the notes


_GROWTH = _Driver("growth rate", "growth", "growth", ".0%")
_MARGIN = _Driver("target EBIT margin", "target EBIT margin",
                  "the target EBIT margin", ".0%")
_WACC = _Driver("discount rate", "discount-rate", "the discount rate", ".2%")


def growth_path_from_y1(
    y1: Optional[float], terminal_growth: float, forecast_years: int
) -> Optional[list[float]]:
    """An explicit growth path fading linearly from a year-1 rate ``y1`` to
    ``terminal_growth`` over ``forecast_years`` years; None without a y1 (the
    DCF then derives its own path) or a horizon below one year.

    With a one-year horizon the single forecast year uses y1 itself
    (``utils.fade_path(y1, g, 1)`` returns ``[g]`` by design, which would drop
    y1)."""
    if y1 is None or forecast_years < 1:
        return None
    if forecast_years == 1:
        return [y1]
    return list(fade_path(y1, terminal_growth, forecast_years))


def _grid(lo: float, hi: float, step: float) -> list[float]:
    """lo, lo + step, ... up to hi; hi is appended when the steps miss it."""
    n = int((hi - lo) / step + 1e-9)
    points = [lo + step * i for i in range(n + 1)]
    if hi - points[-1] > 1e-12:
        points.append(hi)
    return points


def _solve(price_of: Callable[[float], Optional[float]], price: float,
           lo: float, hi: float, step: float,
           driver: _Driver = _GROWTH) -> Optional[dict]:
    """Find the lowest driver value in [lo, hi] at which ``price_of`` equals
    ``price`` (see the module docstring for the method).

    Returns ``{"converged": True, "value": x}`` (plus a ``note`` when more
    than one value fits), ``{"converged": False, "value": None, "note": ...}``
    when nothing in the range reproduces the price, or None when the model
    gives no price anywhere in the range or fails while bisecting."""
    grid = _grid(lo, hi, step)
    vals = [price_of(x) for x in grid]
    known = [v for v in vals if v is not None]
    if not known:
        return None
    tol = STOP_TOL * price
    accept = ACCEPT_TOL * price
    several = (f"More than one {driver.noun} reproduces the market price with these "
               "assumptions; showing the lowest.")

    # A flat curve can match the price for every value; that does not
    # identify a market-implied assumption. Exact grid hits, including a
    # tangency with no sign change, are otherwise valid solutions.
    if len(known) == len(grid) and all(abs(v - price) <= tol for v in known):
        return {"converged": False, "value": None,
                "note": f"The model matches the market price throughout the tested "
                f"{driver.range_name} range; {driver.subject} cannot be uniquely inferred."}
    hits = [(x, p) for x, p in zip(grid, vals) if p is not None and abs(p - price) <= tol]
    brackets = []
    for (x_a, p_a), (x_b, p_b) in zip(zip(grid, vals), zip(grid[1:], vals[1:])):
        if p_a is None or p_b is None:
            continue
        if (abs(p_a - price) > tol and abs(p_b - price) > tol
                and (p_a - price) * (p_b - price) < 0):
            brackets.append((x_a, p_a, x_b))
    crossings = len(hits) + len(brackets)
    bracket = brackets[0] if brackets else None
    if hits and (bracket is None or hits[0][0] < bracket[0]):
        out = {"converged": True, "value": hits[0][0]}
        if crossings > 1:
            out["note"] = several
        return out
    if bracket is None:
        return {"converged": False, "value": None,
                "note": f"Market price is outside the solvable {driver.range_name} range "
                f"({lo:{driver.fmt}} to {hi:{driver.fmt}}) with the current assumptions: the "
                f"model's implied price only spans {min(known):,.2f} to "
                f"{max(known):,.2f} across that range, vs {price:,.2f}."}

    a, p_a, b = bracket
    mid, p_mid = a, p_a
    for _ in range(BISECTION_STEPS):
        if abs(p_a - price) <= tol:
            mid, p_mid = a, p_a
            break
        mid = (a + b) / 2.0
        p_mid = price_of(mid)
        if p_mid is None:
            return None
        if abs(p_mid - price) <= tol:
            break
        if (p_mid - price) * (p_a - price) > 0:
            a, p_a = mid, p_mid
        else:
            b = mid
    if abs(p_mid - price) > accept:
        return {"converged": False, "value": None,
                "note": "The implied price jumps across the market price instead of "
                f"passing through it; no {driver.noun} reproduces it."}
    out = {"converged": True, "value": mid}
    if crossings > 1:
        out["note"] = several
    return out


def _price(result) -> Optional[float]:
    p = getattr(result, "implied_price", None)
    return p if is_num(p) else None


def _growth_leg(company, macro, dcf_assumptions: DCFAssumptions, price: float,
                current_path: Optional[list]) -> Optional[dict]:
    """Year-1 revenue growth that reproduces ``price`` (the top-level keys).
    None when the DCF cannot be re-run anywhere in the range."""
    def growth_path(g1: float) -> list[float]:
        if current_path:
            n = len(current_path)
            delta = g1 - current_path[0]
            return [g + delta * (1 - i / (n - 1) if n > 1 else 1)
                    for i, g in enumerate(current_path)]
        return growth_path_from_y1(g1, dcf_assumptions.terminal_growth,
                                   dcf_assumptions.forecast_years)

    def price_of(g1: float) -> Optional[float]:
        a = replace(dcf_assumptions, revenue_growth=growth_path(g1))
        try:
            # Four positional arguments: tests stand in a 4-argument run_dcf.
            return _price(_dcf.run_dcf(company, macro, a, price))
        except Exception:  # noqa: BLE001
            return None

    lo, hi, step = GROWTH_RANGE
    solved = _solve(price_of, price, lo, hi, step, _GROWTH)
    if solved is None:
        return None
    out = {"converged": solved["converged"], "implied_growth_y1": solved["value"]}
    if solved["converged"]:
        out["implied_revenue_growth"] = growth_path(solved["value"])
    if solved.get("note"):
        out["note"] = solved["note"]
    return out


def _margin_leg(company, macro, dcf_assumptions: DCFAssumptions, price: float,
                resolved: dict) -> Optional[dict]:
    """Target EBIT margin that reproduces ``price``, growth path held."""
    path = list(resolved["revenue_growth_path"])

    def price_of(m: float) -> Optional[float]:
        a = replace(dcf_assumptions, revenue_growth=list(path), target_ebit_margin=m)
        try:
            return _price(_dcf.run_dcf(company, macro, a, price))
        except Exception:  # noqa: BLE001
            return None

    lo, hi, step = MARGIN_RANGE
    solved = _solve(price_of, price, lo, hi, step, _MARGIN)
    if solved is None:
        return None
    return {
        "converged": solved["converged"],
        "implied_target_ebit_margin": solved["value"],
        "current_target_ebit_margin": resolved.get("target_ebit_margin"),
        "start_ebit_margin": resolved.get("start_ebit_margin"),
        "note": solved.get("note"),
    }


def _wacc_leg(company, macro, dcf_assumptions: DCFAssumptions, price: float,
              resolved: dict) -> Optional[dict]:
    """Discount rate that reproduces ``price``, growth path, target margin and
    terminal growth held at the resolved values. The terminal growth held is
    the rate the forward run used: when it was clamped below the input (too
    close to the computed WACC), the clamped rate, so the leg does not swap in
    the higher input rate as it moves the WACC."""
    current = resolved.get("wacc")
    g = next((x for x in (resolved.get("terminal_growth_used"), resolved.get("terminal_growth"),
                          dcf_assumptions.terminal_growth) if is_num(x)), None)
    if resolved.get("terminal_method") == "exit_multiple":
        lo = EXIT_MULTIPLE_WACC_MIN
    else:
        lo = g + config.MAX_TERMINAL_GROWTH_VS_WACC + WACC_CLAMP_MARGIN
    # A negative terminal growth would put the bottom of the range at or below
    # zero, where run_dcf ignores the override.
    lo = max(lo, WACC_STEP)
    if lo >= WACC_MAX:
        return {"converged": False, "implied_wacc": None, "current_wacc": current,
                "note": f"No discount rate to test: terminal growth leaves no rate below "
                f"{WACC_MAX:.0%} that keeps WACC above it by "
                f"{config.MAX_TERMINAL_GROWTH_VS_WACC:.0%}."}
    held = replace(dcf_assumptions, revenue_growth=list(resolved["revenue_growth_path"]),
                   target_ebit_margin=resolved.get("target_ebit_margin"))
    if is_num(g):
        held = replace(held, terminal_growth=g)

    def price_of(w: float) -> Optional[float]:
        try:
            return _price(_dcf.run_dcf(company, macro, held, price, wacc_override=w))
        except Exception:  # noqa: BLE001
            return None

    solved = _solve(price_of, price, lo, WACC_MAX, WACC_STEP, _WACC)
    if solved is None:
        return None
    return {
        "converged": solved["converged"],
        "implied_wacc": solved["value"],
        "current_wacc": current,
        "note": solved.get("note"),
    }


def _guarded(leg, *args) -> Optional[dict]:
    try:
        return leg(*args)
    except Exception:  # noqa: BLE001 - diagnostics only, never block valuation
        return None


def solve_reverse_dcf(company, macro, dcf_assumptions: Optional[DCFAssumptions],
                      price, resolved: Optional[dict], *,
                      include_reference_only: bool = False) -> Optional[dict]:
    """Solve the market price back into year-1 revenue growth, the target EBIT
    margin and the discount rate, one at a time (see the module docstring).

    ``resolved`` is the forward run's ``DCFResult.assumptions``; ``price`` is
    the market price per share. For a flagged company (bank, insurer, REIT,
    lender, captive-finance group, debt-funded lessor) nothing is solved and
    the note says the FCFF DCF is reference-only, unless
    ``include_reference_only`` is True: the solve then runs anyway, and the
    caller must label the result as an inversion of a reference-only DCF
    rather than what the price implies (the dashboard API does). Returns None
    when the DCF gave no valuation (``resolved['valuation_available']`` is
    False) or nothing could be solved; otherwise a dict with

      * ``current_assumption_y1``: the model's year-1 growth;
      * ``converged``, ``implied_growth_y1`` and, when converged,
        ``implied_revenue_growth`` (the shifted path);
      * ``note`` when there is something to say (no solution, several
        solutions, no valid price, or a reference-only DCF);
      * ``margin``: ``{converged, implied_target_ebit_margin,
        current_target_ebit_margin, start_ebit_margin, note}`` or None;
      * ``wacc``: ``{converged, implied_wacc, current_wacc, note}`` or None.

    The margin and discount-rate legs need a valid price and the resolved
    growth path; each leg is None on its own when it cannot be computed.
    Never raises.
    """
    try:
        resolved = resolved or {}
        if resolved.get("valuation_available") is False:
            return None
        dcf_assumptions = dcf_assumptions or DCFAssumptions()
        current_path = resolved.get("revenue_growth_path") or dcf_assumptions.revenue_growth
        base = {"current_assumption_y1": current_path[0] if current_path else None}

        detail = (financial_institution_detail(company)
                  if company is not None and not include_reference_only else None)
        if detail:
            whom = _WHOM.get(detail[1], "a financial institution")
            return {**base, "converged": False, "implied_growth_y1": None,
                    "note": f"Not meaningful for {whom}: the FCFF DCF is reference-only.",
                    "margin": None, "wacc": None}

        if not is_num(price) or price <= 0:
            return {**base, "converged": False, "implied_growth_y1": None,
                    "note": "No valid market price to solve against.",
                    "margin": None, "wacc": None}

        growth = _guarded(_growth_leg, company, macro, dcf_assumptions, price, current_path)
        margin = wacc = None
        if resolved.get("revenue_growth_path"):
            margin = _guarded(_margin_leg, company, macro, dcf_assumptions, price, resolved)
            wacc = _guarded(_wacc_leg, company, macro, dcf_assumptions, price, resolved)
        if growth is None:
            if margin is None and wacc is None:
                return None
            growth = {"converged": False, "implied_growth_y1": None,
                      "note": "The DCF could not be re-run across the growth range."}
        return {**base, **growth, "margin": margin, "wacc": wacc}
    except Exception:  # noqa: BLE001 - diagnostics only, never block valuation
        return None
