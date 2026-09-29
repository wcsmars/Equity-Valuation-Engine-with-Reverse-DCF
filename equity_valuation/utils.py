"""Shared numeric/series helpers used across providers and models.

Pure functions, no I/O. Keep these dependency-light (stdlib + math only) so every
module can import them without pulling in heavy packages.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence


def safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    """Division that returns None on zero/None/NaN inputs instead of raising."""
    if num is None or den is None:
        return None
    try:
        if den == 0 or _isnan(den) or _isnan(num):
            return None
        return num / den
    except (TypeError, ZeroDivisionError):
        return None


def _isnan(x: object) -> bool:
    return isinstance(x, float) and math.isnan(x)


def is_num(x: object) -> bool:
    """True if x is a finite real number."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def clean(values: Sequence[Optional[float]]) -> list[float]:
    """Drop None/NaN/inf from a sequence, returning a list of finite floats."""
    return [float(v) for v in values if is_num(v)]


def cagr(first: Optional[float], last: Optional[float], periods: int) -> Optional[float]:
    """Compound annual growth rate between first and last over `periods` years.

    Returns None if inputs are non-positive (sign change makes CAGR meaningless).
    """
    if not is_num(first) or not is_num(last) or periods <= 0:
        return None
    if first <= 0 or last <= 0:
        return None
    return (last / first) ** (1.0 / periods) - 1.0


def series_cagr(values: Sequence[Optional[float]],
                years: Optional[Sequence[Optional[float]]] = None) -> Optional[float]:
    """CAGR between the first and last *positive* entries of an annual series.

    Providers zero-fill missing years, so zero/None/NaN entries are skipped as
    endpoints, but the period count is the real distance between the endpoints
    (fiscal-year difference when ``years`` aligns with ``values``, else the index
    distance) so a gap in the middle is not compressed into fewer years.
    """
    values = list(values or [])
    pts = [(i, v) for i, v in enumerate(values) if is_num(v) and v > 0]
    if len(pts) < 2:
        return None
    (i0, v0), (i1, v1) = pts[0], pts[-1]
    periods = i1 - i0
    years = list(years or [])
    if len(years) == len(values):
        y0, y1 = years[i0], years[i1]
        if is_num(y0) and is_num(y1) and y0 > 0 and y1 > y0:
            periods = int(round(y1 - y0))
    return cagr(v0, v1, periods)


def incremental_ratio(changes: Sequence[Optional[float]],
                      levels: Sequence[Optional[float]]) -> Optional[float]:
    """Pooled ``sum(changes_i) / sum(levels_i - levels_{i-1})`` over usable years.

    Used for dNWC per unit of revenue change. Pooling (a change-weighted mean)
    keeps one near-flat year from dominating, as a mean of per-year ratios would.
    A year counts only when its change is finite and both levels are positive
    (zero-filled years are gaps, not data). Returns None if nothing is usable or
    the pooled level change is zero.
    """
    return screened_incremental_ratio(changes, levels, one_off_share=None)[0]


# A year whose dNWC exceeds this share of that year's revenue AND the year's
# revenue change is read as a one-off (a tax deposit, a litigation settlement)
# booked through operating working capital, not as working-capital intensity.
NWC_ONE_OFF_REVENUE_SHARE = 0.10


def screened_incremental_ratio(
    changes: Sequence[Optional[float]],
    levels: Sequence[Optional[float]],
    one_off_share: Optional[float] = NWC_ONE_OFF_REVENUE_SHARE,
) -> tuple[Optional[float], list[int]]:
    """(pooled incremental ratio, indices of the years left out as one-offs).

    Same pooled estimator as ``incremental_ratio``, but a year is left out when
    |change_i| > ``one_off_share`` x level_i AND |change_i| > |level_i -
    level_{i-1}|. Both tests must hold: a fast grower's large working-capital
    build stays in (it is in proportion to its revenue change), and so does a
    flat-revenue year with an ordinary dNWC (it is small against revenue). Two
    such KO years (a tax deposit, a settled accrual) otherwise turn a ~0 ratio
    into 0.92. ``one_off_share=None`` disables the screen. The ratio is None if
    no year is usable or the pooled level change is zero.
    """
    changes = list(changes or [])
    levels = list(levels or [])
    num = den = 0.0
    used = False
    dropped: list[int] = []
    for i in range(1, min(len(changes), len(levels))):
        cur, prev, chg = levels[i], levels[i - 1], changes[i]
        if not (is_num(chg) and is_num(cur) and is_num(prev) and cur > 0 and prev > 0):
            continue
        if (one_off_share is not None and abs(chg) > one_off_share * cur
                and abs(chg) > abs(cur - prev)):
            dropped.append(i)
            continue
        num += chg
        den += cur - prev
        used = True
    return (safe_div(num, den) if used else None), dropped


def pooled_ratio(numerators: Sequence[Optional[float]],
                 denominators: Sequence[Optional[float]]) -> Optional[float]:
    """Denominator-weighted ``sum(num_i) / sum(den_i)`` over usable years.

    Used for capex and D&A as a share of revenue. A mean of per-year ratios lets
    one tiny-revenue ramp year dominate (RIVN: capex 32.6x revenue in its first
    sales year, 6.8x on average); pooling weights each year by its revenue. A
    year counts when both values are finite, the denominator is positive and the
    numerator is non-zero (providers zero-fill unreported years, so a 0 is a gap,
    not a reading). Returns None if no year is usable.
    """
    num = den = 0.0
    used = False
    for n, d in zip(numerators or [], denominators or []):
        if is_num(n) and is_num(d) and d > 0 and n != 0:
            num += n
            den += d
            used = True
    return safe_div(num, den) if used else None


# Capex above GROWTH_CAPEX_START x D&A is treated as growth-phase build-out,
# phased in linearly up to GROWTH_CAPEX_FULL x D&A (see ``growth_capex_path``),
# so there is no cliff at the threshold, and never lower for more historical
# capex (a running maximum over capex, which also keeps faded capex at or
# above GROWTH_CAPEX_START x D&A). Ordinary capex a little above D&A
# (replacement at inflated prices, modest growth) is left alone.
GROWTH_CAPEX_START = 1.5
GROWTH_CAPEX_FULL = 2.0
# The models note a fade only when terminal capex ends at least this far (as a
# share of revenue) below history. A smaller fade (AMZN 12.1% -> 12.0%, capex
# just past 1.5x D&A) is still applied, but a note would call ordinary capex
# growth-phase.
GROWTH_CAPEX_NOTE_MIN_FADE = 0.005


def material_capex_fade(capex_pct: Optional[float],
                        path: Optional[Sequence[float]]) -> bool:
    """True when ``growth_capex_path`` returned a path whose final year sits at
    least ``GROWTH_CAPEX_NOTE_MIN_FADE`` below the historical ``capex_pct``."""
    if not path or not is_num(capex_pct) or not is_num(path[-1]):
        return False
    return capex_pct - path[-1] >= GROWTH_CAPEX_NOTE_MIN_FADE


def growth_capex_path(
    capex_pct: Optional[float],
    da_pct: Optional[float],
    ref_growth: Optional[float],
    growth_path: Sequence[float],
) -> Optional[list[float]]:
    """Per-year capex as a share of revenue with growth capex faded, or None.

    Historical capex well above D&A funds growth. Holding the historical
    sales-to-capital ratio, the net capex (capex - D&A) a year needs per unit of
    revenue is proportional to g / (1 + g), so net capex is scaled by that
    factor relative to the growth ``ref_growth`` it was spent at, never above
    its historical level. For a historical capex x (as a share of revenue):

        f_t = min(1, s(g_t) / s(ref_growth)),   s(g) = max(g, 0) / (1 + g)
        c_t(x) = D&A + (x - D&A) x (1 - w(x) x (1 - f_t))

    The weight w(x) is 0 at x <= ``GROWTH_CAPEX_START`` x D&A, 1 at x >=
    ``GROWTH_CAPEX_FULL`` x D&A and linear in between. As growth fades to the
    terminal rate, a build-out's capex fades toward the reinvestment that rate
    needs (down to the floor below), instead of carrying growth-phase capex
    into the perpetuity (a negative terminal FCFF for CAVA, RIVN, CRWV).

    c_t alone is not monotonic in x once little growth is left: capex at 2x
    D&A ended below capex at 1.5x, so ABG's DCF rose when a backfill took its
    capex from 1.6x to 2.2x D&A. Year t's capex is therefore the largest c_t(x)
    for any x up to the historical capex, which is continuous and never lower
    for more capex (so more capex never gives a higher DCF), and equals
    c_t(capex) wherever c_t was already rising (UPS, PEP, AMZN). Between the
    two thresholds c_t is concave in x, with its peak at m* x D&A,
    m* = (START + 1 + (FULL - START) / (1 - f_t)) / 2, so that is
    max(c_t(capex), c_t(min(capex, m* x D&A))).

    Since c_t(START x D&A) = START x D&A, the running maximum is also a floor:
    growth capex fades toward what the remaining growth needs, but never below
    ``GROWTH_CAPEX_START`` x D&A (MSFT: capex 19.2% of revenue, 2.6x D&A, ends
    at 11.0%, 1.5x its 7.3% D&A). At the terminal rate that is more
    reinvestment than growth needs: in steady state at 2.5% nominal growth,
    capex of 1.5x D&A implies asset lives of 35-40 years, so the terminal free
    cash flow is conservative for short-lived build-outs (servers,
    restaurants). Returns None (hold capex flat) when any input is missing,
    D&A or the reference growth is not positive, or capex is at or below the
    start threshold.
    """
    if not (is_num(capex_pct) and is_num(da_pct) and is_num(ref_growth)):
        return None
    if da_pct <= 0 or ref_growth <= 0 or capex_pct <= GROWTH_CAPEX_START * da_pct:
        return None
    span = GROWTH_CAPEX_FULL - GROWTH_CAPEX_START

    def phased(x: float, kept: float) -> float:
        # c_t(x); ``kept`` is f_t, the share of net capex this year's growth needs.
        weight = min(1.0, max(0.0, (x / da_pct - GROWTH_CAPEX_START) / span))
        return da_pct + (x - da_pct) * (1.0 - weight * (1.0 - kept))

    ref = ref_growth / (1.0 + ref_growth)
    path: list[float] = []
    for g in growth_path:
        share = max(g, 0.0) / (1.0 + g) if (is_num(g) and g > -1.0) else 0.0
        kept = min(1.0, share / ref)
        peak = GROWTH_CAPEX_FULL
        if kept < 1.0:
            peak = min(peak, (GROWTH_CAPEX_START + 1.0 + span / (1.0 - kept)) / 2.0)
        path.append(max(phased(capex_pct, kept), phased(min(capex_pct, peak * da_pct), kept)))
    return path


def fiscal_year_labels(years: Optional[Sequence], indices: Sequence[int]) -> str:
    """'FY2024, FY2025' for positions in an annual series (by position if the
    fiscal years are missing or misaligned)."""
    years = list(years or [])
    out = []
    for i in indices:
        y = years[i] if i < len(years) else None
        out.append(f"FY{int(y)}" if is_num(y) else f"year {i + 1}")
    return ", ".join(out)


# Starting-margin screen for one-off spikes (see ``robust_latest_margin``).
MARGIN_SPIKE_WINDOW = 5          # latest year plus up to four prior years
MARGIN_SPIKE_MIN_GAP = 0.05      # deviation must exceed 5 percentage points ...
MARGIN_SPIKE_REL_GAP = 0.25      # ... and 25% of the prior median's size
MARGIN_SPIKE_MAX_REVENUE_MOVE = 0.25  # latest revenue within +/-25% of the prior year


def robust_latest_margin(
    numerators: Sequence[Optional[float]],
    denominators: Sequence[Optional[float]],
) -> tuple[Optional[float], Optional[dict]]:
    """(margin to start a projection from, spike detail or None).

    The latest margin (numerator / denominator of the newest usable year) is
    returned unless it looks like a one-off spike, in which case the median
    margin of the recent window (latest year included) is returned instead and
    the detail dict records ``latest``, ``median``, ``prior_median`` and
    ``years`` (the window length).

    Years count when the denominator is positive and the numerator is finite
    and non-zero (a zero-filled year is a gap). The latest year is a spike when
    all of these hold, with gap = max(MARGIN_SPIKE_MIN_GAP, MARGIN_SPIKE_REL_GAP
    x |median of the prior years in the window|):
      * at least two prior years exist in the window;
      * it has the same sign as that prior median (a company turning
        profitable, or into losses, has changed regime; a median across the
        change would describe neither state);
      * it differs from that prior median by more than the gap;
      * it differs from the year before it by more than the gap (a gradual
        drift is not a spike);
      * the year before did not already move the same way by at least half the
        gap (a margin that keeps expanding is a trend, not a one-off);
      * the denominator moved by at most ``MARGIN_SPIKE_MAX_REVENUE_MOVE`` from
        the year before, so the jump comes from the numerator (a one-off item),
        not from a revenue step such as an acquisition or a revenue-tag change,
        after which the latest margin is the relevant one.
    A one-off gain or charge that lands after an ordinary or opposite year
    (SOLV: 20.8%, 20.6%, 12.6%, then 26.2% with a divestiture gain; JNJ 2025
    with a litigation-reserve reversal) is caught; a lasting step change after
    a weak year looks the same, so the choice is reported and a target margin
    can still be set explicitly.
    """
    pairs = [(n, d) for n, d in zip(numerators or [], denominators or [])
             if is_num(n) and is_num(d) and d > 0 and n != 0][-MARGIN_SPIKE_WINDOW:]
    if not pairs:
        return None, None
    pts = [n / d for n, d in pairs]
    latest = pts[-1]
    prior = pts[:-1]
    if len(prior) < 2:
        return latest, None
    revenue_move = pairs[-1][1] / pairs[-2][1] - 1.0
    if abs(revenue_move) > MARGIN_SPIKE_MAX_REVENUE_MOVE:
        return latest, None
    prior_med = median(prior)
    gap = max(MARGIN_SPIKE_MIN_GAP, MARGIN_SPIKE_REL_GAP * abs(prior_med))
    prev, prev2 = prior[-1], prior[-2]
    jump = latest - prev
    same_sign = latest * prior_med > 0
    trend = (prev - prev2) * jump > 0 and abs(prev - prev2) >= gap / 2.0
    if same_sign and abs(latest - prior_med) > gap and abs(jump) > gap and not trend:
        med = median(pts)
        return med, {"latest": latest, "median": med, "prior_median": prior_med,
                     "years": len(pts)}
    return latest, None


def first_positive_margin(
    numerators: Sequence[Optional[float]],
    denominators: Sequence[Optional[float]],
) -> Optional[dict]:
    """Detail dict if the latest margin is the first positive one after losses.

    ``robust_latest_margin`` keeps such a year on purpose (a company turning
    profitable has changed regime), but a one-off gain can also lift a
    loss-maker into profit for one year. BA FY2025 is an example: a 9.7B
    disposal gain turned a -6.0% EBIT margin into +4.8%, on 34% revenue growth.
    The two cases look the same in the totals, so the caller only notes it.
    It counts when, over the same window and usable years as
    ``robust_latest_margin``, the latest margin is positive, the year before
    is negative and so is the median of the prior years. The dict has
    ``latest`` and ``prior_median``; otherwise the result is None.
    """
    pts = [n / d for n, d in zip(numerators or [], denominators or [])
           if is_num(n) and is_num(d) and d > 0 and n != 0][-MARGIN_SPIKE_WINDOW:]
    if len(pts) < 2 or pts[-1] <= 0 or pts[-2] >= 0:
        return None
    prior_med = median(pts[:-1])
    if prior_med is None or prior_med >= 0:
        return None
    return {"latest": pts[-1], "prior_median": prior_med}


# A loss year after at least this many profitable years is read as a charge
# year (see ``charge_year_margin``).
CHARGE_YEAR_MIN_PRIOR_YEARS = 3


def charge_year_margin(
    numerators: Sequence[Optional[float]],
    denominators: Sequence[Optional[float]],
) -> Optional[dict]:
    """Detail dict if the latest margin is a loss that breaks a profitable run.

    ``robust_latest_margin`` leaves a sign change alone (a median across it
    describes neither state), so a single charge year would be held for the
    whole projection. F FY2025 is an example: impairments turned four
    profitable years (EBIT margin +2.8% to +4.0%) into -4.9% on +1% revenue,
    and a DCF held at -4.9% forever values the company at zero. It counts
    when, over the same window and usable years as ``robust_latest_margin``:
      * the latest margin is negative;
      * the ``CHARGE_YEAR_MIN_PRIOR_YEARS`` years before it are all positive;
      * the denominator moved by at most ``MARGIN_SPIKE_MAX_REVENUE_MOVE`` from
        the year before (a loss that comes with a revenue collapse is not a
        one-off charge).
    The caller keeps the latest margin as the start and fades to ``target``:
    the median of the prior years in the window (positive here), capped at
    the margin of the year just before the loss. The loss may also mark the
    start of a weaker period, and for a business already in decline (30%, 25%,
    3.7%, 0.2%, then a loss) the four-year median, 14.4%, would fade back to a
    level it left years ago; the target is 0.2%. A first profit after losses
    is the opposite case, handled (noted only) by ``first_positive_margin``.
    The dict has ``latest``, ``prior_median``, ``last_profitable`` (the
    margin of the year before the latest), ``target``, ``profitable_years``
    (the run of positive years just before the latest) and ``revenue_move``;
    otherwise the result is None.
    """
    pairs = [(n, d) for n, d in zip(numerators or [], denominators or [])
             if is_num(n) and is_num(d) and d > 0 and n != 0][-MARGIN_SPIKE_WINDOW:]
    if len(pairs) < CHARGE_YEAR_MIN_PRIOR_YEARS + 1:
        return None
    pts = [n / d for n, d in pairs]
    latest, prior = pts[-1], pts[:-1]
    if latest >= 0 or any(p <= 0 for p in prior[-CHARGE_YEAR_MIN_PRIOR_YEARS:]):
        return None
    revenue_move = pairs[-1][1] / pairs[-2][1] - 1.0
    if abs(revenue_move) > MARGIN_SPIKE_MAX_REVENUE_MOVE:
        return None
    prior_med = median(prior)
    if prior_med is None or prior_med <= 0:
        return None
    run = 0
    for p in reversed(prior):
        if p <= 0:
            break
        run += 1
    return {"latest": latest, "prior_median": prior_med, "last_profitable": prior[-1],
            "target": min(prior_med, prior[-1]), "profitable_years": run,
            "revenue_move": revenue_move}


# A positive latest margin below this share of the prior median, after a
# profitable run, is read as a collapse year (see ``collapsed_margin``).
MARGIN_COLLAPSE_SHARE = 1.0 / 3.0


def collapsed_margin(
    numerators: Sequence[Optional[float]],
    denominators: Sequence[Optional[float]],
) -> Optional[dict]:
    """Detail dict if the latest margin is still positive but has collapsed
    against a profitable run, else None.

    ``robust_latest_margin`` only calls a year a spike when it moves by at
    least ``MARGIN_SPIKE_MIN_GAP`` (5 percentage points), so a thin-margin
    business can lose most of its margin in one year and have that trough held
    for the whole projection. JD FY2025 is an example: an EBIT margin of 0.28%
    after 1.75%, 2.67% and 3.41% (a year of heavy investment in a new
    business) on +13% revenue, which gives a DCF below zero. It counts when,
    over the same window and usable years as ``robust_latest_margin``:
      * the latest margin is positive but below ``MARGIN_COLLAPSE_SHARE`` of
        the median of the prior years in the window;
      * the ``CHARGE_YEAR_MIN_PRIOR_YEARS`` years before it are all positive;
      * revenue is stable: the denominator moved by at most
        ``MARGIN_SPIKE_MAX_REVENUE_MOVE`` from one year to the next across the
        whole window (a margin that falls with a revenue collapse is not a
        one-off, and across a revenue step, such as an acquisition or a
        revenue-tag change, the prior margins describe another business);
      * ``robust_latest_margin`` does not already treat it as a spike (the
        projection then starts from the median instead).
    A margin that declines year after year (TSLA: 16.8%, 9.2%, 7.2%, 4.6%)
    stays above the line and is held. The caller keeps the latest margin as
    the start and fades to ``target``, as after a charge year
    (``charge_year_margin``, whose dict shape this shares): the prior median,
    capped at the margin of the year just before the collapse.
    """
    pairs = [(n, d) for n, d in zip(numerators or [], denominators or [])
             if is_num(n) and is_num(d) and d > 0 and n != 0][-MARGIN_SPIKE_WINDOW:]
    if len(pairs) < CHARGE_YEAR_MIN_PRIOR_YEARS + 1:
        return None
    pts = [n / d for n, d in pairs]
    latest, prior = pts[-1], pts[:-1]
    if latest <= 0 or any(p <= 0 for p in prior[-CHARGE_YEAR_MIN_PRIOR_YEARS:]):
        return None
    moves = [b[1] / a[1] - 1.0 for a, b in zip(pairs, pairs[1:])]
    if any(abs(m) > MARGIN_SPIKE_MAX_REVENUE_MOVE for m in moves):
        return None
    revenue_move = moves[-1]
    prior_med = median(prior)
    if prior_med is None or prior_med <= 0 or latest >= MARGIN_COLLAPSE_SHARE * prior_med:
        return None
    if robust_latest_margin(numerators, denominators)[1] is not None:
        return None
    run = 0
    for p in reversed(prior):
        if p <= 0:
            break
        run += 1
    return {"latest": latest, "prior_median": prior_med, "last_profitable": prior[-1],
            "target": min(prior_med, prior[-1]), "profitable_years": run,
            "revenue_move": revenue_move}


# yfinance industries (lower-case prefixes, dashes normalised to " - ") whose
# members are banks, insurers, REITs or mortgage lenders -> kind: an FCFF DCF /
# FCFE does not describe them. A mortgage REIT (AGNC, NLY, STWD) is a lender:
# its repo and warehouse borrowing funds a book of loans and mortgage
# securities, not property, so it is matched before the generic REIT prefix.
_FINANCIAL_INDUSTRY_KINDS = (("banks", "bank"), ("insurance -", "insurer"),
                             ("reit - mortgage", "lender"), ("reit", "reit"),
                             ("mortgage finance", "lender"))
# Industries that mix balance-sheet lenders and BDCs (AXP, COF, SCHW, ARCC) with
# fee businesses the DCF handles (V, MA, BLK, EVR): flagged only when the
# statements look like a lender's.
_LENDER_CHECK_INDUSTRIES = ("credit services", "capital markets", "asset management",
                            "financial conglomerates")
# Latest interest expense at or above this share of revenue marks a lender.
_LENDER_INTEREST_SHARE = 0.10
# Rental and leasing companies whose borrowing funds the fleet they lease out
# (AerCap, Air Lease, GATX: interest 20%+ of revenue) are marked "lessor" by the
# same interest test; equipment renters with ordinary leverage (URI, Ryder, at
# 3-4%) keep their DCF. The data layer marks such filers from their own tags
# too (``_financial_kind == "lessor"``).
_LESSOR_CHECK_INDUSTRIES = ("rental & leasing services",)
# A provider note containing one of these phrases flags a financial filer (the
# EDGAR client's "EDGAR tags mark this company as a bank (...)" warning).
_FINANCIAL_NOTE_MARKERS = ("tags mark this company as", "financial filer",
                           "financial institution")
# Words in such a note that name the kind (the EDGAR client's labels).
_FINANCIAL_NOTE_KINDS = (("bank", "bank"), ("insurer", "insurer"),
                         ("business development", "lender"), ("reit", "reit"))
# The data layer's classification (``AnnualFinancials._financial_kind``) ->
# (label, kind). "captive_finance" marks an industrial filer that consolidates
# a large finance and leasing arm (GM Financial, Ford Credit, Toyota Financial
# Services, John Deere Financial, Cat Financial); it is not a financial
# institution, but its consolidated debt, interest, D&A and capex mix that
# arm's lending book with the industrial business. "lessor" marks a
# debt-funded operating lessor (AerCap: lease income most of revenue, interest
# over 10% of it), whose borrowing funds the fleet it leases out.
_EDGAR_FINANCIAL_KINDS = {"bank": ("bank per its EDGAR tags", "bank"),
                          "insurer": ("insurer per its EDGAR tags", "insurer"),
                          "bdc": ("BDC per its EDGAR tags", "lender"),
                          "reit": ("REIT per its EDGAR tags", "reit"),
                          "captive_finance": ("consolidated captive finance arm",
                                              "captive_finance"),
                          "lessor": ("operating lessor per its filings", "lessor")}
# Words in a leading WARNING source note that mark a captive-finance filer
# (checked before the generic financial-filer markers, whose kind words such as
# "bank" can also appear in such a note).
_CAPTIVE_NOTE_MARKERS = ("captive financ", "captive-financ", "finance arm", "leasing arm")
# The data layer's label in a leading WARNING source note that marks a
# debt-funded lessor ("EDGAR tags mark this company as a debt-funded operating
# lessor"; checked after the captive-finance markers, before the generic
# ones). The label, not the bare word: another WARNING that mentions lessors
# in passing must not reclassify the company.
_LESSOR_NOTE_MARKERS = ("operating lessor",)
# Kinds that are not financial institutions but whose FCFF DCF and FCFE are
# shown for reference only, like a financial's: their DDM is also left out of
# the blend at a low payout (``low_payout_ddm``) and never sets the target on
# its own (``ddm_left_alone``).
REFERENCE_ONLY_KINDS = ("captive_finance", "lessor")
# Kinds (``financial_kind``; None for an unflagged company) whose zero-filled
# EBIT years the DCF rebuilds as pretax income + interest expense. A lessor's
# interest is on the debt that funds its fleet: the data layer derives its
# EBIT the same way for display and the reference models, and an EBIT margin
# left at 0 gives a reference DCF that is only the debt (AerCap: -268 a
# share). Banks, lenders and other flagged kinds keep their gaps: a lender's
# interest is the cost of its lending book, so adding it back would count it
# twice.
EBIT_DERIVED_KINDS = (None, "lessor")
# Kinds whose DCF and FCFE replace a capex history with no usable year by
# capex = D&A (maintenance only) when D&A is above 0: an operating company, a
# captive-finance group or a lessor, whose D&A wears out the plant or fleet
# that capex replaces. Not a bank, insurer, REIT or lender, whose DCF and FCFE
# are reference-only and keep capex at 0: a REIT's D&A is mostly depreciation
# of property it buys outside capex, with little recurring capex (capex = D&A
# took Realty Income's reference DCF from 156 to 52 a share), and a bank's D&A
# says nothing about a reinvestment need the model describes.
MAINTENANCE_CAPEX_KINDS = (None,) + REFERENCE_ONLY_KINDS
# Why an FCFF DCF and an FCFE do not fit each kind (the EDGAR client gives the
# same reasons for its tag-based classification).
FINANCIAL_KIND_REASONS = {
    "bank": "interest on deposits and borrowings is its main operating cost",
    "insurer": "investment income on policyholder funds is part of its operations",
    "lender": "interest on the borrowing that funds its lending is an operating cost",
    "reit": "its growth comes from buying property, which is not in capex",
    "financial": "its business is lending, investing or holding property, which an "
                 "operating-company cash-flow model does not describe",
    "captive_finance": "its consolidated statements include a finance and leasing arm whose "
                       "borrowing funds customer loans and leases, so total debt, interest, "
                       "D&A and capex mix that arm's lending book with the industrial business",
    "lessor": "its borrowing funds the fleet it leases out (aircraft, railcars, vehicles), so "
              "interest is an operating cost, and fleet purchases, often reported outside "
              "capex, are the reinvestment its lease income depends on",
}
# Football-field label suffix for a flagged company's DCF and FCFE bars, which
# are shown but left out of the blended target (the report tables use the same
# words).
NOT_IN_BLEND = " (not in blend)"


def financial_institution_detail(company) -> Optional[tuple[str, str]]:
    """(short reason, kind) if ``company`` is a bank, insurer, REIT or
    lender/BDC, consolidates a captive finance arm or is a debt-funded
    operating lessor, else None.

    ``kind`` is one of "bank", "insurer", "reit", "lender" (mortgage lenders,
    BDCs, credit and capital-markets lenders), "captive_finance" (an
    industrial company with a large consolidated finance and leasing arm),
    "lessor" (a debt-funded operating lessor) or "financial" (flagged, kind
    unknown); ``FINANCIAL_KIND_REASONS`` says why the models do not fit it.
    For these an FCFF DCF and an FCFE do not describe the business: for a
    bank, insurer or lender debt and interest are operating items, a REIT's
    growth comes from buying property, which is outside capex, a captive
    finance arm's funding debt and lease fleet sit in the consolidated
    figures, and a lessor's debt funds the fleet it leases out. The engine
    leaves both out of the blended target, and the DCF rebuilds missing EBIT
    from pretax income + interest only for a lessor (``EBIT_DERIVED_KINDS``).
    Checked in order:
      * the market data's industry: bank, insurance (not brokers), REIT and
        mortgage-finance industries always count (a mortgage REIT as a
        lender, since its borrowing funds loans and mortgage securities);
      * the data layer's classification from the filer's own figures
        (deposits, premiums, fair-value investments, investment property, a
        finance and leasing arm, lease income with heavy interest), carried as
        ``financials._financial_kind``; an unrecognised non-empty kind counts
        as "financial";
      * credit services, capital markets, asset management and financial
        conglomerates (or a bare "Financial Services" sector) count only when
        the latest revenue is missing or interest expense is at least
        ``_LENDER_INTEREST_SHARE`` of it, as for a lender or a BDC, so fee
        businesses (V, MA, BLK) keep their DCF; rental and leasing companies
        count as lessors on the same interest test (AerCap, not URI);
      * a provider source note that flags a captive-finance filer (a leading
        WARNING note naming a captive finance or leasing arm), a lessor (a
        leading WARNING note with the data layer's "operating lessor" label)
        or a financial filer.
    """
    market = getattr(company, "market", None)
    fin = getattr(company, "financials", None)
    industry = str(getattr(market, "industry", None) or "").strip()
    sector = str(getattr(market, "sector", None) or "").strip()
    ind = industry.lower()
    # Yahoo's older industry names use an em dash ("REIT—Mortgage").
    key = " - ".join(part.strip() for part in
                     ind.replace("\u2014", "-").replace("\u2013", "-").split("-"))
    for prefix, kind in _FINANCIAL_INDUSTRY_KINDS:
        if key.startswith(prefix):
            return industry, kind
    edgar_kind = getattr(fin, "_financial_kind", None)
    if edgar_kind in _EDGAR_FINANCIAL_KINDS:
        return _EDGAR_FINANCIAL_KINDS[edgar_kind]
    if isinstance(edgar_kind, str) and edgar_kind.strip():
        return f"{edgar_kind.strip().replace('_', ' ')} per the data provider", "financial"
    if ind in _LENDER_CHECK_INDUSTRIES or (not ind and sector.lower() == "financial services"):
        revenue = (getattr(fin, "revenue", None) or [None])[-1]
        interest = (getattr(fin, "interest_expense", None) or [None])[-1]
        label = industry or sector
        if not is_num(revenue) or revenue <= 0:
            return f"{label}; no operating revenue line", "lender"
        if is_num(interest) and abs(interest) >= _LENDER_INTEREST_SHARE * revenue:
            return f"{label}; interest expense {abs(interest) / revenue:.0%} of revenue", "lender"
    if ind in _LESSOR_CHECK_INDUSTRIES:
        revenue = (getattr(fin, "revenue", None) or [None])[-1]
        interest = (getattr(fin, "interest_expense", None) or [None])[-1]
        if (is_num(revenue) and revenue > 0 and is_num(interest)
                and abs(interest) >= _LENDER_INTEREST_SHARE * revenue):
            return (f"{industry}; interest expense {abs(interest) / revenue:.0%} of revenue",
                    "lessor")
    notes = [str(n) for n in (getattr(company, "source_notes", None) or [])]
    for note in notes:
        text = note.lower()
        if text.startswith("warning") and any(m in text for m in _CAPTIVE_NOTE_MARKERS):
            return _EDGAR_FINANCIAL_KINDS["captive_finance"]
    for note in notes:
        text = note.lower()
        if text.startswith("warning") and any(m in text for m in _LESSOR_NOTE_MARKERS):
            return _EDGAR_FINANCIAL_KINDS["lessor"]
    for note in notes:
        text = note.lower()
        if any(m in text for m in _FINANCIAL_NOTE_MARKERS):
            kind = next((k for word, k in _FINANCIAL_NOTE_KINDS if word in text), "financial")
            return "flagged by the data provider", kind
    return None


def financial_institution(company) -> Optional[str]:
    """Short reason if ``company`` is a bank, insurer, REIT or lender/BDC,
    consolidates a captive finance arm or is a debt-funded operating lessor,
    else None (``financial_institution_detail`` without the kind)."""
    detail = financial_institution_detail(company)
    return detail[0] if detail else None


def financial_kind(company) -> Optional[str]:
    """The kind from ``financial_institution_detail`` ("bank", "insurer",
    "reit", "lender", "captive_finance", "lessor" or "financial"), or None
    for an unflagged company."""
    detail = financial_institution_detail(company)
    return detail[1] if detail else None


# A company with a consolidated captive finance arm (or a debt-funded lessor)
# whose regular dividend is below this share of net income returns most of its
# earnings through buybacks or retains them, which a dividends-only DDM ignores
# (GM 24%, HOG 23%, TM 30%, CAT 34%, DE 35%; PCAR, with its year-end extra
# dividend, 87%). The line sits above DE so the rule does not turn on a
# rounding difference. See ``low_payout_ddm``.
LOW_DDM_PAYOUT_SHARE = 0.40


def ddm_payout(company) -> Optional[float]:
    """Share of the latest net income the DDM capitalises: dividend per share x
    shares (market shares outstanding, else the latest diluted count) / latest
    net income. None when an input is missing or net income is not positive."""
    market = getattr(company, "market", None)
    fin = getattr(company, "financials", None)
    dps = getattr(market, "dividend_per_share", None)
    shares = getattr(market, "shares_outstanding", None)
    if not is_num(shares) or shares <= 0:
        diluted = [s for s in (getattr(fin, "diluted_shares", None) or [])
                   if is_num(s) and s > 0]
        shares = diluted[-1] if diluted else None
    net_income = (getattr(fin, "net_income", None) or [None])[-1]
    if not (is_num(dps) and dps > 0 and is_num(shares) and is_num(net_income)
            and net_income > 0):
        return None
    return dps * shares / net_income


def _reference_only_kind(report) -> Optional[str]:
    """The kind (``REFERENCE_ONLY_KINDS``) of a report whose company has a
    consolidated captive finance arm or is a debt-funded lessor and has a
    DDM, else None."""
    company = getattr(report, "company", None)
    if getattr(report, "ddm", None) is None or company is None:
        return None
    detail = financial_institution_detail(company)
    return detail[1] if detail is not None and detail[1] in REFERENCE_ONLY_KINDS else None


def low_payout_ddm(report) -> Optional[float]:
    """The DDM's payout (``ddm_payout``) when the DDM of a company with a
    consolidated captive finance arm, or of a debt-funded lessor, is shown
    for reference only because its dividend is a low share of net income,
    else None.

    With its DCF and FCFE left out, such a company's blend is comps and the
    DDM, and the median of two is their mean, so the DDM carries half the
    weight; without comps it would be the whole target. The DDM counts
    regular dividends only; below ``LOW_DDM_PAYOUT_SHARE`` of net income the
    rest of the earnings is bought back or retained, so the DDM understates
    the company (GM: comps $52.53, DDM $9.81, blend $31.17 "Overvalued
    -61.6%" at $81.20; without peers, GM $9.81 "-88%", CAT -90%, HOG -67%).
    It is therefore left out whether or not comps give a price. Banks and
    insurers keep their DDM: dividends are their usual valuation basis and
    their payout also reflects capital rules.
    """
    if _reference_only_kind(report) is None:
        return None
    payout = ddm_payout(report.company)
    if payout is None or payout >= LOW_DDM_PAYOUT_SHARE:
        return None
    return payout


def ddm_left_alone(report) -> bool:
    """True when the DDM of a company with a consolidated captive finance arm,
    or of a debt-funded lessor, would be the whole blended target: comps give
    no positive price (no peers) and the payout rule (``low_payout_ddm``) has
    not already left it out.

    Its DCF and FCFE are reference-only, so the DDM would be the only method,
    and it counts regular dividends only (PCAR, at an 87% payout with its
    year-end extra dividend, read "-50%"). The engine then gives no blended
    target and asks for peers; the DDM stays on display.
    """
    if _reference_only_kind(report) is None or low_payout_ddm(report) is not None:
        return False
    comps = getattr(report, "comps", None)
    summary = getattr(comps, "implied_price_summary", None) or {}
    comps_price = summary.get("median") if isinstance(summary, dict) else None
    return not (is_num(comps_price) and comps_price > 0)


def ddm_reference_only(report) -> bool:
    """True when the engine shows the DDM for reference only and leaves it out
    of the blend (``low_payout_ddm`` or ``ddm_left_alone``); the football
    field marks its bar the same way."""
    return low_payout_ddm(report) is not None or ddm_left_alone(report)


def mean(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = clean(values)
    return sum(vals) / len(vals) if vals else None


def median(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = sorted(clean(values))
    n = len(vals)
    if n == 0:
        return None
    mid = n // 2
    return vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def percentile(values: Sequence[Optional[float]], q: float) -> Optional[float]:
    """Linear-interpolation percentile, q in [0, 1]."""
    vals = sorted(clean(values))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    idx = q * (len(vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return vals[lo]
    frac = idx - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def summary_stats(values: Sequence[Optional[float]]) -> dict:
    """{'median','mean','min','max','p25','p75','n'} for a sequence of multiples."""
    vals = clean(values)
    return {
        "n": len(vals),
        "mean": mean(vals),
        "median": median(vals),
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
        "p25": percentile(vals, 0.25),
        "p75": percentile(vals, 0.75),
    }


def net_debt_parts(bs: object) -> tuple[float, list[str]]:
    """(total_debt - cash, notes) from a balance-sheet snapshot, component-wise.

    A missing/non-finite component is treated as 0 on its own (with a note), so
    a missing cash figure no longer discards a known debt balance, and a None
    field cannot raise the way ``BalanceSheetSnapshot.net_debt`` would.
    """
    if bs is None:
        return 0.0, ["balance sheet unavailable; net debt assumed 0"]
    notes: list[str] = []
    debt = getattr(bs, "total_debt", None)
    cash = getattr(bs, "cash_and_investments", None)
    if not is_num(debt):
        debt = 0.0
        notes.append("total debt unavailable; assumed 0 in net debt")
    if not is_num(cash):
        cash = 0.0
        notes.append("cash & investments unavailable; assumed 0 in net debt")
    return float(debt) - float(cash), notes


def trim_outliers(values: Sequence[float], factor: float) -> list[float]:
    """Keep values within [median/factor, median*factor]. factor>1, e.g. 3.0.

    Only trims positive multiples; non-positive values are dropped (a negative P/E
    is meaningless for applying to the target). With fewer than three values there
    is no basis for calling one an outlier (and the arithmetic median of two
    always sits nearer the larger), so they are returned untrimmed.
    """
    pos = [v for v in clean(values) if v > 0]
    if len(pos) < 3:
        return pos
    med = median(pos)
    if med is None or med <= 0:
        return pos
    lo, hi = med / factor, med * factor
    return [v for v in pos if lo <= v <= hi]


def fade_path(start: float, end: float, n: int) -> list[float]:
    """Linearly interpolate from `start` to `end` over n steps (inclusive of end).

    Used to fade growth/margins from a near-term level toward a terminal level.
    Returns n values; the last equals `end`.
    """
    if n <= 0:
        return []
    if n == 1:
        return [end]
    return [start + (end - start) * i / (n - 1) for i in range(n)]
