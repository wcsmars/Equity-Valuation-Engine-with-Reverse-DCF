"""Weighted-average cost of capital (WACC) and its building blocks.

This module derives the discount rate used by the unlevered FCFF DCF:

  * `effective_tax_rate` — a robust, history-based effective tax rate.
  * `compute_wacc`       — CAPM cost of equity + a cost of debt, blended on
                           market-value weights.

Everything here is pure-Python (stdlib + the package's own utils). All rates are
decimals (8% -> 0.08) and all monetary inputs are absolute units (not millions),
matching the conventions in ``schemas.py``.
"""

from __future__ import annotations

import re
from typing import Optional

from ..schemas import AnnualFinancials, CompanyData, MacroAssumptions, WACCResult
from ..utils import is_num, median, safe_div
from .. import config


# --------------------------------------------------------------------------- #
#  Effective tax rate
# --------------------------------------------------------------------------- #
# A historical median needs this many clean years (positive pre-tax income and
# a 0-100% tax charge). One or two years of a young issuer are typically NOL-
# shielded or carry a valuation-allowance release, and would be applied to
# every forecast year and the terminal value.
MIN_TAX_HISTORY_YEARS = 3

# A cost of debt derived as interest / debt is floored at rf + this spread: no
# issuer borrows below the risk-free rate at the margin. The ratio reads low
# when old low-coupon debt dominates, or when the debt balance postdates the
# interest it is divided by (KD: 1.56% after a debt raise late in the year,
# which lifted the debt weight and cut WACC to 4.6%). 0.5% is about the
# tightest investment-grade spread.
MIN_DEBT_SPREAD = 0.005


def effective_tax_rate(fin: AnnualFinancials, fallback: float) -> float:
    """Median historical effective tax rate (tax_expense / pretax_income).

    See ``effective_tax_rate_detail`` for the rules; this returns the rate only.
    """
    return effective_tax_rate_detail(fin, fallback)[0]


def effective_tax_rate_detail(
    fin: AnnualFinancials, fallback: float
) -> tuple[float, str, Optional[str]]:
    """(rate, source, note): median historical effective tax rate, or ``fallback``.

    A year is used only when pre-tax income is *positive* (a negative or zero
    EBT makes the ratio meaningless) and the tax charge is 0-100% of it: a tax
    benefit on a profit (valuation-allowance release) or a charge above the
    profit is a one-off, not a rate. The median of the remaining years is
    clamped to ``[config.MIN_EFFECTIVE_TAX_RATE, config.MAX_EFFECTIVE_TAX_RATE]``
    and the source is ``"effective (historical)"``.

    With fewer than ``MIN_TAX_HISTORY_YEARS`` such years (or no financials) the
    provided ``fallback`` is returned unchanged with source ``"marginal
    (fallback)"``, plus a note explaining why when some profitable year existed
    (the note is None otherwise).
    """
    # Guard against a malformed/empty financials object.
    if fin is None:
        return fallback, "marginal (fallback)", None

    pretax = getattr(fin, "pretax_income", None) or []
    taxes = getattr(fin, "tax_expense", None) or []

    rates: list[float] = []
    profitable = 0
    # Pair up tax/pretax year-by-year; only keep positive-EBT years.
    for tax, ebt in zip(taxes, pretax):
        if not is_num(tax) or not is_num(ebt) or ebt <= 0:
            continue
        profitable += 1
        ratio = safe_div(tax, ebt)
        if ratio is not None and 0.0 <= ratio <= 1.0:
            rates.append(ratio)

    if len(rates) < MIN_TAX_HISTORY_YEARS:
        # Thin or distorted history -- defer to the caller's fallback (often the
        # marginal rate, config.DEFAULT_MARGINAL_TAX_RATE).
        note = None
        if profitable:
            note = (f"effective tax rate: only {len(rates)} clean year(s) among {profitable} "
                    f"with positive pretax income (need {MIN_TAX_HISTORY_YEARS}; tax benefits "
                    f"and charges above pretax income are left out); using {fallback:.1%}")
        return fallback, "marginal (fallback)", note

    # Clamp into the sane band defined in config.
    lo, hi = config.MIN_EFFECTIVE_TAX_RATE, config.MAX_EFFECTIVE_TAX_RATE
    return max(lo, min(hi, median(rates))), "effective (historical)", None


# --------------------------------------------------------------------------- #
#  Beta provenance
# --------------------------------------------------------------------------- #
# The market data layer adjusts Yahoo's raw regression beta toward 1 (Blume:
# 0.67 x raw + 0.33), keeps the raw figure in ``MarketData.raw_beta`` and says
# so in a source note; the models use the beta as given and only report it.
_RAW_BETA_ATTRS = ("raw_beta", "beta_raw")
# The raw value in such a note: "raw beta 2.217", or "beta 1.835 Blume-adjusted".
_RAW_BETA_IN_NOTE = re.compile(
    r"raw(?: [a-z]+)?(?: beta)?(?: of| is|:|=)? *(-?\d+(?:\.\d+)?)"
    r"|beta (-?\d+(?:\.\d+)?) blume-adjusted", re.IGNORECASE)


def beta_adjustment(company) -> Optional[dict]:
    """``{"raw": raw beta or None, "blume": bool, "note": source note or None}``
    when the market beta was adjusted by the data layer, else None.

    Recognised from the raw beta the provider kept on ``company.market``
    (``raw_beta``, when it differs from ``beta``) or from a source note that
    mentions Blume or an adjusted beta (the raw value is then read from the
    note when it gives one, e.g. "raw beta 2.217").
    """
    market = getattr(company, "market", None)
    beta = getattr(market, "beta", None)
    if not is_num(beta):
        return None
    note = None
    for text in (str(n) for n in getattr(company, "source_notes", None) or []):
        low = text.lower()
        if "beta" not in low or not ("blume" in low or "adjust" in low):
            continue
        if "not usable" in low or "default beta" in low:
            continue  # a dropped beta: the models use DEFAULT_BETA and say so
        note = text
        break
    blume = note is not None and "blume" in note.lower()
    for attr in _RAW_BETA_ATTRS:
        raw = getattr(market, attr, None)
        if is_num(raw) and abs(raw - beta) > 1e-9:
            return {"raw": float(raw), "blume": blume, "note": note}
    if note is None:
        return None
    match = _RAW_BETA_IN_NOTE.search(note)
    raw = float(match.group(1) or match.group(2)) if match else None
    return {"raw": raw if (raw is not None and abs(raw - beta) > 1e-9) else None,
            "blume": blume, "note": note}


# --------------------------------------------------------------------------- #
#  WACC
# --------------------------------------------------------------------------- #
def compute_wacc(company: CompanyData, macro: MacroAssumptions) -> WACCResult:
    """Compute WACC for ``company`` under the macro assumptions.

    Cost of equity (CAPM):   ke = rf + beta * ERP
        beta comes from market data if available, else ``config.DEFAULT_BETA``.
        When the data layer adjusted it (``beta_adjustment``), a note gives the
        raw beta and its cost of equity; ``detail`` records ``beta_source`` and
        ``beta_raw``. No adjustment is made here.

    Pre-tax cost of debt (first sane source wins):
        1. ``macro.pretax_cost_of_debt`` if explicitly supplied.
        2. interest_expense(latest) / total_debt — but only accepted if it lands
           in the plausible band [0.01, 0.15] (filters out distorted readings
           when debt is tiny or interest is mismatched), and floored at
           ``rf + MIN_DEBT_SPREAD`` (with a note).
        3. ``rf + config.DEFAULT_CREDIT_SPREAD`` as the last-resort proxy.

    After-tax cost of debt:  kd_at = kd_pretax * (1 - tax).

    Weights use MARKET values: E = market_cap, D = total_debt (book proxy).
        w_e = E / (E + D),  w_d = D / (E + D).
        A missing/zero/negative market cap is the providers' "unknown" sentinel,
        so E falls back to price x market shares_outstanding, then price x the
        latest positive diluted_shares. If E is still unknown -> all-equity
        (w_e = 1, w_d = 0), never all-debt.

    WACC = w_e * ke + w_d * kd_at.

    Never raises: missing fields degrade to documented defaults and every input
    is recorded in ``WACCResult.detail`` (including a ``notes`` list).
    """
    notes: list[str] = []

    # --- macro inputs (guard each) ----------------------------------------- #
    rf = macro.risk_free_rate if (macro is not None and is_num(macro.risk_free_rate)) \
        else config.DEFAULT_RISK_FREE_RATE
    erp = macro.equity_risk_premium if (macro is not None and is_num(macro.equity_risk_premium)) \
        else config.DEFAULT_EQUITY_RISK_PREMIUM

    market = getattr(company, "market", None)
    fin = getattr(company, "financials", None)
    bs = getattr(company, "balance_sheet", None)

    # --- beta -------------------------------------------------------------- #
    beta = getattr(market, "beta", None) if market is not None else None
    adjusted = None
    if not is_num(beta):
        beta = config.DEFAULT_BETA
        beta_source = "DEFAULT_BETA"
        notes.append(f"beta unavailable; using DEFAULT_BETA={config.DEFAULT_BETA}")
    else:
        adjusted = beta_adjustment(company)
        beta_source = "market data (adjusted)" if adjusted else "market data"

    # --- cost of equity (CAPM) --------------------------------------------- #
    cost_of_equity = rf + beta * erp
    if adjusted:
        # The data layer adjusted the beta (and noted it); say what it does to
        # the discount rate, since the DDM and FCFE use the same beta.
        how = "Blume-adjusted toward 1" if adjusted["blume"] else "adjusted by the market data"
        if adjusted["raw"] is not None:
            raw = adjusted["raw"]
            notes.append(
                f"beta {beta:.3f} is {how} from the raw beta {raw:.3f}: cost of equity "
                f"{cost_of_equity:.2%} (raw beta: {rf + raw * erp:.2%})")
        else:
            notes.append(f"beta {beta:.3f} is {how} (see the data notes): cost of equity "
                         f"{cost_of_equity:.2%}")

    # --- effective tax rate ------------------------------------------------ #
    # Prefer an explicit macro tax rate; otherwise derive from history with the
    # statutory marginal rate as the ultimate fallback.
    if macro is not None and is_num(macro.tax_rate):
        tax = macro.tax_rate
        tax_source = "macro.tax_rate"
    else:
        # (A thin-history fallback note is recorded by the DCF, which applies
        # the same rate to NOPAT; tax_source records it here.)
        tax, tax_source, _ = effective_tax_rate_detail(fin, config.DEFAULT_MARGINAL_TAX_RATE)

    # --- pre-tax cost of debt ---------------------------------------------- #
    total_debt = getattr(bs, "total_debt", None) if bs is not None else None
    if not is_num(total_debt) or total_debt < 0:
        total_debt = 0.0

    interest_latest = None
    if fin is not None:
        ints = getattr(fin, "interest_expense", None) or []
        if ints and is_num(ints[-1]):
            interest_latest = abs(ints[-1])  # stored positive, but be safe

    if macro is not None and is_num(macro.pretax_cost_of_debt):
        kd_pretax = macro.pretax_cost_of_debt
        kd_source = "macro.pretax_cost_of_debt"
    else:
        derived = safe_div(interest_latest, total_debt)
        floor = rf + MIN_DEBT_SPREAD
        if derived is not None and 0.01 <= derived <= 0.15 and derived < floor:
            kd_pretax = floor
            kd_source = "interest_expense/total_debt, floored at rf + MIN_DEBT_SPREAD"
            notes.append(
                f"derived cost of debt {derived:.4f} below rf + {MIN_DEBT_SPREAD:.3f}; "
                f"floored at {floor:.4f}"
            )
        elif derived is not None and 0.01 <= derived <= 0.15:
            kd_pretax = derived
            kd_source = "interest_expense/total_debt"
        else:
            kd_pretax = rf + config.DEFAULT_CREDIT_SPREAD
            kd_source = "rf + DEFAULT_CREDIT_SPREAD"
            if derived is not None:
                notes.append(
                    f"derived cost of debt {derived:.4f} outside [0.01,0.15]; "
                    f"using rf+spread"
                )
            elif total_debt > 0:
                # (With no debt the cost of debt carries zero weight: nothing to flag.)
                notes.append("cost of debt not derivable; using rf+spread")

    after_tax_cost_of_debt = kd_pretax * (1.0 - tax)

    # --- market-value weights ---------------------------------------------- #
    # A market cap of 0 is what the market client reports when Yahoo's quote
    # summary fails (price * 0 shares), so treat <= 0 as unknown, not as E = 0
    # (which would put 100% weight on debt and collapse WACC to kd_at).
    equity_value = getattr(market, "market_cap", None) if market is not None else None
    if not is_num(equity_value) or equity_value <= 0:
        equity_value = 0.0
        price = getattr(market, "price", None) if market is not None else None
        shares = getattr(market, "shares_outstanding", None) if market is not None else None
        diluted = [s for s in (getattr(fin, "diluted_shares", None) or []) if is_num(s) and s > 0] \
            if fin is not None else []
        if is_num(price) and price > 0:
            if is_num(shares) and shares > 0:
                equity_value = price * shares
                notes.append("market_cap unavailable; using price x shares_outstanding")
            elif diluted:
                equity_value = price * diluted[-1]
                notes.append("market_cap unavailable; using price x latest diluted_shares")

    total_cap = equity_value + total_debt
    if equity_value <= 0:
        # No usable equity value (or no capital structure at all) -> assume all
        # equity rather than letting the debt weight absorb 100%.
        weight_equity = 1.0
        weight_debt = 0.0
        notes.append("equity market value unavailable; defaulting to all-equity weights")
    else:
        weight_equity = equity_value / total_cap
        weight_debt = total_debt / total_cap

    # --- blend ------------------------------------------------------------- #
    wacc = weight_equity * cost_of_equity + weight_debt * after_tax_cost_of_debt

    detail = {
        "risk_free_rate": rf,
        "equity_risk_premium": erp,
        "beta": beta,
        "beta_source": beta_source,
        "beta_raw": adjusted["raw"] if adjusted else None,
        "cost_of_equity": cost_of_equity,
        "tax_rate": tax,
        "tax_source": tax_source,
        "pretax_cost_of_debt": kd_pretax,
        "cost_of_debt_source": kd_source,
        "after_tax_cost_of_debt": after_tax_cost_of_debt,
        "equity_value": equity_value,
        "total_debt": total_debt,
        "weight_equity": weight_equity,
        "weight_debt": weight_debt,
        "interest_expense_latest": interest_latest,
        "wacc": wacc,
        "notes": notes,
    }

    return WACCResult(
        cost_of_equity=cost_of_equity,
        after_tax_cost_of_debt=after_tax_cost_of_debt,
        weight_equity=weight_equity,
        weight_debt=weight_debt,
        wacc=wacc,
        beta=beta,
        detail=detail,
    )
