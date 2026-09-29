"""Turn the engine's `ValuationReport` dataclass graph into a JSON-safe dict
(the contract the frontend binds to), and build a compact text context that
grounds the AI researcher in the currently-loaded model.

The JSON shape mirrors `equity_valuation/schemas.py` field-for-field (so
`company.market.raw_beta`, Yahoo's beta before the Blume adjustment, comes
through as is), with two additions the engine doesn't emit directly:
  * `company.balance_sheet.net_debt` (a dataclass @property asdict drops)
  * `assumptions_used` (echo of the knobs that produced this run)
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Optional

# The no-target rule every output shares (the Excel and HTML reports, the memo
# and deck, and the dashboard's copy in lib/format.ts).
from equity_valuation.report.excel import needs_peers


def _sanitize(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with None so the payload is valid JSON."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def report_to_dict(report, assumptions_used: Optional[dict] = None) -> dict:
    """Serialize a ValuationReport to a JSON-safe dict."""
    d = dataclasses.asdict(report)

    # asdict() drops @property values — re-attach net_debt where present.
    try:
        d["company"]["balance_sheet"]["net_debt"] = report.company.balance_sheet.net_debt
    except Exception:  # noqa: BLE001 - degrade, never block serialization
        pass

    if assumptions_used is not None:
        d["assumptions_used"] = assumptions_used

    return _sanitize(d)


# --------------------------------------------------------------------------- #
#  AI grounding context
# --------------------------------------------------------------------------- #
def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _spct(x: Optional[float]) -> str:
    """A signed percent for an upside: '+128.3%', '-20.7%'."""
    return f"{x * 100:+.1f}%" if _isnum(x) else "n/a"


def _money(x: Optional[float], sym: str = "") -> str:
    return "n/a" if x is None else f"{sym}{x:,.2f}"


def _isnum(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _mult(x: Optional[float]) -> str:
    return f"{x:.1f}x" if _isnum(x) else "n/a"


def _dec(x: Optional[float]) -> str:
    return f"{x:.4g}" if _isnum(x) else "n/a"


def _big(x: Optional[float], sym: str = "") -> str:
    """Compact large-number formatter (e.g. 391.0B)."""
    if x is None:
        return "n/a"
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(x) >= div:
            return f"{sym}{x / div:,.1f}{unit}"
    return f"{sym}{x:,.0f}"


def _beta(beta: Any, raw: Any, source: Any = None) -> str:
    """'1.815 (Blume-adjusted from raw 2.217)' for a beta the market data
    adjusted toward 1, else the beta alone ('n/a' when missing)."""
    if not _isnum(beta):
        return "n/a"
    if source == "DEFAULT_BETA":
        return f"{beta:.3f} (default; no usable market beta)"
    if _isnum(raw) and abs(raw - beta) > 1e-9:
        return f"{beta:.3f} (Blume-adjusted from raw {raw:.3f})"
    return f"{beta:.3f}"


def _blend(s: dict, sym: str, comps: Any = None) -> str:
    """The blended target and upside, or why there is none, in the words the
    reports use: no target (and whether peers would give one), a verdict the
    engine withholds (a target, no upside, "N/A" at a positive price), or no
    upside because there is no current price."""
    target = s.get("blended_target")
    if not _isnum(target):
        if needs_peers(s, comps):
            return "blended target: none (no target; supply peers to add trading comps)"
        return "blended target: n/a"
    upside = s.get("blended_upside")
    if not _isnum(upside):
        price = s.get("current_price")
        if s.get("recommendation") == "N/A" and _isnum(price) and price > 0:
            return f"blended target {_money(target, sym)} (upside withheld: n/a)"
        return f"blended target {_money(target, sym)} (upside n/a: no current price)"
    return f"blended target {_money(target, sym)} ({_spct(upside)} vs price)"


def build_ai_context(report: dict) -> str:
    """A compact (~1k token) snapshot of the loaded valuation, fed to Claude so
    its analysis and assumption suggestions reference the actual current model."""
    s = report.get("summary", {}) or {}
    company = report.get("company", {}) or {}
    market = company.get("market", {}) or {}
    fin = company.get("financials", {}) or {}
    bs = company.get("balance_sheet", {}) or {}
    macro = report.get("macro", {}) or {}
    dcf = report.get("dcf") or {}
    comps = report.get("comps") or {}
    sym = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}.get(
        s.get("currency"), ""
    )

    lines: list[str] = []
    lines.append(
        f"COMPANY: {s.get('name','?')} ({s.get('ticker','?')}) | "
        f"sector={market.get('sector')} industry={market.get('industry')}"
    )
    lines.append(
        f"PRICE: {_money(s.get('current_price'), sym)} {s.get('currency','')} | "
        f"market cap {_big(market.get('market_cap'), sym)} | "
        f"beta {_beta(market.get('beta'), market.get('raw_beta'))} | "
        f"52w {_money(market.get('fifty_two_week_low'), sym)}-{_money(market.get('fifty_two_week_high'), sym)}"
    )
    lines.append(f"VERDICT: {s.get('recommendation','?')} | {_blend(s, sym, comps)}")
    methods = s.get("methods") or {}
    excluded = s.get("excluded_from_blend") or {}
    if methods:
        # Methods left out of the blend (e.g. a bank's or a lessor's DCF) are
        # marked, so they are not quoted as the model's valuation.
        lines.append(
            "METHOD VALUES: "
            + "; ".join(
                f"{k} {_money(v, sym)}"
                + (f" (reference only, not in blend: {excluded[k]})" if k in excluded else "")
                for k, v in methods.items())
        )

    # Latest fundamentals — every value may be None (sanitized NaN), so all
    # arithmetic must be None-safe or one missing EBIT silently destroys the
    # AI's entire model context (the caller swallows exceptions).
    def last(key):
        seq = fin.get(key) or []
        return seq[-1] if seq else None

    def _ratio(num, den):
        if num is None or den is None or not den:
            return None
        try:
            return num / den
        except TypeError:
            return None

    years = fin.get("fiscal_years") or []
    if years:
        rev = fin.get("revenue") or []
        lines.append(
            f"LATEST FY{years[-1]}: revenue {_big(last('revenue'), sym)}, "
            f"EBIT {_big(last('ebit'), sym)}, net income {_big(last('net_income'), sym)}, "
            f"EBIT margin {_pct(_ratio(last('ebit'), last('revenue')))}"
        )
        first, latest = (rev[0] if rev else None), (rev[-1] if rev else None)
        n = len(rev) - 1
        if (
            n >= 1
            and isinstance(first, (int, float))
            and isinstance(latest, (int, float))
            and first > 0
            and latest > 0
        ):
            cagr = (latest / first) ** (1 / n) - 1
            lines.append(f"HISTORICAL REVENUE CAGR ({n}y): {_pct(cagr)}")
    lines.append(
        f"BALANCE SHEET: total debt {_big(bs.get('total_debt'), sym)}, "
        f"cash {_big(bs.get('cash_and_investments'), sym)}, "
        f"net debt {_big(bs.get('net_debt'), sym)}"
    )

    # DCF drivers (the editable assumptions the user is steering)
    da = (dcf.get("assumptions") or {}) if dcf else {}
    rg = da.get("revenue_growth_path") or da.get("revenue_growth")
    rg = rg if isinstance(rg, list) and rg else None
    if dcf:
        wacc = (dcf.get("wacc") or {})
        # The growth the DCF actually used; it clamps the input below WACC.
        g_req, g_used = da.get("terminal_growth"), da.get("terminal_growth_used")
        growth = _pct(g_used if g_used is not None else g_req)
        if g_used is not None and g_req is not None and abs(g_used - g_req) > 1e-12:
            growth += f" (input {_pct(g_req)}, clamped below WACC)"
        wdetail = wacc.get("detail") or {}
        # A DCF left out of the blend (a bank's, a captive-finance group's or a
        # lessor's) is marked as on the METHOD VALUES line, so its upside is
        # not quoted as the model's view.
        ref_only = (f" (reference only, not in blend: {excluded['DCF']})"
                    if "DCF" in excluded else "")
        lines.append(
            f"DCF: WACC {_pct(wacc.get('wacc'))} (ke {_pct(wacc.get('cost_of_equity'))}, "
            f"beta {_beta(wacc.get('beta'), wdetail.get('beta_raw'), wdetail.get('beta_source'))}), "
            f"terminal growth {growth}, "
            f"terminal method {da.get('terminal_method')}, "
            f"forecast years {da.get('forecast_years')}, "
            f"implied {_money(dcf.get('implied_price'), sym)} "
            f"({_spct(dcf.get('upside'))} vs price){ref_only}"
        )
        if rg:
            lines.append(
                "DCF revenue-growth path: " + ", ".join(_pct(g) for g in rg)
            )
        lines.append(
            f"DCF operating drivers: EBIT margin {_pct(da.get('start_ebit_margin'))} "
            f"fading to target {_pct(da.get('target_ebit_margin'))}, "
            f"tax rate used {_pct(da.get('tax_rate'))} ({da.get('tax_source') or 'n/a'}), "
            f"exit EV/EBITDA {_mult(da.get('exit_ev_ebitda'))}"
        )
    # macro.tax_rate is None when the engine derives the rate; report the
    # rate the DCF actually applied instead of "n/a".
    tax_used = da.get("tax_rate")
    if tax_used is None:
        tax_used = macro.get("tax_rate")
    lines.append(
        f"MACRO: risk-free {_pct(macro.get('risk_free_rate'))}, "
        f"ERP {_pct(macro.get('equity_risk_premium'))}, "
        f"tax {_pct(tax_used)}"
    )
    # The exact current value of every AI-suggestable field, in the units a
    # suggestion must use, so `current_value` is read rather than guessed.
    current = {
        "revenue_growth_y1": rg[0] if rg else None,
        "terminal_growth": da.get("terminal_growth"),
        "forecast_years": da.get("forecast_years"),
        "target_ebit_margin": da.get("target_ebit_margin"),
        "tax_rate": tax_used,
        "risk_free_rate": macro.get("risk_free_rate"),
        "equity_risk_premium": macro.get("equity_risk_premium"),
        "exit_ev_ebitda": da.get("exit_ev_ebitda"),
    }
    lines.append(
        "CURRENT ASSUMPTION VALUES (decimals): "
        + ", ".join(f"{k}={_dec(v)}" for k, v in current.items())
    )
    rdcf = report.get("reverse_dcf") or {}
    if rdcf.get("converged") and rdcf.get("implied_growth_y1") is not None:
        lines.append(
            "REVERSE DCF: the market price implies year-1 revenue growth of "
            f"{_pct(rdcf.get('implied_growth_y1'))} (other assumptions held)"
        )

    # Comps snapshot
    if comps:
        target = comps.get("target") or {}
        stats = comps.get("stats") or {}

        def med(m):
            return (stats.get(m) or {}).get("median")

        lines.append(
            "TARGET MULTIPLES: "
            f"P/E {target.get('pe')}, EV/EBITDA {target.get('ev_ebitda')}, "
            f"EV/Sales {target.get('ev_sales')}, P/B {target.get('pb')}"
        )
        lines.append(
            "PEER MEDIANS: "
            f"P/E {med('pe')}, EV/EBITDA {med('ev_ebitda')}, "
            f"EV/Sales {med('ev_sales')}, P/B {med('pb')} "
            f"({len(comps.get('peers') or [])} peers)"
        )

    warnings = report.get("warnings") or []
    if warnings:
        # Data-quality WARNINGs (unconverted currency, no market cap) first, so
        # the cap never drops them in favour of routine fallback notes.
        ordered = [w for w in warnings if str(w).startswith("WARNING")] + [
            w for w in warnings if not str(w).startswith("WARNING")]
        lines.append("MODEL NOTES: " + " | ".join(str(w) for w in ordered[:10]))

    return "\n".join(lines)
