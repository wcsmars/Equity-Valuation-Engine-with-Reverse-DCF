"""Drive the valuation engine from API requests.

Responsibilities:
  * translate the frontend's flat assumption payload into the engine's
    MacroAssumptions / DCFAssumptions / DDMAssumptions objects;
  * cache the expensive CompanyData fetch per ticker so moving an assumption
    slider re-runs the models instantly instead of re-hitting EDGAR/yfinance,
    and share one data provider so the SEC ticker directory is downloaded once
    rather than for every new ticker;
  * return the serialized report plus an echo of the assumptions actually used.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Optional

from equity_valuation import value_company
from equity_valuation.data.base import DataProvider
from equity_valuation.schemas import (
    DCFAssumptions,
    DDMAssumptions,
    MacroAssumptions,
)
from equity_valuation.utils import is_num

from .serialization import report_to_dict

# Engine config defaults (mirrored here so the API can echo resolved values).
from equity_valuation import config as _cfg

_CACHE_TTL = 600.0  # seconds to reuse fetched CompanyData for slider re-runs
_BASE_TTL = 86400.0  # seconds before the shared provider and its SEC directory are rebuilt
_cache_lock = threading.Lock()
_company_cache: dict[str, tuple[float, Any]] = {}  # ticker -> (ts, CompanyData)
_base: Optional[tuple[float, DataProvider]] = None  # (built at, shared provider)


_peer_cache_lock = threading.Lock()
_peer_cache: dict[tuple, tuple[float, list]] = {}  # tickers tuple -> (ts, rows)
_PEER_TTL = 600.0


class _CachedProvider(DataProvider):
    """Serves a pre-fetched CompanyData; delegates live calls to the underlying
    provider, with a short-lived cache for peer comps so assumption-slider
    recomputes don't re-fetch every peer from yfinance each time."""

    def __init__(self, base: DataProvider, company_data):
        self._base = base
        self._cd = company_data

    def get_company_data(self, ticker: str):
        return self._cd

    def get_market_data(self, ticker: str):
        return self._cd.market

    def get_peer_comp_rows(self, tickers):
        key = tuple(sorted(t.upper() for t in tickers))
        now = time.time()
        with _peer_cache_lock:
            hit = _peer_cache.get(key)
            if hit and now - hit[0] < _PEER_TTL:
                return hit[1]
        rows = self._base.get_peer_comp_rows(tickers)
        if rows:  # don't cache an empty/failed fetch
            with _peer_cache_lock:
                _prune(_peer_cache, now, _PEER_TTL)
                _peer_cache[key] = (now, rows)
        return rows

    def suggest_peers(self, ticker: str):
        return self._base.suggest_peers(ticker)


def _prune(cache: dict, now: float, ttl: float) -> None:
    """Drop entries older than `ttl` (the caller holds the cache's lock), so a
    long-running app keeps only recently used tickers."""
    for key in [k for k, v in cache.items() if now - v[0] >= ttl]:
        del cache[key]


def _shared_base(now: float) -> DataProvider:
    """The process-wide HybridProvider, built on first use.

    Its EdgarClient caches the SEC ticker directory (about 220 KB, over a
    second to fetch), so sharing it downloads the directory once instead of for
    every new ticker. It is rebuilt after _BASE_TTL so companies listed since
    then resolve. The clients keep no per-request state, so request threads can
    share them."""
    global _base
    from equity_valuation.data.provider import HybridProvider

    with _cache_lock:
        if _base is None or now - _base[0] >= _BASE_TTL:
            _base = (now, HybridProvider())
        return _base[1]


def _get_provider(ticker: str, refresh: bool = False) -> DataProvider:
    """Return a provider that yields cached CompanyData when fresh."""
    ticker = ticker.strip().upper()
    now = time.time()
    base = _shared_base(now)
    with _cache_lock:
        hit = _company_cache.get(ticker)
        if hit and not refresh and (now - hit[0]) < _CACHE_TTL:
            return _CachedProvider(base, hit[1])

    # Cold path: fetch once, cache, then serve from cache for this run too.
    company_data = base.get_company_data(ticker)  # the slow network call
    with _cache_lock:
        _prune(_company_cache, now, _CACHE_TTL)
        _company_cache[ticker] = (now, company_data)
    return _CachedProvider(base, company_data)


class AssumptionError(ValueError):
    """An assumption in the request payload is unusable (e.g. NaN/Infinity).
    The API maps it to HTTP 400 with this message."""


def _num(key: str, value: Any) -> Optional[float]:
    """Coerce a number or numeric string. Empty fields use the default;
    malformed values fail visibly instead of silently replacing an assumption.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise AssumptionError(f"{key} must be a number, not a boolean.")
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AssumptionError(f"{key} must be a finite number.") from exc
    if not math.isfinite(f):
        raise AssumptionError(f"{key} must be a finite number (got {value!r}).")
    return f


def _f(payload: dict, *keys) -> Optional[float]:
    for k in keys:
        if payload.get(k) is not None:
            return _num(k, payload[k])
    return None


def _flag(payload: dict, key: str, default: bool = True) -> bool:
    """Strict-ish boolean toggle: 'false'/'0'/'no'/'off' (any case) are False."""
    v = payload.get(key, default)
    if isinstance(v, str):
        return v.strip().lower() not in ("false", "0", "no", "off", "")
    return bool(v)


def _revenue_growth_path(
    y1: Optional[float], terminal_growth: float, forecast_years: int
) -> Optional[list[float]]:
    """Build an explicit fading growth path from a near-term (year-1) override
    down to terminal growth, so the AI's 'raise near-term growth' suggestion is
    directly applicable. Falls back to None (engine derives its own) if no y1.

    With a one-year horizon the single forecast year uses y1 itself
    (utils.fade_path(y1, g, 1) returns [g] by design, which would drop y1)."""
    if y1 is None or forecast_years < 1:
        return None
    if forecast_years == 1:
        return [y1]
    from equity_valuation.utils import fade_path

    return list(fade_path(y1, terminal_growth, forecast_years))


def parse_assumptions(payload: dict):
    """payload (all optional) -> (macro, dcf, ddm, peers, toggles, echo)."""
    payload = payload or {}

    rf = _f(payload, "rf", "risk_free_rate")
    erp = _f(payload, "erp", "equity_risk_premium")
    tax = _f(payload, "tax_rate", "tax")
    cod = _f(payload, "cost_of_debt", "pretax_cost_of_debt")
    fy = _f(payload, "forecast_years")
    forecast_years = int(fy) if fy is not None else _cfg.DEFAULT_FORECAST_YEARS
    forecast_years = max(1, min(forecast_years, 15))
    terminal_growth = _f(payload, "terminal_growth")
    if terminal_growth is None:
        terminal_growth = _cfg.DEFAULT_TERMINAL_GROWTH
    terminal_method = payload.get("terminal_method")
    if terminal_method not in ("gordon", "exit_multiple"):
        terminal_method = "gordon"
    exit_ev_ebitda = _f(payload, "exit_ev_ebitda")
    target_ebit_margin = _f(payload, "target_ebit_margin")
    rev_y1 = _f(payload, "revenue_growth_y1")
    rev_path = payload.get("revenue_growth")
    if isinstance(rev_path, list) and rev_path:
        rev_path = [_num("revenue_growth", g) for g in rev_path]
        if any(g is None for g in rev_path):
            raise AssumptionError("revenue_growth must be a list of numbers.")
    else:
        rev_path = _revenue_growth_path(rev_y1, terminal_growth, forecast_years)

    macro = MacroAssumptions(
        risk_free_rate=rf if rf is not None else _cfg.DEFAULT_RISK_FREE_RATE,
        equity_risk_premium=erp
        if erp is not None
        else _cfg.DEFAULT_EQUITY_RISK_PREMIUM,
        tax_rate=tax,
        pretax_cost_of_debt=cod,
    )
    dcf = DCFAssumptions(
        forecast_years=forecast_years,
        revenue_growth=rev_path,
        terminal_growth=terminal_growth,
        terminal_method=terminal_method,
        exit_ev_ebitda=exit_ev_ebitda,
        target_ebit_margin=target_ebit_margin,
        tax_rate=tax,
    )
    ddm = DDMAssumptions(
        forecast_years=forecast_years, terminal_growth=terminal_growth
    )

    peers_raw = payload.get("peers")
    if isinstance(peers_raw, str):
        peers = [p.strip().upper() for p in peers_raw.split(",") if p.strip()]
    elif isinstance(peers_raw, list):
        peers = [str(p).strip().upper() for p in peers_raw if str(p).strip()]
    else:
        peers = None

    toggles = {
        k: _flag(payload, k)
        for k in ("run_dcf", "run_comps", "run_ddm", "run_fcfe", "run_sensitivity")
    }

    echo = {
        "rf": macro.risk_free_rate,
        "erp": macro.equity_risk_premium,
        "tax_rate": tax,
        "cost_of_debt": cod,
        "forecast_years": forecast_years,
        "terminal_growth": terminal_growth,
        "terminal_method": terminal_method,
        "exit_ev_ebitda": exit_ev_ebitda,
        "target_ebit_margin": target_ebit_margin,
        "revenue_growth_y1": rev_y1,
        "revenue_growth": rev_path,
        # Echo as CSV — the frontend's Assumptions.peers is a string.
        "peers": ",".join(peers) if peers else None,
    }
    return macro, dcf, ddm, peers, toggles, echo


def run_valuation_report(ticker: str, payload: Optional[dict] = None):
    """Provider (cached) -> value_company. Returns (ValuationReport, echo).

    Used directly by the export endpoints, which need the engine's dataclass
    (write_excel / write_html bind to it) rather than the serialized dict."""
    payload = payload or {}
    macro, dcf, ddm, peers, toggles, echo = parse_assumptions(payload)
    provider = _get_provider(ticker, refresh=_flag(payload, "refresh", default=False))

    report = value_company(
        ticker,
        provider=provider,
        macro=macro,
        dcf_assumptions=dcf,
        ddm_assumptions=ddm,
        peers=peers,
        **toggles,
    )
    return report, echo


def _reverse_dcf(report, dcf_assumptions, macro) -> Optional[dict]:
    """Solve for the year-1 revenue growth the market price implies, holding
    other assumptions fixed. Adjust the resolved growth path by a first-year
    delta that fades to zero in the final explicit year. A linear base path
    stays linear; a custom schedule keeps its shape and final-year endpoint.

    This is the 'what do I have to believe?' number: if the market-implied
    growth looks heroic vs history, the price embeds optimism — and vice versa.
    Pure-math re-runs of the engine's run_dcf on cached data (fast).

    The implied price is not guaranteed monotone in growth (e.g. a negative
    target margin makes extra revenue destroy value), so the range is scanned
    on a coarse grid and the lowest-growth crossing is bisected. Convergence is
    judged relative to the price, so penny stocks solve as precisely as $500
    stocks."""
    import dataclasses as _dc

    if report.dcf is None:
        return None
    try:
        from equity_valuation.models.dcf import run_dcf as _run_dcf

        company = report.company
        price = report.current_price
        resolved = getattr(report.dcf, "assumptions", {}) or {}
        if resolved.get("valuation_available") is False:
            return None
        current_path = resolved.get("revenue_growth_path") or dcf_assumptions.revenue_growth
        base = {"current_assumption_y1": current_path[0] if current_path else None}
        if not is_num(price) or price <= 0:
            return {
                **base,
                "converged": False,
                "implied_growth_y1": None,
                "note": "No valid market price to solve against.",
            }

        def growth_path(g1: float) -> list[float]:
            if current_path:
                n = len(current_path)
                delta = g1 - current_path[0]
                return [g + delta * (1 - i / (n - 1) if n > 1 else 1)
                        for i, g in enumerate(current_path)]
            return _revenue_growth_path(
                g1, dcf_assumptions.terminal_growth, dcf_assumptions.forecast_years)

        def implied(g1: float) -> Optional[float]:
            path = growth_path(g1)
            a = _dc.replace(dcf_assumptions, revenue_growth=path)
            try:
                p = _run_dcf(company, macro, a, price).implied_price
            except Exception:  # noqa: BLE001
                return None
            return p if is_num(p) else None

        lo_g, hi_g, step = -0.40, 0.80, 0.05
        grid = [lo_g + step * i for i in range(round((hi_g - lo_g) / step) + 1)]
        vals = [implied(g) for g in grid]
        known = [v for v in vals if v is not None]
        if not known:
            return None
        tol = 1e-9 * price      # stop bisecting once this close
        accept = 1e-6 * price   # report converged only within 0.0001%

        # A flat curve can match the price for every growth rate; that does
        # not identify a market-implied assumption. Exact grid hits, including
        # a tangency with no sign change, are otherwise valid solutions.
        if len(known) == len(grid) and all(abs(v - price) <= tol for v in known):
            return {**base, "converged": False, "implied_growth_y1": None,
                    "note": "The model matches the market price throughout the tested "
                    "growth range; growth cannot be uniquely inferred."}
        hits = [(g, p) for g, p in zip(grid, vals)
                if p is not None and abs(p - price) <= tol]
        brackets = []
        for (g_a, p_a), (g_b, p_b) in zip(zip(grid, vals), zip(grid[1:], vals[1:])):
            if p_a is None or p_b is None:
                continue
            if (abs(p_a - price) > tol and abs(p_b - price) > tol
                    and (p_a - price) * (p_b - price) < 0):
                brackets.append((g_a, p_a, g_b))
        crossings = len(hits) + len(brackets)
        bracket = brackets[0] if brackets else None
        if hits and (bracket is None or hits[0][0] < bracket[0]):
            out = {**base, "converged": True, "implied_growth_y1": hits[0][0],
                   "implied_revenue_growth": growth_path(hits[0][0])}
            if crossings > 1:
                out["note"] = "More than one growth rate reproduces the market price with these assumptions; showing the lowest."
            return out
        if bracket is None:
            return {
                **base,
                "converged": False,
                "implied_growth_y1": None,
                "note": "Market price is outside the solvable growth range "
                f"({lo_g:.0%} to {hi_g:.0%}) with the current assumptions: the "
                f"model's implied price only spans {min(known):,.2f} to "
                f"{max(known):,.2f} across that range, vs {price:,.2f}.",
            }

        lo, p_lo, hi = bracket
        mid, p_mid = lo, p_lo
        for _ in range(60):
            if abs(p_lo - price) <= tol:
                mid, p_mid = lo, p_lo
                break
            mid = (lo + hi) / 2.0
            p_mid = implied(mid)
            if p_mid is None:
                return None
            if abs(p_mid - price) <= tol:
                break
            if (p_mid - price) * (p_lo - price) > 0:
                lo, p_lo = mid, p_mid
            else:
                hi = mid
        if abs(p_mid - price) > accept:
            return {
                **base,
                "converged": False,
                "implied_growth_y1": None,
                "note": "The implied price jumps across the market price "
                "instead of passing through it; no growth rate reproduces it.",
            }
        out = {**base, "converged": True, "implied_growth_y1": mid,
               "implied_revenue_growth": growth_path(mid)}
        if crossings > 1:
            out["note"] = (
                "More than one growth rate reproduces the market price "
                "with these assumptions; showing the lowest."
            )
        return out
    except Exception:  # noqa: BLE001 - diagnostics only, never block valuation
        return None


def run_valuation(ticker: str, payload: Optional[dict] = None) -> dict:
    """Full pipeline: provider (cached) -> value_company -> serialized dict,
    plus the reverse-DCF (market-implied growth) diagnostic."""
    payload = payload or {}
    report, echo = run_valuation_report(ticker, payload)
    d = report_to_dict(report, assumptions_used=echo)

    macro, dcf, _ddm, _peers, _toggles, _echo = parse_assumptions(payload)
    from .serialization import _sanitize  # reverse_dcf rides after report_to_dict

    d["reverse_dcf"] = _sanitize(_reverse_dcf(report, dcf, macro))
    return d
