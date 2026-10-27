"""Drive the valuation engine from API requests.

Responsibilities:
  * translate the frontend's flat assumption payload into the engine's
    MacroAssumptions / DCFAssumptions / DDMAssumptions objects and the comps
    basis (``parse_assumptions``);
  * cache the expensive CompanyData fetch per ticker so moving an assumption
    slider re-runs the models instantly instead of re-hitting EDGAR/yfinance,
    and share one data provider so the SEC ticker directory is downloaded once
    rather than for every new ticker;
  * return the serialized report plus an echo of the assumptions actually
    used (``assumptions_used``) and of the defaults (``assumptions_default``),
    so the dashboard can flag custom inputs;
  * suggest peers on request (``discover_peers``), never applying them.
"""

from __future__ import annotations

import copy
import math
import secrets
import threading
import time
from typing import Any, Optional

from equity_valuation import value_company
from equity_valuation.data.base import DataProvider
from equity_valuation.engine import _discover_peers
from equity_valuation.models.comps import normalize_basis
from equity_valuation.models.dcf import TERMINAL_METHODS
from equity_valuation.models.reverse_dcf import growth_path_from_y1, solve_reverse_dcf
from equity_valuation.schemas import (
    DCFAssumptions,
    DDMAssumptions,
    MacroAssumptions,
)
from equity_valuation.utils import financial_institution_detail

from .serialization import _sanitize, report_to_dict

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

# Retain exact displayed calculations for exports after data caches expire.
# Snapshots are local to this process and bounded in both age and count.
_SNAPSHOT_TTL = 86400.0
_SNAPSHOT_LIMIT = 32
_snapshot_lock = threading.Lock()
_report_snapshots: dict[str, tuple[float, str, Any, dict]] = {}


class SnapshotError(ValueError):
    """A displayed report is no longer available for export."""


def _remember_report(ticker: str, report, echo: dict) -> str:
    now = time.monotonic()
    report_id = secrets.token_urlsafe(24)
    snapshot = (now, ticker.strip().upper(), copy.deepcopy(report), copy.deepcopy(echo))
    with _snapshot_lock:
        _prune(_report_snapshots, now, _SNAPSHOT_TTL)
        while len(_report_snapshots) >= _SNAPSHOT_LIMIT:
            del _report_snapshots[next(iter(_report_snapshots))]
        _report_snapshots[report_id] = snapshot
    return report_id


def get_report_snapshot(ticker: str, report_id: str):
    """Return an isolated copy; never silently recalculate an expired report."""
    with _snapshot_lock:
        _prune(_report_snapshots, time.monotonic(), _SNAPSHOT_TTL)
        snapshot = _report_snapshots.get(report_id) if isinstance(report_id, str) else None
        if snapshot is None or snapshot[1] != ticker.strip().upper():
            raise SnapshotError(
                "This displayed report is no longer available for export. "
                "Recompute the valuation, review it, then export again.")
        return copy.deepcopy(snapshot[2]), copy.deepcopy(snapshot[3])


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

    def discover_peers(self, ticker: str, *, industry_key: Optional[str] = None,
                       limit: Optional[int] = None, name: Optional[str] = None) -> dict:
        """Opt-in peer discovery, passed through to the underlying provider
        (``HybridProvider.discover_peers``: Yahoo's industry list). A provider
        without it (one not built on ``DataProvider``, whose default reads
        ``suggest_peers``) gives its ``suggest_peers`` list, as
        ``engine._discover_peers`` reads it. Uncached: Yahoo's industry
        lists are cached by the market client. Never raises."""
        discover = getattr(self._base, "discover_peers", None)
        if not callable(discover):
            return _discover_peers(self._base, ticker, industry_key, name)
        try:
            return discover(ticker, industry_key=industry_key, limit=limit, name=name)
        except Exception as exc:  # noqa: BLE001 - peers are optional
            return {"ticker": str(ticker or "").strip().upper(), "peers": [],
                    "source": "the data provider",
                    "notes": [f"No peers discovered: peer discovery failed ({exc})."]}


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


# The explicit path a year-1 override would give if it faded to the requested
# terminal growth; None without a y1. parse_assumptions no longer builds it (it
# hands the y1 to the engine as ``revenue_growth_start``, which fades to the
# terminal growth actually used); it stays for callers that re-price a solved
# year-1 rate, as the engine's reverse DCF builds its paths with the same
# function.
_revenue_growth_path = growth_path_from_y1

# Dashboard bounds on the opt-in drivers (decimals). A value outside them is
# almost always a unit slip (15 for 15%), so it is refused rather than run.
TERMINAL_RONIC_MAX = 5.0             # 0 < terminal_ronic <= 500%
DDM_HIGH_GROWTH_BOUNDS = (-0.5, 0.6)  # exclusive
DDM_METHODS = ("gordon", "two_stage", "h_model")
_DDM_DEFAULTS = DDMAssumptions()


def _choice(key: str, value: Any, allowed: tuple, default: str) -> str:
    """A case- and whitespace-insensitive pick from ``allowed``; None or ''
    gives ``default``. Anything else raises AssumptionError."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    choice = value.strip().lower() if isinstance(value, str) else None
    if choice not in allowed:
        raise AssumptionError(f"{key} must be one of {', '.join(allowed)} (got {value!r}).")
    return choice


def _years(payload: dict, key: str, default: int) -> int:
    """A whole number of years clamped to 1-15; empty gives ``default``."""
    v = _f(payload, key)
    return max(1, min(int(v) if v is not None else default, 15))


def parse_assumptions(payload: dict):
    """payload (all optional) -> (macro, dcf, ddm, peers, toggles, echo).

    Beyond the DCF and WACC drivers, the payload may carry:
      * ``terminal_method`` 'gordon' (default), 'exit_multiple' or
        'value_driver' (anything else is Gordon), and ``terminal_ronic``, the
        value-driver terminal's return on new invested capital: a decimal
        above 0 and at most ``TERMINAL_RONIC_MAX`` (empty uses WACC in the
        engine; ignored by the other methods);
      * the DDM drivers ``ddm_method`` ('two_stage' by default, 'gordon' or
        'h_model'), ``ddm_high_growth_rate`` (a decimal strictly inside
        ``DDM_HIGH_GROWTH_BOUNDS``; empty lets the DDM derive it) and
        ``ddm_high_growth_years`` (clamped to 1-15, default 5). The FCFE
        reads only the horizon and terminal growth, so it is unaffected;
      * ``comps_basis`` 'fy' (default) or 'ttm' (case-insensitive), the
        target figures comps apply the peer multiples to
        (``value_company(comps_basis=...)``).
    A ``revenue_growth_y1`` override goes to the engine as
    ``DCFAssumptions.revenue_growth_start``: the path starts at it and fades to
    the terminal growth the DCF actually uses, so a terminal growth clamped
    below WACC ends the path at the clamped rate, as the derived path does, and
    each sensitivity cell's path fades to that cell's terminal growth. An
    explicit ``revenue_growth`` list is used as given and wins over the y1.
    An unusable value raises AssumptionError (HTTP 400). ``echo`` holds every
    resolved value under the payload's key (``ddm_high_growth_rate`` and
    ``terminal_ronic`` are None when left to the engine);
    ``parse_assumptions({})[5]`` is the defaults echo."""
    payload = payload or {}

    rf = _f(payload, "rf", "risk_free_rate")
    erp = _f(payload, "erp", "equity_risk_premium")
    tax = _f(payload, "tax_rate", "tax")
    cod = _f(payload, "cost_of_debt", "pretax_cost_of_debt")
    forecast_years = _years(payload, "forecast_years", _cfg.DEFAULT_FORECAST_YEARS)
    terminal_growth = _f(payload, "terminal_growth")
    if terminal_growth is None:
        terminal_growth = _cfg.DEFAULT_TERMINAL_GROWTH
    terminal_method = payload.get("terminal_method")
    if terminal_method not in TERMINAL_METHODS:
        terminal_method = "gordon"
    terminal_ronic = _f(payload, "terminal_ronic")
    if terminal_ronic is not None and not 0 < terminal_ronic <= TERMINAL_RONIC_MAX:
        raise AssumptionError(
            "terminal_ronic must be a decimal above 0 and at most "
            f"{TERMINAL_RONIC_MAX:g} (e.g. 0.15 for 15%), got {terminal_ronic!r}.")
    ddm_method = _choice("ddm_method", payload.get("ddm_method"), DDM_METHODS,
                         _DDM_DEFAULTS.method)
    ddm_high_growth_rate = _f(payload, "ddm_high_growth_rate")
    lo, hi = DDM_HIGH_GROWTH_BOUNDS
    if ddm_high_growth_rate is not None and not lo < ddm_high_growth_rate < hi:
        raise AssumptionError(
            f"ddm_high_growth_rate must be a decimal between {lo:g} and {hi:g} "
            f"(e.g. 0.08 for 8%), got {ddm_high_growth_rate!r}.")
    ddm_high_growth_years = _years(payload, "ddm_high_growth_years",
                                   _DDM_DEFAULTS.high_growth_years)
    basis = payload.get("comps_basis")
    try:
        comps_basis = normalize_basis(
            None if isinstance(basis, str) and not basis.strip() else basis)
    except (TypeError, ValueError) as exc:
        raise AssumptionError(f"comps_basis must be 'fy' or 'ttm' (got {basis!r}).") from exc
    exit_ev_ebitda = _f(payload, "exit_ev_ebitda")
    target_ebit_margin = _f(payload, "target_ebit_margin")
    rev_y1 = _f(payload, "revenue_growth_y1")
    rev_path = payload.get("revenue_growth")
    if isinstance(rev_path, list) and rev_path:
        rev_path = [_num("revenue_growth", g) for g in rev_path]
        if any(g is None for g in rev_path):
            raise AssumptionError("revenue_growth must be a list of numbers.")
    else:
        # No explicit path: the engine fades from the y1 (if any) to the
        # terminal growth used. The resolved path is in the DCF's
        # assumptions['revenue_growth_path'].
        rev_path = None

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
        revenue_growth_start=rev_y1 if rev_path is None else None,
        terminal_growth=terminal_growth,
        terminal_method=terminal_method,
        exit_ev_ebitda=exit_ev_ebitda,
        target_ebit_margin=target_ebit_margin,
        tax_rate=tax,
        terminal_ronic=terminal_ronic,
    )
    ddm = DDMAssumptions(
        method=ddm_method,
        high_growth_years=ddm_high_growth_years,
        high_growth_rate=ddm_high_growth_rate,
        terminal_growth=terminal_growth,
        forecast_years=forecast_years,
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
        "terminal_ronic": terminal_ronic,
        "exit_ev_ebitda": exit_ev_ebitda,
        "target_ebit_margin": target_ebit_margin,
        "revenue_growth_y1": rev_y1,
        "revenue_growth": rev_path,  # an explicit list only
        # Echo as CSV — the frontend's Assumptions.peers is a string.
        "peers": ",".join(peers) if peers else None,
        "ddm_method": ddm_method,
        "ddm_high_growth_rate": ddm_high_growth_rate,
        "ddm_high_growth_years": ddm_high_growth_years,
        "comps_basis": comps_basis,
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
        comps_basis=echo["comps_basis"],
        **toggles,
    )
    return report, echo


def _reverse_dcf(report, dcf_assumptions, macro) -> Optional[dict]:
    """What the market price implies for year-1 revenue growth (the top-level
    keys), the target EBIT margin (``margin``) and the discount rate
    (``wacc``), other assumptions held: the 'what do I have to believe?'
    numbers. A thin wrapper over the engine's solver
    (``equity_valuation.models.reverse_dcf.solve_reverse_dcf``, which documents
    the method); None without a DCF.

    Unlike the engine's ``report.reverse_dcf``, which says a flagged company's
    reference-only FCFF DCF is not meaningful to invert, the dashboard still
    solves it (``include_reference_only``) and labels the result as an
    inversion of the reference-only DCF, not what the price implies (the
    DCF panel and ``serialization.build_ai_context``). For any other company
    both give the same result."""
    if report.dcf is None:
        return None
    return solve_reverse_dcf(report.company, macro, dcf_assumptions, report.current_price,
                             getattr(report.dcf, "assumptions", {}) or {},
                             include_reference_only=True)


def run_valuation(ticker: str, payload: Optional[dict] = None) -> dict:
    """Full pipeline: provider (cached) -> value_company -> serialized dict.

    Beyond the report's own fields the dict carries ``assumptions_used`` (the
    echo of the resolved inputs), ``assumptions_default`` (the same echo for
    an empty payload, so the dashboard can flag custom inputs; pure, no
    network) and ``report_id`` (the export snapshot).

    ``reverse_dcf`` (market-implied year-1 growth, target margin and discount
    rate) is the engine's ``report.reverse_dcf``, serialized with the rest of
    the report, so it is solved once. For a flagged company (bank, insurer,
    REIT, lender, captive-finance group, debt-funded lessor), where the engine
    declines to invert the reference-only FCFF DCF, the dashboard's
    ``_reverse_dcf`` still solves it and the UI labels the result as an
    inversion of the reference-only DCF (see ``_reverse_dcf``)."""
    payload = payload or {}
    report, echo = run_valuation_report(ticker, payload)
    d = report_to_dict(report, assumptions_used=echo)
    d["assumptions_default"] = parse_assumptions({})[5]
    if report.dcf is not None and financial_institution_detail(report.company):
        macro, dcf, *_rest = parse_assumptions(payload)
        d["reverse_dcf"] = _sanitize(_reverse_dcf(report, dcf, macro))
    d["report_id"] = _remember_report(ticker, report, echo)
    return d


def discover_peers(ticker: str) -> dict:
    """Suggested peers for ``ticker`` from the shared provider's opt-in
    discovery (``HybridProvider.discover_peers``: Yahoo's industry list,
    peers within a market-cap band, with notes asking for review). Nothing is
    applied: the dashboard fills its peer box with them.

    When the ticker's CompanyData is fresh in the valuation cache, its
    Yahoo industry key and company name are passed along, which saves the
    quote lookup and keeps the target's other share class out; otherwise
    the provider reads them itself. Returns the provider's dict
    (``{ticker, industry_key, industry, peers, notes, source}``)."""
    from equity_valuation.data.market import company_name_key

    ticker = ticker.strip().upper()
    now = time.time()
    industry_key = name = None
    with _cache_lock:
        hit = _company_cache.get(ticker)
    if hit and now - hit[0] < _CACHE_TTL:
        company = hit[1]
        market = getattr(company, "market", None)
        industry_key = getattr(market, "industry_key", None)
        # Yahoo's name first, as engine._auto_peers picks it; a name that
        # only repeats the ticker names no company.
        name = next((n.strip() for n in (getattr(market, "name", None),
                                         getattr(company, "name", None))
                     if company_name_key(n) not in (None, company_name_key(ticker))), None)
    return _discover_peers(_shared_base(now), ticker, industry_key, name)
