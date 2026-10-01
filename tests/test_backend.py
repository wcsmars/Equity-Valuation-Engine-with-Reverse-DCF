"""Offline regression tests for the FastAPI backend.

Every test runs without network access: valuations use the synthetic company,
the store and .env writes go to temporary directories, and the Anthropic SDK
talks to an in-process mock transport (no real API calls). Run with:
    python -m unittest tests.test_backend
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import shutil
import stat
import subprocess
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import ai_service, exports, filings, store, valuation_service as vs
from backend import app as backend
from backend.serialization import build_ai_context
from equity_valuation import value_company
from equity_valuation.data.edgar import _TICKERS_URL, EdgarClient
from equity_valuation.data.provider import HybridProvider
from equity_valuation.data.synthetic import (
    DEMO_PEERS,
    DEMO_TICKER,
    SyntheticProvider,
    make_company,
)
from equity_valuation.models.dcf import run_dcf
from equity_valuation.schemas import DCFAssumptions
from equity_valuation.utils import fade_path

LOCAL = "http://127.0.0.1:8000"


def _synthetic(ticker, refresh=False):
    return SyntheticProvider()


class _PricedProvider(SyntheticProvider):
    """The synthetic firm with a different market price and/or share count."""

    def __init__(self, price=None, share_mult=1.0):
        self.price, self.share_mult = price, share_mult

    def get_company_data(self, ticker):
        c = make_company()
        m = c.market
        price = self.price if self.price is not None else m.price / self.share_mult
        c.market = dataclasses.replace(
            m, price=price, shares_outstanding=m.shares_outstanding * self.share_mult)
        c.financials = dataclasses.replace(
            c.financials,
            diluted_shares=[s * self.share_mult for s in c.financials.diluted_shares])
        return c


def _report(provider, payload=None):
    macro, dcf_a, ddm_a, *_ = vs.parse_assumptions(payload or {})
    rep = value_company(DEMO_TICKER, provider=provider, macro=macro,
                        dcf_assumptions=dcf_a, ddm_assumptions=ddm_a, peers=DEMO_PEERS)
    return rep, macro, dcf_a


def _reprice(rep, macro, dcf_a, g1):
    path = vs._revenue_growth_path(g1, dcf_a.terminal_growth, dcf_a.forecast_years)
    a = dataclasses.replace(dcf_a, revenue_growth=path)
    return run_dcf(rep.company, macro, a, rep.current_price).implied_price


def _client():
    return TestClient(backend.app, base_url=LOCAL)


class _StoreSandbox(unittest.TestCase):
    """Points the research store at a temporary directory."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        data = Path(self.tmp.name) / "data"
        self.path = data / "copilot_store.json"
        self.patches = [patch.object(store, "_DATA_DIR", data),
                        patch.object(store, "_STORE_PATH", self.path)]
        for p in self.patches:
            p.start()
        data.mkdir()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def snapshot(self):
        return self.path.read_bytes() if self.path.exists() else None


def _asgi(app, chunks, *, path="/echo", headers=(), scope_type="http", unread=None):
    """Drive an ASGI app directly with a body sent in `chunks` (no
    Content-Length unless given in `headers`); returns the messages it sent.
    `unread`, if a list, receives the number of chunks never read."""
    import asyncio

    scope = {"type": scope_type, "asgi": {"version": "3.0"}, "http_version": "1.1",
             "method": "POST", "scheme": "http", "path": path,
             "raw_path": path.encode(), "root_path": "", "query_string": b"",
             "headers": [(b"host", b"127.0.0.1:8000"),
                         (b"content-type", b"application/json"), *headers],
             "client": ("127.0.0.1", 50000), "server": ("127.0.0.1", 8000)}
    pending = [{"type": "http.request", "body": c, "more_body": i < len(chunks) - 1}
               for i, c in enumerate(chunks)]
    sent: list[dict] = []

    async def receive():
        return pending.pop(0) if pending else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    if unread is not None:
        unread.append(len(pending))
    return sent


# --------------------------------------------------------------------------- #
#  Assumption parsing
# --------------------------------------------------------------------------- #
class ParseAssumptionTests(unittest.TestCase):
    def test_malformed_assumptions_are_not_silently_defaulted(self):
        for bad in ("4 percent", "typo", {}, [0.04], 10 ** 400):
            with self.subTest(value=bad):
                with self.assertRaises(vs.AssumptionError):
                    vs.parse_assumptions({"rf": bad})
        with patch.object(vs, "_get_provider") as provider:
            r = _client().post("/api/valuation", json={"ticker": "SYNT", "rf": "oops"})
            self.assertEqual(r.status_code, 400)
            provider.assert_not_called()

    def test_refresh_false_values_keep_the_cache(self):
        for value in (False, "false", "0", "off", 0, None):
            with self.subTest(value=value), patch.object(vs, "_get_provider", return_value=SyntheticProvider()) as get:
                vs.run_valuation_report(DEMO_TICKER, {"refresh": value, "run_sensitivity": False})
                get.assert_called_once_with(DEMO_TICKER, refresh=False)

    def test_one_year_horizon_keeps_year1_growth(self):
        self.assertEqual(vs._revenue_growth_path(0.15, 0.025, 1), [0.15])
        for n in range(2, 16):
            self.assertEqual(vs._revenue_growth_path(0.15, 0.025, n),
                             fade_path(0.15, 0.025, n))
        _m, dcf_a, *_rest, echo = vs.parse_assumptions(
            {"forecast_years": 1, "revenue_growth_y1": 0.15})
        self.assertEqual(dcf_a.revenue_growth, [0.15])
        self.assertEqual(echo["revenue_growth"], [0.15])

    def test_non_finite_and_boolean_values_rejected(self):
        for key in ("rf", "erp", "tax_rate", "terminal_growth", "forecast_years",
                    "exit_ev_ebitda", "target_ebit_margin", "revenue_growth_y1"):
            for bad in ("inf", "-Infinity", "nan", float("inf"), float("nan"), 1e400, True):
                with self.subTest(key=key, value=bad):
                    with self.assertRaises(vs.AssumptionError) as caught:
                        vs.parse_assumptions({key: bad})
                    self.assertIn(key, str(caught.exception))
        with self.assertRaises(vs.AssumptionError):
            vs.parse_assumptions({"revenue_growth": ["x", None]})
        with self.assertRaises(vs.AssumptionError):
            vs.parse_assumptions({"revenue_growth": [0.1, float("nan")]})

    def test_lenient_inputs_still_parse(self):
        _m, dcf_a, _d, _p, toggles, echo = vs.parse_assumptions(
            {"forecast_years": "7", "rf": "0.04", "tax_rate": "", "revenue_growth": ["0.1", 0.05],
             "run_dcf": "false", "run_ddm": 0, "run_comps": "yes"})
        self.assertEqual(dcf_a.forecast_years, 7)
        self.assertEqual(echo["rf"], 0.04)
        self.assertIsNone(echo["tax_rate"])  # unparseable -> engine default
        self.assertEqual(dcf_a.revenue_growth, [0.1, 0.05])
        self.assertFalse(toggles["run_dcf"])
        self.assertFalse(toggles["run_ddm"])
        self.assertTrue(toggles["run_comps"])
        self.assertTrue(toggles["run_fcfe"])

    def test_api_returns_400_not_500(self):
        with patch.object(vs, "_get_provider", _synthetic):
            c = _client()
            for body in ('{"ticker":"SYNT","forecast_years":"inf"}',
                         '{"ticker":"SYNT","forecast_years":1e400}',
                         '{"ticker":"SYNT","terminal_growth":NaN}',
                         '{"ticker":"SYNT","rf":Infinity}'):
                with self.subTest(body=body):
                    r = c.post("/api/valuation", content=body,
                               headers={"content-type": "application/json"})
                    self.assertEqual(r.status_code, 400)
                    self.assertIsInstance(r.json()["detail"], str)
            r = c.post("/api/export/memo", json={"ticker": "SYNT", "rf": "nan"})
            self.assertEqual(r.status_code, 400)
            r = c.post("/api/valuation",
                       json={"ticker": "SYNT", "forecast_years": 1, "revenue_growth_y1": 0.15})
            self.assertEqual(r.status_code, 200)
            d = r.json()
            self.assertEqual(d["dcf"]["assumptions"]["revenue_growth_path"], [0.15])
            self.assertEqual(d["reverse_dcf"]["current_assumption_y1"], 0.15)


# --------------------------------------------------------------------------- #
#  Reverse DCF
# --------------------------------------------------------------------------- #
class _FakeDCF:
    """Stands in for run_dcf with a chosen price(g1) curve."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, company, macro, a, price):
        return SimpleNamespace(implied_price=self.fn(a.revenue_growth[0]))


def _solve_fake(fn, price, n=5):
    rep = SimpleNamespace(dcf=object(), company=None, current_price=price)
    with patch("equity_valuation.models.dcf.run_dcf", _FakeDCF(fn)):
        return vs._reverse_dcf(rep, DCFAssumptions(forecast_years=n), None)


class ReverseDCFTests(unittest.TestCase):
    def test_reverse_preserves_the_computed_growth_path(self):
        # When the market price equals the computed DCF value, solving must
        # recover that same first-year growth, even after a terminal clamp or
        # with a non-linear explicit schedule.
        for payload in ({"terminal_growth": 0.2},
                        {"revenue_growth": [0.1, 0.2, 0.15, 0.1, 0.05]}):
            with self.subTest(payload=payload):
                rep, macro, dcf_a = _report(SyntheticProvider(), payload)
                rep.current_price = rep.dcf.implied_price
                original = rep.dcf.assumptions["revenue_growth_path"]
                r = vs._reverse_dcf(rep, dcf_a, macro)
                self.assertTrue(r["converged"])
                self.assertAlmostEqual(r["implied_growth_y1"], original[0], delta=1e-6)
                for expected, actual in zip(original, r["implied_revenue_growth"]):
                    self.assertAlmostEqual(expected, actual, delta=1e-6)
                a = dataclasses.replace(dcf_a, revenue_growth=r["implied_revenue_growth"])
                self.assertAlmostEqual(run_dcf(rep.company, macro, a, rep.current_price).implied_price,
                                       rep.current_price, delta=rep.current_price * 1e-6)

    def test_unavailable_dcf_does_not_produce_reverse_diagnostic(self):
        rep = SimpleNamespace(dcf=SimpleNamespace(assumptions={"valuation_available": False}),
                              company=None, current_price=40)
        self.assertIsNone(vs._reverse_dcf(rep, DCFAssumptions(), None))

    def test_auto_growth_is_reported_as_current_assumption(self):
        rep, macro, dcf_a = _report(SyntheticProvider())
        r = vs._reverse_dcf(rep, dcf_a, macro)
        self.assertEqual(r["current_assumption_y1"], rep.dcf.assumptions["revenue_growth_path"][0])

    def test_tangent_grid_hit_is_a_solution(self):
        r = _solve_fake(lambda g: 40 + (g - 0.2) ** 2, price=40)
        self.assertTrue(r["converged"])
        self.assertAlmostEqual(r["implied_growth_y1"], 0.2)

    def test_flat_matching_curve_does_not_identify_growth(self):
        r = _solve_fake(lambda g: 40.0, price=40)
        self.assertFalse(r["converged"])
        self.assertIsNone(r["implied_growth_y1"])
        self.assertIn("uniquely", r["note"])

    def test_solution_does_not_depend_on_share_count(self):
        results = []
        for mult in (1.0, 1_000.0, 100_000.0):
            rep, macro, dcf_a = _report(_PricedProvider(share_mult=mult))
            r = vs._reverse_dcf(rep, dcf_a, macro)
            self.assertTrue(r["converged"], mult)
            g1 = r["implied_growth_y1"]
            self.assertAlmostEqual(_reprice(rep, macro, dcf_a, g1) / rep.current_price, 1.0,
                                   delta=1e-6)
            results.append(g1)
        self.assertAlmostEqual(results[0], results[1], delta=1e-6)
        self.assertAlmostEqual(results[0], results[2], delta=1e-6)

    def test_penny_price_uses_relative_tolerance(self):
        r = _solve_fake(lambda g: 0.001 + 0.01 * g, price=0.001 + 0.01 * 0.1234)
        self.assertTrue(r["converged"])
        self.assertAlmostEqual(r["implied_growth_y1"], 0.1234, delta=1e-7)

    def test_root_inside_non_monotone_range_is_found(self):
        # Both ends of [-40%, 80%] price below the market, but the curve
        # crosses it twice in between; the old endpoint check said "outside".
        r = _solve_fake(lambda g: 50.0 - 400.0 * (g - 0.2) ** 2, price=40.0)
        self.assertTrue(r["converged"])
        self.assertAlmostEqual(r["implied_growth_y1"], 0.2 - (10 / 400) ** 0.5, delta=1e-7)
        self.assertIn("More than one", r["note"])

    def test_jump_across_price_is_not_reported_as_converged(self):
        r = _solve_fake(lambda g: 30.0 if g < 0.1 else 50.0, price=40.0)
        self.assertFalse(r["converged"])
        self.assertIsNone(r["implied_growth_y1"])

    def test_invalid_price_gives_no_solution(self):
        for price in (float("nan"), float("inf"), 0.0, -5.0, None):
            with self.subTest(price=price):
                r = _solve_fake(lambda g: 40.0 + g, price=price)
                self.assertFalse(r["converged"])
                self.assertIsNone(r["implied_growth_y1"])

    def test_one_year_horizon_solves_or_explains(self):
        payload = {"forecast_years": 1, "revenue_growth_y1": 0.15}
        # Market price inside the 1-year model's range: converges and reprices.
        rep, macro, dcf_a = _report(_PricedProvider(price=30.0), payload)
        r = vs._reverse_dcf(rep, dcf_a, macro)
        self.assertEqual(r["current_assumption_y1"], 0.15)
        self.assertTrue(r["converged"])
        self.assertAlmostEqual(_reprice(rep, macro, dcf_a, r["implied_growth_y1"]), 30.0,
                               delta=30.0 * 1e-6)
        # A very high market price is above what one year of growth can reach:
        # the note says so with the model's actual price range.
        rep, macro, dcf_a = _report(_PricedProvider(price=1000.0), payload)
        r = vs._reverse_dcf(rep, dcf_a, macro)
        self.assertFalse(r["converged"])
        self.assertIn("only spans", r["note"])


# --------------------------------------------------------------------------- #
#  Provider sharing and cache bounds
# --------------------------------------------------------------------------- #
class ProviderCacheTests(unittest.TestCase):
    """_get_provider reuses one HybridProvider, so the SEC ticker directory is
    fetched once, and its caches drop expired tickers. EDGAR is stubbed."""

    def setUp(self):
        self.clock = [1_000_000.0]
        self.fetched: list[str] = []
        directory = {str(i): {"cik_str": i + 1, "ticker": t, "title": f"{t} Inc."}
                     for i, t in enumerate(("AAA", "BBB", "CCC"))}

        def get_json(client, url):
            self.fetched.append(url)
            return directory

        def company_data(provider, ticker):
            provider.edgar.resolve_cik(ticker)  # the directory lookup under test
            return make_company()

        self.patches = [
            patch.object(vs, "_base", None),
            patch.dict(vs._company_cache, clear=True),
            patch.dict(vs._peer_cache, clear=True),
            patch.object(vs, "time", SimpleNamespace(time=lambda: self.clock[0])),
            patch.object(EdgarClient, "_get_json", get_json),
            patch.object(HybridProvider, "get_company_data", company_data),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

    def test_directory_fetched_once_across_tickers(self):
        bases = {id(vs._get_provider(t)._base) for t in ("AAA", "bbb", "CCC")}
        vs._get_provider("AAA", refresh=True)  # refetches the data, not the directory
        self.assertEqual(len(bases), 1)
        self.assertEqual(self.fetched, [_TICKERS_URL])
        # Entries hold only (ts, CompanyData); no per-entry provider or map.
        self.assertEqual({len(v) for v in vs._company_cache.values()}, {2})

    def test_shared_provider_rebuilt_after_a_day(self):
        first = vs._get_provider("AAA")._base
        self.clock[0] += vs._BASE_TTL
        second = vs._get_provider("BBB")._base
        self.assertIsNot(first, second)  # new listings become resolvable
        self.assertEqual(self.fetched, [_TICKERS_URL, _TICKERS_URL])

    def test_expired_entries_pruned_on_insert(self):
        stale = self.clock[0] - vs._CACHE_TTL
        vs._company_cache["OLD"] = (stale, object())
        vs._peer_cache[("OLD",)] = (stale, [object()])
        vs._get_provider("AAA")
        self.assertEqual(sorted(vs._company_cache), ["AAA"])
        cached = vs._CachedProvider(SimpleNamespace(get_peer_comp_rows=lambda t: ["row"]),
                                    make_company())
        self.assertEqual(cached.get_peer_comp_rows(["bbb"]), ["row"])
        self.assertEqual(sorted(vs._peer_cache), [("BBB",)])


# --------------------------------------------------------------------------- #
#  AI grounding context
# --------------------------------------------------------------------------- #
class AIContextTests(unittest.TestCase):
    def test_context_carries_the_drivers_actually_used(self):
        with patch.object(vs, "_get_provider", _synthetic):
            d = vs.run_valuation(DEMO_TICKER, {
                "revenue_growth_y1": 0.12, "target_ebit_margin": 0.3, "tax_rate": 0.23,
                "terminal_method": "exit_multiple", "exit_ev_ebitda": 14})
            default = vs.run_valuation(DEMO_TICKER, {})
        ctx = build_ai_context(d)
        self.assertIn("DCF revenue-growth path: 12.0%", ctx)
        self.assertIn("target 30.0%", ctx)
        self.assertIn("tax rate used 23.0%", ctx)
        self.assertIn("exit EV/EBITDA 14.0x", ctx)
        self.assertIn("revenue_growth_y1=0.12", ctx)
        self.assertIn("exit_ev_ebitda=14", ctx)
        ctx = build_ai_context(default)
        self.assertIn("DCF revenue-growth path: 8.0%", ctx)
        self.assertNotIn("tax n/a", ctx)
        self.assertIn("tax 21.0%", ctx)
        self.assertIn("REVERSE DCF", ctx)

    def test_sparse_report_does_not_raise(self):
        ctx = build_ai_context({"summary": {}, "dcf": None, "macro": {}})
        self.assertIn("CURRENT ASSUMPTION VALUES", ctx)


# --------------------------------------------------------------------------- #
#  DNS-rebinding guard
# --------------------------------------------------------------------------- #
class HostGuardTests(unittest.TestCase):
    def test_local_hosts_pass(self):
        c = TestClient(backend.app)
        for host in ("127.0.0.1:8000", "localhost:3000", "localhost", "LOCALHOST:8000",
                     "127.0.0.1", "[::1]:8000", "[::1]", "::1"):
            with self.subTest(host=host):
                self.assertEqual(c.get("/api/health", headers={"host": host}).status_code, 200)

    def test_next_rewrite_proxy_and_desktop_calls_pass(self):
        c = TestClient(backend.app)
        # Next's proxy (changeOrigin) sets Host to the backend target and
        # forwards the browser's Host as X-Forwarded-Host.
        for fwd in ("localhost:3000", "127.0.0.1:3000", ""):
            r = c.get("/api/health", headers={"host": "127.0.0.1:8000",
                                              "x-forwarded-host": fwd,
                                              "origin": "http://localhost:3000"})
            self.assertEqual(r.status_code, 200, fwd)
        # The desktop shell calls the backend's dynamic port directly.
        r = c.get("/api/health", headers={"host": "127.0.0.1:53817",
                                          "origin": "http://127.0.0.1:53816"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers.get("access-control-allow-origin"), "http://127.0.0.1:53816")

    def test_rebound_hosts_rejected(self):
        c = TestClient(backend.app)
        evil = "rebind.attacker.example:8000"
        with patch.object(backend, "_upsert_env_file") as save, \
                patch.dict(backend.os.environ, {}, clear=True):
            r = c.post("/api/settings", json={"anthropic_api_key": "your-attacker-key"},
                       headers={"host": evil, "origin": f"http://{evil}"})
            self.assertEqual(r.status_code, 400)
            save.assert_not_called()
            self.assertNotIn("ANTHROPIC_API_KEY", backend.os.environ)
        for headers in ({"host": evil},
                        {"host": "localhost.evil.example"},
                        {"host": "127.0.0.1.evil.example:8000"},
                        {"host": "127.0.0.1:8000", "x-forwarded-host": evil},
                        {"host": "127.0.0.1:8000", "x-forwarded-host": f"localhost:3000, {evil}"}):
            with self.subTest(headers=headers):
                for path in ("/api/watchlist", "/api/research_state/SYNT", "/api/health"):
                    self.assertEqual(c.get(path, headers=headers).status_code, 400)

    def test_middleware_builds_the_way_older_starlette_does(self):
        # Starlette before 0.41.3 (FastAPI before 0.118) calls cls(app=app,
        # ...); a differently named parameter made every request a 500.
        from fastapi.responses import JSONResponse

        async def inner(scope, receive, send):
            await JSONResponse({"ok": True})(scope, receive, send)

        stack = inner
        for m in reversed(backend.app.user_middleware):
            stack = m.cls(app=stack, *m.args, **m.kwargs)
        sent = _asgi(stack, [b"{}"], path="/api/health")
        self.assertEqual(sent[0]["status"], 200)
        self.assertIn(backend.LocalHostOnlyMiddleware,
                      [m.cls for m in backend.app.user_middleware])


# --------------------------------------------------------------------------- #
#  Cross-site request guard (CSRF), independent of the FastAPI version
# --------------------------------------------------------------------------- #
class CrossSiteGuardTests(_StoreSandbox):
    EVIL = "https://evil.example"

    def setUp(self):
        super().setUp()
        store.save_research("SYNT", {"notes": "keep me"})
        store.upsert_watchlist({"ticker": "KEEP"})
        self.fmp = patch.object(backend, "_fmp")
        self.fmp_mock = self.fmp.start()
        self.fmp_mock.enabled = True
        self.fmp_mock.enrichment.return_value = {"enabled": True}
        self.fmp_mock.transcripts_list.return_value = []

    def tearDown(self):
        self.fmp.stop()
        super().tearDown()

    def test_cross_site_writes_refused_before_any_effect(self):
        c = _client()
        writes = [("/api/settings", {"anthropic_api_key": "your-attacker-key"}),
                  ("/api/research_state/SYNT", {"notes": "overwritten"}),
                  ("/api/watchlist", {"action": "add", "ticker": "PWNED"}),
                  ("/api/ai/chat", {"turns": [{"role": "user", "content": "q"}]})]
        variants = [{"origin": self.EVIL},
                    {"origin": self.EVIL, "sec-fetch-site": "cross-site"},
                    {"origin": "null"},  # sandboxed frame or data: URL
                    {"origin": "http://localhost.evil.example:3000"},
                    {"sec-fetch-site": "cross-site"}]
        before = self.snapshot()
        with patch.object(backend, "_upsert_env_file") as save_env, \
                patch.object(ai_service, "chat") as chat:
            for path, body in writes:
                for headers in variants:
                    for untyped in (False, True):
                        with self.subTest(path=path, headers=headers, untyped=untyped):
                            if untyped:
                                # A no-preflight body with no Content-Type,
                                # which FastAPI before 0.132 parses as JSON.
                                r = c.post(path, content=json.dumps(body).encode(),
                                           headers=headers)
                            else:
                                r = c.post(path, json=body, headers=headers)
                            self.assertEqual(r.status_code, 403)
                            self.assertIn("Cross-site", r.json()["detail"])
            save_env.assert_not_called()
            chat.assert_not_called()
        self.assertEqual(self.snapshot(), before)

    def test_cross_site_side_effect_gets_refused(self):
        # An <img src> or link from another site: GET, no Origin header.
        c = _client()
        with patch.object(filings, "list_filings") as list_filings:
            for path in ("/api/enrichment/AAPL", "/api/transcripts/AAPL",
                         "/api/filings/AAPL", "/api/watchlist"):
                with self.subTest(path=path):
                    r = c.get(path, headers={"sec-fetch-site": "cross-site"})
                    self.assertEqual(r.status_code, 403)
            list_filings.assert_not_called()
        self.fmp_mock.enrichment.assert_not_called()
        self.fmp_mock.transcripts_list.assert_not_called()

    def test_dashboard_calls_pass(self):
        cases = {
            # Next rewrite proxy: same-origin page, Host rewritten to the backend.
            "next proxy": {"host": "127.0.0.1:8000", "x-forwarded-host": "127.0.0.1:3000",
                           "origin": "http://127.0.0.1:3000", "sec-fetch-site": "same-origin"},
            # Desktop app: 127.0.0.1:<front> calls 127.0.0.1:<api> directly.
            "desktop": {"host": "127.0.0.1:53817", "origin": "http://127.0.0.1:53816",
                        "sec-fetch-site": "same-site"},
            # A localhost page calling 127.0.0.1 is cross-site but sends its Origin.
            "localhost page": {"host": "127.0.0.1:8000", "origin": "http://localhost:3000",
                               "sec-fetch-site": "cross-site"},
            "ipv6 page": {"host": "[::1]:8000", "origin": "http://[::1]:3000",
                          "sec-fetch-site": "same-site"},
            "curl": {"host": "127.0.0.1:8000"},
        }
        c = TestClient(backend.app)
        for name, headers in cases.items():
            with self.subTest(name):
                r = c.post("/api/watchlist", json={"action": "add", "ticker": name[:4]},
                           headers=headers)
                self.assertEqual(r.status_code, 200)
                if "origin" in headers:
                    self.assertEqual(r.headers.get("access-control-allow-origin"),
                                     headers["origin"])
                r = c.get("/api/enrichment/AAPL", headers=headers)
                self.assertEqual(r.status_code, 200)
        # Address bar navigation to the API or /docs.
        r = c.get("/api/health", headers={"host": "127.0.0.1:8000", "sec-fetch-site": "none"})
        self.assertEqual(r.status_code, 200)

    def test_websocket_from_foreign_origin_is_closed(self):
        called = []

        async def inner(scope, receive, send):
            called.append(scope)

        guard = backend.LocalHostOnlyMiddleware(inner)
        sent = _asgi(guard, [], headers=[(b"origin", self.EVIL.encode())],
                     scope_type="websocket")
        self.assertEqual(sent, [{"type": "websocket.close", "code": 1008}])
        self.assertEqual(called, [])


# --------------------------------------------------------------------------- #
#  Request-body limits
# --------------------------------------------------------------------------- #
class BodyLimitTests(_StoreSandbox):
    def test_oversized_json_body_refused_and_store_untouched(self):
        store.save_research("SYNT", {"notes": "keep me"})
        before = self.snapshot()
        big = {"notes": "x" * (backend.MAX_BODY_BYTES + 1)}
        r = _client().post("/api/research_state/SYNT", json=big,
                           headers={"origin": "http://127.0.0.1:53816"})
        self.assertEqual(r.status_code, 413)
        self.assertIn("8 MB", r.json()["detail"])
        # CORS headers ride along, so the desktop UI can show the message.
        self.assertEqual(r.headers.get("access-control-allow-origin"), "http://127.0.0.1:53816")
        self.assertEqual(self.snapshot(), before)

    def test_declared_length_refused_before_the_body_is_read(self):
        with patch.object(store, "save_research") as save, \
                patch.object(ai_service, "digest") as digest:
            for path, limit in (("/api/research_state/SYNT", backend.MAX_BODY_BYTES),
                                ("/api/ai/digest", backend.MAX_UPLOAD_BODY_BYTES)):
                with self.subTest(path=path):
                    r = _client().post(path, content=b"{}",
                                       headers={"content-type": "application/json",
                                                "content-length": str(limit + 1)})
                    self.assertEqual(r.status_code, 413)
                    self.assertEqual("PDFs" in r.json()["detail"], path.startswith("/api/ai/"))
            save.assert_not_called()
            digest.assert_not_called()

    def test_pdf_sized_bodies_still_reach_the_ai_routes(self):
        pdf = "A" * (backend.MAX_BODY_BYTES + 1024)  # over the JSON cap, under the upload cap
        with patch.object(ai_service, "digest", return_value={"summary": "ok"}) as digest:
            r = _client().post("/api/ai/digest",
                               json={"pdfs": [{"name": "a.pdf", "data_base64": pdf}]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(digest.call_args.kwargs["pdfs"][0]["data_base64"]), len(pdf))

    def test_streamed_body_cut_off_at_the_cap(self):
        # No Content-Length (chunked): the cap applies to the bytes as they
        # arrive, the route never runs, and only the 413 goes out.
        from fastapi import FastAPI

        mini = FastAPI()
        calls = []

        @mini.post("/echo")
        def echo(body: dict):
            calls.append(body)
            return {"keys": sorted(body)}

        limited = backend.BodySizeLimitMiddleware(mini, max_body=64, max_upload_body=64)
        sent = _asgi(limited, [b'{"a": "', b"x" * 100, b'"}'])
        starts = [m for m in sent if m["type"] == "http.response.start"]
        self.assertEqual([m["status"] for m in starts], [413])
        self.assertIn(b"64 bytes", b"".join(m.get("body", b"") for m in sent))
        self.assertEqual(calls, [])
        sent = _asgi(limited, [b'{"a": ', b"1}"])
        self.assertEqual(sent[0]["status"], 200)
        self.assertEqual(calls, [{"a": 1}])

    def test_rest_of_an_oversized_body_is_read_before_the_413(self):
        # A proxy still streaming the upload (the Next rewrite) relays the 413
        # only if the connection is not closed mid-write, so the body is read
        # and dropped first, up to the upload cap.
        async def route(scope, receive, send):
            while (await receive()).get("more_body"):
                pass
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"route ran"})

        limited = backend.BodySizeLimitMiddleware(route, max_body=64, max_upload_body=1000)
        chunks = [b"x" * 50] * 4
        for label, headers, left in (
                ("declared", [(b"content-length", b"200")], 0),
                ("streamed", [], 0),
                ("past the upload cap", [(b"content-length", b"5000")], 4)):
            with self.subTest(label):
                unread: list[int] = []
                sent = _asgi(limited, chunks, headers=headers, unread=unread)
                self.assertEqual([m.get("status") for m in sent][:1], [413])
                self.assertNotIn(b"route ran", b"".join(m.get("body", b"") for m in sent))
                self.assertEqual(unread, [left])

    def test_upload_cap_does_not_exceed_the_frontend_proxy_cap(self):
        # The Next proxy cuts a body past its own cap short but forwards the
        # full Content-Length. Such a body must be refused from its declared
        # length; under a larger backend cap it would be waited for instead.
        import re

        config = Path(__file__).resolve().parents[1] / "frontend" / "next.config.mjs"
        found = re.search(r'middlewareClientMaxBodySize:\s*"(\d+)mb"',
                          config.read_text(encoding="utf-8")) if config.exists() else None
        if found is None:
            self.skipTest("frontend proxy body cap not found")
        self.assertLessEqual(backend.MAX_UPLOAD_BODY_BYTES, int(found.group(1)) * 1024 * 1024)


# --------------------------------------------------------------------------- #
#  .env persistence
# --------------------------------------------------------------------------- #
class LauncherTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "requires Bash process groups")
    def test_server_failure_stops_the_other_server_and_preserves_exit_code(self):
        for failed in ("backend", "frontend"):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                shutil.copyfile(Path(__file__).resolve().parents[1] / "run_dev.sh", root / "run_dev.sh")
                (root / ".venv/bin").mkdir(parents=True)
                (root / "frontend/node_modules").mkdir(parents=True)
                (root / "bin").mkdir()
                # A long-lived sibling records that the launcher's EXIT trap
                # terminated its process group. No servers or installers run.
                survivor = "trap 'touch stopped; exit 0' TERM\ntouch ready\nwhile :; do sleep 1; done\n"
                failure = "while [ ! -f ready ]; do sleep 0.05; done\nexit 23\n"
                python = root / ".venv/bin/python"
                python.write_text("#!/bin/bash\ncd \"$(dirname \"$0\")/../..\"\n"
                                  "if [ \"$2\" = pip ]; then exit 0; fi\n"
                                  + (failure if failed == "backend" else survivor))
                npm = root / "bin/npm"
                npm.write_text("#!/bin/bash\ncd \"$(dirname \"$0\")/..\"\n"
                               + (failure if failed == "frontend" else survivor))
                for p in (python, npm):
                    p.chmod(0o700)
                env = {**os.environ, "PATH": str(root / "bin") + os.pathsep + os.environ["PATH"]}
                result = subprocess.run(["bash", "run_dev.sh"], cwd=root, env=env,
                                        capture_output=True, text=True, timeout=8)
                self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
                self.assertTrue((root / "stopped").exists(), result.stdout + result.stderr)


class EnvFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = Path(self.tmp.name) / ".env"
        self.patch = patch.object(backend, "_ENV_PATH", self.env)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_new_file_is_owner_only(self):
        backend._upsert_env_file({"FMP_API_KEY": "your-new-fmp-key"})
        self.assertEqual(self.env.read_text(), "FMP_API_KEY=your-new-fmp-key\n")
        self.assertEqual(stat.S_IMODE(self.env.stat().st_mode), 0o600)
        self.assertEqual(os.listdir(self.tmp.name), [".env"])

    def test_concurrent_settings_preserve_both_keys_and_live_values(self):
        first_inside = threading.Event()
        release_first = threading.Event()
        second_started = threading.Event()
        second_inside = threading.Event()
        real_save = backend._upsert_env_file

        def save(updates):
            if "ANTHROPIC_API_KEY" in updates:
                first_inside.set()
                if not release_first.wait(5):
                    raise RuntimeError("settings test timed out")
            else:
                second_inside.set()
            real_save(updates)

        def second():
            second_started.set()
            return backend.settings(backend.SettingsRequest(fmp_api_key="your-fmp-key"))

        with patch.object(backend, "_upsert_env_file", save), \
                patch.dict(backend.os.environ, {}, clear=True), patch.object(backend, "_fmp"), \
                ThreadPoolExecutor(max_workers=2) as pool:
            one = pool.submit(backend.settings, backend.SettingsRequest(anthropic_api_key="your-anthropic-key"))
            self.assertTrue(first_inside.wait(2))
            two = pool.submit(second)
            try:
                self.assertTrue(second_started.wait(2))
                self.assertFalse(second_inside.wait(0.1))
            finally:
                release_first.set()
            one.result(timeout=3)
            two.result(timeout=3)
            self.assertIn("ANTHROPIC_API_KEY=your-anthropic-key", self.env.read_text())
            self.assertIn("FMP_API_KEY=your-fmp-key", self.env.read_text())
            self.assertEqual(backend.os.environ["FMP_API_KEY"], "your-fmp-key")

    def test_every_assignment_replaced_once(self):
        self.env.write_text(
            "# keys\nFMP_API_KEY=old\nANTHROPIC_MODEL=claude-opus-5\n"
            "  export FMP_API_KEY = dupe\nFMP_API_KEY=dupe-later\nOTHER=1\n")
        os.chmod(self.env, 0o644)
        backend._upsert_env_file({"FMP_API_KEY": "your-new-fmp-key",
                                  "ANTHROPIC_API_KEY": "your-anthropic-key"})
        self.assertEqual(
            self.env.read_text(),
            "# keys\nFMP_API_KEY=your-new-fmp-key\nANTHROPIC_MODEL=claude-opus-5\nOTHER=1\n"
            "ANTHROPIC_API_KEY=your-anthropic-key\n")
        self.assertEqual(stat.S_IMODE(self.env.stat().st_mode), 0o600)
        bash = shutil.which("bash")
        if bash:
            out = subprocess.run(
                [bash, "-c", f"set -a; source '{self.env}'; echo \"$FMP_API_KEY\""],
                capture_output=True, text=True, check=True).stdout.strip()
            self.assertEqual(out, "your-new-fmp-key")

    def test_settings_route_writes_through_real_upsert(self):
        with patch.dict(backend.os.environ, {}, clear=True), patch.object(backend, "_fmp"):
            r = _client().post("/api/settings", json={"fmp_api_key": "your-fmp-key"})
            self.assertEqual(r.status_code, 200)
            self.assertIn("FMP_API_KEY=your-fmp-key\n", self.env.read_text())
            multiline = "x\nEVIL=1"
            r = _client().post("/api/settings", json={"fmp_api_key": multiline})
            self.assertEqual(r.status_code, 400)
            self.assertNotIn("EVIL", self.env.read_text())


# --------------------------------------------------------------------------- #
#  Research store
# --------------------------------------------------------------------------- #
def _strict_json(text):
    """Parse `text` as standard JSON: bare NaN/Infinity are errors, as in
    JSON.parse or jq (Python's json.loads accepts them by default)."""
    def refuse(token):
        raise ValueError(f"not standard JSON: {token}")

    return json.loads(text, parse_constant=refuse)


def _post_raw(client, path, body):
    """POST `body` exactly as given, the way a non-browser client may send
    NaN/Infinity tokens."""
    return client.post(path, content=body, headers={"content-type": "application/json"})


class StoreTests(_StoreSandbox):
    def test_null_clears_saved_fields_but_omitted_fields_survive(self):
        store.save_research("SYNT", {"notes": "keep", "note": {"title": "old"},
                                     "assumptions": {"rf": 0.04}})
        c = _client()
        r = c.post("/api/research_state/SYNT", json={"note": None})
        self.assertEqual(r.status_code, 200)
        saved = c.get("/api/research_state/SYNT").json()
        self.assertIsNone(saved["note"])
        self.assertEqual(saved["notes"], "keep")
        self.assertEqual(saved["assumptions"], {"rf": 0.04})

    def test_overflowed_json_exponents_do_not_break_store_responses(self):
        self.path.write_text('{"watchlist": [{"ticker": "X", "price": 1e400}], '
                             '"research": {"X": {"assumptions": {"rf": -1e400}}}}')
        c = _client()
        self.assertIsNone(c.get("/api/watchlist").json()["watchlist"][0]["price"])
        self.assertIsNone(c.get("/api/research_state/X").json()["assumptions"]["rf"])
        store.save_research("X", {"notes": "still usable"})
        _strict_json(self.path.read_text())

    def backups(self):
        return sorted(p for p in self.path.parent.iterdir() if ".corrupt" in p.name)

    def test_round_trip_leaves_no_temp_files(self):
        store.upsert_watchlist({"ticker": "aapl", "price": 1.0})
        store.save_research("AAPL", {"notes": "hello"})
        self.assertEqual([w["ticker"] for w in store.get_watchlist()], ["AAPL"])
        self.assertEqual(store.get_research("aapl")["notes"], "hello")
        self.assertEqual(os.listdir(self.path.parent), ["copilot_store.json"])

    def test_undecodable_store_is_backed_up_not_500(self):
        raw = b'{"watchlist": [], "research": {"X": {"notes": "caf\xe9"}}}'
        self.path.write_bytes(raw)
        r = _client().get("/api/watchlist")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"watchlist": []})
        (backup,) = self.backups()
        self.assertEqual(backup.read_bytes(), raw)

    def test_repeated_corruption_keeps_every_backup(self):
        for content in ("{bad1", "{bad2", '["a list, not a store"]', '{"watchlist": {}}',
                        '{"research": {"X": []}}'):
            self.path.write_text(content)
            self.assertEqual(store.get_watchlist(), [])
            store.upsert_watchlist({"ticker": "T"})
        self.assertEqual(sorted(b.read_text() for b in self.backups()),
                         sorted(['["a list, not a store"]', '{"research": {"X": []}}',
                                 '{"watchlist": {}}', "{bad1", "{bad2"]))

    def test_transient_read_error_does_not_move_valid_store(self):
        store.upsert_watchlist({"ticker": "KEEP"})
        before = self.path.read_bytes()
        with patch("backend.store.open", side_effect=PermissionError("locked"), create=True):
            with self.assertRaises(store.StoreError):
                store.get_watchlist()
            r = _client().get("/api/watchlist")
            self.assertEqual(r.status_code, 503)
            r = _client().post("/api/watchlist", json={"action": "add", "ticker": "NEW"})
            self.assertEqual(r.status_code, 503)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.backups(), [])
        self.assertEqual([w["ticker"] for w in store.get_watchlist()], ["KEEP"])

    def test_oversized_research_state_refused_whole(self):
        store.save_research("SYNT", {"notes": "keep me", "digests": [{"d": 1}]})
        with patch.object(store, "_MAX_RESEARCH_BYTES", 1000):
            store.save_research("SYNT", {"notes": "short edit"})
            before = self.path.read_bytes()
            with self.assertRaises(store.StoreLimitError) as caught:
                store.save_research("SYNT", {"notes": "x" * 1000})
            self.assertEqual(caught.exception.status_code, 413)
            # The merged state counts: small parts that add up are refused too.
            with self.assertRaises(store.StoreLimitError):
                store.save_research("SYNT", {"digests": [{"d": "y" * 990}]})
            r = _client().post("/api/research_state/SYNT", json={"notes": "x" * 1000})
            self.assertEqual(r.status_code, 413)
            self.assertIn("shorten", r.json()["detail"])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(store.get_research("SYNT")["notes"], "short edit")

    def test_implausible_ticker_refused(self):
        long = "X" * (store._MAX_TICKER_LEN + 1)
        c = _client()
        r = c.post(f"/api/research_state/{long}", json={"notes": "n"})
        self.assertEqual(r.status_code, 400)
        r = c.post("/api/watchlist", json={"action": "add", "ticker": long})
        self.assertEqual(r.status_code, 400)
        for blank in ("%20", "%20%20%09"):
            with self.subTest(path=blank):
                r = c.post(f"/api/research_state/{blank}", json={"notes": "n"})
                self.assertEqual(r.status_code, 400)
                self.assertIn("ticker is required", r.json()["detail"])
        with self.assertRaises(store.StoreLimitError) as caught:
            store.save_research("  ", {"notes": "n"})
        self.assertEqual(caught.exception.status_code, 400)
        self.assertFalse(self.path.exists())
        ok = "X" * store._MAX_TICKER_LEN
        self.assertEqual(c.post(f"/api/research_state/{ok}", json={"notes": "n"}).status_code, 200)

    def test_store_growth_past_cap_refused_but_shrinking_allowed(self):
        store.upsert_watchlist({"ticker": "A", "name": "a" * 150})
        store.upsert_watchlist({"ticker": "B", "name": "b" * 150})
        size = self.path.stat().st_size
        with patch.object(store, "_MAX_STORE_BYTES", size - 200):  # already over the cap
            before = self.path.read_bytes()
            with self.assertRaises(store.StoreLimitError):
                store.upsert_watchlist({"ticker": "C"})
            with self.assertRaises(store.StoreLimitError):
                store.save_research("A", {"notes": "more"})
            self.assertEqual(self.path.read_bytes(), before)
            r = _client().post("/api/watchlist", json={"action": "add", "ticker": "C"})
            self.assertEqual(r.status_code, 413)
            # Saves that shrink the file still go through, even above the cap.
            store.upsert_watchlist({"ticker": "A", "name": "a"})
            self.assertGreater(self.path.stat().st_size, store._MAX_STORE_BYTES)
            self.assertEqual([w["ticker"] for w in store.remove_watchlist("B")], ["A"])

    def test_watchlist_fields_keep_their_types(self):
        (entry,) = store.upsert_watchlist({
            "ticker": "t", "name": "n" * 5000, "currency": {"nested": "x" * 5000},
            "price": "12.5", "blended_target": True, "recommendation": "Undervalued"})
        self.assertEqual(len(entry["name"]), store._MAX_TEXT_FIELD)
        self.assertIsNone(entry["currency"])
        self.assertIsNone(entry["price"])
        self.assertIsNone(entry["blended_target"])
        self.assertEqual(entry["recommendation"], "Undervalued")
        (entry,) = store.upsert_watchlist({"ticker": "T", "name": "Synthetic Corp",
                                           "currency": "USD", "price": 41.5,
                                           "blended_target": 50, "recommendation": None})
        self.assertEqual((entry["name"], entry["currency"], entry["price"],
                          entry["blended_target"]), ("Synthetic Corp", "USD", 41.5, 50))
        # Python's JSON parser accepts NaN/Infinity from a non-browser client;
        # they are stored as null and the file stays standard JSON.
        big = 10 ** 400  # a valid JSON integer too large for a float
        r = _post_raw(_client(), "/api/watchlist",
                      '{"action": "add", "ticker": "NANCO", "snapshot": {"price": NaN, '
                      f'"blended_target": Infinity, "name": {big}}}}}')
        self.assertEqual(r.status_code, 200)
        saved = _strict_json(self.path.read_text())["watchlist"][0]
        self.assertEqual(saved["ticker"], "NANCO")
        self.assertIsNone(saved["price"])
        self.assertIsNone(saved["blended_target"])
        self.assertIsNone(saved["name"])
        entry = store.upsert_watchlist({"ticker": "BIG", "price": big,
                                        "blended_target": -float("inf")})[0]
        self.assertEqual(entry["price"], big)
        self.assertIsNone(entry["blended_target"])

    def test_non_finite_research_values_saved_as_null(self):
        r = _post_raw(_client(), "/api/research_state/SYNT",
                      '{"notes": "n", "assumptions": {"rf": NaN, "path": [0.1, -Infinity]}, '
                      '"digests": [{"score": Infinity, "tags": ["ok"]}]}')
        self.assertEqual(r.status_code, 200)
        saved = _strict_json(self.path.read_text())["research"]["SYNT"]
        self.assertEqual(saved["assumptions"], {"rf": None, "path": [0.1, None]})
        self.assertEqual(saved["digests"], [{"score": None, "tags": ["ok"]}])
        self.assertEqual(r.json()["assumptions"], saved["assumptions"])

    def test_store_with_bare_nan_is_read_as_null_and_rewritten_strictly(self):
        # An earlier version could write these tokens; the store stays usable.
        self.path.write_text('{"watchlist": [{"ticker": "OLD", "price": NaN}], '
                             '"research": {"OLD": {"assumptions": {"rf": Infinity}}}}')
        self.assertIsNone(store.get_watchlist()[0]["price"])
        self.assertEqual(store.get_research("OLD")["assumptions"], {"rf": None})
        self.assertEqual(self.backups(), [])
        store.save_research("OLD", {"notes": "n"})
        data = _strict_json(self.path.read_text())
        self.assertIsNone(data["watchlist"][0]["price"])

    def test_save_refuses_non_finite_numbers_as_a_backstop(self):
        store.upsert_watchlist({"ticker": "KEEP"})
        before = self.path.read_bytes()
        with self.assertRaises(store.StoreLimitError) as caught:
            store._save({"watchlist": [{"ticker": "X", "price": float("nan")}],
                         "research": {}})
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(os.listdir(self.path.parent), ["copilot_store.json"])


# --------------------------------------------------------------------------- #
#  Office exports
# --------------------------------------------------------------------------- #
class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [patch.object(vs, "_get_provider", _synthetic),
                        patch.object(exports, "_ensure_out", lambda: Path(self.tmp.name))]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_number_formatting(self):
        self.assertEqual(exports._money(-41.17), "-$41.17")
        self.assertEqual(exports._money(12.5, "EUR"), "€12.50")
        self.assertEqual(exports._money(float("nan")), "n/a")
        self.assertEqual(exports._cap(3e8), "300M")
        self.assertEqual(exports._cap(3.4e9), "3.4B")
        self.assertEqual(exports._cap(2.345e12), "2,345B")
        self.assertEqual(exports._cap(None), "n/a")

    def test_memo_shows_exit_multiple_and_own_metadata(self):
        from docx import Document
        from pptx import Presentation

        payload = {"peers": ",".join(DEMO_PEERS), "terminal_method": "exit_multiple",
                   "exit_ev_ebitda": 14}
        memo = Document(exports.export_memo(DEMO_TICKER, payload))
        text = "\n".join(p.text for p in memo.paragraphs)
        self.assertIn("exit EV/EBITDA 14.0x", text)
        deck = Presentation(exports.export_deck(DEMO_TICKER, payload))
        year = __import__("datetime").datetime.now().year
        for props in (memo.core_properties, deck.core_properties):
            self.assertEqual(props.author, "Equity Research Automation")
            self.assertEqual(props.last_modified_by, "Equity Research Automation")
            self.assertEqual(props.comments, "")
            self.assertIn("Synthetic Corp", props.title)
            self.assertGreaterEqual(props.created.year, year - 1)

    def test_concurrent_same_ticker_exports_are_not_corrupted(self):
        from openpyxl import load_workbook

        c = _client()

        def run(g):
            r = c.post("/api/export/excel", json={"ticker": "SYNT", "revenue_growth_y1": g})
            self.assertEqual(r.status_code, 200)
            self.assertIn("SYNT_valuation.xlsx", r.headers["content-disposition"])
            load_workbook(io.BytesIO(r.content))  # raises on a torn file
            return True

        with ThreadPoolExecutor(3) as ex:
            self.assertTrue(all(ex.map(run, [0.01 * i for i in range(6)])))


# --------------------------------------------------------------------------- #
#  AI service (mock transport; never calls the real API)
# --------------------------------------------------------------------------- #
def _http_lib():
    """The HTTP package the installed anthropic SDK is built on: httpx for the
    0.x line, httpx2 from 1.0. Mock transports, responses and errors must come
    from the SDK's own package, so it is read off the SDK's client class rather
    than imported by name (another one may be installed alongside)."""
    import importlib

    import anthropic

    base = next(c for c in anthropic.DefaultHttpxClient.__mro__ if c.__name__ == "Client")
    return importlib.import_module(base.__module__.split(".")[0])


def _sdk_client(handler):
    import anthropic

    http = _http_lib()
    return anthropic.Anthropic(
        api_key="your-test-key", max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=http.MockTransport(handler)))


def _message(blocks, stop_reason="end_turn"):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5",
            "content": blocks, "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1}}


class _Recorder:
    def __init__(self, status=200, body=None, headers=None, exc=None):
        self.status, self.body, self.headers, self.exc = status, body, headers or {}, exc
        self.requests: list[dict] = []

    def __call__(self, request):
        self.requests.append(json.loads(request.content))
        if self.exc is not None:
            raise self.exc
        return _http_lib().Response(self.status, json=self.body, headers=self.headers)


class AIServiceTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"ANTHROPIC_API_KEY": "your-test-key"})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def _with(self, rec):
        return patch.object(ai_service, "_client", lambda: _sdk_client(rec))

    def test_chat_request_shape_and_reply(self):
        rec = _Recorder(body=_message([{"type": "thinking", "thinking": "", "signature": "s"},
                                       {"type": "text", "text": " The answer. "}]))
        with self._with(rec):
            reply = ai_service.chat("ctx", [{"role": "user", "content": "q"}])
        self.assertEqual(reply, "The answer.")
        req = rec.requests[0]
        self.assertGreaterEqual(req["max_tokens"], 16000)
        self.assertEqual(req["thinking"], {"type": "adaptive"})
        self.assertEqual(req["output_config"], {"effort": "high"})
        self.assertEqual(req["messages"], [{"role": "user", "content": "q"}])

    def test_chat_never_returns_empty(self):
        cases = [
            (_message([{"type": "thinking", "thinking": "", "signature": "s"}], "max_tokens"),
             "output limit"),
            (_message([], "refusal"), "declined"),
            (_message([{"type": "text", "text": "  "}]), "empty"),
        ]
        for body, words in cases:
            with self.subTest(stop=body["stop_reason"]), self._with(_Recorder(body=body)):
                with self.assertRaises(ai_service.AIError) as caught:
                    ai_service.chat("ctx", [{"role": "user", "content": "q"}])
                self.assertIn(words, str(caught.exception))
        body = _message([{"type": "text", "text": "partial"}], "max_tokens")
        with self._with(_Recorder(body=body)):
            reply = ai_service.chat("ctx", [{"role": "user", "content": "q"}])
        self.assertTrue(reply.startswith("partial"))
        self.assertIn("truncated", reply)

    def test_bad_turns_rejected_before_any_call(self):
        bad = [
            [],
            [{"content": "hi"}],
            [{"role": "system", "content": "hi"}],
            [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "q"}],
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": "prefill"}],
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": ""},
             {"role": "user", "content": "q2"}],
            [{"role": "user", "content": ["block"]}],
            ["just a string"],
        ]
        rec = _Recorder(body=_message([{"type": "text", "text": "x"}]))
        with self._with(rec):
            for turns in bad:
                with self.subTest(turns=turns):
                    with self.assertRaises(ai_service.AIError):
                        ai_service.chat("ctx", turns)
            r = _client().post("/api/ai/chat", json={"turns": [{"content": "hi"}]})
            self.assertEqual(r.status_code, 400)
            self.assertIsInstance(r.json()["detail"], str)
        self.assertEqual(rec.requests, [])

    def test_pdf_payloads_are_normalised(self):
        blocks = ai_service._pdf_blocks([
            "not a dict", None, {"data_base64": ""},
            {"data_base64": "data:application/pdf;base64,JVBE\nRi0x\n", "name": "a.pdf"},
            {"data": " JVBE Ri0x "},
        ])
        self.assertEqual([b["source"]["data"] for b in blocks], ["JVBERi0x", "JVBERi0x"])
        self.assertEqual(blocks[0]["title"], "a.pdf")
        rec = _Recorder(body=_message([{"type": "text", "text": "ok"}]))
        with self._with(rec):
            ai_service.chat("ctx", [{"role": "user", "content": "q1"},
                                    {"role": "assistant", "content": "a1"},
                                    {"role": "user", "content": "q2"}],
                            pdfs=[{"data_base64": "JVBERi0x"}])
        last = rec.requests[0]["messages"][-1]
        self.assertEqual(last["role"], "user")
        self.assertEqual(last["content"][0]["type"], "document")
        self.assertEqual(last["content"][-1], {"type": "text", "text": "q2"})

    def test_haiku_override_omits_thinking_and_effort(self):
        digest = {"summary": "s", "sentiment": "neutral", "key_facts": [], "risks": [],
                  "catalysts": [], "suggested_assumptions": []}
        rec = _Recorder(body=_message([{"type": "text", "text": json.dumps(digest)}]))
        with self._with(rec), patch.object(ai_service, "MODEL", "claude-haiku-4-5"):
            self.assertEqual(ai_service.digest("ctx", "material"), digest)
            rec.body = _message([{"type": "text", "text": "hi"}])
            ai_service.chat("ctx", [{"role": "user", "content": "q"}])
        structured, chat = rec.requests
        self.assertNotIn("thinking", structured)
        self.assertEqual(set(structured["output_config"]), {"format"})
        self.assertNotIn("thinking", chat)
        self.assertNotIn("output_config", chat)
        self.assertEqual(chat["model"], "claude-haiku-4-5")
        rec.requests.clear()
        rec.body = _message([{"type": "text", "text": json.dumps(digest)}])
        with self._with(rec):  # default model keeps adaptive thinking + effort
            ai_service.digest("ctx", "material")
        self.assertEqual(rec.requests[0]["thinking"], {"type": "adaptive"})
        self.assertEqual(rec.requests[0]["output_config"]["effort"], "high")
        self.assertGreaterEqual(rec.requests[0]["max_tokens"], 16000)

    def test_sdk_errors_map_to_clean_http_errors(self):
        import anthropic

        http = _http_lib()

        def err(status, message, headers=None):
            body = {"type": "error", "error": {"type": "x", "message": message}}
            return _Recorder(status=status, body=body, headers=headers)

        req = http.Request("POST", "https://api.anthropic.com/v1/messages")
        cases = [
            (err(401, "invalid x-api-key"), 502, "ANTHROPIC_API_KEY"),
            (err(403, "no access"), 502, "ANTHROPIC_API_KEY"),
            (err(404, "model: nope"), 502, "ANTHROPIC_MODEL"),
            (err(400, "messages: bad thing"), 400, "messages: bad thing"),
            (err(413, "too big"), 413, "too big"),
            (err(429, "slow down", {"retry-after": "7"}), 429, "slow down"),
            (err(500, "boom"), 502, "boom"),
            (err(529, "overloaded"), 503, "overloaded"),
            (_Recorder(exc=http.ConnectError("refused", request=req)), 503, "Connection"),
        ]
        body = {"turns": [{"role": "user", "content": "q"}]}
        for rec, code, words in cases:
            with self.subTest(code=code, words=words), self._with(rec):
                r = _client().post("/api/ai/chat", json=body)
                self.assertEqual(r.status_code, code)
                detail = r.json()["detail"]
                self.assertIsInstance(detail, str)
                self.assertIn(words, detail)
                self.assertNotIn("Error code:", detail)
                if code == 429:
                    self.assertEqual(r.headers.get("retry-after"), "7")
        self.assertTrue(issubclass(anthropic.RateLimitError, anthropic.APIStatusError))

    def test_installed_sdk_accepts_the_parameters_we_send(self):
        import inspect

        import anthropic

        params = inspect.signature(anthropic.resources.messages.Messages.create).parameters
        self.assertIn("output_config", params)
        self.assertIn("thinking", params)


# --------------------------------------------------------------------------- #
#  Filing section extraction
# --------------------------------------------------------------------------- #
class FilingSectionTests(unittest.TestCase):
    @staticmethod
    def _words(tag, n):
        return " ".join(f"{tag}{i}" for i in range(n))

    def test_cross_references_do_not_end_a_section(self):
        w = self._words
        text = "\n".join([
            "Item 1. Business  3", "Item 1A. Risk Factors  12",
            "Item 7. Management's Discussion and Analysis  30",
            "Item 1. Business",
            f"BUSINESS_START {w('b', 150)}. For risks, see Part I, Item 1A of this "
            f"Form 10-K under Risk Factors. {w('c', 150)} BUSINESS_END",
            "Item 1A. Risk Factors",
            f"RISK_START {w('r', 200)} RISK_END",
            "Item 1B. Unresolved Staff Comments", "None.",
            "Item 7. Management's Discussion and Analysis of Financial Condition",
            f"MDNA_START {w('m', 150)} read with the financial statements in Item 8. "
            f"Financial Statements and Supplementary Data. {w('n', 150)} Market risk is "
            f"in Item 7A, Quantitative and Qualitative Disclosures. {w('o', 150)} MDNA_END",
            "Item 7A. Quantitative and Qualitative Disclosures About Market Risk",
            w("q", 50),
            "Item 8. Financial Statements and Supplementary Data",
            w("f", 50),
        ])
        s = filings.extract_sections(text, "10-K")
        self.assertTrue(s["business"].rstrip().endswith("BUSINESS_END"))
        self.assertTrue(s["risk_factors"].rstrip().endswith("RISK_END"))
        self.assertTrue(s["mdna"].rstrip().endswith("MDNA_END"))

    def test_falls_back_to_inline_end_heading(self):
        w = self._words
        text = (f"Item 1A. Risk Factors\n{w('r', 200)} RISK_END Item 1B. Unresolved "
                f"Staff Comments {w('z', 200)}")
        s = filings.extract_sections(text, "10-K")
        self.assertTrue(s["risk_factors"].rstrip().endswith("RISK_END"))


if __name__ == "__main__":
    unittest.main()
