"""Offline tests for opt-in peer discovery (``--auto-peers``).

``YFinanceClient.discover_peers`` reads Yahoo's industry list
(``yf.Industry(key).top_companies``); here ``_industry`` is patched to return a
stand-in whose ``top_companies`` is a DataFrame shaped like yfinance 1.x
output (index ``symbol``; columns ``name``, ``rating``, ``market weight``).
Covered: the candidate filter, the market-weight size band, its one
widening and the closest-in-size ranking, a target missing from the list,
the target's other share class (Yahoo lists one class per company: GOOG,
not GOOGL; BRK-B, not BRK-A) matched by company name, the per-industry cache
and its TTL, failure paths that return no peers with a note, ``industry_key``
in ``get_market_data``, the HybridProvider hand-off, the engine's
``auto_peers`` switch (off: no discovery request and unchanged warnings),
the ``DataProvider`` default that reads ``suggest_peers``,
the summary wording for discovered peers that give no price, and the CLI's
``--auto-peers`` flag. No network access is needed.
Run with:  python -m unittest tests.test_peer_discovery
"""

from __future__ import annotations

import contextlib
from dataclasses import replace
import io
import math
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from equity_valuation import config, value_company
from equity_valuation.cli import build_parser, main
from equity_valuation.data import market as market_mod
from equity_valuation.data.provider import HybridProvider
from equity_valuation.data.synthetic import DEMO_PEERS, DEMO_TICKER, SyntheticProvider, make_company
from equity_valuation.data.market import YFinanceClient, company_name_key
from equity_valuation.schemas import CompRow

NaN = np.nan
KEY = "beverages-non-alcoholic"
INDUSTRY = "Beverages - Non-Alcoholic"

# Yahoo's beverages list for KO (market weight: share of the industry's
# market cap). BUDA has no name; GOGY is tiny.
KO_ROWS = [
    ("KO", "The Coca-Cola Company", 0.527),
    ("PEP", "PepsiCo, Inc.", 0.243),
    ("MNST", "Monster Beverage Corporation", 0.121),
    ("KDP", "Keurig Dr Pepper Inc.", 0.062),
    ("COKE", "Coca-Cola Consolidated, Inc.", 0.018),
    ("PRMB", "Primo Brands Corporation", 0.0096),
    ("CELH", "Celsius Holdings, Inc.", 0.0095),
    ("BUDA", NaN, 0.004),
    ("GOGY", "Golden Grail Technology Corp.", 0.000003),
]

# Yahoo's lists as read on 2026-10-27 (abridged): one share class per
# company, so GOOGL and BRK-A targets are not listed under their own symbol.
GOOG_KEY = "internet-content-information"
GOOG_ROWS = [
    ("GOOG", "Alphabet Inc.", 0.674743),
    ("META", "Meta Platforms, Inc.", 0.282074),
    ("SPOT", "Spotify Technology S.A.", 0.020666),
    ("NBIS", NaN, 0.006833),
    ("RDDT", "Reddit, Inc.", 0.004677),
    ("PINS", "Pinterest, Inc.", 0.002088),
    ("SNAP", "Snap Inc.", 0.001883),
    ("MTCH", "Match Group, Inc.", 0.001822),
    ("ZG", "Zillow Group, Inc.", 0.001332),
]
BRK_KEY = "insurance-diversified"
BRK_ROWS = [
    ("BRK-B", "Berkshire Hathaway Inc.", 0.908111),
    ("AIG", "American International Group, Inc.", 0.032806),
    ("HIG", "The Hartford Insurance Group, Inc.", 0.028475),
    ("ACGL", "Arch Capital Group Ltd.", 0.026987),
    ("PLGO", NaN, 0.001663),
    ("XZO", "Exzeo Group, Inc.", 0.001147),
    ("IGIC", "International General Insurance Holdings Ltd.", 0.000811),
]


def top_companies(rows) -> pd.DataFrame:
    """A ``top_companies`` frame as yfinance 1.x builds it."""
    return pd.DataFrame(
        [(sym, name, "Buy", w) for sym, name, w in rows],
        columns=["symbol", "name", "rating", "market weight"],
    ).set_index("symbol")


def industry(rows=KO_ROWS, name=INDUSTRY):
    """A yfinance ``Industry`` stand-in."""
    ind = mock.NonCallableMock(spec=["top_companies", "name"])
    ind.top_companies = top_companies(rows) if rows is not None else None
    ind.name = name
    return ind


def ticker(info: dict):
    tk = mock.NonCallableMock(spec=["info", "fast_info"])
    tk.info = info
    tk.fast_info = mock.NonCallableMock(spec=["last_price", "currency", "shares"],
                                        last_price=None, currency=None, shares=None)
    return tk


def client(ind=None, tickers=None) -> YFinanceClient:
    c = YFinanceClient()
    c._industry = mock.Mock(return_value=ind if ind is not None else industry())
    c._ticker = mock.Mock(side_effect=lambda sym: (tickers or {})[sym])
    return c


def symbols(result: dict) -> list[str]:
    return [p["ticker"] for p in result["peers"]]


class _FreshCache(unittest.TestCase):
    def setUp(self) -> None:
        market_mod._peer_cache.clear()
        self.addCleanup(market_mod._peer_cache.clear)


class DiscoverPeersTests(_FreshCache):
    def test_ko_band_widens_once(self) -> None:
        # 0.1-10x KO's weight holds PEP, MNST, KDP (3 < 4): widened to
        # 0.02-50x, which adds COKE (0.034x) but not PRMB (0.0182x).
        c = client()
        r = c.discover_peers("ko", industry_key=KEY)
        self.assertEqual(symbols(r), ["PEP", "MNST", "KDP", "COKE"])
        self.assertEqual(r["ticker"], "KO")
        self.assertEqual(r["industry_key"], KEY)
        self.assertEqual(r["industry"], INDUSTRY)
        self.assertEqual(r["source"], "Yahoo Finance industry list")
        pep = r["peers"][0]
        self.assertEqual(pep["name"], "PepsiCo, Inc.")
        self.assertAlmostEqual(pep["market_weight"], 0.243)
        self.assertAlmostEqual(pep["size_ratio"], 0.243 / 0.527)
        self.assertTrue(any("widened" in n for n in r["notes"]), r["notes"])
        self.assertTrue(r["notes"][0].startswith(
            f"Peers suggested from Yahoo's {INDUSTRY} list (4 of 9 names"), r["notes"])
        self.assertIn("closest in size first, 0.02-50x KO's size", r["notes"][0])
        self.assertTrue(r["notes"][0].endswith("review them before relying on comps."))
        self.assertIn("Size band widened from 0.1-10x to 0.02-50x KO's market weight: only 3 "
                      "names were within the narrower band.", r["notes"])
        c._industry.assert_called_once_with(KEY)
        c._ticker.assert_not_called()  # the key was given: no quote request

    def test_band_is_not_widened_with_four_names_in_it(self) -> None:
        rows = [("AAA", "A", 0.30), ("BBB", "B", 0.25), ("CCC", "C", 0.20),
                ("DDD", "D", 0.10), ("EEE", "E", 0.05), ("TINY", "T", 0.001)]
        r = client(industry(rows)).discover_peers("BBB", industry_key="k")
        self.assertEqual(symbols(r), ["AAA", "CCC", "DDD", "EEE"])
        self.assertFalse(any("widened" in n for n in r["notes"]), r["notes"])
        self.assertIn("0.1-10x BBB's size", r["notes"][0])

    def test_absent_target_gets_the_largest_names_without_a_band(self) -> None:
        r = client().discover_peers("XYZ", industry_key=KEY, limit=3)
        self.assertEqual(symbols(r), ["KO", "PEP", "MNST"])
        self.assertTrue(all(p["size_ratio"] is None for p in r["peers"]))
        self.assertIn(f"Size band not applied: XYZ has no market weight in Yahoo's {INDUSTRY} "
                      "list, so its largest names are proposed; they may differ widely in size "
                      "from XYZ.", r["notes"])
        self.assertIn("(3 of 9 names, largest market weight first)", r["notes"][0])

    def test_in_band_names_are_ranked_closest_in_size_first(self) -> None:
        # Six names within 0.1-10x of T: the limit keeps the three closest
        # in size (A 0.9x, B 3x, C 0.25x), not the largest (E 9.5x, BIG 5x).
        rows = [("E", "E", 0.95), ("BIG", "Big", 0.5), ("B", "B", 0.3), ("T", "Target", 0.1),
                ("A", "A", 0.09), ("C", "C", 0.025), ("D", "D", 0.011)]
        r = client(industry(rows)).discover_peers("T", industry_key="k", limit=3)
        self.assertEqual(symbols(r), ["A", "B", "C"])
        self.assertIn("(3 of 7 names, closest in size first, 0.1-10x T's size)", r["notes"][0])
        everyone = client(industry(rows)).discover_peers("T", industry_key="k")
        self.assertEqual(symbols(everyone), ["A", "B", "C", "BIG", "D", "E"])

    def test_default_limit_and_nameless_rows(self) -> None:
        r = client().discover_peers("XYZ", industry_key=KEY)
        self.assertEqual(len(r["peers"]), config.PEER_DISCOVERY_LIMIT)
        self.assertNotIn("BUDA", symbols(r))  # Yahoo gives no name for it
        self.assertEqual(symbols(r)[-1], "GOGY")

    def test_candidate_filter(self) -> None:
        rows = [("^GSPC", "Index", 0.2), ("EURUSD=X", "FX", 0.2), ("0700.HK", "Num", 0.2),
                ("low", "Lower", 0.2), ("TOOLONGX", "Long", 0.2), ("ZERO", "Zero", 0.0),
                ("NEG", "Neg", -0.1), ("NANW", "NaN", NaN), ("INF", "Inf", math.inf),
                ("BLANK", "  ", 0.2), ("BRK-B", "Berkshire B", 0.3), ("SHEL.L", "Shell", 0.25),
                ("OK", "Okay", 0.2), ("OK", "Okay again", 0.19)]
        r = client(industry(rows)).discover_peers("XYZ", industry_key="k")
        self.assertEqual(symbols(r), ["BRK-B", "SHEL.L", "OK"])

    def test_other_share_classes_of_a_listed_target_are_not_peers(self) -> None:
        # Both of the target's classes listed (Yahoo lists one; see below).
        rows = [("GOOGL", "Alphabet Inc.", 0.40), ("GOOG", "Alphabet Inc.", 0.38),
                ("META", "Meta Platforms, Inc.", 0.30), ("NWSA", "News Corporation", 0.05),
                ("NWS", "news corporation", 0.05), ("PINS", "Pinterest, Inc.", 0.04),
                ("RDDT", "Reddit, Inc.", 0.03)]
        r = client(industry(rows)).discover_peers("GOOGL", industry_key="k")
        self.assertEqual(symbols(r), ["META", "NWSA", "PINS", "RDDT"])

    def test_googl_target_finds_goog_by_name(self) -> None:
        # Yahoo's list holds GOOG, not GOOGL: GOOG is the target itself, never
        # a peer, and its weight sets the band (META 0.42x; SPOT 0.031x after
        # widening; RDDT 0.0069x is out).
        r = client(industry(GOOG_ROWS)).discover_peers(
            "GOOGL", industry_key=GOOG_KEY, name="Alphabet Inc.")
        self.assertEqual(symbols(r), ["META", "SPOT"])
        self.assertAlmostEqual(r["peers"][0]["size_ratio"], 0.282074 / 0.674743)
        self.assertIn("GOOG is not proposed: Yahoo lists Alphabet Inc. under it, another share "
                      "class of GOOGL; GOOG's market weight sets the size band.", r["notes"])
        self.assertIn("Size band widened from 0.1-10x to 0.02-50x GOOGL's market weight: only 1 "
                      "name was within the narrower band.", r["notes"])
        self.assertFalse(any("not applied" in n for n in r["notes"]), r["notes"])

    def test_brk_a_target_finds_brk_b_by_name(self) -> None:
        # No name within 0.1-10x of BRK-B's 0.908; widened: AIG, HIG, ACGL.
        for name in ("Berkshire Hathaway Inc.", "BERKSHIRE HATHAWAY INC"):
            with self.subTest(name=name):
                r = client(industry(BRK_ROWS)).discover_peers(
                    "BRK-A", industry_key=BRK_KEY, name=name)
                self.assertEqual(symbols(r), ["AIG", "HIG", "ACGL"])
                self.assertTrue(any(n.startswith("BRK-B is not proposed") for n in r["notes"]))
                self.assertIn("Size band widened from 0.1-10x to 0.02-50x BRK-A's market "
                              "weight: no name was within the narrower band.", r["notes"])

    def test_target_name_defaults_to_the_quote_name(self) -> None:
        for info in ({"longName": "Alphabet Inc.", "shortName": "Alphabet"},
                     {"shortName": "Alphabet Inc."}):
            with self.subTest(info=info):
                c = client(industry(GOOG_ROWS),
                           tickers={"GOOGL": ticker({"industryKey": GOOG_KEY, **info})})
                self.assertEqual(symbols(c.discover_peers("GOOGL")), ["META", "SPOT"])
        # A name passed in wins over the quote's.
        c = client(industry(GOOG_ROWS), tickers={"GOOGL": ticker(
            {"industryKey": GOOG_KEY, "longName": "Something Else"})})
        self.assertEqual(symbols(c.discover_peers("GOOGL", name="Alphabet Inc.")),
                         ["META", "SPOT"])

    def test_without_a_name_the_other_class_cannot_be_told_apart(self) -> None:
        # With the key given and no name (or only the ticker as its name) no
        # quote is read: GOOG is then just the list's largest name, and the
        # band is not applied, as for any unlisted target.
        for name in (None, "GOOGL", "  "):
            with self.subTest(name=name):
                c = client(industry(GOOG_ROWS))
                r = c.discover_peers("GOOGL", industry_key=GOOG_KEY, name=name, limit=2)
                self.assertEqual(symbols(r), ["GOOG", "META"])
                c._ticker.assert_not_called()

    def test_company_name_key(self) -> None:
        self.assertEqual(company_name_key("Alphabet Inc."), "alphabet inc")
        self.assertEqual(company_name_key(" ALPHABET  INC "), "alphabet inc")
        self.assertEqual(company_name_key("The Coca-Cola Company"), "the coca cola company")
        for blank in (None, NaN, "", "  ", "..."):
            self.assertIsNone(company_name_key(blank))

    def test_dotted_class_share_target_is_found_under_its_dash(self) -> None:
        rows = [("BRK-B", "Berkshire", 0.5), ("AAA", "A", 0.2), ("BBB", "B", 0.1),
                ("CCC", "C", 0.08), ("DDD", "D", 0.06)]
        r = client(industry(rows)).discover_peers("BRK.B", industry_key="k")
        self.assertEqual(symbols(r), ["AAA", "BBB", "CCC", "DDD"])
        self.assertAlmostEqual(r["peers"][0]["size_ratio"], 0.4)

    def test_no_name_in_band_gives_no_peers_and_a_note(self) -> None:
        rows = [("BIG", "Big", 0.99), ("DUST", "Dust", 0.0001)]
        r = client(industry(rows)).discover_peers("BIG", industry_key="k")
        self.assertEqual(r["peers"], [])
        self.assertEqual(r["notes"], [
            f"No peers discovered: no other company in Yahoo's {INDUSTRY} list is within "
            "0.02-50x BIG's size.",
            "Size band widened from 0.1-10x to 0.02-50x BIG's market weight: no name was "
            "within the narrower band."])

    def test_a_list_of_only_the_target_gives_no_peers(self) -> None:
        r = client(industry([("KO", "The Coca-Cola Company", 1.0)])).discover_peers(
            "KO", industry_key=KEY)
        self.assertEqual(r["peers"], [])
        self.assertEqual(r["notes"], [
            f"No peers discovered: Yahoo's {INDUSTRY} list (1 name) has no listed company "
            "with a market weight other than KO."])

    def test_industry_name_falls_back_to_the_quote_then_the_key(self) -> None:
        c = client(industry(name=None),
                   tickers={"KO": ticker({"industryKey": KEY, "industry": INDUSTRY})})
        self.assertEqual(c.discover_peers("KO")["industry"], INDUSTRY)
        self.assertIsNone(market_mod._peer_cache[KEY][1])  # Yahoo gave no name
        # The key was passed, so no quote is read: the key labels the notes.
        r = client(industry(name=None)).discover_peers("KO", industry_key=KEY)
        self.assertEqual(r["industry"], KEY)
        self.assertIn(f"Yahoo's {KEY} list", r["notes"][0])

    def test_key_is_read_from_the_quote_when_not_given(self) -> None:
        c = client(tickers={"KO": ticker({"industryKey": f" {KEY} ", "industry": INDUSTRY})})
        r = c.discover_peers("KO")
        c._industry.assert_called_once_with(KEY)
        self.assertEqual(symbols(r), ["PEP", "MNST", "KDP", "COKE"])

    def test_missing_industry_key_gives_no_peers(self) -> None:
        c = client(tickers={"KO": ticker({"industry": INDUSTRY})})
        r = c.discover_peers("KO")
        self.assertEqual(r["peers"], [])
        self.assertEqual(r["notes"], ["No peers discovered: Yahoo gives no industry for KO."])
        c._industry.assert_not_called()

    def test_industry_error_gives_no_peers_and_never_raises(self) -> None:
        for exc in (RuntimeError("HTTP 404"), ImportError("No module named 'yfinance'")):
            with self.subTest(exc=type(exc).__name__):
                c = client()
                c._industry = mock.Mock(side_effect=exc)
                r = c.discover_peers("KO", industry_key=KEY)
                self.assertEqual(r["peers"], [])
                self.assertEqual(len(r["notes"]), 1)
                self.assertIn("could not be read", r["notes"][0])
                self.assertIn(str(exc), r["notes"][0])

    def test_quote_error_gives_no_peers(self) -> None:
        c = client()
        c._ticker = mock.Mock(side_effect=ImportError("no yfinance"))
        r = c.discover_peers("KO")
        self.assertEqual(r["peers"], [])
        self.assertIn("no yfinance", r["notes"][0])

    def test_empty_or_missing_list_gives_no_peers(self) -> None:
        for rows in (None, []):
            with self.subTest(rows=rows):
                r = client(industry(rows)).discover_peers("KO", industry_key=KEY)
                self.assertEqual(r["peers"], [])
                self.assertIn("empty or unavailable", r["notes"][0])

    def test_suggest_peers_stays_empty_without_any_request(self) -> None:
        c = client()
        self.assertEqual(c.suggest_peers("KO"), [])
        c._industry.assert_not_called()
        c._ticker.assert_not_called()


class PeerCacheTests(_FreshCache):
    def test_one_fetch_per_industry_until_the_ttl_passes(self) -> None:
        clock = [1000.0]
        with mock.patch.object(market_mod, "_clock", side_effect=lambda: clock[0]):
            c = client()
            first = c.discover_peers("KO", industry_key=KEY)
            # Another client and another target in the same industry reuse the list.
            second = client(ind=industry(rows=[])).discover_peers("PEP", industry_key=KEY)
            self.assertEqual(c._industry.call_count, 1)
            self.assertEqual(symbols(second)[:2], ["MNST", "KO"])  # closest to PEP first
            clock[0] += config.PEER_DISCOVERY_TTL - 1
            c.discover_peers("KO", industry_key=KEY)
            self.assertEqual(c._industry.call_count, 1)
            clock[0] += 2  # past the TTL
            third = c.discover_peers("KO", industry_key=KEY)
            self.assertEqual(c._industry.call_count, 2)
        self.assertEqual(symbols(first), symbols(third))

    def test_empty_lists_are_not_cached(self) -> None:
        c = client(industry(rows=[]))
        c.discover_peers("KO", industry_key=KEY)
        c.discover_peers("KO", industry_key=KEY)
        self.assertEqual(c._industry.call_count, 2)
        self.assertNotIn(KEY, market_mod._peer_cache)

    def test_failed_fetches_are_not_cached(self) -> None:
        c = client()
        c._industry = mock.Mock(side_effect=RuntimeError("rate limited"))
        c.discover_peers("KO", industry_key=KEY)
        c.discover_peers("KO", industry_key=KEY)
        self.assertEqual(c._industry.call_count, 2)


class MarketDataIndustryKeyTests(unittest.TestCase):
    def test_get_market_data_reads_industry_key(self) -> None:
        info = {"currency": "USD", "currentPrice": 60.0, "sharesOutstanding": 10.0,
                "industry": INDUSTRY, "industryKey": f"  {KEY} "}
        md = client(tickers={"KO": ticker(info)}).get_market_data("KO")
        self.assertEqual(md.industry_key, KEY)
        self.assertEqual(md.industry, INDUSTRY)

    def test_missing_industry_key_is_none(self) -> None:
        for value in (None, "", "   "):
            with self.subTest(value=value):
                info = {"currency": "USD", "currentPrice": 60.0, "sharesOutstanding": 10.0}
                if value is not None:
                    info["industryKey"] = value
                md = client(tickers={"KO": ticker(info)}).get_market_data("KO")
                self.assertIsNone(md.industry_key)

    def test_schema_default_is_none(self) -> None:
        self.assertIsNone(make_company().market.industry_key)


class HybridDiscoverPeersTests(_FreshCache):
    def test_delegates_to_the_market_client(self) -> None:
        c = client()
        r = HybridProvider(edgar=mock.Mock(), market=c).discover_peers(
            "KO", industry_key=KEY, limit=2)
        self.assertEqual(symbols(r), ["PEP", "MNST"])

    def test_name_is_passed_on(self) -> None:
        market = mock.Mock()
        HybridProvider(edgar=mock.Mock(), market=market).discover_peers(
            "GOOGL", industry_key=GOOG_KEY, name="Alphabet Inc.")
        market.discover_peers.assert_called_once_with(
            "GOOGL", industry_key=GOOG_KEY, limit=None, name="Alphabet Inc.")
        c = client(industry(GOOG_ROWS))
        r = HybridProvider(edgar=mock.Mock(), market=c).discover_peers(
            "GOOGL", industry_key=GOOG_KEY, name="Alphabet Inc.")
        self.assertEqual(symbols(r), ["META", "SPOT"])

    def test_client_error_gives_no_peers(self) -> None:
        market = mock.Mock()
        market.discover_peers.side_effect = RuntimeError("boom")
        r = HybridProvider(edgar=mock.Mock(), market=market).discover_peers("ko")
        self.assertEqual(r["peers"], [])
        self.assertEqual(r["ticker"], "KO")
        self.assertIn("boom", r["notes"][0])

    def test_suggest_peers_is_unchanged(self) -> None:
        c = client()
        self.assertEqual(HybridProvider(edgar=mock.Mock(), market=c).suggest_peers("KO"), [])
        c._industry.assert_not_called()


class _NoSuggestion(SyntheticProvider):
    """The synthetic company with no default peers (like the live provider)."""

    def suggest_peers(self, ticker):
        return []


class _Guarded(_NoSuggestion):
    """Discovery must not be asked for."""

    def discover_peers(self, ticker, *, industry_key=None, limit=None, name=None):
        raise AssertionError("discover_peers called")


class _Stub(_NoSuggestion):
    """Discovers P1, P2 (and the target, which must be dropped). ``peers``
    items are tickers or ``(ticker, name)`` pairs; ``rows`` is True (the
    synthetic peer rows), False (none) or a list of rows to serve."""

    def __init__(self, peers=("P1", "P2", "SYNT"), notes=("Peers suggested from a stub.",),
                 rows=True, industry_key="stub-industry", company=None):
        self.peers, self.notes, self.rows, self.key = list(peers), list(notes), rows, industry_key
        self.company = company
        self.calls = []

    def get_company_data(self, ticker):
        company = self.company or make_company()
        return replace(company, market=replace(company.market, industry_key=self.key))

    def discover_peers(self, ticker, *, industry_key=None, limit=None, name=None):
        self.calls.append((ticker, industry_key, name))
        peers = [{"ticker": p[0], "name": p[1]} if isinstance(p, tuple) else {"ticker": p}
                 for p in self.peers]
        return {"ticker": ticker, "peers": peers, "notes": list(self.notes),
                "source": "a stub list"}

    def get_peer_comp_rows(self, tickers):
        if isinstance(self.rows, list):
            return list(self.rows)
        return super().get_peer_comp_rows(tickers) if self.rows else []


def _with_kind(company, kind):
    """``company`` with the data layer's ``financials._financial_kind`` set."""
    fin = replace(company.financials)
    fin._financial_kind = kind
    return replace(company, financials=fin)


class EngineAutoPeersTests(unittest.TestCase):
    def test_off_by_default_makes_no_discovery_request(self) -> None:
        for peers in (None, [], list(DEMO_PEERS)):
            with self.subTest(peers=peers):
                base = value_company(DEMO_TICKER, provider=_NoSuggestion(), peers=peers)
                guarded = value_company(DEMO_TICKER, provider=_Guarded(), peers=peers)
                self.assertEqual(guarded.warnings, base.warnings)
                self.assertEqual(guarded.summary, base.summary)

    def test_default_path_keeps_the_no_peers_warning(self) -> None:
        report = value_company(DEMO_TICKER, provider=_Guarded())
        self.assertIn("Comps: no usable peers found (pass --peers to supply them).",
                      report.warnings)
        self.assertFalse(any("discovered" in w for w in report.warnings))

    def test_discovered_peers_feed_comps_and_are_listed(self) -> None:
        # Three peers: a multiple needs config.COMPS_MIN_PEER_VALUES of them
        # to give an implied price.
        stub = _Stub(peers=("P1", "P2", "P3", "SYNT"))
        report = value_company(DEMO_TICKER, provider=stub, auto_peers=True)
        self.assertEqual(stub.calls, [("SYNT", "stub-industry", "Synthetic Corp")])
        self.assertEqual([r.ticker for r in report.comps.peers], ["P1", "P2", "P3"])
        self.assertIn("Comps: peers discovered automatically from a stub list: P1, P2, P3; "
                      "review them.", report.warnings)
        self.assertIn("Comps: Peers suggested from a stub.", report.warnings)
        self.assertIn("Comps (median)", report.summary["methods"])

    def test_a_peer_under_the_target_name_never_reaches_comps(self) -> None:
        # A provider that does not drop the target's other share class: the
        # engine drops it by name (case and punctuation aside).
        stub = _Stub(peers=[("SYNTB", "SYNTHETIC CORP."), ("P1", "Peer One"),
                            ("P2", "Peer Two"), ("SYNT", "Synthetic Corp")])
        report = value_company(DEMO_TICKER, provider=stub, auto_peers=True)
        self.assertEqual([r.ticker for r in report.comps.peers], ["P1", "P2"])
        self.assertIn("Comps: peers discovered automatically from a stub list: P1, P2; "
                      "review them.", report.warnings)

    def test_explicit_peers_or_no_comps_skip_discovery(self) -> None:
        stub = _Stub()
        report = value_company(DEMO_TICKER, provider=stub, auto_peers=True, peers=["PEER1"])
        self.assertEqual([r.ticker for r in report.comps.peers], ["PEER1"])
        value_company(DEMO_TICKER, provider=stub, auto_peers=True, run_comps=False)
        self.assertEqual(stub.calls, [])

    def test_no_peers_discovered_keeps_the_notes(self) -> None:
        stub = _Stub(peers=[], notes=["No peers discovered: Yahoo gives no industry for SYNT."])
        report = value_company(DEMO_TICKER, provider=stub, auto_peers=True)
        self.assertEqual(report.comps.peers, [])
        self.assertIn("Comps: No peers discovered: Yahoo gives no industry for SYNT.",
                      report.warnings)
        self.assertIn("Comps: no usable peers found (pass --peers to supply them).",
                      report.warnings)
        self.assertFalse(any("discovered automatically" in w for w in report.warnings))

    def test_discovered_peers_without_multiples_say_so(self) -> None:
        report = value_company(DEMO_TICKER, provider=_Stub(rows=False), auto_peers=True)
        self.assertIn("Comps: none of the peers discovered (P1, P2) returned usable "
                      "multiples; supply peers with --peers.", report.warnings)

    def test_provider_without_discovery_uses_its_suggestions(self) -> None:
        # SyntheticProvider keeps the DataProvider default, which reads
        # suggest_peers: its three synthetic peers.
        report = value_company(DEMO_TICKER, provider=SyntheticProvider(), auto_peers=True)
        self.assertEqual([r.ticker for r in report.comps.peers], DEMO_PEERS)
        self.assertIn("Comps: peers discovered automatically from the data provider: "
                      "PEER1, PEER2, PEER3; review them.", report.warnings)
        pinned = value_company(DEMO_TICKER, provider=SyntheticProvider(), peers=DEMO_PEERS)
        self.assertEqual(report.summary, pinned.summary)

    def test_data_provider_default_reads_suggest_peers(self) -> None:
        provider = SyntheticProvider()
        self.assertEqual(provider.discover_peers("SYNT", industry_key="k", name="N"),
                         {"ticker": "SYNT", "notes": [], "source": "the data provider",
                          "peers": [{"ticker": t} for t in DEMO_PEERS]})
        self.assertEqual(provider.discover_peers("SYNT", limit=2)["peers"],
                         [{"ticker": "PEER1"}, {"ticker": "PEER2"}])
        self.assertEqual(provider.discover_peers("SYNT", limit=0)["peers"], [])
        with mock.patch.object(config, "PEER_DISCOVERY_LIMIT", 1):
            self.assertEqual(provider.discover_peers("SYNT")["peers"], [{"ticker": "PEER1"}])
        self.assertEqual(_NoSuggestion().discover_peers("SYNT")["peers"], [])

    def test_duck_typed_provider_without_discovery_uses_its_suggestions(self) -> None:
        # Not a DataProvider and no discover_peers: the engine reads its
        # suggest_peers list the way the DataProvider default does.
        class Duck:
            def __init__(self) -> None:
                self._base = SyntheticProvider()

            def __getattr__(self, attr):
                if attr == "discover_peers":
                    raise AttributeError(attr)
                return getattr(self._base, attr)

        report = value_company(DEMO_TICKER, provider=Duck(), auto_peers=True)
        self.assertEqual([r.ticker for r in report.comps.peers], DEMO_PEERS)
        self.assertIn("Comps: peers discovered automatically from the data provider: "
                      "PEER1, PEER2, PEER3; review them.", report.warnings)

    def test_discovered_peers_that_give_no_price_say_discovered(self) -> None:
        # A lessor is valued on P/E and P/B only, so peers with EV multiples
        # only give rows but no price. The peers were not the user's, so the
        # wording says "discovered" and still asks for peers.
        lessor = _with_kind(make_company(), "lessor")
        ev_only = [CompRow(ticker=t, name=t, market_cap=5e10, enterprise_value=5.2e10,
                           ev_ebitda=18.0 + i, ev_sales=4.0)
                   for i, t in enumerate(["P1", "P2", "P3"])]
        stub = _Stub(peers=["P1", "P2", "P3"], rows=ev_only, company=lessor)
        report = value_company("LESS", provider=stub, auto_peers=True)
        self.assertTrue(report.comps.peers)
        self.assertIsNone(report.summary["blended_target"])
        self.assertIn("No blended target: the DCF and FCFE are left out for a debt-funded "
                      "lessor; the DDM is shown for reference only, and the peers discovered "
                      "give no usable comps price. Supply peers with --peers to get a target.",
                      report.warnings)
        # A high-payout lessor's DDM, left out because it would stand alone.
        net_income = lessor.financials.net_income[-1]
        high = replace(lessor, market=replace(lessor.market,
                                              dividend_per_share=0.87 * net_income / 1e10))
        report = value_company("LESS", provider=_Stub(peers=["P1", "P2", "P3"], rows=ev_only,
                                                      company=high), auto_peers=True)
        self.assertEqual(report.summary["excluded_from_blend"]["DDM"],
                         "dividends only and no comps price from the peers discovered to "
                         "check it; supply peers")
        # A bank whose blend is the DDM alone.
        bank = _with_kind(make_company(), "bank")
        report = value_company("BANK", provider=_Stub(peers=["P1", "P2", "P3"], rows=ev_only,
                                                      company=bank),
                               auto_peers=True, run_dcf=False)
        self.assertEqual(report.summary["recommendation"], "N/A")
        self.assertIn("Blended target rests on the DDM alone (FCFE left out for a financial "
                      "institution; no usable comps price from the peers discovered), so no "
                      "verdict or upside is given: the DDM counts dividends only (regular plus "
                      "any recurring variable dividends; buybacks and one-off specials are "
                      "ignored) and there is no second method to check it against. Supply "
                      "peers to add trading comps.", report.warnings)
        self.assertFalse(any("peers supplied" in w for w in report.warnings), report.warnings)

    def test_supplied_peers_that_give_no_price_keep_their_wording(self) -> None:
        lessor = _with_kind(make_company(), "lessor")
        ev_only = [CompRow(ticker=t, name=t, market_cap=5e10, enterprise_value=5.2e10,
                           ev_ebitda=18.0, ev_sales=4.0) for t in ["P1", "P2"]]
        stub = _Stub(rows=ev_only, company=lessor)
        report = value_company("LESS", provider=stub, auto_peers=True, peers=["P1", "P2"])
        self.assertEqual(stub.calls, [])
        self.assertIn("No blended target: the DCF and FCFE are left out for a debt-funded "
                      "lessor; the DDM is shown for reference only, and the peers supplied give "
                      "no usable comps price.", report.warnings)

    def test_discovery_error_never_sinks_the_run(self) -> None:
        class Broken(_NoSuggestion):
            def discover_peers(self, ticker, *, industry_key=None, limit=None, name=None):
                raise RuntimeError("list down")

        report = value_company(DEMO_TICKER, provider=Broken(), auto_peers=True)
        self.assertIsNotNone(report.comps)
        self.assertIn("Comps: No peers discovered: peer discovery failed (list down).",
                      report.warnings)
        self.assertFalse(any(w.startswith("Comps failed") for w in report.warnings))

    def test_hybrid_provider_end_to_end(self) -> None:
        market_mod._peer_cache.clear()
        self.addCleanup(market_mod._peer_cache.clear)
        company = make_company()
        company = replace(company, ticker="KO",
                          market=replace(company.market, ticker="KO", industry_key=KEY))
        hybrid = HybridProvider(edgar=mock.Mock(), market=client())
        hybrid.get_company_data = mock.Mock(return_value=company)
        hybrid.get_peer_comp_rows = SyntheticProvider().get_peer_comp_rows
        report = value_company("KO", provider=hybrid, auto_peers=True, run_sensitivity=False,
                               run_reverse_dcf=False)
        self.assertEqual([r.ticker for r in report.comps.peers], ["PEP", "MNST", "KDP"])
        self.assertIn("Comps: peers discovered automatically from Yahoo Finance industry "
                      "list: PEP, MNST, KDP, COKE; review them.", report.warnings)
        self.assertTrue(any(w.startswith("Comps: Size band widened") for w in report.warnings))

    def test_hybrid_provider_keeps_the_targets_other_class_out_of_comps(self) -> None:
        # GOOGL is valued against Yahoo's list, which holds GOOG (the same
        # company) but not GOOGL: GOOG must never be asked for as a peer.
        market_mod._peer_cache.clear()
        self.addCleanup(market_mod._peer_cache.clear)
        company = make_company()
        company = replace(company, ticker="GOOGL", name="Alphabet Inc.",
                          market=replace(company.market, ticker="GOOGL", name="Alphabet Inc.",
                                         industry_key=GOOG_KEY))
        hybrid = HybridProvider(edgar=mock.Mock(), market=client(industry(GOOG_ROWS)))
        hybrid.get_company_data = mock.Mock(return_value=company)
        hybrid.get_peer_comp_rows = mock.Mock(
            side_effect=SyntheticProvider().get_peer_comp_rows)
        report = value_company("GOOGL", provider=hybrid, auto_peers=True,
                               run_sensitivity=False, run_reverse_dcf=False)
        requested = [t for call in hybrid.get_peer_comp_rows.call_args_list
                     for t in call.args[0]]
        self.assertEqual(requested, ["META", "SPOT"])
        self.assertEqual([r.ticker for r in report.comps.peers], ["META", "SPOT"])
        self.assertNotIn("GOOG", [r.ticker for r in report.comps.peers])
        self.assertIn("Comps: GOOG is not proposed: Yahoo lists Alphabet Inc. under it, another "
                      "share class of GOOGL; GOOG's market weight sets the size band.",
                      report.warnings)
        self.assertFalse(any("not applied" in w for w in report.warnings), report.warnings)


def _run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = main(argv)
        except SystemExit as exc:
            code = exc.code
    return code, out.getvalue(), err.getvalue()


class CliAutoPeersTests(unittest.TestCase):
    def _kwargs(self, argv):
        with mock.patch("equity_valuation.engine.value_company",
                        side_effect=RuntimeError("stop")) as vc:
            code, _, err = _run_cli(argv + ["--quiet"])
        self.assertEqual(code, 1)
        return vc.call_args.kwargs, err

    def test_flag_is_forwarded(self) -> None:
        kwargs, _ = self._kwargs(["KO", "--auto-peers"])
        self.assertIs(kwargs["auto_peers"], True)
        self.assertIsNone(kwargs["peers"])

    def test_off_by_default(self) -> None:
        kwargs, err = self._kwargs(["KO"])
        self.assertIs(kwargs["auto_peers"], False)
        self.assertNotIn("--auto-peers", err)

    def test_ignored_flags_are_noted(self) -> None:
        _, err = self._kwargs(["KO", "--auto-peers", "--peers", "PEP"])
        self.assertIn("note: --auto-peers is ignored when --peers is given.", err)
        _, err = self._kwargs(["KO", "--auto-peers", "--no-comps"])
        self.assertIn("note: --auto-peers is ignored with --no-comps.", err)

    def test_help_text(self) -> None:
        text = build_parser().format_help()
        self.assertIn("--auto-peers", text)
        self.assertIn("Yahoo's industry list", text)

    def test_demo_discovers_its_synthetic_peers_offline(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            code, stdout, _ = _run_cli(["--demo", "--auto-peers", "--html", "--out", out])
        self.assertEqual(code, 0)
        self.assertIn("Comps: peers discovered automatically from the data provider: "
                      "PEER1, PEER2, PEER3; review them.", stdout)
        self.assertIn("$32.40", stdout)  # the pinned demo blend is unchanged


if __name__ == "__main__":
    unittest.main()
