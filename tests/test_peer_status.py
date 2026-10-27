"""Offline tests for the per-peer resolution status in comps.

``run_comps`` records what it did with every requested peer in
``CompsResult.peer_status`` ('used', 'not found', 'no usable multiple',
'is target', 'not fetched') and names the peers it could not use in one
"Peers not used: ..." note, which the engine lists among the report warnings
when comps ran on the other peers. The status is read only from the rows the
provider returned (one ``get_peer_comp_rows`` call, no extra lookups), and the
statistics and the pinned demo figures are unchanged. Covered: an unknown
ticker next to the synthetic peers, the target in its own peer list (in
either class-share spelling), a row without a usable multiple, a failed
fetch, the HybridProvider's class-share retry, a provider that renames its
rows, the backward-compatible ``_collect_peer_rows`` signature, the CLI
notes and the backend JSON. No network access is needed.
Run with:  python -m unittest tests.test_peer_status
"""

from __future__ import annotations

import contextlib
from dataclasses import replace
import io
import json
import unittest
from unittest import mock

from equity_valuation import value_company
from equity_valuation.cli import _print_summary
from equity_valuation.data.base import DataError
from equity_valuation.data.provider import HybridProvider
from equity_valuation.data.synthetic import (
    DEMO_PEERS,
    DEMO_TICKER,
    SyntheticProvider,
    make_company,
)
from equity_valuation.models.comps import (
    PEER_FETCH_FAILED_NOTE_PREFIX,
    PEER_IS_TARGET,
    PEER_NO_USABLE_MULTIPLE,
    PEER_NOT_FETCHED,
    PEER_NOT_FOUND,
    PEER_USED,
    PEERS_NOT_USED_NOTE_PREFIX,
    _collect_peer_rows,
    run_comps,
)
from equity_valuation.schemas import CompRow, CompsResult
from tests.test_market_data import client, ticker

PRICE = 40.84  # the synthetic company's price (about 20x trailing EPS)


def _demo(peers=None, provider=None, **kwargs):
    return value_company(DEMO_TICKER, provider=provider or SyntheticProvider(),
                         peers=DEMO_PEERS if peers is None else peers, **kwargs)


def _not_used_notes(notes):
    return [n for n in notes or [] if str(n).startswith(PEERS_NOT_USED_NOTE_PREFIX)]


def _comps_warnings(report):
    return [w for w in report.warnings if w.startswith("Comps:")]


class _RowsProvider(SyntheticProvider):
    """The synthetic company with fixed peer rows; records each request."""

    def __init__(self, rows):
        self.rows = rows
        self.requests: list[list[str]] = []
        self.suggested = 0

    def get_peer_comp_rows(self, tickers):
        self.requests.append(list(tickers))
        return list(self.rows)

    def suggest_peers(self, ticker):
        self.suggested += 1
        return super().suggest_peers(ticker)


def _row(tk, **multiples):
    return CompRow(ticker=tk, name=tk, **multiples)


class DemoPeerStatusTests(unittest.TestCase):
    """The synthetic peers, with and without an unknown ticker."""

    def test_plain_demo_uses_every_peer_with_no_note_and_pinned_figures(self) -> None:
        report = _demo()
        self.assertEqual(report.comps.peer_status, {t: PEER_USED for t in DEMO_PEERS})
        self.assertEqual(list(report.comps.peer_status), DEMO_PEERS)  # request order
        self.assertEqual(_not_used_notes(report.comps.notes), [])
        self.assertFalse(any(PEERS_NOT_USED_NOTE_PREFIX in w for w in report.warnings))
        s = report.summary
        expected = {"DCF": 33.38, "Comps (median)": 46.96, "DDM": 10.99, "FCFE": 31.41}
        for method, price in expected.items():
            self.assertAlmostEqual(s["methods"][method], price, delta=0.005, msg=method)
        self.assertAlmostEqual(s["blended_target"], 32.40, delta=0.005)
        self.assertAlmostEqual(s["blended_upside"], -0.207, delta=0.0005)

    def test_unknown_ticker_is_not_found_and_named_without_moving_the_figures(self) -> None:
        base = _demo()
        report = _demo(DEMO_PEERS + ["xyznotreal"])
        comps = report.comps
        self.assertEqual(comps.peer_status, {"PEER1": PEER_USED, "PEER2": PEER_USED,
                                             "PEER3": PEER_USED, "XYZNOTREAL": PEER_NOT_FOUND})
        note = "Peers not used: XYZNOTREAL (not found)."
        self.assertEqual(_not_used_notes(comps.notes), [note])
        self.assertIn(f"Comps: {note}", report.warnings)
        # The comps figures are the three-peer demo's.
        self.assertEqual([r.ticker for r in comps.peers], DEMO_PEERS)
        self.assertEqual(comps.stats, base.comps.stats)
        self.assertEqual(comps.implied, base.comps.implied)
        self.assertEqual(comps.implied_price_summary, base.comps.implied_price_summary)
        self.assertEqual(report.summary["blended_target"], base.summary["blended_target"])
        self.assertEqual(report.summary["recommendation"], base.summary["recommendation"])

    def test_cli_notes_name_the_unused_peer(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _print_summary(_demo(DEMO_PEERS + ["XYZNOTREAL"]))
        self.assertIn("    - Comps: Peers not used: XYZNOTREAL (not found).", out.getvalue())

    def test_status_is_read_from_one_fetch(self) -> None:
        provider = _RowsProvider([_row("PEER1", pe=20.0)])
        result = run_comps(make_company(), provider, ["PEER1", "MISSING"], PRICE)
        self.assertEqual(provider.requests, [["PEER1", "MISSING"]])
        self.assertEqual(provider.suggested, 0)
        self.assertEqual(result.peer_status, {"PEER1": PEER_USED, "MISSING": PEER_NOT_FOUND})


class TargetInPeerListTests(unittest.TestCase):
    """The target among its own peers is 'is target' and never requested."""

    def test_target_is_flagged_and_left_out(self) -> None:
        provider = _RowsProvider([_row("PEER1", pe=20.0), _row("PEER2", pe=24.0)])
        result = run_comps(make_company(), provider, ["PEER1", " synt ", "PEER2"], PRICE)
        self.assertEqual(provider.requests, [["PEER1", "PEER2"]])
        self.assertEqual(result.peer_status, {"PEER1": PEER_USED, "SYNT": PEER_IS_TARGET,
                                              "PEER2": PEER_USED})
        self.assertEqual(_not_used_notes(result.notes), ["Peers not used: SYNT (is target)."])

    def test_engine_lists_the_target_note_next_to_the_blend(self) -> None:
        report = _demo(DEMO_PEERS + [DEMO_TICKER])
        self.assertEqual(report.comps.peer_status[DEMO_TICKER], PEER_IS_TARGET)
        self.assertIn("Comps: Peers not used: SYNT (is target).", report.warnings)
        self.assertAlmostEqual(report.summary["blended_target"], 32.40, delta=0.005)

    def test_only_the_target_gives_no_comps_and_says_why(self) -> None:
        provider = _RowsProvider([_row("PEER1", pe=20.0)])
        result = run_comps(make_company(), provider, ["synt"], PRICE)
        self.assertEqual(provider.requests, [])
        self.assertEqual(result.peers, [])
        self.assertEqual(result.peer_status, {"SYNT": PEER_IS_TARGET})
        self.assertEqual(_not_used_notes(result.notes), ["Peers not used: SYNT (is target)."])
        self.assertIn("No peer tickers available; comps could not be computed.", result.notes)

    def test_class_share_target_in_the_other_spelling(self) -> None:
        # BRK.B and BRK-B are one listing: the data layer retries a dotted
        # peer in the dashed form, so the target cannot be its own peer.
        for target, peer in (("BRK-B", "BRK.B"), ("BRK.B", "brk-b")):
            with self.subTest(target=target):
                company = replace(make_company(), ticker=target)
                provider = _RowsProvider([_row("BRK-B", pe=9.0), _row("PEER1", pe=20.0)])
                result = run_comps(company, provider, [peer, "PEER1"], PRICE)
                self.assertEqual(provider.requests, [["PEER1"]])
                self.assertEqual([r.ticker for r in result.peers], ["PEER1"])
                self.assertEqual(result.peer_status, {peer.upper(): PEER_IS_TARGET,
                                                      "PEER1": PEER_USED})
        # A provider that returns the target unasked still drops it, unlisted.
        provider = _RowsProvider([_row("SYNT", pe=9.0), _row("PEER1", pe=20.0)])
        result = run_comps(make_company(), provider, ["PEER1"], PRICE)
        self.assertEqual([r.ticker for r in result.peers], ["PEER1"])
        self.assertEqual(result.peer_status, {"PEER1": PEER_USED})


class UnusableRowTests(unittest.TestCase):
    """A row that came back without a positive multiple."""

    def test_row_without_a_usable_multiple(self) -> None:
        provider = _RowsProvider([
            _row("GOOD", ev_ebitda=18.0, pe=20.0),
            _row("EMPTY", market_cap=5e10),             # every multiple None
            _row("LOSS", market_cap=5e10, pe=-12.0),    # only a negative P/E
        ])
        result = run_comps(make_company(), provider, ["GOOD", "EMPTY", "LOSS", "TYPO"], PRICE)
        self.assertEqual([r.ticker for r in result.peers], ["GOOD"])
        self.assertEqual(result.peer_status, {
            "GOOD": PEER_USED, "EMPTY": PEER_NO_USABLE_MULTIPLE,
            "LOSS": PEER_NO_USABLE_MULTIPLE, "TYPO": PEER_NOT_FOUND})
        self.assertEqual(_not_used_notes(result.notes), [
            "Peers not used: EMPTY (no usable multiple), LOSS (no usable multiple), "
            "TYPO (not found)."])
        self.assertEqual(result.stats["pe"]["n"], 1)

    def test_no_usable_peer_keeps_the_status_and_the_engine_warning(self) -> None:
        provider = _RowsProvider([_row("EMPTY", market_cap=5e10)])
        report = _demo(["EMPTY", "TYPO"], provider=provider)
        self.assertEqual(report.comps.peers, [])
        self.assertEqual(report.comps.peer_status,
                         {"EMPTY": PEER_NO_USABLE_MULTIPLE, "TYPO": PEER_NOT_FOUND})
        self.assertEqual(_not_used_notes(report.comps.notes), [
            "Peers not used: EMPTY (no usable multiple), TYPO (not found)."])
        # The no-usable-peers warning names the peers; the reasons stay on comps.
        self.assertEqual(_comps_warnings(report), [
            "Comps: none of the peers supplied (EMPTY, TYPO) returned usable multiples; "
            "check the tickers or supply other peers."])

    def test_a_used_repeat_outranks_an_unusable_copy(self) -> None:
        provider = _RowsProvider([_row("DUP", market_cap=5e10), _row("dup", pe=20.0),
                                  _row(" DUP ", pe=30.0)])
        result = run_comps(make_company(), provider, ["DUP"], PRICE)
        self.assertEqual(len(result.peers), 1)  # one issuer, one vote
        self.assertEqual(result.peer_status, {"DUP": PEER_USED})
        self.assertEqual(_not_used_notes(result.notes), [])


class FetchAndProviderShapeTests(unittest.TestCase):
    """A failed fetch, no peers at all, and providers whose rows are renamed."""

    def test_failed_fetch_is_not_fetched_not_not_found(self) -> None:
        class _Down(SyntheticProvider):
            def get_peer_comp_rows(self, tickers):
                raise ConnectionError("network unreachable")

        report = _demo(["ZZZZQ", "SYNT", "QQQQZ"], provider=_Down())
        comps = report.comps
        self.assertEqual(comps.peer_status, {"ZZZZQ": PEER_NOT_FETCHED, "SYNT": PEER_IS_TARGET,
                                             "QQQQZ": PEER_NOT_FETCHED})
        self.assertIn(f"{PEER_FETCH_FAILED_NOTE_PREFIX}: network unreachable", comps.notes)
        # Only the target is named: the unfetched peers were never checked.
        self.assertEqual(_not_used_notes(comps.notes), ["Peers not used: SYNT (is target)."])
        self.assertEqual(_comps_warnings(report), [
            "Comps: peer data for ZZZZQ, QQQQZ could not be fetched (network unreachable), "
            "so comps give no price."])

    def test_no_peers_requested_gives_an_empty_status(self) -> None:
        provider = _RowsProvider([])
        provider.suggest_peers = lambda ticker: []
        for peers in (None, [], ["", "  "]):
            with self.subTest(peers=peers):
                result = run_comps(make_company(), provider, peers, PRICE)
                self.assertEqual(result.peer_status, {})
                self.assertEqual(_not_used_notes(result.notes), [])
        self.assertEqual(CompsResult(target=None, peers=[], stats={}, implied={},
                                     implied_price_summary={}).peer_status, {})

    def test_suggested_peers_get_a_status_too(self) -> None:
        result = run_comps(make_company(), SyntheticProvider(), None, PRICE)
        self.assertEqual(result.peer_status, {t: PEER_USED for t in DEMO_PEERS})

    def test_renamed_rows_are_listed_as_used(self) -> None:
        # A provider that answers under other tickers: the requested names had
        # no row of their own, and every peer row still has a status.
        provider = _RowsProvider([_row("P0", pe=20.0), _row("P1", pe=22.0)])
        result = run_comps(make_company(), provider, ["PEER1"], PRICE)
        self.assertEqual(list(result.peer_status.items()), [
            ("PEER1", PEER_NOT_FOUND), ("P0", PEER_USED), ("P1", PEER_USED)])
        self.assertEqual(_not_used_notes(result.notes), ["Peers not used: PEER1 (not found)."])

    def test_collect_peer_rows_is_backward_compatible(self) -> None:
        provider = _RowsProvider([_row("A", pe=20.0), _row("b", market_cap=1.0)])
        notes: list[str] = []
        rows = _collect_peer_rows(provider, ["A", "B"], "SYNT", notes)
        self.assertEqual([r.ticker for r in rows], ["A"])
        rejected: set = set()
        rows = _collect_peer_rows(provider, ["A", "B"], "SYNT", notes, rejected)
        self.assertEqual(([r.ticker for r in rows], rejected), (["A"], {"B"}))
        self.assertEqual(notes, [])


class HybridProviderStatusTests(unittest.TestCase):
    """The live provider's peer rows (mocked Yahoo and SEC directory)."""

    @staticmethod
    def _provider():
        used = {"currency": "USD", "financialCurrency": "USD", "marketCap": 1000.0,
                "trailingPE": 20.0}
        loss = {"currency": "USD", "financialCurrency": "USD", "marketCap": 1000.0,
                "trailingPE": -5.0}
        market = client({"BF.B": ticker({}), "BF-B": ticker(used), "KO": ticker(used),
                         "LOSSCO": ticker(loss), "XYZNOTREAL": ticker({})})

        def resolve(sym):
            if sym in ("BF-B", "KO"):
                return "0000000001", "Fixture Co"
            raise DataError(f"ticker {sym!r} not found in SEC EDGAR directory")

        edgar = mock.Mock()
        edgar.resolve_cik.side_effect = resolve
        return HybridProvider(edgar=edgar, market=market)

    def test_unresolved_alias_and_unusable_rows(self) -> None:
        result = run_comps(make_company(), self._provider(),
                           ["BF.B", "KO", "LOSSCO", "XYZNOTREAL"], PRICE)
        self.assertEqual(sorted(r.ticker for r in result.peers), ["BF-B", "KO"])
        self.assertEqual(result.peer_status, {
            "BF.B": PEER_USED,                    # retried and found as BF-B
            "KO": PEER_USED,
            "LOSSCO": PEER_NO_USABLE_MULTIPLE,    # a row with only a negative P/E
            "XYZNOTREAL": PEER_NOT_FOUND,         # Yahoo had no data
        })
        self.assertEqual(_not_used_notes(result.notes), [
            "Peers not used: LOSSCO (no usable multiple), XYZNOTREAL (not found)."])


class SerializationTests(unittest.TestCase):
    def test_backend_json_carries_the_peer_status(self) -> None:
        from backend.serialization import report_to_dict

        d = report_to_dict(_demo(DEMO_PEERS + ["XYZNOTREAL"]))
        self.assertEqual(d["comps"]["peer_status"], {
            "PEER1": "used", "PEER2": "used", "PEER3": "used", "XYZNOTREAL": "not found"})
        self.assertEqual(json.loads(json.dumps(d))["comps"]["peer_status"]["XYZNOTREAL"],
                         "not found")
        # A run without comps has no comps block at all.
        self.assertIsNone(report_to_dict(_demo(run_comps=False))["comps"])


if __name__ == "__main__":
    unittest.main()
