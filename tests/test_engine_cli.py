"""Regression tests for the engine and CLI fixes: the default-beta note without
a DCF, the no-usable-peers warning (including a failed peer fetch), the DDM
basis wording, blank-argument errors, the console currency marker and its map
pinned to the dashboard's.

Everything runs on the synthetic company (``equity_valuation.data.synthetic``)
with no network. Run with:  python -m unittest tests.test_engine_cli
"""

from __future__ import annotations

from dataclasses import replace
import contextlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from equity_valuation import config, value_company
from equity_valuation.cli import _print_summary, main
from equity_valuation.data.synthetic import SyntheticProvider, make_company


class _OneCompanyProvider(SyntheticProvider):
    """SyntheticProvider that serves a caller-supplied (perturbed) company and,
    when given, a fixed list of peer rows (``[]``: no usable peer)."""

    def __init__(self, company, rows=None, suggested=None):
        self._company = company
        self._rows = rows
        self._suggested = suggested

    def get_company_data(self, ticker):
        return self._company

    def get_peer_comp_rows(self, tickers):
        if self._rows is None:
            return super().get_peer_comp_rows(tickers)
        return list(self._rows)

    def suggest_peers(self, ticker):
        if self._suggested is None:
            return super().suggest_peers(ticker)
        return list(self._suggested)


def _with_market(company, **changes):
    return replace(company, market=replace(company.market, **changes))


def _with_kind(company, kind):
    """``company`` with the data layer's ``financials._financial_kind`` set."""
    fin = replace(company.financials)
    fin._financial_kind = kind
    return replace(company, financials=fin)


def _run_cli(argv):
    """(exit code, stdout, stderr) of cli.main, with SystemExit mapped to its code."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = main(argv)
        except SystemExit as exc:
            code = exc.code
    return code, out.getvalue(), err.getvalue()


def _beta_notes(report):
    return [w for w in report.warnings if "beta unavailable" in w]


class DefaultBetaNoteTests(unittest.TestCase):
    """The DDM and FCFE price at DEFAULT_BETA when the market beta is missing;
    the note must reach the warnings whether or not the DCF ran."""

    def setUp(self):
        self.no_beta = _with_market(make_company(), beta=None)
        self.provider = _OneCompanyProvider(self.no_beta)
        self.default_ke = config.DEFAULT_RISK_FREE_RATE + (
            config.DEFAULT_BETA * config.DEFAULT_EQUITY_RISK_PREMIUM)

    def test_dcf_off_keeps_the_default_beta_note(self):
        report = value_company("SYNT", provider=self.provider, run_dcf=False)
        self.assertIsNone(report.dcf)
        self.assertAlmostEqual(report.ddm.cost_of_equity, self.default_ke, places=12)
        self.assertEqual(_beta_notes(report), [
            f"CAPM: beta unavailable; using DEFAULT_BETA={config.DEFAULT_BETA} for the DDM "
            "and FCFE cost of equity"])

    def test_only_the_models_that_priced_are_named(self):
        report = value_company("SYNT", provider=self.provider, run_dcf=False, run_ddm=False)
        self.assertEqual(_beta_notes(report), [
            f"CAPM: beta unavailable; using DEFAULT_BETA={config.DEFAULT_BETA} for the FCFE "
            "cost of equity"])
        # A failed DCF (no revenue) also loses the WACC note; the FCFE that
        # returned placeholder zeros is not named.
        fin = self.no_beta.financials
        no_revenue = replace(self.no_beta, financials=replace(
            fin, revenue=[0.0] * len(fin.revenue)))
        report = value_company("SYNT", provider=_OneCompanyProvider(no_revenue),
                               run_comps=False)
        self.assertIsNone(report.dcf)
        self.assertTrue(any(w.startswith("DCF failed:") for w in report.warnings))
        self.assertEqual(_beta_notes(report), [
            f"CAPM: beta unavailable; using DEFAULT_BETA={config.DEFAULT_BETA} for the DDM "
            "cost of equity"])

    def test_no_extra_note_when_the_dcf_ran_or_the_beta_is_known(self):
        report = value_company("SYNT", provider=self.provider)
        self.assertIsNotNone(report.dcf)
        self.assertEqual(_beta_notes(report),
                         [f"WACC: beta unavailable; using DEFAULT_BETA={config.DEFAULT_BETA}"])
        report = value_company("SYNT", provider=SyntheticProvider(), run_dcf=False)
        self.assertEqual(_beta_notes(report), [])
        report = value_company("SYNT", provider=self.provider, run_dcf=False, run_ddm=False,
                               run_fcfe=False)
        self.assertEqual(_beta_notes(report), [])


class NoUsablePeersWarningTests(unittest.TestCase):
    """Peers that were supplied but gave no usable row are named; asking the
    user to pass --peers is kept for runs that had none."""

    def test_supplied_peers_are_named_not_asked_for_again(self):
        provider = _OneCompanyProvider(make_company(), rows=[])
        report = value_company("SYNT", provider=provider,
                               peers=["zzzzq", " QQQQZ ", "SYNT", "ZZZZQ", ""])
        self.assertFalse(report.comps.peers)
        comps = [w for w in report.warnings if w.startswith("Comps:")]
        self.assertEqual(comps, ["Comps: none of the peers supplied (ZZZZQ, QQQQZ) returned "
                                 "usable multiples; check the tickers or supply other peers."])

    def test_a_failed_peer_fetch_is_reported_not_blamed_on_the_tickers(self):
        class _Down(_OneCompanyProvider):
            def get_peer_comp_rows(self, tickers):
                raise ConnectionError("network unreachable")

        report = value_company("SYNT", provider=_Down(make_company()),
                               peers=["zzzzq", "QQQQZ"])
        self.assertFalse(report.comps.peers)
        self.assertIn("Failed to fetch peer comp rows: network unreachable", report.comps.notes)
        comps = [w for w in report.warnings if w.startswith("Comps:")]
        self.assertEqual(comps, ["Comps: peer data for ZZZZQ, QQQQZ could not be fetched "
                                 "(network unreachable), so comps give no price."])

        # Auto-suggested peers (none supplied) and an exception with no message.
        class _DownQuiet(_OneCompanyProvider):
            def get_peer_comp_rows(self, tickers):
                raise ConnectionError()

        report = value_company("SYNT", provider=_DownQuiet(make_company(), suggested=["PEER1"]))
        comps = [w for w in report.warnings if w.startswith("Comps:")]
        self.assertEqual(comps, ["Comps: peer data could not be fetched, so comps give no price."])

    def test_no_peers_still_asks_for_them(self):
        provider = _OneCompanyProvider(make_company(), rows=[], suggested=[])
        for peers in (None, [], ["SYNT"]):
            with self.subTest(peers=peers):
                report = value_company("SYNT", provider=provider, peers=peers)
                self.assertIn("Comps: no usable peers found (pass --peers to supply them).",
                              report.warnings)
                self.assertFalse([w for w in report.warnings
                                  if w.startswith("Comps: none of the peers supplied")])


class DdmBasisWordingTests(unittest.TestCase):
    """D0 holds the regular rate plus recurring variable dividends, so the
    DDM-alone warnings must not say special dividends are ignored."""

    BASIS = ("counts dividends only (regular plus any recurring variable dividends; "
             "buybacks and one-off specials are ignored)")

    def test_financial_ddm_alone_warning(self):
        bank = _with_kind(make_company(), "bank")
        report = value_company("BANK", provider=_OneCompanyProvider(bank), run_comps=False)
        self.assertEqual(report.summary["recommendation"], "N/A")
        texts = [w for w in report.warnings
                 if w.startswith("Blended target rests on the DDM alone")]
        self.assertEqual(len(texts), 1, report.warnings)
        self.assertIn(f"the DDM {self.BASIS} and there is no second method", texts[0])
        self.assertNotIn("regular dividends only", texts[0])

    def test_captive_ddm_alone_warning(self):
        captive = _with_kind(make_company(), "captive_finance")
        net_income = captive.financials.net_income[-1]
        pcar = _with_market(captive, dividend_per_share=0.87 * net_income / 1e10)
        report = value_company("PCAR", provider=_OneCompanyProvider(pcar), run_comps=False)
        self.assertIn("Captive finance arm: the DDM is shown for reference only and left out of "
                      "the blended target, which it would otherwise set on its own: it "
                      f"{self.BASIS} and no second method checks it.", report.warnings)
        self.assertFalse([w for w in report.warnings if "regular dividends only" in w])


class CliArgumentTests(unittest.TestCase):
    """A blank ticker or an empty --out is an argument error (exit 2) raised
    before any valuation runs."""

    def test_blank_ticker_and_empty_out_exit_2_before_valuing(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cases = [
            (["   ", "--quiet", "--out", tmp.name], "a ticker is required"),
            (["", "--quiet", "--out", tmp.name], "a ticker is required"),
            (["--demo", "--quiet", "--out", ""], "--out must not be empty"),
            (["AAPL", "--quiet", "--out", ""], "--out must not be empty"),
        ]

        def boom(*_args, **_kwargs):
            raise AssertionError("value_company must not run")

        for argv, hint in cases:
            with self.subTest(argv=argv), patch("equity_valuation.engine.value_company", boom):
                code, out, err = _run_cli(argv)
                self.assertEqual(code, 2, err)
                self.assertIn(hint, err)
                self.assertNotIn("Traceback", err)
                self.assertEqual(out, "")
        self.assertEqual(os.listdir(tmp.name), [])

    def test_version_flag(self):
        from equity_valuation import __version__

        def boom(*_args, **_kwargs):
            raise AssertionError("value_company must not run")

        with patch("equity_valuation.engine.value_company", boom):
            code, out, err = _run_cli(["--version"])
        self.assertEqual(code, 0, err)
        self.assertEqual(out.strip(), f"equity_valuation {__version__}")


class CurrencyMarkerTests(unittest.TestCase):
    """Every console price carries a unit: a glyph when mapped, else the ISO
    code."""

    def test_currency_symbol(self):
        self.assertEqual(config.currency_symbol("USD"), "$")
        self.assertEqual(config.currency_symbol("CAD"), "C$")
        self.assertEqual(config.currency_symbol("CHF"), "CHF ")
        self.assertEqual(config.currency_symbol("SEK"), "SEK ")
        self.assertEqual(config.currency_symbol(None), "")
        self.assertEqual(config.currency_symbol(""), "")

    def _summary(self, currency):
        company = _with_market(make_company(), currency=currency)
        report = value_company("SYNT", provider=_OneCompanyProvider(company))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _print_summary(report)
        return report, out.getvalue()

    def test_unmapped_currency_prints_its_iso_code(self):
        report, text = self._summary("SEK")
        price = report.summary["current_price"]
        target = report.summary["blended_target"]
        self.assertIn(f"Current price: SEK {price:,.2f}", text)
        self.assertIn(f"SEK {target:,.2f}", text)
        self.assertNotIn("$", text)

    def test_mapped_currency_keeps_its_glyph(self):
        report, text = self._summary("CAD")
        self.assertIn(f"Current price: C${report.summary['current_price']:,.2f}", text)
        report, text = self._summary("USD")
        self.assertIn(f"Current price: ${report.summary['current_price']:,.2f}", text)

    def test_yuan_and_yen_are_told_apart(self):
        self.assertEqual(config.currency_symbol("CNY"), "CN¥")
        self.assertEqual(config.currency_symbol("JPY"), "¥")

    FORMAT_TS = Path(__file__).resolve().parent.parent / "frontend" / "lib" / "format.ts"

    def test_map_matches_the_dashboard(self):
        """The dashboard's lib/format.ts SYMBOLS is a copy of CURRENCY_SYMBOLS."""
        if not self.FORMAT_TS.is_file():
            self.skipTest("no frontend in this tree")
        ts = self.FORMAT_TS.read_text(encoding="utf-8")
        block = re.search(r"const SYMBOLS: Record<string, string> = \{(.*?)\n\};", ts, re.S)
        self.assertIsNotNone(block, "SYMBOLS object not found in format.ts")
        body = re.sub(r"//[^\n]*", "", block.group(1))
        pairs = re.findall(r"^\s*(\w+):\s*(\"(?:[^\"\\]|\\.)*\"),?\s*$", body, re.M)
        entries = [line.strip() for line in body.splitlines() if line.strip()]
        self.assertEqual(len(pairs), len(entries), f"unparsed SYMBOLS lines: {entries}")
        self.assertEqual({code: json.loads(sym) for code, sym in pairs},
                         config.CURRENCY_SYMBOLS)


class ConfigTests(unittest.TestCase):
    def test_dead_peer_limit_constant_is_gone(self):
        # Peers are not suggested by default (opt-in discovery uses
        # config.PEER_DISCOVERY_LIMIT); the dashboard's cap on auto-fed peers
        # lives in the frontend.
        self.assertFalse(hasattr(config, "DEFAULT_PEER_LIMIT"))


if __name__ == "__main__":
    unittest.main()
