"""Offline checks for the demo run, every export format, and the reverse DCF.

All tests use the synthetic company from ``equity_valuation.data.synthetic``,
so none of them touch the network. Run with:  python -m unittest tests.test_exports
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from equity_valuation import value_company
from equity_valuation.data.synthetic import DEMO_PEERS, DEMO_TICKER, SyntheticProvider


def _demo_report():
    return value_company(DEMO_TICKER, provider=SyntheticProvider(), peers=DEMO_PEERS)


class DemoResultTests(unittest.TestCase):
    def test_summary_matches_readme_results(self):
        # The README "Results" table quotes these figures; keep them in step.
        s = _demo_report().summary
        self.assertAlmostEqual(s["current_price"], 40.84, delta=0.005)
        expected = {"DCF": 33.38, "Comps (median)": 42.88, "DDM": 10.99, "FCFE": 31.41}
        self.assertEqual(set(s["methods"]), set(expected))
        for method, price in expected.items():
            self.assertAlmostEqual(s["methods"][method], price, delta=0.005, msg=method)
        self.assertAlmostEqual(s["blended_target"], 32.40, delta=0.005)
        self.assertAlmostEqual(s["blended_upside"], -0.207, delta=0.0005)
        self.assertEqual(s["recommendation"], "Overvalued")

    def test_cli_demo_writes_excel_and_html(self):
        from equity_valuation.cli import main

        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--demo", "--quiet", "--out", tmp]), 0)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "SYNT_valuation.xlsx")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "SYNT_valuation.html")))

    def test_cli_demo_rejects_a_live_ticker(self):
        from equity_valuation.cli import main

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["AAPL", "--demo"])


class EngineExportTests(unittest.TestCase):
    def test_excel_keeps_formulas_and_html_renders(self):
        from openpyxl import load_workbook

        from equity_valuation.report.excel import write_excel
        from equity_valuation.report.html import write_html

        report = _demo_report()
        with tempfile.TemporaryDirectory() as tmp:
            xlsx = write_excel(report, os.path.join(tmp, "SYNT_valuation.xlsx"))
            html = write_html(report, os.path.join(tmp, "SYNT_valuation.html"))

            wb = load_workbook(xlsx)
            for sheet in ("Summary", "DCF", "Comps", "DDM_FCFE", "Sensitivity"):
                self.assertIn(sheet, wb.sheetnames)
            formulas = [
                c for row in wb["DCF"].iter_rows() for c in row
                if isinstance(c.value, str) and c.value.startswith("=")
            ]
            self.assertTrue(formulas, "DCF sheet should contain live formulas")

            text = Path(html).read_text(encoding="utf-8")
            self.assertIn("Synthetic Corp", text)
            self.assertIn("<html", text.lower())


class OfficeExportTests(unittest.TestCase):
    """The dashboard's Word memo and PowerPoint briefing, without a live provider."""

    def test_memo_and_deck(self):
        from docx import Document
        from pptx import Presentation

        from backend import exports, valuation_service

        payload = {"peers": ",".join(DEMO_PEERS)}
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            valuation_service, "_get_provider",
            lambda ticker, refresh=False: SyntheticProvider(),
        ), patch.object(exports, "_ensure_out", lambda: Path(tmp)):
            memo = exports.export_memo(DEMO_TICKER, payload)
            deck = exports.export_deck(DEMO_TICKER, payload)

            self.assertEqual(Path(memo).parent, Path(tmp))
            text = "\n".join(p.text for p in Document(memo).paragraphs)
            self.assertIn("Synthetic Corp", text)
            self.assertIn("Overvalued", text)
            self.assertGreaterEqual(len(Presentation(deck).slides), 1)


class SerializationTests(unittest.TestCase):
    """The dashboard payload and the AI context for an adjusted beta and for a
    summary with no blended target or a withheld upside."""

    def _adjusted_report(self):
        from equity_valuation.data.synthetic import make_company

        class _Adjusted(SyntheticProvider):
            def get_company_data(self, ticker):
                company = make_company()
                company.market.raw_beta = 2.217
                company.market.beta = 0.67 * 2.217 + 0.33
                return company

        return value_company(DEMO_TICKER, provider=_Adjusted(), peers=DEMO_PEERS)

    def test_raw_beta_reaches_the_payload_and_the_ai_context(self):
        from backend.serialization import build_ai_context, report_to_dict

        d = report_to_dict(self._adjusted_report())
        self.assertEqual(d["company"]["market"]["raw_beta"], 2.217)
        self.assertAlmostEqual(d["company"]["market"]["beta"], 1.81539, places=9)
        self.assertEqual(d["dcf"]["wacc"]["detail"]["beta_raw"], 2.217)
        ctx = build_ai_context(d)
        self.assertIn("beta 1.815 (Blume-adjusted from raw 2.217)", ctx)
        self.assertEqual(ctx.count("Blume-adjusted from raw 2.217"), 2)  # PRICE and DCF

        # The synthetic beta is not adjusted: raw_beta is null, no adjustment is claimed.
        d = report_to_dict(_demo_report())
        self.assertIsNone(d["company"]["market"]["raw_beta"])
        ctx = build_ai_context(d)
        self.assertIn("beta 1.100 |", ctx)
        self.assertNotIn("Blume", ctx)

    @staticmethod
    def _kind_payload(kind=None, industry=None):
        """The dashboard payload of the engine's run for the synthetic company
        flagged as that kind (or presented as a bank), without peers."""
        from backend.serialization import report_to_dict
        from equity_valuation.data.synthetic import make_company

        class _Kind(SyntheticProvider):
            def get_company_data(self, ticker):
                company = make_company()
                if kind:
                    company.financials._financial_kind = kind
                if industry:
                    company.market.sector = "Financial Services"
                    company.market.industry = industry
                return company

        return report_to_dict(value_company(DEMO_TICKER, provider=_Kind(), run_comps=False))

    @staticmethod
    def _line(ctx, prefix):
        return next(ln for ln in ctx.splitlines() if ln.startswith(prefix))

    def test_ai_context_says_when_there_is_no_target_or_no_upside(self):
        from backend.serialization import build_ai_context, report_to_dict

        ctx = build_ai_context(report_to_dict(_demo_report()))
        self.assertIn("VERDICT: Overvalued | blended target $32.40 (-20.7% vs price)", ctx)
        # The demo's DCF is in the blend: a signed upside, no marker.
        dcf = self._line(ctx, "DCF: ")
        self.assertTrue(dcf.endswith("implied $33.38 (-18.3% vs price)"), dcf)
        self.assertNotIn("reference only", dcf)

        # A lessor without peers (as AER): no target, every method reference only.
        d = self._kind_payload("lessor")
        excluded = d["summary"]["excluded_from_blend"]
        self.assertEqual(set(excluded), {"DCF", "DDM", "FCFE"})
        ctx = build_ai_context(d)
        self.assertIn("VERDICT: N/A | blended target: none (no target; supply peers to add "
                      "trading comps)", ctx)
        for name in ("DCF", "DDM", "FCFE"):
            self.assertIn(f"{name} ${d['summary']['methods'][name]:,.2f} (reference only, "
                          f"not in blend: {excluded[name]})", ctx)
        # The DCF line is marked too, so its upside is not quoted as the view.
        dcf = self._line(ctx, "DCF: ")
        self.assertTrue(dcf.endswith(f"(-18.3% vs price) (reference only, not in blend: "
                                     f"{excluded['DCF']})"), dcf)

        # A bank without peers (as BAC): the DDM-only target stays, the verdict
        # is withheld; the DDM is in the blend, the DCF is not.
        d = self._kind_payload(industry="Banks - Diversified")
        ctx = build_ai_context(d)
        self.assertIn("VERDICT: N/A | blended target $10.99 (upside withheld: n/a)", ctx)
        self.assertIn("DDM $10.99;", ctx)
        self.assertIn("(reference only, not in blend: not meaningful for a financial "
                      "institution)", self._line(ctx, "DCF: "))

        # Hand-built (the engine needs a price): no upside because there is no
        # current price is not a withheld verdict.
        d["summary"].update(current_price=None)
        self.assertIn("VERDICT: N/A | blended target $10.99 (upside n/a: no current price)",
                      build_ai_context(d))
        # A positive upside carries its sign.
        d = report_to_dict(_demo_report())
        d["summary"].update(blended_target=49.0, blended_upside=0.2)
        self.assertIn("blended target $49.00 (+20.0% vs price)", build_ai_context(d))
        # An ordinary company whose methods gave no valuation: no peers asked for.
        d["summary"].update(methods={"DCF": 0.0}, blended_target=None, blended_upside=None,
                            recommendation="N/A", financial_kind=None,
                            financial_institution=None)
        d["comps"] = None
        self.assertIn("VERDICT: N/A | blended target: n/a", build_ai_context(d))


class ReverseDCFTests(unittest.TestCase):
    def test_solved_growth_reprices_to_market(self):
        from backend import valuation_service as vs
        from equity_valuation.models.dcf import run_dcf

        macro, dcf_a, ddm_a, _peers, _toggles, _echo = vs.parse_assumptions({})
        report = value_company(
            DEMO_TICKER, provider=SyntheticProvider(), macro=macro,
            dcf_assumptions=dcf_a, ddm_assumptions=ddm_a, peers=DEMO_PEERS,
        )
        result = vs._reverse_dcf(report, dcf_a, macro)
        self.assertTrue(result["converged"])
        g1 = result["implied_growth_y1"]
        # The synthetic firm grew 8% a year; a 20x P/E price needs more than that.
        self.assertGreater(g1, 0.08)

        path = vs._revenue_growth_path(g1, dcf_a.terminal_growth, dcf_a.forecast_years)
        solved = dataclasses.replace(dcf_a, revenue_growth=path)
        price = run_dcf(report.company, macro, solved, report.current_price).implied_price
        self.assertAlmostEqual(price, report.current_price, delta=0.01)


if __name__ == "__main__":
    unittest.main()
