"""Exports must reproduce the displayed valuation without refreshing inputs."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from docx import Document
from fastapi.testclient import TestClient

from backend import app as api, exports, valuation_service as vs
from backend.serialization import report_to_dict
from equity_valuation import value_company
from equity_valuation.data.synthetic import DEMO_TICKER, SyntheticProvider


class ReportSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.report = value_company(DEMO_TICKER, provider=SyntheticProvider())
        self.echo = {'forecast_years': 5, 'terminal_growth': 0.025}
        with vs._snapshot_lock:
            vs._report_snapshots.clear()

    def tearDown(self):
        with vs._snapshot_lock:
            vs._report_snapshots.clear()

    def test_api_returns_snapshot_and_timestamp(self):
        with patch.object(vs, '_get_provider', return_value=SyntheticProvider()):
            response = TestClient(api.app, base_url='http://127.0.0.1').post(
                '/api/valuation', json={'ticker': DEMO_TICKER, 'run_sensitivity': False})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        stored, echo = vs.get_report_snapshot(DEMO_TICKER, data['report_id'])
        self.assertEqual(stored.current_price, data['current_price'])
        self.assertEqual(stored.summary, data['summary'])
        self.assertEqual(data['assumptions_used'], echo)
        self.assertTrue(data['valued_at'].endswith('+00:00'))

    def test_snapshot_isolated_from_source_and_callers(self):
        token = vs._remember_report(DEMO_TICKER, self.report, self.echo)
        self.report.summary['blended_target'] = -123
        self.echo['forecast_years'] = 99
        first, echo = vs.get_report_snapshot(DEMO_TICKER.lower(), token)
        self.assertNotEqual(first.summary['blended_target'], -123)
        self.assertEqual(echo['forecast_years'], 5)
        first.summary['blended_target'] = -456
        again, _ = vs.get_report_snapshot(DEMO_TICKER, token)
        self.assertNotEqual(again.summary['blended_target'], -456)

    def test_snapshot_expiry_and_capacity(self):
        with patch.object(vs, '_SNAPSHOT_LIMIT', 2), patch.object(vs.time, 'monotonic', return_value=10):
            tokens = [vs._remember_report(DEMO_TICKER, self.report, self.echo) for _ in range(3)]
        with patch.object(vs.time, 'monotonic', return_value=11):
            with self.assertRaises(vs.SnapshotError):
                vs.get_report_snapshot(DEMO_TICKER, tokens[0])
            vs.get_report_snapshot(DEMO_TICKER, tokens[-1])
        with patch.object(vs.time, 'monotonic', return_value=10 + vs._SNAPSHOT_TTL):
            with self.assertRaises(vs.SnapshotError):
                vs.get_report_snapshot(DEMO_TICKER, tokens[-1])

    def test_ticker_mismatch_and_invalid_ids_do_not_recompute(self):
        token = vs._remember_report(DEMO_TICKER, self.report, self.echo)
        for ticker, report_id in [('OTHER', token), (DEMO_TICKER, None), (DEMO_TICKER, []), (DEMO_TICKER, 'missing')]:
            with self.subTest(ticker=ticker, report_id=report_id), self.assertRaises(vs.SnapshotError):
                exports._export_report(ticker, {'report_id': report_id})

    def test_all_exporters_use_exact_snapshot_even_with_changed_payload(self):
        token = vs._remember_report(DEMO_TICKER, self.report, self.echo)
        with tempfile.TemporaryDirectory() as tmp, patch.object(exports, '_OUT_DIR', Path(tmp)), \
             patch.object(exports, 'run_valuation_report', side_effect=AssertionError('No refetch allowed')):
            for kind, (fn, _) in api._EXPORTERS.items():
                with self.subTest(kind=kind):
                    path = fn(DEMO_TICKER, {'report_id': token, 'rf': 0.9, 'refresh': True})
                    self.assertGreater(Path(path).stat().st_size, 100)
            text = (Path(tmp) / 'SYNT_valuation.html').read_text()
            self.assertIn('40.84', text)
            # The snapshot's reverse DCF, as displayed, in the HTML and the memo.
            self.assertIn('Year-1 revenue growth</th><td>8.0%</td><td>17.5%</td>', text)
            memo = '\n'.join(p.text for p in
                             Document(Path(tmp) / 'SYNT_research_memo.docx').paragraphs)
            self.assertIn('The price implies 17.5% year-1 growth (model 8.0%)', memo)

    def test_expired_export_returns_conflict_instead_of_new_value(self):
        client = TestClient(api.app, base_url='http://127.0.0.1')
        with patch.object(exports, 'run_valuation_report', side_effect=AssertionError('No refetch allowed')):
            response = client.post('/api/export/excel', json={'ticker': DEMO_TICKER, 'report_id': 'expired'})
        self.assertEqual(response.status_code, 409)
        self.assertIn('Recompute', response.json()['detail'])

    def test_programmatic_export_without_id_still_supported(self):
        with patch.object(exports, 'run_valuation_report', return_value=(self.report, self.echo)) as run:
            report, echo = exports._export_report(DEMO_TICKER, {'rf': 0.04})
        self.assertIs(report, self.report)
        run.assert_called_once_with(DEMO_TICKER, {'rf': 0.04})

    def test_unavailable_model_prices_are_null_but_completed_zero_is_zero(self):
        for name, meta in [('dcf', 'assumptions'), ('fcfe', 'detail')]:
            with self.subTest(name=name):
                report = copy.deepcopy(self.report)
                model = getattr(report, name)
                model.implied_price = 0.0
                report.summary['methods'][name.upper()] = 0.0
                getattr(model, meta)['valuation_available'] = False
                self.assertIsNone(report_to_dict(report)[name]['implied_price'])
                self.assertIsNone(report_to_dict(report)['summary']['methods'][name.upper()])
                self.assertIsNone(exports._method_values(report)[name.upper()])
                getattr(model, meta)['valuation_available'] = True
                self.assertEqual(report_to_dict(report)[name]['implied_price'], 0.0)
                self.assertEqual(report_to_dict(report)['summary']['methods'][name.upper()], 0.0)
                self.assertEqual(exports._method_values(report)[name.upper()], 0.0)


class AIContextPeriodTests(unittest.TestCase):
    def test_cagr_uses_elapsed_fiscal_years(self):
        from backend.serialization import build_ai_context
        context = build_ai_context({'company': {'financials': {
            'fiscal_years': [2020, 2024], 'revenue': [100, 146.41]}}})
        self.assertIn('HISTORICAL REVENUE CAGR (4y): 10.0%', context)
