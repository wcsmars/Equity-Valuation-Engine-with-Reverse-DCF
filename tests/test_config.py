"""Offline checks for resolving the EDGAR contact from the environment or .env."""

import tempfile
import unittest
from pathlib import Path

from equity_valuation import config


class SecUserAgentTests(unittest.TestCase):
    def resolve(self, text, environ=None, encoding="utf-8"):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(text, encoding=encoding)
            return config._resolve_sec_user_agent(environ or {}, path)

    def test_environment_wins_over_env_file(self):
        self.assertEqual(
            self.resolve("SEC_USER_AGENT=file@example.com\n", {"SEC_USER_AGENT": " env@example.com "}),
            "env@example.com",
        )

    def test_missing_file_or_key_disables_edgar(self):
        self.assertEqual(config._resolve_sec_user_agent({}, Path("/nonexistent/.env")), "")
        self.assertEqual(self.resolve("OTHER=1\nSEC_USER_AGENT_X=nope\n"), "")

    def test_quoting_export_and_comments(self):
        cases = {
            "SEC_USER_AGENT='app one@example.com'\n": "app one@example.com",
            'export SEC_USER_AGENT="app two@example.com"\n': "app two@example.com",
            "SEC_USER_AGENT=app@example.com # contact\n": "app@example.com",
            "SEC_USER_AGENT='app three@example.com' # contact\n": "app three@example.com",
            'SEC_USER_AGENT="app \\"four\\"@example.com"\n': 'app "four"@example.com',
            "# SEC_USER_AGENT=commented@example.com\n": "",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.resolve(text), expected)

    def test_last_assignment_wins(self):
        text = "SEC_USER_AGENT=first@example.com\nSEC_USER_AGENT='last one@example.com'\n"
        self.assertEqual(self.resolve(text), "last one@example.com")

    def test_byte_order_mark_does_not_hide_the_key(self):
        self.assertEqual(
            self.resolve("SEC_USER_AGENT=bom@example.com\n", encoding="utf-8-sig"),
            "bom@example.com",
        )


if __name__ == "__main__":
    unittest.main()
