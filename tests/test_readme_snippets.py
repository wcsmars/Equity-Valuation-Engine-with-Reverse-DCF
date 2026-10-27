"""README drift checks: what the README shows must match the code.

The README is the first thing a reader runs, so these tests execute or verify
what it documents instead of trusting it to stay current:

  * The first ```python block under "## Programmatic use" runs (offline: it
    values the synthetic company) in a temporary directory, and what it prints
    must equal the ```text block that follows it.
  * The demo figures the README quotes (method prices and upsides, the blended
    target and verdict, the reverse-DCF drivers and the terminal reinvestment
    check) are the ones the engine computes.
  * Every ``python -m equity_valuation`` / ``equity-valuation`` command in a
    ```bash block passes the CLI's argument checks, and every option named in
    inline code exists. The engine and the exporters are stubbed, so no
    network or file outside a temporary directory is touched.
  * Every path in the "## Project layout" module map exists, every engine and
    backend module is listed there, and every module the "## Architecture"
    diagram names exists.
  * Rules the README states in words carry the values the code uses: the
    verdict bands, the comps minimum of peer values per multiple, the
    trailing-gap note threshold, the number of discovered peers, the
    low-payout DDM cut-off, the reverse-DCF tolerance, the Blume beta weights,
    the dividends the DDM counts and the split factors a likely spin-off is
    flagged for.

README.md is found next to the tests folder (``Path(__file__).resolve()
.parents[1]``), so the same file checks the README of whichever checkout it
runs in. Run with:  python -m unittest tests.test_readme_snippets
"""

from __future__ import annotations

import contextlib
import datetime
import functools
import io
import re
import shlex
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from equity_valuation import config, value_company
from equity_valuation.cli import build_parser, main
from equity_valuation.data.market import blume_adjusted_beta
from equity_valuation.data.provider import _STOCK_DIVIDEND_TOLERANCE, _corporate_action_warning
from equity_valuation.data.synthetic import DEMO_PEERS, DEMO_TICKER, SyntheticProvider
from equity_valuation.engine import _DDM_BASIS, _recommendation
from equity_valuation.models.comps import TTM_GAP_THRESHOLD
from equity_valuation.models.reverse_dcf import ACCEPT_TOL
from equity_valuation.utils import LOW_DDM_PAYOUT_SHARE

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

# Pinned in tests/test_exports.py and tests/test_reverse_dcf.py as well.
DEMO_BLENDED_TARGET = 32.40
DEMO_IMPLIED_GROWTH = 0.17470256313681604

# A shell line that runs the CLI: `python -m equity_valuation ...` (any python
# path) or the `equity-valuation` console script; group 1 holds the arguments.
CLI_COMMAND = re.compile(r"(?:^|\s)(?:\S*python[\d.]*\s+-m\s+equity_valuation|equity-valuation)(?=\s|$)(.*)$")
# Inline code that starts with a long option (`--comps-basis ttm`).
INLINE_OPTION = re.compile(r"`(--[a-z][a-z0-9{},-]*)[^`]*`")
# A module path in the architecture diagram (data/edgar.py, backend/app.py).
DIAGRAM_MODULE = re.compile(r"[A-Za-z_][\w/]*\.py\b")


@functools.lru_cache(maxsize=None)
def _readme() -> str:
    return README.read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """The body of the level-2 section ``## heading``, up to the next level-2
    heading outside a code fence."""
    lines = text.splitlines(keepends=True)
    start = None
    in_fence = False
    for i, line in enumerate(lines):
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line.startswith("## "):
            continue
        if start is not None:
            return "".join(lines[start:i])
        if line[3:].strip() == heading:
            start = i + 1
    if start is None:
        raise AssertionError(f"README.md has no '## {heading}' section")
    return "".join(lines[start:])


def _blocks(text: str) -> list[tuple[str, str]]:
    """Fenced code blocks in order, as (info string, body)."""
    blocks: list[tuple[str, str]] = []
    lang, body = None, []
    for line in text.splitlines(keepends=True):
        if line.startswith("```"):
            if lang is None:
                lang, body = line[3:].strip(), []
            else:
                blocks.append((lang, "".join(body)))
                lang = None
        elif lang is not None:
            body.append(line)
    return blocks


def _demo_report():
    return value_company(DEMO_TICKER, provider=SyntheticProvider(), peers=DEMO_PEERS)


@functools.lru_cache(maxsize=None)
def _run_programmatic_example() -> tuple[str, str, dict, tuple[str, ...]]:
    """Runs the first python block under "## Programmatic use" once, in a
    temporary working directory, and returns (code, expected output, the
    namespace it left, the files it wrote)."""
    blocks = _blocks(_section(_readme(), "Programmatic use"))
    langs = [lang for lang, _ in blocks]
    if "python" not in langs:
        raise AssertionError("'## Programmatic use' has no ```python block")
    i = langs.index("python")
    code = blocks[i][1]
    if i + 1 >= len(blocks) or blocks[i + 1][0] != "text":
        raise AssertionError("the ```python block under '## Programmatic use' must be "
                             "followed by a ```text block with its output")
    expected = blocks[i + 1][1]
    namespace: dict = {"__name__": "readme_programmatic_use"}
    out = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp), \
            contextlib.redirect_stdout(out):
        exec(compile(code, "README.md (Programmatic use)", "exec"), namespace)
        written = tuple(sorted(str(p.relative_to(tmp)) for p in Path(tmp).rglob("*")
                               if p.is_file()))
    namespace["__stdout__"] = out.getvalue()
    return code, expected, namespace, written


class ProgrammaticUseTests(unittest.TestCase):
    def test_example_runs_offline_and_values_the_demo(self):
        _, _, namespace, _ = _run_programmatic_example()
        report = namespace.get("report")
        self.assertIsNotNone(report, "the example should leave its ValuationReport in `report`")
        self.assertEqual(round(report.summary["blended_target"], 2), DEMO_BLENDED_TARGET)
        self.assertEqual(report.summary["recommendation"], "Overvalued")
        self.assertAlmostEqual(report.reverse_dcf["implied_growth_y1"], DEMO_IMPLIED_GROWTH,
                               places=9)

    def test_printed_output_matches_the_documented_output(self):
        _, expected, namespace, _ = _run_programmatic_example()
        self.assertEqual(namespace["__stdout__"].strip().splitlines(),
                         expected.strip().splitlines())
        self.assertIn("Overvalued", namespace["__stdout__"])

    def test_exports_written_by_the_example_land_in_its_directory(self):
        code, _, _, written = _run_programmatic_example()
        for writer, suffix in (("write_excel", ".xlsx"), ("write_html", ".html")):
            if writer in code:
                self.assertTrue(any(name.endswith(suffix) for name in written),
                                f"{writer} wrote nothing: {written}")


class QuotedDemoFigureTests(unittest.TestCase):
    def test_readme_quotes_the_figures_the_engine_computes(self):
        report = _demo_report()
        s, dcf, rev = report.summary, report.dcf, report.reverse_dcf
        price = s["current_price"]
        figures = [f"${price:.2f}", f"${s['blended_target']:.2f}",
                   f"{s['blended_upside']:+.1%}", s["recommendation"]]
        for method in ("DCF", "Comps (median)", "DDM", "FCFE"):
            implied = s["methods"][method]
            figures += [f"${implied:.2f}", f"{implied / price - 1:+.1%}"]
        figures += [
            f"{dcf.wacc.wacc:.1%}",
            f"{dcf.pv_terminal / dcf.enterprise_value:.0%}",
            f"{rev['implied_growth_y1']:.1%}",
            f"{rev['margin']['implied_target_ebit_margin']:.1%}",
            f"{rev['wacc']['implied_wacc']:.1%}",
            f"{dcf.assumptions['terminal_reinvestment_rate']:.1%}",
            f"{dcf.assumptions['implied_terminal_ronic']:.0%}",
        ]
        text = _readme()
        for figure in figures:
            with self.subTest(figure=figure):
                self.assertIn(figure, text)


def _cli_examples(text: str) -> list[tuple[str, list[str]]]:
    """(command line, arguments) for each CLI command in a shell code block,
    with backslash continuations joined and comments dropped."""
    examples = []
    for lang, body in _blocks(text):
        if lang not in ("bash", "sh", "shell", "console"):
            continue
        for line in re.sub(r"[ \t]*\\\n\s*", " ", body).splitlines():
            match = CLI_COMMAND.search(line)
            if match:
                examples.append((line.strip(), shlex.split(match.group(1), comments=True)))
    return examples


class CliExampleTests(unittest.TestCase):
    def test_readme_has_cli_examples(self):
        args = [a for _, a in _cli_examples(_readme())]
        self.assertTrue(args, "no `python -m equity_valuation` command found in README.md")
        self.assertIn(["--demo"], args)

    def test_every_documented_command_passes_the_cli_checks(self):
        demo = _demo_report()
        for line, args in _cli_examples(_readme()):
            with self.subTest(command=line):
                stdout, stderr = io.StringIO(), io.StringIO()
                if {"-h", "--help"} & set(args):
                    with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit) as cm:
                        build_parser().parse_args(args)
                    self.assertEqual(cm.exception.code, 0)
                    continue
                # The engine and the writers are stubbed: only the argument
                # handling runs, offline, and nothing is written.
                with tempfile.TemporaryDirectory() as tmp, \
                        mock.patch("equity_valuation.engine.value_company",
                                   return_value=demo) as engine, \
                        mock.patch("equity_valuation.report.excel.write_excel",
                                   side_effect=lambda report, path: path), \
                        mock.patch("equity_valuation.report.html.write_html",
                                   side_effect=lambda report, path: path), \
                        contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    try:
                        code = main([*args, "--out", tmp, "--quiet"])
                    except SystemExit as exc:  # argparse rejected the command
                        code = exc.code
                self.assertEqual(code, 0, stderr.getvalue())
                engine.assert_called_once()

    def test_options_named_in_inline_code_exist(self):
        known = set(build_parser()._option_string_actions)
        for match in INLINE_OPTION.finditer(_readme()):
            word = match.group(1)
            braces = re.search(r"\{([^}]*)\}", word)
            names = ([word[:braces.start()] + part + word[braces.end():]
                      for part in braces.group(1).split(",")] if braces else [word])
            for name in names:
                with self.subTest(option=name):
                    self.assertIn(name, known)


def _module_map(text: str) -> list[str]:
    """Paths listed in the first ```text block under "## Project layout".

    Each line is ``<indent><name> <description>``; two spaces of indent per
    level nest a name under the last directory (a name ending in "/") above it.
    """
    blocks = [body for lang, body in _blocks(_section(text, "Project layout")) if lang == "text"]
    if not blocks:
        raise AssertionError("'## Project layout' has no ```text module map")
    paths: list[str] = []
    stack: list[str] = []
    for line in blocks[0].splitlines():
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2 or indent // 2 > len(stack):
            raise AssertionError(f"module map line is mis-indented: {line!r}")
        name = line.split()[0]
        stack = stack[:indent // 2]
        paths.append("/".join([*stack, name]))
        stack.append(name.rstrip("/"))
    return paths


class ModuleMapTests(unittest.TestCase):
    def test_every_listed_path_exists(self):
        paths = _module_map(_readme())
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path):
                target = ROOT / path
                if path.endswith("/"):
                    self.assertTrue(target.is_dir(), f"{path} is not a directory")
                else:
                    self.assertTrue(target.is_file(), f"{path} is not a file")

    def test_every_engine_and_backend_module_is_listed(self):
        listed = set(_module_map(_readme()))
        modules = [*(ROOT / "equity_valuation").rglob("*.py"), *(ROOT / "backend").glob("*.py")]
        for module in modules:
            relative = module.relative_to(ROOT).as_posix()
            if module.name.startswith("__") or "__pycache__" in relative:
                continue
            with self.subTest(module=relative):
                self.assertIn(relative, listed)

    def test_architecture_diagram_names_existing_modules(self):
        blocks = [body for lang, body in _blocks(_section(_readme(), "Architecture"))
                  if lang == "mermaid"]
        self.assertTrue(blocks, "'## Architecture' has no ```mermaid diagram")
        diagram = blocks[0]
        self.assertTrue(diagram.lstrip().startswith("flowchart"))
        modules = set(DIAGRAM_MODULE.findall(diagram))
        self.assertTrue(modules)
        for module in sorted(modules):
            with self.subTest(module=module):
                self.assertTrue((ROOT / module).is_file()
                                or (ROOT / "equity_valuation" / module).is_file(),
                                f"{module} not found at the root or in equity_valuation/")


_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                 "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12}


def _flat(text: str) -> str:
    """`text` with each run of whitespace as one space, so a phrase that wraps
    across lines still matches."""
    return re.sub(r"\s+", " ", text)


def _count(word: str) -> int:
    """A count written as digits or as a word ("three")."""
    return int(word) if word.isdigit() else _NUMBER_WORDS[word.lower()]


class DocumentedRuleTests(unittest.TestCase):
    """Thresholds and rules the README states in prose match the code, so a
    changed constant cannot leave the README describing the old one."""

    def _stated(self, pattern: str) -> list[str]:
        """The first matched group of each match of `pattern` in the README
        (whitespace-flattened); fails when the README no longer states it."""
        found = [next(g for g in m.groups() if g is not None)
                 for m in re.finditer(pattern, _flat(_readme()))]
        self.assertTrue(found, f"README.md no longer states the rule matched by {pattern!r}")
        return found

    def test_verdict_bands(self):
        for pct in self._stated(r"Undervalued at (?:>= )?\+(\d+)%"):
            band = int(pct) / 100
            with self.subTest(band=f"+{pct}%"):
                self.assertEqual(_recommendation(100 * (1 + band), 100.0), "Undervalued")
                self.assertEqual(_recommendation(100 * (1 + band) - 0.01, 100.0),
                                 "Fairly valued")
        for pct in self._stated(r"Overvalued at (?:<= )?-(\d+)%"):
            band = int(pct) / 100
            with self.subTest(band=f"-{pct}%"):
                self.assertEqual(_recommendation(100 * (1 - band), 100.0), "Overvalued")
                self.assertEqual(_recommendation(100 * (1 - band) + 0.01, 100.0),
                                 "Fairly valued")

    def test_comps_minimum_peer_values(self):
        for word in self._stated(r"(?:at least|fewer than) (\w+) (?:usable )?peer values"):
            with self.subTest(stated=word):
                self.assertEqual(_count(word), config.COMPS_MIN_PEER_VALUES)

    def test_trailing_gap_note_threshold(self):
        for pct in self._stated(r"differ by more than (\d+)%|more than (\d+)% away"):
            with self.subTest(stated=pct):
                self.assertAlmostEqual(int(pct) / 100, TTM_GAP_THRESHOLD)

    def test_share_count_mismatch_tolerance(self):
        from equity_valuation.data.market import _SHARE_MISMATCH_TOL

        for pct in self._stated(r"are more than (\d+)% apart"):
            with self.subTest(stated=pct):
                self.assertAlmostEqual(int(pct) / 100, _SHARE_MISMATCH_TOL)

    def test_discovered_peer_limit(self):
        for word in self._stated(r"proposes up to (\w+) names"):
            with self.subTest(stated=word):
                self.assertEqual(_count(word), config.PEER_DISCOVERY_LIMIT)

    def test_low_payout_ddm_cut_off(self):
        for pct in self._stated(r"below (?:a )?(\d+)% payout|payout is below (\d+)%"):
            with self.subTest(stated=pct):
                self.assertAlmostEqual(int(pct) / 100, LOW_DDM_PAYOUT_SHARE)

    def test_reverse_dcf_tolerance(self):
        for pct in self._stated(r"within ([\d.]+)% of the price"):
            with self.subTest(stated=pct):
                self.assertAlmostEqual(float(pct) / 100, ACCEPT_TOL, delta=1e-12)

    def test_blume_weights(self):
        stated = re.findall(r"Blume-adjusted \((\d\.\d+) x raw \+ (\d\.\d+)\)",
                            _flat(_readme()))
        self.assertTrue(stated, "README.md no longer states the Blume weights")
        for raw, market in stated:
            with self.subTest(raw=raw, market=market):
                self.assertAlmostEqual(blume_adjusted_beta(0.0), float(market))
                self.assertAlmostEqual(blume_adjusted_beta(1.0) - blume_adjusted_beta(0.0),
                                       float(raw))

    def test_dividends_the_ddm_counts(self):
        # The engine's own wording (engine._DDM_BASIS): regular plus recurring
        # variable dividends; buybacks and one-off specials are ignored.
        self.assertIn("recurring variable", _DDM_BASIS)
        self.assertIn("one-off specials", _DDM_BASIS)
        text = _flat(_readme())
        self.assertNotIn("regular dividends only", text)
        self.assertRegex(text, r"regular (?:plus|and) recurring variable dividends")
        self.assertIn("one-off specials", text)

    def test_spin_off_factor_band(self):
        # provider._corporate_action_warning: a factor below 1 always warns
        # (HON's 0.9535); one just above 1 with the share count unchanged
        # passes for a stock dividend within _STOCK_DIVIDEND_TOLERANCE, so the
        # silent band the README states is about 1 + that tolerance.
        fin = SimpleNamespace(fiscal_years=[2024, 2025], diluted_shares=[100.0, 100.0])

        def flagged(factor: float) -> bool:
            return _corporate_action_warning([(datetime.date(2026, 3, 2), factor)], fin,
                                             datetime.date(2025, 12, 31), 100.0) is not None

        self.assertRegex(_flat(_readme()), r"factor below 1 (?:\([^)]*\) )?is always flagged")
        self.assertTrue(flagged(0.9535))
        self.assertTrue(flagged(0.99))
        for stated in self._stated(r"factor just above 1 \(under about ([\d.]+)\)"):
            band = float(stated)
            with self.subTest(stated=stated):
                self.assertAlmostEqual(band, 1.0 + _STOCK_DIVIDEND_TOLERANCE, delta=0.005)
                self.assertFalse(flagged(band - 0.01))
                self.assertTrue(flagged(band + 0.01))


if __name__ == "__main__":
    unittest.main()
