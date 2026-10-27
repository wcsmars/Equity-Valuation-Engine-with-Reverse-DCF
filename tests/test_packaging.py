"""Offline checks for the package metadata, the CI gates and the installed layout.

pyproject.toml reads its dependency lists from the requirements files, so these
tests pin that wiring, build the wheel without network access (with the
installed setuptools) and check what ships in it. The .env tests cover the
installed case, where the project .env path points into site-packages.
"""

from __future__ import annotations

import ast
import importlib
import importlib.metadata
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
import zipfile
from pathlib import Path
from unittest import mock

try:
    import yaml  # PyYAML, installed with uvicorn[standard] (backend requirements)
except ImportError:  # pragma: no cover - the engine-only install has no PyYAML
    yaml = None

import equity_valuation
from equity_valuation import config

ROOT = Path(__file__).resolve().parents[1]
# A plain requirement line as setuptools' `file:` dependencies accept it: a
# name, optional extras, one or more version specifiers (a floor is expected)
# and an optional environment marker. No -r includes or pip options.
_SPECIFIER = r"(==|>=|~=|<=|<|>|!=)\s*[0-9][^\s,;#]*"
REQUIREMENT_LINE = re.compile(
    rf"^[A-Za-z0-9][A-Za-z0-9._-]*\s*(\[[A-Za-z0-9,._\s-]+\])?\s*{_SPECIFIER}(\s*,\s*{_SPECIFIER})*"
    r"(\s*;\s*\S.*)?$"
)
_REQUIREMENT_PARTS = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[([^\]]*)\])?\s*([^;]*?)\s*(?:;.*)?$")


def _requirement_key(line: str) -> tuple:
    """Name, extras and version specifiers of a requirement line, compared the
    way setuptools writes them to METADATA: the name is normalised, spacing
    and specifier order do not matter, and the environment marker is left
    out (setuptools rewrites it, adding `extra == ...` for optional groups)."""
    match = _REQUIREMENT_PARTS.match(line)
    if match is None:
        raise ValueError(f"not a requirement line: {line!r}")
    name, extras, specifiers = match.groups()
    return (
        re.sub(r"[-_.]+", "-", name).lower(),
        tuple(sorted(part.strip().lower() for part in (extras or "").split(",") if part.strip())),
        tuple(sorted(re.sub(r"\s+", "", part) for part in specifiers.split(",") if part.strip())),
    )


def _pyproject() -> dict:
    with open(ROOT / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)


def _requirements(relative: str) -> list[str]:
    lines = (ROOT / relative).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def _setuptools_version() -> tuple[int, ...] | None:
    try:
        text = importlib.metadata.version("setuptools")
    except importlib.metadata.PackageNotFoundError:
        return None
    return tuple(int(part) for part in re.findall(r"\d+", text)[:2])


class SecUserAgentFallbackTests(unittest.TestCase):
    """A non-editable install has no project .env; the working directory's is used."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.cwd_env = self.tmp / "work" / ".env"
        self.cwd_env.parent.mkdir()
        self.cwd_env.write_text("SEC_USER_AGENT='cwd app@example.com'\n", encoding="utf-8")
        self.missing = self.tmp / "site-packages" / ".env"

    def test_missing_project_env_reads_the_fallback(self):
        value = config._resolve_sec_user_agent({}, self.missing, fallback_path=self.cwd_env)
        self.assertEqual(value, "cwd app@example.com")

    def test_environment_still_wins(self):
        value = config._resolve_sec_user_agent(
            {"SEC_USER_AGENT": "env@example.com"}, self.missing, fallback_path=self.cwd_env)
        self.assertEqual(value, "env@example.com")

    def test_existing_project_env_does_not_fall_through(self):
        project_env = self.tmp / "project.env"
        project_env.write_text("OTHER=1\n", encoding="utf-8")
        self.assertEqual(config._resolve_sec_user_agent({}, project_env, fallback_path=self.cwd_env), "")

    def test_missing_or_unreadable_fallback_disables_edgar(self):
        self.assertEqual(config._resolve_sec_user_agent({}, self.missing, fallback_path=self.tmp / "none"), "")
        # A directory cannot be read as a file.
        self.assertEqual(config._resolve_sec_user_agent({}, self.missing, fallback_path=self.tmp), "")

    def test_working_directory_env_file(self):
        previous = os.getcwd()
        os.chdir(self.cwd_env.parent)
        try:
            found = config._working_directory_env_file()
        finally:
            os.chdir(previous)
        self.assertEqual(found.resolve(), self.cwd_env.resolve())

    def test_installed_layout_reads_the_working_directory_env(self):
        # config.py under <site-packages>/equity_valuation/, as `pip install` lays it out.
        package = self.tmp / "site-packages" / "equity_valuation"
        package.mkdir(parents=True)
        shutil.copy(ROOT / "equity_valuation" / "config.py", package / "config.py")
        spec = importlib.util.spec_from_file_location("_installed_config_copy", package / "config.py")
        module = importlib.util.module_from_spec(spec)
        previous = os.getcwd()
        os.chdir(self.cwd_env.parent)
        try:
            with mock.patch.dict(os.environ):
                os.environ.pop("SEC_USER_AGENT", None)
                spec.loader.exec_module(module)
        finally:
            os.chdir(previous)
        self.assertEqual(module.PROJECT_ENV_FILE, self.missing.resolve())
        self.assertFalse(module.PROJECT_ENV_FILE.exists())
        self.assertEqual(module.SEC_USER_AGENT, "cwd app@example.com")


class PackageMetadataTests(unittest.TestCase):
    def setUp(self):
        self.pyproject = _pyproject()

    def test_version_is_a_static_literal(self):
        # setuptools reads `attr` versions without importing the package (and
        # so without pandas) only when the assignment is a plain literal.
        dynamic = self.pyproject["tool"]["setuptools"]["dynamic"]
        self.assertEqual(dynamic["version"], {"attr": "equity_valuation.__version__"})
        tree = ast.parse((ROOT / "equity_valuation" / "__init__.py").read_text(encoding="utf-8"))
        literals = [
            node.value.value for node in tree.body
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
            and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
        ]
        self.assertEqual(literals, [equity_valuation.__version__])
        self.assertRegex(equity_valuation.__version__, r"^\d+\.\d+\.\d+$")

    def test_dependency_lists_come_from_the_requirements_files(self):
        project = self.pyproject["project"]
        self.assertEqual(set(project["dynamic"]), {"version", "dependencies", "optional-dependencies"})
        self.assertNotIn("dependencies", project)
        self.assertEqual(project["requires-python"], ">=3.11")
        dynamic = self.pyproject["tool"]["setuptools"]["dynamic"]
        self.assertEqual(dynamic["dependencies"], {"file": ["requirements.txt"]})
        self.assertEqual(dynamic["optional-dependencies"], {
            "backend": {"file": ["backend/requirements-backend.txt"]},
            "dev": {"file": ["requirements-dev.txt"]},
        })
        # setuptools accepts requirement lines only (no -r includes or pip options).
        for relative in ("requirements.txt", "backend/requirements-backend.txt", "requirements-dev.txt"):
            entries = _requirements(relative)
            self.assertTrue(entries, relative)
            for entry in entries:
                with self.subTest(file=relative, entry=entry):
                    self.assertRegex(entry, REQUIREMENT_LINE)

    def test_requirement_line_rules(self):
        # Forms setuptools accepts in a `file:` dependency list.
        for line in ("pandas>=2.2.2", "uvicorn[standard]>=0.29", "foo >= 1.0 , < 2",
                     "Foo_Bar[b, a]~=1.4; python_version < '3.12'"):
            with self.subTest(line=line):
                self.assertRegex(line, REQUIREMENT_LINE)
        # pip-only syntax, and lines without a version floor.
        for line in ("-r requirements.txt", "--index-url https://example.com/simple", "pandas",
                     "git+https://example.com/repo.git", "pandas>=2 # comment"):
            with self.subTest(line=line):
                self.assertNotRegex(line, REQUIREMENT_LINE)
        # Spacing, specifier order, name spelling and markers do not change the key.
        self.assertEqual(_requirement_key("Foo_Bar[b, a] >= 1.0 , < 2 ; python_version < '3.12'"),
                         _requirement_key("foo-bar[a,b]<2,>=1.0"))
        self.assertNotEqual(_requirement_key("foo>=1.0"), _requirement_key("foo>=1.1"))

    def test_console_script_runs_the_cli(self):
        target = self.pyproject["project"]["scripts"]["equity-valuation"]
        module_name, _, attr = target.partition(":")
        main = getattr(importlib.import_module(module_name), attr)
        from equity_valuation import __main__ as entry
        self.assertIs(main, entry.main)

    def test_only_the_engine_ships(self):
        self.assertEqual(self.pyproject["tool"]["setuptools"]["packages"]["find"],
                         {"include": ["equity_valuation*"]})

    def test_lint_and_coverage_settings(self):
        tool = self.pyproject["tool"]
        self.assertEqual(tool["ruff"]["lint"]["select"], ["E9", "F"])
        self.assertEqual(tool["coverage"]["run"]["source"], ["equity_valuation", "backend"])
        floor = tool["coverage"]["report"]["fail_under"]
        self.assertIsInstance(floor, int)
        self.assertGreater(floor, 50)
        self.assertLess(floor, 100)

    def test_dev_requirements(self):
        entries = {re.split(r"[\[<>=~!]", entry, maxsplit=1)[0].lower(): entry
                   for entry in _requirements("requirements-dev.txt")}
        self.assertEqual(set(entries), {"ruff", "coverage", "httpx2", "setuptools"})
        self.assertRegex(entries["ruff"], r"^ruff==\d+\.\d+\.\d+$")  # gate must not drift
        # The packaging test builds with the same setuptools floor as the backend.
        build_requires = self.pyproject["build-system"]["requires"]
        self.assertEqual(build_requires, [entries["setuptools"]])
        self.assertEqual(self.pyproject["build-system"]["build-backend"], "setuptools.build_meta")


class CheckEntryPointTests(unittest.TestCase):
    """CI and the Makefile run the gates pyproject.toml configures."""

    def test_workflow_installs_dev_tools_and_runs_the_gates(self):
        text = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
        self.assertIn("-r requirements.txt -r backend/requirements-backend.txt -r requirements-dev.txt", text)
        for command in (
            "ruff check equity_valuation backend tests",
            "coverage run -m unittest discover -s tests -p 'test_*.py'",
            "coverage run -a -m tests.test_synthetic",
            "run: coverage report\n",
        ):
            with self.subTest(command=command):
                self.assertIn(command, text)
        self.assertEqual(text.count("runs-on:"), text.count("timeout-minutes:"))

    @unittest.skipIf(yaml is None, "PyYAML comes with the backend requirements")
    def test_workflow_step_conditions(self):
        with open(ROOT / ".github" / "workflows" / "tests.yml", encoding="utf-8") as handle:
            workflow = yaml.safe_load(handle)
        # Superseded runs are cancelled, but never on the default branch,
        # where each commit's SHA is its own concurrency group.
        concurrency = workflow["concurrency"]
        self.assertIs(concurrency["cancel-in-progress"], True)
        self.assertIn("github.event.repository.default_branch", concurrency["group"])
        self.assertIn("&& github.sha || github.ref", concurrency["group"])
        for name, job in workflow["jobs"].items():
            with self.subTest(job=name):
                self.assertIsInstance(job.get("timeout-minutes"), int)

        steps = {step.get("id") or step.get("name") or step.get("uses"): step
                 for step in workflow["jobs"]["python"]["steps"]}
        self.assertIn("-r requirements-dev.txt", steps["install"]["run"])
        lint = steps["Lint (syntax errors and pyflakes, rules in pyproject.toml)"]
        self.assertNotIn("if", lint)
        self.assertNotIn("continue-on-error", lint)  # a lint error still fails the job
        # The suites run after a lint (or each other's) failure, not after a failed install.
        for step_id in ("unit", "synthetic"):
            with self.subTest(step=step_id):
                self.assertEqual(steps[step_id]["if"],
                                 "${{ !cancelled() && steps.install.outcome == 'success' }}")
        # Coverage is reported and gated only when both suites passed.
        both_passed = ("${{ !cancelled() && steps.unit.outcome == 'success' "
                       "&& steps.synthetic.outcome == 'success' }}")
        summary, gate = steps["Coverage summary"], steps["Coverage gate (fail_under in pyproject.toml)"]
        self.assertEqual(summary["if"], both_passed)
        self.assertEqual(gate["if"], both_passed)
        self.assertIn("--fail-under=0", summary["run"])
        self.assertEqual(gate["run"], "coverage report")
        names = list(steps)
        self.assertLess(names.index("Coverage summary"), names.index("Coverage gate (fail_under in pyproject.toml)"))

    def test_makefile_targets(self):
        text = (ROOT / "Makefile").read_text(encoding="utf-8")
        targets = set(re.findall(r"(?m)^([a-z]+):", text))
        self.assertTrue({"test", "demo", "lint", "coverage", "check"} <= targets)
        self.assertIn("-m tests.test_synthetic", text)
        self.assertIn("-m ruff check equity_valuation backend tests", text)


@unittest.skipIf((_setuptools_version() or (0,)) < (77,) or importlib.util.find_spec("pip") is None,
                 "building the wheel offline needs pip and setuptools>=77 (requirements-dev.txt)")
class WheelBuildTests(unittest.TestCase):
    """Builds the wheel from a copy of the tree, offline, and inspects it."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        src, dist = cls.tmp / "src", cls.tmp / "dist"
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
        for name in ("pyproject.toml", "README.md", "requirements.txt", "requirements-dev.txt"):
            (src / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / name, src / name)
        # backend/ and tests/ come along so the wheel can be checked to leave them out.
        for name in ("equity_valuation", "backend", "tests"):
            shutil.copytree(ROOT / name, src / name, ignore=ignore)
        env = dict(os.environ, PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INPUT="1")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--no-index",
             "--quiet", "--wheel-dir", str(dist), str(src)],
            capture_output=True, text=True, env=env, timeout=300,
        )
        if result.returncode:
            shutil.rmtree(cls.tmp, ignore_errors=True)
            raise AssertionError(f"pip wheel failed:\n{result.stdout}\n{result.stderr}")
        cls.wheels = sorted(dist.glob("*.whl"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _read(self, archive: zipfile.ZipFile, suffix: str) -> str:
        (name,) = [n for n in archive.namelist() if n.endswith(suffix)]
        return archive.read(name).decode("utf-8")

    def test_wheel_name_and_contents(self):
        version = equity_valuation.__version__
        self.assertEqual([w.name for w in self.wheels], [f"equity_valuation-{version}-py3-none-any.whl"])
        with zipfile.ZipFile(self.wheels[0]) as archive:
            names = archive.namelist()
        tops = {name.split("/", 1)[0] for name in names}
        self.assertEqual(tops, {"equity_valuation", f"equity_valuation-{version}.dist-info"})
        for module in ("cli.py", "config.py", "engine.py", "data/synthetic.py", "models/dcf.py",
                       "report/excel.py", "report/html.py"):
            self.assertIn(f"equity_valuation/{module}", names)

    def test_metadata_matches_the_requirements_files(self):
        with zipfile.ZipFile(self.wheels[0]) as archive:
            metadata = self._read(archive, ".dist-info/METADATA")
            entry_points = self._read(archive, ".dist-info/entry_points.txt")
        headers = metadata.split("\n\n", 1)[0].splitlines()
        self.assertIn("Name: equity-valuation", headers)
        self.assertIn(f"Version: {equity_valuation.__version__}", headers)
        self.assertIn("Requires-Python: >=3.11", headers)
        self.assertIn("License-Expression: MIT", headers)
        requires = [h.removeprefix("Requires-Dist: ") for h in headers if h.startswith("Requires-Dist: ")]

        def keys(lines):
            return sorted(_requirement_key(line) for line in lines)

        core = [r for r in requires if "extra ==" not in r]
        self.assertEqual(keys(core), keys(_requirements("requirements.txt")))
        for extra, relative in (("backend", "backend/requirements-backend.txt"), ("dev", "requirements-dev.txt")):
            with self.subTest(extra=extra):
                self.assertIn(f"Provides-Extra: {extra}", headers)
                listed = [r for r in requires if f'extra == "{extra}"' in r]
                self.assertEqual(keys(listed), keys(_requirements(relative)))
        self.assertIn("equity-valuation = equity_valuation.cli:main", entry_points)


if __name__ == "__main__":
    unittest.main()
