# Shortcuts for the checks CI runs (.github/workflows/tests.yml).
# Uses the project's .venv when it exists; override with `make test PYTHON=python3.13`.
# lint and coverage need the tools in requirements-dev.txt.
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
OUT ?= output

.PHONY: test demo lint coverage check

## Unit tests, then the synthetic end-to-end checks (offline).
test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'
	$(PYTHON) -m tests.test_synthetic

## Offline demo on the synthetic company; writes Excel and HTML to $(OUT)/.
demo:
	$(PYTHON) -m equity_valuation --demo --out $(OUT)

## Syntax errors and pyflakes checks (rules in pyproject.toml).
lint:
	$(PYTHON) -m ruff check equity_valuation backend tests

## Both suites under coverage; fails below fail_under in pyproject.toml.
coverage:
	$(PYTHON) -m coverage run -m unittest discover -s tests -p 'test_*.py'
	$(PYTHON) -m coverage run -a -m tests.test_synthetic
	$(PYTHON) -m coverage report

## Everything the Python CI job runs: lint, then both suites with the coverage gate.
## make stops at the first failure; `make -k check` still runs the suites after a
## lint error, as CI does.
check: lint coverage
