"""Default assumptions and tunable constants for the valuation engine.

These are sensible, documented defaults for a US large-cap as of ~2025. They can
be overridden per-run via the CLI or by passing custom assumption objects to the
engine. Keep all magic numbers here so the models stay clean.
"""

from __future__ import annotations

import os
from pathlib import Path
import re

# --- Macro / CAPM defaults (decimals) --------------------------------------- #
DEFAULT_RISK_FREE_RATE = 0.042       # ~10y US Treasury yield
DEFAULT_EQUITY_RISK_PREMIUM = 0.05   # Damodaran-style mature-market ERP
DEFAULT_MARGINAL_TAX_RATE = 0.21     # US federal statutory corporate rate
DEFAULT_CREDIT_SPREAD = 0.015        # added to rf when cost of debt can't be derived
MIN_EFFECTIVE_TAX_RATE = 0.0
MAX_EFFECTIVE_TAX_RATE = 0.35        # clamp derived effective tax rates to sane band
DEFAULT_BETA = 1.0                   # fallback if market beta is unavailable

# --- DCF defaults ----------------------------------------------------------- #
DEFAULT_FORECAST_YEARS = 5
DEFAULT_TERMINAL_GROWTH = 0.025      # ~ long-run nominal GDP, must be < WACC
MAX_TERMINAL_GROWTH_VS_WACC = 0.01   # require WACC - g >= this gap; else clamp g
DEFAULT_REVENUE_GROWTH_CAP = 0.30    # cap derived near-term growth at 30%/yr
DEFAULT_REVENUE_GROWTH_FLOOR = -0.05

# --- Comps defaults --------------------------------------------------------- #
DEFAULT_PEER_LIMIT = 8               # max peers pulled when auto-suggesting
COMPS_MULTIPLES = ("ev_ebitda", "ev_sales", "pe", "pb", "peg")
# Outlier trimming: drop peers whose multiple is outside [median/k, median*k]
COMPS_OUTLIER_FACTOR = 3.0

# --- Sensitivity grid defaults ---------------------------------------------- #
SENSITIVITY_WACC_DELTAS = (-0.015, -0.0075, 0.0, 0.0075, 0.015)   # absolute +/- on WACC
SENSITIVITY_GROWTH_DELTAS = (-0.01, -0.005, 0.0, 0.005, 0.01)     # absolute +/- on terminal g
SENSITIVITY_MARGIN_DELTAS = (-0.02, -0.01, 0.0, 0.01, 0.02)       # absolute +/- on EBIT margin
SENSITIVITY_EXIT_MULTIPLE_DELTAS = (-2.0, -1.0, 0.0, 1.0, 2.0)    # absolute +/- on EV/EBITDA

# --- HTTP / EDGAR ----------------------------------------------------------- #
# SEC requires a descriptive User-Agent with contact info on every request.
PROJECT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def _dotenv_value(raw: str) -> str:
    """Parse one .env value the way the desktop launcher and `source` do.

    A leading double-quoted token keeps backslash escapes of `"`, `\\`, `$` and
    `` ` ``; a leading single-quoted token is literal; an unquoted value drops a
    trailing ` # comment`.
    """
    text = raw.strip()
    double = re.match(r'"((?:[^"\\]|\\.)*)"', text)
    if double:
        return re.sub(r'\\(["\\$`])', r"\1", double.group(1))
    single = re.match(r"'([^']*)'", text)
    if single:
        return single.group(1)
    return re.sub(r"\s+#.*$", "", raw).strip()


def _resolve_sec_user_agent(environ=os.environ, env_path: Path = PROJECT_ENV_FILE) -> str:
    """Return SEC_USER_AGENT from the environment, else from the project .env.

    The .env file is parsed like the desktop launcher parses it: blank and `#`
    lines are skipped, an optional `export ` prefix is allowed, quoting and
    inline comments follow `_dotenv_value`, and the last assignment wins. An
    empty result disables EDGAR requests.
    """
    value = (environ.get("SEC_USER_AGENT") or "").strip()
    if value:
        return value
    try:
        lines = env_path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError):
        return ""
    for line in lines:
        text = re.sub(r"^export\s+", "", line.strip())
        if not text or text.startswith("#"):
            continue
        key, sep, raw = text.partition("=")
        if sep and key.strip() == "SEC_USER_AGENT":
            value = _dotenv_value(raw).strip()
    return value


SEC_USER_AGENT = _resolve_sec_user_agent()
SEC_REQUEST_TIMEOUT = 20             # seconds
SEC_MAX_RETRIES = 3
HTTP_RETRY_BACKOFF = 1.5             # seconds, exponential

# --- Output ----------------------------------------------------------------- #
DEFAULT_OUTPUT_DIR = "output"
CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
