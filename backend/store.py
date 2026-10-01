"""Tiny JSON persistence for the app's personal layer.

Holds the watchlist and per-ticker research state (notes, digests, generated
research note, applied assumptions) in one file under the project's `data/`
directory, so research survives app restarts. Single-process, lock-guarded —
deliberately simple; this is a personal tool, not a multi-tenant service.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_STORE_PATH = _DATA_DIR / "copilot_store.json"
_lock = threading.Lock()

_MAX_DIGESTS_PER_TICKER = 25
# Size bounds. Every watchlist or research call re-reads the whole file, so
# what persists is capped, not only what one request may carry. The limits sit
# far above real use (25 digests plus notes is well under 2 MB per ticker).
_MAX_TICKER_LEN = 32
_MAX_TEXT_FIELD = 200  # watchlist name / currency / recommendation
_MAX_RESEARCH_BYTES = 8 * 1024 * 1024  # one ticker's saved research state
_MAX_STORE_BYTES = 64 * 1024 * 1024  # the whole file; only growth past it is refused


class StoreError(RuntimeError):
    """The store file exists but can't be read or written right now (e.g. a
    sync client or antivirus holds a lock). Surfaced as HTTP 503; the file is
    left untouched."""


class StoreLimitError(ValueError):
    """A save the store refuses: a missing or implausible ticker (HTTP 400) or
    data past the size bounds above (HTTP 413). Nothing is written."""

    def __init__(self, message: str, status_code: int = 413) -> None:
        super().__init__(message)
        self.status_code = status_code


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _valid(data) -> bool:
    return (
        isinstance(data, dict)
        and isinstance(data.get("watchlist", []), list)
        and all(isinstance(w, dict) for w in data.get("watchlist", []))
        and isinstance(data.get("research", {}), dict)
        and all(isinstance(r, dict) for r in data.get("research", {}).values())
    )


def _quarantine() -> None:
    """Move an unusable store aside under a name that never overwrites an
    earlier backup, so the next save can't silently wipe the user's data."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"{_STORE_PATH}.corrupt-{stamp}"
    dest, n = base, 1
    while os.path.exists(dest):
        dest, n = f"{base}-{n}", n + 1
    try:
        os.replace(_STORE_PATH, dest)
    except OSError as exc:
        raise StoreError(f"Research store is unreadable and could not be moved aside: {exc}") from exc


def _not_a_number(_token: str) -> None:
    """json's hook for the bare NaN/Infinity tokens: read them as null. A
    store written by an earlier version may still hold them."""
    return None


def _load() -> dict:
    try:
        with open(_STORE_PATH, encoding="utf-8") as f:
            data = json.load(f, parse_constant=_not_a_number)
    except FileNotFoundError:
        return {"watchlist": [], "research": {}}
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Corrupt store: preserve it for recovery, start fresh.
        _quarantine()
        return {"watchlist": [], "research": {}}
    except OSError as exc:
        # Possibly transient (lock, permissions): don't touch a store that may
        # be perfectly valid.
        raise StoreError(f"Research store is temporarily unreadable: {exc}") from exc
    if not _valid(data):
        _quarantine()
        return {"watchlist": [], "research": {}}
    data.setdefault("watchlist", [])
    data.setdefault("research", {})
    # Valid JSON exponents such as 1e400 also overflow to infinity; unlike bare
    # NaN/Infinity, json's parse_constant hook never sees them.
    return _finite(data)


def _stored_size() -> int:
    try:
        return os.path.getsize(_STORE_PATH)
    except OSError:
        return 0


def _save(data: dict) -> None:
    try:
        # Strict JSON: NaN/Infinity would make the file unreadable to other
        # JSON tools. Callers turn them into null first; this is the backstop.
        # The output is ASCII-only, so len() is the byte size.
        text = json.dumps(data, indent=1, allow_nan=False)
    except ValueError as exc:
        raise StoreLimitError(f"The research store can't save this data: {exc}",
                              status_code=400) from exc
    if len(text) > _MAX_STORE_BYTES and len(text) > _stored_size():
        # Shrinking saves (removals, smaller notes) still go through.
        raise StoreLimitError(
            f"The research store would exceed {_MAX_STORE_BYTES // (1024 * 1024)} MB; "
            "remove watchlist entries or shorten saved notes first.")
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(_DATA_DIR), prefix=".copilot_store.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, _STORE_PATH)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError as exc:
        raise StoreError(f"Could not save the research store: {exc}") from exc


def _check_ticker(ticker: str) -> str:
    if not ticker:
        raise StoreLimitError("A ticker is required.", status_code=400)
    if len(ticker) > _MAX_TICKER_LEN:
        raise StoreLimitError(
            f"Ticker is longer than {_MAX_TICKER_LEN} characters.", status_code=400)
    return ticker


def _text(value):
    """A watchlist text field: strings are cut to length, anything else
    becomes null."""
    return value[:_MAX_TEXT_FIELD] if isinstance(value, str) else None


def _number(value):
    """A watchlist number field: a finite int or float, anything else
    (booleans, text, NaN, Infinity) becomes null."""
    if isinstance(value, bool):
        return None
    ok = isinstance(value, int) or (isinstance(value, float) and math.isfinite(value))
    return value if ok else None


def _finite(value):
    """`value` with every NaN/Infinity, at any depth, replaced by null, as a
    browser's JSON.stringify would send it. Python's JSON parser accepts those
    tokens from other clients, but a strict JSON file can't hold them."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


# --- watchlist -------------------------------------------------------------- #
def get_watchlist() -> list[dict]:
    with _lock:
        return _load()["watchlist"]


def upsert_watchlist(snapshot: dict) -> list[dict]:
    """Add or refresh one ticker's snapshot {ticker, name, currency, price,
    blended_target, recommendation}. Fields keep their expected types (text
    cut to 200 characters, finite numbers or null), so an entry stays small."""
    ticker = str(snapshot.get("ticker", "")).upper().strip()
    if not ticker:
        return get_watchlist()
    _check_ticker(ticker)
    entry = {
        "ticker": ticker,
        "name": _text(snapshot.get("name")),
        "currency": _text(snapshot.get("currency")),
        "price": _number(snapshot.get("price")),
        "blended_target": _number(snapshot.get("blended_target")),
        "recommendation": _text(snapshot.get("recommendation")),
        "updated_at": _now(),
    }
    with _lock:
        data = _load()
        data["watchlist"] = [w for w in data["watchlist"] if w.get("ticker") != ticker]
        data["watchlist"].insert(0, entry)
        _save(data)
        return data["watchlist"]


def remove_watchlist(ticker: str) -> list[dict]:
    ticker = ticker.upper().strip()
    with _lock:
        data = _load()
        data["watchlist"] = [w for w in data["watchlist"] if w.get("ticker") != ticker]
        _save(data)
        return data["watchlist"]


# --- per-ticker research state ---------------------------------------------- #
def get_research(ticker: str) -> dict:
    with _lock:
        return _load()["research"].get(ticker.upper().strip(), {})


def save_research(ticker: str, state: dict) -> dict:
    """Persist {notes, digests, note, assumptions} for a ticker (partial ok).
    Omitted fields stay unchanged; explicit null clears a field.
    NaN/Infinity anywhere in the state are saved as null. An empty or overlong
    ticker, or a state over _MAX_RESEARCH_BYTES, is refused whole
    (StoreLimitError), never truncated, so the last good save survives."""
    ticker = _check_ticker(ticker.upper().strip())
    with _lock:
        data = _load()
        cur = data["research"].get(ticker, {})
        for key in ("notes", "digests", "note", "assumptions"):
            if key in state:
                cur[key] = _finite(state[key])
        digests = cur.get("digests")
        if isinstance(digests, list) and len(digests) > _MAX_DIGESTS_PER_TICKER:
            cur["digests"] = digests[-_MAX_DIGESTS_PER_TICKER:]
        if len(json.dumps(cur)) > _MAX_RESEARCH_BYTES:
            raise StoreLimitError(
                f"Research state for {ticker} would exceed "
                f"{_MAX_RESEARCH_BYTES // (1024 * 1024)} MB; shorten the notes "
                "or remove older digests.")
        cur["updated_at"] = _now()
        data["research"][ticker] = cur
        _save(data)
        return cur
