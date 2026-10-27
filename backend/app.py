"""FastAPI app: valuation + peer suggestions + filings intelligence + FMP
enrichment + AI research + Office exports + watchlist/persistence. All JSON
except the export downloads.

Run from the repo root:
    uvicorn backend.app:app --reload --port 8000
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from equity_valuation.data.base import DataError

from . import ai_service, exports, filings, store
from .fmp_client import FMPClient
from .serialization import _sanitize, build_ai_context
from .valuation_service import AssumptionError, SnapshotError, discover_peers, run_valuation

app = FastAPI(title="Equity Research Automation API", version="2.0.0")


# --------------------------------------------------------------------------- #
#  Request guards (pure ASGI, so they behave the same on every Starlette the
#  FastAPI floor allows). Order, outermost first:
#    LocalHostOnlyMiddleware -> CORSMiddleware -> BodySizeLimitMiddleware
# --------------------------------------------------------------------------- #
# DNS-rebinding guard. CORS cannot stop a page on a rebound hostname: it is
# same-origin with itself, so the browser sends Host (and Origin) set to the
# attacker's name. Only answer requests addressed to a loopback name. The Next
# dev/prod rewrite proxy rewrites Host to the backend target (changeOrigin)
# and passes the browser's original Host as X-Forwarded-Host, so check that
# too. The desktop app calls http://127.0.0.1:<port> directly.
_LOCAL_HOST_RE = re.compile(
    r"(?:localhost|127\.0\.0\.1|\[::1\])(?::[0-9]{1,5})?|::1", re.IGNORECASE
)
# Browser origins of the dashboard itself. The desktop wrapper assigns a new
# local frontend port on each launch, so any port is accepted. Shared by CORS
# and the cross-site check below.
_LOCAL_ORIGIN = r"https?://(?:localhost|127\.0\.0\.1|\[::1\])(?::[0-9]{1,5})?"
_LOCAL_ORIGIN_RE = re.compile(_LOCAL_ORIGIN)
_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# Request-body caps. /api/ai/* carries base64 PDFs and matches the Next
# proxy's 50 MB body cap (frontend/next.config.mjs), which it must not exceed:
# the proxy cuts longer bodies short. This is a transport cap only: what goes
# on to the Anthropic API must also fit its 32 MB request limit, which
# ai_service checks before sending (HTTP 413). Every other route takes plain
# JSON, the largest being one ticker's saved research state.
MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_UPLOAD_BODY_BYTES = 50 * 1024 * 1024
_UPLOAD_PREFIX = "/api/ai/"


async def _refuse(scope, receive, send, status: int, detail: str) -> None:
    if scope["type"] == "websocket":
        await send({"type": "websocket.close", "code": 1008})
        return
    await JSONResponse({"detail": detail}, status_code=status)(scope, receive, send)


def _guard_verdict(scope) -> Optional[tuple[int, str]]:
    """(status, detail) when a request must be refused, else None.

    - Host and every X-Forwarded-Host must be a loopback name (400).
    - Cross-site request forgery (403), which the Host check cannot see: a
      page on another site that targets http://127.0.0.1:<port> sends the
      genuine loopback Host. Refused are (a) unsafe methods whose Origin is
      present and not a loopback origin, which covers bodies a browser sends
      without a CORS preflight on any FastAPI version, and (b) any request a
      browser labels Sec-Fetch-Site: cross-site unless its Origin is a
      loopback origin, which covers side-effect GETs such as /api/enrichment
      (image tags, links). The dashboard's own calls pass: through the Next
      proxy they are same-origin, the desktop app's 127.0.0.1:<front> ->
      127.0.0.1:<api> calls are same-site with a loopback Origin, and a
      localhost page calling 127.0.0.1 is cross-site but sends a loopback
      Origin. Clients that send neither header (curl, scripts) pass.
    """
    hosts: list[str] = []
    forwarded: list[str] = []
    origins: list[str] = []
    sites: list[str] = []
    for name, value in scope.get("headers") or []:
        text = value.decode("latin-1").strip()
        if name == b"host":
            hosts.append(text)
        elif name == b"x-forwarded-host":
            forwarded += [h.strip() for h in text.split(",") if h.strip()]
        elif name == b"origin":
            origins.append(text)
        elif name == b"sec-fetch-site":
            sites.append(text.lower())
    if not (len(hosts) == 1 and all(_LOCAL_HOST_RE.fullmatch(h) for h in hosts + forwarded)):
        return 400, "Invalid host header: this API only serves localhost."
    local_origin = bool(origins) and all(_LOCAL_ORIGIN_RE.fullmatch(o) for o in origins)
    unsafe = scope["type"] == "websocket" or scope.get("method", "").upper() in _UNSAFE_METHODS
    if (unsafe and origins and not local_origin) or ("cross-site" in sites and not local_origin):
        return 403, "Cross-site request refused: this API only serves the local dashboard."
    return None


class LocalHostOnlyMiddleware:
    """Refuse requests that are not the local dashboard's own (see
    _guard_verdict) before any other layer runs."""

    def __init__(self, app) -> None:
        # Must be named `app`: Starlette before 0.41.3 (FastAPI before 0.118)
        # builds middleware as cls(app=app).
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] in ("http", "websocket"):
            verdict = _guard_verdict(scope)
            if verdict is not None:
                await _refuse(scope, receive, send, *verdict)
                return
        await self.app(scope, receive, send)


async def _discard_body(receive, budget: int, more: bool = True) -> None:
    """Read and drop the rest of a request body, up to about `budget` bytes,
    before a 413 goes out. A proxy still streaming the upload (the Next
    rewrite) then relays the 413 instead of failing on a connection closed
    mid-write. Memory stays flat."""
    while more and budget >= 0:
        message = await receive()
        if message["type"] != "http.request":
            return
        budget -= len(message.get("body", b""))
        more = message.get("more_body", False)


class BodySizeLimitMiddleware:
    """Answer 413 once a request body passes its route's cap: before the route
    runs when Content-Length declares more, otherwise as soon as the streamed
    bytes do. The route then reads a disconnect and its own reply is dropped,
    so an oversized body is never buffered or parsed."""

    def __init__(self, app, max_body: int = MAX_BODY_BYTES,
                 max_upload_body: int = MAX_UPLOAD_BODY_BYTES) -> None:
        self.app = app  # named `app` for the same reason as above
        self.max_body = max_body
        self.max_upload_body = max_upload_body

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        upload = scope.get("path", "").startswith(_UPLOAD_PREFIX)
        limit = self.max_upload_body if upload else self.max_body
        size = f"{limit / 1024 / 1024:g} MB" if limit >= 1024 * 1024 else f"{limit} bytes"
        detail = f"Request body is larger than {size}."
        if upload:
            detail += " Attach fewer or smaller PDFs."
        for name, value in scope.get("headers") or []:
            if name == b"content-length" and value.strip().isdigit() and int(value) > limit:
                # A body declared past the upload cap is answered at once: the
                # Next proxy cuts those short, so waiting for the rest would hang.
                if int(value) <= self.max_upload_body:
                    await _discard_body(receive, int(value))
                await _refuse(scope, receive, send, 413, detail)
                return

        received = 0
        cut = started = answered = False

        async def limited_receive():
            nonlocal received, cut, answered
            if cut:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    cut = True
                    if not started:
                        await _discard_body(receive, self.max_upload_body - received,
                                            message.get("more_body", False))
                        await _refuse(scope, receive, send, 413, detail)
                        answered = True
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message) -> None:
            nonlocal started
            if answered:
                return  # the 413 already went out; drop the route's reply
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        await self.app(scope, limited_receive, guarded_send)


# add_middleware wraps the current stack, so register the innermost first.
app.add_middleware(BodySizeLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=_LOCAL_ORIGIN,
    allow_methods=["*"],
    allow_headers=["*"],
    # Lets the desktop app (cross-origin ?api= mode) read export file names.
    expose_headers=["Content-Disposition"],
)
app.add_middleware(LocalHostOnlyMiddleware)

_fmp = FMPClient()


# --------------------------------------------------------------------------- #
#  Request models
# --------------------------------------------------------------------------- #
class ValuationRequest(BaseModel):
    ticker: str
    model_config = {"extra": "allow"}  # assumption overrides ride along


class DigestRequest(BaseModel):
    report: Optional[dict] = None
    extra_context: Optional[str] = None
    material_text: str = ""
    pdfs: Optional[list[dict]] = None


class FilingDigestRequest(BaseModel):
    ticker: str
    form: str
    filed: str = ""
    accession_number: str
    primary_document: str
    report: Optional[dict] = None
    extra_context: Optional[str] = None


class TranscriptDigestRequest(BaseModel):
    ticker: str
    year: int
    quarter: int
    report: Optional[dict] = None
    extra_context: Optional[str] = None


class NoteRequest(BaseModel):
    report: Optional[dict] = None
    extra_context: Optional[str] = None
    pdfs: Optional[list[dict]] = None


class ChatRequest(BaseModel):
    report: Optional[dict] = None
    extra_context: Optional[str] = None
    turns: list[dict]
    pdfs: Optional[list[dict]] = None


class ExportRequest(BaseModel):
    ticker: str
    note: Optional[dict] = None
    model_config = {"extra": "allow"}  # assumption overrides ride along


class WatchlistRequest(BaseModel):
    action: str  # "add" | "remove"
    ticker: str
    snapshot: Optional[dict] = None


class ResearchStateRequest(BaseModel):
    notes: Optional[str] = None
    digests: Optional[list[dict]] = None
    note: Optional[dict] = None
    assumptions: Optional[dict] = None


class SettingsRequest(BaseModel):
    anthropic_api_key: Optional[str] = None
    fmp_api_key: Optional[str] = None


def _context(report: Optional[dict], extra: Optional[str]) -> str:
    parts: list[str] = []
    if report:
        try:
            parts.append(build_ai_context(report))
        except Exception:  # noqa: BLE001
            pass
    if extra and extra.strip():
        # The research log holds model-written summaries of third-party
        # documents. Fenced, and the system prompt treats it as reference
        # data, so instructions planted in a document carry no authority.
        parts.append("USER RESEARCH NOTES / FED MATERIAL SO FAR:\n"
                     + ai_service.fence("research_log", extra.strip()))
    return "\n\n".join(parts)


def _anthropic_http(exc: Exception) -> Optional[HTTPException]:
    """Map an Anthropic SDK exception to a clean HTTP error, or None if `exc`
    is not one. Checks sys.modules so the SDK import stays lazy."""
    anthropic = sys.modules.get("anthropic")
    if anthropic is None or not isinstance(exc, anthropic.APIError):
        return None
    msg = getattr(exc, "message", None) or str(exc)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") if isinstance(body.get("error"), dict) else body
        if isinstance(err.get("message"), str) and err["message"].strip():
            msg = err["message"]
    msg = msg.strip()
    if msg and msg[-1] not in ".!?":
        msg += "."
    slow = HTTPException(
        status_code=504,
        detail="The Anthropic API took too long to respond. Try again, or "
        "send less material.",
    )
    if isinstance(exc, anthropic.APITimeoutError) and (
            exc.__cause__ is None or type(exc.__cause__).__name__ == "ReadTimeout"):
        # Sent, then no response in time: a slow or stalled API, not the
        # user's network. Connect, write (upload) and pool timeouts are local
        # or network trouble and fall through to the 503 below.
        return slow
    if isinstance(exc, anthropic.APIConnectionError):
        what = "timed out" if isinstance(exc, anthropic.APITimeoutError) else "failed"
        return HTTPException(
            status_code=503,
            detail=f"Connection to the Anthropic API {what}. Check your network and try again.",
        )
    if not isinstance(exc, anthropic.APIStatusError):
        return HTTPException(status_code=502, detail=f"Anthropic API error: {msg}")
    # A streamed call that fails after the API answered HTTP 200 (overloaded,
    # rate limited, server error) gets an `error` event, which the SDK raises
    # with status_code 200; error_status() reads the status it stands for off
    # the event's error.type, so it maps like the plain HTTP error would.
    in_stream = exc.status_code < 400
    code = ai_service.error_status(exc)
    if code in (401, 403):
        return HTTPException(
            status_code=502,
            detail=f"Anthropic rejected the API key (HTTP {code}): {msg} "
            "Replace it under API keys at the top of the dashboard, or edit "
            "ANTHROPIC_API_KEY in .env.",
        )
    if code == 404:
        return HTTPException(
            status_code=502,
            detail=f"Anthropic API returned not found: {msg} Check that "
            f"ANTHROPIC_MODEL ({ai_service.MODEL}) is available to your account.",
        )
    if code == 429:
        headers = {}
        response = getattr(exc, "response", None)
        retry_after = response.headers.get("retry-after") if response is not None else None
        if retry_after:
            headers["Retry-After"] = retry_after
        return HTTPException(
            status_code=429,
            detail=f"Anthropic rate limit reached: {msg} Wait a moment and retry.",
            headers=headers or None,
        )
    if code == 413:
        return HTTPException(
            status_code=413,
            detail=f"Request too large for the Anthropic API: {msg} "
            "Attach fewer or smaller PDFs.",
        )
    if code in (400, 422):
        return HTTPException(status_code=400, detail=f"Anthropic API rejected the request: {msg}")
    if code in (503, 529):
        return HTTPException(
            status_code=503,
            detail=f"The Anthropic API is overloaded or unavailable: {msg} Try again shortly.",
        )
    if code == 504:
        return slow
    if in_stream:
        return HTTPException(status_code=502,
                             detail=f"Anthropic API error during the response: {msg}")
    return HTTPException(status_code=502, detail=f"Anthropic API error (HTTP {code}): {msg}")


def _http(exc: Exception) -> HTTPException:
    if isinstance(exc, SnapshotError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, ai_service.AIRequestTooLargeError):
        return HTTPException(status_code=413, detail=str(exc))
    if isinstance(exc, ai_service.AIError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, AssumptionError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, DataError):
        return HTTPException(status_code=404, detail=str(exc))
    mapped = _anthropic_http(exc)
    if mapped is not None:
        return mapped
    return HTTPException(status_code=500, detail=str(exc))


@app.exception_handler(store.StoreError)
def _store_unavailable(_request, exc: store.StoreError) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=503)


@app.exception_handler(store.StoreLimitError)
def _store_limit(_request, exc: store.StoreLimitError) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=exc.status_code)


# --------------------------------------------------------------------------- #
#  Core routes
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "fmp_enabled": _fmp.enabled,
        "anthropic_enabled": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }


# A plausible exchange ticker (BRK.B, BRK-B, 7203.T): checked before any
# lookup so arbitrary text never reaches the data provider.
_TICKER_RE = re.compile(r"[A-Za-z0-9.\-]{1,12}")

# How the market-data client words "Yahoo has no price for this symbol"
# (YFinanceClient.get_market_data), almost always a mistyped or delisted
# ticker. The API answers it in plain words instead of the client's.
_NO_QUOTE_PREFIX = "No obtainable price for "


def _request_ticker(raw: Any) -> str:
    """The upper-cased ticker of a valuation or export request, or HTTP 400
    before any lookup: 'A ticker is required.' when it is empty, 'Invalid
    ticker.' when it is not a plausible exchange ticker (``_TICKER_RE``)."""
    ticker = (raw or "").strip()
    if not ticker:
        raise HTTPException(status_code=400, detail="A ticker is required.")
    if _TICKER_RE.fullmatch(ticker) is None:
        raise HTTPException(status_code=400, detail="Invalid ticker.")
    return ticker.upper()


def _data_http(ticker: str, exc: DataError) -> HTTPException:
    """A DataError from loading `ticker` as HTTP 404. No quote at all reads
    'No market data for X; check the symbol...'; any other failure (no usable
    fundamentals, a missing exchange rate) keeps the provider's reason."""
    if str(exc).startswith(_NO_QUOTE_PREFIX):
        return HTTPException(
            status_code=404,
            detail=f"No market data for {ticker}; check the symbol, or try again "
            "shortly if the market-data service is unreachable.",
        )
    return HTTPException(status_code=404, detail=f"Could not load {ticker}: {exc}")


@app.post("/api/valuation")
def valuation(req: ValuationRequest) -> dict:
    payload = req.model_dump()
    ticker = _request_ticker(payload.pop("ticker", ""))
    try:
        return run_valuation(ticker, payload)
    except AssumptionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DataError as exc:
        raise _data_http(ticker, exc) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/peers/{ticker}")
def peers(ticker: str) -> dict:
    """Suggested peers for the dashboard's 'Suggest peers' button
    (``valuation_service.discover_peers``: Yahoo's industry list within a
    market-cap band). Suggestions only: nothing is applied, and the notes ask
    for review. An implausible ticker is a 400; a failed lookup is still a
    200 with no peers and a note saying why, so the button never errors."""
    if _TICKER_RE.fullmatch(ticker) is None:
        raise HTTPException(status_code=400, detail="Invalid ticker.")
    symbol = ticker.upper()
    try:
        return _sanitize(discover_peers(symbol))
    except Exception as exc:  # noqa: BLE001 - suggestions are optional
        return {"ticker": symbol, "industry_key": None, "industry": None, "peers": [],
                "notes": [f"No peers discovered: peer discovery failed ({exc})."],
                "source": "the data provider"}


@app.get("/api/enrichment/{ticker}")
def enrichment(ticker: str) -> dict:
    return _fmp.enrichment(ticker)


# --------------------------------------------------------------------------- #
#  Filings & transcripts (the "read primary sources" layer)
# --------------------------------------------------------------------------- #
@app.get("/api/filings/{ticker}")
def filings_list(ticker: str) -> dict:
    try:
        return filings.list_filings(ticker)
    except Exception as exc:  # noqa: BLE001
        raise _http(exc) from exc


@app.post("/api/filings/digest")
def filings_digest(req: FilingDigestRequest) -> dict:
    try:
        material, meta = filings.build_filing_material(
            req.ticker, req.form, req.filed, req.accession_number,
            req.primary_document,
        )
        digest = ai_service.digest(
            context=_context(req.report, req.extra_context),
            material_text=material,
        )
        return {"digest": digest, "meta": meta,
                "source": f"{req.form} filed {req.filed}"}
    except Exception as exc:  # noqa: BLE001
        raise _http(exc) from exc


@app.get("/api/transcripts/{ticker}")
def transcripts(ticker: str) -> dict:
    if not _fmp.enabled:
        return {"enabled": False, "transcripts": []}
    return {"enabled": True, "transcripts": _fmp.transcripts_list(ticker)}


@app.post("/api/transcripts/digest")
def transcripts_digest(req: TranscriptDigestRequest) -> dict:
    try:
        if not _fmp.enabled:
            raise ai_service.AIError(
                "FMP_API_KEY is not set — transcripts come from FMP.")
        t = _fmp.transcript(req.ticker, req.year, req.quarter)
        content = (t or {}).get("content") or ""
        if not content:
            raise ai_service.AIError(
                f"No transcript found for {req.ticker} Q{req.quarter} {req.year} "
                "(may not be covered by your FMP plan).")
        material = (
            f"EARNINGS CALL TRANSCRIPT: {req.ticker.upper()} Q{req.quarter} "
            f"FY{req.year} ({(t or {}).get('date', '')})\n\n" + content[:150_000]
        )
        digest = ai_service.digest(
            context=_context(req.report, req.extra_context),
            material_text=material,
        )
        return {"digest": digest,
                "meta": {"chars": len(content)},
                "source": f"Q{req.quarter} FY{req.year} earnings call"}
    except Exception as exc:  # noqa: BLE001
        raise _http(exc) from exc


# --------------------------------------------------------------------------- #
#  AI research
# --------------------------------------------------------------------------- #
@app.post("/api/ai/digest")
def ai_digest(req: DigestRequest) -> dict:
    try:
        return ai_service.digest(
            context=_context(req.report, req.extra_context),
            material_text=req.material_text,
            pdfs=req.pdfs,
        )
    except Exception as exc:  # noqa: BLE001
        raise _http(exc) from exc


@app.post("/api/ai/research_note")
def ai_research_note(req: NoteRequest) -> dict:
    try:
        return ai_service.research_note(
            context=_context(req.report, req.extra_context), pdfs=req.pdfs
        )
    except Exception as exc:  # noqa: BLE001
        raise _http(exc) from exc


@app.post("/api/ai/chat")
def ai_chat(req: ChatRequest) -> dict:
    try:
        reply = ai_service.chat(
            context=_context(req.report, req.extra_context),
            turns=req.turns,
            pdfs=req.pdfs,
        )
        return {"reply": reply}
    except Exception as exc:  # noqa: BLE001
        raise _http(exc) from exc


# --------------------------------------------------------------------------- #
#  Office exports
# --------------------------------------------------------------------------- #
_EXPORTERS = {
    "excel": (exports.export_excel,
              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    "html": (exports.export_html, "text/html"),
    "memo": (exports.export_memo,
             "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    "deck": (exports.export_deck,
             "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
}


_export_lock = threading.Lock()


@app.post("/api/export/{kind}")
def export(kind: str, req: ExportRequest):
    if kind not in _EXPORTERS:
        raise HTTPException(status_code=404, detail=f"Unknown export '{kind}'.")
    fn, media = _EXPORTERS[kind]
    payload = req.model_dump()
    ticker = _request_ticker(payload.pop("ticker", ""))
    note = payload.pop("note", None)
    try:
        # Exporters write a fixed output/{TICKER}_*.ext path. Build and read the
        # file under a lock, then send the bytes: a concurrent export of the
        # same ticker can no longer rewrite the file mid-download.
        with _export_lock:
            if kind in ("memo", "deck"):
                path = fn(ticker, payload, note)
            else:
                path = fn(ticker, payload)
            with open(path, "rb") as f:
                data = f.read()
        return Response(
            content=data,
            media_type=media,
            headers={"Content-Disposition":
                     f'attachment; filename="{os.path.basename(path)}"'},
        )
    except DataError as exc:
        raise _data_http(ticker, exc) from exc
    except Exception as exc:  # noqa: BLE001
        raise _http(exc) from exc


# --------------------------------------------------------------------------- #
#  Watchlist + research persistence
# --------------------------------------------------------------------------- #
@app.get("/api/watchlist")
def watchlist_get() -> dict:
    return {"watchlist": store.get_watchlist()}


@app.post("/api/watchlist")
def watchlist_post(req: WatchlistRequest) -> dict:
    if req.action == "add":
        wl = store.upsert_watchlist(
            {**(req.snapshot or {}), "ticker": req.ticker})
    elif req.action == "remove":
        wl = store.remove_watchlist(req.ticker)
    else:
        raise HTTPException(status_code=400, detail="action must be add|remove")
    return {"watchlist": wl}


@app.get("/api/research_state/{ticker}")
def research_get(ticker: str) -> dict:
    return store.get_research(ticker)


@app.post("/api/research_state/{ticker}")
def research_post(ticker: str, req: ResearchStateRequest) -> dict:
    # Omitted fields are unchanged; an explicit null clears a saved field.
    return store.save_research(ticker, req.model_dump(exclude_unset=True))


# --------------------------------------------------------------------------- #
#  Settings — lets the user paste API keys in the UI instead of editing .env.
#  Localhost-only personal tool: keys are written to the project .env and into
#  this process's environment, taking effect immediately.
# --------------------------------------------------------------------------- #
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
_settings_lock = threading.Lock()


def _upsert_env_file(updates: dict[str, str]) -> None:
    """Set KEY=value lines in the project .env. Every existing assignment of a
    key (including `export KEY=` and duplicates) is replaced by one line, so a
    later duplicate can't win when the launcher sources the file. The write is
    atomic (temp file + os.replace) and the file is owner-only (0600). Bytes
    that are not UTF-8 (a .env saved as Latin-1 by another editor) are kept
    exactly as they were (surrogateescape); new values are ASCII."""
    env_path = Path(os.path.realpath(_ENV_PATH))  # keep a symlinked .env a symlink
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8", errors="surrogateescape").splitlines()
    for key, value in updates.items():
        pat = re.compile(rf"\s*(?:export\s+)?{re.escape(key)}\s*=")
        out: list[str] = []
        placed = False
        for line in lines:
            if pat.match(line):
                if not placed:
                    out.append(f"{key}={value}")
                    placed = True
                continue
            out.append(line)
        if not placed:
            out.append(f"{key}={value}")
        lines = out
    fd, tmp = tempfile.mkstemp(dir=str(env_path.parent), prefix=".env.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", errors="surrogateescape") as f:
            f.write("\n".join(lines) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, env_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


@app.post("/api/settings")
def settings(req: SettingsRequest) -> dict:
    global _fmp
    updates: dict[str, str] = {}
    if req.anthropic_api_key and req.anthropic_api_key.strip():
        updates["ANTHROPIC_API_KEY"] = req.anthropic_api_key.strip()
    if req.fmp_api_key and req.fmp_api_key.strip():
        updates["FMP_API_KEY"] = req.fmp_api_key.strip()
    if not updates:
        raise HTTPException(status_code=400, detail="No keys provided.")
    for key, value in updates.items():
        # The launcher loads .env with a shell. Keep persisted keys literal and
        # reject line breaks, quoting, substitutions, and other shell syntax.
        if re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None:
            raise HTTPException(
                status_code=400,
                detail=f"{key} must contain only letters, digits, dots, underscores, or hyphens.",
            )
    # Atomic replacement alone cannot prevent two read/modify/write requests
    # from losing each other's keys. Keep file and live configuration ordered.
    with _settings_lock:
        try:
            _upsert_env_file(updates)
        except (OSError, UnicodeError) as exc:
            raise HTTPException(
                status_code=500, detail=f"Could not write .env: {exc}"
            ) from exc
        os.environ.update(updates)
        if "FMP_API_KEY" in updates:
            _fmp = FMPClient()  # picks up the new key
        return health()
