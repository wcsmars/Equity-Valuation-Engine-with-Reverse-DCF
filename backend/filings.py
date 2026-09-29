"""SEC EDGAR filings intelligence (key-less, free).

Lists a company's recent filings (10-K / 10-Q / 8-K / proxies …) and fetches a
filing's primary document as clean text, with best-effort extraction of the
sections an analyst actually reads (Risk Factors, MD&A, Business). The text is
fed to the AI researcher so the app can read primary sources itself instead of
relying on the user to paste excerpts.

Endpoints used (same User-Agent rules as the engine's EDGAR provider):
  * submissions index: https://data.sec.gov/submissions/CIK{cik}.json
  * documents:         https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}
"""

from __future__ import annotations

import re
import threading
import time
from html.parser import HTMLParser
from typing import Optional

import requests

from equity_valuation import config
from equity_valuation.data.base import DataError
from equity_valuation.data.edgar import EdgarClient

_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"

# Forms worth surfacing to an analyst (matched as prefixes of the form type, so
# amendments and variants such as 10-K/A or S-1/A count). Prospectus
# supplements (424B*) and free-writing prospectuses (FWP) are left out: large
# bank issuers file hundreds of them a month for structured notes, and they
# would push every 10-K and 10-Q out of the list.
_INTERESTING_FORMS = (
    "10-K", "10-Q", "8-K", "20-F", "40-F", "6-K", "DEF 14A", "S-1",
)

_HEADERS = {"User-Agent": config.SEC_USER_AGENT}

# Character budgets so a 300-page 10-K doesn't blow the AI context.
_SECTION_CAPS = {"risk_factors": 60_000, "mdna": 80_000, "business": 40_000}
_WHOLE_DOC_CAP = 60_000
_TOTAL_CAP = 150_000

_cache_lock = threading.Lock()
_text_cache: dict[str, tuple[float, str]] = {}  # url -> (ts, text)
_TEXT_TTL = 3600.0

_edgar = EdgarClient()


class _TextExtractor(HTMLParser):
    """Strip an SEC HTML/iXBRL document down to readable text."""

    _SKIP = {"script", "style", "head", "title"}
    _BLOCK = {"p", "div", "tr", "table", "br", "li", "h1", "h2", "h3", "h4"}
    _CELL = {"td", "th"}  # need a separator or table numbers concatenate

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in self._CELL:
            self.parts.append("  ")
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip_depth and data:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        raw = re.sub(r"[ \t\xa0]+", " ", raw)
        raw = re.sub(r"\n\s*\n+", "\n\n", raw)
        return raw.strip()


def _html_to_text(html: str) -> str:
    # Drop the iXBRL header (huge hidden metadata block) before parsing.
    html = re.sub(
        r"<ix:header.*?</ix:header>", " ", html, flags=re.S | re.I
    )
    p = _TextExtractor()
    try:
        p.feed(html)
    except Exception:  # noqa: BLE001 - malformed HTML; use what we got
        pass
    return p.text()


def list_filings(ticker: str, limit: int = 40) -> dict:
    """Recent interesting filings for `ticker` (see `_INTERESTING_FORMS`),
    newest first, at most `limit` of them."""
    cik, name = _edgar.resolve_cik(ticker)
    url = _SUBMISSIONS_URL.format(cik=cik)
    r = requests.get(url, headers=_HEADERS, timeout=config.SEC_REQUEST_TIMEOUT)
    if r.status_code != 200:
        raise DataError(f"EDGAR submissions returned HTTP {r.status_code}")
    data = r.json()
    recent = (data.get("filings") or {}).get("recent") or {}

    forms = recent.get("form") or []
    out: list[dict] = []
    for i, form in enumerate(forms):
        base = form.split("/")[0]
        if not any(base.startswith(f) for f in _INTERESTING_FORMS):
            continue
        acc = (recent.get("accessionNumber") or [""])[i]
        doc = (recent.get("primaryDocument") or [""])[i]
        out.append(
            {
                "form": form,
                "filed": (recent.get("filingDate") or [""])[i],
                "report_date": (recent.get("reportDate") or [""])[i],
                "accession_number": acc,
                "primary_document": doc,
                "description": (recent.get("primaryDocDescription") or [""])[i],
                "url": _DOC_URL.format(
                    cik_int=int(cik), acc_nodash=acc.replace("-", ""), doc=doc
                ),
            }
        )
        if len(out) >= limit:
            break
    return {"ticker": ticker.upper(), "cik": cik, "name": name, "filings": out}


def fetch_filing_text(ticker: str, accession_number: str, primary_document: str) -> str:
    """Download a filing's primary document and return clean text (cached)."""
    cik, _ = _edgar.resolve_cik(ticker)
    url = _DOC_URL.format(
        cik_int=int(cik),
        acc_nodash=accession_number.replace("-", ""),
        doc=primary_document,
    )
    now = time.time()
    with _cache_lock:
        hit = _text_cache.get(url)
        if hit and now - hit[0] < _TEXT_TTL:
            return hit[1]

    r = requests.get(url, headers=_HEADERS, timeout=60)
    if r.status_code != 200:
        raise DataError(f"EDGAR document returned HTTP {r.status_code} ({url})")
    text = _html_to_text(r.text) if "<" in r.text[:1000] else r.text

    with _cache_lock:
        _text_cache[url] = (now, text)
    return text


_ITEM_PATTERNS = {
    # (start pattern, [next-item patterns]) — matched case-insensitively on
    # line-ish boundaries. SEC docs repeat headings in the TOC, so we take the
    # LAST match of the start pattern (the actual section, not the TOC entry).
    "risk_factors": (r"item\s+1a\.?\s*[—:\-–\s]*risk\s+factors",
                     [r"item\s+1b\.?", r"item\s+2\.?\s"]),
    "business": (r"item\s+1\.?\s*[—:\-–\s]*business",
                 [r"item\s+1a\.?"]),
    "mdna": (r"item\s+[27]\.?\s*[—:\-–\s]*management.{0,5}s\s+discussion",
             [r"item\s+[38]\.?\s*[—:\-–\s]*(quantitative|financial\s+statements)",
              r"item\s+7a\.?", r"item\s+3\.?\s"]),
}


def _at_line_start(text: str, pos: int) -> bool:
    """True if a heading match at `pos` begins a line (only whitespace before
    it on that line). Mid-sentence cross-references ("see Item 1A ...") fail."""
    before = text[max(0, pos - 2) : pos]
    return pos == 0 or before.rstrip(" \t") == "" or "\n" in before


def extract_sections(text: str, form: str) -> dict[str, str]:
    """Pull the analyst-relevant sections out of a 10-K/10-Q; for short forms
    (8-K etc.) return the whole document capped."""
    base = form.split("/")[0].upper()
    if base not in ("10-K", "10-Q", "20-F"):
        return {"document": text[:_WHOLE_DOC_CAP]}

    low = text.lower()
    sections: dict[str, str] = {}
    for key, (start_pat, end_pats) in _ITEM_PATTERNS.items():
        starts = [m.start() for m in re.finditer(start_pat, low)]
        if not starts:
            continue
        # Prefer matches at a line start — mid-sentence cross-references
        # ("see Item 1A. Risk Factors") and TOC entries are then excluded;
        # among line-start matches the LAST one is the section body (the TOC
        # comes first). Fall back to the raw last match if none qualify.
        line_starts = [s for s in starts if _at_line_start(text, s)]
        start = (line_starts or starts)[-1]
        # End at the EARLIEST next-item heading across all end patterns —
        # taking the first pattern that matches (in list order) can fold a
        # later section (e.g. Item 7A) into this one. Like the start, the end
        # must be a heading at a line start: in-text cross-references ("see
        # Item 1A", "in Item 8. Financial Statements") would otherwise cut
        # the section short. Fall back to any match if no heading qualifies.
        end = len(text)
        loose_end = len(text)
        for ep in end_pats:
            for m in re.finditer(ep, low[start + 50 :]):
                pos = start + 50 + m.start()
                loose_end = min(loose_end, pos)
                if _at_line_start(text, pos):
                    end = min(end, pos)
                    break
        if end == len(text):
            end = loose_end
        chunk = text[start:end].strip()
        if len(chunk) > 500:  # ignore degenerate matches
            sections[key] = chunk[: _SECTION_CAPS.get(key, 50_000)]

    if not sections:
        sections["document"] = text[:_WHOLE_DOC_CAP]
    return sections


def build_filing_material(ticker: str, form: str, filed: str,
                          accession_number: str, primary_document: str) -> tuple[str, dict]:
    """Fetch + sectionize a filing into AI-ready material text.

    Returns (material_text, meta) where meta reports what was extracted."""
    text = fetch_filing_text(ticker, accession_number, primary_document)
    sections = extract_sections(text, form)

    titles = {
        "business": "ITEM 1 — BUSINESS",
        "risk_factors": "ITEM 1A — RISK FACTORS",
        "mdna": "MANAGEMENT'S DISCUSSION & ANALYSIS",
        "document": "FILING TEXT",
    }
    parts = [f"SEC FILING: {form} for {ticker.upper()}, filed {filed} "
             f"(accession {accession_number})"]
    total = len(parts[0])
    for key, chunk in sections.items():
        room = _TOTAL_CAP - total
        if room <= 0:
            break
        piece = chunk[:room]
        parts.append(f"\n\n===== {titles.get(key, key.upper())} =====\n\n{piece}")
        total += len(piece)

    meta = {
        "form": form,
        "filed": filed,
        "doc_chars": len(text),
        "sections": {k: len(v) for k, v in sections.items()},
        "material_chars": total,
    }
    return "".join(parts), meta
