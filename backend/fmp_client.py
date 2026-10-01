"""Financial Modeling Prep enrichment layer.

The valuation engine runs on free EDGAR + yfinance data. FMP is used only where
it's clearly better: a real news feed, analyst estimates / price targets, and a
richer set of display multiples. Everything degrades gracefully — if no
FMP_API_KEY is set (or a call fails / the plan doesn't cover an endpoint), the
field comes back empty and the UI shows a "connect FMP" hint instead of breaking.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import requests

DEFAULT_BASE = os.environ.get("FMP_BASE_URL", "https://financialmodelingprep.com")


class FMPClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base: str = DEFAULT_BASE,
        timeout: float = 12.0,
    ):
        self.api_key = api_key or os.environ.get("FMP_API_KEY")
        self.base = base.rstrip("/")
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        if not self.enabled:
            return None
        params = dict(params or {})
        params["apikey"] = self.api_key
        try:
            r = requests.get(
                f"{self.base}{path}", params=params, timeout=self.timeout
            )
            if r.status_code != 200:
                return None
            data = r.json()
            # FMP returns {"Error Message": ...} on bad symbol / plan limits.
            if isinstance(data, dict) and ("Error Message" in data or "error" in data):
                return None
            return data
        except Exception:  # noqa: BLE001 - enrichment is best-effort
            return None

    @staticmethod
    def _first(data: Any) -> Optional[dict]:
        if isinstance(data, list) and data:
            return data[0] if isinstance(data[0], dict) else None
        if isinstance(data, dict):
            return data
        return None

    # --- individual endpoints ---------------------------------------------- #
    def profile(self, ticker: str) -> Optional[dict]:
        return self._first(self._get(f"/api/v3/profile/{ticker}"))

    def quote(self, ticker: str) -> Optional[dict]:
        return self._first(self._get(f"/api/v3/quote/{ticker}"))

    def ratios_ttm(self, ticker: str) -> Optional[dict]:
        return self._first(self._get(f"/api/v3/ratios-ttm/{ticker}"))

    def key_metrics_ttm(self, ticker: str) -> Optional[dict]:
        return self._first(self._get(f"/api/v3/key-metrics-ttm/{ticker}"))

    def price_target(self, ticker: str) -> Optional[dict]:
        return self._first(
            self._get("/api/v4/price-target-consensus", {"symbol": ticker})
        )

    def rating(self, ticker: str) -> Optional[dict]:
        return self._first(self._get(f"/api/v3/rating/{ticker}"))

    def peers(self, ticker: str) -> list[str]:
        data = self._first(self._get("/api/v4/stock_peers", {"symbol": ticker}))
        if data and isinstance(data.get("peersList"), list):
            return [str(p) for p in data["peersList"]]
        return []

    def news(self, ticker: str, limit: int = 30) -> list[dict]:
        data = self._get("/api/v3/stock_news", {"tickers": ticker, "limit": limit})
        return [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []

    def analyst_estimates(self, ticker: str, limit: int = 6) -> list[dict]:
        data = self._get(
            f"/api/v3/analyst-estimates/{ticker}", {"limit": limit, "period": "annual"}
        )
        return [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []

    def transcripts_list(self, ticker: str, limit: int = 12) -> list[dict]:
        """Available earnings-call transcripts as [{quarter, year, date}]."""
        data = self._get(
            "/api/v4/earning_call_transcript", {"symbol": ticker.upper()}
        )
        out: list[dict] = []
        if isinstance(data, list):
            for row in data[:limit]:
                # FMP returns rows as [quarter, year, date]
                if isinstance(row, (list, tuple)) and len(row) >= 3:
                    out.append(
                        {"quarter": row[0], "year": row[1], "date": str(row[2])}
                    )
                elif isinstance(row, dict):
                    out.append(
                        {
                            "quarter": row.get("quarter"),
                            "year": row.get("year"),
                            "date": str(row.get("date", "")),
                        }
                    )
        return out

    def transcript(self, ticker: str, year: int, quarter: int) -> Optional[dict]:
        """One earnings-call transcript {content, date, ...} or None."""
        data = self._get(
            f"/api/v3/earning_call_transcript/{ticker.upper()}",
            {"year": year, "quarter": quarter},
        )
        return self._first(data)

    # --- aggregate --------------------------------------------------------- #
    def enrichment(self, ticker: str) -> dict:
        """Everything the Multiples + News tabs need, in one call."""
        ticker = ticker.strip().upper()
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "profile": self.profile(ticker),
            "quote": self.quote(ticker),
            "ratios_ttm": self.ratios_ttm(ticker),
            "key_metrics_ttm": self.key_metrics_ttm(ticker),
            "price_target": self.price_target(ticker),
            "rating": self.rating(ticker),
            "peers": self.peers(ticker),
            "estimates": self.analyst_estimates(ticker),
            "news": self.news(ticker),
        }
