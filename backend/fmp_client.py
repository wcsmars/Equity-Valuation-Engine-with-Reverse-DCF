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

    @staticmethod
    def _aliases(row: Optional[dict], aliases: dict[str, str]) -> Optional[dict]:
        """Keep the dashboard contract while accepting FMP's stable field names."""
        if row is None:
            return None
        out = dict(row)
        for old, current in aliases.items():
            if out.get(old) is None and current in out:
                out[old] = out[current]
        return out

    # --- individual endpoints ---------------------------------------------- #
    def profile(self, ticker: str) -> Optional[dict]:
        return self._first(self._get("/stable/profile", {"symbol": ticker.upper()}))

    def quote(self, ticker: str) -> Optional[dict]:
        return self._first(self._get("/stable/quote", {"symbol": ticker.upper()}))

    def ratios_ttm(self, ticker: str) -> Optional[dict]:
        return self._aliases(
            self._first(self._get("/stable/ratios-ttm", {"symbol": ticker.upper()})),
            {"peRatioTTM": "priceToEarningsRatioTTM",
             "priceToFreeCashFlowsRatioTTM": "priceToFreeCashFlowRatioTTM",
             "payoutRatioTTM": "dividendPayoutRatioTTM",
             "debtEquityRatioTTM": "debtToEquityRatioTTM"},
        )

    def key_metrics_ttm(self, ticker: str) -> Optional[dict]:
        return self._aliases(
            self._first(self._get("/stable/key-metrics-ttm", {"symbol": ticker.upper()})),
            {"roicTTM": "returnOnInvestedCapitalTTM"},
        )

    def price_target(self, ticker: str) -> Optional[dict]:
        return self._first(
            self._get("/stable/price-target-consensus", {"symbol": ticker.upper()})
        )

    def rating(self, ticker: str) -> Optional[dict]:
        return self._first(self._get("/stable/ratings-snapshot", {"symbol": ticker.upper()}))

    def peers(self, ticker: str) -> list[str]:
        data = self._get("/stable/stock-peers", {"symbol": ticker.upper()})
        rows = data if isinstance(data, list) else [data]
        symbols: list[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            # Stable returns one {symbol, companyName, ...} row per peer.
            # Retain support for a cached legacy response's peersList too.
            candidates = row.get("peersList")
            if not isinstance(candidates, list):
                candidates = [row.get("symbol")]
            for value in candidates:
                if isinstance(value, str) and value.strip():
                    symbol = value.strip().upper()
                    if symbol != ticker.strip().upper() and symbol not in symbols:
                        symbols.append(symbol)
        return symbols

    def news(self, ticker: str, limit: int = 30) -> list[dict]:
        data = self._get("/stable/news/stock", {"symbols": ticker.upper(), "limit": limit})
        rows = []
        for row in data if isinstance(data, list) else []:
            if not isinstance(row, dict):
                continue
            normalized = self._aliases(row, {"site": "publisher"})
            for key in ("title", "text", "url", "site", "publishedDate", "image"):
                if key in normalized and not isinstance(normalized[key], str):
                    normalized[key] = None
            rows.append(normalized)
        return rows

    def analyst_estimates(self, ticker: str, limit: int = 6) -> list[dict]:
        data = self._get(
            "/stable/analyst-estimates",
            {"symbol": ticker.upper(), "limit": limit, "period": "annual", "page": 0},
        )
        # Amounts use the reporting currency, which may differ from the quote.
        # Do not attach the profile's quote currency to these financial estimates.
        return [self._aliases(row, {"estimatedRevenueAvg": "revenueAvg",
                                   "estimatedEpsAvg": "epsAvg"})
                for row in data if isinstance(row, dict)] if isinstance(data, list) else []

    def transcripts_list(self, ticker: str, limit: int = 12) -> list[dict]:
        """Available earnings-call transcripts as [{quarter, year, date}]."""
        data = self._get(
            "/stable/earning-call-transcript-dates", {"symbol": ticker.upper()}
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
                            "year": row.get("fiscalYear", row.get("year")),
                            "date": str(row.get("date", "")),
                        }
                    )
        return out

    def transcript(self, ticker: str, year: int, quarter: int) -> Optional[dict]:
        """One earnings-call transcript {content, date, ...} or None."""
        data = self._get(
            "/stable/earning-call-transcript",
            {"symbol": ticker.upper(), "year": year, "quarter": quarter},
        )
        return self._first(data)

    # --- aggregate --------------------------------------------------------- #
    def enrichment(self, ticker: str) -> dict:
        """Everything the Multiples + News tabs need, in one call."""
        ticker = ticker.strip().upper()
        if not self.enabled:
            return {"enabled": False}
        ratios = self.ratios_ttm(ticker)
        metrics = self.key_metrics_ttm(ticker)
        # ROE moved from ratios to key metrics in the stable API.
        if metrics and metrics.get("returnOnEquityTTM") is not None:
            ratios = dict(ratios or {})
            if ratios.get("returnOnEquityTTM") is None:
                ratios["returnOnEquityTTM"] = metrics["returnOnEquityTTM"]
        return {
            "enabled": True,
            "profile": self.profile(ticker),
            "quote": self.quote(ticker),
            "ratios_ttm": ratios,
            "key_metrics_ttm": metrics,
            "price_target": self.price_target(ticker),
            "rating": self.rating(ticker),
            "peers": self.peers(ticker),
            "estimates": self.analyst_estimates(ticker),
            "news": self.news(ticker),
        }
