"""Live market data + trading-comps multiples via yfinance.

`YFinanceClient` supplies the *market* side of the world (price, market cap, beta,
annual dividend, 52-week range, sector/industry) and the trailing valuation
multiples used by the comps model. It can also, as a fallback, reconstruct a rough
`AnnualFinancials` + `BalanceSheetSnapshot` from yfinance's statement DataFrames
for non-US issuers that SEC EDGAR cannot serve.

Currencies
----------
* Market data is returned in the MAJOR unit of the quote currency: Yahoo quotes
  London, Johannesburg and Tel Aviv listings in pence / cents / agorot (``GBp``,
  ``ZAc``, ``ILA``), which are converted to GBP / ZAR / ILS. Yahoo's
  ``dividendRate`` for those lines is already in the major unit, while the
  dividend payment history is in the quote's minor unit.
* ``trailingAnnualDividendRate`` is per ORDINARY share in the reporting
  currency (TSM: 26.0 TWD, BP ADR: 0.336 USD per share, 2.02 per ADS), so it
  is only a last-resort D0 and only when that currency is the quote currency.
  Known gap: the ADR of a USD-reporting issuer (BP) passes that check with a
  per-ordinary-share figure, since Yahoo does not mark ADRs. The field is
  reached only when both ``dividendRate`` and the payment history are missing.
* yfinance statements are in the issuer's reporting currency
  (``info['financialCurrency']``), e.g. TWD for the USD-quoted TSM ADR. The
  fallback converts them into the quote currency at one spot FX rate (a Yahoo
  ``XXXYYY=X`` quote) so price and fundamentals are comparable; if no rate can
  be fetched it raises DataError, preventing valuations across mixed units.
  Exception: for some 20-F filers the statement tables hold the USD 20-F
  figures while ``financialCurrency`` and the ``.info`` amounts stay local
  (VALE and PBR, also on their Sao Paulo lines: BRL; YPF: ARS). Comparing a
  table's Total Debt and Total Revenue with ``.info``'s ``totalDebt`` and
  ``totalRevenue`` shows this (see `YFinanceClient._tables_currency`), and the
  tables are then read in that currency, with a note.
* Yahoo's ``.info`` enterprise value and EV multiples mix the quote currency
  (market cap) with the reporting currency (debt, cash, revenue, EBITDA), and
  for some ADRs the ordinary-share and ADS bases (TSM EV/EBITDA 5.1 instead of
  about 23; the BP ADR's EV 510B against about 151B). `get_comp_row` rebuilds
  them from market cap, total debt, total cash, minority interest, preferred
  stock, revenue and EBITDA at one spot FX rate, and P/B from market cap and
  the balance sheet's common equity (or drops it) where Yahoo's book value
  per share is not on the quoted basis. A same-currency EV that only minority
  interest could explain is kept when Yahoo's balance sheet cannot be read.
* Data-quality notes ride on the returned objects as a ``_source_notes``
  attribute (MarketData, the fallback's AnnualFinancials and rebuilt
  CompRows); HybridProvider copies the first two into
  ``CompanyData.source_notes``.

Beta
----
* Yahoo's beta is a raw 5-year monthly regression beta. `get_market_data`
  Blume-adjusts it (0.67 x raw + 0.33) toward the market's 1.0, keeps the
  raw figure in ``MarketData.raw_beta`` and notes both. A beta <= 0 is not
  usable for CAPM and is dropped instead, so the models use
  ``config.DEFAULT_BETA``.

Captive finance arms (statement fallback)
-----------------------------------------
* A durable-goods maker that consolidates a finance and leasing arm (Toyota
  Financial Services, Honda Finance) is marked like the EDGAR client marks
  one: ``financials._financial_kind = "captive_finance"`` and a leading
  WARNING note. Yahoo's statements show the arm reliably only through large
  non-current receivables funded with debt, or, for automakers and
  farm/heavy-machinery makers, receivables and debt that are both a large
  share of total assets (see `_captive_finance_evidence`). Where Yahoo lumps
  the finance book into unlabelled lines (BMW, Mercedes-Benz, Hyundai) an
  automaker is not marked but gets a note.

Design notes
------------
* yfinance is imported LAZILY inside each method so that merely importing this
  module (and the wider package) never requires the dependency. EDGAR-only use
  therefore works without yfinance installed.
* yfinance's `.info` dictionary is notoriously sparse and inconsistent — many keys
  are absent for any given ticker, ETFs, or non-US listings. EVERY access goes
  through `.get(...)` with a default and `_num()` coercion so a missing/`NaN`/
  string field degrades to `None` instead of crashing.
* `get_market_data` raises `DataError` when a price is genuinely unobtainable,
  since price underpins the whole valuation.
  The fundamentals fallback also raises DataError if its known reporting
  currency cannot be converted to the quote currency. Optional peer enrichment
  returns `None` / `[]` on failure.
"""

from __future__ import annotations

import dataclasses
import datetime
import math
import statistics
from typing import Optional

from ..schemas import (
    AnnualFinancials,
    BalanceSheetSnapshot,
    CompRow,
    MarketData,
)
from ..utils import is_num
from .base import DataError

# Yahoo quote currencies expressed in minor units -> (major ISO code, factor).
_MINOR_UNITS = {
    "GBp": ("GBP", 0.01),
    "GBX": ("GBP", 0.01),
    "ZAc": ("ZAR", 0.01),
    "ILA": ("ILS", 0.01),
}

# sharesOutstanding more than this far from marketCap/price is on a different
# share basis (one class of a multi-class issuer, or ordinary shares vs ADSs).
_SHARE_MISMATCH_TOL = 0.05

# D0 from the five-year dividend payment history (see `_read_dividends`). A
# payment above `_OUTSIZED_PAYMENT` times the median payment within a year
# either side of it is a special or variable dividend. Such payments RECUR when
# they fall in at least two separate years (payments under `_DRIFT_DAYS` apart
# share a year) of the `_VARIABLE_YEARS` years ending with the latest one, and
# are current while that one is under `_VARIABLE_CURRENT_DAYS` old. Anchoring
# the years on the payments rather than on today keeps D0 steady as ex-dates
# drift (CME's annual variable dividend moved from late December to March).
_HISTORY_PERIOD = "5y"
_TTM_DAYS = 365
_OUTSIZED_PAYMENT = 2.0
_VARIABLE_YEARS = 3
_DRIFT_DAYS = 183
_VARIABLE_CURRENT_DAYS = 548
# Trailing-12-month dividends this far above the indicated rate earn a note.
_D0_TTM_MARGIN = 1.25

# A dividend implying a yield above this is taken to be in the quote's minor
# unit (pence) rather than Yahoo's usual major unit.
_MAX_PLAUSIBLE_YIELD = 0.5

# Blume adjustment of Yahoo's raw beta: regression betas drift toward the
# market's 1.0, so CAPM uses 0.67 x raw + 0.33 (the "adjusted beta" data
# services quote). NVDA 2.22 -> 1.82, JNJ 0.24 -> 0.49.
_BLUME_RAW_WEIGHT = 0.67
_BLUME_MARKET_WEIGHT = 0.33

# Comps rows: Yahoo's enterprise value more than this far from market cap +
# (total debt - total cash), with minority interest and preferred stock added
# when that explains the gap (BN, KKR), is on another currency or share basis
# (BP ADR, DEO), so the EV, EV multiples and P/B are rebuilt; they always are
# when the reporting currency is not the quote currency.
_COMP_EV_TOLERANCE = 0.25
# Without Yahoo's balance sheet minority interest cannot be checked: a
# same-currency Yahoo EV above that check by up to this factor could be
# minority interest and preferred stock (BN 1.37x) and is kept; beyond it (BP
# ADR 3.4x, DEO 12.8x), below the check or with the other sign it is not.
_COMP_EV_MINORITY_MAX = 2.0

# Statement tables in another currency than financialCurrency (see
# `YFinanceClient._tables_currency`). Both table/.info ratios must lie within
# this factor of the financialCurrency->X rate (VALE: 0.202 and 0.176
# against BRLUSD 0.192), and X must be at least `_TABLE_CCY_MIN_GAP` away
# from financialCurrency: nearer pairs (EUR, GBP or CHF against USD) are not
# told apart from the gaps between the table and .info dates and definitions
# (ULVR.L's table debt is 0.86 of totalDebt, EUR->GBP is 0.87).
_TABLE_CCY_BAND = 1.5
_TABLE_CCY_MIN_GAP = 2.0

# Captive finance arms in yfinance statements (see `_captive_finance_evidence`).
# Sectors whose members sell durable goods on credit (autos, motorcycles,
# machinery, trucks, IT equipment, timeshares); banks, lessors and utilities
# with concession receivables are outside them.
_CAPTIVE_SECTORS = ("consumer cyclical", "industrials", "technology")
# Non-current receivables at least this share of total assets (TM 24%, HMC
# 20%; Dell 6%, HPE 8%, IBM 5%, NVO 1%) and funded at least half by debt.
_CAPTIVE_MIN_NONCURRENT_SHARE = 0.15
_CAPTIVE_MIN_DEBT_FUNDING = 0.5
# Industries where Yahoo often lumps the finance book into current receivables
# (Renault 48%, Nissan 40%, Deere 54%): current plus non-current receivables
# and total debt at least these shares of total assets (Ferrari 22%, Komatsu
# 22%, Daimler Truck 21%, Tesla 3% stay below). The same industries get a
# note when nothing is found, since Yahoo hides some arms (BMW, Hyundai).
_CAPTIVE_INDUSTRIES = ("auto manufacturers", "farm & heavy construction machinery",
                       "recreational vehicles")
_CAPTIVE_MIN_RECEIVABLES_SHARE = 0.35
_CAPTIVE_MIN_DEBT_SHARE = 0.30

# Balance-sheet backfill (HybridProvider, for EDGAR filers whose debt is under
# company-specific tags): the Yahoo column must be this close to EDGAR's date.
# Public because the provider's note quotes it.
BACKFILL_MAX_DAYS = 100

# Capital expenditure rows of Yahoo's cash-flow statement, in order of
# preference: its standard line (purchases of property, plant and equipment,
# plus intangibles where Yahoo counts them), else the PP&E purchases alone.
# Read by the statement fallback and by `YFinanceClient.get_capex_by_year`
# (HybridProvider's backfill of EDGAR years without a capex tag).
_CAPEX_ROWS = ("Capital Expenditure", "CapitalExpenditure", "Purchase Of PPE")

# Monetary fields scaled by an FX conversion (share counts and years are not).
_MONEY_SERIES = (
    "revenue", "ebit", "ebitda", "net_income", "dep_amort", "capex",
    "change_in_nwc", "interest_expense", "tax_expense", "pretax_income",
    "dividends_paid",
)
_MONEY_BALANCE = (
    "total_debt", "cash_and_investments", "total_equity", "minority_interest",
    "preferred_equity",
)


# --------------------------------------------------------------------------- #
#  Small coercion helpers (kept local; these are yfinance-specific janitorial
#  chores rather than reusable numeric primitives).
# --------------------------------------------------------------------------- #
def _num(x: object) -> Optional[float]:
    """Coerce a yfinance value to a finite float, or None.

    yfinance frequently returns strings ('Infinity'), NaNs, None, or sentinel
    zeros for missing fields. We accept only genuinely finite real numbers.
    """
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError, OverflowError):
        return None
    return v if is_num(v) else None


def _pos(x: object) -> Optional[float]:
    """Like `_num` but also rejects non-positive values.

    Useful for fields where a non-positive number is meaningless (a price, a
    P/E we intend to display, shares outstanding, etc.).
    """
    v = _num(x)
    return v if (v is not None and v > 0) else None


def _str(x: object) -> Optional[str]:
    """Coerce to a non-empty stripped string, or None."""
    if x is None:
        return None
    try:
        s = str(x).strip()
    except Exception:  # pragma: no cover - defensive
        return None
    return s or None


def major_currency(code: Optional[str]) -> tuple[Optional[str], float]:
    """``(major ISO code, factor)`` for a Yahoo currency code.

    ``GBp`` (pence) -> ``('GBP', 0.01)``, ``ZAc`` -> ``('ZAR', 0.01)``,
    ``ILA`` -> ``('ILS', 0.01)``; anything else is already a major unit and is
    returned upper-cased with factor 1.0. The minor codes are case-sensitive
    (``GBp`` is pence, ``GBP`` is pounds).
    """
    s = _str(code)
    if s is None:
        return None, 1.0
    if s in _MINOR_UNITS:
        return _MINOR_UNITS[s]
    return s.upper(), 1.0


def blume_adjusted_beta(raw: float) -> float:
    """Blume-adjusted beta: ``0.67 x raw + 0.33`` (pulled toward 1.0)."""
    return _BLUME_RAW_WEIGHT * raw + _BLUME_MARKET_WEIGHT


def captive_finance_warning(source: str, evidence: str) -> str:
    """The leading WARNING for a group with a consolidated captive finance arm.

    Worded like the EDGAR client's (``WARNING: EDGAR tags mark this company as
    a group with a captive finance arm ...``), with `source` naming what
    showed it (e.g. "yfinance statements") and `evidence` the shares found.
    """
    return (
        f"WARNING: {source} mark this company as a group with a captive finance arm "
        f"({evidence}); the finance arm's debt, leases and receivables are consolidated "
        "(total debt includes the borrowing that funds its customer loans and leases), "
        "so the FCFF DCF and FCFE do not fit it and the DDM and comps are the better guides"
    )


def _today() -> datetime.date:
    """Today's date; the reference for the trailing-12-month dividend windows."""
    return datetime.date.today()


@dataclasses.dataclass(frozen=True)
class _DividendRecord:
    """What a payment history says about D0 (amounts in the quote's major unit).

    ``ttm`` is the cash paid with an ex-date in the last 12 months and
    ``regular_ttm`` the same without the special/variable part. The special
    tuples hold ``(ex-date, amount above the regular dividend)``: ``ttm_specials``
    those in the last 12 months, ``recurring`` those averaged into ``variable``
    (the annual special/variable dividend; both empty/zero unless they recur
    and are current).
    """

    ttm: float
    regular_ttm: float
    ttm_specials: tuple
    recurring: tuple
    variable: float


def _read_dividends(
    history: list[tuple[datetime.date, float]], today: datetime.date
) -> _DividendRecord:
    """Split a payment history into its regular and special/variable parts.

    * A payment is OUTSIZED (special or variable) above `_OUTSIZED_PAYMENT` x
      the median of the payments within a year either side of it (at least
      three of them), so a fast-growing dividend, or one after a cut, is
      judged against its own recent level rather than the five-year median.
    * An outsized payment with no regular ex-date within half the regular
      spacing of it is paid together with a regular dividend (PGR's 13.60 in
      January = 13.50 variable + the 0.10 quarterly; CME's 7.45 in March 2026 =
      6.15 + 1.30): only its excess over the next regular payment (else the
      previous one) is special. One on its own (CME's late-December payments,
      18 days after the quarterly) is special in full.
    * The specials recur when they fall in at least two separate years of the
      `_VARIABLE_YEARS` years ending with the latest (tolerating `_DRIFT_DAYS`
      of ex-date drift), which must be under `_VARIABLE_CURRENT_DAYS` old.
      ``variable`` is then their sum over those years / `_VARIABLE_YEARS`.

    Payments with an ex-date after `today` are ignored.
    """
    pays = sorted((d, a) for d, a in history if d <= today)

    def outsized(day: datetime.date, amount: float) -> bool:
        near = [a for d, a in pays if abs((d - day).days) <= _TTM_DAYS]
        return len(near) >= 3 and amount > _OUTSIZED_PAYMENT * statistics.median(near)

    flags = [outsized(d, a) for d, a in pays]
    regular = [p for p, big in zip(pays, flags) if not big]
    gaps = [(b[0] - a[0]).days for a, b in zip(regular, regular[1:])]
    half_spacing = (statistics.median(gaps) if gaps else _TTM_DAYS) / 2

    ttm = regular_ttm = 0.0
    specials: list[tuple[datetime.date, float]] = []
    ttm_specials: list[tuple[datetime.date, float]] = []
    for (day, amount), big in zip(pays, flags):
        in_ttm = (today - day).days < _TTM_DAYS
        base = amount
        if big:
            base = 0.0
            if regular and all(abs((d - day).days) > half_spacing for d, _ in regular):
                later = [a for d, a in regular if d > day]
                base = later[0] if later else regular[-1][1]
            specials.append((day, amount - base))
            if in_ttm:
                ttm_specials.append(specials[-1])
        if in_ttm:
            ttm += amount
            regular_ttm += base

    recurring: list[tuple[datetime.date, float]] = []
    if specials and (today - specials[-1][0]).days < _VARIABLE_CURRENT_DAYS:
        last = specials[-1][0]
        span = _VARIABLE_YEARS * _TTM_DAYS - _DRIFT_DAYS
        window = [s for s in specials if (last - s[0]).days < span]
        years = 1 + sum((b[0] - a[0]).days >= _DRIFT_DAYS for a, b in zip(window, window[1:]))
        if years >= 2:
            recurring = window
    return _DividendRecord(
        ttm=ttm,
        regular_ttm=regular_ttm,
        ttm_specials=tuple(ttm_specials),
        recurring=tuple(recurring),
        variable=sum(x for _, x in recurring) / _VARIABLE_YEARS,
    )


def _specials_text(specials: tuple) -> str:
    """``'0.75 on 2024-01-18, 4.5 on 2025-01-10'`` for a D0 note."""
    return ", ".join(f"{x:,.4g} on {d.isoformat()}" for d, x in specials)


def _dividend_to_major(
    amount: float, price: float, quote_ccy: str, field: str, notes: list[str]
) -> float:
    """A Yahoo dividend rate in the quote currency's MAJOR unit.

    Yahoo reports ``dividendRate`` in the major unit even when the quote is in
    minor units: BP.L shows 0.25 (GBP) at a 566.3p price and a 4.49% yield.
    Only for a minor-unit quote whose amount would imply a yield above
    `_MAX_PLAUSIBLE_YIELD` of the (major-unit) `price` is the amount read as
    minor units and converted, with a note.
    """
    major, unit = major_currency(quote_ccy)
    if unit != 1.0 and price > 0 and amount > _MAX_PLAUSIBLE_YIELD * price:
        notes.append(
            f"Yahoo {field} ({amount:,.4g}) would be a yield above "
            f"{_MAX_PLAUSIBLE_YIELD:.0%} in {major}; read as {quote_ccy} and converted"
        )
        return amount * unit
    return amount


def _unit_scale(ratio_if_minor: Optional[float], factor: float) -> float:
    """Multiplier that takes a Yahoo amount to the MAJOR currency unit.

    For a minor-unit quote (``factor`` 0.01) Yahoo is not consistent about
    whether the market cap is in the minor or the major unit (dividend rates
    are handled by :func:`_dividend_to_major`). `ratio_if_minor` compares the
    amount with the same quantity rebuilt from the minor-unit price: ~1 means
    it is in minor units (scale by `factor`), ~`factor` means it is already
    major (scale 1.0). The closer reading wins; without a comparison the
    amount is taken to be in the quote's own (minor) unit.
    """
    if factor == 1.0 or ratio_if_minor is None or not ratio_if_minor > 0:
        return factor
    if abs(math.log(ratio_if_minor / factor)) < abs(math.log(ratio_if_minor)):
        return 1.0
    return factor


def scale_fundamentals(
    financials: AnnualFinancials, balance_sheet: BalanceSheetSnapshot, rate: float
) -> tuple[AnnualFinancials, BalanceSheetSnapshot]:
    """Copies of the fundamentals with every monetary amount multiplied by `rate`.

    Used for FX conversion: all flow series and balance-sheet amounts are
    scaled; fiscal years and share counts are not. Of the dynamic attributes
    the financial-kind and income-attribution markers are carried over (the
    models use them); ``_source_notes`` are handled by the caller.
    """
    fin = dataclasses.replace(
        financials,
        **{k: [v * rate for v in getattr(financials, k)] for k in _MONEY_SERIES},
    )
    for marker in ("_financial_kind", "_income_attribution_adjusted"):
        if hasattr(financials, marker):
            setattr(fin, marker, getattr(financials, marker))
    bs = dataclasses.replace(
        balance_sheet,
        **{k: getattr(balance_sheet, k) * rate for k in _MONEY_BALANCE},
    )
    return fin, bs


class YFinanceClient:
    """Thin, defensive wrapper around yfinance for market data and comps."""

    # ----------------------------------------------------------------- #
    #  Internal: fetch the (.info dict, Ticker object) for a symbol.
    # ----------------------------------------------------------------- #
    def _ticker(self, ticker: str):
        """Return a yfinance Ticker object (lazy import). Raises on import error."""
        import yfinance as yf  # lazy: package imports without yfinance installed

        return yf.Ticker(ticker)

    def _info(self, tk) -> dict:
        """Best-effort `.info` dict for a Ticker; empty dict on any failure.

        `.info` triggers a network call and can throw (HTTP errors, JSON decode
        errors, rate limits) or return None — all of which we swallow here.
        """
        try:
            info = tk.info
        except Exception:
            return {}
        return info if isinstance(info, dict) else {}

    def _fast_value(self, tk, keys: tuple[str, ...], coerce=_pos):
        """First usable `fast_info` field among `keys` (e.g. last_price, shares).

        `fast_info` supports both attribute and mapping access depending on the
        yfinance version, so we try both; `coerce` validates each candidate.
        `fast_info` is a lighter endpoint (price history) than `.info`, so it
        often still works when `.info` is rate-limited. Any failure -> None.
        """
        fi = None
        try:
            fi = tk.fast_info
        except Exception:
            return None
        if fi is None:
            return None
        # Try attribute access first, then mapping-style access.
        for key in keys:
            try:
                val = coerce(getattr(fi, key))
            except Exception:
                val = None
            if val is not None:
                return val
        for key in keys:
            try:
                val = coerce(fi[key])  # type: ignore[index]
            except Exception:
                val = None
            if val is not None:
                return val
        return None

    def _fast_last_price(self, tk) -> Optional[float]:
        """Pull a last price from `fast_info`, or None."""
        return self._fast_value(tk, ("last_price", "lastPrice"))

    def _quote_currency(self, tk, info: dict) -> Optional[str]:
        """Raw Yahoo quote currency: ``info['currency']``, else fast_info's."""
        return _str(info.get("currency")) or self._fast_value(tk, ("currency",), _str)

    # ----------------------------------------------------------------- #
    #  Public: spot FX rate (never raises).
    # ----------------------------------------------------------------- #
    def get_fx_rate(self, from_ccy: str, to_ccy: str) -> Optional[tuple[float, str]]:
        """Spot units of `to_ccy` per one `from_ccy`, plus the quote(s) used.

        Tries Yahoo's direct pair (``TWDUSD=X``), then the inverse pair
        (``1/USDTWD=X``), then a cross through USD. Returns None when no
        positive rate can be obtained. Never raises.
        """
        f, t = (_str(from_ccy) or "").upper(), (_str(to_ccy) or "").upper()
        if not f or not t:
            return None
        if f == t:
            return 1.0, "same currency"
        hit = self._fx_pair(f, t)
        if hit is not None:
            return hit
        if "USD" not in (f, t):
            a, b = self._fx_pair(f, "USD"), self._fx_pair("USD", t)
            if a is not None and b is not None:
                return a[0] * b[0], f"{a[1]} x {b[1]}"
        return None

    def _fx_pair(self, f: str, t: str) -> Optional[tuple[float, str]]:
        """`t` per one `f` from the direct Yahoo pair or its inverse, or None."""
        for sym, invert in ((f"{f}{t}=X", False), (f"{t}{f}=X", True)):
            try:
                tk = self._ticker(sym)
            except Exception:
                continue
            px = self._fast_last_price(tk)
            if px is None:
                info = self._info(tk)
                px = _pos(info.get("regularMarketPrice")) or _pos(info.get("previousClose"))
            if px is not None:
                return (1.0 / px, f"1/{sym}") if invert else (px, sym)
        return None

    # ----------------------------------------------------------------- #
    #  Public: live market data for the target.
    # ----------------------------------------------------------------- #
    def get_market_data(self, ticker: str) -> MarketData:
        """Return live :class:`MarketData` for ``ticker``.

        Field mapping (first non-None wins):
          price              <- info.currentPrice -> fast_info.last_price -> previousClose
          shares_outstanding <- info.sharesOutstanding -> fast_info.shares, replaced by
                                marketCap/price when they differ by >5% (multi-class
                                issuers / ADRs report one class or ordinary shares)
          market_cap         <- info.marketCap (fallback price * shares)
          beta               <- info.beta Blume-adjusted (0.67 x raw + 0.33), with a
                                note; a beta <= 0 is dropped with a note, so the
                                models use config.DEFAULT_BETA
          raw_beta           <- info.beta as Yahoo reports it (None when dropped)
          dividend_per_share <- D0 for the DDM (see :meth:`_dividend_per_share`):
                                info.dividendRate (indicated regular rate), plus the
                                3-year average of recurring variable/special
                                dividends from the payment history
                                -> TTM regular history -> trailingAnnualDividendRate
          52wk low/high      <- info.fiftyTwoWeekLow / fiftyTwoWeekHigh
          sector/industry    <- info.sector / industry
          currency           <- info.currency -> fast_info.currency (default 'USD'),
                                minor units (GBp/ZAc/ILA) converted to the major unit
          name               <- info.longName -> shortName -> ticker

        When neither a share count nor a market cap is available both are left
        at 0.0 (the schema types them as float); the hybrid provider backfills
        them from the statements, with a note, or warns. Data-quality notes are
        attached as ``_source_notes`` on the returned object.

        Raises :class:`DataError` if no price can be obtained.
        """
        try:
            tk = self._ticker(ticker)
        except Exception as exc:  # yfinance missing or construction failed
            raise DataError(
                f"Could not initialize yfinance for '{ticker}': {exc}"
            ) from exc

        info = self._info(tk)
        notes: list[str] = []
        if not info:
            notes.append(
                "Yahoo quote summary (.info) unavailable; market data limited to "
                "the price endpoint (no beta, indicated dividend rate or 52-week range "
                "from Yahoo)"
            )

        # --- price: try .info fields, then the lighter fast_info endpoint ---- #
        price = _pos(info.get("currentPrice"))
        if price is None:
            price = self._fast_last_price(tk)
        if price is None:
            price = _pos(info.get("previousClose"))
        if price is None:
            price = _pos(info.get("regularMarketPrice"))
        if price is None:
            raise DataError(
                f"No obtainable price for '{ticker}' (yfinance .info/fast_info "
                "returned no usable price field)."
            )

        # --- currency: convert minor-unit quotes (pence etc.) to major ------ #
        quote_ccy = self._quote_currency(tk, info)
        if quote_ccy is None:
            quote_ccy = "USD"
            notes.append("quote currency unavailable from Yahoo; assumed USD")
        currency, unit = major_currency(quote_ccy)
        if unit != 1.0:
            notes.append(
                f"Yahoo quotes {ticker.upper()} in {quote_ccy} ({currency} minor "
                f"units); price, 52-week range and dividend history converted to "
                f"{currency} (Yahoo's dividend rate is already in {currency})"
            )
        price_quote = price  # in the quote's own (possibly minor) unit
        price = price_quote * unit

        # --- shares & market cap (with mutual fallbacks) --------------------- #
        shares = _pos(info.get("sharesOutstanding"))
        mcap_raw = _pos(info.get("marketCap"))
        if shares is None and mcap_raw is None:
            shares = self._fast_value(tk, ("shares",))
        market_cap: Optional[float] = None
        if mcap_raw is not None:
            ratio = mcap_raw / (price_quote * shares) if shares is not None else None
            market_cap = mcap_raw * _unit_scale(ratio, unit)
            implied = market_cap / price
            if shares is None:
                # Back out an implied share count so per-share math works.
                shares = implied
            elif abs(implied / shares - 1.0) > _SHARE_MISMATCH_TOL:
                # Multi-class issuers (GOOGL, BRK-B) report one class's shares
                # and ADRs sometimes ordinary shares, while marketCap covers the
                # whole company on the quoted line's basis.
                notes.append(
                    f"sharesOutstanding ({shares:,.0f}) differs from marketCap/price "
                    f"({implied:,.0f}); using marketCap/price (multi-class or ADR "
                    "share basis)"
                )
                shares = implied
        elif shares is not None:
            market_cap = price * shares
        # The dataclass types these as float: when neither is known leave 0.0;
        # HybridProvider backfills them from the statements (with a note) or
        # adds a WARNING that the market cap is unavailable.
        if shares is None or market_cap is None:
            shares, market_cap = 0.0, 0.0

        # --- dividend per share (D0 for the DDM) ---------------------------- #
        dps = self._dividend_per_share(tk, info, quote_ccy, price, notes)

        # --- beta: CAPM needs a positive one, Blume-adjusted ----------------- #
        # Yahoo's beta for BP and Shell (-0.22) would put the cost of equity
        # below the risk-free rate and the DDM's ke - g at its floor; leaving it
        # unset makes the WACC and DDM use config.DEFAULT_BETA (with their note).
        # A positive raw beta is pulled toward 1.0 (NVDA 2.22 -> 1.82, JNJ 0.24
        # -> 0.49); the raw figure stays in MarketData.raw_beta.
        beta = _num(info.get("beta"))
        raw_beta = None
        if beta is not None and beta <= 0:
            notes.append(
                f"Yahoo beta {beta:,.3g} is not usable for CAPM (not positive); left "
                "unset so the models use the default beta"
            )
            beta = None
        elif beta is not None:
            raw_beta = beta
            beta = blume_adjusted_beta(raw_beta)
            notes.append(
                f"Beta {beta:.3f} is Yahoo's 5-year monthly beta {raw_beta:.3f} "
                f"Blume-adjusted toward 1.0 ({_BLUME_RAW_WEIGHT:g} x raw + "
                f"{_BLUME_MARKET_WEIGHT:g}) for CAPM"
            )

        def _px(key: str) -> Optional[float]:
            v = _num(info.get(key))
            return v * unit if v is not None else None

        name = (
            _str(info.get("longName"))
            or _str(info.get("shortName"))
            or ticker.upper()
        )

        md = MarketData(
            ticker=ticker.upper(),
            name=name,
            currency=currency,
            price=price,
            shares_outstanding=shares,
            market_cap=market_cap,
            beta=beta,
            dividend_per_share=dps,
            fifty_two_week_low=_px("fiftyTwoWeekLow"),
            fifty_two_week_high=_px("fiftyTwoWeekHigh"),
            sector=_str(info.get("sector")),
            industry=_str(info.get("industry")),
            raw_beta=raw_beta,
        )
        md._source_notes = notes  # type: ignore[attr-defined]
        return md

    # ----------------------------------------------------------------- #
    #  Internal: dividend per share (D0) and the payment history.
    # ----------------------------------------------------------------- #
    def _dividend_per_share(
        self, tk, info: dict, quote_ccy: str, price: float, notes: list[str]
    ) -> Optional[float]:
        """Annual dividend per quoted share (D0), in the quote's major unit.

        * ``dividendRate`` (Yahoo's indicated regular annual rate) is primary.
          It is in the MAJOR unit even for minor-unit quotes (BP.L: 0.25 GBP
          at a 566p price); only a value implying a yield above 50% is read as
          minor units (see :func:`_dividend_to_major`).
        * The indicated rate leaves out variable and special dividends (PGR:
          0.40 a year while 13.90 was paid in the last 12 months). When the
          payment history shows recurring ones (see :func:`_read_dividends`)
          and the indicated rate is nearer the trailing-12-month regular
          dividends than regular plus variable (so it does not already hold
          them, as it does for HSBC's large final dividend), D0 = indicated
          rate + their annual average over three years (PGR: 0.40 + 6.25).
          A one-off special, or trailing dividends above the rate for another
          reason (a cut, supplementals, an extra ex-date), keeps the indicated
          rate. Each case is noted.
        * Without an indicated rate: the history's trailing-12-month regular
          dividends plus any recurring variable average (a one-off special is
          left out), then ``trailingAnnualDividendRate`` when the reporting
          currency is the quote currency (the field is per ordinary share in
          the reporting currency; see the module notes for the ADR gap).
          Otherwise None; HybridProvider then derives DPS from the statements.
        """
        currency, unit = major_currency(quote_ccy)
        history = self._dividend_history(tk, quote_ccy)
        rec = None
        if history:
            rec = _read_dividends([(d, a * unit) for d, a in history], _today())

        indicated = _num(info.get("dividendRate"))
        if indicated is not None:
            indicated = _dividend_to_major(indicated, price, quote_ccy, "dividendRate", notes)
            if rec is None:
                return indicated
            if rec.recurring and indicated < rec.regular_ttm + rec.variable / 2:
                d0 = indicated + rec.variable
                notes.append(
                    f"DDM D0 = Yahoo's indicated regular rate ({indicated:,.4g}) + "
                    f"{rec.variable:,.4g} {currency} a year of recurring special/variable "
                    f"dividends = {d0:,.4g} {currency} per share; the variable part averages "
                    f"{_specials_text(rec.recurring)} (amounts above the regular dividend) "
                    f"over {_VARIABLE_YEARS} years"
                )
                return d0
            if rec.ttm > _D0_TTM_MARGIN * indicated:
                if rec.ttm_specials and not rec.recurring:
                    notes.append(
                        f"trailing-12-month cash dividends of {rec.ttm:,.4g} {currency} per "
                        "share include a one-off special payment "
                        f"({_specials_text(rec.ttm_specials)} above the regular dividend; "
                        "none in the two years before); DDM D0 keeps Yahoo's indicated "
                        f"rate ({indicated:,.4g})"
                    )
                else:
                    notes.append(
                        f"trailing-12-month cash dividends of {rec.ttm:,.4g} {currency} per "
                        f"share exceed Yahoo's indicated rate ({indicated:,.4g}) (a cut, "
                        "supplemental or special payments, or an extra ex-date in the "
                        "window); DDM D0 keeps the indicated rate"
                    )
            return indicated

        if rec is not None:
            d0 = rec.regular_ttm + rec.variable
            note = (
                "Yahoo's indicated dividend rate unavailable; DDM D0 = trailing-12-month "
                f"regular cash dividends from the payment history ({rec.regular_ttm:,.4g} "
                f"{currency})"
            )
            if rec.recurring:
                note += (
                    f" + {rec.variable:,.4g} {currency} a year of recurring special/variable "
                    f"dividends ({_specials_text(rec.recurring)} above the regular dividend, "
                    f"averaged over {_VARIABLE_YEARS} years) = {d0:,.4g} {currency}"
                )
            elif rec.ttm_specials:
                note += (
                    f"; a one-off special payment ({_specials_text(rec.ttm_specials)} above "
                    "the regular dividend; none in the two years before) is left out"
                )
            notes.append(note)
            return d0
        trailing = _num(info.get("trailingAnnualDividendRate"))
        if trailing is None or trailing == 0.0:
            return trailing  # a reported zero is zero in any currency
        fin_ccy, _ = major_currency(info.get("financialCurrency"))
        if fin_ccy is not None and fin_ccy != currency:
            notes.append(
                f"Yahoo's trailing dividend rate ({trailing:,.4g}) is in {fin_ccy} per "
                f"ordinary share, not {currency} per quoted share; not used as D0"
            )
            return None
        return _dividend_to_major(trailing, price, quote_ccy, "trailingAnnualDividendRate", notes)

    def _dividend_history(
        self, tk, quote_ccy: str
    ) -> Optional[list[tuple[datetime.date, float]]]:
        """``[(ex-date, amount)]`` of the cash dividends of the last five years.

        Amounts are per quoted share (per ADS for an ADR) in the quote's own
        unit (pence for a GBp line), as Yahoo's chart history reports them.
        An empty list means no dividends in the period; None means the history
        could not be fetched, or it reports payments in a currency other than
        the quote's (yfinance then returns a frame with a currency column).
        """
        try:
            import pandas as pd  # lazy, like the statement fallback

            try:
                hist = tk.get_dividends(period=_HISTORY_PERIOD)
            except TypeError:  # yfinance without the `period` argument
                hist = tk.dividends
            if isinstance(hist, pd.DataFrame):
                ccys = hist["currency"] if "currency" in hist.columns else []
                if any(isinstance(c, str) and c.strip() not in ("", quote_ccy) for c in ccys):
                    return None
                hist = hist["Dividends"]
            if not isinstance(hist, pd.Series):
                return None
            out: list[tuple[datetime.date, float]] = []
            for idx, val in hist.items():
                amount = _pos(val)
                day = idx.date() if hasattr(idx, "date") else None
                if amount is not None and isinstance(day, datetime.date):
                    out.append((day, amount))
            return out
        except Exception:
            return None

    # ----------------------------------------------------------------- #
    #  Public: a single comps row (never raises).
    # ----------------------------------------------------------------- #
    def get_comp_row(self, ticker: str) -> Optional[CompRow]:
        """Build a :class:`CompRow` of trailing multiples from ``.info``.

        Field mapping:
          market_cap       <- marketCap (in the quote's major unit)
          enterprise_value <- enterpriseValue, rebuilt (see below)
          ev_ebitda        <- enterpriseToEbitda, rebuilt (see below)
          ev_sales         <- enterpriseToRevenue, rebuilt (see below)
          pe               <- trailingPE
          pb               <- priceToBook; where Yahoo's basis is mixed, market cap /
                              common equity from the balance sheet, else None
          peg              <- pegRatio (fallback trailingPegRatio)

        Yahoo's EV fields mix currencies and share bases for ADRs and foreign
        lines, so they are rebuilt (see :meth:`_comp_ev_fields`) whenever the
        reporting currency (``financialCurrency``) is not the quote currency,
        or Yahoo's EV is more than `_COMP_EV_TOLERANCE` away from
        marketCap + (totalDebt - totalCash) x FX and minority interest does
        not explain the gap. A rebuilt row carries a ``_source_notes`` list
        saying what changed. If that check itself fails, Yahoo's figures are
        kept for a single-currency row and left out (None) for a row whose
        reporting and quote currencies differ, with a note either way.

        Returns ``None`` (never raises) if the ticker can't be resolved or yields
        no usable data at all.
        """
        try:
            tk = self._ticker(ticker)
        except Exception:
            return None

        info = self._info(tk)
        if not info:
            return None

        name = (
            _str(info.get("longName"))
            or _str(info.get("shortName"))
            or ticker.upper()
        )

        # PEG can live under either key depending on yfinance version.
        peg = _num(info.get("pegRatio"))
        if peg is None:
            peg = _num(info.get("trailingPegRatio"))

        notes: list[str] = []
        try:
            market_cap, ev, ev_ebitda, ev_sales, pb = self._comp_ev_fields(
                tk, info, ticker.upper(), notes
            )
        except Exception as exc:  # noqa: BLE001 -- a comps row must never raise
            # Yahoo's own figures, except where its currencies are known to be
            # mixed (TSM EV/EBITDA 5.1, ASML about 2,758): those are left out.
            market_cap, ev, ev_ebitda, ev_sales, pb = (
                _pos(info.get("marketCap")), _num(info.get("enterpriseValue")),
                _num(info.get("enterpriseToEbitda")), _num(info.get("enterpriseToRevenue")),
                _num(info.get("priceToBook")),
            )
            quote, _ = major_currency(info.get("currency"))
            fin, _ = major_currency(info.get("financialCurrency"))
            failed = f"{ticker.upper()}: the EV and P/B check failed ({type(exc).__name__})"
            if quote is not None and fin is not None and quote != fin:
                ev = ev_ebitda = ev_sales = pb = None
                notes = [f"{failed}; Yahoo's enterprise value, EV/EBITDA, EV/Sales and P/B "
                         f"mix {fin} statements with the {quote} quote and are left out of "
                         "the comps"]
            else:
                notes = [f"{failed}; Yahoo's enterprise value, EV multiples and P/B are "
                         "kept as reported"]

        row = CompRow(
            ticker=ticker.upper(),
            name=name,
            market_cap=market_cap,
            enterprise_value=ev,
            ev_ebitda=ev_ebitda,
            ev_sales=ev_sales,
            pe=_num(info.get("trailingPE")),
            pb=pb,
            peg=peg,
            currency=major_currency(self._quote_currency(tk, info))[0],
        )
        if notes:
            row._source_notes = notes  # type: ignore[attr-defined]

        # If literally every valuation field is missing, the row is useless as a
        # comp; treat that as an unresolved ticker.
        has_any = any(
            v is not None
            for v in (
                row.ev_ebitda,
                row.ev_sales,
                row.pe,
                row.pb,
                row.peg,
                row.enterprise_value,
                row.market_cap,
            )
        )
        return row if has_any else None

    def _comp_ev_fields(
        self, tk, info: dict, symbol: str, notes: list[str]
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float],
               Optional[float]]:
        """``(market cap, EV, EV/EBITDA, EV/Sales, P/B)`` for a comps row, on one basis.

        * Market cap: Yahoo's ``marketCap`` in the quote's major unit (a
          minor-unit figure is detected against price x shares, as in
          :meth:`get_market_data`; without both it is kept as reported).
        * Yahoo's EV is market cap + total debt - total cash + minority
          interest + preferred stock (BN, KKR, BEP match it), but it takes the
          balance-sheet items in the reporting currency and, for some ADRs, on
          another share basis. It is checked against market cap + (totalDebt -
          totalCash) x FX, where FX takes the reporting currency to the quote
          currency (one spot rate from :meth:`get_fx_rate`; a missing debt or
          cash figure counts as 0 when the other is reported).
        * When the currencies differ, Yahoo reports no EV, or its EV is more
          than `_COMP_EV_TOLERANCE` away from that check, Yahoo's newest
          balance sheet (see :meth:`_comp_balance_items`) adds minority
          interest and preferred stock (x FX). A same-currency EV within the
          tolerance of the result is kept (the gap was minority interest).
          Otherwise EV is that rebuild and EV/Sales and EV/EBITDA are it over
          totalRevenue and ebitda x FX (None unless both positive).
        * If that balance sheet cannot be read, a same-currency Yahoo EV
          above the check by up to `_COMP_EV_MINORITY_MAX` times is kept with
          its multiples and P/B (minority interest could explain it; BN), with
          a note that it could not be checked.
        * The balance-sheet amounts are converted from the statement tables'
          own currency, which :meth:`_tables_currency` checks: VALE's and
          PBR's tables are in USD while financialCurrency and the ``.info``
          amounts are BRL, so their equity and minority interest take no FX
          (P/B 1.58, not 8.2).
        * P/B is then rebuilt as market cap / (common equity x FX) from the
          same balance sheet, or None without one: Yahoo divides the quoted
          price by a book value per share in the reporting currency, on the
          ordinary-share basis for an ADR (TSM 92.8, ASML 1,519, BP ADR 7.9).
        * Differing currencies with nothing to rebuild from (no FX rate, no
          market cap, or neither debt nor cash) leave the EV fields and P/B
          None: Yahoo's are known to be mixed.
        Each change is noted in `notes`; Yahoo's figures are kept otherwise.
        """
        quote, unit = major_currency(self._quote_currency(tk, info))
        fin, fin_unit = major_currency(info.get("financialCurrency"))

        mcap = _pos(info.get("marketCap"))
        if mcap is not None and unit != 1.0:
            price = (_pos(info.get("currentPrice")) or _pos(info.get("regularMarketPrice"))
                     or _pos(info.get("previousClose")))
            shares = _pos(info.get("sharesOutstanding"))
            if price is not None and shares is not None:
                mcap *= _unit_scale(mcap / (price * shares), unit)

        ev = _num(info.get("enterpriseValue"))
        ev_ebitda = _num(info.get("enterpriseToEbitda"))
        ev_sales = _num(info.get("enterpriseToRevenue"))
        pb = _num(info.get("priceToBook"))

        def off(check: Optional[float]) -> bool:
            """Yahoo's EV is further than the tolerance from `check` (a sign
            flip counts: BRK-B's Yahoo EV is -234B against about +842B)."""
            return (check is not None and ev is not None
                    and abs(ev - check) > _COMP_EV_TOLERANCE * abs(check))

        mixed = fin is not None and quote is not None and fin != quote
        fx = self.get_fx_rate(fin, quote) if mixed else (1.0, "same currency")
        debt, cash = _num(info.get("totalDebt")), _num(info.get("totalCash"))
        rate = fx[0] * fin_unit if fx is not None else None
        rebuilt = None
        if rate is not None and mcap is not None and (debt is not None or cash is not None):
            rebuilt = mcap + ((debt or 0.0) - (cash or 0.0)) * rate
        if not mixed and not off(rebuilt) and not (rebuilt is not None and ev is None):
            return mcap, ev, ev_ebitda, ev_sales, pb

        if rebuilt is None:  # differing currencies, nothing to rebuild from
            why = (f"no {fin}->{quote} exchange rate" if rate is None
                   else "no market cap, total debt or total cash from Yahoo")
            notes.append(
                f"{symbol}: Yahoo's enterprise value, EV/EBITDA, EV/Sales and P/B mix "
                f"{fin} statements with the {quote} quote and could not be rebuilt "
                f"({why}); left out of the comps"
            )
            return mcap, None, None, None, None

        bal = self._comp_balance_items(tk)
        if (not mixed and bal["as_of"] is None and ev is not None
                and 0 < rebuilt < ev <= _COMP_EV_MINORITY_MAX * rebuilt):
            # No balance sheet (an outage or rate limit): the gap may be the
            # minority interest and preferred stock Yahoo's EV includes (BN).
            notes.append(
                f"{symbol}: Yahoo's enterprise value {ev:,.0f} is {ev / rebuilt - 1:.0%} above "
                f"market cap + total debt - total cash ({rebuilt:,.0f}); kept with its EV "
                "multiples and P/B, since the minority interest and preferred stock that "
                "Yahoo's EV includes could not be checked without Yahoo's balance sheet"
            )
            return mcap, ev, ev_ebitda, ev_sales, pb

        # Minority interest, preferred stock and equity come from the statement
        # tables, which for a few 20-F filers are not in financialCurrency.
        table_rate, table_note = rate, None
        if bal["as_of"] is not None:
            tables, table_fx, evidence = self._tables_currency(
                tk, info, bal["debt"], known={quote: fx} if mixed else None
            )
            if table_fx is not None:
                table_rate = rate / table_fx[0]  # table currency -> quote currency
                table_note = (
                    f"{symbol}: Yahoo's statement tables are in {tables}, not the {fin} of its "
                    f"financialCurrency ({evidence}); minority interest, preferred stock and "
                    f"common equity are read in {tables}"
                )
        claims = (bal["minority"] or 0.0) + (bal["preferred"] or 0.0)
        rebuilt += claims * table_rate
        if not mixed and ev is not None and not off(rebuilt):
            return mcap, ev, ev_ebitda, ev_sales, pb  # the gap was minority interest

        revenue, ebitda = _num(info.get("totalRevenue")), _num(info.get("ebitda"))
        new_sales = (rebuilt / (revenue * rate)
                     if rebuilt > 0 and revenue is not None and revenue > 0 else None)
        new_ebitda = (rebuilt / (ebitda * rate)
                      if rebuilt > 0 and ebitda is not None and ebitda > 0 else None)

        def fmt(x: Optional[float]) -> str:
            return "n/a" if x is None else f"{x:,.3g}"

        if mixed:
            how = (f" x {rate:.6g} ({fx[1]})",
                   f"Yahoo mixes {fin} statements with the {quote} quote")
        elif ev is not None:
            how = ("", f"Yahoo's {ev:,.0f} is more than {_COMP_EV_TOLERANCE:.0%} away, "
                       "so it is on another share or currency basis")
        else:
            how = ("", "Yahoo reports no enterprise value")
        if table_rate == rate:
            parts = "total debt - total cash"
            if claims:
                parts += " + minority interest + preferred stock"
            formula = f"({parts}){how[0]}"
        else:  # the .info amounts and the statement tables convert differently
            formula = f"(total debt - total cash){how[0]}"
            if claims:
                formula += " + minority interest + preferred stock" + (
                    "" if table_rate == 1.0 else f" x {table_rate:.6g}")
        note = (
            f"{symbol}: EV rebuilt as market cap + {formula} = "
            f"{rebuilt:,.0f} {quote or ''}".rstrip() + f" ({how[1]}); EV/EBITDA "
            f"{fmt(ev_ebitda)} -> {fmt(new_ebitda)}, EV/Sales {fmt(ev_sales)} -> {fmt(new_sales)}"
        )
        if mixed or ev is not None:
            equity = bal["equity"]
            new_pb = (mcap / (equity * table_rate)
                      if equity is not None and equity > 0 else None)
            if new_pb is not None:
                note += (f"; P/B {fmt(pb)} -> {fmt(new_pb)} (market cap / common equity at "
                         f"{bal['as_of']})")
            elif pb is not None:
                note += (f"; P/B {fmt(pb)} left out (Yahoo's book value per share is not on "
                         "the quoted currency and share basis)")
            pb = new_pb
        notes.append(note)
        if table_note is not None:
            notes.append(table_note)
        return mcap, (rebuilt if rebuilt > 0 else None), new_ebitda, new_sales, pb

    def _comp_balance_items(self, tk) -> dict:
        """Minority interest, preferred stock, common equity and total debt
        from Yahoo's newest balance sheet (quarterly, else annual), in the
        statement tables' currency (see :meth:`_tables_currency`), with its
        date. Missing items (or no balance sheet) are None.
        """
        out = {"as_of": None, "minority": None, "preferred": None, "equity": None,
               "debt": None}
        for attr in ("quarterly_balance_sheet", "balance_sheet"):
            frame = self._frame(tk, attr)
            if frame is None:
                continue
            dated = [c for c in frame.columns if self._col_date(c) is not None]
            for col in sorted(dated, key=self._col_date, reverse=True):
                preferred = self._cell(frame, ("Preferred Stock", "PreferredStock"), col)
                equity = self._cell(frame, ("Common Stock Equity", "CommonStockEquity"), col)
                if equity is None:
                    parent = self._cell(frame, ("Stockholders Equity", "StockholdersEquity"), col)
                    equity = parent - (preferred or 0.0) if parent is not None else None
                minority = self._cell(frame, ("Minority Interest", "MinorityInterest"), col)
                if equity is None and minority is None:
                    continue  # a sparse column
                out.update(
                    as_of=self._col_date(col).isoformat(), equity=equity, minority=minority,
                    preferred=preferred,
                    debt=self._cell(frame, ("Total Debt", "TotalDebt"), col),
                )
                return out
        return out

    def _tables_currency(
        self, tk, info: dict, debt: Optional[float], income=None,
        known: Optional[dict] = None,
    ) -> tuple[Optional[str], Optional[tuple[float, str]], str]:
        """``(currency, fx, evidence)``: the currency of Yahoo's statement tables.

        The tables (balance sheet, income and cash-flow statements) are
        normally in the reporting currency, ``info['financialCurrency']``, as
        are the ``.info`` amounts. For some 20-F filers they hold the USD
        20-F figures while financialCurrency and ``.info`` stay local: VALE
        and VALE3.SA show a table Total Debt of 22.2B (USD) against a
        ``totalDebt`` of 110.0B (BRL); PBR and YPF likewise. The tables are
        taken to be in currency X only when the table Total Debt over
        ``totalDebt`` and the newest annual Total Revenue over
        ``totalRevenue`` BOTH lie within `_TABLE_CCY_BAND` of the
        financialCurrency->X spot rate and nearer it than 1, where X is the
        quote currency or USD and at least `_TABLE_CCY_MIN_GAP` away from
        financialCurrency. One comparison alone is not enough: bank debt
        definitions differ (ITUB's table debt is 0.50 of ``totalDebt``,
        HMC's 0.34), and TTM revenue differs from the last fiscal year's.

        `debt` is the table Total Debt to compare (newest column); `income`
        the annual income statement, fetched only when the debt comparison
        does not settle it; `known` maps codes to financialCurrency->code
        rates already fetched. Returns ``(financialCurrency or None, None,
        "")`` in the usual case or without the comparisons, else X, the
        financialCurrency->X rate with the quote(s) used, and the comparison
        as text for a note. Rates are fetched only when both ratios are far
        from 1.
        """
        fin, fin_unit = major_currency(info.get("financialCurrency"))
        usual = (fin, None, "")
        base_debt, base_revenue = _pos(info.get("totalDebt")), _pos(info.get("totalRevenue"))
        if (fin is None or fin_unit != 1.0 or base_debt is None or base_revenue is None
                or debt is None or not debt > 0):
            return usual
        # A ratio this near 1 is nearer 1 than any rate that qualifies.
        near_one = math.log(_TABLE_CCY_MIN_GAP) / 2
        debt_ratio = debt / base_debt
        if abs(math.log(debt_ratio)) < near_one:
            return usual
        if income is None:
            income = self._frame(tk, "financials")
        revenue = self._newest_cell(income, ("Total Revenue", "TotalRevenue"))
        if revenue is None or not revenue > 0:
            return usual
        revenue_ratio = revenue / base_revenue
        if abs(math.log(revenue_ratio)) < near_one:
            return usual
        quote, _ = major_currency(self._quote_currency(tk, info))
        for code in dict.fromkeys(c for c in (quote, "USD") if c and c != fin):
            fx = (known or {}).get(code) or self.get_fx_rate(fin, code)
            if fx is None or abs(math.log(fx[0])) < math.log(_TABLE_CCY_MIN_GAP):
                continue
            if all(abs(math.log(r / fx[0])) <= math.log(_TABLE_CCY_BAND)
                   and abs(math.log(r / fx[0])) < abs(math.log(r))
                   for r in (debt_ratio, revenue_ratio)):
                return code, fx, (
                    f"table Total Debt / totalDebt {debt_ratio:.4g} and Total Revenue / "
                    f"totalRevenue {revenue_ratio:.4g}, against {fin}->{code} {fx[0]:.4g} "
                    f"({fx[1]})"
                )
        return usual

    # ----------------------------------------------------------------- #
    #  Public: many comps rows, skipping failures.
    # ----------------------------------------------------------------- #
    def get_comp_rows(self, tickers: list[str]) -> list[CompRow]:
        """Map :meth:`get_comp_row` over ``tickers``, dropping unresolved ones.

        De-duplicates by upper-cased symbol so the same peer passed twice (or the
        target appearing in its own peer list) yields a single row.
        """
        rows: list[CompRow] = []
        seen: set[str] = set()
        for t in tickers or []:
            sym = _str(t)
            if not sym:
                continue
            key = sym.upper()
            if key in seen:
                continue
            seen.add(key)
            row = self.get_comp_row(sym)
            if row is not None:
                rows.append(row)
        return rows

    # ----------------------------------------------------------------- #
    #  Public: best-effort peer suggestions (yfinance has no robust API).
    # ----------------------------------------------------------------- #
    def suggest_peers(self, ticker: str) -> list[str]:
        """Best-effort peer tickers for ``ticker``: always ``[]``.

        yfinance exposes no peer/screener data: none of the modules its `.info`
        requests (financialData, quoteType, defaultKeyStatistics, assetProfile,
        summaryDetail, the v7 quote) carries a related-tickers field. We never
        fabricate peers from a sector string either, so peers must be passed
        explicitly (README: "Peers are not auto-discovered"). Returning without
        a network call matters because the comps model asks on every
        (cached) recompute. Never raises.
        """
        return []

    # ----------------------------------------------------------------- #
    #  Public: fundamentals fallback from yfinance statement DataFrames.
    # ----------------------------------------------------------------- #
    def get_annual_financials_fallback(
        self, ticker: str
    ) -> Optional[tuple[AnnualFinancials, BalanceSheetSnapshot]]:
        """Reconstruct (AnnualFinancials, BalanceSheetSnapshot) from yfinance.

        Parses ``Ticker(...).financials`` (income statement), ``.cashflow`` and
        ``.balance_sheet`` — pandas DataFrames whose COLUMNS are period-end
        Timestamps ordered NEWEST-first. We reverse them to OLDEST->NEWEST to match
        the package's series convention, and keep only the periods where the
        income statement has both revenue and net income (yfinance often adds a
        sparse oldest column), mirroring the EDGAR year axis. Each line takes,
        per period, the first candidate row with a value; lines still missing
        are zero-filled and noted.

        Statements are in the issuer's reporting currency
        (``info['financialCurrency']``), unless :meth:`_tables_currency`
        shows them to be in another (VALE, PBR: USD tables under a BRL
        financialCurrency; noted); when that differs from the quote
        currency they are converted at one spot FX rate (see
        :meth:`get_fx_rate`), or raise DataError if no rate is available.
        Notes ride on the returned financials as
        ``_source_notes``.

        Used by the hybrid provider to backfill non-US issuers that EDGAR cannot
        serve. Returns ``None`` for missing dependencies, empty statements or no
        revenue; raises ``DataError`` for a missing required FX conversion.
        """
        try:
            import pandas as pd  # noqa: F401  (lazy; only needed on this path)

            tk = self._ticker(ticker)
            income = self._frame(tk, "financials")
            cashflow = self._frame(tk, "cashflow")
            balance = self._frame(tk, "balance_sheet")
        except Exception:
            return None

        # Without an income statement there is nothing to anchor the series on.
        if income is None or income.empty:
            return None

        notes: list[str] = []
        rev_names = ("Total Revenue", "TotalRevenue", "Operating Revenue", "OperatingRevenue")
        ni_names = ("Net Income Common Stockholders", "NetIncomeCommonStockholders",
                    "Net Income", "NetIncome")

        # Period-end columns, oldest -> newest. yfinance gives newest-first.
        cols = list(income.columns)
        try:
            cols = sorted(cols)  # Timestamps sort chronologically -> oldest first
        except Exception:
            cols = list(reversed(cols))  # fall back to a simple reversal

        # Keep periods with revenue, and with net income too where that line
        # exists at all (a sparse oldest column otherwise enters the history
        # as a year of real revenue with zero EBIT, capex, shares, ...).
        with_rev = [c for c in cols if self._cell(income, rev_names, c) is not None]
        both = [c for c in with_rev if self._cell(income, ni_names, c) is not None]
        kept = both or with_rev
        if not kept:
            return None  # no revenue anywhere: nothing to value
        dropped = [c for c in cols if c not in kept]
        if dropped:
            notes.append(
                "yfinance fallback: dropped "
                + ", ".join(self._date_iso(c) for c in dropped)
                + " (period without both revenue and net income)"
            )
        cols = kept

        # Derive fiscal years from the column timestamps (period-end year).
        fiscal_years: list[int] = []
        for c in cols:
            yr = self._year_of(c)
            fiscal_years.append(yr if yr is not None else 0)

        # --- helper to pull a row series aligned to `cols` (oldest->newest) --- #
        def series(
            sources: list[tuple[object, tuple[str, ...]]],
            *,
            positive: bool = False,
            label: str = "",
        ) -> list[float]:
            """Per period, the first candidate row (across `sources`) with a value.

            Missing values -> 0.0 (noted when `label` is given); if `positive`,
            store the absolute magnitude (yfinance reports capex / dividends /
            D&A with varying signs).
            """
            out: list[float] = []
            missing: list[int] = []
            for c, fy in zip(cols, fiscal_years):
                val = None
                for df, names in sources:
                    val = self._cell(df, names, c)
                    if val is not None:
                        break
                if val is None:
                    out.append(0.0)
                    missing.append(fy)
                else:
                    out.append(abs(val) if positive else val)
            if label and missing:
                if len(missing) == len(cols):
                    notes.append(f"yfinance fallback: {label} unavailable; filled with 0.0")
                else:
                    notes.append(
                        f"yfinance fallback: {label} missing for "
                        + ", ".join(f"FY{y}" for y in missing)
                        + "; filled with 0.0"
                    )
            return out

        # --- income-statement driven series --------------------------------- #
        revenue = series([(income, rev_names)])
        # Operating income first: Yahoo's 'EBIT' row is pretax income + interest
        # expense (non-operating items included), so it is only a last resort
        # (e.g. banks with no operating-income line), matching EDGAR's basis.
        ebit = series(
            [(income, ("Operating Income", "OperatingIncome",
                       "Total Operating Income As Reported", "EBIT"))],
            label="EBIT (operating income)",
        )
        net_income = series([(income, ni_names)])
        attribution_adjusted = False
        common_derived: list[int] = []
        for i, (col, year) in enumerate(zip(cols, fiscal_years)):
            common = self._cell(income, ni_names[:2], col)
            parent = self._cell(income, ni_names[2:], col)
            gross = self._cell(income, ("Net Income Including Noncontrolling Interests",), col)
            if common is not None:
                references = [v for v in (parent, gross) if v is not None]
                if not references or any(common != v for v in references):
                    attribution_adjusted = True
            elif parent is not None:
                preferred = self._cell(income, ("Preferred Stock Dividends",), col) or 0.0
                other = self._cell(income, ("Otherunder Preferred Stock Dividend",), col) or 0.0
                if preferred or other:
                    net_income[i] = parent - preferred - other
                    common_derived.append(year)
                    attribution_adjusted = True
                if gross is not None and parent != gross:
                    attribution_adjusted = True
        if common_derived:
            notes.append(
                "yfinance fallback: common net income derived from parent net income less "
                "preferred dividends and other preferred adjustments for "
                + ", ".join(f"FY{y}" for y in common_derived)
            )
        pretax_income = series(
            [(income, ("Pretax Income", "PretaxIncome", "Income Before Tax"))],
            label="pretax income",
        )
        # Signed: a tax benefit stays negative (the effective-tax median needs it).
        tax_expense = series(
            [(income, ("Tax Provision", "TaxProvision", "Income Tax Expense"))],
            label="tax expense",
        )
        interest_expense = series(
            [(income, ("Interest Expense", "InterestExpense",
                       "Interest Expense Non Operating"))],
            positive=True,
        )

        # D&A: prefer the income statement, then the cash-flow statement.
        dep_amort = series(
            [
                (income, (
                    "Reconciled Depreciation",
                    "Depreciation And Amortization In Income Statement",
                    "Depreciation Amortization Depletion Income Statement",
                )),
                (cashflow, (
                    "Depreciation And Amortization",
                    "DepreciationAndAmortization",
                    "Depreciation Amortization Depletion",
                    "Depreciation",
                )),
            ],
            positive=True,
            label="D&A",
        )

        # EBITDA = EBIT + D&A (per the package convention; never looked up).
        ebitda = [e + d for e, d in zip(ebit, dep_amort)]

        # --- cash-flow driven series ---------------------------------------- #
        capex = series(
            [(cashflow, _CAPEX_ROWS)],
            positive=True,
            label="capex",
        )
        dividends_paid = series(
            [(cashflow, (
                "Cash Dividends Paid",
                "Common Stock Dividend Paid",
                "CommonStockDividendPaid",
                "Dividends Paid",
            ))],
            positive=True,
        )
        # ΔNWC: yfinance's "Change In Working Capital" is signed as a cash-flow
        # contribution (a NWC *increase* is a cash *use* -> negative). Our schema
        # stores ΔNWC as positive = increase in NWC, so negate the cash-flow sign.
        cf_wc = series([(cashflow, ("Change In Working Capital", "ChangeInWorkingCapital"))])
        change_in_nwc = [-v for v in cf_wc]

        # Diluted shares (weighted average), falling back to basic per period.
        diluted_shares = series(
            [(income, (
                "Diluted Average Shares",
                "DilutedAverageShares",
                "Basic Average Shares",
                "BasicAverageShares",
            ))],
            label="diluted shares",
        )

        financials = AnnualFinancials(
            fiscal_years=fiscal_years,
            revenue=revenue,
            ebit=ebit,
            ebitda=ebitda,
            net_income=net_income,
            dep_amort=dep_amort,
            capex=capex,
            change_in_nwc=change_in_nwc,
            interest_expense=interest_expense,
            tax_expense=tax_expense,
            pretax_income=pretax_income,
            dividends_paid=dividends_paid,
            diluted_shares=diluted_shares,
        )

        balance_sheet = self._build_balance_sheet(balance, notes)

        # Reporting currency -> quote currency (FX notes lead the list).
        fx_notes: list[str] = []
        financials, balance_sheet = self._to_quote_currency(
            tk, financials, balance_sheet, fx_notes, balance=balance, income=income
        )

        # A consolidated captive finance arm (Toyota, Honda) is marked like the
        # EDGAR client marks one; its WARNING leads the notes.
        lead: list[str] = []
        kind = None
        info = self._info(tk)
        col = self._newest_column(balance)
        if col is not None:
            evidence, note = self._captive_finance_evidence(
                balance, col, info.get("sector"), info.get("industry")
            )
            if evidence is not None:
                kind = "captive_finance"
                lead.append(captive_finance_warning("yfinance statements", evidence))
            elif note is not None:
                notes.append(note)
        financials._source_notes = lead + fx_notes + notes  # type: ignore[attr-defined]
        financials._financial_kind = kind  # type: ignore[attr-defined]
        financials._income_attribution_adjusted = attribution_adjusted  # type: ignore[attr-defined]
        return financials, balance_sheet

    # ----------------------------------------------------------------- #
    #  Captive finance arms and the EDGAR debt backfill (Yahoo balance sheet).
    # ----------------------------------------------------------------- #
    def _captive_finance_evidence(
        self, balance, col, sector: object, industry: object
    ) -> tuple[Optional[str], Optional[str]]:
        """``(evidence, note)`` for a consolidated captive finance arm.

        `evidence` (for the WARNING) is set when the Yahoo balance sheet at
        `col` shows one. Otherwise `note` is set for a company in
        `_CAPTIVE_INDUSTRIES` (Yahoo hides some arms in unlabelled lines, so
        the note says one cannot be ruled out); both are None for the rest. A
        captive arm shows as either
          * non-current receivables (``Non Current Accounts Receivable`` plus
            ``Non Current Note Receivables``: loans and leases due after a year)
            of at least `_CAPTIVE_MIN_NONCURRENT_SHARE` of total assets, funded
            at least `_CAPTIVE_MIN_DEBT_FUNDING` by total debt, in a
            `_CAPTIVE_SECTORS` sector (TM 24%, HMC 20%, F 21%, CAT 17%); or
          * for `_CAPTIVE_INDUSTRIES`, current plus non-current receivables of
            at least `_CAPTIVE_MIN_RECEIVABLES_SHARE` and total debt of at least
            `_CAPTIVE_MIN_DEBT_SHARE` of total assets (Renault, Nissan, Deere,
            PACCAR, where Yahoo reports the finance book as current).
        Lessors and rental companies (URI, AerCap) hold their fleets as
        property, not receivables, and are not marked.
        """
        assets = self._cell(balance, ("Total Assets", "TotalAssets"), col)
        if assets is None or assets <= 0:
            return None, None
        debt = self._cell(balance, ("Total Debt", "TotalDebt"), col) or 0.0
        parts = [self._cell(balance, (n,), col)
                 for n in ("Non Current Accounts Receivable", "Non Current Note Receivables")]
        noncurrent = sum(p for p in parts if p is not None and p > 0)
        sec = (_str(sector) or "").lower()
        ind = (_str(industry) or "").lower()
        if (sec in _CAPTIVE_SECTORS and noncurrent >= _CAPTIVE_MIN_NONCURRENT_SHARE * assets
                and debt >= _CAPTIVE_MIN_DEBT_FUNDING * noncurrent):
            return (f"non-current receivables are {noncurrent / assets:.0%} of total assets "
                    f"and total debt {debt / assets:.0%}"), None
        if ind not in _CAPTIVE_INDUSTRIES:
            return None, None
        current = self._cell(balance, ("Receivables",), col)
        if current is None:
            subs = [self._cell(balance, (n,), col) for n in (
                "Accounts Receivable", "Other Receivables", "Notes Receivable", "Loans Receivable")]
            current = sum(s for s in subs if s is not None and s > 0)
        receivables = max(current, 0.0) + noncurrent
        if (receivables >= _CAPTIVE_MIN_RECEIVABLES_SHARE * assets
                and debt >= _CAPTIVE_MIN_DEBT_SHARE * assets):
            return (f"receivables are {receivables / assets:.0%} of total assets and total "
                    f"debt {debt / assets:.0%}, for a maker in {_str(industry)}"), None
        return None, (
            f"{_str(industry)} companies often consolidate a captive finance arm, which "
            "Yahoo's statements do not always show; none is visible here (receivables "
            f"{receivables / assets:.0%} and total debt {debt / assets:.0%} of total assets), "
            "but if there is one, its debt and receivables are in the DCF, FCFE and WACC inputs"
        )

    def get_balance_sheet_items(self, ticker: str, as_of: str) -> Optional[dict]:
        """Debt, cash and captive-finance evidence from Yahoo's balance sheet.

        Reads the quarterly and annual balance sheets and takes the column
        (with total assets) closest to `as_of`, if it is within
        `BACKFILL_MAX_DAYS`. Amounts are in the statement tables' currency:
        Yahoo's reporting currency (``financialCurrency``) unless
        :meth:`_tables_currency` shows another. Used by HybridProvider to
        backfill an EDGAR filer whose debt sits under company-specific tags.
        Returns None (never raises) when nothing usable is found, else a dict:
          as_of               ISO date of the column used
          currency            the tables' currency (major ISO code) or None
          currency_note       why that is not financialCurrency, or None
          total_debt          long-term + current debt, else Total Debt less
                              lease obligations, else Total Debt. A side
                              without its plain line (``Long Term Debt``,
                              ``Current Debt``) takes the line that includes
                              leases, less its lease line when there is one
          debt_basis          which of those, in words
          cash                cash and cash equivalents alone
          cash_and_investments  plus short-term investments
          captive             evidence of a captive finance arm, or None
        """
        try:
            target = datetime.date.fromisoformat(str(as_of or "")[:10])
            tk = self._ticker(ticker)
        except Exception:
            return None
        best = None
        newest_debt = None  # the tables' newest Total Debt, for their currency
        for attr in ("quarterly_balance_sheet", "balance_sheet"):
            frame = self._frame(tk, attr)
            if frame is None:
                continue
            if newest_debt is None:
                newest_debt = self._newest_cell(frame, ("Total Debt", "TotalDebt"))
            for col in frame.columns:
                day = self._col_date(col)
                if day is None or self._cell(frame, ("Total Assets", "TotalAssets"), col) is None:
                    continue
                gap = abs((day - target).days)
                if best is None or gap < best[0]:
                    best = (gap, frame, col, day)
        if best is None or best[0] > BACKFILL_MAX_DAYS:
            return None
        _gap, frame, col, day = best

        def cell(*names: str) -> Optional[float]:
            return self._cell(frame, names, col)

        def debt_part(plain: str, with_leases: str, lease: str) -> tuple[Optional[float], bool]:
            """``(amount, includes leases)`` for the long-term or current side
            (labels match without regard to case or spaces, see `_row`)."""
            value = cell(plain)
            if value is not None:
                return value, False
            value = cell(with_leases)
            if value is None:
                return None, False
            lease_part = cell(lease)
            return (value - lease_part, False) if lease_part is not None else (value, True)

        long_term, lt_leases = debt_part("Long Term Debt",
                                         "Long Term Debt And Capital Lease Obligation",
                                         "Long Term Capital Lease Obligation")
        current, cur_leases = debt_part("Current Debt",
                                        "Current Debt And Capital Lease Obligation",
                                        "Current Capital Lease Obligation")
        # Only when a plain line shows the split (else Total Debt less leases).
        has_plain = cell("Long Term Debt") is not None or cell("Current Debt") is not None
        total, leases = cell("Total Debt", "TotalDebt"), cell("Capital Lease Obligations")
        if has_plain:
            debt = (long_term or 0.0) + (current or 0.0)
            with_leases = [side for side, flag in (("long-term", lt_leases),
                                                   ("current", cur_leases)) if flag]
            basis = ("long-term plus current debt, lease obligations excluded" if not with_leases
                     else "long-term plus current debt, "
                          f"{' and '.join(with_leases)} lease obligations included")
        elif total is not None and leases:
            debt, basis = total - leases, "total debt less lease obligations"
        elif total is not None:
            debt, basis = total, "total debt, which may include lease obligations"
        else:
            debt, basis = None, ""
        cash = cell("Cash And Cash Equivalents", "CashAndCashEquivalents")
        combined = cell("Cash Cash Equivalents And Short Term Investments",
                        "CashCashEquivalentsAndShortTermInvestments")
        if combined is None and cash is not None:
            short_term = cell("Other Short Term Investments", "Short Term Investments")
            combined = cash + (short_term or 0.0)
        info = self._info(tk)
        evidence, _note = self._captive_finance_evidence(
            frame, col, info.get("sector"), info.get("industry"))
        tables, currency_note = self._tables_currency_and_note(tk, info, newest_debt)
        return {
            "as_of": day.isoformat(),
            "currency": tables,
            "currency_note": currency_note,
            "total_debt": debt,
            "debt_basis": basis,
            "cash": cash,
            "cash_and_investments": combined,
            "captive": evidence,
        }

    def get_capex_by_year(self, ticker: str) -> Optional[dict]:
        """Capital expenditure by fiscal year from Yahoo's annual cash-flow statement.

        Used by HybridProvider to backfill EDGAR years that no capex tag
        covers (Phillips 66 tags none; AerCap's aircraft purchases sit under
        company-specific tags). Each column is labelled with the fiscal year
        of its period end by the EDGAR parser's rule (see `_year_of`: the
        calendar year of the end, a 52/53-week year ending in early January
        taking the prior year), so a year matches the EDGAR year that ends
        on the same date. Per column the first of `_CAPEX_ROWS` with a value
        is read, as a positive magnitude; zero or missing cells are left out,
        and so is a fiscal year that two columns share (a changed year end).
        Every other year read is returned, not only the ones asked about, so
        the caller can compare Yahoo's figures with EDGAR's where both report.
        Amounts are in the statement tables' currency, as in
        :meth:`get_balance_sheet_items`. Never raises. Returns None when the
        statement could not be read (the request failed or came back empty,
        as yfinance does when rate-limited), else a dict:
          currency       the tables' currency (major ISO code) or None
          currency_note  why that is not financialCurrency, or None
          by_year        {fiscal year: (ISO period end, capex)}; empty (with
                         currency and currency_note None) when the statement
                         was read but shows no capital expenditure
        """
        try:
            tk = self._ticker(ticker)
            cashflow = self._frame(tk, "cashflow")
            if cashflow is None:
                return None
            cols = list(cashflow.columns)
        except Exception:
            return None
        by_year: dict = {}
        seen: set = set()
        shared: set = set()
        for col in cols:
            day, fy = self._col_date(col), self._year_of(col)
            if day is None or fy is None:
                continue
            if fy in seen:
                shared.add(fy)
                continue
            seen.add(fy)
            value = self._cell(cashflow, _CAPEX_ROWS, col)
            if value is not None and value != 0:
                by_year[fy] = (day.isoformat(), abs(value))
        for fy in shared:
            by_year.pop(fy, None)
        if not by_year:
            return {"currency": None, "currency_note": None, "by_year": {}}
        info = self._info(tk)
        try:
            balance = self._frame(tk, "balance_sheet")
            tables, currency_note = self._tables_currency_and_note(
                tk, info, self._newest_cell(balance, ("Total Debt", "TotalDebt")))
        except Exception:  # the usual case: tables in the reporting currency
            tables, currency_note = major_currency(info.get("financialCurrency"))[0], None
        return {"currency": tables, "currency_note": currency_note, "by_year": by_year}

    def _tables_currency_and_note(
        self, tk, info: dict, newest_debt: Optional[float]
    ) -> tuple[Optional[str], Optional[str]]:
        """The statement tables' currency (see :meth:`_tables_currency`) and,
        when that is not financialCurrency, a note saying why."""
        tables, table_fx, table_evidence = self._tables_currency(tk, info, newest_debt)
        if table_fx is None:
            return tables, None
        return tables, (
            f"Yahoo's statement tables are in {tables}, not the "
            f"{major_currency(info.get('financialCurrency'))[0]} of its financialCurrency "
            f"({table_evidence})"
        )

    def _to_quote_currency(
        self,
        tk,
        financials: AnnualFinancials,
        balance_sheet: BalanceSheetSnapshot,
        notes: list[str],
        balance=None,
        income=None,
    ) -> tuple[AnnualFinancials, BalanceSheetSnapshot]:
        """Convert fallback statements into the quote currency's major unit.

        One spot rate for every year: the DCF, FCFE and comps are linear in the
        monetary inputs, so this equals valuing in the reporting currency and
        converting at spot, and the WACC weights need debt and market cap in the
        same currency. If no rate can be fetched, DataError prevents statements
        and the quote from entering valuation models with different units.
        `balance` and `income` are the statement tables the fallback read;
        they show when the tables are not in financialCurrency (see
        :meth:`_tables_currency`), in which case the tables' own currency is
        converted, or nothing when it is the quote currency (VALE's USD
        tables for the USD ADR), with a note.
        """
        info = self._info(tk)
        fin_ccy, fin_unit = major_currency(info.get("financialCurrency"))
        quote_ccy, _ = major_currency(self._quote_currency(tk, info))
        if fin_ccy is None or quote_ccy is None:
            notes.append(
                "yfinance fallback: reporting or quote currency unavailable from "
                "Yahoo; statements assumed to be in the quote currency"
            )
            return financials, balance_sheet
        tables, table_fx, evidence = self._tables_currency(
            tk, info, self._newest_cell(balance, ("Total Debt", "TotalDebt")), income=income
        )
        if table_fx is not None:
            notes.append(
                f"yfinance fallback: Yahoo gives {fin_ccy} as the reporting currency "
                f"(financialCurrency), but its statement tables are in {tables} "
                f"({evidence}); they are read as {tables}"
            )
            fin_ccy, fin_unit = tables, 1.0
        if fin_ccy == quote_ccy and fin_unit == 1.0:
            return financials, balance_sheet
        fx = self.get_fx_rate(fin_ccy, quote_ccy)
        if fx is None:
            raise DataError(
                f"Financial statements are in {fin_ccy} but the share "
                f"price is in {quote_ccy}, and no {fin_ccy}->{quote_ccy} exchange "
                "rate could be fetched. Valuation stopped to avoid mixing currencies; "
                "retry when exchange-rate data is available."
            )
        rate, how = fx
        notes.append(
            f"Fundamentals converted from {fin_ccy} to {quote_ccy} at spot "
            f"{rate:.6g} ({how}); every year uses this one rate"
        )
        return scale_fundamentals(financials, balance_sheet, rate * fin_unit)

    # ----------------------------------------------------------------- #
    #  Balance-sheet snapshot assembly (most recent period).
    # ----------------------------------------------------------------- #
    def _build_balance_sheet(
        self, balance, notes: Optional[list[str]] = None
    ) -> BalanceSheetSnapshot:
        """Build a :class:`BalanceSheetSnapshot` from the newest balance-sheet column.

        Always returns a snapshot (zero-filled if data is missing, with a note
        in `notes`) so the caller gets a usable object.
        """
        notes = notes if notes is not None else []
        if balance is None or getattr(balance, "empty", True):
            notes.append("yfinance fallback: balance sheet unavailable; debt, cash and equity set to 0.0")
            return BalanceSheetSnapshot(
                as_of="",
                total_debt=0.0,
                cash_and_investments=0.0,
                total_equity=0.0,
            )

        # Newest period-end column.
        try:
            col = max(balance.columns)
        except Exception:
            col = balance.columns[0]
        as_of = self._date_iso(col)

        def val(names: tuple[str, ...]) -> Optional[float]:
            return self._cell(balance, names, col)

        # Total debt: prefer an explicit total, else sum LT + current debt.
        total_debt = val(("Total Debt", "TotalDebt"))
        if total_debt is None:
            lt = val(("Long Term Debt And Capital Lease Obligation", "Long Term Debt", "LongTermDebt"))
            cur = val((
                "Current Debt And Capital Lease Obligation", "Current Debt",
                "CurrentDebt", "Short Term Debt", "ShortTermDebt",
            ))
            if lt is None and cur is None:
                notes.append("yfinance fallback: total debt unavailable; set to 0.0")
            total_debt = (lt or 0.0) + (cur or 0.0)

        # Cash + short-term investments. The combined line already includes the
        # short-term investments, so use it alone; otherwise add the parts.
        combined = val(("Cash Cash Equivalents And Short Term Investments",
                        "CashCashEquivalentsAndShortTermInvestments"))
        if combined is not None:
            cash_and_investments = combined
        else:
            cash = val(("Cash And Cash Equivalents", "CashAndCashEquivalents"))
            sti = val(("Other Short Term Investments", "Short Term Investments"))
            if cash is None and sti is None:
                notes.append("yfinance fallback: cash & equivalents unavailable; set to 0.0")
            cash_and_investments = (cash or 0.0) + (sti or 0.0)

        minority = val(("Minority Interest", "MinorityInterest")) or 0.0
        preferred_book = val(("Preferred Stock", "PreferredStock")) or 0.0
        preferred = val(("Preferred Stock", "PreferredStock", "Preferred Securities Outside Stock Equity")) or 0.0

        # Book equity for common shareholders, matching the P/B and ROE models.
        # A common-equity line already excludes preferred stock. Parent equity
        # does not, and consolidated equity also includes minority interests.
        total_equity = val(("Common Stock Equity", "CommonStockEquity"))
        if total_equity is None:
            parent = val(("Stockholders Equity", "StockholdersEquity"))
            gross = val(("Total Equity Gross Minority Interest",))
            if parent is not None:
                total_equity = parent - preferred_book
            elif gross is not None:
                total_equity = gross - minority - preferred_book
            else:
                notes.append("yfinance fallback: total equity unavailable; set to 0.0")

        return BalanceSheetSnapshot(
            as_of=as_of,
            total_debt=float(total_debt),
            cash_and_investments=float(cash_and_investments),
            total_equity=float(total_equity or 0.0),
            minority_interest=float(minority),
            preferred_equity=float(preferred),
        )

    # ----------------------------------------------------------------- #
    #  DataFrame access helpers.
    # ----------------------------------------------------------------- #
    def _frame(self, tk, attr: str):
        """Return a statement DataFrame for `attr`, or None on any failure."""
        try:
            df = getattr(tk, attr)
        except Exception:
            return None
        # Guard: yfinance can return None or a non-DataFrame on errors.
        if df is None:
            return None
        try:
            if df.empty:
                return None
        except Exception:
            return None
        return df

    def _row(self, df, names: tuple[str, ...]):
        """Return the first DataFrame row (a Series) whose index label matches.

        Matching is case/whitespace-insensitive against the requested `names`.
        Returns None if no candidate label is present.
        """
        if df is None:
            return None
        try:
            index_labels = list(df.index)
        except Exception:
            return None
        # Build a normalized lookup once.
        norm = {self._norm(lbl): lbl for lbl in index_labels}
        for name in names:
            key = self._norm(name)
            if key in norm:
                try:
                    return df.loc[norm[key]]
                except Exception:
                    return None
        return None

    def _cell(self, df, names: tuple[str, ...], col) -> Optional[float]:
        """Value at period `col` from the first candidate row that has one.

        Unlike taking the first row that merely exists, this coalesces per
        period, so a preferred label that is present but NaN for a period falls
        through to the next candidate instead of becoming 0.0.
        """
        for name in names:
            row = self._row(df, (name,))
            if row is None or not hasattr(row, "get"):
                continue
            try:
                v = _num(row.get(col))
            except Exception:
                v = None
            if v is not None:
                return v
        return None

    @staticmethod
    def _norm(label: object) -> str:
        """Normalize a row label for tolerant matching."""
        return "".join(str(label).lower().split())

    def _newest_cell(self, frame, names: tuple[str, ...]) -> Optional[float]:
        """The first of `names` at the newest dated column that has a value
        (sparse newer columns are skipped), or None."""
        if frame is None or getattr(frame, "empty", True):
            return None
        try:
            dated = [c for c in frame.columns if self._col_date(c) is not None]
        except Exception:
            return None
        for col in sorted(dated, key=self._col_date, reverse=True):
            value = self._cell(frame, names, col)
            if value is not None:
                return value
        return None

    @staticmethod
    def _newest_column(frame):
        """The newest period-end column of a statement frame, or None."""
        if frame is None or getattr(frame, "empty", True):
            return None
        try:
            return max(frame.columns)
        except Exception:
            return frame.columns[0]

    @staticmethod
    def _col_date(col: object) -> Optional[datetime.date]:
        """A period-end column label as a date, or None."""
        try:
            day = col.date()  # type: ignore[attr-defined]
        except Exception:
            try:
                day = datetime.date.fromisoformat(str(col)[:10])
            except ValueError:
                return None
        return day if isinstance(day, datetime.date) else None

    @staticmethod
    def _year_of(col: object) -> Optional[int]:
        """Fiscal-year label from a period-end column label.

        The calendar year of the period end, except that a 52/53-week year
        ending in the first two weeks of January takes the prior year (the same
        rule as the EDGAR parser), so it does not share a label with the next
        fiscal year ending in late December.
        """
        # pandas Timestamp / datetime expose `.year`.
        yr = getattr(col, "year", None)
        if isinstance(yr, int):
            month, day = getattr(col, "month", 0), getattr(col, "day", 0)
            return yr - 1 if (month == 1 and isinstance(day, int) and day <= 14) else yr
        # Fallback: parse a leading 4-digit year from the string form.
        s = str(col)
        if len(s) >= 10 and s[:4].isdigit() and s[5:7] == "01" and s[8:10].isdigit():
            return int(s[:4]) - 1 if int(s[8:10]) <= 14 else int(s[:4])
        if len(s) >= 4 and s[:4].isdigit():
            return int(s[:4])
        return None

    @staticmethod
    def _date_iso(col: object) -> str:
        """Render a period-end column label as an ISO date string (best effort)."""
        try:
            # pandas Timestamp / datetime -> 'YYYY-MM-DD'
            return col.date().isoformat()  # type: ignore[attr-defined]
        except Exception:
            pass
        s = str(col)
        return s[:10] if len(s) >= 10 else s
