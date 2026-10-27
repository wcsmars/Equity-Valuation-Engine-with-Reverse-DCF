"""Hybrid data provider: EDGAR fundamentals + yfinance market data.

This is the concrete ``DataProvider`` the engine instantiates by default. It
combines two specialized clients:

  * ``EdgarClient`` (``.edgar``) -- authoritative US-GAAP fundamentals straight
    from the SEC's XBRL JSON API. Works for US filers (10-K / 20-F).
  * ``YFinanceClient`` (``.market``) -- live market data (price, shares, beta,
    sector) plus trading-comp multiples, and a best-effort fundamentals fallback
    built from yfinance financial statements for tickers EDGAR can't serve
    (e.g. non-US companies).

Data selection:
  1. Always pull market data from yfinance first. A class-share ticker typed
     with a dot (BRK.B) that Yahoo cannot quote is valued under the dashed
     form the SEC directory lists (BRK-B), with a note.
  2. Try EDGAR for fundamentals (AnnualFinancials + BalanceSheetSnapshot + CIK).
  3. If EDGAR fails for any reason (ticker not in the SEC map, non-US filer,
     network/parse error), fall back to the yfinance fundamentals builder with
     ``cik = None``. EDGAR data whose latest fiscal year ended more than two
     years ago (a filer that stopped tagging us-gaap facts; Toyota's run
     FY2010-2013 only) is replaced by the fallback, with a WARNING note, when
     that has a newer fiscal year, and kept with a WARNING note otherwise.
  4. When the EDGAR client warns that interest expense implies debt its tags
     do not cover (Ford, Berkshire: company-specific debt elements) and it
     found no debt, backfill total debt (and short-term investments EDGAR
     clearly missed, or all of the cash when EDGAR's is 0 too, as for
     PACCAR) from Yahoo's balance sheet nearest the EDGAR date, and mark a
     captive finance arm that balance sheet shows (see
     `HybridProvider._backfill_edgar_debt`). Debt EDGAR found only in part
     is not backfilled (a known limitation; its WARNING stays). Likewise,
     D&A years EDGAR zero-filled because no D&A tag covers them (PG&E,
     Carvana) come from Yahoo's annual income or cash-flow statement for
     the same fiscal year, with EBITDA recomputed where EBIT is reported (see
     `HybridProvider._backfill_edgar_da`), and so do capex years no capex
     tag covers (Phillips 66, NextEra, AerCap), from its cash-flow statement
     (see `HybridProvider._backfill_edgar_capex`). A reported EDGAR figure
     is never replaced, and nothing is filled for a bank, insurer, BDC or
     REIT (nor D&A for a debt-funded lessor), or when Yahoo's figures do
     not match EDGAR's where both report.
  5. Put fundamentals and price in one currency: EDGAR statements are USD, the
     yfinance fallback converts to the quote currency itself; a missing required
     FX rate stops valuation with DataError rather than mixing currencies.
  6. Backfill market fields yfinance could not supply (shares outstanding,
     market cap, dividend per share) from the statements, with a note each,
     and replace a D0 of 0 read from a payment history the statements show
     to be incomplete. Warn when Yahoo records a corporate action after the
     latest fiscal year that is not a plain split (a spin-off; see
     `_corporate_action_warning`).
  7. Assemble and return ``CompanyData``. ``source_notes`` records which
     fundamentals source was used plus every data-quality note from the
     clients (EDGAR/yfinance gaps, derivations, FX conversion, backfills), so
     the engine can surface them as report warnings.
  8. Raise ``DataError`` only if *neither* source yields usable financials.

Peers: none are suggested by default (``suggest_peers`` returns ``[]``);
``discover_peers`` hands opt-in discovery from Yahoo's industry list (the
CLI's ``--auto-peers``) to the yfinance client.

The clients are constructed once in ``__init__`` (injectable for testing). We
never crash on a missing field -- every failure path degrades into either the
fallback source or a clear ``DataError``.
"""

from __future__ import annotations

import dataclasses
import datetime
import math
import re
import statistics
from fractions import Fraction
from typing import Optional

from ..schemas import AnnualFinancials, BalanceSheetSnapshot, CompanyData, CompRow, MarketData
from ..utils import is_num
from .base import DataError, DataProvider
from .edgar import _EARLY_JANUARY_DAYS, EdgarClient, _fy_list
from .market import (
    _PEER_SOURCE,
    BACKFILL_MAX_DAYS,
    YFinanceClient,
    captive_finance_warning,
    major_currency,
    scale_fundamentals,
)

# EDGAR fundamentals whose latest fiscal year (see `_latest_period_end`) ended
# more than this many days ago are stale. A 10-K filer's latest fiscal year
# ended at most about 15 months ago and an annual-only 20-F filer's about 16,
# so only a filer that stopped tagging us-gaap facts or stopped filing trips it.
_EDGAR_MAX_AGE_DAYS = 730

# The EDGAR client's WARNING when interest expense implies debt its tags do not
# cover ("WARNING: FY2025 interest expense is ... but no debt was found under
# the SEC debt tags this parser reads (the filer may tag its debt with
# company-specific elements); net debt and the WACC debt weight may be
# understated"). Matched on stable phrases: a WARNING that names interest
# expense and contains one of these.
_MISSING_DEBT_PHRASES = ("no debt was found", "company-specific",
                         "debt weight may be understated")
# Cash backfill in that case (Berkshire's Treasury bills): EDGAR's cash equals
# Yahoo's cash and equivalents alone to within this share, while Yahoo's cash
# plus short-term investments is at least this multiple of it.
_CASH_MATCH_TOLERANCE = 0.02
_CASH_MISSING_MULTIPLE = 1.5
# The EDGAR client's gap notes that put debt at 0 ("total debt unavailable on
# EDGAR; set to 0.0"; "<debt label> last reported <date> (...); treated as 0
# at the <date> balance sheet"), marked as superseded once debt is backfilled.
_EDGAR_ZERO_DEBT_NOTE = "total debt unavailable on EDGAR"
_EDGAR_DEBT_LABELS = ("long-term debt", "current portion of long-term debt",
                      "short-term borrowings", "current debt", "total debt", "debt line (")
_SUPERSEDED_DEBT_NOTE = ("; replaced by the Yahoo balance-sheet backfill (see the "
                         "interest-expense WARNING)")
# Its gap notes that put cash at 0 ("cash & equivalents unavailable on EDGAR;
# set to 0.0"; "cash & equivalents last reported <date> (...); treated as 0 at
# the <date> balance sheet"; likewise for short-term investments), marked the
# same way once cash is backfilled with the debt.
_EDGAR_ZERO_CASH_NOTE = "cash & equivalents unavailable on EDGAR"
_EDGAR_CASH_LABELS = ("cash & equivalents", "short-term investments")

# The EDGAR client's gap notes for capex years no capex tag covers ("capex
# unavailable on EDGAR; filled with 0.0" for every year; "capex not reported
# on EDGAR for FY2019-2025; filled with 0.0" for those years).
_EDGAR_CAPEX_ALL_NOTE = "capex unavailable on EDGAR"
_EDGAR_CAPEX_YEARS_NOTE = "capex not reported on EDGAR for "
_FY_SPAN = re.compile(r"FY(\d{4})(?:-(\d{4}))?")
# EDGAR financial kinds whose capex gaps are not backfilled: capex enters only
# the FCFF DCF and FCFE, which for these are shown for reference only, so a
# Yahoo request would only move reference figures (kind -> label for the note).
_CAPEX_BACKFILL_SKIP_KINDS = {"bank": "a bank", "insurer": "an insurer",
                              "bdc": "a business development company", "reit": "a REIT"}
# Yahoo's capex over EDGAR's, as a median over the years both report, must lie
# in this band for Yahoo's figures to fill EDGAR's gaps: outside it the two
# define capex differently (Oportun FY2022: Yahoo 48.9M against EDGAR's 6.0M,
# 8.2x) and the filled years would not be comparable with the reported ones.
# United Rentals (1.00-1.05) and Asbury (1.14) match.
_CAPEX_MATCH_BAND = (0.8, 1.25)
# A lessor's Yahoo capex above this multiple of EDGAR's D&A for the same years
# (Avis Budget: 10-15B of vehicle purchases against 0.2B) includes fleet
# purchases whose depreciation EDGAR's D&A leaves out, so it is not filled.
# AerCap's aircraft purchases, about 2.4x its D&A, are.
_LESSOR_MAX_CAPEX_TO_DA = 5.0

# The EDGAR client's gap notes for D&A years no D&A tag covers ("D&A
# unavailable on EDGAR; filled with 0.0"; "D&A not reported on EDGAR for
# FY2021-2022; filled with 0.0"). PG&E and Carvana tag their D&A only with
# company-specific elements, and a D&A of 0 against their reported capex
# leaves FCFF far too low.
_EDGAR_DA_ALL_NOTE = "D&A unavailable on EDGAR"
_EDGAR_DA_YEARS_NOTE = "D&A not reported on EDGAR for "
# Kinds whose D&A gaps are not backfilled (kind -> label for the note): those
# of the capex backfill, and a debt-funded lessor. For each, the FCFF DCF and
# FCFE that D&A feeds (with capex) are shown for reference only. A lessor's
# Yahoo D&A may also include the lease fleet's depreciation, which EDGAR's
# D&A can leave out (Avis Budget), and the lessor check of the capex backfill
# (`_LESSOR_MAX_CAPEX_TO_DA`) needs EDGAR's own D&A.
_DA_BACKFILL_SKIP_KINDS = dict(_CAPEX_BACKFILL_SKIP_KINDS, lessor="a debt-funded lessor")
# The EDGAR client's notes for EBIT years it left at 0.0: no operating income
# and nothing to derive it from ("EBIT (operating income) not reported on
# EDGAR for FY2018-2021; filled with 0.0", as for Carvana), or a lender's
# years it does not derive ("EBIT (OperatingIncomeLoss) not reported on EDGAR
# for FY2020; left at 0.0 rather than derived as ..."). The D&A backfill does
# not recompute those years' EBITDA: D&A alone would pass for a small EBITDA.
_EDGAR_EBIT_ALL_NOTE = "EBIT (operating income) unavailable on EDGAR"
_EDGAR_EBIT_YEARS_NOTE = "EBIT (operating income) not reported on EDGAR for "
_EDGAR_EBIT_LEFT_NOTE = "EBIT (OperatingIncomeLoss) not reported on EDGAR for "
_EDGAR_EBIT_LEFT_MARK = "; left at 0.0"
# Yahoo's D&A over EDGAR's, as a median over the years both report, must lie in
# this band (the capex backfill's) for Yahoo's figures to fill EDGAR's gaps.
_DA_MATCH_BAND = _CAPEX_MATCH_BAND

# A D0 of 0 from Yahoo's payment history (no payment in the last 12 months)
# is replaced from the statements when the latest fiscal year's dividends
# paid per share exceed this multiple of the payments the history shows with
# an ex-date in that year or the `_DIVIDEND_LAG_DAYS` before it (cash follows
# the ex-date). Yahoo's ADR histories can stop early: TM's ends 2025-03-31
# (3.46 a share in the window) against FY2026 dividends of about 6.69. When
# the year's end date is not known (see `_fiscal_year_end`), the window
# starts `_UNDATED_YEAR_DAYS` before the upper bound instead, so that it
# covers the year wherever it ended in its label year.
_STALE_HISTORY_MULTIPLE = 1.5
_DIVIDEND_LAG_DAYS = 90
_UNDATED_YEAR_DAYS = 2 * 365

# Yahoo split records after the latest fiscal year (see
# `_corporate_action_warning`): a factor (or its reciprocal, for a reverse
# split such as 1-for-20) within this share of a ratio p/q with q at most
# `_SPLIT_MAX_DENOMINATOR` is a plain split or consolidation (CRWD 4.0, a
# 1-for-2 0.5); others adjust the price for value paid out (HON 0.9535: the
# Aerospace spin-off with a 1-for-2 consolidation; CMCSA 1.067).
_SPLIT_RATIO_TOLERANCE = 0.001
_SPLIT_MAX_DENOMINATOR = 10
# Unless the share count moved with the factors (a stock dividend or bonus
# issue, whose factors multiply to more than 1): Yahoo's share count over the
# latest fiscal year's diluted shares within this share of their product.
_STOCK_DIVIDEND_TOLERANCE = 0.05

# A class-share ticker typed with a dot (BRK.B, BF.B, HEI.A); Yahoo and the
# SEC directory write it with a dash. The EDGAR client's error for a ticker
# the directory does not list (any other error means it could not be read).
_CLASS_SHARE_DOT = re.compile(r"[A-Z]{1,6}\.[A-Z]")
_SEC_NOT_LISTED = "not found in SEC EDGAR directory"


def _today() -> datetime.date:
    """Today's date; the reference for the EDGAR staleness guard."""
    return datetime.date.today()


def _notes_of(obj: object) -> list[str]:
    """The ``_source_notes`` a client attached to `obj` (tolerates stubs)."""
    notes = getattr(obj, "_source_notes", None)
    return [str(n) for n in notes] if isinstance(notes, list) else []


def _latest_fiscal_year(financials: object) -> Optional[int]:
    """Newest positive fiscal-year label of `financials`, or None."""
    years = [y for y in getattr(financials, "fiscal_years", None) or []
             if isinstance(y, int) and y > 0]
    return max(years) if years else None


def _latest_period_end(
    financials: object, balance_sheet: object
) -> Optional[datetime.date]:
    """Latest period end the fundamentals' flows cover, or None if undated.

    31 December of the latest fiscal-year label: EDGAR labels a year by the
    calendar year of its end (give or take the early-January 52/53-week rule),
    so no year so labelled ends later. The DCF runs on the flows, so a current
    balance-sheet date does not make years-old flows current, and a stale one
    does not make fresh flows old. The balance-sheet date is used only when no
    fiscal year is labelled.
    """
    year = _latest_fiscal_year(financials)
    if year is not None and year <= 9999:
        return datetime.date(year, 12, 31)
    try:
        return datetime.date.fromisoformat(str(getattr(balance_sheet, "as_of", ""))[:10])
    except ValueError:
        return None


def _missing_debt_warning(notes: list[str]) -> Optional[int]:
    """Index of the EDGAR client's interest-without-debt WARNING, or None."""
    for i, note in enumerate(notes):
        low = note.lower()
        if (note.startswith("WARNING") and "interest expense" in low
                and any(p in low for p in _MISSING_DEBT_PHRASES)):
            return i
    return None


def _edgar_zero_debt_note(note: str) -> bool:
    """True for an EDGAR gap note that put (part of) total debt at 0."""
    return note.startswith(_EDGAR_ZERO_DEBT_NOTE) or (
        " treated as 0 at the " in note and note.startswith(_EDGAR_DEBT_LABELS)
    )


def _edgar_zero_cash_note(note: str) -> bool:
    """True for an EDGAR gap note that put (part of) cash and short-term
    investments at 0."""
    return note.startswith(_EDGAR_ZERO_CASH_NOTE) or (
        " treated as 0 at the " in note and note.startswith(_EDGAR_CASH_LABELS)
    )


def _edgar_capex_gap(notes: list[str], years: list) -> tuple[Optional[int], set]:
    """``(index, years)`` of the EDGAR client's capex gap note: the fiscal
    years it zero-filled (all of `years` for "capex unavailable"), or
    ``(None, set())`` when there is no such note."""
    return _edgar_gap(notes, years, _EDGAR_CAPEX_ALL_NOTE, _EDGAR_CAPEX_YEARS_NOTE)


def _edgar_da_gap(notes: list[str], years: list) -> tuple[Optional[int], set]:
    """As `_edgar_capex_gap`, for the EDGAR client's D&A gap note."""
    return _edgar_gap(notes, years, _EDGAR_DA_ALL_NOTE, _EDGAR_DA_YEARS_NOTE)


def _edgar_gap(
    notes: list[str], years: list, all_note: str, years_note: str
) -> tuple[Optional[int], set]:
    """``(index, years)`` of the first note starting with `all_note` (every
    one of `years`) or `years_note` (the fiscal years it lists, as
    "FY2019-2021, FY2024"), or ``(None, set())``."""
    for i, note in enumerate(notes):
        if note.startswith(all_note):
            return i, set(years)
        if note.startswith(years_note):
            return i, _listed_years(note[len(years_note):])
    return None, set()


def _listed_years(text: str) -> set:
    """The fiscal years `text` lists before its first ';' ("FY2019-2021,
    FY2024")."""
    years: set = set()
    for first, last in _FY_SPAN.findall(text.split(";")[0]):
        years.update(range(int(first), int(last or first) + 1))
    return years


def _edgar_ebit_gap(notes: list[str], years: list) -> set:
    """The fiscal years whose EBIT the EDGAR client left at 0.0 (see
    `_EDGAR_EBIT_ALL_NOTE`), or an empty set."""
    _, gap = _edgar_gap(notes, years, _EDGAR_EBIT_ALL_NOTE, _EDGAR_EBIT_YEARS_NOTE)
    for note in notes:
        if note.startswith(_EDGAR_EBIT_LEFT_NOTE) and _EDGAR_EBIT_LEFT_MARK in note:
            gap |= _listed_years(note[len(_EDGAR_EBIT_LEFT_NOTE):])
    return gap


def _times(ratio: float) -> str:
    """A ratio for a note: '8.2x', or '0.43x' below 1."""
    return f"{ratio:.1f}x" if ratio >= 1 else f"{ratio:.2f}x"


def _copy_financials(financials: AnnualFinancials, **changes) -> Optional[AnnualFinancials]:
    """A ``dataclasses.replace`` copy of `financials` with `changes` that keeps
    the client's dynamic attributes (``_source_notes``, ``_financial_kind``,
    ``_annual_form``, ...), or None when it is not a dataclass (a test stub)."""
    try:
        extras = {k: v for k, v in vars(financials).items() if k.startswith("_")}
        result = dataclasses.replace(financials, **changes)
        for k, v in extras.items():
            setattr(result, k, v)
    except TypeError:
        return None
    return result


def _latest_positive(values: list, years: list) -> tuple[Optional[float], Optional[int]]:
    """Newest positive finite value of an oldest->newest series, with its year."""
    for i in range(len(values) - 1, -1, -1):
        v = values[i]
        if is_num(v) and v > 0:
            return float(v), (years[i] if i < len(years) else None)
    return None, None


def _fiscal_year_end(
    financials: object,
    balance_sheet: object,
    from_fallback: bool,
    reported: object = None,
) -> tuple[Optional[datetime.date], bool]:
    """``(end, exact)`` for the latest fiscal year the flows cover.

    A fiscal-year label L covers year ends from 15 January of L to 14
    January of L+1: a 52/53-week year ending in the first two weeks of
    January takes the prior year's label, on EDGAR and in the yfinance
    fallback alike. The exact end is the first of these that falls in that
    range:
      * for the yfinance fallback, its balance-sheet date, which is the
        fiscal year-end column itself (TM: 2026-03-31);
      * for EDGAR, the period end of its latest fiscal year (the EDGAR
        client's ``_latest_fy_end`` marker; see
        ``EdgarClient._latest_annual_report``);
      * `reported`, Yahoo's ``lastFiscalYearEnd`` (the market client's
        ``_last_fiscal_year_end``; FDX: 2026-05-31), or the same day a year
        earlier or later, for when Yahoo already has a year the statements
        do not (its annual report is not out yet) or still lacks one they have.
    Otherwise it is ``(31 December of L, False)`` (`_latest_period_end`), an
    upper bound except for a year ending in early January, and ``(None,
    False)`` when the flows are undated.
    """
    end = _latest_period_end(financials, balance_sheet)
    year = _latest_fiscal_year(financials)
    if end is None or year is None or year > 9998:
        return end, False
    first = datetime.date(year, 1, _EARLY_JANUARY_DAYS + 1)
    last = datetime.date(year + 1, 1, _EARLY_JANUARY_DAYS)
    candidates: list[datetime.date] = []
    own = (getattr(balance_sheet, "as_of", "") if from_fallback
           else getattr(financials, "_latest_fy_end", ""))
    try:
        candidates.append(datetime.date.fromisoformat(str(own or "")[:10]))
    except ValueError:
        pass
    if isinstance(reported, datetime.datetime):
        reported = reported.date()
    if isinstance(reported, datetime.date):
        for shift in (0, -1, 1):
            try:
                candidates.append(reported.replace(year=reported.year + shift))
            except ValueError:  # 29 February, or out of the calendar's range
                pass
    for day in candidates:
        if first <= day <= last:
            return day, True
    return end, False


def _plain_split(factor: float) -> bool:
    """True when a Yahoo split factor is a simple share ratio (4.0, 0.5, 1.5)
    or the reciprocal of one (a 1-for-20 reverse split's 0.05)."""
    for value in (factor, 1.0 / factor):
        ratio = Fraction(value).limit_denominator(_SPLIT_MAX_DENOMINATOR)
        if ratio > 0 and abs(value / ratio - 1.0) <= _SPLIT_RATIO_TOLERANCE:
            return True
    return False


def _corporate_action_warning(
    actions: object,
    financials: object,
    fy_end: Optional[datetime.date],
    yahoo_shares: object,
) -> Optional[str]:
    """A WARNING for Yahoo split records after the latest fiscal year that are
    not plain splits, or None.

    The statements of that year still include anything the company has
    since spun off, while the price and share count no longer do (HON
    after the Aerospace spin-off: FY2025 earnings over the post-spin share
    count gave 'Undervalued +47%'). A plain split (CRWD's 4-for-1) moves
    price and share count together and is silent. So is a stock dividend or
    bonus issue: the other factors multiply to more than 1 (it adds shares)
    and Yahoo's share count over the latest diluted shares is within
    `_STOCK_DIVIDEND_TOLERANCE` of their product (statement counts already
    restated for the plain splits, as EDGAR's are for a split after the
    annual report and Yahoo's for every split) or of the product of every
    factor (counts on the old basis). A product below 1 (HON's 0.9535, a
    distribution with a 1-for-2 consolidation) always warns: HON's EDGAR
    count, restated for the consolidation, moved about as much as that
    factor. Limits: a spin-off Yahoo records only as a share ratio is not
    caught, nor one whose factor is under about 1.05 (an unchanged share
    count matches it within the tolerance); a stock dividend above about 5%
    that the statement counts already include warns. `fy_end` comes from
    `_fiscal_year_end` (without a known year-end date it is an upper bound,
    so events just after a non-December year end can be missed).
    """
    if not isinstance(actions, (list, tuple)) or fy_end is None:
        return None
    after = sorted(
        (d, float(f)) for d, f in (a for a in actions if isinstance(a, tuple) and len(a) == 2)
        if isinstance(d, datetime.date) and d > fy_end and is_num(f) and f > 0
    )
    paid_out = [(d, f) for d, f in after if not _plain_split(f)]
    if not paid_out:
        return None
    years = list(getattr(financials, "fiscal_years", None) or [])
    fy_shares, _ = _latest_positive(list(getattr(financials, "diluted_shares", None) or []),
                                    years)
    issued = math.prod(f for _, f in paid_out)
    if issued > 1.0 and fy_shares is not None and is_num(yahoo_shares) and yahoo_shares > 0:
        moved = yahoo_shares / fy_shares
        if any(abs(moved / product - 1.0) <= _STOCK_DIVIDEND_TOLERANCE
               for product in (issued, math.prod(f for _, f in after))):
            return None
    fy = _latest_fiscal_year(financials)
    events = ", ".join(f"{d.isoformat()} (factor {f:.4g})" for d, f in paid_out)
    return (
        f"WARNING: Yahoo records a corporate action after FY{fy} that is not a plain "
        f"split: {events}; likely a spin-off or other distribution of value, so the FY{fy} "
        "statements may still include a business the company no longer owns and the "
        "per-share values and verdict are unreliable until the next annual report"
    )


class HybridProvider(DataProvider):
    """Combine SEC EDGAR fundamentals with yfinance market data/comps."""

    def __init__(
        self,
        edgar: Optional[EdgarClient] = None,
        market: Optional[YFinanceClient] = None,
    ) -> None:
        # Construct the underlying clients once. Allow injection so tests (and the
        # engine, if it wants custom config) can swap in stubs/preconfigured
        # instances. We do NOT make live calls here -- only on demand.
        self.edgar: EdgarClient = edgar if edgar is not None else EdgarClient()
        self.market: YFinanceClient = market if market is not None else YFinanceClient()

    # ------------------------------------------------------------------ #
    #  Primary entry point: full CompanyData assembly
    # ------------------------------------------------------------------ #
    def get_company_data(self, ticker: str) -> CompanyData:
        """Return fully-populated ``CompanyData`` for ``ticker``.

        Market data is mandatory (we need a live price for every downstream
        valuation). Fundamentals are sourced from EDGAR when possible, otherwise
        from the yfinance fallback; EDGAR data more than two years old gives way
        to a fallback with a newer fiscal year (WARNING note either way).
        ``DataError`` is raised only when no usable fundamentals can be
        obtained from either source.
        """
        symbol = (ticker or "").strip().upper()
        source_notes: list[str] = []

        # --- 1) Market data (required). Let DataError propagate; without a price
        #         the company cannot be valued. A class-share ticker typed with a
        #         dot (BRK.B) is retried in the dashed form the SEC lists. ------
        try:
            market: MarketData = self.market.get_market_data(symbol)
        except DataError:
            alias = self._class_share_alias(symbol)
            if alias is None:
                raise
            market = self.market.get_market_data(alias)
            source_notes.append(
                f"Yahoo has no quote for {symbol}; valued as {alias}, the SEC directory's "
                "form of this class-share ticker (Yahoo writes class shares with a dash)"
            )
            symbol = alias

        # --- 2) Fundamentals: try EDGAR first. ------------------------------------
        financials = None
        balance_sheet = None
        cik: Optional[str] = None
        edgar_name: Optional[str] = None

        from_fallback = False
        try:
            financials, balance_sheet, cik, edgar_name = (
                self.edgar.get_annual_financials(symbol)
            )
        except DataError as exc:
            # Expected, well-understood failure (e.g. ticker not in SEC map / no
            # XBRL facts). Record it and fall through to the fallback.
            source_notes.append(f"EDGAR unavailable: {exc}")
            financials = balance_sheet = None
            cik = edgar_name = None
        except Exception as exc:  # noqa: BLE001 -- never let a provider bug crash us
            # Any other EDGAR error (network hiccup, unexpected JSON shape, etc.).
            source_notes.append(f"EDGAR error: {exc}")
            financials = balance_sheet = None
            cik = edgar_name = None

        # A history that ended years ago (Toyota: us-gaap facts stop at FY2013)
        # is only used when the yfinance fallback has nothing newer.
        stale = None
        if financials is not None and balance_sheet is not None:
            stale = self._edgar_staleness(financials, balance_sheet, cik)
            if stale is None:
                source_notes.append(f"Fundamentals: SEC EDGAR (CIK {cik})")

        # --- 3) Fall back to yfinance-built fundamentals if EDGAR gave us nothing
        #         or only a stale history. --------------------------------------
        stale_warning: Optional[str] = None
        if financials is None or balance_sheet is None or stale is not None:
            fallback_notes: list[str] = []
            fallback = self._fallback_fundamentals(symbol, fallback_notes)
            if stale is not None:
                edgar_fy = _latest_fiscal_year(financials)
                fallback_fy = _latest_fiscal_year(fallback[0]) if fallback else None
                if fallback_fy is not None and (edgar_fy is None or fallback_fy > edgar_fy):
                    source_notes.append(
                        f"WARNING: {stale}; using the yfinance fallback (to "
                        f"FY{fallback_fy}) instead"
                    )
                else:
                    fallback = None  # nothing newer: keep EDGAR, with a warning
                    source_notes.append(f"Fundamentals: SEC EDGAR (CIK {cik})")
                    stale_warning = (
                        f"WARNING: {stale}, and the yfinance fallback has no newer fiscal "
                        "year; the valuation uses this stale history"
                    )
            source_notes.extend(fallback_notes)
            if fallback is not None:
                financials, balance_sheet = fallback
                cik = None  # yfinance fundamentals have no CIK
                from_fallback = True
                source_notes.append("Fundamentals: yfinance fallback")

        # --- 4) If both sources failed, we cannot value the company. -------------
        if financials is None or balance_sheet is None:
            detail = " ".join(source_notes)
            raise DataError(
                f"No usable fundamentals for {symbol!r} from SEC EDGAR or yfinance."
                + (f" {detail}" if detail else "")
            )

        # --- 5) Data-quality notes, currency alignment and market backfills. ------
        # Read the clients' notes before any dataclass copy drops them.
        fundamentals_notes = _notes_of(financials)
        extra = _notes_of(market)
        if not from_fallback:
            balance_sheet = self._backfill_edgar_debt(
                symbol, financials, balance_sheet, fundamentals_notes
            )
            financials = self._backfill_edgar_da(symbol, financials, fundamentals_notes)
            financials = self._backfill_edgar_capex(symbol, financials, fundamentals_notes)
            financials, balance_sheet = self._edgar_to_quote_currency(
                financials, balance_sheet, market, extra
            )
        # Read Yahoo's year-end date, split records and share count before
        # the backfill copy drops the client's markers.
        fy_end, fy_end_exact = _fiscal_year_end(
            financials, balance_sheet, from_fallback,
            getattr(market, "_last_fiscal_year_end", None),
        )
        action_warning = _corporate_action_warning(
            getattr(market, "_corporate_actions", None), financials, fy_end,
            getattr(market, "shares_outstanding", None),
        )
        market = self._backfill_market(
            market, financials, from_fallback, extra, fy_end, fy_end_exact)
        if action_warning is not None:
            extra.append(action_warning)
        extra.extend(fundamentals_notes)
        if stale_warning is not None:
            extra.append(stale_warning)
        # WARNING notes (unconverted currencies, no market cap) lead the list.
        source_notes.extend(n for n in extra if n.startswith("WARNING"))
        source_notes.extend(n for n in extra if not n.startswith("WARNING"))

        # --- 6) Resolve the display name. Prefer EDGAR's registered name, then
        #         the market name, then the ticker as a last resort. --------------
        name = edgar_name or getattr(market, "name", None) or symbol

        return CompanyData(
            ticker=symbol,
            name=name,
            cik=cik,
            financials=financials,
            balance_sheet=balance_sheet,
            market=market,
            source_notes=source_notes,
        )

    # ------------------------------------------------------------------ #
    #  Helpers: fundamentals source selection
    # ------------------------------------------------------------------ #
    @staticmethod
    def _edgar_staleness(
        financials: AnnualFinancials, balance_sheet: BalanceSheetSnapshot, cik: Optional[str]
    ) -> Optional[str]:
        """Why the EDGAR fundamentals are stale, or None when they are current.

        Stale means their latest fiscal year (the balance-sheet date only when
        no year is labelled; see `_latest_period_end`) ended more than
        `_EDGAR_MAX_AGE_DAYS` before today. Undated data is not judged here.
        """
        latest_end = _latest_period_end(financials, balance_sheet)
        today = _today()
        if latest_end is None or (today - latest_end).days <= _EDGAR_MAX_AGE_DAYS:
            return None
        as_of = str(getattr(balance_sheet, "as_of", "") or "") or "undated"
        return (
            f"EDGAR fundamentals (CIK {cik}) end FY{_latest_fiscal_year(financials)} "
            f"(balance sheet {as_of}), more than {_EDGAR_MAX_AGE_DAYS // 365} years "
            f"before {today.isoformat()}: the filer no longer tags us-gaap facts "
            "(e.g. it moved to IFRS) or stopped filing"
        )

    def _fallback_fundamentals(
        self, symbol: str, notes: list[str]
    ) -> Optional[tuple[AnnualFinancials, BalanceSheetSnapshot]]:
        """The yfinance statement fallback, or None (an error is noted).

        The builder is best-effort and returns None on failure; a raise or a
        result that is not a ``(financials, balance_sheet)`` pair also gives None.
        """
        try:
            fallback = self.market.get_annual_financials_fallback(symbol)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"yfinance fallback error: {exc}")
            return None
        if isinstance(fallback, tuple) and len(fallback) == 2:
            return fallback
        return None

    # ------------------------------------------------------------------ #
    #  Helpers: EDGAR debt backfill, currency alignment, market backfill
    # ------------------------------------------------------------------ #
    def _backfill_edgar_debt(
        self,
        symbol: str,
        financials: AnnualFinancials,
        balance_sheet: BalanceSheetSnapshot,
        notes: list[str],
    ) -> BalanceSheetSnapshot:
        """Fill EDGAR debt that company-specific tags hid, from Yahoo.

        Runs only when `notes` hold the EDGAR client's interest-without-debt
        WARNING (see `_MISSING_DEBT_PHRASES`) and EDGAR's total debt is 0: a
        non-zero EDGAR figure is never overwritten. Known limitation: debt
        that EDGAR's tags cover only in part (the WARNING then reads "total
        debt read is only ...") stays understated; that WARNING remains in
        the notes. Yahoo's balance sheet
        nearest the EDGAR date (quarterly or annual, within
        `market.BACKFILL_MAX_DAYS`; see
        :meth:`YFinanceClient.get_balance_sheet_items`) supplies
          * total debt: long-term plus current debt, lease obligations
            excluded (Ford 0 -> about 157B, Berkshire 0 -> about 129B);
          * cash and short-term investments, only when EDGAR's cash equals
            Yahoo's cash and equivalents alone (within
            `_CASH_MATCH_TOLERANCE`) and Yahoo's total with short-term
            investments is at least `_CASH_MISSING_MULTIPLE` times it
            (Berkshire's Treasury bills: 40.6B -> 365.5B), or when EDGAR's
            cash and short-term investments are 0 and the debt was
            backfilled (PACCAR: its cash tags went stale with its debt tags;
            0 -> about 8.8B);
          * the "captive_finance" mark with a leading WARNING, when that
            balance sheet shows a captive finance arm and EDGAR's tags gave no
            kind (Ford Credit's receivables).
        When the backfilled debt includes (or may include) lease obligations
        (``debt_includes_leases`` not False), EDGAR's
        ``operating_lease_liability`` is cleared so leases are not counted
        twice.
        What was done is appended to the EDGAR WARNING, which stays: the
        figures now come from another source and date. EDGAR's own notes that
        put debt (or, when it was backfilled, cash) at 0 are marked as
        replaced. Yahoo amounts in another
        currency (its statement tables' currency; see
        :meth:`YFinanceClient.get_balance_sheet_items`) are converted to USD
        at spot; without a rate nothing is filled. Returns a copy;
        `balance_sheet` is not mutated.
        """
        idx = _missing_debt_warning(notes)
        debt = getattr(balance_sheet, "total_debt", None)
        if idx is None or not is_num(debt) or debt > 0:
            return balance_sheet
        as_of = str(getattr(balance_sheet, "as_of", "") or "")
        items = None
        get_items = getattr(self.market, "get_balance_sheet_items", None)
        if callable(get_items):
            try:
                items = get_items(symbol, as_of)
            except Exception:  # noqa: BLE001 -- best-effort
                items = None
        if not isinstance(items, dict):
            notes[idx] += (f"; no Yahoo balance sheet within {BACKFILL_MAX_DAYS} days of "
                           f"{as_of or 'the EDGAR date'} to backfill it from")
            return balance_sheet

        ccy = items.get("currency")
        rate = self._usd_rate(ccy)
        if rate is None:
            notes[idx] += (f"; Yahoo's balance sheet is in {ccy} and no {ccy}->USD rate "
                           "could be fetched, so nothing was backfilled")
            return balance_sheet

        changes: dict = {}
        done: list[str] = []
        cash_was_zero = False
        y_debt = items.get("total_debt")
        if is_num(y_debt) and y_debt > 0:
            changes["total_debt"] = y_debt * rate
            basis = items.get("debt_basis") or "total debt"
            done.append(f"total debt {y_debt * rate:,.0f} ({basis})")
            # EDGAR's operating-lease liabilities are a separate claim only
            # while the debt leaves leases out; Yahoo's debt that includes (or
            # may include) them would count them twice in the comps bridge.
            if (items.get("debt_includes_leases") is not False
                    and getattr(balance_sheet, "operating_lease_liability", None) is not None):
                changes["operating_lease_liability"] = None
        cash = getattr(balance_sheet, "cash_and_investments", None)
        y_cash, y_all = items.get("cash"), items.get("cash_and_investments")
        if (is_num(cash) and cash > 0 and is_num(y_cash) and is_num(y_all)
                and abs(y_cash * rate / cash - 1.0) <= _CASH_MATCH_TOLERANCE
                and y_all * rate >= _CASH_MISSING_MULTIPLE * cash):
            changes["cash_and_investments"] = y_all * rate
            done.append(
                f"cash and short-term investments {y_all * rate:,.0f} (EDGAR's {cash:,.0f} "
                "is cash and equivalents alone; the short-term investments, e.g. Treasury "
                "bills, are under company-specific tags too)"
            )
        elif ("total_debt" in changes and is_num(cash) and cash <= 0
              and is_num(y_all) and y_all > 0):
            # PACCAR: its cash tags went stale when the debt tags did, so
            # EDGAR's cash is 0 at the date the debt now comes from.
            changes["cash_and_investments"] = y_all * rate
            cash_was_zero = True
            done.append(
                f"cash and short-term investments {y_all * rate:,.0f} (EDGAR's are 0: its "
                "cash tags are stale or company-specific, like its debt tags)"
            )
        y_as_of = items.get("as_of") or "an undated column"
        if done:
            ccy_note = items.get("currency_note")
            notes[idx] += (f"; backfilled from Yahoo's balance sheet at {y_as_of}: "
                           + "; ".join(done)
                           + (f" ({ccy_note})" if isinstance(ccy_note, str) and ccy_note else ""))
            for i, note in enumerate(notes):
                if (("total_debt" in changes and _edgar_zero_debt_note(note))
                        or (cash_was_zero and _edgar_zero_cash_note(note))):
                    notes[i] = note + _SUPERSEDED_DEBT_NOTE
        else:
            notes[idx] += (f"; Yahoo's balance sheet at {y_as_of} shows no debt either, so "
                           "nothing was backfilled")

        captive = items.get("captive")
        if isinstance(captive, str) and not getattr(financials, "_financial_kind", None):
            try:
                setattr(financials, "_financial_kind", "captive_finance")
                notes.insert(0, captive_finance_warning("Yahoo's balance-sheet lines", captive))
            except Exception:  # noqa: BLE001 -- a frozen stub: leave it unmarked
                pass

        if not changes:
            return balance_sheet
        try:
            return dataclasses.replace(balance_sheet, **changes)
        except TypeError:  # not a dataclass (a test stub)
            return balance_sheet

    def _usd_rate(self, ccy: object) -> Optional[float]:
        """Multiplier from Yahoo statement amounts in `ccy` to EDGAR's USD:
        1.0 for USD or an unknown currency, the spot rate otherwise, None
        when that rate cannot be fetched."""
        if not (isinstance(ccy, str) and ccy and ccy != "USD"):
            return 1.0
        fx = None
        try:
            fx = self.market.get_fx_rate(ccy, "USD")
        except Exception:  # noqa: BLE001 -- best-effort
            fx = None
        if not (isinstance(fx, tuple) and len(fx) == 2 and is_num(fx[0]) and fx[0] > 0):
            return None
        return float(fx[0])

    def _backfill_edgar_da(
        self,
        symbol: str,
        financials: AnnualFinancials,
        notes: list[str],
    ) -> AnnualFinancials:
        """Fill EDGAR D&A years that no D&A tag covered, from Yahoo.

        EDGAR zero-fills a D&A year no D&A tag covers (PG&E and Carvana tag
        their D&A only with company-specific elements). A D&A of 0 against
        reported capex makes FCFF far too low and leaves EBITDA equal to
        EBIT: PG&E's DCF goes from -126.21 a share to -7.22 on Yahoo's
        FY2022-2025 D&A, and Carvana's from 8.19 to 13.15. Runs only for
        the years the EDGAR client's D&A gap note names (see
        `_EDGAR_DA_ALL_NOTE`) whose D&A is still 0: a non-zero EDGAR figure
        is never overwritten. Each such year takes Yahoo's D&A for the same
        fiscal year (see :meth:`YFinanceClient.get_da_by_year`: the income
        statement's line, else the cash-flow statement's), converted to USD
        at spot when Yahoo's tables are in another currency, and its EBITDA
        is recomputed as EBIT + D&A, except in a year whose EBIT the EDGAR
        client left at 0.0 as well (`_EDGAR_EBIT_ALL_NOTE`): there EBITDA
        stays at EDGAR's 0.0, since D&A alone would pass for a small EBITDA.
        Nothing is filled, and the gap note says why, when
          * EDGAR marks a bank, insurer, BDC, REIT or debt-funded lessor
            (`_DA_BACKFILL_SKIP_KINDS`; no Yahoo request is made);
          * Yahoo's statements could not be read, show no D&A for those
            years, or are in a currency with no rate to USD;
          * in the years both report, Yahoo's D&A is a median outside
            `_DA_MATCH_BAND` of EDGAR's (another D&A definition).
        Years Yahoo does not cover (it shows about four) stay 0.0. What was
        done is appended to the EDGAR gap note. Runs before
        `_backfill_edgar_capex`. Returns a copy that keeps the client's
        dynamic attributes; `financials` is not mutated.
        """
        years = list(getattr(financials, "fiscal_years", None) or [])
        da = list(getattr(financials, "dep_amort", None) or [])
        idx, gap = _edgar_da_gap(notes, years)
        todo = [y for y, v in zip(years, da) if y in gap and is_num(v) and v == 0]
        if idx is None or not todo:
            return financials
        kind = getattr(financials, "_financial_kind", None)
        if kind in _DA_BACKFILL_SKIP_KINDS:
            notes[idx] += (f"; not backfilled from Yahoo: for {_DA_BACKFILL_SKIP_KINDS[kind]} "
                           "the FCFF DCF and FCFE, which D&A feeds, are shown for reference "
                           "only (EBITDA for those years is EBIT alone)")
            return financials
        items = None
        get_da = getattr(self.market, "get_da_by_year", None)
        if callable(get_da):
            try:
                items = get_da(symbol)
            except Exception:  # noqa: BLE001 -- best-effort
                items = None
        if not isinstance(items, dict):
            notes[idx] += ("; Yahoo's annual income and cash-flow statements could not be read "
                           "(the request failed or returned nothing), so nothing was backfilled")
            return financials
        by_year = items.get("by_year")
        by_year = by_year if isinstance(by_year, dict) else {}
        found = {y: by_year[y] for y in todo if y in by_year}
        if not found:
            notes[idx] += (f"; Yahoo's annual statements show no D&A for {_fy_list(todo)} "
                           "either, so nothing was backfilled")
            return financials
        ccy = items.get("currency")
        rate = self._usd_rate(ccy)
        if rate is None:
            notes[idx] += (f"; Yahoo's statements are in {ccy} and no {ccy}->USD rate could "
                           "be fetched, so nothing was backfilled")
            return financials
        # Same definition? Compare the years both report (not EDGAR's gap years).
        both = sorted(y for y, v in zip(years, da)
                      if y not in gap and is_num(v) and v > 0 and y in by_year)
        if both:
            edgar_da = dict(zip(years, da))
            ratio = statistics.median(by_year[y][1] * rate / edgar_da[y] for y in both)
            low, high = _DA_MATCH_BAND
            if not low <= ratio <= high:
                notes[idx] += (
                    f"; not backfilled from Yahoo's annual statements: their D&A for "
                    f"{_fy_list(both)}, which EDGAR also reports, is "
                    f"{'a median ' if len(both) > 1 else ''}{_times(ratio)} EDGAR's, so the "
                    f"two likely define D&A differently; {_fy_list(todo)} left at 0.0")
                return financials

        filled = [found[y][1] * rate if y in found else v for y, v in zip(years, da)]
        filled += da[len(filled):]
        ebit = list(getattr(financials, "ebit", None) or [])
        ebitda = list(getattr(financials, "ebitda", None) or [])
        # Not where EDGAR zero-filled EBIT too: D&A alone is no EBITDA.
        no_ebit = sorted(_edgar_ebit_gap(notes, years) & set(found))
        for i, y in enumerate(years):
            if (y in found and y not in no_ebit and i < len(ebit) and i < len(ebitda)
                    and is_num(ebit[i])):
                ebitda[i] = ebit[i] + filled[i]
        result = _copy_financials(financials, dep_amort=filled, ebitda=ebitda)
        if result is None:  # not a dataclass (a test stub)
            return financials
        amounts = ", ".join(f"FY{y} {found[y][1] * rate:,.0f} (year to {found[y][0]})"
                            for y in sorted(found))
        if rate != 1.0:
            amounts += f" (Yahoo's {ccy} figures at {ccy}->USD spot {rate:.6g})"
        left = [y for y in todo if y not in found]
        ccy_note = items.get("currency_note")
        if not no_ebit:
            ebitda_note = "; EBITDA recomputed as EBIT + D&A for those years"
        else:
            recomputed = [y for y in found if y not in no_ebit]
            ebitda_note = (
                (f"; EBITDA recomputed as EBIT + D&A for {_fy_list(recomputed)}"
                 if recomputed else "")
                + f"; EBITDA for {_fy_list(no_ebit)} left at 0.0, since EBIT is not "
                "reported on EDGAR there either (D&A alone is no EBITDA)")
        notes[idx] += (
            f"; backfilled from Yahoo's annual income or cash-flow statement (D&A for the "
            f"same fiscal year): {amounts}"
            + (f" ({ccy_note})" if isinstance(ccy_note, str) and ccy_note else "")
            + ebitda_note
            + (f"; {_fy_list(left)} not in Yahoo's statements, left at 0.0" if left else "")
        )
        return result

    def _backfill_edgar_capex(
        self,
        symbol: str,
        financials: AnnualFinancials,
        notes: list[str],
    ) -> AnnualFinancials:
        """Fill EDGAR capex years that no capex tag covered, from Yahoo.

        EDGAR zero-fills a capex year no capex tag covers (Phillips 66 tags
        none in any year; NextEra; AerCap's aircraft purchases). The models
        skip such a 0 as a gap and, with no capex year at all, use capex =
        D&A. This replaces that proxy, or widens the window of capex years
        the models pool, with reported figures: Phillips 66's DCF goes from
        233.09 on the proxy to 242.86 on Yahoo's FY2022-2025 capex, and
        NextEra's from 50.86 to 5.07 (capex about 38% of revenue against
        D&A of 23%). Runs only for the years the EDGAR client's capex gap note
        names (see `_EDGAR_CAPEX_ALL_NOTE`) whose capex is still 0: a
        non-zero EDGAR figure is never overwritten. Each such year takes the
        capex of the column of Yahoo's annual cash-flow statement labelled
        with the same fiscal year (see :meth:`YFinanceClient.get_capex_by_year`),
        converted to USD at spot when Yahoo's tables are in another currency.
        Nothing is filled, and the gap note says why, when
          * EDGAR marks a bank, insurer, BDC or REIT
            (`_CAPEX_BACKFILL_SKIP_KINDS`; no Yahoo request is made);
          * Yahoo's statement could not be read, shows no capex for those
            years, or is in a currency with no rate to USD;
          * in the years both report, Yahoo's capex is a median outside
            `_CAPEX_MATCH_BAND` of EDGAR's (another capex definition);
          * for a lessor, Yahoo's capex is a median over
            `_LESSOR_MAX_CAPEX_TO_DA` times EDGAR's D&A for the same years.
        Years Yahoo does not cover (it shows about four) stay 0.0. What was
        done is appended to the EDGAR gap note. Returns a copy that keeps the
        client's dynamic attributes (``_source_notes``, ``_financial_kind``);
        `financials` is not mutated.
        """
        years = list(getattr(financials, "fiscal_years", None) or [])
        capex = list(getattr(financials, "capex", None) or [])
        idx, gap = _edgar_capex_gap(notes, years)
        todo = [y for y, v in zip(years, capex) if y in gap and is_num(v) and v == 0]
        if idx is None or not todo:
            return financials
        kind = getattr(financials, "_financial_kind", None)
        if kind in _CAPEX_BACKFILL_SKIP_KINDS:
            notes[idx] += ("; not backfilled from Yahoo: capex enters only the FCFF DCF and "
                           f"FCFE, which for {_CAPEX_BACKFILL_SKIP_KINDS[kind]} are shown for "
                           "reference only")
            return financials
        items = None
        get_capex = getattr(self.market, "get_capex_by_year", None)
        if callable(get_capex):
            try:
                items = get_capex(symbol)
            except Exception:  # noqa: BLE001 -- best-effort
                items = None
        if not isinstance(items, dict):
            notes[idx] += ("; Yahoo's annual cash-flow statement could not be read (the "
                           "request failed or returned nothing), so nothing was backfilled")
            return financials
        by_year = items.get("by_year")
        by_year = by_year if isinstance(by_year, dict) else {}
        found = {y: by_year[y] for y in todo if y in by_year}
        if not found:
            notes[idx] += (f"; Yahoo's cash-flow statement has no capital expenditure for "
                           f"{_fy_list(todo)} either, so nothing was backfilled")
            return financials
        ccy = items.get("currency")
        rate = self._usd_rate(ccy)
        if rate is None:
            notes[idx] += (f"; Yahoo's cash-flow statement is in {ccy} and no {ccy}->USD "
                           "rate could be fetched, so nothing was backfilled")
            return financials
        not_filled = ("; not backfilled from Yahoo's annual cash-flow statement: its capital "
                      "expenditure for ")
        left_all = f"; {_fy_list(todo)} left at 0.0"
        # Same definition? Compare the years both report (not EDGAR's gap years).
        both = sorted(y for y, v in zip(years, capex)
                      if y not in gap and is_num(v) and v > 0 and y in by_year)
        edgar_capex = dict(zip(years, capex))
        if both:
            ratio = statistics.median(by_year[y][1] * rate / edgar_capex[y] for y in both)
            low, high = _CAPEX_MATCH_BAND
            if not low <= ratio <= high:
                notes[idx] += (
                    f"{not_filled}{_fy_list(both)}, which EDGAR also reports, is "
                    f"{'a median ' if len(both) > 1 else ''}{_times(ratio)} EDGAR's, so the "
                    f"two likely define capex differently{left_all}")
                return financials

        # A lessor's fleet purchases against a D&A without the fleet (Avis Budget).
        if kind == "lessor":
            da = dict(zip(years, getattr(financials, "dep_amort", None) or []))
            da_years = sorted(y for y in found if is_num(da.get(y)) and da[y] > 0)
            if da_years:
                ratio = statistics.median(found[y][1] * rate / da[y] for y in da_years)
                if ratio > _LESSOR_MAX_CAPEX_TO_DA:
                    notes[idx] += (
                        f"{not_filled}{_fy_list(da_years)} is "
                        f"{'a median ' if len(da_years) > 1 else ''}{_times(ratio)} EDGAR's "
                        "D&A for the same years, so it likely includes purchases of the "
                        f"lease fleet, whose depreciation EDGAR's D&A leaves out{left_all}")
                    return financials

        filled = [found[y][1] * rate if y in found else v for y, v in zip(years, capex)]
        filled += capex[len(filled):]
        result = _copy_financials(financials, capex=filled)
        if result is None:  # not a dataclass (a test stub)
            return financials
        amounts = ", ".join(f"FY{y} {found[y][1] * rate:,.0f} (year to {found[y][0]})"
                            for y in sorted(found))
        left = [y for y in todo if y not in found]
        ccy_note = items.get("currency_note")
        if rate != 1.0:
            amounts += f" (Yahoo's {ccy} figures at {ccy}->USD spot {rate:.6g})"
        notes[idx] += (
            f"; backfilled from Yahoo's annual cash-flow statement (capital expenditure "
            f"for the same fiscal year): {amounts}"
            + (f" ({ccy_note})" if isinstance(ccy_note, str) and ccy_note else "")
            + (f"; {_fy_list(left)} not in Yahoo's statement, left at 0.0" if left else "")
        )
        return result

    def _edgar_to_quote_currency(
        self,
        financials: AnnualFinancials,
        balance_sheet: BalanceSheetSnapshot,
        market: MarketData,
        notes: list[str],
    ) -> tuple[AnnualFinancials, BalanceSheetSnapshot]:
        """Convert EDGAR's USD statements when the stock is quoted in another
        currency (rare: EDGAR tickers normally quote in USD)."""
        quote_ccy, _ = major_currency(getattr(market, "currency", None))
        if quote_ccy is None or quote_ccy == "USD":
            return financials, balance_sheet
        fx = None
        get_rate = getattr(self.market, "get_fx_rate", None)
        if callable(get_rate):
            try:
                fx = get_rate("USD", quote_ccy)
            except Exception:  # noqa: BLE001 -- best-effort
                fx = None
        if not (isinstance(fx, tuple) and len(fx) == 2 and is_num(fx[0]) and fx[0] > 0):
            raise DataError(
                f"EDGAR statements are in USD but the share price is in "
                f"{quote_ccy}, and no USD->{quote_ccy} exchange rate could be "
                "fetched. Valuation stopped to avoid mixing currencies; retry when "
                "exchange-rate data is available."
            )
        rate, how = fx
        notes.append(
            f"Fundamentals converted from USD to {quote_ccy} at spot {rate:.6g} "
            f"({how}); every year uses this one rate"
        )
        return scale_fundamentals(financials, balance_sheet, float(rate))

    @staticmethod
    def _backfill_market(
        market: MarketData,
        financials: AnnualFinancials,
        from_fallback: bool,
        notes: list[str],
        fy_end: Optional[datetime.date] = None,
        fy_end_exact: bool = True,
    ) -> MarketData:
        """Fill shares / market cap / DPS that yfinance could not supply.

        A missing (0.0) market cap would otherwise enter the WACC as a zero
        equity weight, and a missing DPS would skip the DDM for a dividend
        payer. A missing DPS (None) is filled from the latest fiscal year's
        dividends paid. A 0.0 is kept, unless it was read from Yahoo's payment
        history (``_dividend_history``, set by the market client) and the
        dividends paid per share in the fiscal year ending `fy_end` exceed
        `_STALE_HISTORY_MULTIPLE` times the payments that history shows for
        it (TM's ADR history stops at 2025-03-31): the statements' figure
        then replaces it, with a note. When `fy_end` is only an upper bound
        (`fy_end_exact` False; see `_fiscal_year_end`) the history is counted
        from `_UNDATED_YEAR_DAYS` earlier, so a June year's own payments are
        not left out and a suspended dividend stays 0. Share counts: Yahoo's
        ADR statements count ADSs, the quoted basis (TM, TSM, BP), but EDGAR
        counts a 20-F filer's ordinary shares, so an EDGAR count for
        financials marked ``_annual_form`` "20-F" gets a WARNING. Returns a
        copy (dataclasses.replace); the caller's MarketData is not mutated.
        """
        source = "yfinance" if from_fallback else "EDGAR"
        years = list(getattr(financials, "fiscal_years", None) or [])
        changes: dict = {}
        price = getattr(market, "price", None)

        shares = getattr(market, "shares_outstanding", None)
        if not (is_num(shares) and shares > 0):
            shares, fy = _latest_positive(
                list(getattr(financials, "diluted_shares", None) or []), years
            )
            if shares is not None:
                changes["shares_outstanding"] = shares
                note = (
                    f"shares outstanding unavailable from Yahoo; using FY{fy} "
                    f"diluted weighted-average shares from {source} ({shares:,.0f})"
                )
                form = str(getattr(financials, "_annual_form", "") or "").upper()
                if not from_fallback and form.startswith("20-F"):
                    note = (
                        f"WARNING: {note}; EDGAR counts a 20-F filer's ordinary shares, so "
                        "for an ADR the market cap, dividend per share and per-share "
                        "values are off by the ADS ratio"
                    )
                notes.append(note)

        mcap = getattr(market, "market_cap", None)
        if not (is_num(mcap) and mcap > 0):
            if shares is not None and is_num(price) and price > 0:
                changes["market_cap"] = price * shares
                notes.append(
                    f"market cap unavailable from Yahoo; set to price x shares "
                    f"({price * shares:,.0f})"
                )
            else:
                notes.append(
                    "WARNING: market cap unavailable (no share count from Yahoo or "
                    f"{source}); the WACC equity weight and market multiples are "
                    "unreliable"
                )

        dps = getattr(market, "dividend_per_share", None)
        if (dps is None or dps == 0.0) and shares is not None:
            div, fy = _latest_positive(
                list(getattr(financials, "dividends_paid", None) or [])[-1:], years[-1:]
            )
            history = getattr(market, "_dividend_history", None)
            if div is not None and dps is None:
                changes["dividend_per_share"] = div / shares
                notes.append(
                    f"dividend per share unavailable from Yahoo; derived from FY{fy} "
                    f"dividends paid / shares ({div / shares:.4f})"
                )
            elif div is not None and isinstance(history, tuple) and history and fy_end:
                span = 365 if fy_end_exact else _UNDATED_YEAR_DAYS
                start = fy_end - datetime.timedelta(days=span + _DIVIDEND_LAG_DAYS)
                shown = sum(a for d, a in history if start < d <= fy_end)
                if div / shares > _STALE_HISTORY_MULTIPLE * shown:
                    changes["dividend_per_share"] = div / shares
                    last = max(d for d, _ in history).isoformat()
                    period = ("that year" if fy_end_exact else
                              "that year and the one before (its end date is not known)")
                    notes.append(
                        f"Yahoo's dividend payment history gives D0 = 0 (last ex-date {last}), "
                        f"but FY{fy} dividends paid / shares ({div / shares:.4f}) is more than "
                        f"{_STALE_HISTORY_MULTIPLE:g}x the {shown:,.4g} that history shows for "
                        f"{period}, so it looks incomplete (Yahoo's ADR histories can stop "
                        f"early); DDM D0 = {div / shares:.4f} from the statements"
                    )

        if not changes:
            return market
        try:
            return dataclasses.replace(market, **changes)
        except TypeError:  # not a dataclass (e.g. a test stub): set in place
            for k, v in changes.items():
                setattr(market, k, v)
            return market

    # ------------------------------------------------------------------ #
    #  Thin delegations to the market client
    # ------------------------------------------------------------------ #
    def get_market_data(self, ticker: str) -> MarketData:
        """Delegate live market-data lookup to the yfinance client."""
        return self.market.get_market_data(ticker)

    def get_peer_comp_rows(self, tickers: list[str]) -> list[CompRow]:
        """Delegate trading-comps row construction to the yfinance client.

        Unresolvable tickers are skipped (not raised on) by ``get_comp_rows``;
        a class-share peer typed with a dot (BF.B) that yields no row is
        retried in the dashed form (see :meth:`_class_share_alias`).
        """
        if not tickers:
            return []
        rows = self.market.get_comp_rows(tickers)
        got = {str(getattr(r, "ticker", "") or "").upper() for r in rows or []}
        retry = []
        for t in tickers:
            sym = t.strip().upper() if isinstance(t, str) else ""
            if sym and sym not in got:
                alias, reachable = self._class_share_lookup(sym)
                if not reachable:
                    # Every further lookup would retry the directory download
                    # (about a minute each during an SEC outage).
                    break
                if alias is not None and alias not in got and alias not in retry:
                    retry.append(alias)
        if retry:
            rows = list(rows or []) + list(self.market.get_comp_rows(retry) or [])
        return rows

    def _class_share_alias(self, symbol: str) -> Optional[str]:
        """``BRK-B`` for ``BRK.B``: the dashed form of a class-share ticker
        typed with a dot, when the SEC directory lists it (Yahoo, like the SEC,
        writes class shares with a dash), else None. Only a one-letter class
        after the dot qualifies, so exchange suffixes such as ``7203.T`` or
        ``SHOP.TO`` are never looked up; ``VOD.L`` is, and is not listed.
        Called only after Yahoo found nothing for the dotted form. Never raises.
        """
        return self._class_share_lookup(symbol)[0]

    def _class_share_lookup(self, symbol: str) -> tuple[Optional[str], bool]:
        """``(alias, reachable)``: the `_class_share_alias` answer, and False
        when the SEC directory could not be read (any error other than the
        EDGAR client's not-listed one), so that callers stop asking."""
        if not isinstance(symbol, str) or not _CLASS_SHARE_DOT.fullmatch(symbol):
            return None, True
        alias = symbol.replace(".", "-")
        try:
            self.edgar.resolve_cik(alias)
        except Exception as exc:  # noqa: BLE001 -- not listed, or the directory is unreachable
            return None, isinstance(exc, DataError) and _SEC_NOT_LISTED in str(exc)
        return alias, True

    def suggest_peers(self, ticker: str) -> list[str]:
        """Delegate best-effort peer suggestion to the yfinance client."""
        try:
            return self.market.suggest_peers(ticker)
        except Exception:  # noqa: BLE001 -- peers are optional; never crash.
            return []

    def discover_peers(self, ticker: str, *, industry_key: Optional[str] = None,
                       limit: Optional[int] = None, name: Optional[str] = None) -> dict:
        """Opt-in peer discovery from Yahoo's industry list, delegated to the
        yfinance client (``YFinanceClient.discover_peers``: same dict, peers
        within a market-weight size band, notes asking for review; ``name``,
        the target's company name, keeps its other share class out). A
        client without it, or any error, gives no peers and a note. Never
        raises."""
        try:
            return self.market.discover_peers(ticker, industry_key=industry_key, limit=limit,
                                              name=name)
        except Exception as exc:  # noqa: BLE001 -- peers are optional; never crash.
            return {"ticker": str(ticker or "").strip().upper(), "industry_key": industry_key,
                    "industry": None, "peers": [],
                    "notes": [f"No peers discovered: peer discovery failed ({exc})."],
                    "source": _PEER_SOURCE}
