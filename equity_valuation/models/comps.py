"""Trading-comparables ("comps") valuation.

Given a target company and a set of peer tickers, this module builds a comps
table of trailing multiples, trims outliers per multiple, computes summary
statistics, and derives an implied share price by applying each peer-median
multiple to the target's own metric.

Valuation methods:

  * EV multiples (ev_ebitda, ev_sales) imply an enterprise value, from which the
    non-equity claims (net debt + minority interest + preferred) are stripped to
    get equity value, then divided by shares. Peer EVs come from the market
    data, whose total debt includes operating-lease liabilities (MCD: 54.6B
    with 14.7B of leases, against 41.4B of borrowings in its filings). When the
    balance sheet carries the target's operating-lease liabilities
    (``operating_lease_liability``) they are claims on the same basis: added
    to the target's EV and stripped in the EV-to-equity bridge, with a note.
    That field is optional: EDGAR filers get it from their operating-lease
    tags; while a provider leaves it None (Yahoo's statements, whose debt
    already includes leases) the bridge carries no lease term
    (``operating_lease_claim``).
  * Equity multiples (pe, pb) apply directly to per-share earnings / book value.
  * Basis. Peer multiples are Yahoo's trailing-twelve-month ones. By default
    (``basis="fy"``) they are applied to the target's latest fiscal-year
    revenue, EBITDA and net income, and a note flags trailing net income or
    EBITDA more than 15% away from the fiscal year's (NVDA: trailing net
    income 192.9B against 120.1B for FY2026, +60%), or a trailing loss
    against a fiscal-year profit, since the P/E- and EV/EBITDA-implied prices
    may then be off. The gap may be growth or one-off items: Yahoo's trailing
    net income includes gains, impairments and tax items below EBITDA and is
    not screened for them (GOOGL: trailing net income +85% against EBITDA
    +15%), so the note says when the net income gap is mostly below EBITDA.
    ``basis="ttm"`` (opt-in, the CLI's ``--comps-basis ttm``) uses Yahoo's
    trailing figures for the target too (``MarketData.ttm_*``) when they run
    past the statements' fiscal year end and are recent, with a note; the
    DCF stays on the fiscal year. Book value (P/B) is the balance sheet's
    under both.
  * PEG is display-only: the target uses historical diluted EPS growth, while
    provider peer PEGs may use forward growth over an unspecified horizon.
    Those figures cannot support a comparable implied P/E or target price.
  * For a bank, insurer, lender/BDC, a company with a consolidated captive
    finance arm or a debt-funded operating lessor
    (``utils.financial_institution_detail``), the EV multiples give no implied
    price: its debt funds its lending or its lease fleet, so it is not a
    financing claim to strip out of EV (JPM: EV/Sales implied $106 against
    $277 from P/E and $200 from P/B). The implied price uses P/E and P/B
    only, with a note; peer EV multiples and the reference-only PEG are still
    shown. Equity REITs keep the EV and equity multiples: their debt finances
    property like an operating company's, and
    EV/EBITDA is the usual REIT multiple (P/E is distorted by property
    depreciation). A mortgage REIT (AGNC, NLY, STWD) is classed as a lender:
    its repo and warehouse borrowing funds a book of loans and mortgage
    securities, so it gets P/E and P/B only.
  * When the target's latest net margin has collapsed below operating profit
    (the FCFE's screen, ``ddm_fcfe.net_margin_collapse``: under a quarter of
    what its EBIT implies after interest and tax, with the EBIT margin
    positive and not a one-off) and is also under a quarter of its prior
    median, the P/E implied price is left out, with a note: a peer
    P/E applied to that EPS prices the impairment or tax charge, not the
    business (BP FY2025: a P/E-implied GBP 0.04 took the comps median from
    9.15 to 6.97). A filer whose provider marks noncontrolling or preferred
    income (BP among them) is screened on a common basis: that income comes
    off the margin EBIT implies (``net_margin_collapse``). The same happens
    in a year whose EBIT margin is itself out of line, which that screen
    skips (``ddm_fcfe.ebit_charge_collapse``). An EBIT margin that is a
    one-off drop, or under a third of its prior median, points to a charge
    inside operating profit, so also inside EBITDA (EBIT + D&A): with the net
    margin under a third of its own prior median, EV/EBITDA is left out too,
    and EV/Sales and P/B are kept (GM FY2025: EBIT margin 1.6% against 6.7%,
    net 1.72% against 5.76%; a P/E-implied 25.48 against 79.91 from P/B had
    put the comps median at 52.70, 'Overvalued -35%'). Next to a one-off rise
    in the EBIT margin the charge is below operating profit: with the net
    margin under a quarter of its prior median only P/E is left out.

A multiple with fewer than ``config.COMPS_MIN_PEER_VALUES`` (3) usable peer
values after trimming gives no implied price, with a note starting with
``FEW_PEER_VALUES_NOTE_PREFIX``: the outlier trim cannot act on one or two
values, so one peer's multiple (or the mean of two) would set the price. Its
peer statistics are still shown.

Every requested peer gets a status in ``CompsResult.peer_status`` ('used',
'not found', 'no usable multiple', 'is target', or 'not fetched' when the
peer-data fetch failed), and the peers left out are named in one note
("Peers not used: XYZNOTREAL (not found), ABC (no usable multiple)."), so a
mistyped ticker is not dropped silently. The status is read from the rows
the provider already returned; no extra lookups are made. A class-share
ticker counts as one symbol in either spelling (BF.B and BF-B), as the data
layer retries a dotted peer in the dashed form.

All monetary inputs are absolute units (not millions); multiples are pure ratios.
The provider is touched ONLY through the DataProvider interface
(`suggest_peers` / `get_peer_comp_rows`). Never crashes on missing data: every
access is guarded and human-readable issues are appended to `CompsResult.notes`.

"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from .. import config
from ..data.base import DataProvider
from ..schemas import CompanyData, CompRow, CompsResult
from ..utils import (
    cagr,
    financial_institution_detail,
    is_num,
    median,
    net_debt_parts,
    safe_div,
    share_scale_break,
    summary_stats,
    trim_outliers,
)
from .ddm_fcfe import NET_MARGIN_COLLAPSE_SHARE, ebit_charge_collapse, net_margin_collapse
from .wacc import tax_gap_years

# EV multiples: the target's net debt is stripped from the implied EV.
EV_MULTIPLES = ("ev_ebitda", "ev_sales")
# Kinds (``utils.financial_institution_detail``) whose debt funds a lending
# book, so the implied price uses P/E and P/B only. "financial" is a flagged
# company of unknown kind. Equity REITs are not listed (see the module notes);
# a mortgage REIT is a "lender".
EQUITY_MULTIPLES_ONLY_KINDS = ("bank", "insurer", "lender", "financial", "captive_finance",
                               "lessor")
# Multiples that give such a company no implied price: the EV multiples, and
# PEG (reference-only for every issuer because growth horizons do not align).
EQUITY_ONLY_DROPPED = EV_MULTIPLES + ("peg",)
# Start of the note that says so; the engine also lists it among the warnings.
EQUITY_MULTIPLES_NOTE_PREFIX = "Equity multiples only"
# Earnings multiples left out when the latest net margin has collapsed (with
# EV/EBITDA when the charge is inside EBIT; see the module notes), and the
# start of the note that says so (the engine also lists it among the warnings).
EARNINGS_MULTIPLES = ("pe", "peg")
EARNINGS_COLLAPSE_NOTE_PREFIX = "P/E-implied price left out"
# How that note names the multiples it leaves out.
_MULTIPLE_NAMES = {"pe": "P/E", "peg": "PEG", "ev_ebitda": "EV/EBITDA",
                   "ev_sales": "EV/Sales", "pb": "P/B"}
# Start of the note naming the multiples with fewer than
# ``config.COMPS_MIN_PEER_VALUES`` usable peer values, which give no implied
# price (the engine also lists it among the warnings).
FEW_PEER_VALUES_NOTE_PREFIX = "Too few peer values"
# Values of ``CompsResult.peer_status``: what comps did with each requested
# peer (see ``run_comps``).
PEER_USED = "used"
PEER_NOT_FOUND = "not found"
PEER_NO_USABLE_MULTIPLE = "no usable multiple"
PEER_IS_TARGET = "is target"
PEER_NOT_FETCHED = "not fetched"
# Start of the note naming the requested peers that were not used (the engine
# also lists it among the warnings).
PEERS_NOT_USED_NOTE_PREFIX = "Peers not used"
# Start of the note recorded when the provider raised while fetching the peer
# rows (the engine quotes that failure in its no-usable-peers warning).
PEER_FETCH_FAILED_NOTE_PREFIX = "Failed to fetch peer comp rows"
# Target metric bases (``run_comps(basis=...)``): the latest fiscal year (the
# default) or Yahoo's trailing twelve months (``MarketData.ttm_*``).
COMPS_BASES = ("fy", "ttm")
# Start of every note about the trailing basis (used, partly used or not
# usable), and of the default basis's note on a large fiscal-year vs trailing
# gap; the engine lists both among the warnings.
COMPS_TTM_NOTE_PREFIX = "Comps on trailing twelve months"
COMPS_GAP_NOTE_PREFIX = "Fiscal-year vs trailing gap"
# The gap note needs trailing net income or EBITDA more than this far from the
# fiscal year's (|TTM / FY - 1|, both positive), or a trailing figure at or
# below zero against a positive fiscal-year one.
TTM_GAP_THRESHOLD = 0.15
# The gap note calls the net income gap "mostly below EBITDA" (non-operating
# or one-off items) when it is more than this many points (as a decimal) away
# from the EBITDA gap, and further from it than the EBITDA gap is from zero
# (GOOGL: +84.7% against +15.3%). Operating leverage alone (EBITDA +15%, net
# income +30%) does not qualify.
TTM_BELOW_EBITDA_POINTS = 0.25
# Trailing figures whose latest quarter ended more than this many days before
# today are stale and not used.
TTM_MAX_AGE_DAYS = 200
# A fiscal year ending in the first this-many days of January takes the prior
# year's label (as the EDGAR client labels a 52/53-week year).
_EARLY_JANUARY_DAYS = 14
# The target metrics the trailing basis replaces: (key, MarketData field,
# name in the notes, the multiple that applies to it).
_TTM_METRICS = (
    ("revenue", "ttm_revenue", "revenue", "ev_sales"),
    ("ebitda", "ttm_ebitda", "EBITDA", "ev_ebitda"),
    ("net_income", "ttm_net_income", "net income", "pe"),
)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _empty_summary() -> dict:
    """The shape of `implied_price_summary` when nothing usable was produced."""
    return {"low": None, "median": None, "high": None}


def _target_shares(company: CompanyData) -> Optional[float]:
    """Best share count for the target: market shares_outstanding, else latest
    diluted weighted-average shares. Returns None if neither is usable."""
    market = getattr(company, "market", None)
    if market is not None:
        so = getattr(market, "shares_outstanding", None)
        if is_num(so) and so > 0:
            return float(so)
    fin = getattr(company, "financials", None)
    if fin is not None:
        try:
            ds = fin.diluted_shares[-1] if fin.diluted_shares else None
        except (IndexError, AttributeError):
            ds = None
        if is_num(ds) and ds > 0:
            return float(ds)
    return None


def operating_lease_claim(bs) -> float:
    """The target's operating-lease liabilities from the balance sheet
    (``operating_lease_liability``) when positive, else 0.0.

    It is an optional ``BalanceSheetSnapshot`` field; 0.0 while a provider
    leaves it None. It holds only leases the target's own total debt leaves
    out (EDGAR's borrowings, not Yahoo's or an IFRS filer's lease-inclusive
    debt), so that it is a claim on the peers' lease-inclusive EV basis, not
    a double count."""
    lease = getattr(bs, "operating_lease_liability", None) if bs is not None else None
    return float(lease) if is_num(lease) and lease > 0 else 0.0


def _latest(series: Optional[list]) -> Optional[float]:
    """Most-recent (last) finite element of an oldest->newest series, else None."""
    if not series:
        return None
    try:
        v = series[-1]
    except (IndexError, TypeError):
        return None
    return float(v) if is_num(v) else None


def _today() -> date:
    """Today's date: the reference for the trailing figures' age (patched in tests)."""
    return date.today()


def normalize_basis(basis) -> str:
    """``basis`` as one of ``COMPS_BASES`` ('fy' or 'ttm', any case; None is
    'fy'). Raises ValueError for anything else."""
    key = "fy" if basis is None else (basis.strip().lower() if isinstance(basis, str) else basis)
    if key not in COMPS_BASES:
        raise ValueError(f"unknown comps basis {basis!r}; expected one of: "
                         f"{', '.join(COMPS_BASES)}")
    return key


def _iso_date(value) -> Optional[date]:
    """A ``date`` from an ISO string ('2026-07-26', a time part ignored), a
    date or a datetime (its date part), else None."""
    if isinstance(value, datetime):  # a date subclass: compare as a date
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


def _fiscal_year_text(fin) -> str:
    """'fiscal year 2026' for the statements' latest year, else 'the latest
    fiscal year'."""
    years = getattr(fin, "fiscal_years", None) if fin is not None else None
    try:
        year = years[-1] if years else None
    except (IndexError, TypeError):
        year = None
    if is_num(year):
        return f"fiscal year {int(year)}"
    return "the latest fiscal year"


def _join(items: list[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _statements_end_before(fin, day: date) -> bool:
    """Whether the statements' latest fiscal year (``fiscal_years[-1]``)
    certainly ends before ``day``, a fiscal year end Yahoo reports: its label
    is below the lowest label a fiscal year ending on ``day`` gets (the
    calendar year of the end, or the year before for a 52/53-week year ending
    in the first ``_EARLY_JANUARY_DAYS`` of January, as the EDGAR client
    labels it; Yahoo's statements use the calendar year). False when the
    label is missing or not positive."""
    years = getattr(fin, "fiscal_years", None) if fin is not None else None
    try:
        year = years[-1] if years else None
    except (IndexError, TypeError):
        year = None
    if not is_num(year) or year <= 0:
        return False
    lowest = day.year - 1 if (day.month == 1 and day.day <= _EARLY_JANUARY_DAYS) else day.year
    return int(year) < lowest


def _trailing_figures(company: CompanyData) -> tuple[Optional[dict], Optional[str]]:
    """Yahoo's trailing-twelve-month figures for the target, when usable.

    Returns ``({"revenue", "ebitda", "net_income": float or None, "as_of":
    date}, None)`` or ``(None, reason)``. Usable means: at least one of
    ``MarketData.ttm_revenue / ttm_ebitda / ttm_net_income`` is a number,
    ``ttm_as_of`` parses as a date, it is after ``fiscal_year_end`` (Yahoo's
    latest fiscal year end, when known) and at most ``TTM_MAX_AGE_DAYS``
    before ``_today()``. The reason says which test failed (not available,
    a different reporting currency, no date, the fiscal year end, stale).

    A latest quarter that is Yahoo's fiscal year end makes the trailing
    figures that fiscal year, so they are not used, unless the statements'
    latest fiscal year ends before it (``_statements_end_before``): between
    a company's year-end results and its annual report reaching the
    statements (banks report in mid-January and file weeks later), Yahoo's
    trailing figures are the newer year and are used. The freshness test
    reads Yahoo's dates and the statements' fiscal-year label only (the
    statements carry no period-end date), so statements newer than Yahoo's
    trailing figures are not detected; the age limit bounds that case.
    """
    market = getattr(company, "market", None)
    figures: dict = {}
    for key, attr, _, _ in _TTM_METRICS:
        v = getattr(market, attr, None) if market is not None else None
        figures[key] = float(v) if is_num(v) else None
    if all(v is None for v in figures.values()):
        ttm_ccy = getattr(market, "ttm_currency", None) if market is not None else None
        quote = getattr(market, "currency", None) if market is not None else None
        if (isinstance(ttm_ccy, str) and isinstance(quote, str) and ttm_ccy.strip()
                and ttm_ccy.strip().upper() != quote.strip().upper()):
            return None, (f"Yahoo reports the target's trailing figures in {ttm_ccy.strip()}, "
                          f"a different reporting currency from the {quote.strip()} quote, and "
                          "they are not converted")
        return None, "Yahoo's trailing-twelve-month figures are not available for the target"
    as_of = _iso_date(getattr(market, "ttm_as_of", None))
    if as_of is None:
        return None, "Yahoo gives no date for the target's trailing figures"
    fy_end = _iso_date(getattr(market, "fiscal_year_end", None))
    newer_year = (fy_end is not None and as_of == fy_end
                  and _statements_end_before(getattr(company, "financials", None), fy_end))
    if fy_end is not None and as_of <= fy_end and not newer_year:
        if as_of == fy_end:
            where, so = "is", "its trailing twelve months are that fiscal year"
        else:
            where, so = "ends before", "its trailing figures are older than the fiscal year"
        return None, (f"the latest quarter Yahoo reports ({as_of.isoformat()}) {where} the "
                      f"fiscal year end ({fy_end.isoformat()}), so {so}")
    age = (_today() - as_of).days
    if age > TTM_MAX_AGE_DAYS:
        return None, (f"Yahoo's trailing figures are stale: their latest quarter ended "
                      f"{as_of.isoformat()}, {age} days ago (over the {TTM_MAX_AGE_DAYS}-day "
                      "limit)")
    figures["as_of"] = as_of
    return figures, None


def _fiscal_metrics(fin) -> dict:
    """The latest fiscal year's revenue, EBITDA and net income (None when missing)."""
    return {key: (_latest(getattr(fin, key, None)) if fin is not None else None)
            for key, _, _, _ in _TTM_METRICS}


def _non_positive_text(key: str, value: float) -> str:
    """'a loss' (net income below zero), 'negative' or 'zero'."""
    if value == 0:
        return "zero"
    return "a loss" if key == "net_income" else "negative"


def _target_metrics(
    company: CompanyData, basis: str, fy_metrics: dict, notes: list[str],
    equity_only: bool = False,
) -> tuple[dict, Optional[dict]]:
    """The target's revenue, EBITDA and net income for the comps, and the
    usable trailing figures (``_trailing_figures``; None when not usable).

    Basis 'fy': the latest fiscal year's (``fy_metrics``). Basis 'ttm' with
    usable trailing figures: Yahoo's trailing ones, each falling back to the
    fiscal year's when Yahoo has none, with a note naming the date and the
    metrics (``COMPS_TTM_NOTE_PREFIX + " to YYYY-MM-DD (Yahoo) ..."``; the
    fallbacks are named once the implied prices are known, see
    ``_trailing_fallback_note``). The note marks a metric whose multiple
    gives no implied price: '(display only)' for an EV metric of an
    ``equity_only`` target (a lender, whose EV multiples are only shown), and
    '(a loss, so P/E gives no price)' or '(negative, ...)' for a trailing
    figure at or below zero. Basis 'ttm' without usable trailing figures: the
    fiscal year's, with a note saying why. Book value (P/B) is always the
    balance sheet's."""
    fin = getattr(company, "financials", None)
    metrics = dict(fy_metrics)
    ttm, why_not = _trailing_figures(company)
    if basis != "ttm":
        return metrics, ttm
    fy_text = _fiscal_year_text(fin)
    if ttm is None:
        notes.append(f"{COMPS_TTM_NOTE_PREFIX} not used: {why_not}; comps use the target's "
                     f"{fy_text} figures.")
        return metrics, None
    used = []
    for key, _, name, multiple in _TTM_METRICS:
        value = ttm[key]
        if value is None:
            continue
        metrics[key] = value
        if equity_only and multiple in EQUITY_ONLY_DROPPED:
            used.append(f"{name} (display only)")
        elif value <= 0:
            used.append(f"{name} ({_non_positive_text(key, value)}, so "
                        f"{_MULTIPLE_NAMES[multiple]} gives no price)")
        else:
            used.append(name)
    notes.append(
        f"{COMPS_TTM_NOTE_PREFIX} to {ttm['as_of'].isoformat()} (Yahoo) for the target's "
        f"{_join(used)}, matching peers' trailing multiples; the DCF stays on {fy_text}.")
    return metrics, ttm


def _trailing_fallback_note(fin, ttm: dict, implied: dict) -> Optional[str]:
    """Under the trailing basis, the note naming each metric Yahoo gives no
    trailing figure for whose multiple still gives a positive implied price,
    on the fiscal year's figure instead (JPM: no trailing EBITDA). A multiple
    that gives no price anyway (a lender's EV/EBITDA) is not named. None when
    there is nothing to name."""
    missing = [(name, _MULTIPLE_NAMES[multiple]) for key, _, name, multiple in _TTM_METRICS
               if ttm.get(key) is None and is_num(implied.get(multiple))
               and implied[multiple] > 0]
    if not missing:
        return None
    names = _join([name for name, _ in missing])
    multiples = _join([multiple for _, multiple in missing])
    verb, figure = ("uses", "figure") if len(missing) == 1 else ("use", "figures")
    return (f"{COMPS_TTM_NOTE_PREFIX}: Yahoo gives no trailing {names} for the target, so "
            f"{multiples} {verb} the {_fiscal_year_text(fin)} {figure}.")


def _trailing_gap_note(fin, fy_metrics: dict, ttm: Optional[dict],
                       implied: dict, equity_only: bool = False) -> Optional[str]:
    """The default (fiscal-year) basis's note when Yahoo's usable trailing
    net income or EBITDA differs from the fiscal year's by more than
    ``TTM_GAP_THRESHOLD`` (both positive), or is at or below zero against a
    positive fiscal-year figure (a trailing loss), for the multiples that
    still give an implied price (P/E, EV/EBITDA). Peers' multiples are
    trailing, so a higher trailing figure means the implied price may be
    understated, a lower one (or a loss) overstated.

    The note does not call the gap timing: Yahoo's trailing net income
    includes gains, impairments and tax items below EBITDA, and no one-off
    screen reads the trailing figures (the collapse screens read the fiscal
    year). It says the gap may reflect the business or one-off items, or,
    when the net income gap is more than ``TTM_BELOW_EBITDA_POINTS`` away
    from the EBITDA gap and further from it than the EBITDA gap is from zero
    (not for an ``equity_only`` target, whose EBITDA means little), that it
    is mostly below EBITDA. It names the opt-in trailing basis without
    recommending it. None otherwise (no usable trailing figures, gaps within
    the threshold, or no positive implied price left for them)."""
    if ttm is None:
        return None
    # (key, name, P/E or EV/EBITDA name, trailing, fiscal-year, TTM / FY - 1)
    gaps = []
    for key, name, multiple in (("net_income", "net income", "pe"),
                                ("ebitda", "EBITDA", "ev_ebitda")):
        t, f = ttm.get(key), fy_metrics.get(key)
        if not (is_num(t) and is_num(f) and f > 0):
            continue
        price = implied.get(multiple)
        if not (is_num(price) and price > 0):
            continue  # left out (a lender's EV/EBITDA, a collapsed margin's P/E)
        gap = t / f - 1.0
        if t <= 0 or abs(gap) > TTM_GAP_THRESHOLD:
            gaps.append((key, name, _MULTIPLE_NAMES[multiple], t, gap))
    if not gaps:
        return None
    fy_text = _fiscal_year_text(fin)
    numeric = [g for g in gaps if g[3] > 0]
    losses = [g for g in gaps if g[3] <= 0]
    parts = []
    if numeric:
        moves = [f"{name} is {gap:+.1%}" if i == 0 else f"{name} {gap:+.1%}"
                 for i, (_, name, _, _, gap) in enumerate(numeric)]
        parts.append(f"{_join(moves)} vs {fy_text}")
    if losses:
        moves = [f"{name} is {_non_positive_text(key, t)}" for key, name, _, t, _ in losses]
        was = [("a profit" if key == "net_income" else f"positive {name}")
               for key, name, _, _, _ in losses]
        parts.append(f"{_join(moves)}, against {_join(was)} in "
                     f"{'that year' if numeric else fy_text}")
    cause = "the gap may reflect the business or one-off items (gains, impairments, tax)"
    ni_gap = next((gap for key, _, _, _, gap in gaps if key == "net_income"), None)
    t_eb, f_eb = ttm.get("ebitda"), fy_metrics.get("ebitda")
    if (ni_gap is not None and not equity_only
            and is_num(t_eb) and is_num(f_eb) and t_eb > 0 and f_eb > 0):
        eb_gap = t_eb / f_eb - 1.0
        below = ni_gap - eb_gap
        if abs(below) > TTM_BELOW_EBITDA_POINTS and abs(below) > abs(eb_gap):
            shown = any(key == "ebitda" for key, _, _, _, _ in numeric)
            cause = ("the net income gap is mostly below EBITDA"
                     + ("" if shown else f" (trailing EBITDA {eb_gap:+.1%})")
                     + ", so it may be non-operating or one-off items (gains, impairments, "
                     "tax)")
    effects = []
    for word, up in (("understated", True), ("overstated", False)):
        names = [m for _, _, m, t, gap in gaps if (t > 0 and gap > 0) == up]
        if names:
            label = (f"the {names[0]}-implied price" if len(names) == 1
                     else f"the {'- and '.join(names)}-implied prices")
            effects.append(f"{label} may be {word}")
    return (f"{COMPS_GAP_NOTE_PREFIX}: trailing-twelve-month {' and '.join(parts)} (Yahoo, "
            f"to {ttm['as_of'].isoformat()}); {cause}; comps apply peers' trailing multiples "
            f"to fiscal-year figures, so {' and '.join(effects)} (an opt-in trailing basis, "
            "--comps-basis ttm in the CLI, uses Yahoo's trailing figures, which are not "
            "screened for one-off items).")


def _eps_cagr(fin) -> Optional[float]:
    """Diluted EPS CAGR across the available history (oldest->newest).

    Returns None if fewer than two positive endpoints exist (a sign change makes
    growth meaningless for a PEG), or if the share history changes scale by a
    unit factor (``utils.share_scale_break``: MCD's FY2021-25 counts tagged in
    millions gave 668.6%/yr and a PEG of 0.03). The result is a decimal
    (0.12 == 12%).
    """
    if fin is None:
        return None
    ni = getattr(fin, "net_income", None)
    shares = getattr(fin, "diluted_shares", None) or []
    if not ni or len(ni) < 2:
        return None
    if len(shares) != len(ni) or not all(is_num(s) and s > 0 for s in (shares[0], shares[-1])):
        return None
    if share_scale_break(shares) is not None:
        return None
    first, last = safe_div(ni[0], shares[0]), safe_div(ni[-1], shares[-1])
    periods = len(ni) - 1
    years = getattr(fin, "fiscal_years", None) or []
    if (len(years) == len(ni) and is_num(years[0]) and is_num(years[-1])
            and years[-1] > years[0]):
        periods = years[-1] - years[0]
    return cagr(first, last, periods)


# --------------------------------------------------------------------------- #
#  Target row construction
# --------------------------------------------------------------------------- #
def _build_target_row(
    company: CompanyData,
    current_price: Optional[float],
    shares: Optional[float],
    net_debt: float,
    minority: float,
    preferred: float,
    notes: list[str],
    lease: float = 0.0,
    metrics: Optional[dict] = None,
) -> CompRow:
    """Assemble the target's own CompRow for display.

    enterprise_value = market_cap + net_debt + minority + preferred + lease
    (``lease``: operating-lease liabilities, on the peers' EV basis).
    Trailing multiples are computed from the latest fundamentals; any that cannot
    be derived (missing / non-positive denominator) are left as None.
    ``metrics`` ({"revenue", "ebitda", "net_income"}; see ``_target_metrics``)
    replaces the latest fiscal year's figures, e.g. with Yahoo's trailing
    twelve months; None uses the fiscal year's.
    """
    market = getattr(company, "market", None)
    fin = getattr(company, "financials", None)

    ticker = getattr(company, "ticker", "") or ""
    name = getattr(company, "name", "") or ticker

    # Market cap: prefer the provider's market cap, else price * shares.
    market_cap: Optional[float] = None
    if market is not None:
        mc = getattr(market, "market_cap", None)
        if is_num(mc) and mc > 0:
            market_cap = float(mc)
    if market_cap is None and is_num(current_price) and shares:
        market_cap = float(current_price) * float(shares)

    # Enterprise value = equity value + net non-equity claims.
    enterprise_value: Optional[float] = None
    if market_cap is not None:
        enterprise_value = market_cap + net_debt + minority + preferred + lease

    # Latest fundamental metrics for the trailing multiples.
    if metrics is not None:
        ebitda_latest = metrics.get("ebitda")
        revenue_latest = metrics.get("revenue")
        ni_latest = metrics.get("net_income")
    else:
        ebitda_latest = _latest(getattr(fin, "ebitda", None)) if fin is not None else None
        revenue_latest = _latest(getattr(fin, "revenue", None)) if fin is not None else None
        ni_latest = _latest(getattr(fin, "net_income", None)) if fin is not None else None
    total_equity = None
    bs = getattr(company, "balance_sheet", None)
    if bs is not None:
        te = getattr(bs, "total_equity", None)
        total_equity = float(te) if is_num(te) else None

    # EV-based multiples (only meaningful for a positive EV and denominator).
    # A net-cash target with EV <= 0 must show None/NM, not a negative multiple.
    ev_ebitda = None
    ev_sales = None
    if enterprise_value is not None and enterprise_value > 0:
        if is_num(ebitda_latest) and ebitda_latest > 0:
            ev_ebitda = safe_div(enterprise_value, ebitda_latest)
        if is_num(revenue_latest) and revenue_latest > 0:
            ev_sales = safe_div(enterprise_value, revenue_latest)

    # Equity multiples.
    pe = None
    if is_num(current_price) and is_num(ni_latest) and ni_latest > 0 and shares:
        eps = safe_div(ni_latest, shares)
        pe = safe_div(current_price, eps)

    pb = None
    if market_cap is not None and is_num(total_equity) and total_equity > 0:
        pb = safe_div(market_cap, total_equity)

    # PEG = P/E divided by the earnings growth expressed in percentage points.
    peg = None
    growth = _eps_cagr(fin)
    if pe is not None and is_num(growth) and growth > 0:
        peg = safe_div(pe, growth * 100.0)

    if market_cap is None:
        notes.append("Target market cap unavailable; target multiples are limited.")

    return CompRow(
        ticker=ticker,
        name=name,
        market_cap=market_cap,
        enterprise_value=enterprise_value,
        ev_ebitda=ev_ebitda,
        ev_sales=ev_sales,
        pe=pe,
        pb=pb,
        peg=peg,
        currency=getattr(market, "currency", None),
    )


# --------------------------------------------------------------------------- #
#  Peer collection
# --------------------------------------------------------------------------- #
def _row_has_usable_multiple(row: CompRow) -> bool:
    """True if a peer row carries at least one finite, positive multiple."""
    for m in config.COMPS_MULTIPLES:
        v = getattr(row, m, None)
        if is_num(v) and v > 0:
            return True
    return False


def _symbol_key(ticker) -> str:
    """``ticker`` upper-cased and stripped, with a class-share dot written as
    a dash: BF.B and BF-B are one listing (the data layer retries a dotted
    peer in the dashed form), so the target, repeats and a peer's status
    are matched in either spelling."""
    return str(ticker or "").strip().upper().replace(".", "-")


def _collect_peer_rows(
    provider: DataProvider,
    peer_tickers: list[str],
    target_ticker: str,
    notes: list[str],
    rejected: Optional[set] = None,
) -> list[CompRow]:
    """Fetch peer comp rows via the provider, dropping the target and any row with
    no usable multiples. Never raises -- a provider failure yields an empty list
    plus a note.

    ``rejected``, when given, receives the upper-case ticker of each row that
    came back without a usable multiple (``_row_has_usable_multiple``), for
    ``CompsResult.peer_status``; callers that omit it are unaffected."""
    try:
        raw_rows = provider.get_peer_comp_rows(list(peer_tickers))
    except Exception as exc:  # pragma: no cover - defensive against provider bugs
        notes.append(f"{PEER_FETCH_FAILED_NOTE_PREFIX}: {exc}")
        return []

    if not raw_rows:
        return []

    target_key = _symbol_key(target_ticker)
    kept: list[CompRow] = []
    seen: set[str] = set()
    for row in raw_rows:
        if row is None:
            continue
        rt = (getattr(row, "ticker", "") or "").strip().upper()
        key = _symbol_key(rt)
        if key and key == target_key:
            # Exclude the target if the provider returned it among the peers.
            continue
        if not _row_has_usable_multiple(row):
            if rejected is not None and rt:
                rejected.add(rt)
            continue
        if key and key in seen:
            continue  # one issuer must have one vote in the peer distribution
        seen.add(key)
        kept.append(row)
    return kept


def _peer_status(
    requested: list[str],
    target_ticker: str,
    kept: list[CompRow],
    rejected: set,
    fetched: bool = True,
) -> dict:
    """``{TICKER: status}`` for the requested peers (upper-case, request
    order), read only from what the provider returned (no extra lookups):
    'is target' for the target itself; 'not fetched' for the rest when the
    fetch failed (``fetched`` False); else 'used' when a kept row matches,
    'no usable multiple' when only a row without a positive multiple came
    back, and 'not found' when no row came back. A kept row under a ticker
    that was not requested (a provider that renames its rows) is listed as
    'used' after them, so every peer row has a status."""
    target_key = _symbol_key(target_ticker)
    used = {_symbol_key(getattr(r, "ticker", "")) for r in kept}
    bad = {_symbol_key(t) for t in rejected}
    status: dict = {}
    for t in requested:
        key = _symbol_key(t)
        if not key or t in status:
            continue
        if key == target_key:
            status[t] = PEER_IS_TARGET
        elif not fetched:
            status[t] = PEER_NOT_FETCHED
        elif key in used:
            status[t] = PEER_USED
        elif key in bad:
            status[t] = PEER_NO_USABLE_MULTIPLE
        else:
            status[t] = PEER_NOT_FOUND
    listed = {_symbol_key(t) for t in status}
    for row in kept:
        rt = (getattr(row, "ticker", "") or "").strip().upper()
        if rt and _symbol_key(rt) not in listed:
            status[rt] = PEER_USED
            listed.add(_symbol_key(rt))
    return status


def _peers_not_used_note(peer_status: dict) -> Optional[str]:
    """``"Peers not used: XYZNOTREAL (not found), ABC (no usable multiple)."``
    for the peers in ``peer_status`` that were not used, else None. Peers that
    were never fetched are left to the fetch-failure note."""
    left_out = [f"{t} ({s})" for t, s in peer_status.items()
                if s not in (PEER_USED, PEER_NOT_FETCHED)]
    if not left_out:
        return None
    return f"{PEERS_NOT_USED_NOTE_PREFIX}: {', '.join(left_out)}."


# --------------------------------------------------------------------------- #
#  Implied-price math
# --------------------------------------------------------------------------- #
def _implied_from_multiple(
    multiple: str,
    med: Optional[float],
    *,
    shares: Optional[float],
    net_debt: float,
    minority: float,
    preferred: float,
    ebitda_latest: Optional[float],
    revenue_latest: Optional[float],
    ni_latest: Optional[float],
    total_equity: Optional[float],
    lease: float = 0.0,
) -> Optional[float]:
    """Apply a peer-median multiple to the target's own metric -> implied price.

    EV multiples back out equity value (EV - net debt - minority - preferred -
    operating-lease liabilities) before dividing by shares; equity multiples
    apply per-share. ``ebitda_latest``, ``revenue_latest`` and ``ni_latest``
    are the latest fiscal year's, or Yahoo's trailing twelve months' under the
    trailing basis (``run_comps(basis="ttm")``). Returns None when any
    required input is missing / non-positive.
    """
    if med is None or not is_num(med) or med <= 0:
        return None

    if multiple == "ev_ebitda":
        if not (is_num(ebitda_latest) and ebitda_latest > 0 and shares):
            return None
        ev_star = med * ebitda_latest
        equity_star = ev_star - net_debt - minority - preferred - lease
        return safe_div(equity_star, shares)

    if multiple == "ev_sales":
        if not (is_num(revenue_latest) and revenue_latest > 0 and shares):
            return None
        ev_star = med * revenue_latest
        equity_star = ev_star - net_debt - minority - preferred - lease
        return safe_div(equity_star, shares)

    if multiple == "pe":
        if not (is_num(ni_latest) and ni_latest > 0 and shares):
            return None
        eps = safe_div(ni_latest, shares)
        if eps is None:
            return None
        return med * eps

    if multiple == "pb":
        if not (is_num(total_equity) and total_equity > 0 and shares):
            return None
        bvps = safe_div(total_equity, shares)
        if bvps is None:
            return None
        return med * bvps

    if multiple == "peg":
        return None  # provider growth horizons are not aligned with the target

    return None


# --------------------------------------------------------------------------- #
#  Public entry point
# --------------------------------------------------------------------------- #
def run_comps(
    company: CompanyData,
    provider: DataProvider,
    peers: Optional[list[str]],
    current_price: float,
    tax_rate: Optional[float] = None,
    basis: str = "fy",
) -> CompsResult:
    """Run a trading-comps valuation for `company`.

    Parameters
    ----------
    company : CompanyData
        The target. Money fields are absolute units; series oldest->newest.
    provider : DataProvider
        Used ONLY via `suggest_peers` / `get_peer_comp_rows`.
    peers : list[str] | None
        Explicit peer tickers. If None/empty, the provider is asked to suggest
        peers; if that is still empty, an empty CompsResult (with a note) returns.
        The target itself (in either class-share spelling) is never a peer.
    current_price : float
        Latest market price per share for the target.
    tax_rate : float | None
        An explicit tax rate (the CLI's ``--tax``) for the net-margin collapse
        screen, as the FCFE uses; else the effective rate.
    basis : str
        The target's revenue, EBITDA and net income behind its multiples and
        implied prices (the CLI's ``--comps-basis``). 'fy' (the default): the
        latest fiscal year's, so peers' trailing multiples meet fiscal-year
        figures; when Yahoo's usable trailing net income or EBITDA differs
        from the fiscal year's by more than ``TTM_GAP_THRESHOLD``, or is a
        loss against a fiscal-year profit, a note starting with
        ``COMPS_GAP_NOTE_PREFIX`` gives the gap, which may be growth or
        one-off items (``_trailing_gap_note``). 'ttm': Yahoo's trailing twelve
        months (``MarketData.ttm_*``, not screened for one-off items) when
        usable (see ``_trailing_figures``: dated after the fiscal year end
        and at most ``TTM_MAX_AGE_DAYS`` old), each metric falling back to
        the fiscal year's when Yahoo has none; otherwise the fiscal year's,
        with the reason. A trailing loss gives no P/E price, as on the fiscal
        year, and the note says so. Those notes start with
        ``COMPS_TTM_NOTE_PREFIX``. Either way
        P/B uses the balance sheet's book value, the DCF and the other models
        stay on the fiscal year, and the net-margin and EBIT collapse screens
        that drop P/E (and EV/EBITDA) read the fiscal-year statements. Any
        other value raises ValueError.

    Returns
    -------
    CompsResult
        Always returned -- never raises on missing data. ``peer_status`` maps
        each requested peer (upper-case, request order) to 'used', 'not found'
        (the provider returned no row: an unknown or mistyped ticker), 'no
        usable multiple' (a row without any positive multiple), 'is target',
        or 'not fetched' (the provider raised, so nothing was checked); it is
        empty when no peers were requested. The peers not used are named in
        one note starting with ``PEERS_NOT_USED_NOTE_PREFIX`` (none when every
        peer was used; peers that were not fetched are left to the
        fetch-failure note). Statistics and trimming use the 'used' rows only,
        as before.

    Raises
    ------
    ValueError
        For a ``basis`` that is not one of ``COMPS_BASES``.
    """
    basis = normalize_basis(basis)
    notes: list[str] = []

    ticker = getattr(company, "ticker", "") or ""

    # --- Non-equity claims (used in both target EV and implied EV->equity) ----- #
    bs = getattr(company, "balance_sheet", None)
    net_debt = 0.0
    minority = 0.0
    preferred = 0.0
    if bs is not None:
        # Component-wise, so a missing cash figure does not discard known debt.
        net_debt, nd_notes = net_debt_parts(bs)
        notes.extend(nd_notes)
        mi = getattr(bs, "minority_interest", 0.0)
        minority = float(mi) if is_num(mi) else 0.0
        pe_eq = getattr(bs, "preferred_equity", 0.0)
        preferred = float(pe_eq) if is_num(pe_eq) else 0.0
    else:
        notes.append("Balance sheet unavailable; net debt / minority / preferred treated as 0.")
    lease = operating_lease_claim(bs)

    shares = _target_shares(company)
    if shares is None:
        notes.append("Share count unavailable; implied prices cannot be computed.")

    # --- The target's revenue, EBITDA and net income (fiscal year or TTM) ------ #
    fin = getattr(company, "financials", None)
    fy_metrics = _fiscal_metrics(fin)
    # A lender's EV multiples give no implied price (see below); its trailing
    # revenue and EBITDA only change the displayed multiples.
    flag = financial_institution_detail(company)
    equity_only = flag is not None and flag[1] in EQUITY_MULTIPLES_ONLY_KINDS
    metrics, ttm = _target_metrics(company, basis, fy_metrics, notes, equity_only)

    # --- Build the target's own display row ------------------------------------ #
    target_row = _build_target_row(
        company,
        current_price,
        shares,
        net_debt,
        minority,
        preferred,
        notes,
        lease,
        metrics,
    )

    # --- Resolve peer tickers --------------------------------------------------- #
    peer_tickers: list[str] = []
    if peers:
        peer_tickers = [t for t in peers if t]
    else:
        try:
            suggested = provider.suggest_peers(ticker)
            peer_tickers = [t for t in (suggested or []) if t]
        except Exception as exc:  # defensive: suggest_peers should never blow up the run
            notes.append(f"Peer suggestion failed: {exc}")
            peer_tickers = []

    # Drop the target itself from the requested set (case-insensitive, either
    # class-share spelling); ``peer_status`` records it as 'is target'.
    requested = list(dict.fromkeys(t.strip().upper() for t in peer_tickers if t.strip()))
    target_key = _symbol_key(ticker)
    peer_tickers = [t for t in requested if _symbol_key(t) != target_key]

    if not peer_tickers:
        peer_status = _peer_status(requested, ticker, [], set())
        not_used = _peers_not_used_note(peer_status)
        if not_used:
            notes.append(not_used)
        notes.append("No peer tickers available; comps could not be computed.")
        return CompsResult(
            target=target_row,
            peers=[],
            stats={},
            implied={},
            implied_price_summary=_empty_summary(),
            notes=notes,
            peer_status=peer_status,
        )

    # --- Fetch and clean peer rows --------------------------------------------- #
    rejected: set = set()
    before = len(notes)
    peer_rows = _collect_peer_rows(provider, peer_tickers, ticker, notes, rejected)
    fetched = not any(str(n).startswith(PEER_FETCH_FAILED_NOTE_PREFIX) for n in notes[before:])
    peer_status = _peer_status(requested, ticker, peer_rows, rejected, fetched)
    not_used = _peers_not_used_note(peer_status)
    if not_used:
        notes.append(not_used)
    if not peer_rows:
        notes.append("No usable peers returned by the data provider; comps could not be computed.")
        return CompsResult(
            target=target_row,
            peers=[],
            stats={},
            implied={},
            implied_price_summary=_empty_summary(),
            notes=notes,
            peer_status=peer_status,
        )

    # --- Per-multiple trimming + summary statistics ---------------------------- #
    stats: dict = {}
    medians: dict = {}
    for m in config.COMPS_MULTIPLES:
        raw_vals = [getattr(row, m, None) for row in peer_rows]
        trimmed = trim_outliers(raw_vals, config.COMPS_OUTLIER_FACTOR)
        stats[m] = summary_stats(trimmed)
        medians[m] = median(trimmed)  # peer median used for implied price
        if not trimmed:
            notes.append(f"No usable peer values for {_MULTIPLE_NAMES.get(m, m)} after outlier "
                         "trimming.")

    # --- Target metrics needed for implied prices ------------------------------ #
    # The basis's figures (``_target_metrics``); book value is the balance sheet's.
    ebitda_latest = metrics.get("ebitda")
    revenue_latest = metrics.get("revenue")
    ni_latest = metrics.get("net_income")
    total_equity = None
    if bs is not None:
        te = getattr(bs, "total_equity", None)
        total_equity = float(te) if is_num(te) else None

    # --- Implied price per multiple -------------------------------------------- #
    implied: dict = {}
    for m in config.COMPS_MULTIPLES:
        implied[m] = _implied_from_multiple(
            m,
            medians.get(m),
            shares=shares,
            net_debt=net_debt,
            minority=minority,
            preferred=preferred,
            ebitda_latest=ebitda_latest,
            revenue_latest=revenue_latest,
            ni_latest=ni_latest,
            total_equity=total_equity,
            lease=lease,
        )

    # --- A multiple from too few peers gives no implied price ----------------- #
    # Below ``COMPS_MIN_PEER_VALUES`` the outlier trim cannot act and the
    # "median" is one peer's multiple or the mean of two: GM's discovered
    # peers gave a P/E from TSLA alone (about 430x) and a P/E-implied price
    # over 20 times GM's share price. A lender's EV multiples give no price
    # anyway (below), so they are not named here.
    thin = [(m, stats[m]["n"]) for m in config.COMPS_MULTIPLES
            if implied.get(m) is not None
            and 0 < stats[m]["n"] < config.COMPS_MIN_PEER_VALUES
            and not (equity_only and m in EQUITY_ONLY_DROPPED)]
    if thin:
        for m, _n in thin:
            implied[m] = None
        counts = [f"{_MULTIPLE_NAMES.get(m, m)} has {n} usable value{'s' if n != 1 else ''}"
                  for m, n in thin]
        which = "it gives" if len(thin) == 1 else "they give"
        notes.append(
            f"{FEW_PEER_VALUES_NOTE_PREFIX} (at least {config.COMPS_MIN_PEER_VALUES} "
            f"needed): {_join(counts)}, so {which} no implied price: a multiple from fewer "
            f"than {config.COMPS_MIN_PEER_VALUES} peers is their valuation, not a peer "
            "median.")
    # Not for an equity-only target: its EV multiples give no price (below),
    # so a note on how its EV is bridged would contradict that one.
    if lease > 0 and not equity_only and any(implied.get(m) is not None for m in EV_MULTIPLES):
        currency = getattr(getattr(company, "market", None), "currency", None) or ""
        notes.append(
            "EV multiples: the peers' EVs include operating-lease liabilities (in the market "
            f"data's total debt), so the target's {lease / 1e9:,.2f}B {currency}".rstrip()
            + " of operating-lease liabilities is added to its EV and subtracted in the "
            "EV-to-equity bridge, on the same basis.")
    if is_num(target_row.peg) or is_num(medians.get("peg")):
        notes.append(
            "PEG is shown for reference only and gives no implied price: the target uses "
            "historical diluted EPS growth, while provider peer PEGs may use forward "
            "growth over an unspecified horizon. Their growth bases are not comparable."
        )

    # --- A lender's EV multiples give no implied price ------------------------ #
    # Its debt (deposits, the borrowing behind a loan book or a finance arm's
    # leases) funds its operations, so stripping it from a peer-multiple EV
    # does not isolate equity (JPM: EV/Sales $106 vs P/E $277 and P/B $200).
    if equity_only:
        for m in EQUITY_ONLY_DROPPED:
            implied[m] = None
        funds = {"captive_finance": "its finance arm's loans and leases",
                 "lessor": "the fleet it leases out"}.get(flag[1], "its lending and investing")
        notes.append(
            f"{EQUITY_MULTIPLES_NOTE_PREFIX} ({flag[0]}): EV/EBITDA and EV/Sales give no "
            f"implied price, because most of the target's debt funds {funds} and is not a "
            "financing claim to strip out of enterprise value; the implied price uses P/E "
            "and P/B (PEG, whose growth horizon is not comparable, and the peer EV "
            "multiples are shown for reference only).")

    # --- A collapsed latest net margin gives no earnings-based price ---------- #
    # The FCFE's screen (net margin under a quarter of what EBIT implies, EBIT
    # in line), plus the net margin under a quarter of its own prior median:
    # a peer P/E on that EPS prices the charge, not the business (BP FY2025).
    # Or the EBIT margin is itself out of line and the net margin is under a
    # third (next to an EBIT gain, a quarter) of its prior median
    # (``ebit_charge_collapse``): a charge inside EBIT (GM FY2025) also sits
    # in EBITDA, which every provider builds as EBIT + D&A, so EV/EBITDA goes
    # too; next to a one-off EBIT gain the charge is below EBIT, and only the
    # EPS multiples go. Both screens read the fiscal-year statements under
    # either basis: Yahoo's trailing figures give one year, not the history
    # the prior medians need.
    collapse = (net_margin_collapse(fin, tax_rate, tax_gap_years(company))
                if fin is not None else None)
    prior_med = collapse.get("prior_median") if collapse else None
    reason = None
    to_drop = EARNINGS_MULTIPLES
    if (collapse is not None and is_num(prior_med) and prior_med > 0
            and collapse["latest"] < NET_MARGIN_COLLAPSE_SHARE * prior_med):
        after = ("interest, tax and noncontrolling/preferred income"
                 if collapse.get("noncommon", 0.0) > 0 else "interest and tax")
        reason = (
            f"the latest net margin {collapse['latest']:.2%} is under a quarter of both its "
            f"prior median {prior_med:.2%} and the {collapse['implied']:.1%} its EBIT margin "
            f"{collapse['ebit_margin']:.1%} implies after {after} (items below "
            "operating profit such as impairments, or an abnormal tax charge)")
    else:
        charge = ebit_charge_collapse(fin) if fin is not None else None
        if charge is not None:
            ebit_m, ebit_prior = charge["ebit_margin"], charge["ebit_prior_median"]
            inside = (" (likely a charge inside operating profit, such as restructuring, "
                      "impairments or a write-down)")
            if charge["ebit_move"] == "gain":
                ebit_text = (
                    f"the latest EBIT margin {ebit_m:.1%} is a one-off rise above its prior "
                    f"median {ebit_prior:.1%} (likely a charge below operating profit, such as "
                    "impairments or an abnormal tax charge, next to a one-off gain within it)")
            elif charge["ebit_move"] == "drop":
                ebit_text = (f"the latest EBIT margin {ebit_m:.1%} is a one-off drop from its "
                             f"prior median {ebit_prior:.1%}{inside}")
            else:  # "collapse": a thin margin, shown to two decimals as in the DCF
                ebit_text = (f"the latest EBIT margin {ebit_m:.2%} is under a third of its "
                             f"prior median {ebit_prior:.2%}{inside}")
            if charge["ebit_move"] != "gain":
                to_drop = EARNINGS_MULTIPLES + ("ev_ebitda",)
            share = "a quarter" if charge["net_share"] == NET_MARGIN_COLLAPSE_SHARE else "a third"
            reason = (
                f"the latest net margin {charge['latest']:.2%} is under {share} of its prior "
                f"median {charge['prior_median']:.2%}, and {ebit_text}")
    if reason is not None:
        dropped = [m for m in to_drop if implied.get(m) is not None]
        for m in dropped:
            implied[m] = None
        if dropped:
            names = [_MULTIPLE_NAMES[m] for m in dropped]
            which = (f"{names[0]} gives" if len(names) == 1 else
                     f"{', '.join(names[:-1])} and {names[-1]} give")
            measure = "EPS or EBITDA" if "ev_ebitda" in dropped else "EPS"
            # Under the trailing basis the screen still reads the fiscal year.
            scope = (f" (screened on {_fiscal_year_text(fin)}, also under the trailing basis)"
                     if basis == "ttm" and ttm is not None else "")
            notes.append(
                f"{EARNINGS_COLLAPSE_NOTE_PREFIX}: {reason}{scope}, so {which} no implied "
                f"price: a peer multiple of that {measure} would price the charge, not the "
                "business.")

    # --- Basis notes: trailing fallbacks, or the fiscal-year vs trailing gap -- #
    # Under the trailing basis, name the metrics that stayed on the fiscal year
    # and still give a price. Under the default basis peers' trailing multiples
    # meet fiscal-year figures: flag a large gap to Yahoo's trailing ones.
    if basis == "ttm":
        basis_note = _trailing_fallback_note(fin, ttm, implied) if ttm is not None else None
    else:
        basis_note = _trailing_gap_note(fin, fy_metrics, ttm, implied, equity_only)
    if basis_note:
        notes.append(basis_note)

    # --- Flag EV multiples whose bridge yields a non-positive equity value ----- #
    # These are dropped from the summary below; explain why so net-debt-heavy
    # targets are not silently excluded.
    for m in EV_MULTIPLES:
        p = implied.get(m)
        if is_num(p) and p <= 0:
            notes.append(
                f"{m} implies non-positive equity value (net debt exceeds implied EV); "
                "excluded from summary."
            )

    # --- Summary across all non-None implied prices ---------------------------- #
    implied_prices = [p for p in implied.values() if is_num(p) and p > 0]
    if implied_prices:
        implied_price_summary = {
            "low": min(implied_prices),
            "median": median(implied_prices),
            "high": max(implied_prices),
        }
    else:
        implied_price_summary = _empty_summary()
        notes.append("No implied prices could be derived from the peer medians.")

    return CompsResult(
        target=target_row,
        peers=peer_rows,
        stats=stats,
        implied=implied,
        implied_price_summary=implied_price_summary,
        notes=notes,
        peer_status=peer_status,
        target_metrics=dict(metrics),
    )
