"""Trading-comparables ("comps") valuation.

Given a target company and a set of peer tickers, this module builds a comps
table of trailing multiples, trims outliers per multiple, computes summary
statistics, and derives an implied share price by applying each peer-median
multiple to the target's own metric.

Valuation methods:

  * EV multiples (ev_ebitda, ev_sales) imply an enterprise value, from which the
    non-equity claims (net debt + minority interest + preferred) are stripped to
    get equity value, then divided by shares.
  * Equity multiples (pe, pb) apply directly to per-share earnings / book value.
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
    9.15 to 6.97). The same happens when the net margin is under a quarter of
    its own prior median in a year whose EBIT margin is itself out of line,
    which that screen skips (``ddm_fcfe.ebit_charge_collapse``). An EBIT
    margin that is a one-off drop, or under a third of its prior median,
    points to a charge inside operating profit, so also inside EBITDA (EBIT +
    D&A): EV/EBITDA is left out too, and EV/Sales and P/B are kept (GM
    FY2025: EBIT margin 1.6% against 6.7%, net 1.46% against 6.1%, a
    P/E-implied 25.48 against 79.91 from P/B, so the comps median was 52.70,
    'Overvalued -35%'). Next to a one-off rise in the EBIT margin the charge
    is below operating profit, and only P/E is left out.

All monetary inputs are absolute units (not millions); multiples are pure ratios.
The provider is touched ONLY through the DataProvider interface
(`suggest_peers` / `get_peer_comp_rows`). Never crashes on missing data: every
access is guarded and human-readable issues are appended to `CompsResult.notes`.

"""

from __future__ import annotations

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
    summary_stats,
    trim_outliers,
)
from .ddm_fcfe import NET_MARGIN_COLLAPSE_SHARE, ebit_charge_collapse, net_margin_collapse

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
_MULTIPLE_NAMES = {"pe": "P/E", "peg": "PEG", "ev_ebitda": "EV/EBITDA"}


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


def _latest(series: Optional[list]) -> Optional[float]:
    """Most-recent (last) finite element of an oldest->newest series, else None."""
    if not series:
        return None
    try:
        v = series[-1]
    except (IndexError, TypeError):
        return None
    return float(v) if is_num(v) else None


def _eps_cagr(fin) -> Optional[float]:
    """Diluted EPS CAGR across the available history (oldest->newest).

    Returns None if fewer than two positive endpoints exist (a sign change makes
    growth meaningless for a PEG). The result is a decimal (0.12 == 12%).
    """
    if fin is None:
        return None
    ni = getattr(fin, "net_income", None)
    shares = getattr(fin, "diluted_shares", None) or []
    if not ni or len(ni) < 2:
        return None
    if len(shares) != len(ni) or not all(is_num(s) and s > 0 for s in (shares[0], shares[-1])):
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
) -> CompRow:
    """Assemble the target's own CompRow for display.

    enterprise_value = market_cap + net_debt + minority + preferred.
    Trailing multiples are computed from the latest fundamentals; any that cannot
    be derived (missing / non-positive denominator) are left as None.
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
        enterprise_value = market_cap + net_debt + minority + preferred

    # Latest fundamental metrics for the trailing multiples.
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


def _collect_peer_rows(
    provider: DataProvider,
    peer_tickers: list[str],
    target_ticker: str,
    notes: list[str],
) -> list[CompRow]:
    """Fetch peer comp rows via the provider, dropping the target and any row with
    no usable multiples. Never raises -- a provider failure yields an empty list
    plus a note."""
    try:
        raw_rows = provider.get_peer_comp_rows(list(peer_tickers))
    except Exception as exc:  # pragma: no cover - defensive against provider bugs
        notes.append(f"Failed to fetch peer comp rows: {exc}")
        return []

    if not raw_rows:
        return []

    target_upper = (target_ticker or "").upper()
    kept: list[CompRow] = []
    seen: set[str] = set()
    for row in raw_rows:
        if row is None:
            continue
        rt = (getattr(row, "ticker", "") or "").strip().upper()
        if rt and rt == target_upper:
            # Exclude the target if the provider returned it among the peers.
            continue
        if not _row_has_usable_multiple(row):
            continue
        if rt and rt in seen:
            continue  # one issuer must have one vote in the peer distribution
        seen.add(rt)
        kept.append(row)
    return kept


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
) -> Optional[float]:
    """Apply a peer-median multiple to the target's own metric -> implied price.

    EV multiples back out equity value (EV - net debt - minority - preferred)
    before dividing by shares; equity multiples apply per-share. Returns None when
    any required input is missing / non-positive.
    """
    if med is None or not is_num(med) or med <= 0:
        return None

    if multiple == "ev_ebitda":
        if not (is_num(ebitda_latest) and ebitda_latest > 0 and shares):
            return None
        ev_star = med * ebitda_latest
        equity_star = ev_star - net_debt - minority - preferred
        return safe_div(equity_star, shares)

    if multiple == "ev_sales":
        if not (is_num(revenue_latest) and revenue_latest > 0 and shares):
            return None
        ev_star = med * revenue_latest
        equity_star = ev_star - net_debt - minority - preferred
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
    current_price : float
        Latest market price per share for the target.
    tax_rate : float | None
        An explicit tax rate (the CLI's ``--tax``) for the net-margin collapse
        screen, as the FCFE uses; else the effective rate.

    Returns
    -------
    CompsResult
        Always returned -- never raises on missing data.
    """
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

    shares = _target_shares(company)
    if shares is None:
        notes.append("Share count unavailable; implied prices cannot be computed.")

    # --- Build the target's own display row ------------------------------------ #
    target_row = _build_target_row(
        company,
        current_price,
        shares,
        net_debt,
        minority,
        preferred,
        notes,
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

    # Drop the target itself from the requested set (case-insensitive).
    target_upper = ticker.upper()
    peer_tickers = list(dict.fromkeys(
        t.strip().upper() for t in peer_tickers
        if t.strip() and t.strip().upper() != target_upper
    ))

    if not peer_tickers:
        notes.append("No peer tickers available; comps could not be computed.")
        return CompsResult(
            target=target_row,
            peers=[],
            stats={},
            implied={},
            implied_price_summary=_empty_summary(),
            notes=notes,
        )

    # --- Fetch and clean peer rows --------------------------------------------- #
    peer_rows = _collect_peer_rows(provider, peer_tickers, ticker, notes)
    if not peer_rows:
        notes.append("No usable peers returned by the data provider; comps could not be computed.")
        return CompsResult(
            target=target_row,
            peers=[],
            stats={},
            implied={},
            implied_price_summary=_empty_summary(),
            notes=notes,
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
            notes.append(f"No usable peer values for {m} after outlier trimming.")

    # --- Target metrics needed for implied prices ------------------------------ #
    fin = getattr(company, "financials", None)
    ebitda_latest = _latest(getattr(fin, "ebitda", None)) if fin is not None else None
    revenue_latest = _latest(getattr(fin, "revenue", None)) if fin is not None else None
    ni_latest = _latest(getattr(fin, "net_income", None)) if fin is not None else None
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
        )
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
    flag = financial_institution_detail(company)
    if flag is not None and flag[1] in EQUITY_MULTIPLES_ONLY_KINDS:
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
    # quarter of its prior median (``ebit_charge_collapse``): a charge inside
    # EBIT (GM FY2025) also sits in EBITDA, which every provider builds as
    # EBIT + D&A, so EV/EBITDA goes too; next to a one-off EBIT gain the charge
    # is below EBIT, and only the EPS multiples go.
    collapse = net_margin_collapse(fin, tax_rate) if fin is not None else None
    prior_med = collapse.get("prior_median") if collapse else None
    reason = None
    to_drop = EARNINGS_MULTIPLES
    if (collapse is not None and is_num(prior_med) and prior_med > 0
            and collapse["latest"] < NET_MARGIN_COLLAPSE_SHARE * prior_med):
        reason = (
            f"the latest net margin {collapse['latest']:.2%} is under a quarter of both its "
            f"prior median {prior_med:.2%} and the {collapse['implied']:.1%} its EBIT margin "
            f"{collapse['ebit_margin']:.1%} implies after interest and tax (items below "
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
            reason = (
                f"the latest net margin {charge['latest']:.2%} is under a quarter of its prior "
                f"median {charge['prior_median']:.2%}, and {ebit_text}")
    if reason is not None:
        dropped = [m for m in to_drop if implied.get(m) is not None]
        for m in dropped:
            implied[m] = None
        if dropped:
            names = [_MULTIPLE_NAMES[m] for m in dropped]
            which = (f"{names[0]} gives" if len(names) == 1 else
                     f"{', '.join(names[:-1])} and {names[-1]} give")
            basis = "EPS or EBITDA" if "ev_ebitda" in dropped else "EPS"
            notes.append(
                f"{EARNINGS_COLLAPSE_NOTE_PREFIX}: {reason}, so {which} no implied price: a "
                f"peer multiple of that {basis} would price the charge, not the business.")

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
    )
