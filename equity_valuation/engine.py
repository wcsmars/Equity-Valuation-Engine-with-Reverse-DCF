"""Orchestration: pull data, run every model, assemble a ValuationReport.

This is the public entry point. Each model is run defensively so that one model
failing (e.g. no dividends -> no DDM, or a thin EDGAR record) never sinks the
whole valuation -- failures become `warnings` on the report, as do the fallbacks
and clamps each model records in its notes (prefixed with the model name) and
any method left out of the blended target. Comps run on the peers passed in;
``auto_peers=True`` (opt-in) asks the provider to discover them when none
are, and lists the names in the warnings for review. Each requested peer's
status is in ``comps.peer_status``; when comps ran on the others, a peer they
could not use (not found, no usable multiple, or the target itself) is named
in the warnings ("Comps: Peers not used: ..."). With no usable peer at all the
no-usable-peers warning names them instead. Comps apply the peers' trailing
multiples to the target's latest fiscal-year figures, with a warning when
Yahoo's trailing net income or EBITDA is more than 15% away or a loss (the
gap may be growth or one-off items); ``comps_basis="ttm"`` (opt-in) uses
Yahoo's trailing figures for the target instead, not screened for one-off
items, while the DCF, DDM and FCFE stay on the fiscal year.

Banks, insurers, REITs and lenders/BDCs are detected from the market data's
industry, the EDGAR client's tag-based classification and provider notes (see
``utils.financial_institution_detail``). Their FCFF DCF and FCFE are still
computed and shown (marked "not in blend" in the football field), but left out
of the blended target: for a bank, insurer or lender debt and interest are
operating items, and a REIT grows by buying property, which is outside capex,
so neither model describes the business. An industrial company that
consolidates a large captive finance arm (the data layer's "captive_finance"
kind: GM, F, TM, DE, CAT) takes the same path: its consolidated debt,
interest, D&A and capex mix the finance arm's lending book with the industrial
business, so its DCF and FCFE are reference-only too. So does a debt-funded
operating lessor (the "lessor" kind: AerCap), whose borrowing funds the fleet
it leases out. If only the DDM is left for a financial institution (no
comps), the blended target is still reported but no verdict and no upside are
given: the DDM counts dividends only (regular plus any recurring variable
ones), so it understates a company that also returns capital through
buybacks, and no second method checks it. The same holds after a corporate
action the data layer flags as likely a spin-off (HON after the Aerospace
spin-off): the target is kept for reference, with no verdict or upside, until
the next annual report.
For a captive-finance group or a lessor the DDM is shown for reference only
and left out of the blend when its dividend is a low share of net income
(``utils.low_payout_ddm``: next to comps a two-method median is the mean, so
the DDM would carry half the weight), and it never sets the target on its own
(``utils.ddm_left_alone``): without a comps price there is no blended
target, and a warning asks for peers (or, when the peers supplied or
discovered gave no price, says so).
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import isclose
from typing import Optional

from . import config
from .schemas import (
    DCFAssumptions,
    DDMAssumptions,
    FootballFieldRow,
    MacroAssumptions,
    ValuationReport,
)
from .utils import (
    FINANCIAL_KIND_REASONS,
    LOW_DDM_PAYOUT_SHARE,
    NO_OPERATING_MARGIN_NOTE_PREFIX,
    NOT_IN_BLEND,
    REFERENCE_ONLY_KINDS,
    available_model_price,
    ddm_left_alone,
    ddm_reference_only,
    financial_institution,
    financial_institution_detail,
    is_num,
    low_payout_ddm,
    median,
)

# What the DDM's D0 holds (``MarketData.dividend_per_share``: Yahoo's indicated
# regular rate plus the 3-year average of recurring variable dividends), as
# quoted in the DDM-alone warnings.
_DDM_BASIS = ("counts dividends only (regular plus any recurring variable dividends; "
              "buybacks and one-off specials are ignored)")

# Start of the data layer's note (``data.provider._corporate_action_warning``)
# for a corporate action after the latest fiscal year that is not a plain
# split (likely a spin-off): the statements may still include a business the
# company no longer owns, so ``_build_summary`` withholds the verdict.
CORPORATE_ACTION_NOTE_PREFIX = "WARNING: Yahoo records a corporate action after FY"

# Prefix of the note ``models.comps`` records when the provider raised while
# fetching the peer rows (``_collect_peer_rows``); the no-usable-peers warning
# then reports the failure instead of blaming the tickers.
_PEER_FETCH_FAILED = "Failed to fetch peer comp rows:"


def value_company(
    ticker: str,
    *,
    provider=None,
    macro: Optional[MacroAssumptions] = None,
    dcf_assumptions: Optional[DCFAssumptions] = None,
    ddm_assumptions: Optional[DDMAssumptions] = None,
    peers: Optional[list[str]] = None,
    run_dcf: bool = True,
    run_comps: bool = True,
    run_ddm: bool = True,
    run_fcfe: bool = True,
    run_sensitivity: bool = True,
    run_reverse_dcf: bool = True,
    auto_peers: bool = False,
    comps_basis: str = "fy",
) -> ValuationReport:
    """Value `ticker` and return a fully-assembled ValuationReport.

    Parameters mirror the CLI flags. `provider` defaults to the EDGAR+yfinance
    HybridProvider. Any model can be toggled off. With a DCF,
    ``report.reverse_dcf`` holds what the market price implies for year-1
    revenue growth, the target EBIT margin and the discount rate, other
    inputs held (``models.reverse_dcf``; ``run_reverse_dcf=False`` skips it).

    ``auto_peers`` (the CLI's ``--auto-peers``, off by default) lets comps
    run on peers the provider discovers (``discover_peers``: Yahoo's
    industry list for the HybridProvider) when no ``peers`` are given; the
    names and the discovery notes are added to the warnings so they can be
    reviewed. Without it no discovery request is made.

    ``comps_basis`` (the CLI's ``--comps-basis``): the target figures the
    peer multiples are applied to. 'fy' (the default) uses the latest fiscal
    year's and warns when Yahoo's trailing net income or EBITDA is more than
    15% away, or a loss against a fiscal-year profit ("Comps: Fiscal-year vs
    trailing gap: ..."); 'ttm' uses Yahoo's trailing twelve months for the
    target's revenue, EBITDA and net income when they are usable, and says
    so or why not ("Comps: Comps on trailing twelve months ..."). The DCF,
    DDM and FCFE stay on the fiscal year (see ``models.comps.run_comps``).
    Anything else raises ValueError before any data is fetched.
    """
    from .models.comps import normalize_basis

    comps_basis = normalize_basis(comps_basis)
    ticker = ticker.strip().upper()
    macro = macro or MacroAssumptions()
    dcf_assumptions = dcf_assumptions or DCFAssumptions()
    ddm_assumptions = ddm_assumptions or DDMAssumptions()

    # Lazy imports so a missing optional dep surfaces only when actually used.
    if provider is None:
        from .data.provider import HybridProvider

        provider = HybridProvider()

    company = provider.get_company_data(ticker)
    current_price = company.market.price

    report = ValuationReport(
        company=company,
        macro=macro,
        current_price=current_price,
        valued_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    report.warnings.extend(company.source_notes)

    # --- DCF ---------------------------------------------------------------- #
    if run_dcf:
        try:
            from .models.dcf import run_dcf as _run_dcf

            report.dcf = _run_dcf(company, macro, dcf_assumptions, current_price)
            _add_notes(report, "WACC", report.dcf.wacc.detail.get("notes"))
            _drop_restated_beta_note(report)
            _add_notes(report, "DCF", report.dcf.assumptions.get("notes"))
        except Exception as exc:  # noqa: BLE001 - degrade, don't crash
            report.warnings.append(f"DCF failed: {exc}")

    # --- Reverse DCF (what the price implies; a diagnostic) ----------------- #
    if run_reverse_dcf and report.dcf is not None:
        try:
            from .models.reverse_dcf import solve_reverse_dcf

            report.reverse_dcf = solve_reverse_dcf(
                company, macro, dcf_assumptions, current_price, report.dcf.assumptions)
        except Exception as exc:  # noqa: BLE001
            report.warnings.append(f"Reverse DCF failed: {exc}")

    # --- Trading comps ------------------------------------------------------ #
    peers_discovered = False  # comps ran on auto_peers names, not the caller's
    if run_comps:
        try:
            from .models.comps import (
                COMPS_GAP_NOTE_PREFIX,
                COMPS_TTM_NOTE_PREFIX,
                EARNINGS_COLLAPSE_NOTE_PREFIX,
                EQUITY_MULTIPLES_NOTE_PREFIX,
                FEW_PEER_VALUES_NOTE_PREFIX,
                PEERS_NOT_USED_NOTE_PREFIX,
                run_comps as _run_comps,
            )

            discover = bool(auto_peers and not peers)
            comps_peers = _auto_peers(report, provider, ticker) if discover else peers
            peers_discovered = discover and bool(comps_peers)
            report.comps = _run_comps(company, provider, comps_peers, current_price,
                                      tax_rate=macro.tax_rate, basis=comps_basis)
            if report.comps is not None and not report.comps.peers:
                report.warnings.append(_no_usable_peers_warning(
                    comps_peers, ticker, report.comps.notes,
                    discovered=discover))
                # (That warning names the peers; why each gave nothing stays
                # in comps.notes and comps.peer_status.)
            elif report.comps is not None:
                # Requested peers left out (a mistyped ticker is "not found"),
                # a lender's, insurer's, captive-finance group's or lessor's
                # comps use P/E and P/B only, a collapsed net margin drops
                # the P/E price, and a multiple with too few peer values gives
                # no price; the trailing basis's notes and a large
                # fiscal-year vs trailing gap say which figures the target's
                # multiples use. Say so next to the blend (the other comps
                # notes stay on the comps result).
                _add_notes(report, "Comps", [
                    n for n in report.comps.notes or []
                    if str(n).startswith((PEERS_NOT_USED_NOTE_PREFIX,
                                          EQUITY_MULTIPLES_NOTE_PREFIX,
                                          EARNINGS_COLLAPSE_NOTE_PREFIX,
                                          FEW_PEER_VALUES_NOTE_PREFIX,
                                          COMPS_TTM_NOTE_PREFIX,
                                          COMPS_GAP_NOTE_PREFIX))])
        except Exception as exc:  # noqa: BLE001
            report.warnings.append(f"Comps failed: {exc}")

    # --- DDM ---------------------------------------------------------------- #
    if run_ddm:
        try:
            from .models.ddm_fcfe import run_ddm as _run_ddm

            report.ddm = _run_ddm(company, macro, ddm_assumptions, current_price)
            if report.ddm is None:
                report.warnings.append("DDM skipped: company pays no dividend.")
            else:
                _add_notes(report, "DDM", report.ddm.detail.get("notes"))
        except Exception as exc:  # noqa: BLE001
            report.warnings.append(f"DDM failed: {exc}")

    # --- FCFE --------------------------------------------------------------- #
    if run_fcfe:
        try:
            from .models.ddm_fcfe import run_fcfe as _run_fcfe

            report.fcfe = _run_fcfe(company, macro, ddm_assumptions, current_price)
            _add_notes(report, "FCFE", report.fcfe.detail.get("notes"))
        except Exception as exc:  # noqa: BLE001
            report.warnings.append(f"FCFE failed: {exc}")

    # The DDM and FCFE price at DEFAULT_BETA when the market beta is missing;
    # the WACC note says so only when the DCF ran, so say it here otherwise.
    _add_default_beta_note(report)

    # --- Sensitivity (depends on DCF being viable) -------------------------- #
    if run_sensitivity and report.dcf is not None:
        try:
            from .models.sensitivity import dcf_sensitivity

            report.sensitivities = dcf_sensitivity(
                company, macro, dcf_assumptions, current_price
            )
            _drop_off_centre_value_driver_grids(report)
        except Exception as exc:  # noqa: BLE001
            report.warnings.append(f"Sensitivity failed: {exc}")

    # --- Football field + blended target ------------------------------------ #
    try:
        from .models.sensitivity import build_football_field

        report.football_field = build_football_field(report)
    except Exception as exc:  # noqa: BLE001
        report.warnings.append(f"Football field failed: {exc}")
        report.football_field = _fallback_football_field(report)

    report.summary = _build_summary(report, peers_discovered=peers_discovered)
    return report


def _discover_peers(provider, ticker: str, industry_key: Optional[str] = None,
                    name: Optional[str] = None) -> dict:
    """The provider's ``discover_peers(ticker, industry_key=..., name=...)``
    dict (``{"peers": [{"ticker": ...}, ...], "notes": [...], "source":
    ...}``); ``name`` is the target's company name, which keeps its other
    share class out of the peers.

    A provider without that method (one not built on ``DataProvider``) is
    read the way the ``DataProvider.discover_peers`` default (data/base.py)
    reads it: its ``suggest_peers`` list, up to
    ``config.PEER_DISCOVERY_LIMIT`` names, with source "the data provider".
    An error gives no peers and a note. Never raises."""
    try:
        discover = getattr(provider, "discover_peers", None)
        if callable(discover):
            result = discover(ticker, industry_key=industry_key, name=name)
        else:
            suggested = provider.suggest_peers(ticker) or []
            result = {"ticker": ticker, "notes": [], "source": "the data provider",
                      "peers": [{"ticker": t} for t in suggested][:config.PEER_DISCOVERY_LIMIT]}
    except Exception as exc:  # noqa: BLE001 - discovery is optional
        return {"ticker": ticker, "peers": [], "source": "the data provider",
                "notes": [f"No peers discovered: peer discovery failed ({exc})."]}
    return result if isinstance(result, dict) else {"ticker": ticker, "peers": [], "notes": []}


def _auto_peers(report: ValuationReport, provider, ticker: str) -> list[str]:
    """Peer tickers for ``value_company(auto_peers=True)`` when none were
    given: the provider's discovered names (see ``_discover_peers``), passed
    to comps like ``--peers``. The target is dropped, and so is any peer
    listed under its company name (the market name, else the registered
    one): another share class of the same company (GOOG for GOOGL) would
    pull the comps median toward the market price. Repeats are dropped too.
    The names go into the warnings ("Comps: peers discovered automatically
    from <source>: A, B; review them.") with each discovery note as
    ``"Comps: <note>"``. An empty list leaves comps to the provider's
    ``suggest_peers`` and the usual no-peers warning."""
    from .data.market import company_name_key

    company = report.company
    market = getattr(company, "market", None)
    # Company names, Yahoo's (which its industry lists use) first; a name
    # that only repeats the ticker (the data layer's last fallback) is none.
    own_names = [nm.strip() for nm in (getattr(market, "name", None),
                                        getattr(company, "name", None))
                 if isinstance(nm, str) and company_name_key(nm) is not None
                 and company_name_key(nm) != company_name_key(ticker)]
    found = _discover_peers(provider, ticker, getattr(market, "industry_key", None),
                            own_names[0] if own_names else None)
    own_keys = {company_name_key(nm) for nm in own_names}
    targets = {ticker, ticker.replace(".", "-")}
    peers: list[str] = []
    for item in found.get("peers") or []:
        symbol = item.get("ticker") if isinstance(item, dict) else item
        symbol = symbol.strip().upper() if isinstance(symbol, str) else ""
        if isinstance(item, dict) and company_name_key(item.get("name")) in own_keys:
            continue  # another share class of the target
        if symbol and symbol.replace(".", "-") not in targets and symbol not in peers:
            peers.append(symbol)
    if peers:
        source = found.get("source") or "the data provider"
        report.warnings.append(f"Comps: peers discovered automatically from {source}: "
                               f"{', '.join(peers)}; review them.")
    _add_notes(report, "Comps", found.get("notes"))
    return peers


def _drop_off_centre_value_driver_grids(report: ValuationReport) -> None:
    """Drop a value-driver DCF's sensitivity grids when their centre cell is
    not the headline price.

    A consistency check, not a routine step: ``dcf_sensitivity`` prices a
    value-driver headline's cells with the value-driver method and its RONIC,
    so the WACC grid's centre cell (no WACC shift, the headline terminal
    growth) re-prices the headline's own inputs and equals it, and the grids
    are kept. Only grids priced some other way (e.g. Gordon cells around a
    value-driver headline) are dropped, since they would show, and widen the
    football-field DCF bar with, a valuation the headline did not use; a
    warning says so and the DCF bar falls back to its +/-15% band. A blank
    centre cell (a clamped or fallback cell) is left as is. Gordon and
    exit-multiple runs are not checked.
    """
    dcf = report.dcf
    if (getattr(dcf, "assumptions", None) or {}).get("terminal_method") != "value_driver":
        return
    headline = getattr(dcf, "implied_price", None)
    try:
        i = list(config.SENSITIVITY_WACC_DELTAS).index(0.0)
        j = list(config.SENSITIVITY_GROWTH_DELTAS).index(0.0)
    except ValueError:
        return  # no unshifted cell to compare
    for grid in report.sensitivities or []:
        if "wacc" not in (getattr(grid, "row_label", "") or "").lower():
            continue
        try:
            centre = grid.grid[i][j]
        except (IndexError, TypeError):
            return
        if is_num(centre) and is_num(headline) and not isclose(
                centre, headline, rel_tol=1e-9, abs_tol=1e-6):
            report.warnings.append(
                f"Sensitivity: grids omitted: their centre cell ({centre:,.2f}) does not "
                f"reproduce the value-driver DCF price ({headline:,.2f}), so the cells were "
                "not priced with the headline's terminal method.")
            report.sensitivities = []
        return


def _add_notes(report: ValuationReport, label: str, notes) -> None:
    """Surface a model's recorded fallbacks/clamps as ``"<label>: <note>"`` warnings.

    A note that starts with ``"WARNING: "`` keeps that prefix in front
    (``"WARNING: <label>: ..."``), so readers that list data-quality warnings
    first (the AI context) pick it up.
    """
    for note in notes or []:
        text = str(note)
        if text.startswith("WARNING: "):
            warning = f"WARNING: {label}: {text[len('WARNING: '):]}"
        else:
            warning = f"{label}: {text}"
        if warning not in report.warnings:
            report.warnings.append(warning)


def _no_usable_peers_warning(peers: Optional[list[str]], ticker: str,
                             notes=None, *, discovered: bool = False) -> str:
    """Warning for comps that ran without a usable peer. When the provider
    failed to fetch the peer rows (comps' "Failed to fetch peer comp rows"
    note, e.g. a network outage) it quotes that failure, since the tickers
    were never checked. Otherwise, when peers were supplied (other than the
    target itself) it names them, since asking for peers again would repeat
    what the user just did; with none it asks for them. ``discovered``: the
    peers came from ``auto_peers`` discovery, not from the caller."""
    requested = list(dict.fromkeys(
        str(t).strip().upper() for t in (peers or [])
        if t and str(t).strip() and str(t).strip().upper() != ticker
    ))
    failure = next((str(n)[len(_PEER_FETCH_FAILED):].strip() for n in notes or []
                    if str(n).startswith(_PEER_FETCH_FAILED)), None)
    if failure is not None:
        whose = f" for {', '.join(requested)}" if requested else ""
        detail = f" ({failure})" if failure else ""
        return (f"Comps: peer data{whose} could not be fetched{detail}, so comps give "
                "no price.")
    if requested and discovered:
        return (f"Comps: none of the peers discovered ({', '.join(requested)}) returned "
                "usable multiples; supply peers with --peers.")
    if requested:
        return (f"Comps: none of the peers supplied ({', '.join(requested)}) returned "
                "usable multiples; check the tickers or supply other peers.")
    return "Comps: no usable peers found (pass --peers to supply them)."


def _add_default_beta_note(report: ValuationReport) -> None:
    """Say that the DDM/FCFE cost of equity uses ``config.DEFAULT_BETA`` when
    the market beta is missing and no DCF carries the WACC's own note (DCF
    off or failed). With a DCF the "WACC: beta unavailable" note covers it.
    Only the models that produced a price are named."""
    if report.dcf is not None:
        return
    if is_num(getattr(getattr(report.company, "market", None), "beta", None)):
        return
    ran = [name for name, result in (("DDM", report.ddm), ("FCFE", report.fcfe))
           if available_model_price(result) is not None]
    if not ran:
        return
    note = (f"CAPM: beta unavailable; using DEFAULT_BETA={config.DEFAULT_BETA} for the "
            f"{' and '.join(ran)} cost of equity")
    if note not in report.warnings:
        report.warnings.append(note)


def _drop_restated_beta_note(report: ValuationReport) -> None:
    """Drop the market data's beta-adjustment note from ``report.warnings``
    once the WACC note restates it with the raw beta and that beta's cost of
    equity, so one adjustment does not take two lines (the note stays in
    ``company.source_notes``). Kept when the WACC note has no raw beta and
    refers to the data notes instead."""
    if report.dcf is None or report.dcf.wacc.detail.get("beta_raw") is None:
        return
    if not any(str(w).startswith("WACC: beta ") for w in report.warnings):
        return
    from .models.wacc import beta_adjustment

    adjusted = beta_adjustment(report.company)
    note = adjusted.get("note") if adjusted else None
    if note and note in report.warnings:
        report.warnings.remove(note)


def _fallback_football_field(report: ValuationReport) -> list[FootballFieldRow]:
    """Minimal football field if the model helper failed -- one bar per method.

    Self-contained on purpose (it runs when models.sensitivity failed). Bands are
    ordered so low <= base <= high holds for negative prices too.
    """
    def band(name: str, p, pct: float) -> Optional[FootballFieldRow]:
        if not is_num(p):
            return None
        a, b = p * (1.0 - pct), p * (1.0 + pct)
        return FootballFieldRow(name, min(a, b), p, max(a, b))

    rows: list[FootballFieldRow] = []
    m = report.company.market
    # A financial institution's DCF and FCFE bars are marked, as in the blend.
    mark = NOT_IN_BLEND if financial_institution(report.company) else ""
    if m.fifty_two_week_low and m.fifty_two_week_high:
        rows.append(
            FootballFieldRow(
                "52-week range", m.fifty_two_week_low, report.current_price, m.fifty_two_week_high
            )
        )
    if report.dcf and (getattr(report.dcf, "assumptions", None) or {}).get("valuation_available") is not False:
        rows.append(band("DCF" + mark, report.dcf.implied_price, 0.15))
    if report.comps and report.comps.implied_price_summary:
        s = report.comps.implied_price_summary
        if s.get("low") and s.get("high"):
            if s["low"] == s["high"]:
                # A single comps price: a +/-10% band, as models.sensitivity
                # draws it, rather than a zero-width sliver.
                rows.append(band("Comps", s.get("median") or s["low"], 0.10))
            else:
                rows.append(FootballFieldRow("Comps", s["low"],
                                             s.get("median", report.current_price), s["high"]))
    if available_model_price(report.ddm) is not None:
        ddm_mark = NOT_IN_BLEND if ddm_reference_only(report) else ""
        rows.append(band("DDM" + ddm_mark, report.ddm.implied_price, 0.10))
    if report.fcfe and (getattr(report.fcfe, "detail", None) or {}).get("valuation_available") is not False:
        rows.append(band("FCFE" + mark, report.fcfe.implied_price, 0.10))
    return [r for r in rows if r is not None]


def _build_summary(report: ValuationReport, *, peers_discovered: bool = False) -> dict:
    """Collect each method's central estimate and a blended (median) target.

    ``methods`` keeps every method's value for display. The blend differs in
    four ways, each recorded in ``report.warnings`` and, for a method left out,
    in ``excluded_from_blend`` (method -> reason):
      * an unavailable model or a non-finite price is left out. A zero is kept
        only when the model explicitly marks a completed valuation, so an
        actual zero-equity estimate is distinct from a failure placeholder;
      * for a bank, insurer, REIT or lender/BDC, a company with a consolidated
        captive finance arm or a debt-funded operating lessor
        (``financial_institution_detail`` gives the reason and the kind, whose
        rationale the warning quotes) the DCF and FCFE are left out;
      * for a captive-finance group or a lessor (``REFERENCE_ONLY_KINDS``) the
        DDM is left out when its dividend is below ``LOW_DDM_PAYOUT_SHARE`` of
        net income (``low_payout_ddm``: it ignores buybacks), and also when it
        would be the only method left (``ddm_left_alone``: no comps). The
        blend then rests on comps, or there is no blended target (None, "N/A")
        and a warning asks for peers, or says the peers supplied gave no
        price (``peers_discovered``: the peers came from ``auto_peers``
        discovery, so the warning says "discovered" and still asks for
        peers);
      * a negative price (claims senior to common equity exceed the value)
        counts as 0, because equity cannot be worth less.
    A blend of 0 is therefore a real target (-100%, "Overvalued"); an empty
    blend has no target (upside None, "N/A"). One more case gets "N/A" although
    it has a target: a financial institution whose blend is the DDM alone (no
    comps). The DDM counts dividends only (regular plus any recurring variable
    ones), not buybacks or one-off specials, so on its own it reads a bank or
    insurer that also buys back stock as overvalued, and no second method
    checks it. The target is kept, the upside is None (so no downside is
    shown next to "N/A"), and a warning says why no verdict is given. The
    same applies to any target when ``company.source_notes`` holds the data
    layer's corporate-action warning (``CORPORATE_ACTION_NOTE_PREFIX``): the
    latest statements may still include a business spun off since, while the
    price and share count no longer do. ``verdict_withheld`` gives the reason
    in either case (None when a verdict is given or there is no target).

    ``financial_institution`` is the flag's short reason (or None) and
    ``financial_kind`` its kind ("bank", "insurer", "reit", "lender",
    "captive_finance", "lessor", "financial" or None).
    """
    methods: dict[str, float] = {}
    if report.dcf:
        methods["DCF"] = report.dcf.implied_price
    if report.comps and report.comps.implied_price_summary.get("median"):
        methods["Comps (median)"] = report.comps.implied_price_summary["median"]
    if report.ddm:
        methods["DDM"] = report.ddm.implied_price
    if report.fcfe:
        methods["FCFE"] = report.fcfe.implied_price

    def warn(text: str) -> None:
        if text not in report.warnings:
            report.warnings.append(text)

    detail = financial_institution_detail(report.company)
    financial = detail[0] if detail else None
    kind = detail[1] if detail else None
    captive = kind == "captive_finance"
    reference_only = kind in REFERENCE_ONLY_KINDS
    # Warning prefix and "left out for ..." wording per kind.
    if captive:
        prefix, whom = "Captive finance arm", "a company with a consolidated captive finance arm"
        excluded_reason = "not meaningful with a consolidated captive finance arm"
    elif kind == "lessor":
        prefix, whom = "Debt-funded lessor", "a debt-funded lessor"
        excluded_reason = "not meaningful for a debt-funded lessor"
    else:
        prefix, whom = "Financial institution", "a financial institution"
        excluded_reason = "not meaningful for a financial institution"
    if detail and ("DCF" in methods or "FCFE" in methods):
        # A DCF with no operating income to project gives no reference price
        # (models.dcf: GS, JPM), so only the FCFE is shown for reference.
        dcf_notes = (getattr(report.dcf, "assumptions", None) or {}).get("notes") or []
        dcf_unpriced = ("DCF" in methods and available_model_price(report.dcf) is None
                        and any(str(n).startswith(NO_OPERATING_MARGIN_NOTE_PREFIX)
                                for n in dcf_notes))
        no_dcf = "the DCF has no reference price (no operating income is reported)"
        if not dcf_unpriced:
            shown = ("both are shown for reference only and left out of the blended "
                     "target.")
        elif "FCFE" in methods:
            shown = ("the FCFE is shown for reference only and left out of the blended "
                     f"target; {no_dcf}.")
        else:
            shown = f"{no_dcf}."
        if captive:
            warn(f"Captive finance arm: {FINANCIAL_KIND_REASONS[kind]}. An FCFF DCF and an "
                 f"FCFE on the consolidated figures describe neither business, so {shown}")
        else:
            warn(f"{prefix} ({financial}): {FINANCIAL_KIND_REASONS[kind]}, so "
                 "an FCFF DCF and an FCFE do not describe the business. "
                 f"{shown[0].upper()}{shown[1:]}")
    has_comps = "Comps (median)" in methods
    # Peers that ran but gave no positive price (a lessor's peers without P/E
    # or P/B) are not a missing input: the wording below then says so rather
    # than asking for peers (the reports show "n/a", not "supply peers").
    peers_gave_no_price = bool(getattr(report.comps, "peers", None)) and not has_comps
    # Whose peers they were: discovered ones were not the user's choice, so
    # the wording still asks for peers.
    peer_source = "discovered" if peers_discovered else "supplied"
    ask_for_peers = not peers_gave_no_price or peers_discovered
    low_payout = low_payout_ddm(report) if "DDM" in methods else None
    ddm_alone = "DDM" in methods and ddm_left_alone(report)
    if low_payout is not None:
        text = (f"{prefix}: the DDM is shown for reference only and left out of the "
                "blended target")
        text += ", which rests on comps. " if has_comps else ". "
        text += (f"Its dividend is {low_payout:.0%} of net income (below "
                 f"{LOW_DDM_PAYOUT_SHARE:.0%}), so most earnings are bought back or retained, "
                 "which a dividends-only model ignores")
        text += (", and with comps as the only other method it would carry half the weight "
                 "of the blend." if has_comps else ".")
        warn(text)
    elif ddm_alone:
        warn(f"{prefix}: the DDM is shown for reference only and left out of the blended "
             f"target, which it would otherwise set on its own: it {_DDM_BASIS} and no second "
             "method checks it.")

    blend: list[float] = []
    blended_names: list[str] = []
    excluded: dict[str, str] = {}
    floored = False
    availability = {
        "DCF": (getattr(report.dcf, "assumptions", None) or {}).get("valuation_available"),
        "FCFE": (getattr(report.fcfe, "detail", None) or {}).get("valuation_available"),
    }
    for name, price in methods.items():
        if financial and name in ("DCF", "FCFE"):
            excluded[name] = excluded_reason
            continue
        if name == "DDM" and low_payout is not None:
            excluded[name] = (f"dividends only ({low_payout:.0%} of net income); "
                              "buybacks ignored")
            continue
        if name == "DDM" and ddm_alone:
            if not peers_gave_no_price:
                excluded[name] = "dividends only and no comps to check it; supply peers"
            else:
                excluded[name] = (f"dividends only and no comps price from the peers "
                                  f"{peer_source} to check it"
                                  + ("; supply peers" if ask_for_peers else ""))
            continue
        if is_num(price) and (price > 0 or (price == 0 and availability.get(name) is True)):
            blend.append(price)
            blended_names.append(name)
            continue
        if is_num(price) and price < 0:
            blend.append(0.0)
            blended_names.append(name)
            floored = True
            warn(f"{name} implies negative equity ({price:.2f} per share); "
                 "counted as 0.00 in the blended target.")
        else:
            reason = "non-finite implied price" if not is_num(price) else "no valuation (0.00)"
            excluded[name] = reason
            warn(f"{name} excluded from blended target: {reason}.")

    blended = median(blend)
    if blended == 0.0 and floored:
        warn("Blended target is 0.00: the median method implies negative equity, floored at "
             "zero (limited liability), so the upside is -100%.")
    cur = report.current_price
    has_price = is_num(cur) and cur > 0
    recommendation = _recommendation(blended, cur)
    upside = (blended / cur - 1.0) if (blended is not None and has_price) else None
    # The two warnings below name only the reference-only models that ran, and
    # ask for peers only when none were supplied or none was usable
    # (``peers_gave_no_price`` above), or when the peers were discovered
    # (``ask_for_peers``).
    ref_models = " and ".join(n for n in ("DCF", "FCFE") if n in methods)
    withheld: Optional[str] = None  # why a target gets no verdict
    if reference_only and blended is None and not has_comps:
        # The DCF and FCFE are reference-only and the DDM never stands alone
        # here: without a comps price there is nothing left to blend.
        parts = []
        if ref_models:
            verb = "are" if " and " in ref_models else "is"
            parts.append(f"the {ref_models} {verb} left out for {whom}")
        if "DDM" in methods:
            parts.append("the DDM is shown for reference only" if ref_models
                         else f"the DDM of {whom} is shown for reference only")
        comps_text = (f"the peers {peer_source} give no usable comps price"
                      if peers_gave_no_price else "there are no comps")
        text = (f"No blended target: {'; '.join(parts)}, and {comps_text}." if parts
                else f"No blended target for {whom}: {comps_text}.")
        if ask_for_peers:
            text += " Supply peers with --peers to get a target."
        warn(text)
    if financial and blended_names == ["DDM"]:
        # One-sided on its own: dividends only, while banks and insurers also
        # return capital through buybacks (JPM, C, PGR read 60-93%
        # overvalued), and nothing cross-checks it for a REIT or BDC. The
        # target is kept for reference; the verdict and upside are withheld.
        recommendation = "N/A"
        upside = None
        left = f"{ref_models} left out for {whom}; " if ref_models else ""
        comps_text = (f"no usable comps price from the peers {peer_source}"
                      if peers_gave_no_price else "no comps")
        text = (f"Blended target rests on the DDM alone ({left}{comps_text}), so no verdict or "
                f"upside is given: the DDM {_DDM_BASIS} and there is no "
                "second method to check it against.")
        if ask_for_peers:
            text += " Supply peers to add trading comps."
        warn(text)
        withheld = "the target rests on the DDM alone"
    action = next((str(n) for n in getattr(report.company, "source_notes", None) or []
                   if str(n).startswith(CORPORATE_ACTION_NOTE_PREFIX)), None)
    if action is not None and blended is not None:
        # The statements behind every method may still include a business
        # spun off since the fiscal year (HON: FY2025 earnings, with
        # Aerospace, over the post-spin price), so no direction is given.
        recommendation = "N/A"
        upside = None
        fy = action[len(CORPORATE_ACTION_NOTE_PREFIX):].split(" ", 1)[0]
        withheld = f"a corporate action after FY{fy} (likely a spin-off)"
        warn(f"Corporate action: no verdict or upside is given. The blended target is shown "
             f"for reference only: Yahoo records a corporate action after FY{fy} that is not "
             f"a plain split, so the FY{fy} statements behind it may still include a business "
             "the company no longer owns, while the price and share count no longer do.")
    return {
        "ticker": report.company.ticker,
        "name": report.company.name,
        "currency": report.company.market.currency,
        "current_price": cur,
        "methods": methods,
        "blended_target": blended,
        "blended_upside": upside,
        "recommendation": recommendation,
        "excluded_from_blend": excluded,
        "financial_institution": financial,
        "financial_kind": kind,
        "verdict_withheld": withheld,
    }


def _recommendation(target: Optional[float], price: Optional[float]) -> str:
    """Verdict from the blended target: +/-15% bands; "N/A" when there is no
    target or no positive price (a 0.0 target is a valid -100% "Overvalued").
    ``_build_summary`` also withholds it (and the upside) for a financial
    institution's DDM-only blend and after a flagged corporate action."""
    if target is None or not is_num(target) or not is_num(price) or price <= 0:
        return "N/A"
    upside = target / price - 1.0
    if upside >= 0.15 or isclose(upside, 0.15, rel_tol=0.0, abs_tol=1e-12):
        return "Undervalued"
    if upside <= -0.15 or isclose(upside, -0.15, rel_tol=0.0, abs_tol=1e-12):
        return "Overvalued"
    return "Fairly valued"
