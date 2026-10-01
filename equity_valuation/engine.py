"""Orchestration: pull data, run every model, assemble a ValuationReport.

This is the public entry point. Each model is run defensively so that one model
failing (e.g. no dividends -> no DDM, or a thin EDGAR record) never sinks the
whole valuation -- failures become `warnings` on the report, as do the fallbacks
and clamps each model records in its notes (prefixed with the model name) and
any method left out of the blended target.

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
given: the DDM counts regular dividends only, so it understates a company
that also returns capital through buybacks, and no second method checks it.
For a captive-finance group or a lessor the DDM is shown for reference only
and left out of the blend when its dividend is a low share of net income
(``utils.low_payout_ddm``: next to comps a two-method median is the mean, so
the DDM would carry half the weight), and it never sets the target on its own
(``utils.ddm_left_alone``): without a comps price there is no blended
target, and a warning asks for peers (or, when the peers supplied gave no
price, says so).
"""

from __future__ import annotations

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
    NOT_IN_BLEND,
    REFERENCE_ONLY_KINDS,
    ddm_left_alone,
    ddm_reference_only,
    financial_institution,
    financial_institution_detail,
    is_num,
    low_payout_ddm,
    median,
)


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
) -> ValuationReport:
    """Value `ticker` and return a fully-assembled ValuationReport.

    Parameters mirror the CLI flags. `provider` defaults to the EDGAR+yfinance
    HybridProvider. Any model can be toggled off.
    """
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

    # --- Trading comps ------------------------------------------------------ #
    if run_comps:
        try:
            from .models.comps import (
                EARNINGS_COLLAPSE_NOTE_PREFIX,
                EQUITY_MULTIPLES_NOTE_PREFIX,
                run_comps as _run_comps,
            )

            report.comps = _run_comps(company, provider, peers, current_price,
                                      tax_rate=macro.tax_rate)
            if report.comps is not None and not report.comps.peers:
                report.warnings.append(
                    "Comps: no usable peers found (pass --peers to supply them)."
                )
            elif report.comps is not None:
                # A lender's, insurer's, captive-finance group's or lessor's
                # comps use P/E and P/B only, and a collapsed net margin drops
                # the P/E price; say so next to the blend (the other comps
                # notes stay on the comps result).
                _add_notes(report, "Comps", [
                    n for n in report.comps.notes or []
                    if str(n).startswith((EQUITY_MULTIPLES_NOTE_PREFIX,
                                          EARNINGS_COLLAPSE_NOTE_PREFIX))])
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

    # --- Sensitivity (depends on DCF being viable) -------------------------- #
    if run_sensitivity and report.dcf is not None:
        try:
            from .models.sensitivity import dcf_sensitivity

            report.sensitivities = dcf_sensitivity(
                company, macro, dcf_assumptions, current_price
            )
        except Exception as exc:  # noqa: BLE001
            report.warnings.append(f"Sensitivity failed: {exc}")

    # --- Football field + blended target ------------------------------------ #
    try:
        from .models.sensitivity import build_football_field

        report.football_field = build_football_field(report)
    except Exception as exc:  # noqa: BLE001
        report.warnings.append(f"Football field failed: {exc}")
        report.football_field = _fallback_football_field(report)

    report.summary = _build_summary(report)
    return report


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
            rows.append(
                FootballFieldRow("Comps", s["low"], s.get("median", report.current_price), s["high"])
            )
    if report.ddm:
        ddm_mark = NOT_IN_BLEND if ddm_reference_only(report) else ""
        rows.append(band("DDM" + ddm_mark, report.ddm.implied_price, 0.10))
    if report.fcfe and (getattr(report.fcfe, "detail", None) or {}).get("valuation_available") is not False:
        rows.append(band("FCFE" + mark, report.fcfe.implied_price, 0.10))
    return [r for r in rows if r is not None]


def _build_summary(report: ValuationReport) -> dict:
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
        price;
      * a negative price (claims senior to common equity exceed the value)
        counts as 0, because equity cannot be worth less.
    A blend of 0 is therefore a real target (-100%, "Overvalued"); an empty
    blend has no target (upside None, "N/A"). One more case gets "N/A" although
    it has a target: a financial institution whose blend is the DDM alone (no
    comps). The DDM counts regular dividends only, not buybacks or special
    dividends, so on its own it reads a bank or insurer that also buys back
    stock as overvalued, and no second method checks it. The target is kept,
    the upside is None (so no downside is shown next to "N/A"), and a
    warning says why no verdict is given.

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
        if captive:
            warn(f"Captive finance arm: {FINANCIAL_KIND_REASONS[kind]}. An FCFF DCF and an "
                 "FCFE on the consolidated figures describe neither business, so both are "
                 "shown for reference only and left out of the blended target.")
        else:
            warn(f"{prefix} ({financial}): {FINANCIAL_KIND_REASONS[kind]}, so "
                 "an FCFF DCF and an FCFE do not describe the business. Both are shown for "
                 "reference only and left out of the blended target.")
    has_comps = "Comps (median)" in methods
    # Peers that ran but gave no positive price (a lessor's peers without P/E
    # or P/B) are not a missing input: the wording below then says so rather
    # than asking for peers (the reports show "n/a", not "supply peers").
    peers_gave_no_price = bool(getattr(report.comps, "peers", None)) and not has_comps
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
             "target, which it would otherwise set on its own: it counts regular dividends only "
             "(buybacks and special dividends are ignored) and no second method checks it.")

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
            excluded[name] = ("dividends only and no comps price from the peers supplied to "
                              "check it" if peers_gave_no_price else
                              "dividends only and no comps to check it; supply peers")
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
    # (``peers_gave_no_price`` above).
    ref_models = " and ".join(n for n in ("DCF", "FCFE") if n in methods)
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
        comps_text = ("the peers supplied give no usable comps price" if peers_gave_no_price
                      else "there are no comps")
        text = (f"No blended target: {'; '.join(parts)}, and {comps_text}." if parts
                else f"No blended target for {whom}: {comps_text}.")
        if not peers_gave_no_price:
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
        comps_text = ("no usable comps price from the peers supplied" if peers_gave_no_price
                      else "no comps")
        text = (f"Blended target rests on the DDM alone ({left}{comps_text}), so no verdict or "
                "upside is given: the DDM counts regular dividends only (buybacks and special "
                "dividends are ignored) and there is no second method to check it against.")
        if not peers_gave_no_price:
            text += " Supply peers to add trading comps."
        warn(text)
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
    }


def _recommendation(target: Optional[float], price: Optional[float]) -> str:
    """Verdict from the blended target: +/-15% bands; "N/A" when there is no
    target or no positive price (a 0.0 target is a valid -100% "Overvalued").
    ``_build_summary`` also withholds it (and the upside) for a financial
    institution's DDM-only blend."""
    if target is None or not is_num(target) or not is_num(price) or price <= 0:
        return "N/A"
    upside = target / price - 1.0
    if upside >= 0.15:
        return "Undervalued"
    if upside <= -0.15:
        return "Overvalued"
    return "Fairly valued"
