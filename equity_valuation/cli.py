"""Command-line interface.

    python -m equity_valuation AAPL
    python -m equity_valuation AAPL --peers MSFT,GOOGL,META --out output
    python -m equity_valuation --demo          # synthetic company, no network
    python -m equity_valuation MSFT --rf 0.043 --erp 0.05 --terminal-growth 0.025 \
        --forecast-years 6 --no-ddm --excel --html
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys

from . import config
from .schemas import DCFAssumptions, DDMAssumptions, MacroAssumptions


def _fmt_money(x, sym="$"):
    if x is None:
        return "n/a"
    return f"{sym}{x:,.2f}"


def _fmt_pct(x):
    if x is None:
        return "n/a"
    return f"{x * 100:+.1f}%"


# Matches the dashboard API, which clamps forecast_years to this range.
MAX_FORECAST_YEARS = 15


def _number(text: str) -> float:
    try:
        x = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid number: {text!r}") from None
    if not math.isfinite(x):
        raise argparse.ArgumentTypeError(f"must be a finite number, got {text!r}")
    return x


def _rate(minimum: float = -1.0):
    """argparse type for a decimal rate: finite, |x| < 1 and x >= `minimum`."""

    def parse(text: str) -> float:
        x = _number(text)
        if abs(x) >= 1.0:
            raise argparse.ArgumentTypeError(
                f"{text} looks like a percentage; rates are decimals "
                f"(use {x / 100:g} for {x:g}%)"
            )
        if x < minimum:
            raise argparse.ArgumentTypeError(f"must be >= {minimum:g}, got {text}")
        return x

    parse.__name__ = "rate"  # argparse names the type in its error messages
    return parse


def _positive(text: str) -> float:
    x = _number(text)
    if x <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {text}")
    return x


def _forecast_years(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid integer: {text!r}") from None
    if not 1 <= n <= MAX_FORECAST_YEARS:
        raise argparse.ArgumentTypeError(
            f"must be between 1 and {MAX_FORECAST_YEARS}, got {text}"
        )
    return n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="equity_valuation",
        description="Automated equity valuation: DCF, comps, DDM/FCFE, sensitivity "
        "-> Excel + HTML. Rates are decimals: --rf 0.043 means 4.3%.",
        epilog="Exit status: 0 on success, 1 if the valuation or an export failed, "
        "2 for invalid arguments.",
    )
    p.add_argument("ticker", nargs="?", help="Target ticker, e.g. AAPL")
    p.add_argument(
        "--demo",
        action="store_true",
        help="Value the built-in synthetic company (SYNT) with three synthetic "
        "peers, offline. Takes no ticker.",
    )
    p.add_argument(
        "--peers",
        default=None,
        help="Comma-separated peer tickers for comps (comps are skipped if omitted).",
    )
    p.add_argument("--out", default=config.DEFAULT_OUTPUT_DIR, help="Output directory.")

    # Macro / CAPM
    p.add_argument("--rf", type=_rate(), default=config.DEFAULT_RISK_FREE_RATE,
                   help="Risk-free rate (decimal, e.g. 0.043 for 4.3%%).")
    p.add_argument("--erp", type=_rate(0.0), default=config.DEFAULT_EQUITY_RISK_PREMIUM,
                   help="Equity risk premium (decimal).")
    p.add_argument("--tax", type=_rate(0.0), default=None,
                   help="Marginal tax rate (decimal, e.g. 0.21); default derives effective.")
    p.add_argument("--cost-of-debt", type=_rate(0.0), default=None,
                   help="Pre-tax cost of debt (decimal); default derived.")

    # DCF (--forecast-years also sets the FCFE horizon; --terminal-growth is
    # shared by the DCF, DDM and FCFE, as in the dashboard)
    p.add_argument("--forecast-years", type=_forecast_years,
                   default=config.DEFAULT_FORECAST_YEARS,
                   help=f"Explicit forecast years for the DCF and FCFE "
                   f"(1-{MAX_FORECAST_YEARS}).")
    p.add_argument("--terminal-growth", type=_rate(), default=config.DEFAULT_TERMINAL_GROWTH,
                   help="Long-run growth rate (decimal) for the DCF, DDM and FCFE "
                   "terminal values.")
    p.add_argument("--terminal-method", choices=["gordon", "exit_multiple"], default="gordon")
    p.add_argument("--exit-ev-ebitda", type=_positive, default=None,
                   help="Exit EV/EBITDA multiple (required if terminal-method=exit_multiple).")
    p.add_argument("--target-ebit-margin", type=_rate(), default=None,
                   help="Terminal EBIT margin (decimal, e.g. 0.25) to fade toward.")

    # Toggles
    p.add_argument("--no-dcf", action="store_true")
    p.add_argument("--no-comps", action="store_true")
    p.add_argument("--no-ddm", action="store_true")
    p.add_argument("--no-fcfe", action="store_true")
    p.add_argument("--no-sensitivity", action="store_true")

    # Output formats
    p.add_argument("--excel", action="store_true", help="Write Excel (default: on if neither flag).")
    p.add_argument("--html", action="store_true", help="Write HTML (default: on if neither flag).")
    p.add_argument("--quiet", action="store_true", help="Suppress the console summary.")
    return p


def _print_summary(report) -> None:
    s = report.summary
    sym = config.CURRENCY_SYMBOLS.get(s.get("currency"), "")
    line = "=" * 64
    print(line)
    print(f"  {s['name']}  ({s['ticker']})")
    print(f"  Current price: {_fmt_money(s['current_price'], sym)}   "
          f"Recommendation: {s['recommendation']}")
    print(line)
    # Methods shown for reference but left out of the blend (e.g. a bank's DCF)
    # are marked; the reason is in the notes below. The name column widens past
    # its usual 22 characters only for a long marked name ("Comps (median)
    # (excluded)"), so the price and upside columns stay aligned.
    excluded = s.get("excluded_from_blend") or {}
    labels = {name: (f"{name} (excluded)" if name in excluded else name) for name in s["methods"]}
    width = max([22] + [len(label) for label in labels.values()])
    extra = width - 22
    print("  Method" + " " * (17 + extra) + "Implied price      Upside")
    print("  " + "-" * (50 + extra))
    for name, price in s["methods"].items():
        up = (price / s["current_price"] - 1.0) if (price and s["current_price"]) else None
        print(f"  {labels[name]:<{width}} {_fmt_money(price, sym):>14}   {_fmt_pct(up):>8}")
    print("  " + "-" * (50 + extra))
    print(f"  {'Blended target':<{width}} {_fmt_money(s['blended_target'], sym):>14}   "
          f"{_fmt_pct(s['blended_upside']):>8}")
    print(line)
    if report.warnings:
        print("  Notes:")
        for w in report.warnings:
            print(f"    - {w}")
        print(line)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.terminal_method == "exit_multiple" and args.exit_ev_ebitda is None:
        parser.error("--terminal-method exit_multiple requires --exit-ev-ebitda.")
    if args.exit_ev_ebitda is not None and args.terminal_method != "exit_multiple":
        print("note: --exit-ev-ebitda is ignored unless --terminal-method exit_multiple.",
              file=sys.stderr)
    if os.path.exists(args.out) and not os.path.isdir(args.out):
        parser.error(f"--out {args.out} exists and is not a directory.")

    provider = None
    if args.demo:
        from .data.synthetic import DEMO_PEERS, DEMO_TICKER, SyntheticProvider

        if args.ticker and args.ticker.strip().upper() != DEMO_TICKER:
            parser.error("--demo values the synthetic company; omit the ticker.")
        args.ticker = DEMO_TICKER
        provider = SyntheticProvider()
        if not args.peers:
            args.peers = ",".join(DEMO_PEERS)
        elif not {t.strip().upper() for t in args.peers.split(",") if t.strip()} \
                <= set(DEMO_PEERS):
            # The synthetic provider would attach made-up multiples to real tickers.
            parser.error(f"--demo uses its synthetic peers ({','.join(DEMO_PEERS)}); "
                         "omit --peers.")
    elif not args.ticker:
        parser.error("a ticker is required (or use --demo for the offline example).")

    macro = MacroAssumptions(
        risk_free_rate=args.rf,
        equity_risk_premium=args.erp,
        tax_rate=args.tax,
        pretax_cost_of_debt=args.cost_of_debt,
    )
    dcf_assumptions = DCFAssumptions(
        forecast_years=args.forecast_years,
        terminal_growth=args.terminal_growth,
        terminal_method=args.terminal_method,
        exit_ev_ebitda=args.exit_ev_ebitda,
        target_ebit_margin=args.target_ebit_margin,
        tax_rate=args.tax,
    )
    # Same terminal growth for DDM/FCFE as for the DCF, as the dashboard does.
    ddm_assumptions = DDMAssumptions(
        forecast_years=args.forecast_years, terminal_growth=args.terminal_growth
    )
    peers = [t.strip().upper() for t in args.peers.split(",")] if args.peers else None

    from .engine import value_company

    try:
        report = value_company(
            args.ticker,
            provider=provider,
            macro=macro,
            dcf_assumptions=dcf_assumptions,
            ddm_assumptions=ddm_assumptions,
            peers=peers,
            run_dcf=not args.no_dcf,
            run_comps=not args.no_comps,
            run_ddm=not args.no_ddm,
            run_fcfe=not args.no_fcfe,
            run_sensitivity=not args.no_sensitivity,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not value {args.ticker}: {exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        _print_summary(report)

    # Default: write both if neither flag is given.
    want_excel = args.excel or not (args.excel or args.html)
    want_html = args.html or not (args.excel or args.html)
    try:
        os.makedirs(args.out, exist_ok=True)
    except OSError as exc:
        print(f"ERROR: cannot create output directory {args.out}: {exc}", file=sys.stderr)
        return 1
    # Same sanitizing as the dashboard's exports: a ticker like BRK/B must not
    # turn into a sub-directory of the output path.
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", report.company.ticker or "report")
    failed = False

    if want_excel:
        try:
            from .report.excel import write_excel

            path = write_excel(report, os.path.join(args.out, f"{stem}_valuation.xlsx"))
            print(f"  Excel : {path}")
        except Exception as exc:  # noqa: BLE001
            print(f"  Excel export failed: {exc}", file=sys.stderr)
            failed = True

    if want_html:
        try:
            from .report.html import write_html

            path = write_html(report, os.path.join(args.out, f"{stem}_valuation.html"))
            print(f"  HTML  : {path}")
        except Exception as exc:  # noqa: BLE001
            print(f"  HTML export failed: {exc}", file=sys.stderr)
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
