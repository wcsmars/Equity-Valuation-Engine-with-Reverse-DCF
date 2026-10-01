"""Deterministic synthetic company for offline runs.

``SyntheticProvider`` implements the ``DataProvider`` interface without any
network access. It serves one hand-built company (ticker ``SYNT``: a steady
grower with modest leverage and a 30% dividend payout) and three peer rows, so
DCF, comps, DDM and FCFE all have inputs. The CLI's ``--demo`` flag and the
offline tests use it to run every model and exporter without EDGAR or yfinance.
"""

from __future__ import annotations

from ..schemas import (
    AnnualFinancials,
    BalanceSheetSnapshot,
    CompanyData,
    CompRow,
    MarketData,
)
from .base import DataProvider

DEMO_TICKER = "SYNT"
DEMO_PEERS = ["PEER1", "PEER2", "PEER3"]

# (EV/EBITDA, EV/Sales, P/E, P/B, PEG) for each synthetic peer.
_PEER_MULTIPLES = [
    (18.0, 4.0, 19.0, 5.0, 1.6),
    (22.0, 5.5, 24.0, 6.5, 2.1),
    (20.0, 4.8, 21.0, 5.8, 1.8),
]


def _ramp(start: float, growth: float, n: int) -> list[float]:
    return [start * (1 + growth) ** i for i in range(n)]


def make_company() -> CompanyData:
    """Five fiscal years of a $80B-revenue firm growing 8% a year.

    25% EBIT margin, 21% tax, $20B debt against $8B cash, 30% payout, and a
    share price set at 20x trailing earnings.
    """
    years = [2020, 2021, 2022, 2023, 2024]
    rev = _ramp(80_000_000_000.0, 0.08, 5)          # $80B growing 8%/yr
    ebit = [r * 0.25 for r in rev]                  # 25% EBIT margin
    da = [r * 0.04 for r in rev]
    ebitda = [e + d for e, d in zip(ebit, da)]
    pretax = [e * 0.95 for e in ebit]               # small interest drag
    tax = [p * 0.21 for p in pretax]
    ni = [p - t for p, t in zip(pretax, tax)]
    capex = [r * 0.05 for r in rev]
    dnwc = [r * 0.01 for r in rev]
    interest = [e * 0.05 for e in ebit]
    div = [n * 0.30 for n in ni]                    # 30% payout
    shares = [10_000_000_000.0] * 5

    fin = AnnualFinancials(
        fiscal_years=years, revenue=rev, ebit=ebit, ebitda=ebitda, net_income=ni,
        dep_amort=da, capex=capex, change_in_nwc=dnwc, interest_expense=interest,
        tax_expense=tax, pretax_income=pretax, dividends_paid=div, diluted_shares=shares,
    )
    bs = BalanceSheetSnapshot(
        as_of="2024-12-31", total_debt=20_000_000_000.0,
        cash_and_investments=8_000_000_000.0, total_equity=60_000_000_000.0,
    )
    eps = ni[-1] / shares[-1]
    price = eps * 20.0                              # ~20x trailing P/E
    mkt = MarketData(
        ticker=DEMO_TICKER, name="Synthetic Corp", currency="USD", price=price,
        shares_outstanding=shares[-1], market_cap=price * shares[-1], beta=1.1,
        dividend_per_share=div[-1] / shares[-1], fifty_two_week_low=price * 0.8,
        fifty_two_week_high=price * 1.25, sector="Technology", industry="Software",
    )
    return CompanyData(ticker=DEMO_TICKER, name="Synthetic Corp", cik=None,
                       financials=fin, balance_sheet=bs, market=mkt,
                       source_notes=["Synthetic demo data: no live market or filing data."])


class SyntheticProvider(DataProvider):
    """Offline DataProvider: the synthetic company plus three synthetic peers.

    Any requested ticker resolves to the synthetic company, and peer tickers are
    paired in order with the fixed multiples above.
    """

    def get_company_data(self, ticker: str) -> CompanyData:
        return make_company()

    def get_market_data(self, ticker: str) -> MarketData:
        return make_company().market

    def suggest_peers(self, ticker: str) -> list[str]:
        return list(DEMO_PEERS)

    def get_peer_comp_rows(self, tickers: list[str]) -> list[CompRow]:
        return [
            CompRow(ticker=tk, name=tk, market_cap=5e10, enterprise_value=5.2e10,
                    ev_ebitda=eve, ev_sales=evs, pe=pe, pb=pb, peg=peg, currency="USD")
            for tk, (eve, evs, pe, pb, peg) in zip(tickers, _PEER_MULTIPLES)
        ]
