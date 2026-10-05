"""
Portfolio: holdings, lots, tracking-error measurement, and benchmark construction.

Design:
- Each client Portfolio holds a list of TaxLots per ticker (not just an aggregate
  share count). This is required for correct tax accounting: when we sell, we
  pick specific lots and realize their individual cost bases.
- The benchmark is the full universe weighted by index weights. The portfolio
  tracks the benchmark via re-weighted holdings in the allowed universe.
- Tracking error is annualized realized std dev of (portfolio return minus
  benchmark return), computed from historical price data.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from .data import SP500_UNIVERSE, universe_df
from .screens import ScreenSet
from .tax import RealizedTrade, TaxLot, is_wash_sale


@dataclass
class Portfolio:
    """
    A single client account.

    Holds lot-level positions, a cash balance, a realized-trade history, and
    a recent-purchase log used by the wash sale rule.
    """
    name: str
    aum_target: float                    # how much we're managing
    screens: ScreenSet = field(default_factory=ScreenSet)
    lots: dict[str, list[TaxLot]] = field(default_factory=lambda: defaultdict(list))
    cash: float = 0.0
    realized_trades: list[RealizedTrade] = field(default_factory=list)
    purchase_history: list[tuple[str, date]] = field(default_factory=list)

    def add_lot(self, lot: TaxLot) -> None:
        self.lots[lot.ticker].append(lot)
        self.purchase_history.append((lot.ticker, lot.purchase_date))

    def shares(self, ticker: str) -> float:
        return sum(l.shares for l in self.lots.get(ticker, []))

    def cost_basis(self, ticker: str) -> float:
        return sum(l.cost_basis for l in self.lots.get(ticker, []))

    def market_value(self, prices: dict[str, float]) -> float:
        total = self.cash
        for ticker, lots in self.lots.items():
            price = prices.get(ticker, 0.0)
            total += sum(l.market_value(price) for l in lots)
        return total

    def weights(self, prices: dict[str, float]) -> dict[str, float]:
        """Current portfolio weights by ticker (ex-cash)."""
        mv = self.market_value(prices) - self.cash
        if mv <= 0:
            return {}
        return {
            t: sum(l.market_value(prices.get(t, 0.0)) for l in lots) / mv
            for t, lots in self.lots.items()
            if lots
        }

    def unrealized_pnl(self, prices: dict[str, float]) -> float:
        total = 0.0
        for ticker, lots in self.lots.items():
            price = prices.get(ticker, 0.0)
            for lot in lots:
                total += lot.unrealized_pnl(price)
        return total

    def sell_lot(self, lot: TaxLot, price: float, sale_date: date) -> RealizedTrade:
        """Fully liquidate a specific lot."""
        proceeds = lot.shares * price
        trade = RealizedTrade(
            ticker=lot.ticker,
            shares=lot.shares,
            cost_basis=lot.cost_basis,
            proceeds=proceeds,
            purchase_date=lot.purchase_date,
            sale_date=sale_date,
        )
        self.lots[lot.ticker].remove(lot)
        self.cash += proceeds
        self.realized_trades.append(trade)
        return trade

    def summary_row(self, prices: dict[str, float], as_of: date) -> dict:
        mv = self.market_value(prices)
        upnl = self.unrealized_pnl(prices)
        ytd_realized = sum(
            t.pnl for t in self.realized_trades if t.sale_date.year == as_of.year
        )
        harvested_ytd = sum(
            -t.pnl for t in self.realized_trades
            if t.sale_date.year == as_of.year and t.pnl < 0
        )
        return {
            "name": self.name,
            "market_value": mv,
            "unrealized_pnl": upnl,
            "ytd_realized_pnl": ytd_realized,
            "ytd_harvested_losses": harvested_ytd,
            "n_positions": sum(1 for lots in self.lots.values() if lots),
            "screens": "; ".join(self.screens.description()),
        }


# ---------------------------------------------------------------------------
# Benchmark construction
# ---------------------------------------------------------------------------

def benchmark_weights() -> pd.Series:
    """Full-universe weights, our S&P 500 proxy."""
    u = universe_df()
    return u["weight"]


def tracking_weights(screens: ScreenSet) -> pd.Series:
    """
    Weights for the tracking portfolio given a client's screens.

    Policy: start from benchmark weights, drop excluded names, redistribute
    their weight to the remaining names in the SAME sector first, then across
    the whole surviving universe. This keeps the sector profile close to the
    benchmark, which is the main driver of tracking error.
    """
    u = universe_df()
    bench = u["weight"].copy()
    allowed = set(screens.allowed_universe())

    weights = bench.copy()
    excluded_weight_by_sector: dict[str, float] = defaultdict(float)
    for ticker in u.index:
        if ticker not in allowed:
            excluded_weight_by_sector[u.loc[ticker, "sector"]] += weights[ticker]
            weights[ticker] = 0.0

    # Redistribute within-sector where possible
    for sector, excess in excluded_weight_by_sector.items():
        sector_survivors = [
            t for t in u.index
            if u.loc[t, "sector"] == sector and t in allowed and weights[t] > 0
        ]
        if sector_survivors:
            total = sum(weights[t] for t in sector_survivors)
            for t in sector_survivors:
                weights[t] += excess * (weights[t] / total)
        else:
            # Entire sector excluded: redistribute across all survivors
            all_survivors = [t for t in allowed if weights[t] > 0]
            total = sum(weights[t] for t in all_survivors)
            if total > 0:
                for t in all_survivors:
                    weights[t] += excess * (weights[t] / total)

    # Apply max-weight cap if set
    if screens.max_single_weight is not None:
        cap = screens.max_single_weight
        over = weights > cap
        while over.any():
            excess = (weights[over] - cap).sum()
            weights[over] = cap
            under = (weights < cap) & (weights > 0)
            if not under.any():
                break
            under_total = weights[under].sum()
            weights[under] += excess * weights[under] / under_total
            over = weights > cap

    # Normalize to exactly 1.0 after rounding drift
    total = weights.sum()
    if total > 0:
        weights = weights / total
    return weights


def construct_portfolio_from_cash(
    name: str,
    aum: float,
    screens: ScreenSet,
    prices_on_date: pd.Series,
    purchase_date: date,
) -> Portfolio:
    """Build a brand-new portfolio from cash by buying the tracking weights."""
    port = Portfolio(name=name, aum_target=aum, screens=screens, cash=aum)
    weights = tracking_weights(screens)
    for ticker, w in weights.items():
        if w <= 0:
            continue
        dollars = aum * w
        price = float(prices_on_date.get(ticker, 0.0))
        if price <= 0:
            continue
        shares = dollars / price
        lot = TaxLot(
            ticker=ticker,
            shares=shares,
            cost_basis_per_share=price,
            purchase_date=purchase_date,
        )
        port.add_lot(lot)
        port.cash -= shares * price
    return port


# ---------------------------------------------------------------------------
# Tracking error
# ---------------------------------------------------------------------------

def portfolio_daily_returns(
    weights: pd.Series, returns: pd.DataFrame
) -> pd.Series:
    """Weighted daily return of a static-weight portfolio."""
    cols = [c for c in weights.index if c in returns.columns]
    w = weights.reindex(cols).fillna(0.0)
    r = returns[cols].fillna(0.0)
    return r.dot(w)


def tracking_error(
    port_weights: pd.Series,
    bench_weights: pd.Series,
    returns: pd.DataFrame,
    annualize: bool = True,
) -> float:
    """
    Annualized tracking error: std dev of (portfolio return minus benchmark return).

    This is the simplest common definition. Production SMAs use ex-ante tracking
    error from a factor risk model (Barra). This is the ex-post / realized version,
    which is honest and reproducible for a portfolio project.
    """
    port_r = portfolio_daily_returns(port_weights, returns)
    bench_r = portfolio_daily_returns(bench_weights, returns)
    diff = port_r - bench_r
    te = diff.std()
    if annualize:
        te = te * np.sqrt(252)
    return float(te)
