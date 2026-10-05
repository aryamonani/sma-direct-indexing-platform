"""
Backtest: run a client portfolio through historical prices, harvesting losses
along the way, and record what the account did and what it would have done
without harvesting.

The engine is intentionally rule-based rather than a true constrained optimizer.
That's a scoping decision I'd flag in the interview: production would use a
quadratic programming formulation (minimize tracking error subject to screens,
lot-level wash-sale constraints, and transaction cost penalties). The rule-based
version here captures the structure of the decision without the solver weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from .client import Client
from .data import universe_df
from .portfolio import (
    Portfolio,
    benchmark_weights,
    construct_portfolio_from_cash,
    portfolio_daily_returns,
    tracking_error,
    tracking_weights,
)
from .tax import (
    RealizedTrade,
    TaxLot,
    TaxYearResult,
    apply_loss_ordering,
    classify_realized_trades,
    is_wash_sale,
    select_lots_to_harvest,
)


@dataclass
class BacktestResult:
    """Outcome of running one client through a historical window."""
    client_id: str
    start_date: date
    end_date: date
    portfolio: Portfolio
    harvested_losses_by_year: dict[int, float] = field(default_factory=dict)
    tax_savings_by_year: dict[int, float] = field(default_factory=dict)
    harvest_events: list[dict] = field(default_factory=list)
    carryforward_losses: float = 0.0
    final_market_value: float = 0.0
    benchmark_final_value: float = 0.0
    tracking_error: float = 0.0
    daily_values: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    benchmark_daily_values: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    @property
    def total_harvested(self) -> float:
        return sum(self.harvested_losses_by_year.values())

    @property
    def total_tax_savings(self) -> float:
        return sum(self.tax_savings_by_year.values())

    @property
    def after_tax_excess_return(self) -> float:
        """
        After-tax excess return of SMA vs holding the benchmark.

        Simplified: (SMA final MV + cumulative tax savings) / initial - benchmark return.
        Annualized.
        """
        years = (self.end_date - self.start_date).days / 365.25
        if years <= 0:
            return 0.0
        initial = self.portfolio.aum_target
        sma_total = self.final_market_value + self.total_tax_savings
        sma_return = (sma_total / initial) ** (1 / years) - 1
        bench_return = (self.benchmark_final_value / initial) ** (1 / years) - 1
        return sma_return - bench_return


def run_backtest(
    client: Client,
    prices: pd.DataFrame,
    harvest_frequency_days: int = 30,
    min_loss_pct: float = 0.05,
    min_loss_dollars: float = 500.0,
) -> BacktestResult:
    """
    Run a client's SMA account through the backtest window.

    Steps:
    1. Build the initial tracking portfolio on day one, buying each name at
       benchmark weight (filtered by screens).
    2. On each harvest-check day, scan for lots at a loss. If a lot clears
       both the pct and dollar thresholds AND doesn't trigger a wash sale,
       sell the lot and immediately buy a replacement (another name in the
       same sector that we're not already holding at over-cap weight).
    3. Measure tax savings per year using the client's tax rates.
    4. Compare final after-tax value to a buy-and-hold benchmark of SPY.
    """
    priorities = client.priorities
    # Priorities tweak thresholds: a tracking-weight-heavy account harvests less.
    effective_min_loss_pct = min_loss_pct / max(0.01, priorities.harvest_weight)
    effective_min_loss_dollars = min_loss_dollars / max(0.01, priorities.harvest_weight)

    dates = prices.index
    start = dates[0].date()
    end = dates[-1].date()

    # Build initial portfolio on day one
    first_prices = prices.iloc[0]
    portfolio = construct_portfolio_from_cash(
        name=client.display_name,
        aum=client.aum,
        screens=client.screens,
        prices_on_date=first_prices,
        purchase_date=start,
    )

    # Buy-and-hold benchmark: same AUM in full universe at index weights
    bench_shares: dict[str, float] = {}
    bw = benchmark_weights()
    for ticker, w in bw.items():
        price = float(first_prices.get(ticker, 0.0))
        if price > 0:
            bench_shares[ticker] = (client.aum * w) / price

    harvest_events: list[dict] = []
    harvested_by_year: dict[int, float] = {}
    daily_values = []
    bench_values = []

    last_harvest_check = start
    for ts in dates:
        d = ts.date()
        price_row = prices.loc[ts]
        prices_today = {t: float(price_row[t]) for t in price_row.index if pd.notna(price_row[t])}

        # Harvest check every N days (not every day; realistic ops cadence)
        if (d - last_harvest_check).days >= harvest_frequency_days:
            last_harvest_check = d
            # Only harvest for tax-sensitive clients
            if client.tax_rates.st_total > 0.05:
                _scan_and_harvest(
                    portfolio=portfolio,
                    prices_today=prices_today,
                    sale_date=d,
                    min_loss_pct=effective_min_loss_pct,
                    min_loss_dollars=effective_min_loss_dollars,
                    harvest_events=harvest_events,
                    harvested_by_year=harvested_by_year,
                )

        # Record daily market values
        daily_values.append(portfolio.market_value(prices_today))
        bench_mv = sum(
            shares * prices_today.get(t, 0.0) for t, shares in bench_shares.items()
        )
        bench_values.append(bench_mv)

    final_prices = {t: float(prices.iloc[-1][t]) for t in prices.columns if pd.notna(prices.iloc[-1][t])}
    final_mv = portfolio.market_value(final_prices)
    bench_final = sum(
        shares * final_prices.get(t, 0.0) for t, shares in bench_shares.items()
    )

    # Compute tax savings per year using IRS loss ordering
    tax_savings_by_year: dict[int, float] = {}
    years = sorted({t.sale_date.year for t in portfolio.realized_trades})
    prior_st_carry = prior_lt_carry = 0.0
    for year in years:
        st_g, st_l, lt_g, lt_l = classify_realized_trades(portfolio.realized_trades, year)
        # Assume client has ~2% of AUM in outside realized gains each year
        outside_gains = client.aum * 0.02
        st_g += outside_gains * 0.3
        lt_g += outside_gains * 0.7
        result = apply_loss_ordering(
            st_gains=st_g, st_losses=st_l, lt_gains=lt_g, lt_losses=lt_l,
            rates=client.tax_rates,
            prior_st_carry=prior_st_carry, prior_lt_carry=prior_lt_carry,
        )
        tax_savings_by_year[year] = result.tax_savings
        prior_st_carry = result.carryforward_st
        prior_lt_carry = result.carryforward_lt

    # Compute realized tracking error over the window
    daily_series = pd.Series(daily_values, index=dates)
    bench_series = pd.Series(bench_values, index=dates)
    sma_rets = daily_series.pct_change().dropna()
    bench_rets = bench_series.pct_change().dropna()
    diff = (sma_rets - bench_rets).dropna()
    te = float(diff.std() * np.sqrt(252)) if len(diff) > 1 else 0.0

    return BacktestResult(
        client_id=client.client_id,
        start_date=start,
        end_date=end,
        portfolio=portfolio,
        harvested_losses_by_year=harvested_by_year,
        tax_savings_by_year=tax_savings_by_year,
        harvest_events=harvest_events,
        carryforward_losses=prior_st_carry + prior_lt_carry,
        final_market_value=final_mv,
        benchmark_final_value=bench_final,
        tracking_error=te,
        daily_values=daily_series,
        benchmark_daily_values=bench_series,
    )


def _scan_and_harvest(
    portfolio: Portfolio,
    prices_today: dict[str, float],
    sale_date: date,
    min_loss_pct: float,
    min_loss_dollars: float,
    harvest_events: list[dict],
    harvested_by_year: dict[int, float],
) -> None:
    """Scan every held position; sell loss lots and buy replacements."""
    univ = universe_df()
    tickers_held = list(portfolio.lots.keys())

    for ticker in tickers_held:
        price = prices_today.get(ticker, 0.0)
        if price <= 0:
            continue
        lots = list(portfolio.lots[ticker])
        candidates = select_lots_to_harvest(
            lots=lots,
            current_price=price,
            as_of=sale_date,
            min_loss_pct=min_loss_pct,
            min_loss_dollars=min_loss_dollars,
            purchase_history=portfolio.purchase_history[-200:],
        )
        for lot in candidates:
            trade = portfolio.sell_lot(lot, price, sale_date)
            harvested_by_year[sale_date.year] = (
                harvested_by_year.get(sale_date.year, 0.0) + (-trade.pnl)
            )
            # Pick a replacement: a same-sector name that's screen-allowed and
            # we're not currently overweight in. The replacement must not have
            # been bought in the last 30 days (would still trigger wash on the
            # OTHER ticker if the engine treated them as substantially identical;
            # we treat distinct stocks as not substantially identical, matching
            # typical SMA practice).
            sector = univ.loc[ticker, "sector"] if ticker in univ.index else None
            replacement = _pick_replacement(
                portfolio=portfolio,
                sold_ticker=ticker,
                sector=sector,
                prices_today=prices_today,
            )
            if replacement:
                rp = prices_today[replacement]
                rshares = trade.proceeds / rp
                new_lot = TaxLot(
                    ticker=replacement,
                    shares=rshares,
                    cost_basis_per_share=rp,
                    purchase_date=sale_date,
                )
                portfolio.add_lot(new_lot)
                portfolio.cash -= rshares * rp
            harvest_events.append({
                "date": sale_date.isoformat(),
                "sold": ticker,
                "sold_shares": lot.shares,
                "loss_realized": -trade.pnl,
                "replacement": replacement or "(cash held)",
            })


def _pick_replacement(
    portfolio: Portfolio,
    sold_ticker: str,
    sector: Optional[str],
    prices_today: dict[str, float],
) -> Optional[str]:
    """
    Pick a sector-matched replacement that is screen-allowed and not currently
    held at a dominant weight.
    """
    univ = universe_df()
    allowed = set(portfolio.screens.allowed_universe())
    candidates = []
    for t in univ.index:
        if t == sold_ticker:
            continue
        if t not in allowed:
            continue
        if univ.loc[t, "sector"] != sector:
            continue
        if prices_today.get(t, 0.0) <= 0:
            continue
        candidates.append(t)
    if not candidates:
        # Fall back to any allowed name
        candidates = [
            t for t in allowed
            if t != sold_ticker and prices_today.get(t, 0.0) > 0
        ]
    if not candidates:
        return None
    # Prefer the lowest-weight current holding (room to add)
    weights = portfolio.weights(prices_today)
    candidates.sort(key=lambda t: weights.get(t, 0.0))
    return candidates[0]
