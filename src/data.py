"""
Data layer: S&P 500 universe, sector classification, and historical prices.

Design decisions:
- Universe is a static curated subset of the S&P 500 (50 names across 11 GICS
  sectors). Full 500 is unnecessary for a portfolio project and triples yfinance
  calls. The simplified universe keeps the demo fast while preserving sector
  diversity needed to show tracking error vs the index.
- Prices are fetched from yfinance and cached to parquet on disk. If network is
  unavailable, we fall back to deterministic synthetic prices so the Streamlit
  app always runs. In production, this would be a market data vendor (Bloomberg,
  Refinitiv) accessed through a licensed feed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# A curated 50-name slice of the S&P 500 across all 11 GICS sectors.
# Weights are approximate index weights normalized to sum to 1.0 within the slice.
# In production this would come from an index provider (S&P Global).
SP500_UNIVERSE: dict[str, dict] = {
    # Information Technology
    "AAPL": {"name": "Apple", "sector": "Information Technology", "sub_industry": "Technology Hardware", "weight": 0.070},
    "MSFT": {"name": "Microsoft", "sector": "Information Technology", "sub_industry": "Systems Software", "weight": 0.065},
    "NVDA": {"name": "NVIDIA", "sector": "Information Technology", "sub_industry": "Semiconductors", "weight": 0.055},
    "AVGO": {"name": "Broadcom", "sector": "Information Technology", "sub_industry": "Semiconductors", "weight": 0.020},
    "ORCL": {"name": "Oracle", "sector": "Information Technology", "sub_industry": "Systems Software", "weight": 0.015},
    "CRM": {"name": "Salesforce", "sector": "Information Technology", "sub_industry": "Application Software", "weight": 0.012},
    "ADBE": {"name": "Adobe", "sector": "Information Technology", "sub_industry": "Application Software", "weight": 0.010},
    # Communication Services
    "GOOGL": {"name": "Alphabet", "sector": "Communication Services", "sub_industry": "Interactive Media", "weight": 0.040},
    "META": {"name": "Meta Platforms", "sector": "Communication Services", "sub_industry": "Interactive Media", "weight": 0.030},
    "NFLX": {"name": "Netflix", "sector": "Communication Services", "sub_industry": "Entertainment", "weight": 0.012},
    "DIS": {"name": "Walt Disney", "sector": "Communication Services", "sub_industry": "Entertainment", "weight": 0.008},
    # Consumer Discretionary
    "AMZN": {"name": "Amazon", "sector": "Consumer Discretionary", "sub_industry": "Broadline Retail", "weight": 0.045},
    "TSLA": {"name": "Tesla", "sector": "Consumer Discretionary", "sub_industry": "Automobiles", "weight": 0.025},
    "HD": {"name": "Home Depot", "sector": "Consumer Discretionary", "sub_industry": "Home Improvement Retail", "weight": 0.012},
    "MCD": {"name": "McDonald's", "sector": "Consumer Discretionary", "sub_industry": "Restaurants", "weight": 0.008},
    "NKE": {"name": "Nike", "sector": "Consumer Discretionary", "sub_industry": "Footwear", "weight": 0.005},
    # Consumer Staples
    "WMT": {"name": "Walmart", "sector": "Consumer Staples", "sub_industry": "Consumer Staples Merchandise Retail", "weight": 0.015},
    "PG": {"name": "Procter & Gamble", "sector": "Consumer Staples", "sub_industry": "Household Products", "weight": 0.012},
    "KO": {"name": "Coca-Cola", "sector": "Consumer Staples", "sub_industry": "Soft Drinks", "weight": 0.010},
    "PEP": {"name": "PepsiCo", "sector": "Consumer Staples", "sub_industry": "Soft Drinks", "weight": 0.009},
    "COST": {"name": "Costco", "sector": "Consumer Staples", "sub_industry": "Consumer Staples Merchandise Retail", "weight": 0.012},
    "MO": {"name": "Altria", "sector": "Consumer Staples", "sub_industry": "Tobacco", "weight": 0.004},
    "PM": {"name": "Philip Morris", "sector": "Consumer Staples", "sub_industry": "Tobacco", "weight": 0.006},
    # Financials
    "JPM": {"name": "JPMorgan Chase", "sector": "Financials", "sub_industry": "Diversified Banks", "weight": 0.030},
    "BAC": {"name": "Bank of America", "sector": "Financials", "sub_industry": "Diversified Banks", "weight": 0.015},
    "WFC": {"name": "Wells Fargo", "sector": "Financials", "sub_industry": "Diversified Banks", "weight": 0.010},
    "GS": {"name": "Goldman Sachs", "sector": "Financials", "sub_industry": "Investment Banking", "weight": 0.008},
    "V": {"name": "Visa", "sector": "Financials", "sub_industry": "Transaction & Payment Processing", "weight": 0.018},
    "MA": {"name": "Mastercard", "sector": "Financials", "sub_industry": "Transaction & Payment Processing", "weight": 0.015},
    "BRK-B": {"name": "Berkshire Hathaway", "sector": "Financials", "sub_industry": "Multi-Sector Holdings", "weight": 0.020},
    # Health Care
    "UNH": {"name": "UnitedHealth Group", "sector": "Health Care", "sub_industry": "Managed Health Care", "weight": 0.025},
    "JNJ": {"name": "Johnson & Johnson", "sector": "Health Care", "sub_industry": "Pharmaceuticals", "weight": 0.018},
    "LLY": {"name": "Eli Lilly", "sector": "Health Care", "sub_industry": "Pharmaceuticals", "weight": 0.020},
    "PFE": {"name": "Pfizer", "sector": "Health Care", "sub_industry": "Pharmaceuticals", "weight": 0.010},
    "ABBV": {"name": "AbbVie", "sector": "Health Care", "sub_industry": "Biotechnology", "weight": 0.012},
    "MRK": {"name": "Merck", "sector": "Health Care", "sub_industry": "Pharmaceuticals", "weight": 0.012},
    # Industrials
    "CAT": {"name": "Caterpillar", "sector": "Industrials", "sub_industry": "Construction Machinery", "weight": 0.008},
    "BA": {"name": "Boeing", "sector": "Industrials", "sub_industry": "Aerospace & Defense", "weight": 0.006},
    "LMT": {"name": "Lockheed Martin", "sector": "Industrials", "sub_industry": "Aerospace & Defense", "weight": 0.005},
    "RTX": {"name": "RTX Corp", "sector": "Industrials", "sub_industry": "Aerospace & Defense", "weight": 0.006},
    "GE": {"name": "GE Aerospace", "sector": "Industrials", "sub_industry": "Aerospace & Defense", "weight": 0.006},
    "UNP": {"name": "Union Pacific", "sector": "Industrials", "sub_industry": "Rail Transportation", "weight": 0.005},
    # Energy
    "XOM": {"name": "ExxonMobil", "sector": "Energy", "sub_industry": "Integrated Oil & Gas", "weight": 0.015},
    "CVX": {"name": "Chevron", "sector": "Energy", "sub_industry": "Integrated Oil & Gas", "weight": 0.010},
    "COP": {"name": "ConocoPhillips", "sector": "Energy", "sub_industry": "Oil & Gas E&P", "weight": 0.006},
    # Utilities
    "NEE": {"name": "NextEra Energy", "sector": "Utilities", "sub_industry": "Electric Utilities", "weight": 0.006},
    "DUK": {"name": "Duke Energy", "sector": "Utilities", "sub_industry": "Electric Utilities", "weight": 0.004},
    # Real Estate
    "PLD": {"name": "Prologis", "sector": "Real Estate", "sub_industry": "Industrial REITs", "weight": 0.004},
    "AMT": {"name": "American Tower", "sector": "Real Estate", "sub_industry": "Telecom Tower REITs", "weight": 0.005},
    # Materials
    "LIN": {"name": "Linde", "sector": "Materials", "sub_industry": "Industrial Gases", "weight": 0.005},
    "SHW": {"name": "Sherwin-Williams", "sector": "Materials", "sub_industry": "Specialty Chemicals", "weight": 0.003},
}


def universe_df() -> pd.DataFrame:
    """Return the universe as a DataFrame indexed by ticker."""
    df = pd.DataFrame.from_dict(SP500_UNIVERSE, orient="index")
    # Normalize weights to sum to exactly 1.0 within our slice
    df["weight"] = df["weight"] / df["weight"].sum()
    df.index.name = "ticker"
    return df


def sectors() -> list[str]:
    """All GICS sectors represented in our universe."""
    return sorted(set(v["sector"] for v in SP500_UNIVERSE.values()))


def sub_industries() -> list[str]:
    """All sub-industries represented in our universe."""
    return sorted(set(v["sub_industry"] for v in SP500_UNIVERSE.values()))


# -----------------------------------------------------------------------------
# Price data
# -----------------------------------------------------------------------------

def _cache_path(start: date, end: date) -> Path:
    return DATA_DIR / f"prices_{start.isoformat()}_{end.isoformat()}.parquet"


def _synthetic_prices(tickers: list[str], start: date, end: date, seed: int = 42) -> pd.DataFrame:
    """
    Deterministic synthetic price history used when yfinance is unreachable.

    We generate a daily geometric Brownian motion per ticker with sector-level
    correlation injected so sector exclusions move tracking error realistically.
    Includes a COVID-style drawdown in Mar 2020 and a 2022 bear market so TLH
    opportunities actually exist in the backtest window.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    univ = universe_df()
    # Sector shocks each day shared by all names in that sector.
    sector_list = sorted(univ["sector"].unique())
    sector_shocks = pd.DataFrame(
        rng.normal(0.0, 0.008, size=(n, len(sector_list))),
        index=dates, columns=sector_list,
    )

    prices = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    for t in tickers:
        if t not in univ.index:
            continue
        sector = univ.loc[t, "sector"]
        drift = 0.08 / 252  # 8% annual drift
        idio = rng.normal(0.0, 0.012, size=n)
        returns = drift + 0.6 * sector_shocks[sector].values + 0.4 * idio

        # Inject COVID drawdown Feb 20 - Mar 23 2020
        for i, d in enumerate(dates):
            if date(2020, 2, 20) <= d.date() <= date(2020, 3, 23):
                returns[i] -= 0.015
            # 2022 bear market
            if date(2022, 1, 1) <= d.date() <= date(2022, 10, 15):
                returns[i] -= 0.001

        start_price = 100.0 * rng.uniform(0.5, 3.0)
        prices[t] = start_price * np.exp(np.cumsum(returns))

    return prices


def fetch_prices(
    tickers: Optional[list[str]] = None,
    start: date = date(2020, 1, 1),
    end: Optional[date] = None,
    use_cache: bool = True,
    force_synthetic: bool = False,
) -> pd.DataFrame:
    """
    Fetch daily close prices for the given tickers.

    Returns a DataFrame indexed by date with one column per ticker.
    Caches results to parquet. Falls back to synthetic prices if yfinance fails.
    """
    if end is None:
        end = date.today()
    if tickers is None:
        tickers = list(SP500_UNIVERSE.keys())

    cache = _cache_path(start, end)
    if use_cache and cache.exists():
        df = pd.read_parquet(cache)
        missing = [t for t in tickers if t not in df.columns]
        if not missing:
            return df[tickers]

    if force_synthetic:
        df = _synthetic_prices(tickers, start, end)
        df.to_parquet(cache)
        return df

    try:
        import yfinance as yf
        # yfinance uses "-" as a separator for some tickers (BRK-B becomes BRK-B).
        # yfinance expects "BRK-B" as-is; keep tickers unchanged.
        raw = yf.download(
            tickers=tickers,
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if raw.empty:
            raise RuntimeError("yfinance returned empty frame")
        # When multiple tickers, columns are a MultiIndex (field, ticker).
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw["Close"].copy()
        else:
            df = raw[["Close"]].copy()
            df.columns = tickers
        df = df.dropna(how="all")
        df.to_parquet(cache)
        return df
    except Exception as e:
        # Fall back to synthetic so the app always runs.
        print(f"[data] yfinance unavailable ({e}); using synthetic prices.")
        df = _synthetic_prices(tickers, start, end)
        df.to_parquet(cache)
        return df


def returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns."""
    return prices.pct_change().dropna(how="all")
