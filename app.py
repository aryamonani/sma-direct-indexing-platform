"""
SMA Direct-Indexing Platform (demo).

Multi-account book that tracks the S&P 500, honors per-client screens, and
continuously harvests tax losses within wash-sale and tracking-error limits.

Run: streamlit run app.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.backtest import BacktestResult, run_backtest
from src.client import Book, Client, load_book
from src.data import (
    SP500_UNIVERSE,
    fetch_prices,
    returns_from_prices,
    sectors,
    universe_df,
)
from src.portfolio import (
    benchmark_weights,
    tracking_error,
    tracking_weights,
)
from src.screens import THEMATIC_EXCLUSIONS, ScreenSet
from src.segments import SEGMENT_PROFILES, Segment, get_profile
from src.tax import TaxRates


SAMPLE_DIR = Path(__file__).resolve().parent / "sample_clients"
BACKTEST_START = date(2020, 1, 1)
BACKTEST_END = date(2024, 12, 31)


# ---------------------------------------------------------------------------
# Page setup and caching
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SMA Direct-Indexing Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading price history...")
def load_prices() -> pd.DataFrame:
    tickers = list(SP500_UNIVERSE.keys())
    return fetch_prices(tickers=tickers, start=BACKTEST_START, end=BACKTEST_END)


@st.cache_resource(show_spinner="Loading book of clients...")
def load_the_book() -> Book:
    return load_book(SAMPLE_DIR)


@st.cache_data(show_spinner="Computing returns...")
def load_returns(prices_hash: int) -> pd.DataFrame:
    # Cache keyed on an id so Streamlit knows to recompute if prices change
    return returns_from_prices(load_prices())


@st.cache_data(show_spinner=False)
def run_backtest_cached(client_id: str) -> BacktestResult:
    book = load_the_book()
    client = book.by_id(client_id)
    prices = load_prices()
    return run_backtest(client, prices)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("SMA Platform")
st.sidebar.caption(
    "A multi-account direct-indexing platform. Tracks the S&P 500, honors "
    "per-client screens, and continuously harvests tax losses within "
    "wash-sale and tracking-error limits."
)

view = st.sidebar.radio(
    "View",
    options=["Book overview", "Client drill-in", "Screen simulator", "About"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Universe: {len(SP500_UNIVERSE)} S&P 500 names across {len(sectors())} sectors.\n\n"
    f"Backtest window: {BACKTEST_START.isoformat()} to {BACKTEST_END.isoformat()}."
)


# ---------------------------------------------------------------------------
# BOOK OVERVIEW
# ---------------------------------------------------------------------------

def view_book_overview():
    st.title("Book overview")
    st.caption("Every account the platform manages, across all three client segments.")

    book = load_the_book()
    prices = load_prices()

    with st.spinner(f"Backtesting {len(book)} clients..."):
        results = {c.client_id: run_backtest_cached(c.client_id) for c in book}

    # Summary tiles
    total_aum = book.total_aum()
    total_harvested = sum(r.total_harvested for r in results.values())
    total_tax_savings = sum(r.total_tax_savings for r in results.values())
    avg_te = np.mean([r.tracking_error for r in results.values()])

    cols = st.columns(4)
    cols[0].metric("Total AUM", f"${total_aum / 1e6:,.1f}M")
    cols[1].metric("Clients", f"{len(book)}")
    cols[2].metric("Harvested losses (lifetime)", f"${total_harvested / 1e6:,.2f}M")
    cols[3].metric("Est. tax savings (lifetime)", f"${total_tax_savings / 1e6:,.2f}M")

    st.markdown("---")

    # Per-client table
    rows = []
    for client in book:
        res = results[client.client_id]
        rows.append({
            "Client": client.display_name,
            "Segment": get_profile(client.segment).label,
            "AUM": client.aum,
            "Final value": res.final_market_value,
            "Harvested $": res.total_harvested,
            "Tax savings $": res.total_tax_savings,
            "Tracking error": res.tracking_error,
            "Carryforward $": res.carryforward_losses,
            "client_id": client.client_id,
        })
    df = pd.DataFrame(rows)

    styled = df.drop(columns=["client_id"]).style.format({
        "AUM": "${:,.0f}",
        "Final value": "${:,.0f}",
        "Harvested $": "${:,.0f}",
        "Tax savings $": "${:,.0f}",
        "Tracking error": "{:.2%}",
        "Carryforward $": "${:,.0f}",
    })
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Tracking error by segment")
        st.caption("Segments carry different screen loads; tracking error reflects that.")
        seg_df = pd.DataFrame({
            "Segment": [get_profile(c.segment).label for c in book],
            "TE": [results[c.client_id].tracking_error for c in book],
            "Client": [c.display_name for c in book],
        })
        fig = px.bar(
            seg_df, x="Client", y="TE", color="Segment",
            labels={"TE": "Annualized tracking error"},
        )
        fig.update_layout(yaxis_tickformat=".1%", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("After-tax excess return vs SPY buy-and-hold")
        st.caption("Positive means the SMA (with harvesting) beat holding the index after taxes.")
        exc_df = pd.DataFrame({
            "Client": [c.display_name for c in book],
            "Excess": [results[c.client_id].after_tax_excess_return for c in book],
            "Segment": [get_profile(c.segment).label for c in book],
        })
        fig = px.bar(
            exc_df, x="Client", y="Excess", color="Segment",
            labels={"Excess": "Annualized after-tax excess return"},
        )
        fig.update_layout(yaxis_tickformat=".2%", height=380)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# CLIENT DRILL-IN
# ---------------------------------------------------------------------------

def view_client_drillin():
    st.title("Client drill-in")

    book = load_the_book()
    client_names = {c.display_name: c for c in book}
    selected_name = st.selectbox("Account", options=list(client_names.keys()))
    client = client_names[selected_name]
    profile = get_profile(client.segment)

    cols = st.columns([2, 1])
    with cols[0]:
        st.subheader(client.display_name)
        st.caption(profile.label)
        st.markdown(f"**AUM:** ${client.aum:,.0f}")
        st.markdown("**Active screens:**")
        for s in client.screens.description():
            st.markdown(f"- {s}")
        if client.notes:
            st.markdown("**Notes:**")
            for n in client.notes:
                st.markdown(f"- {n}")

    with cols[1]:
        st.markdown("**Tax rates**")
        st.markdown(f"- Short-term: {client.tax_rates.short_term:.1%}")
        st.markdown(f"- Long-term: {client.tax_rates.long_term:.1%}")
        st.markdown(f"- State: {client.tax_rates.state:.1%}")
        st.markdown(f"- NIIT: {client.tax_rates.niit:.1%}")
        st.markdown("**Optimizer priorities**")
        st.markdown(f"- Harvest weight: {client.priorities.harvest_weight:.2f}")
        st.markdown(f"- Tracking weight: {client.priorities.tracking_weight:.2f}")
        st.markdown(f"- Turnover weight: {client.priorities.turnover_weight:.2f}")

    st.markdown("---")

    with st.spinner("Running backtest..."):
        result = run_backtest_cached(client.client_id)

    m = st.columns(4)
    m[0].metric("Final market value", f"${result.final_market_value:,.0f}")
    m[1].metric("Harvested losses", f"${result.total_harvested:,.0f}")
    m[2].metric("Est. tax savings", f"${result.total_tax_savings:,.0f}")
    m[3].metric("Tracking error", f"{result.tracking_error:.2%}")

    st.subheader("Portfolio value vs benchmark (buy-and-hold SPY)")
    both = pd.DataFrame({
        "SMA account": result.daily_values,
        "Benchmark": result.benchmark_daily_values,
    })
    fig = px.line(both)
    fig.update_layout(yaxis_title="Market value ($)", height=400, legend_title="")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Harvested losses by year")
        yrs = sorted(result.harvested_losses_by_year.keys())
        yr_df = pd.DataFrame({
            "Year": yrs,
            "Harvested $": [result.harvested_losses_by_year[y] for y in yrs],
            "Tax savings $": [result.tax_savings_by_year.get(y, 0.0) for y in yrs],
        })
        st.dataframe(
            yr_df.style.format({"Harvested $": "${:,.0f}", "Tax savings $": "${:,.0f}"}),
            use_container_width=True, hide_index=True,
        )

    with col2:
        st.subheader("Recent harvest events")
        events = result.harvest_events[-15:]
        if events:
            ev_df = pd.DataFrame(events)
            ev_df = ev_df.rename(columns={
                "date": "Date", "sold": "Sold", "sold_shares": "Shares",
                "loss_realized": "Loss realized", "replacement": "Replacement",
            })
            st.dataframe(
                ev_df.style.format({"Shares": "{:,.1f}", "Loss realized": "${:,.0f}"}),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info(
                "No harvest events for this account. Tax-exempt institutions see "
                "no harvesting benefit, so the engine does not act."
            )

    st.markdown("---")
    st.subheader("Current holdings")
    final_prices = {t: float(result.daily_values.index[-1] and 0) for t in client.screens.allowed_universe()}
    # Use the last price row from the backtest window instead
    prices = load_prices()
    last_prices = {t: float(prices.iloc[-1][t]) for t in prices.columns if pd.notna(prices.iloc[-1][t])}
    rows = []
    for ticker, lots in result.portfolio.lots.items():
        if not lots:
            continue
        shares = sum(l.shares for l in lots)
        mv = sum(l.market_value(last_prices.get(ticker, 0.0)) for l in lots)
        upnl = sum(l.unrealized_pnl(last_prices.get(ticker, 0.0)) for l in lots)
        cost = sum(l.cost_basis for l in lots)
        rows.append({
            "Ticker": ticker,
            "Sector": SP500_UNIVERSE.get(ticker, {}).get("sector", "?"),
            "Shares": shares,
            "Cost basis": cost,
            "Market value": mv,
            "Unrealized P/L": upnl,
            "Lots": len(lots),
        })
    holdings_df = pd.DataFrame(rows).sort_values("Market value", ascending=False)
    st.dataframe(
        holdings_df.style.format({
            "Shares": "{:,.1f}",
            "Cost basis": "${:,.0f}",
            "Market value": "${:,.0f}",
            "Unrealized P/L": "${:,.0f}",
        }),
        use_container_width=True, hide_index=True,
    )


# ---------------------------------------------------------------------------
# SCREEN SIMULATOR
# ---------------------------------------------------------------------------

def view_screen_simulator():
    st.title("Screen simulator")
    st.caption(
        "Build a custom screen set and watch the tracking error move. This is "
        "the core tension in direct indexing: every exclusion pulls the "
        "portfolio away from the benchmark."
    )

    all_sectors = sectors()
    thematics = list(THEMATIC_EXCLUSIONS.keys())

    cols = st.columns(2)
    with cols[0]:
        selected_sectors = st.multiselect(
            "Exclude sectors", options=all_sectors, default=[],
        )
        selected_thematics = st.multiselect(
            "Thematic exclusions",
            options=thematics,
            format_func=lambda k: THEMATIC_EXCLUSIONS[k]["label"],
        )
    with cols[1]:
        tickers_list = sorted(SP500_UNIVERSE.keys())
        selected_tickers = st.multiselect(
            "Exclude individual tickers", options=tickers_list,
        )
        max_wt = st.slider(
            "Max single-security weight", 0.02, 0.15, 0.10, 0.01,
            format="%.0f%%",
        )

    screens = ScreenSet(
        excluded_sectors=set(selected_sectors),
        excluded_tickers=set(selected_tickers),
        thematic_screens=set(selected_thematics),
        max_single_weight=max_wt,
    )

    prices = load_prices()
    rets = returns_from_prices(prices)

    bench_w = benchmark_weights()
    port_w = tracking_weights(screens)

    te = tracking_error(port_w, bench_w, rets)
    allowed = screens.allowed_universe()
    excluded_count = len(SP500_UNIVERSE) - len(allowed)
    excluded_weight = 1.0 - bench_w[allowed].sum() if allowed else 1.0

    m = st.columns(4)
    m[0].metric("Allowed names", f"{len(allowed)} / {len(SP500_UNIVERSE)}")
    m[1].metric("Benchmark weight excluded", f"{excluded_weight:.1%}")
    m[2].metric("Tracking error (annualized)", f"{te:.2%}")
    cost_bps = te * 100 * 100  # bps
    m[3].metric("Tracking error (bps)", f"{cost_bps:.0f} bps")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Sector weights: portfolio vs benchmark")
        univ = universe_df()
        bench_sector = univ.groupby("sector").apply(
            lambda g: bench_w.reindex(g.index).sum()
        )
        port_sector = univ.groupby("sector").apply(
            lambda g: port_w.reindex(g.index).sum()
        )
        comp = pd.DataFrame({"Benchmark": bench_sector, "Portfolio": port_sector})
        comp = comp.sort_values("Benchmark", ascending=False)
        fig = go.Figure()
        fig.add_bar(x=comp.index, y=comp["Benchmark"], name="Benchmark")
        fig.add_bar(x=comp.index, y=comp["Portfolio"], name="Portfolio")
        fig.update_layout(
            barmode="group", height=440,
            yaxis_tickformat=".1%",
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Top 15 portfolio holdings")
        top = port_w[port_w > 0].sort_values(ascending=False).head(15)
        top_df = pd.DataFrame({
            "Ticker": top.index,
            "Weight": top.values,
            "Sector": [SP500_UNIVERSE[t]["sector"] for t in top.index],
        })
        st.dataframe(
            top_df.style.format({"Weight": "{:.2%}"}),
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")
    st.caption(
        "**Reading this screen:** Tracking error below ~50 bps is light-touch, "
        "customization that barely changes performance. 50-150 bps is a typical "
        "SMA. Above 300 bps, the account meaningfully deviates from the index, "
        "which is sometimes exactly what a concentration-risk client wants."
    )


# ---------------------------------------------------------------------------
# ABOUT
# ---------------------------------------------------------------------------

def view_about():
    st.title("About this project")
    st.markdown(
        """
A working demo of a BlackRock-style SMA direct-indexing platform. The intent
is to show the shape of the real product, not to replicate it.

### What it does

- Manages a **book** of client accounts (not a single portfolio).
- Each account tracks the S&P 500 using a sampled subset weighted to the index.
- Honors per-client **screens**: sector exclusions, ticker exclusions, thematic
  exclusions (fossil-fuel-free, tobacco-free, weapons-free), and single-security
  weight caps.
- Continuously **harvests tax losses** at lot level while respecting:
  - IRC Section 1091 wash-sale rule (30 days before and after the sale)
  - Short-term vs long-term classification
  - IRS loss ordering (ST losses offset ST gains first, then LT)
  - $3,000 ordinary-income offset cap and indefinite carryforward
  - 3.8% Net Investment Income Tax
- Measures realized **tracking error** of each account against a buy-and-hold
  benchmark.
- Pre-loads constraints and optimizer priorities by **client segment**
  (institution, family office, HNW individual).

### Where this simplifies vs production

- Universe is 50 names, not the full S&P 500. Preserves sector diversity;
  cuts yfinance load.
- Optimizer is **rule-based** (sort losses by priority, pick replacements by
  sector match). Production uses quadratic programming (minimize tracking
  error subject to screens, lot-level wash constraints, transaction costs).
- Replacement selection is sector-matched, not factor-matched. A real
  platform selects replacements that preserve the factor loadings of the
  sold name (value, momentum, quality, size).
- Tracking error is **realized**, not ex-ante from a factor risk model
  (e.g. Barra). Realized TE is honest and reproducible for a demo.
- ESG screens are approximated by sector / sub-industry / ticker lookups.
  Production uses licensed classification data (MSCI, Sustainalytics).
- No transition management for new accounts arriving with legacy
  concentrated holdings. Noted as the biggest extension for the family-office
  segment.
- No fixed income, no derivatives, no multi-asset. Equity direct indexing only.

### Tech stack

Python, Streamlit for the UI, pandas and numpy for data, Plotly for charts,
yfinance for prices (with a deterministic synthetic fallback), YAML for
client specifications.
        """
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

if view == "Book overview":
    view_book_overview()
elif view == "Client drill-in":
    view_client_drillin()
elif view == "Screen simulator":
    view_screen_simulator()
elif view == "About":
    view_about()
