"""Unit tests for the tax engine."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.tax import (
    LONG_TERM_THRESHOLD_DAYS,
    ORDINARY_INCOME_OFFSET_CAP,
    WASH_SALE_WINDOW_DAYS,
    RealizedTrade,
    TaxLot,
    TaxRates,
    apply_loss_ordering,
    classify_realized_trades,
    is_wash_sale,
    select_lots_to_harvest,
)


# ---------------------------------------------------------------------------
# TaxLot
# ---------------------------------------------------------------------------

class TestTaxLot:
    def test_cost_basis(self):
        lot = TaxLot("AAPL", 100, 150.0, date(2023, 1, 1))
        assert lot.cost_basis == 15000.0

    def test_market_value(self):
        lot = TaxLot("AAPL", 100, 150.0, date(2023, 1, 1))
        assert lot.market_value(170.0) == 17000.0

    def test_unrealized_pnl_gain(self):
        lot = TaxLot("AAPL", 100, 150.0, date(2023, 1, 1))
        assert lot.unrealized_pnl(170.0) == 2000.0

    def test_unrealized_pnl_loss(self):
        lot = TaxLot("AAPL", 100, 150.0, date(2023, 1, 1))
        assert lot.unrealized_pnl(130.0) == -2000.0

    def test_short_term_boundary(self):
        purchase = date(2023, 1, 1)
        # 365 days later is still short-term
        lot = TaxLot("AAPL", 100, 150.0, purchase)
        assert not lot.is_long_term(purchase + timedelta(days=365))

    def test_long_term_boundary(self):
        purchase = date(2023, 1, 1)
        # 366 days later is long-term
        lot = TaxLot("AAPL", 100, 150.0, purchase)
        assert lot.is_long_term(purchase + timedelta(days=366))


# ---------------------------------------------------------------------------
# Wash sale
# ---------------------------------------------------------------------------

class TestWashSale:
    def test_no_recent_purchase(self):
        assert not is_wash_sale(date(2024, 6, 1), "AAPL", [])

    def test_purchase_30_days_before(self):
        history = [("AAPL", date(2024, 5, 2))]  # exactly 30 days before
        assert is_wash_sale(date(2024, 6, 1), "AAPL", history)

    def test_purchase_31_days_before(self):
        history = [("AAPL", date(2024, 5, 1))]  # 31 days before: outside window
        assert not is_wash_sale(date(2024, 6, 1), "AAPL", history)

    def test_purchase_30_days_after(self):
        history = [("AAPL", date(2024, 7, 1))]  # 30 days after
        assert is_wash_sale(date(2024, 6, 1), "AAPL", history)

    def test_different_ticker(self):
        history = [("MSFT", date(2024, 5, 15))]
        assert not is_wash_sale(date(2024, 6, 1), "AAPL", history)

    def test_substantially_identical_set(self):
        history = [("SPY", date(2024, 5, 15))]
        # Treat SPY and IVV as substantially identical
        assert is_wash_sale(date(2024, 6, 1), "IVV", history, {"SPY", "IVV"})


# ---------------------------------------------------------------------------
# Loss ordering
# ---------------------------------------------------------------------------

class TestLossOrdering:
    def test_pure_gains_no_losses(self):
        rates = TaxRates(short_term=0.37, long_term=0.20, state=0.0, niit=0.0)
        r = apply_loss_ordering(
            st_gains=10000, st_losses=0, lt_gains=20000, lt_losses=0, rates=rates
        )
        assert r.net_st == 10000
        assert r.net_lt == 20000
        assert r.tax_after_harvest == pytest.approx(10000 * 0.37 + 20000 * 0.20)

    def test_st_losses_offset_st_gains_first(self):
        rates = TaxRates(short_term=0.37, long_term=0.20, state=0.0, niit=0.0)
        r = apply_loss_ordering(
            st_gains=10000, st_losses=4000, lt_gains=20000, lt_losses=0, rates=rates
        )
        # Net ST = 6000, no cross-category needed
        assert r.net_st == 6000
        assert r.net_lt == 20000

    def test_st_losses_offset_lt_gains_when_excess(self):
        rates = TaxRates(short_term=0.37, long_term=0.20, state=0.0, niit=0.0)
        r = apply_loss_ordering(
            st_gains=1000, st_losses=5000, lt_gains=10000, lt_losses=0, rates=rates
        )
        # Net ST = -4000, offsets LT: net_lt = 10000 - 4000 = 6000
        assert r.net_st == 0
        assert r.net_lt == 6000

    def test_three_k_ordinary_offset(self):
        rates = TaxRates(short_term=0.37, long_term=0.20, state=0.0, niit=0.0)
        r = apply_loss_ordering(
            st_gains=0, st_losses=5000, lt_gains=0, lt_losses=0, rates=rates
        )
        assert r.ordinary_offset == 3000
        assert r.carryforward_st == 2000

    def test_carryforward_lt_classification(self):
        rates = TaxRates(short_term=0.37, long_term=0.20, state=0.0, niit=0.0)
        r = apply_loss_ordering(
            st_gains=0, st_losses=0, lt_gains=0, lt_losses=10000, rates=rates
        )
        assert r.ordinary_offset == 3000
        assert r.carryforward_lt == 7000
        assert r.carryforward_st == 0

    def test_prior_carryforward_applied(self):
        rates = TaxRates(short_term=0.37, long_term=0.20, state=0.0, niit=0.0)
        r = apply_loss_ordering(
            st_gains=5000, st_losses=0, lt_gains=0, lt_losses=0, rates=rates,
            prior_st_carry=2000,
        )
        # Prior carry wipes out $2K of ST gain
        assert r.net_st == 3000

    def test_tax_savings_vs_no_harvest(self):
        rates = TaxRates(short_term=0.37, long_term=0.20, state=0.05, niit=0.038)
        r = apply_loss_ordering(
            st_gains=10000, st_losses=5000, lt_gains=20000, lt_losses=3000, rates=rates
        )
        # tax_before uses gross gains; tax_after uses netted
        assert r.tax_before_harvest > r.tax_after_harvest


# ---------------------------------------------------------------------------
# Lot selection
# ---------------------------------------------------------------------------

class TestLotSelection:
    def test_skips_gaining_lots(self):
        lots = [TaxLot("AAPL", 100, 150, date(2023, 1, 1))]
        selected = select_lots_to_harvest(lots, 170, date(2024, 6, 1))
        assert selected == []

    def test_picks_losing_lot(self):
        lots = [TaxLot("AAPL", 100, 150, date(2023, 1, 1))]
        selected = select_lots_to_harvest(
            lots, 130, date(2024, 6, 1), min_loss_pct=0.05, min_loss_dollars=100
        )
        assert len(selected) == 1

    def test_respects_min_loss_pct(self):
        lots = [TaxLot("AAPL", 100, 150, date(2023, 1, 1))]
        # Only 1% loss; threshold is 5%
        selected = select_lots_to_harvest(
            lots, 148.5, date(2024, 6, 1), min_loss_pct=0.05, min_loss_dollars=0
        )
        assert selected == []

    def test_respects_min_loss_dollars(self):
        lots = [TaxLot("AAPL", 10, 150, date(2023, 1, 1))]
        # 10 shares * $20 loss = $200; threshold is $500
        selected = select_lots_to_harvest(
            lots, 130, date(2024, 6, 1), min_loss_pct=0.0, min_loss_dollars=500
        )
        assert selected == []

    def test_prefers_short_term_losses(self):
        # One ST and one LT loss lot of same magnitude
        today = date(2024, 6, 1)
        st_lot = TaxLot("AAPL", 100, 150, today - timedelta(days=200))
        lt_lot = TaxLot("AAPL", 100, 150, today - timedelta(days=400))
        lots = [lt_lot, st_lot]  # intentionally wrong order
        selected = select_lots_to_harvest(lots, 130, today, min_loss_pct=0.05, min_loss_dollars=100)
        assert selected[0] is st_lot

    def test_wash_sale_filters_out(self):
        lots = [TaxLot("AAPL", 100, 150, date(2023, 1, 1))]
        # Recent purchase triggers wash
        history = [("AAPL", date(2024, 5, 15))]
        selected = select_lots_to_harvest(
            lots, 130, date(2024, 6, 1),
            min_loss_pct=0.05, min_loss_dollars=100,
            purchase_history=history,
        )
        assert selected == []


# ---------------------------------------------------------------------------
# Trade classification
# ---------------------------------------------------------------------------

class TestTradeClassification:
    def test_classify_mixed_year(self):
        trades = [
            RealizedTrade("AAPL", 100, 15000, 17000, date(2023, 1, 1), date(2024, 6, 1)),  # LT gain
            RealizedTrade("MSFT", 50, 10000, 8000, date(2024, 1, 1), date(2024, 6, 1)),    # ST loss
            RealizedTrade("GOOG", 20, 5000, 7000, date(2024, 1, 1), date(2024, 6, 1)),     # ST gain
            RealizedTrade("TSLA", 10, 3000, 1000, date(2022, 1, 1), date(2024, 6, 1)),     # LT loss
        ]
        st_g, st_l, lt_g, lt_l = classify_realized_trades(trades, 2024)
        assert st_g == 2000   # GOOG
        assert st_l == 2000   # MSFT
        assert lt_g == 2000   # AAPL
        assert lt_l == 2000   # TSLA
