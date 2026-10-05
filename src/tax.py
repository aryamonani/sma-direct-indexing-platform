"""
Tax engine: lot accounting, wash sale detection, and after-tax calculations.

Implements the US tax rules relevant to equity SMAs:
- IRC Section 1091 wash sale rule (30 days before + 30 days after)
- Short-term vs long-term classification (> 365 days = long-term)
- Loss ordering: ST losses offset ST gains first, then LT; vice versa
- Annual $3,000 ordinary-income offset cap (IRC 1211)
- Indefinite loss carryforward
- Net Investment Income Tax (3.8%) as additive rate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterator, Optional


WASH_SALE_WINDOW_DAYS = 30
LONG_TERM_THRESHOLD_DAYS = 365
ORDINARY_INCOME_OFFSET_CAP = 3_000.0


@dataclass
class TaxRates:
    """Marginal rates applicable to the client."""
    short_term: float = 0.37      # top ordinary rate
    long_term: float = 0.20       # top LT cap gains rate
    state: float = 0.05           # state cap gains, if any
    niit: float = 0.038           # Net Investment Income Tax

    @property
    def st_total(self) -> float:
        return self.short_term + self.state + self.niit

    @property
    def lt_total(self) -> float:
        return self.long_term + self.state + self.niit


@dataclass
class TaxLot:
    """A single purchase of a security. The unit of tax accounting."""
    ticker: str
    shares: float
    cost_basis_per_share: float
    purchase_date: date
    lot_id: str = ""

    def __post_init__(self):
        if not self.lot_id:
            self.lot_id = f"{self.ticker}-{self.purchase_date.isoformat()}-{id(self):x}"

    @property
    def cost_basis(self) -> float:
        return self.shares * self.cost_basis_per_share

    def market_value(self, current_price: float) -> float:
        return self.shares * current_price

    def unrealized_pnl(self, current_price: float) -> float:
        return self.market_value(current_price) - self.cost_basis

    def is_long_term(self, as_of: date) -> bool:
        return (as_of - self.purchase_date).days > LONG_TERM_THRESHOLD_DAYS

    def days_held(self, as_of: date) -> int:
        return (as_of - self.purchase_date).days


@dataclass
class RealizedTrade:
    """A completed sale, recorded for tax reporting."""
    ticker: str
    shares: float
    cost_basis: float
    proceeds: float
    purchase_date: date
    sale_date: date

    @property
    def pnl(self) -> float:
        return self.proceeds - self.cost_basis

    @property
    def is_long_term(self) -> bool:
        return (self.sale_date - self.purchase_date).days > LONG_TERM_THRESHOLD_DAYS


# ---------------------------------------------------------------------------
# Wash sale rule
# ---------------------------------------------------------------------------

def is_wash_sale(
    sale_date: date,
    ticker: str,
    purchase_history: list[tuple[str, date]],
    replacement_universe: Optional[set[str]] = None,
) -> bool:
    """
    True if selling `ticker` on `sale_date` would trigger a wash sale.

    A wash sale is triggered if a 'substantially identical' security was bought
    within 30 days before OR after the sale. We use ticker identity as the
    strict test. In production, substantial-identity analysis is a separate
    product decision; mutual funds tracking the same index are commonly treated
    as substantially identical, individual stocks almost never.

    purchase_history: list of (ticker, purchase_date) for recent buys.
    replacement_universe: optional set of tickers treated as substantially
        identical to the sold ticker. Defaults to {ticker}.
    """
    identical = replacement_universe or {ticker}
    window_start = sale_date - timedelta(days=WASH_SALE_WINDOW_DAYS)
    window_end = sale_date + timedelta(days=WASH_SALE_WINDOW_DAYS)
    for t, d in purchase_history:
        if t in identical and window_start <= d <= window_end:
            return True
    return False


# ---------------------------------------------------------------------------
# Loss ordering and carryforward
# ---------------------------------------------------------------------------

@dataclass
class TaxYearResult:
    """What the client actually owes (or saves) for a tax year."""
    year: int
    st_gains: float = 0.0
    st_losses: float = 0.0   # positive number representing loss magnitude
    lt_gains: float = 0.0
    lt_losses: float = 0.0

    # Computed by apply_loss_ordering
    net_st: float = 0.0
    net_lt: float = 0.0
    ordinary_offset: float = 0.0
    carryforward_st: float = 0.0
    carryforward_lt: float = 0.0
    tax_before_harvest: float = 0.0
    tax_after_harvest: float = 0.0

    @property
    def tax_savings(self) -> float:
        return self.tax_before_harvest - self.tax_after_harvest


def classify_realized_trades(
    trades: list[RealizedTrade], year: int
) -> tuple[float, float, float, float]:
    """Bucket a year's realized trades into (st_gains, st_losses, lt_gains, lt_losses)."""
    st_gains = st_losses = lt_gains = lt_losses = 0.0
    for t in trades:
        if t.sale_date.year != year:
            continue
        if t.is_long_term:
            if t.pnl >= 0:
                lt_gains += t.pnl
            else:
                lt_losses += -t.pnl
        else:
            if t.pnl >= 0:
                st_gains += t.pnl
            else:
                st_losses += -t.pnl
    return st_gains, st_losses, lt_gains, lt_losses


def apply_loss_ordering(
    st_gains: float,
    st_losses: float,
    lt_gains: float,
    lt_losses: float,
    rates: TaxRates,
    prior_st_carry: float = 0.0,
    prior_lt_carry: float = 0.0,
) -> TaxYearResult:
    """
    Apply IRS loss ordering: ST losses offset ST gains first, excess offsets LT,
    then up to $3K offsets ordinary income, rest carries forward.
    """
    r = TaxYearResult(year=0)
    r.st_gains = st_gains
    r.st_losses = st_losses
    r.lt_gains = lt_gains
    r.lt_losses = lt_losses

    # Include prior-year carryforwards
    st_losses_total = st_losses + prior_st_carry
    lt_losses_total = lt_losses + prior_lt_carry

    # Step 1: net within each category
    net_st = st_gains - st_losses_total
    net_lt = lt_gains - lt_losses_total

    # Step 2: cross-category netting if signs differ
    if net_st < 0 and net_lt > 0:
        offset = min(-net_st, net_lt)
        net_st += offset
        net_lt -= offset
    elif net_lt < 0 and net_st > 0:
        offset = min(-net_lt, net_st)
        net_lt += offset
        net_st -= offset

    # Step 3: $3K ordinary income offset
    ordinary_offset = 0.0
    carry_st = carry_lt = 0.0
    if net_st < 0 or net_lt < 0:
        total_net_loss = max(0.0, -(net_st + net_lt))
        ordinary_offset = min(ORDINARY_INCOME_OFFSET_CAP, total_net_loss)
        remaining_loss = total_net_loss - ordinary_offset
        # Carry forward: ST losses carry as ST, LT as LT
        if net_st < 0 and net_lt < 0:
            carry_st = min(remaining_loss, -net_st)
            carry_lt = remaining_loss - carry_st
        elif net_st < 0:
            carry_st = remaining_loss
        else:
            carry_lt = remaining_loss
        net_st = max(0.0, net_st)
        net_lt = max(0.0, net_lt)

    r.net_st = net_st
    r.net_lt = net_lt
    r.ordinary_offset = ordinary_offset
    r.carryforward_st = carry_st
    r.carryforward_lt = carry_lt

    # Tax calculation
    r.tax_after_harvest = (
        net_st * rates.st_total
        + net_lt * rates.lt_total
        - ordinary_offset * rates.short_term  # saved at ordinary rate
    )
    r.tax_before_harvest = (
        st_gains * rates.st_total + lt_gains * rates.lt_total
    )
    return r


# ---------------------------------------------------------------------------
# Lot selection
# ---------------------------------------------------------------------------

def select_lots_to_harvest(
    lots: list[TaxLot],
    current_price: float,
    as_of: date,
    min_loss_pct: float = 0.03,
    min_loss_dollars: float = 100.0,
    purchase_history: Optional[list[tuple[str, date]]] = None,
    replacement_universe: Optional[set[str]] = None,
) -> list[TaxLot]:
    """
    Pick lots to sell for a given ticker.

    Policy: sell loss lots that clear BOTH thresholds (percent and dollars),
    skip any that would trigger a wash sale. Prefers short-term losses first
    (they offset at the higher ordinary rate), then long-term.
    """
    purchase_history = purchase_history or []

    candidates = []
    for lot in lots:
        pnl = lot.unrealized_pnl(current_price)
        if pnl >= 0:
            continue
        loss_pct = -pnl / lot.cost_basis
        if loss_pct < min_loss_pct:
            continue
        if -pnl < min_loss_dollars:
            continue
        if is_wash_sale(as_of, lot.ticker, purchase_history, replacement_universe):
            continue
        candidates.append(lot)

    # Prefer ST losses first (they save at the higher ordinary rate),
    # then within each class pick largest loss first.
    def sort_key(l: TaxLot):
        is_st = not l.is_long_term(as_of)
        loss_mag = -l.unrealized_pnl(current_price)
        return (0 if is_st else 1, -loss_mag)

    candidates.sort(key=sort_key)
    return candidates
