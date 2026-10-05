# SMA Direct-Indexing Platform

A working multi-account direct-indexing platform. The platform manages a book
of client Separately Managed Accounts (SMAs), each tracking the S&P 500 while
honoring per-client screens and continuously harvesting tax losses, within
wash-sale and tracking-error limits.

Built as a portfolio project oriented at BlackRock SMA Solutions.

![Streamlit UI](docs/screenshot.png)

## What this is (and isn't)

Most tax-loss-harvesting demos you'll see are **single-account tools**: feed
one portfolio in, get back a list of what to sell. That's a tax tool, not an
SMA platform.

This project is shaped around the actual business:

- **Multi-account book** of clients, each with their own rules and tax profile
- **Three customer segments** (institution, family office, HNW individual) each
  pre-loading a typical constraint template and optimizer priority profile
- **Direct indexing**: each portfolio is built to track the S&P 500, with
  excluded names reweighted within-sector to keep tracking error contained
- **Continuous harvesting** at tax-lot level with full IRS compliance
- **Live tracking-error measurement** vs the benchmark, with a screen simulator
  that shows the cost of each additional exclusion

## Why segmentation matters

The scale challenge of a real SMA platform isn't managing one account well; it
is managing ten thousand accounts, each with different rules, without
hand-building each one. The segment layer is the templating mechanism:

| Segment | Typical screens | Tax rate | Optimizer priority |
|---|---|---|---|
| Institution | Mandate-driven (fossil fuel, tobacco, weapons) | Tax-exempt | Tracking error |
| Family office | Idiosyncratic (specific tickers, legacy aversions) | Highest marginal | Harvesting |
| HNW individual | Concentration risk (own sector) | High marginal | Harvesting |

The optimizer takes these segment-level priorities and tunes its own weighting
of harvesting-aggressiveness vs tracking-error discipline. A tax-exempt
endowment never harvests; a Chevron executive always does.

## IRS rules implemented

| Rule | Where |
|---|---|
| Wash sale rule (IRC §1091, 30 days before + after) | `src/tax.py:is_wash_sale` |
| Short-term vs long-term classification (>365 days) | `src/tax.py:TaxLot.is_long_term` |
| Loss ordering (ST offsets ST first, then cross-category) | `src/tax.py:apply_loss_ordering` |
| $3,000 ordinary-income offset cap | `src/tax.py:apply_loss_ordering` |
| Indefinite loss carryforward | `src/tax.py:apply_loss_ordering` |
| Net Investment Income Tax (3.8%) | `src/tax.py:TaxRates` |

Covered by `tests/test_tax.py` (run `pytest tests/`).

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

First launch fetches ~50 tickers from yfinance (Jan 2020 to Dec 2024) and
caches them to parquet in `data/`. If yfinance is unreachable, the data layer
falls back to deterministic synthetic prices so the app always runs.

## Project layout

```
sma_platform/
├── app.py                    Streamlit UI (book overview + drill-in + simulator)
├── requirements.txt
├── README.md
├── src/
│   ├── data.py               S&P 500 universe + price fetching
│   ├── tax.py                Lots, wash sale, loss ordering, carryforward
│   ├── screens.py            Client exclusion rules
│   ├── portfolio.py          Portfolio, benchmark, tracking error
│   ├── segments.py           Institution / family office / HNW profiles
│   ├── client.py             Client, Book, YAML loader
│   └── backtest.py           Historical simulation and tax calculation
├── sample_clients/           Six YAML client specs across the three segments
└── tests/
    └── test_tax.py           Tax engine unit tests
```

## Three numbers the demo surfaces

A platform like this exists to deliver three outcomes a client can see:

1. **Harvested losses in dollars** — total capital losses realized across the
   lifetime of the account
2. **After-tax excess return** — annualized return of the SMA plus tax savings,
   minus the return of simply holding SPY
3. **Tracking-error cost of each screen** — basis points added to tracking
   error for each exclusion the client applied

Each one is called out as a tile on the dashboard.

## Where this simplifies vs production

Honesty on the limits is part of the project. The two biggest ones:

- **The optimizer is rule-based.** Pick a replacement by sector match and
  current underweight. Production is a quadratic program minimizing tracking
  error subject to lot-level wash constraints and transaction cost penalties.
- **Tracking error is realized, not ex-ante.** No factor risk model. Production
  uses a model like Barra US4 to forecast tracking error before trading.

Full list in the "About" tab of the Streamlit app.

## License

MIT.
