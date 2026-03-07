# Prediction Market Analysis

> Forked from [jon-becker/prediction-market-analysis](https://github.com/jon-becker/prediction-market-analysis). Original data collection, analysis framework, and datasets by [Jon Becker](https://github.com/jon-becker).

## Overview

Collect, analyze, and backtest prediction market data from Kalshi and Polymarket. The original project handles data collection and analysis over 36GiB of market and trade data. This fork adds a backtesting engine for running trading strategies on historical data.

## Features

- 36GiB pre-collected dataset from Polymarket and Kalshi
- Data collection indexers for Kalshi API and Polymarket blockchain
- Parquet storage with automatic progress saving
- Analysis script framework (`make analyze`)
- Backtesting engine with strategy framework *(fork addition)*

## Additions in This Fork

### Backtesting Engine

Replays historical trades chronologically, feeding each to a pluggable strategy and resolving positions against actual outcomes. No lookahead bias.

- DuckDB-powered Parquet queries
- Portfolio tracking with bankroll enforcement and position averaging
- Fee models for Kalshi (7% profit fee) and Polymarket (2% winnings fee)
- Metrics: Sharpe ratio, max drawdown, profit factor, win rate, equity curves, P&L by category
- Kelly criterion sizing with fractional Kelly and max-bet caps
- Per-market exposure limits

### Strategy Framework

`Strategy` base class with `on_trade()` and `on_resolution()` hooks. Strategies receive trade context (price, side, volume, ticker) and current bankroll, then return `BUY_YES`, `BUY_NO`, or `SKIP`.

### Implemented Strategies

- **Longshot Fade** — bets against extreme prices (<15c or >85c) where markets overprice longshots
- **Calibration Mispricing** — trades where empirical win rates diverge from market-implied probability, using either default offsets or a fitted calibration curve

### Research Analytics

- **Calibration curve** — predicted probability vs realized win frequency, showing where markets misprice
- **6-panel figure** — calibration curve, equity curves, return comparison, P&L by category, win rate vs Sharpe, metrics table
- **Fitted calibration** — data-driven edge estimates from historical outcomes, fed into the mispricing strategy

### Data Pipeline

- **Sample fetcher** (`scripts/fetch_sample_data.py`) — pulls ~300 resolved markets from Kalshi API across 3 time windows, no need to download the full dataset
- Auto-category mapping from event tickers

### Testing

33 tests covering portfolio, fees, sizing, calibration, strategies, engine integration, and exposure limits. Uses conftest fixtures with synthetic Parquet data.

## Project Structure

```
├── src/
│   ├── analysis/           # Analysis scripts
│   │   ├── kalshi/
│   │   └── polymarket/
│   ├── backtesting/        # Fork addition
│   │   ├── engine.py       # Trade replay engine
│   │   ├── strategy.py     # Strategy base class
│   │   ├── portfolio.py    # Position tracking + settlement
│   │   ├── metrics.py      # Performance metrics
│   │   ├── sizing.py       # Kelly + fixed sizing
│   │   ├── fees.py         # Platform fee models
│   │   ├── calibration.py  # Calibration curve fitting
│   │   └── strategies/
│   │       ├── longshot_fade.py
│   │       └── mispricing.py
│   ├── indexers/           # Data collection
│   └── common/             # Shared utilities
├── scripts/
│   └── fetch_sample_data.py
├── tests/
│   └── test_backtesting.py
├── docs/
└── output/
```

## Running the Project

Requires Python 3.9+. Install with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

**Full dataset (36GiB):**

```bash
make setup     # Download and extract
make analyze   # Select backtest_strategies
```

**Sample only (no large download):**

```bash
uv run python3 scripts/fetch_sample_data.py
make analyze
```

**Programmatic:**

```python
from src.backtesting import BacktestEngine, KalshiFees
from src.backtesting.strategies.longshot_fade import LongshotFade

engine = BacktestEngine(
    trades_dir="data/sample/trades",
    markets_dir="data/sample/markets",
    strategy=LongshotFade(),
    fee_model=KalshiFees(),
    initial_bankroll=10_000.0,
)
result = engine.run()
print(f"Return: {result.metrics.total_return_pct:.2f}%")
print(f"Sharpe: {result.metrics.sharpe_ratio:.2f}")
```

## License

See the original repository for license information.

## Citations

- Becker, J. (2026). _The Microstructure of Wealth Transfer in Prediction Markets_. https://jbecker.dev/research/prediction-market-microstructure
- Le, N. A. (2026). _Decomposing Crowd Wisdom: Domain-Specific Calibration Dynamics in Prediction Markets_. https://arxiv.org/abs/2602.19520
