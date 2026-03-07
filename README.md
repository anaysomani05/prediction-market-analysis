# Prediction Market Analysis

> Forked from [jon-becker/prediction-market-analysis](https://github.com/jon-becker/prediction-market-analysis) — the largest publicly available dataset of Polymarket and Kalshi prediction market data (36GiB of trades and market metadata). All original data collection infrastructure, analysis framework, and datasets are the work of [Jon Becker](https://github.com/jon-becker).

## Overview

A framework for collecting, analyzing, and backtesting prediction market data from Kalshi and Polymarket. The original project provides data collection indexers, Parquet-based storage, and an extensible analysis script framework. This fork adds a backtesting engine for developing and evaluating trading strategies on historical data.

## Features

- Pre-collected datasets from Polymarket and Kalshi (36GiB compressed)
- Data collection indexers for Kalshi API and Polymarket blockchain
- Parquet-based storage with automatic progress saving
- Extensible analysis script framework (`make analyze`)
- **Backtesting engine with strategy framework and research analytics** *(fork addition)*

## Additions in This Fork

### Backtesting Engine

Chronological trade replay engine (`src/backtesting/engine.py`) that walks through historical trades in timestamp order, feeding each to a pluggable strategy. Resolves positions against actual market outcomes with no lookahead bias. Supports per-market exposure limits to prevent concentration risk.

- DuckDB-powered Parquet queries for fast data loading
- Portfolio tracker with position averaging, bankroll enforcement, and settlement
- Platform fee models — Kalshi (7% profit fee) and Polymarket (2% winnings fee)
- Performance metrics — Sharpe ratio, max drawdown, profit factor, win rate, equity curves, P&L by category
- Kelly criterion position sizing with fractional Kelly (0.15x) and configurable max-bet caps

### Strategy Framework

Abstract `Strategy` base class (`src/backtesting/strategy.py`) with:
- `on_trade(ctx: TradeContext) -> StrategyDecision` — receives price, side, volume, ticker, and current bankroll for each trade
- `on_resolution(ticker, result)` — callback when a market resolves, for state cleanup
- Signal types: `BUY_YES`, `BUY_NO`, `SKIP`
- Real-time bankroll tracking — strategies can adapt sizing to current portfolio value

### Implemented Strategies

- **Longshot Fade** (`src/backtesting/strategies/longshot_fade.py`) — bets against extreme prices (below 15c or above 85c) where markets are systematically overconfident on longshots. Once-per-market entry with conservative Kelly sizing.
- **Calibration Mispricing** (`src/backtesting/strategies/mispricing.py`) — trades based on empirical calibration gaps between market-implied probability and actual win frequency. Accepts either hardcoded defaults or a fitted calibration curve from historical data. Interpolates edge between calibration points.

### Research Analytics

- **Calibration curve** (`src/backtesting/calibration.py`) — fits predicted probability vs realized win frequency from historical data, producing the core research plot that shows where and how markets misprice. Points sized by sample count, with mispricing zones highlighted.
- **6-panel backtest figure** — calibration curve, equity curves, return comparison, P&L by category, win rate vs Sharpe scatter, and summary metrics table
- **Fitted calibration offsets** — data-driven edge estimates fed directly into the mispricing strategy, replacing guesswork with empirical measurement

### Data Pipeline

- **Sample data fetcher** (`scripts/fetch_sample_data.py`) — pulls ~300 resolved markets across 3 time windows from the Kalshi API, so you can run backtests without downloading the full 36GiB dataset
- Auto-category mapping from event tickers using the existing categories utility

### Testing

33 tests (`tests/test_backtesting.py`) covering portfolio mechanics, fee models, position sizing, calibration fitting, strategy behavior, engine integration, exposure limits, and bankroll tracking. Uses conftest.py fixtures with synthetic Parquet data.

## Project Structure

```
├── src/
│   ├── analysis/           # Analysis scripts
│   │   ├── kalshi/         # Kalshi-specific analyses
│   │   └── polymarket/     # Polymarket-specific analyses
│   ├── backtesting/        # Backtesting engine (fork addition)
│   │   ├── engine.py       # Chronological trade replay engine
│   │   ├── strategy.py     # Strategy base class + signals
│   │   ├── portfolio.py    # Position tracking + P&L resolution
│   │   ├── metrics.py      # Sharpe, max drawdown, profit factor
│   │   ├── sizing.py       # Kelly criterion + fixed position sizing
│   │   ├── fees.py         # Kalshi/Polymarket fee models
│   │   ├── calibration.py  # Calibration curve fitting + plotting data
│   │   └── strategies/     # Built-in strategies
│   │       ├── longshot_fade.py
│   │       └── mispricing.py
│   ├── indexers/           # Data collection indexers
│   │   ├── kalshi/         # Kalshi API client and indexers
│   │   └── polymarket/     # Polymarket API/blockchain indexers
│   └── common/             # Shared utilities and interfaces
├── data/                   # Data directory (extracted from data.tar.zst)
├── scripts/
│   └── fetch_sample_data.py  # Fetch sample from Kalshi API
├── tests/
│   └── test_backtesting.py   # Backtesting framework tests
├── docs/                   # Documentation
└── output/                 # Analysis outputs (figures, CSVs)
```

## Running the Project

Requires Python 3.9+. Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

**Option A — Use the full dataset (36GiB):**

```bash
make setup     # Download and extract data
make analyze   # Run analyses (select backtest_strategies)
```

**Option B — Use a sample (no large download):**

```bash
uv run python3 scripts/fetch_sample_data.py   # Fetch ~300 markets from Kalshi API
make analyze                                    # Select backtest_strategies
```

**Programmatic usage:**

```python
from src.backtesting import BacktestEngine, KalshiFees
from src.backtesting.strategies.longshot_fade import LongshotFade

engine = BacktestEngine(
    trades_dir="data/sample/trades",
    markets_dir="data/sample/markets",
    strategy=LongshotFade(low_threshold=15, high_threshold=85),
    fee_model=KalshiFees(),
    initial_bankroll=10_000.0,
)
result = engine.run()
print(f"Return: {result.metrics.total_return_pct:.2f}%")
print(f"Sharpe: {result.metrics.sharpe_ratio:.2f}")
```

## Research Goals

- Quantify prediction market calibration — where do market prices diverge from actual outcome frequencies?
- Develop systematic strategies that exploit persistent mispricing patterns
- Measure the impact of platform fees on strategy viability
- Understand which market categories (politics, finance, sports, crypto) offer the most consistent edges

## Future Work

- Walk-forward optimization with train/test splits to avoid overfitting calibration
- Multi-platform backtesting (Polymarket support)
- Limit order simulation and slippage modeling
- Live paper trading integration with the Kalshi API
- More strategies: momentum, mean-reversion, cross-market arbitrage

## License

See the original repository for license information.

## Research & Citations

- Becker, J. (2026). _The Microstructure of Wealth Transfer in Prediction Markets_. https://jbecker.dev/research/prediction-market-microstructure
- Le, N. A. (2026). _Decomposing Crowd Wisdom: Domain-Specific Calibration Dynamics in Prediction Markets_. https://arxiv.org/abs/2602.19520
