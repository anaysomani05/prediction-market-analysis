"""Backtesting engine: chronological trade replay over Parquet data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from src.backtesting.fees import FeeModel, KalshiFees, NoFees
from src.backtesting.metrics import BacktestMetrics, compute_metrics
from src.backtesting.portfolio import Portfolio
from src.backtesting.sizing import FixedSizer, PositionSizer
from src.backtesting.strategy import Signal, Strategy, StrategyDecision, TradeContext


@dataclass
class BacktestResult:
    """Full output from a backtest run."""

    strategy_name: str
    metrics: BacktestMetrics
    portfolio: Portfolio


class BacktestEngine:
    """Replays historical trades chronologically through a strategy.

    Usage:
        engine = BacktestEngine(
            trades_dir="data/kalshi/trades",
            markets_dir="data/kalshi/markets",
            strategy=MyStrategy(),
        )
        result = engine.run()
        print(result.metrics)
    """

    def __init__(
        self,
        trades_dir: Path | str,
        markets_dir: Path | str,
        strategy: Strategy,
        fee_model: FeeModel | None = None,
        sizer: PositionSizer | None = None,
        initial_bankroll: float = 10_000.0,
        category_map: dict[str, str] | None = None,
        max_exposure_per_market: float = 0.0,
    ):
        self.trades_dir = Path(trades_dir)
        self.markets_dir = Path(markets_dir)
        self.strategy = strategy
        self.fee_model = fee_model or KalshiFees()
        self.sizer = sizer or FixedSizer()
        self.initial_bankroll = initial_bankroll
        self.category_map = category_map
        self.max_exposure_per_market = max_exposure_per_market

    def run(self) -> BacktestResult:
        """Execute the backtest: load data, replay trades, resolve markets."""
        con = duckdb.connect()

        # Load resolved markets into a lookup
        markets_df = con.execute(
            f"""
            SELECT ticker, result, event_ticker
            FROM '{self.markets_dir}/*.parquet'
            WHERE status = 'finalized'
              AND result IN ('yes', 'no')
            """
        ).df()

        market_results = dict(zip(markets_df["ticker"], markets_df["result"]))
        market_events = dict(zip(markets_df["ticker"], markets_df["event_ticker"]))

        # Auto-build category map from event_ticker if not provided
        if self.category_map is None:
            try:
                from src.analysis.kalshi.util.categories import get_group
                self.category_map = {}
                for ticker, event_ticker in market_events.items():
                    if event_ticker:
                        self.category_map[ticker] = get_group(event_ticker)
                    else:
                        self.category_map[ticker] = "Other"
            except ImportError:
                pass

        # Load all trades for resolved markets, sorted chronologically
        resolved_tickers = list(market_results.keys())
        if not resolved_tickers:
            portfolio = Portfolio(initial_bankroll=self.initial_bankroll)
            return BacktestResult(
                strategy_name=self.strategy.name,
                metrics=compute_metrics([], self.initial_bankroll, [self.initial_bankroll]),
                portfolio=portfolio,
            )

        trades_df = con.execute(
            f"""
            SELECT trade_id, ticker, count, yes_price, no_price, taker_side, created_time
            FROM '{self.trades_dir}/*.parquet'
            WHERE ticker IN ({','.join(f"'{t}'" for t in resolved_tickers)})
            ORDER BY created_time ASC
            """
        ).df()

        # Run replay
        portfolio = Portfolio(
            initial_bankroll=self.initial_bankroll,
            max_exposure_per_market=self.max_exposure_per_market,
        )
        equity_curve = [self.initial_bankroll]
        resolved_so_far: set[str] = set()

        # Track which trades have been seen per market (for resolution timing)
        # We resolve a market after processing its last trade
        last_trade_per_market: dict[str, int] = {}
        for idx, row in trades_df.iterrows():
            last_trade_per_market[row["ticker"]] = idx

        for idx, row in trades_df.iterrows():
            ticker = row["ticker"]

            ctx = TradeContext(
                trade_id=str(row["trade_id"]),
                ticker=ticker,
                yes_price=int(row["yes_price"]),
                no_price=int(row["no_price"]),
                taker_side=row["taker_side"],
                count=int(row["count"]),
                timestamp=str(row["created_time"]),
                event_ticker=market_events.get(ticker),
            )

            # Feed current bankroll to strategy before decision
            self.strategy.update_bankroll(portfolio.bankroll)

            # Strategy decision (no peeking at result)
            decision: StrategyDecision = self.strategy.on_trade(ctx)

            if decision.signal != Signal.SKIP and decision.contracts > 0:
                side = "yes" if decision.signal == Signal.BUY_YES else "no"
                price = ctx.yes_price if side == "yes" else ctx.no_price

                # Let sizer override contract count if strategy didn't set a specific size
                if decision.contracts == 0:
                    edge = 0.0  # strategy should estimate this
                    contracts = self.sizer.size(price, edge, portfolio.bankroll)
                else:
                    contracts = decision.contracts

                entry_fee = self.fee_model.entry_fee(price, contracts)
                portfolio.add_position(ticker, side, contracts, price, entry_fee)

            # Resolve market after its last trade
            if idx == last_trade_per_market.get(ticker) and ticker not in resolved_so_far:
                result = market_results[ticker]

                def settlement_fee_fn(profit_cents, num_contracts):
                    if isinstance(self.fee_model, KalshiFees):
                        return self.fee_model.profit_fee(profit_cents, num_contracts)
                    return self.fee_model.settlement_fee(int(profit_cents), num_contracts)

                portfolio.resolve_market(ticker, result, settlement_fee_fn)
                self.strategy.on_resolution(ticker, result)
                resolved_so_far.add(ticker)
                equity_curve.append(portfolio.equity)

        # Resolve any remaining open positions (markets we traded but haven't resolved)
        for ticker in list(portfolio.open_positions.keys()):
            if ticker in market_results and ticker not in resolved_so_far:
                result = market_results[ticker]

                def settlement_fee_fn(profit_cents, num_contracts):
                    if isinstance(self.fee_model, KalshiFees):
                        return self.fee_model.profit_fee(profit_cents, num_contracts)
                    return self.fee_model.settlement_fee(int(profit_cents), num_contracts)

                portfolio.resolve_market(ticker, result, settlement_fee_fn)
                self.strategy.on_resolution(ticker, result)
                equity_curve.append(portfolio.equity)

        metrics = compute_metrics(
            portfolio.closed_positions,
            self.initial_bankroll,
            equity_curve,
            self.category_map,
        )

        return BacktestResult(
            strategy_name=self.strategy.name,
            metrics=metrics,
            portfolio=portfolio,
        )
