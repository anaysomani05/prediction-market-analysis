"""Performance metrics for prediction market backtesting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.backtesting.portfolio import ClosedPosition


@dataclass
class BacktestMetrics:
    """Summary performance metrics from a backtest run."""

    total_pnl: float
    total_return_pct: float
    num_trades: int
    win_rate: float
    avg_pnl_per_trade: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    profit_factor: float
    avg_winner: float
    avg_loser: float
    total_fees: float
    pnl_by_category: dict[str, float]
    equity_curve: list[float]


def compute_metrics(
    closed_positions: list[ClosedPosition],
    initial_bankroll: float,
    equity_curve: list[float],
    category_map: dict[str, str] | None = None,
) -> BacktestMetrics:
    """Compute backtest performance metrics from closed positions.

    Args:
        closed_positions: List of resolved positions.
        initial_bankroll: Starting bankroll.
        equity_curve: Equity value after each trade resolution.
        category_map: Optional ticker -> category mapping for P&L breakdown.
    """
    if not closed_positions:
        return BacktestMetrics(
            total_pnl=0.0, total_return_pct=0.0, num_trades=0, win_rate=0.0,
            avg_pnl_per_trade=0.0, sharpe_ratio=0.0, max_drawdown=0.0,
            max_drawdown_pct=0.0, profit_factor=0.0, avg_winner=0.0,
            avg_loser=0.0, total_fees=0.0, pnl_by_category={},
            equity_curve=equity_curve,
        )

    pnls = np.array([p.net_pnl for p in closed_positions])
    winners = pnls[pnls > 0]
    losers = pnls[pnls < 0]

    total_pnl = float(pnls.sum())
    num_trades = len(pnls)
    win_rate = float(np.mean(pnls > 0))

    # Sharpe: annualize assuming ~250 trades/year as rough baseline
    if len(pnls) > 1 and pnls.std() > 0:
        sharpe = float(pnls.mean() / pnls.std() * np.sqrt(min(num_trades, 250)))
    else:
        sharpe = 0.0

    # Max drawdown from equity curve
    eq = np.array(equity_curve)
    if len(eq) > 1:
        running_max = np.maximum.accumulate(eq)
        drawdowns = eq - running_max
        max_dd = float(abs(drawdowns.min()))
        max_dd_pct = float(abs((drawdowns / running_max).min())) if running_max.max() > 0 else 0.0
    else:
        max_dd = 0.0
        max_dd_pct = 0.0

    # Profit factor
    gross_profit = float(winners.sum()) if len(winners) > 0 else 0.0
    gross_loss = float(abs(losers.sum())) if len(losers) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

    # Fees
    total_fees = sum(p.settlement_fees for p in closed_positions)

    # P&L by category
    pnl_by_category: dict[str, float] = {}
    if category_map:
        for pos in closed_positions:
            cat = category_map.get(pos.ticker, "unknown")
            pnl_by_category[cat] = pnl_by_category.get(cat, 0.0) + pos.net_pnl

    return BacktestMetrics(
        total_pnl=round(total_pnl, 2),
        total_return_pct=round(total_pnl / initial_bankroll * 100, 2) if initial_bankroll > 0 else 0.0,
        num_trades=num_trades,
        win_rate=round(win_rate, 4),
        avg_pnl_per_trade=round(float(pnls.mean()), 4),
        sharpe_ratio=round(sharpe, 4),
        max_drawdown=round(max_dd, 2),
        max_drawdown_pct=round(max_dd_pct, 4),
        profit_factor=round(profit_factor, 4) if profit_factor != float('inf') else float('inf'),
        avg_winner=round(float(winners.mean()), 4) if len(winners) > 0 else 0.0,
        avg_loser=round(float(losers.mean()), 4) if len(losers) > 0 else 0.0,
        total_fees=round(total_fees, 2),
        pnl_by_category={k: round(v, 2) for k, v in pnl_by_category.items()},
        equity_curve=equity_curve,
    )
