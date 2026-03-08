"""Backtest analysis: run trading strategies on historical Kalshi data.

Runs longshot-fade and mispricing strategies through the backtesting engine,
produces equity curves, compares performance metrics, and shows P&L by category.

Uses a proper train/test split: the calibration curve is fitted on markets
that closed *before* a cutoff date, then the fitted strategy is evaluated
only on markets that closed *after* that date.  This prevents look-ahead
bias and gives an honest out-of-sample estimate.
"""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtesting.calibration import calibration_curve_data, fit_calibration
from src.backtesting.engine import BacktestEngine
from src.backtesting.fees import FeeModel, KalshiFees, NoFees, PolymarketFees
from src.backtesting.metrics import BacktestMetrics
from src.backtesting.sizing import FixedSizer, KellySizer
from src.backtesting.strategies.longshot_fade import LongshotFade
from src.backtesting.strategies.mispricing import MispricingStrategy
from src.common.analysis import Analysis, AnalysisOutput
from src.common.interfaces.chart import ChartConfig, ChartType, UnitType


def _find_train_test_cutoff(markets_dir: Path | str) -> str:
    """Pick a cutoff so the last time-window becomes the test set.

    Finds the median close_time of the last third of markets
    (sorted by close_time) and returns the boundary timestamp.
    """
    con = duckdb.connect()
    # Get distinct close months and pick the start of the last window
    df = con.execute(
        f"""
        SELECT close_time
        FROM '{markets_dir}/*.parquet'
        WHERE status = 'finalized' AND result IN ('yes', 'no')
        ORDER BY close_time
        """
    ).df()
    if len(df) < 10:
        # Not enough data to split — use midpoint
        mid = len(df) // 2
        return str(df.iloc[mid]["close_time"])
    # Use the last third as test set
    cutoff_idx = len(df) * 2 // 3
    return str(df.iloc[cutoff_idx]["close_time"])


class BacktestStrategiesAnalysis(Analysis):
    """Run and compare backtesting strategies on prediction market data.

    Runs each platform (Kalshi, Polymarket) separately with its own fee
    model, then reports all results together.
    """

    def __init__(
        self,
        trades_dir: Path | str | None = None,
        markets_dir: Path | str | None = None,
        poly_trades_dir: Path | str | None = None,
        poly_markets_dir: Path | str | None = None,
        initial_bankroll: float = 10_000.0,
    ):
        super().__init__(
            name="backtest_strategies",
            description="Backtest trading strategies on prediction market data",
        )
        base_dir = Path(__file__).parent.parent.parent.parent
        self.trades_dir = Path(trades_dir or base_dir / "data" / "kalshi" / "trades")
        self.markets_dir = Path(markets_dir or base_dir / "data" / "kalshi" / "markets")
        self.poly_trades_dir = Path(poly_trades_dir) if poly_trades_dir else None
        self.poly_markets_dir = Path(poly_markets_dir) if poly_markets_dir else None
        self.initial_bankroll = initial_bankroll

    def _run_platform(
        self,
        platform: str,
        trades_dir: Path,
        markets_dir: Path,
        fee_model: FeeModel,
        calibration: dict[int, float],
        cutoff_ts: str,
    ) -> list[tuple[str, BacktestMetrics]]:
        """Run all strategies for a single platform."""
        max_exposure = self.initial_bankroll * 0.05

        strategies = [
            (f"{platform} Longshot Fade", LongshotFade(), fee_model),
            (f"{platform} Mispricing/default", MispricingStrategy(), fee_model),
            (f"{platform} Mispricing/fitted", MispricingStrategy(calibration=calibration), fee_model),
        ]

        results = []
        for name, strat, fees in strategies:
            with self.progress(f"Running {name}"):
                engine = BacktestEngine(
                    trades_dir=trades_dir,
                    markets_dir=markets_dir,
                    strategy=strat,
                    fee_model=fees,
                    initial_bankroll=self.initial_bankroll,
                    max_exposure_per_market=max_exposure,
                    min_close_ts=cutoff_ts,
                )
                result = engine.run()
                results.append((name, result.metrics))

        return results

    def run(self) -> AnalysisOutput:
        # Build list of platforms to run
        platforms: list[tuple[str, Path, Path, FeeModel]] = [
            ("Kalshi", self.trades_dir, self.markets_dir, KalshiFees()),
        ]
        if (self.poly_trades_dir and self.poly_markets_dir
                and self.poly_trades_dir.exists() and self.poly_markets_dir.exists()):
            platforms.append(
                ("Polymarket", self.poly_trades_dir, self.poly_markets_dir, PolymarketFees()),
            )

        # Use the first platform for calibration fitting (largest dataset)
        train_trades = self.trades_dir
        train_markets = self.markets_dir

        with self.progress("Computing train/test split"):
            cutoff_ts = _find_train_test_cutoff(train_markets)

        with self.progress("Fitting calibration curve (train set only)"):
            calibration = fit_calibration(
                train_trades, train_markets, max_close_ts=cutoff_ts,
            )
            cal_df = calibration_curve_data(train_trades, train_markets)

        # Run each platform with its own fee model
        results: list[tuple[str, BacktestMetrics]] = []
        for platform, t_dir, m_dir, fees in platforms:
            with self.progress(f"Computing {platform} train/test split"):
                plat_cutoff = _find_train_test_cutoff(m_dir)
            plat_results = self._run_platform(
                platform, t_dir, m_dir, fees, calibration, plat_cutoff,
            )
            results.extend(plat_results)

        fig = self._create_figure(results, cal_df)
        df = self._create_dataframe(results)
        chart = self._create_chart(results)

        return AnalysisOutput(figure=fig, data=df, chart=chart)

    def _create_figure(self, results: list[tuple[str, BacktestMetrics]], cal_df: pd.DataFrame) -> plt.Figure:
        fig = plt.figure(figsize=(16, 16))
        gs = fig.add_gridspec(4, 2, hspace=0.40, wspace=0.3)

        colors = ["#2ecc71", "#27ae60", "#1abc9c", "#e74c3c", "#c0392b", "#e67e22"]

        # 1. Calibration curve — predicted vs realized probability
        ax = fig.add_subplot(gs[0, :])
        ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1.2, label="Perfect calibration")
        if not cal_df.empty:
            implied = cal_df["implied_prob"].values
            actual = cal_df["actual_win_rate"].values
            sizes = np.clip(cal_df["n"].values / cal_df["n"].max() * 200, 20, 200)
            ax.scatter(implied, actual, s=sizes, color="#3498db", alpha=0.7, edgecolors="white", linewidth=0.5, zorder=5)
            # Highlight mispricing zones (where actual != implied)
            for _, row in cal_df.iterrows():
                ip = row["implied_prob"]
                aw = row["actual_win_rate"]
                if abs(aw - ip) >= 0.02:
                    color = "#2ecc71" if aw > ip else "#e74c3c"
                    ax.plot([ip, ip], [ip, aw], color=color, linewidth=1.5, alpha=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("Market Implied Probability", fontsize=10)
        ax.set_ylabel("Realized Win Frequency", fontsize=10)
        ax.set_title("Calibration Curve: Where Markets Misprice", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        # Annotate: green = underpriced (buy YES edge), red = overpriced (buy NO edge)
        ax.text(0.15, 0.85, "Above line = underpriced\n(YES edge)", transform=ax.transAxes,
                fontsize=7, color="#2ecc71", ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#2ecc71", alpha=0.8))
        ax.text(0.85, 0.15, "Below line = overpriced\n(NO edge)", transform=ax.transAxes,
                fontsize=7, color="#e74c3c", ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#e74c3c", alpha=0.8))

        # 2. Equity curves
        ax = fig.add_subplot(gs[1, 0])
        for i, (name, m) in enumerate(results):
            ax.plot(m.equity_curve, label=name, color=colors[i], linewidth=1.5, alpha=0.85)
        ax.axhline(y=self.initial_bankroll, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title("Equity Curves")
        ax.set_xlabel("Trade Resolution #")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend(fontsize=6, loc="best")
        ax.grid(True, alpha=0.3)

        # 3. Return comparison bar chart
        ax = fig.add_subplot(gs[1, 1])
        names = [n for n, _ in results]
        returns = [m.total_return_pct for _, m in results]
        bar_colors = ["#2ecc71" if r > 0 else "#e74c3c" for r in returns]
        ax.bar(range(len(names)), returns, color=bar_colors, alpha=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=6)
        ax.set_title("Total Return (%)")
        ax.set_ylabel("Return %")
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)
        ax.grid(True, alpha=0.3, axis="y")

        # 4. P&L by category (for best strategy)
        ax = fig.add_subplot(gs[2, 0])
        best_idx = max(range(len(results)), key=lambda i: results[i][1].total_pnl)
        best_name, best_m = results[best_idx]
        if best_m.pnl_by_category:
            cats = sorted(best_m.pnl_by_category.items(), key=lambda x: x[1])
            cat_names = [c for c, _ in cats]
            cat_pnls = [p for _, p in cats]
            cat_colors = ["#2ecc71" if p > 0 else "#e74c3c" for p in cat_pnls]
            ax.barh(cat_names, cat_pnls, color=cat_colors, alpha=0.8)
            ax.set_title(f"P&L by Category ({best_name})")
            ax.set_xlabel("P&L ($)")
            ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.8)
        else:
            ax.text(0.5, 0.5, "No category data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("P&L by Category")
        ax.grid(True, alpha=0.3, axis="x")

        # 5. Win rate vs Sharpe scatter
        ax = fig.add_subplot(gs[2, 1])
        for i, (name, m) in enumerate(results):
            ax.scatter(m.win_rate * 100, m.sharpe_ratio, color=colors[i], s=100, zorder=5, label=name)
        ax.set_title("Win Rate vs Sharpe Ratio")
        ax.set_xlabel("Win Rate (%)")
        ax.set_ylabel("Sharpe Ratio")
        ax.legend(fontsize=6, loc="best")
        ax.grid(True, alpha=0.3)

        # 6. Summary table (spans bottom row)
        ax = fig.add_subplot(gs[3, :])
        ax.axis("off")
        col_labels = ["Strategy", "P&L", "Return", "Trades", "Win%", "Sharpe", "MaxDD%", "PF", "Fees"]
        table_data = []
        for name, m in results:
            pf = f"{m.profit_factor:.2f}" if not math.isinf(m.profit_factor) else "inf"
            table_data.append([
                name,
                f"${m.total_pnl:,.0f}",
                f"{m.total_return_pct:.1f}%",
                str(m.num_trades),
                f"{m.win_rate * 100:.1f}%",
                f"{m.sharpe_ratio:.2f}",
                f"{m.max_drawdown_pct * 100:.1f}%",
                pf,
                f"${m.total_fees:,.0f}",
            ])
        table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.5)

        plt.suptitle("Prediction Market Backtesting Results", fontsize=14, fontweight="bold")
        return fig

    def _create_dataframe(self, results: list[tuple[str, BacktestMetrics]]) -> pd.DataFrame:
        rows = []
        for name, m in results:
            row = {
                "strategy": name,
                "total_pnl": m.total_pnl,
                "total_return_pct": m.total_return_pct,
                "num_trades": m.num_trades,
                "win_rate": m.win_rate,
                "sharpe_ratio": m.sharpe_ratio,
                "max_drawdown": m.max_drawdown,
                "max_drawdown_pct": m.max_drawdown_pct,
                "profit_factor": m.profit_factor,
                "avg_winner": m.avg_winner,
                "avg_loser": m.avg_loser,
                "total_fees": m.total_fees,
            }
            # Add category P&L columns
            for cat, pnl in sorted(m.pnl_by_category.items()):
                row[f"pnl_{cat.lower().replace(' ', '_')}"] = pnl
            rows.append(row)
        return pd.DataFrame(rows)

    def _create_chart(self, results: list[tuple[str, BacktestMetrics]]) -> ChartConfig:
        chart_data = []
        for name, m in results:
            chart_data.append({
                "strategy": name,
                "total_return_pct": round(m.total_return_pct, 2),
                "sharpe_ratio": round(m.sharpe_ratio, 2),
                "win_rate": round(m.win_rate * 100, 1),
                "max_drawdown_pct": round(m.max_drawdown_pct * 100, 1),
            })

        return ChartConfig(
            type=ChartType.BAR,
            data=chart_data,
            xKey="strategy",
            yKeys=["total_return_pct"],
            title="Strategy Backtesting Results",
            yUnit=UnitType.PERCENT,
            xLabel="Strategy",
            yLabel="Total Return (%)",
        )
