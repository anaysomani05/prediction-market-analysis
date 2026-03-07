"""Tests for the backtesting framework using conftest.py fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.backtesting.engine import BacktestEngine, BacktestResult
from src.backtesting.fees import KalshiFees, NoFees, PolymarketFees
from src.backtesting.metrics import compute_metrics
from src.backtesting.portfolio import Portfolio
from src.backtesting.sizing import FixedSizer, KellySizer
from src.backtesting.strategy import Signal, Strategy, StrategyDecision, TradeContext
from src.backtesting.strategies.longshot_fade import LongshotFade
from src.backtesting.strategies.mispricing import MispricingStrategy


# -- Simple test strategy --

class AlwaysBuyYes(Strategy):
    """Buy YES on every trade for testing."""

    def __init__(self):
        super().__init__(name="always_buy_yes")

    def on_trade(self, ctx: TradeContext) -> StrategyDecision:
        return StrategyDecision(
            signal=Signal.BUY_YES,
            contracts=1,
            price=ctx.yes_price,
        )


class AlwaysBuyNo(Strategy):
    """Buy NO on every trade for testing."""

    def __init__(self):
        super().__init__(name="always_buy_no")

    def on_trade(self, ctx: TradeContext) -> StrategyDecision:
        return StrategyDecision(
            signal=Signal.BUY_NO,
            contracts=1,
            price=ctx.no_price,
        )


class AlwaysSkip(Strategy):
    """Skip every trade."""

    def __init__(self):
        super().__init__(name="always_skip")

    def on_trade(self, ctx: TradeContext) -> StrategyDecision:
        return StrategyDecision(signal=Signal.SKIP)


# -- Portfolio tests --

class TestPortfolio:
    def test_add_and_resolve_winner(self):
        p = Portfolio(initial_bankroll=1000.0)
        p.add_position("MKT-A", "yes", 10, 40, 0.0)
        assert p.bankroll == pytest.approx(996.0)  # 10 * 0.40 = 4.00
        assert "MKT-A" in p.open_positions

        closed = p.resolve_market("MKT-A", "yes")
        assert closed is not None
        assert closed.won is True
        assert closed.payout == 10.0  # 10 contracts * $1
        assert p.bankroll == pytest.approx(1006.0)  # 996 + 10

    def test_add_and_resolve_loser(self):
        p = Portfolio(initial_bankroll=1000.0)
        p.add_position("MKT-A", "yes", 10, 60, 0.0)
        assert p.bankroll == pytest.approx(994.0)

        closed = p.resolve_market("MKT-A", "no")
        assert closed is not None
        assert closed.won is False
        assert closed.payout == 0.0
        assert closed.net_pnl == pytest.approx(-6.0)

    def test_bankroll_cap(self):
        p = Portfolio(initial_bankroll=5.0)
        p.add_position("MKT-A", "yes", 100, 50, 0.0)
        # Can only afford 10 contracts at 50c
        assert p.open_positions["MKT-A"].contracts == 10

    def test_resolve_nonexistent(self):
        p = Portfolio(initial_bankroll=1000.0)
        assert p.resolve_market("FAKE", "yes") is None

    def test_multiple_positions(self):
        p = Portfolio(initial_bankroll=1000.0)
        p.add_position("MKT-A", "yes", 5, 30, 0.0)
        p.add_position("MKT-B", "no", 5, 40, 0.0)
        assert len(p.open_positions) == 2
        assert p.bankroll == pytest.approx(1000 - 5 * 0.30 - 5 * 0.40)

    def test_add_to_existing_position(self):
        p = Portfolio(initial_bankroll=1000.0)
        p.add_position("MKT-A", "yes", 5, 30, 0.0)
        p.add_position("MKT-A", "yes", 5, 40, 0.0)
        assert p.open_positions["MKT-A"].contracts == 10
        assert p.open_positions["MKT-A"].avg_price_cents == pytest.approx(35.0)


# -- Fee model tests --

class TestFees:
    def test_kalshi_profit_fee(self):
        fees = KalshiFees(profit_fee_rate=0.07)
        # Win: bought at 40c, won -> profit = 60c per contract
        fee = fees.profit_fee(60.0, 10)
        assert fee == pytest.approx(0.07 * 60.0 * 10 / 100.0)

    def test_kalshi_no_fee_on_loss(self):
        fees = KalshiFees()
        assert fees.profit_fee(-40.0, 10) == 0.0

    def test_polymarket_settlement(self):
        fees = PolymarketFees(winnings_fee_rate=0.02)
        fee = fees.settlement_fee(100, 10)  # $10 payout
        assert fee == pytest.approx(0.02 * 100 * 10 / 100.0)

    def test_no_fees(self):
        fees = NoFees()
        assert fees.entry_fee(50, 10) == 0.0
        assert fees.settlement_fee(100, 10) == 0.0


# -- Sizing tests --

class TestSizing:
    def test_fixed_sizer(self):
        sizer = FixedSizer(dollars_per_trade=10.0)
        # At 50c, $10 buys 20 contracts
        assert sizer.size(50, 0.05, 1000.0) == 20

    def test_fixed_sizer_bankroll_limit(self):
        sizer = FixedSizer(dollars_per_trade=100.0)
        assert sizer.size(50, 0.05, 5.0) == 0

    def test_kelly_positive_edge(self):
        sizer = KellySizer(fraction=0.25, max_bet_pct=0.05)
        contracts = sizer.size(50, 0.10, 10_000.0)
        assert contracts > 0

    def test_kelly_no_edge(self):
        sizer = KellySizer()
        assert sizer.size(50, 0.0, 10_000.0) == 0

    def test_kelly_negative_edge(self):
        sizer = KellySizer()
        assert sizer.size(50, -0.05, 10_000.0) == 0


# -- Engine integration tests --

class TestEngine:
    def test_always_buy_yes(self, kalshi_trades_dir: Path, kalshi_markets_dir: Path):
        engine = BacktestEngine(
            trades_dir=kalshi_trades_dir,
            markets_dir=kalshi_markets_dir,
            strategy=AlwaysBuyYes(),
            fee_model=NoFees(),
            initial_bankroll=10_000.0,
        )
        result = engine.run()
        assert isinstance(result, BacktestResult)
        assert result.metrics.num_trades > 0
        assert len(result.metrics.equity_curve) > 1
        assert result.strategy_name == "always_buy_yes"

    def test_always_buy_no(self, kalshi_trades_dir: Path, kalshi_markets_dir: Path):
        engine = BacktestEngine(
            trades_dir=kalshi_trades_dir,
            markets_dir=kalshi_markets_dir,
            strategy=AlwaysBuyNo(),
            fee_model=NoFees(),
            initial_bankroll=10_000.0,
        )
        result = engine.run()
        assert result.metrics.num_trades > 0

    def test_always_skip(self, kalshi_trades_dir: Path, kalshi_markets_dir: Path):
        engine = BacktestEngine(
            trades_dir=kalshi_trades_dir,
            markets_dir=kalshi_markets_dir,
            strategy=AlwaysSkip(),
            fee_model=NoFees(),
            initial_bankroll=10_000.0,
        )
        result = engine.run()
        assert result.metrics.num_trades == 0
        assert result.metrics.total_pnl == 0.0

    def test_with_kalshi_fees(self, kalshi_trades_dir: Path, kalshi_markets_dir: Path):
        engine = BacktestEngine(
            trades_dir=kalshi_trades_dir,
            markets_dir=kalshi_markets_dir,
            strategy=AlwaysBuyYes(),
            fee_model=KalshiFees(),
            initial_bankroll=10_000.0,
        )
        result = engine.run()
        assert result.metrics.num_trades > 0
        # With fees, P&L should be lower than without
        engine_nofee = BacktestEngine(
            trades_dir=kalshi_trades_dir,
            markets_dir=kalshi_markets_dir,
            strategy=AlwaysBuyYes(),
            fee_model=NoFees(),
            initial_bankroll=10_000.0,
        )
        result_nofee = engine_nofee.run()
        assert result.metrics.total_pnl <= result_nofee.metrics.total_pnl

    def test_longshot_fade(self, kalshi_trades_dir: Path, kalshi_markets_dir: Path):
        engine = BacktestEngine(
            trades_dir=kalshi_trades_dir,
            markets_dir=kalshi_markets_dir,
            strategy=LongshotFade(low_threshold=15, high_threshold=85),
            fee_model=KalshiFees(),
            initial_bankroll=10_000.0,
        )
        result = engine.run()
        assert isinstance(result, BacktestResult)
        # Fixture has prices at 10, 20, 30, 50, 60, 70, 90 — should trade at 10 and 90
        assert result.metrics.num_trades > 0

    def test_mispricing_strategy(self, kalshi_trades_dir: Path, kalshi_markets_dir: Path):
        engine = BacktestEngine(
            trades_dir=kalshi_trades_dir,
            markets_dir=kalshi_markets_dir,
            strategy=MispricingStrategy(),
            fee_model=KalshiFees(),
            initial_bankroll=10_000.0,
        )
        result = engine.run()
        assert isinstance(result, BacktestResult)

    def test_bankroll_never_negative(self, kalshi_trades_dir: Path, kalshi_markets_dir: Path):
        engine = BacktestEngine(
            trades_dir=kalshi_trades_dir,
            markets_dir=kalshi_markets_dir,
            strategy=AlwaysBuyYes(),
            fee_model=NoFees(),
            initial_bankroll=100.0,  # small bankroll
        )
        result = engine.run()
        assert all(e >= 0 for e in result.metrics.equity_curve)

    def test_equity_curve_starts_at_initial(self, kalshi_trades_dir: Path, kalshi_markets_dir: Path):
        engine = BacktestEngine(
            trades_dir=kalshi_trades_dir,
            markets_dir=kalshi_markets_dir,
            strategy=AlwaysBuyYes(),
            fee_model=NoFees(),
            initial_bankroll=5_000.0,
        )
        result = engine.run()
        assert result.metrics.equity_curve[0] == 5_000.0


# -- Max exposure tests --

class TestMaxExposure:
    def test_max_exposure_limits_position(self):
        p = Portfolio(initial_bankroll=10_000.0, max_exposure_per_market=5.0)
        # Try to add $50 position, should be capped to ~$5
        p.add_position("MKT-A", "yes", 100, 50, 0.0)
        assert p.open_positions["MKT-A"].contracts == 10  # 10 * 0.50 = $5

    def test_max_exposure_blocks_additional(self):
        p = Portfolio(initial_bankroll=10_000.0, max_exposure_per_market=5.0)
        p.add_position("MKT-A", "yes", 10, 50, 0.0)  # $5.00 exactly
        p.add_position("MKT-A", "yes", 10, 50, 0.0)  # should be blocked
        assert p.open_positions["MKT-A"].contracts == 10  # unchanged

    def test_engine_with_max_exposure(self, kalshi_trades_dir: Path, kalshi_markets_dir: Path):
        engine = BacktestEngine(
            trades_dir=kalshi_trades_dir,
            markets_dir=kalshi_markets_dir,
            strategy=AlwaysBuyYes(),
            fee_model=NoFees(),
            initial_bankroll=10_000.0,
            max_exposure_per_market=50.0,
        )
        result = engine.run()
        assert result.metrics.num_trades > 0


# -- Strategy bankroll tracking --

class TestStrategyBankroll:
    def test_strategy_receives_bankroll(self, kalshi_trades_dir: Path, kalshi_markets_dir: Path):
        class BankrollTracker(Strategy):
            def __init__(self):
                super().__init__(name="bankroll_tracker")
                self.bankrolls = []

            def on_trade(self, ctx):
                self.bankrolls.append(self.bankroll)
                return StrategyDecision(signal=Signal.SKIP)

        strat = BankrollTracker()
        engine = BacktestEngine(
            trades_dir=kalshi_trades_dir,
            markets_dir=kalshi_markets_dir,
            strategy=strat,
            fee_model=NoFees(),
            initial_bankroll=10_000.0,
        )
        engine.run()
        assert len(strat.bankrolls) > 0
        assert strat.bankrolls[0] == 10_000.0


# -- Category mapping --

class TestCategoryMap:
    def test_auto_category_map(self, kalshi_trades_dir: Path, kalshi_markets_dir: Path):
        engine = BacktestEngine(
            trades_dir=kalshi_trades_dir,
            markets_dir=kalshi_markets_dir,
            strategy=AlwaysBuyYes(),
            fee_model=NoFees(),
            initial_bankroll=10_000.0,
        )
        result = engine.run()
        # Fixture markets have event_tickers INXD-24JAN01 and NFLGAME-25FEB01
        assert len(result.metrics.pnl_by_category) > 0


# -- Calibration --

class TestCalibration:
    def test_fit_calibration(self, kalshi_trades_dir: Path, kalshi_markets_dir: Path):
        from src.backtesting.calibration import fit_calibration
        cal = fit_calibration(kalshi_trades_dir, kalshi_markets_dir, bin_width=10, min_samples=5)
        assert isinstance(cal, dict)
        # Should have some price bins
        assert len(cal) > 0
        for price, edge in cal.items():
            assert 1 <= price <= 99
            assert -1.0 <= edge <= 1.0

    def test_calibration_curve_data(self, kalshi_trades_dir: Path, kalshi_markets_dir: Path):
        from src.backtesting.calibration import calibration_curve_data
        df = calibration_curve_data(kalshi_trades_dir, kalshi_markets_dir, bin_width=10, min_samples=5)
        assert len(df) > 0
        assert "actual_win_rate" in df.columns
        assert "implied_prob" in df.columns
        assert "n" in df.columns


# -- Longshot fade once-per-market --

class TestLongshotFadeOncePerMarket:
    def test_trades_each_market_once(self, kalshi_trades_dir: Path, kalshi_markets_dir: Path):
        strat = LongshotFade(low_threshold=15, high_threshold=85)
        engine = BacktestEngine(
            trades_dir=kalshi_trades_dir,
            markets_dir=kalshi_markets_dir,
            strategy=strat,
            fee_model=NoFees(),
            initial_bankroll=10_000.0,
        )
        result = engine.run()
        # Should have at most 2 closed positions (2 markets in fixture)
        assert result.metrics.num_trades <= 2


# -- Metrics tests --

class TestMetrics:
    def test_empty_positions(self):
        metrics = compute_metrics([], 10_000.0, [10_000.0])
        assert metrics.num_trades == 0
        assert metrics.total_pnl == 0.0
        assert metrics.sharpe_ratio == 0.0

    def test_win_rate(self):
        from src.backtesting.portfolio import ClosedPosition
        positions = [
            ClosedPosition("A", "yes", 1, 50, 0.5, 1.0, 0.0, 0.5, "yes", True),
            ClosedPosition("B", "yes", 1, 50, 0.5, 0.0, 0.0, -0.5, "no", False),
        ]
        metrics = compute_metrics(positions, 1000.0, [1000.0, 1000.5, 1000.0])
        assert metrics.win_rate == 0.5
        assert metrics.num_trades == 2
