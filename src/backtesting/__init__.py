"""Prediction market backtesting framework."""

from src.backtesting.calibration import calibration_curve_data, fit_calibration
from src.backtesting.engine import BacktestEngine
from src.backtesting.fees import FeeModel, KalshiFees, PolymarketFees
from src.backtesting.metrics import compute_metrics
from src.backtesting.portfolio import Portfolio
from src.backtesting.sizing import KellySizer, FixedSizer, PositionSizer
from src.backtesting.strategy import Signal, Strategy, TradeContext

__all__ = [
    "calibration_curve_data",
    "BacktestEngine",
    "FeeModel",
    "FixedSizer",
    "KalshiFees",
    "KellySizer",
    "PolymarketFees",
    "Portfolio",
    "PositionSizer",
    "Signal",
    "Strategy",
    "TradeContext",
    "compute_metrics",
    "fit_calibration",
]
