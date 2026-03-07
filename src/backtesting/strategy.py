"""Strategy base class for prediction market backtesting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class Signal(Enum):
    BUY_YES = "buy_yes"
    BUY_NO = "buy_no"
    SKIP = "skip"


@dataclass
class TradeContext:
    """Context passed to strategy for each trade in the replay."""

    trade_id: str
    ticker: str
    yes_price: int  # cents (1-99)
    no_price: int  # cents (1-99)
    taker_side: str  # "yes" or "no"
    count: int  # number of contracts in this trade
    timestamp: str  # ISO timestamp
    # Market-level info (if available)
    event_ticker: str | None = None
    market_result: str | None = None  # only set after resolution (not during live replay)


@dataclass
class StrategyDecision:
    """What the strategy wants to do."""

    signal: Signal
    contracts: int = 0  # how many contracts to trade
    price: int = 0  # price in cents we're paying


class Strategy(ABC):
    """Base class for prediction market trading strategies.

    Subclasses implement `on_trade()` which receives each trade chronologically
    and returns a decision (buy yes, buy no, or skip).

    The engine calls `update_bankroll()` before each trade so strategies
    can use `self.bankroll` for position sizing.
    """

    def __init__(self, name: str):
        self.name = name
        self.bankroll: float = 0.0

    def update_bankroll(self, bankroll: float) -> None:
        """Called by the engine before each trade with the current bankroll."""
        self.bankroll = bankroll

    @abstractmethod
    def on_trade(self, ctx: TradeContext) -> StrategyDecision:
        """Evaluate a trade and decide whether to take a position.

        Args:
            ctx: Trade context with price/market info.

        Returns:
            StrategyDecision with signal, size, and price.
        """
        ...

    def on_resolution(self, ticker: str, result: str) -> None:
        """Called when a market resolves. Override for stateful strategies."""
        pass
