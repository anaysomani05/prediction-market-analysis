"""Fee models for prediction market platforms."""

from __future__ import annotations

from abc import ABC, abstractmethod


class FeeModel(ABC):
    """Base class for platform fee calculation."""

    @abstractmethod
    def entry_fee(self, price_cents: int, contracts: int) -> float:
        """Fee paid when entering a position (in dollars)."""
        ...

    @abstractmethod
    def settlement_fee(self, payout_cents: int, contracts: int) -> float:
        """Fee paid on winning settlement (in dollars)."""
        ...


class KalshiFees(FeeModel):
    """Kalshi fee model: no entry fee, 7% fee on profits capped at price paid.

    Example: buy YES at 40c, wins -> payout $1, profit 60c, fee = 0.07 * 60 = 4.2c
    Example: buy YES at 40c, loses -> payout $0, no fee
    """

    def __init__(self, profit_fee_rate: float = 0.07):
        self.profit_fee_rate = profit_fee_rate

    def entry_fee(self, price_cents: int, contracts: int) -> float:
        return 0.0

    def settlement_fee(self, payout_cents: int, contracts: int) -> float:
        # Kalshi only charges on profit (payout - cost), but we don't know cost here.
        # The engine handles this by passing profit directly.
        return 0.0

    def profit_fee(self, profit_cents: float, contracts: int) -> float:
        """Fee on net profit from a winning position."""
        if profit_cents <= 0:
            return 0.0
        return self.profit_fee_rate * profit_cents * contracts / 100.0


class PolymarketFees(FeeModel):
    """Polymarket fee model: no trading fees, 2% on winnings.

    Polymarket charges no maker/taker fees on CLOB but takes 2% of net winnings.
    """

    def __init__(self, winnings_fee_rate: float = 0.02):
        self.winnings_fee_rate = winnings_fee_rate

    def entry_fee(self, price_cents: int, contracts: int) -> float:
        return 0.0

    def settlement_fee(self, payout_cents: int, contracts: int) -> float:
        if payout_cents <= 0:
            return 0.0
        return self.winnings_fee_rate * payout_cents * contracts / 100.0


class NoFees(FeeModel):
    """Zero-fee model for baseline comparison."""

    def entry_fee(self, price_cents: int, contracts: int) -> float:
        return 0.0

    def settlement_fee(self, payout_cents: int, contracts: int) -> float:
        return 0.0
