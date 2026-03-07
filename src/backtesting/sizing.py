"""Position sizing models including Kelly criterion."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


class PositionSizer(ABC):
    """Base class for position sizing."""

    @abstractmethod
    def size(self, price_cents: int, edge: float, bankroll: float) -> int:
        """Determine number of contracts to buy.

        Args:
            price_cents: Price in cents (1-99).
            edge: Estimated edge (win_prob - implied_prob).
            bankroll: Current bankroll in dollars.

        Returns:
            Number of contracts to purchase.
        """
        ...


class FixedSizer(PositionSizer):
    """Fixed dollar amount per trade."""

    def __init__(self, dollars_per_trade: float = 10.0):
        self.dollars_per_trade = dollars_per_trade

    def size(self, price_cents: int, edge: float, bankroll: float) -> int:
        if price_cents <= 0 or bankroll < self.dollars_per_trade:
            return 0
        cost_per_contract = price_cents / 100.0
        return max(1, int(self.dollars_per_trade / cost_per_contract))


class KellySizer(PositionSizer):
    """Kelly criterion position sizing for binary outcomes.

    Full Kelly: f* = (bp - q) / b
    where b = (100/price - 1), p = implied_prob + edge, q = 1 - p

    Uses fractional Kelly (default 0.25) for safety.
    """

    def __init__(self, fraction: float = 0.25, max_bet_pct: float = 0.05):
        self.fraction = fraction
        self.max_bet_pct = max_bet_pct  # max % of bankroll per bet

    def size(self, price_cents: int, edge: float, bankroll: float) -> int:
        if price_cents <= 0 or price_cents >= 100 or edge <= 0 or bankroll <= 0:
            return 0

        implied_prob = price_cents / 100.0
        win_prob = min(implied_prob + edge, 0.99)
        lose_prob = 1.0 - win_prob

        # Odds: win pays (100 - price) cents per price cents risked
        b = (100.0 - price_cents) / price_cents

        # Kelly fraction: f* = (b*p - q) / b
        kelly_f = (b * win_prob - lose_prob) / b
        if kelly_f <= 0:
            return 0

        # Apply fractional Kelly and cap
        bet_fraction = min(kelly_f * self.fraction, self.max_bet_pct)
        bet_dollars = bankroll * bet_fraction
        cost_per_contract = price_cents / 100.0

        return max(1, int(bet_dollars / cost_per_contract))
