"""Mispricing strategy: trade based on historical calibration gaps.

Uses the empirical observation that prediction markets are miscalibrated
at certain price points — e.g., contracts priced at 30c might win 35% of
the time, creating a systematic edge for YES buyers at that price.

Can use either hardcoded defaults or a fitted calibration curve from
`src.backtesting.calibration.fit_calibration()`.
"""

from __future__ import annotations

from src.backtesting.sizing import KellySizer, PositionSizer
from src.backtesting.strategy import Signal, Strategy, StrategyDecision, TradeContext

# Conservative default offsets based on published prediction market research.
# These are intentionally small — use fit_calibration() for data-driven values.
DEFAULT_CALIBRATION = {
    10: -0.04,
    20: -0.02,
    30: 0.02,
    40: 0.03,
    50: 0.01,
    60: 0.02,
    70: 0.01,
    80: -0.01,
    90: -0.03,
}


class MispricingStrategy(Strategy):
    """Trade contracts where calibration data shows consistent mispricing."""

    def __init__(
        self,
        calibration: dict[int, float] | None = None,
        min_edge: float = 0.02,
        sizer: PositionSizer | None = None,
        max_contracts_per_trade: int = 50,
    ):
        super().__init__(name="mispricing")
        self.calibration = calibration or DEFAULT_CALIBRATION
        self.min_edge = min_edge
        self.sizer = sizer or KellySizer(fraction=0.15, max_bet_pct=0.02)
        self.max_contracts_per_trade = max_contracts_per_trade
        self._traded_markets: set[str] = set()

    def _lookup_edge(self, price_cents: int) -> float:
        """Interpolate calibration edge for a given price."""
        if price_cents in self.calibration:
            return self.calibration[price_cents]
        prices = sorted(self.calibration.keys())
        if not prices:
            return 0.0
        if price_cents <= prices[0]:
            return self.calibration[prices[0]]
        if price_cents >= prices[-1]:
            return self.calibration[prices[-1]]
        for i in range(len(prices) - 1):
            if prices[i] <= price_cents <= prices[i + 1]:
                frac = (price_cents - prices[i]) / (prices[i + 1] - prices[i])
                return self.calibration[prices[i]] * (1 - frac) + self.calibration[prices[i + 1]] * frac
        return 0.0

    def on_trade(self, ctx: TradeContext) -> StrategyDecision:
        # Only enter each market once
        if ctx.ticker in self._traded_markets:
            return StrategyDecision(signal=Signal.SKIP)

        edge = self._lookup_edge(ctx.yes_price)
        bankroll = self.bankroll if self.bankroll > 0 else 10_000.0

        if edge >= self.min_edge:
            contracts = self.sizer.size(ctx.yes_price, edge, bankroll)
            contracts = min(contracts, self.max_contracts_per_trade, ctx.count)
            if contracts <= 0:
                return StrategyDecision(signal=Signal.SKIP)
            self._traded_markets.add(ctx.ticker)
            return StrategyDecision(
                signal=Signal.BUY_YES,
                contracts=contracts,
                price=ctx.yes_price,
            )

        if edge <= -self.min_edge:
            no_edge = abs(edge)
            contracts = self.sizer.size(ctx.no_price, no_edge, bankroll)
            contracts = min(contracts, self.max_contracts_per_trade, ctx.count)
            if contracts <= 0:
                return StrategyDecision(signal=Signal.SKIP)
            self._traded_markets.add(ctx.ticker)
            return StrategyDecision(
                signal=Signal.BUY_NO,
                contracts=min(contracts, ctx.count),
                price=ctx.no_price,
            )

        return StrategyDecision(signal=Signal.SKIP)

    def on_resolution(self, ticker: str, result: str) -> None:
        self._traded_markets.discard(ticker)
