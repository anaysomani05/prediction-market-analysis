"""Longshot fade strategy: bet against extreme-priced contracts.

Prediction markets systematically overprice longshots (contracts at <15c)
and underprice near-certainties (>85c). This strategy fades those extremes
by buying NO on cheap contracts and YES on expensive ones.

Key risk: buying NO at 90c risks $90 to win $10 — a single loss wipes
out many wins. We mitigate this with:
- Fractional Kelly sizing using actual bankroll
- Max contracts per trade cap
- Only trading once per market (no piling on)
"""

from __future__ import annotations

from src.backtesting.sizing import KellySizer, PositionSizer
from src.backtesting.strategy import Signal, Strategy, StrategyDecision, TradeContext


class LongshotFade(Strategy):
    """Fade extreme-priced contracts where calibration gaps are largest.

    - If YES price < low_threshold (e.g. 15c): buy NO (fade the longshot)
    - If YES price > high_threshold (e.g. 85c): buy YES (back the near-certainty)
    """

    def __init__(
        self,
        low_threshold: int = 15,
        high_threshold: int = 85,
        sizer: PositionSizer | None = None,
        max_contracts_per_trade: int = 50,
    ):
        super().__init__(name="longshot_fade")
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.sizer = sizer or KellySizer(fraction=0.15, max_bet_pct=0.02)
        self.max_contracts_per_trade = max_contracts_per_trade
        self._traded_markets: set[str] = set()

    def on_trade(self, ctx: TradeContext) -> StrategyDecision:
        # Only enter each market once to avoid concentration
        if ctx.ticker in self._traded_markets:
            return StrategyDecision(signal=Signal.SKIP)

        bankroll = self.bankroll if self.bankroll > 0 else 10_000.0

        if ctx.yes_price <= self.low_threshold:
            # Longshot: market overprices YES, so buy NO
            # Edge: longshots empirically win ~40-60% of implied odds
            implied_no = ctx.no_price / 100.0
            actual_no_prob = 1.0 - ctx.yes_price / 100.0 * 0.5  # longshots win ~half as often
            edge = actual_no_prob - implied_no
            if edge <= 0:
                return StrategyDecision(signal=Signal.SKIP)
            contracts = self.sizer.size(ctx.no_price, edge, bankroll)
            contracts = min(contracts, self.max_contracts_per_trade, ctx.count)
            if contracts <= 0:
                return StrategyDecision(signal=Signal.SKIP)
            self._traded_markets.add(ctx.ticker)
            return StrategyDecision(
                signal=Signal.BUY_NO,
                contracts=contracts,
                price=ctx.no_price,
            )

        if ctx.yes_price >= self.high_threshold:
            # Near-certainty: YES contracts slightly underpriced
            implied_yes = ctx.yes_price / 100.0
            actual_yes_prob = implied_yes + (1.0 - implied_yes) * 0.5  # near-certainties win ~midpoint above implied
            edge = actual_yes_prob - implied_yes
            if edge <= 0:
                return StrategyDecision(signal=Signal.SKIP)
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

        return StrategyDecision(signal=Signal.SKIP)

    def on_resolution(self, ticker: str, result: str) -> None:
        self._traded_markets.discard(ticker)
