"""Portfolio and position tracking for prediction market backtesting."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Position:
    """A position in a single market."""

    ticker: str
    side: str  # "yes" or "no"
    contracts: int
    avg_price_cents: float  # average entry price in cents
    entry_cost: float  # total dollars spent entering (including fees)

    @property
    def cost_basis(self) -> float:
        """Total cost in dollars (contracts * avg_price / 100)."""
        return self.contracts * self.avg_price_cents / 100.0


@dataclass
class ClosedPosition:
    """A resolved position with P&L."""

    ticker: str
    side: str
    contracts: int
    avg_price_cents: float
    entry_cost: float
    payout: float  # gross payout before settlement fees
    settlement_fees: float
    net_pnl: float
    result: str  # market result ("yes" or "no")
    won: bool


@dataclass
class Portfolio:
    """Tracks open positions, resolved P&L, and bankroll."""

    initial_bankroll: float
    bankroll: float = 0.0
    max_exposure_per_market: float = 0.0  # max dollars risked per market (0 = no limit)
    open_positions: dict[str, Position] = field(default_factory=dict)
    closed_positions: list[ClosedPosition] = field(default_factory=list)
    total_entry_fees: float = 0.0
    total_settlement_fees: float = 0.0

    def __post_init__(self):
        if self.bankroll == 0.0:
            self.bankroll = self.initial_bankroll

    def add_position(self, ticker: str, side: str, contracts: int, price_cents: int, entry_fee: float) -> None:
        """Add or increase a position in a market."""
        cost = contracts * price_cents / 100.0 + entry_fee

        # Enforce per-market exposure limit
        if self.max_exposure_per_market > 0:
            existing_cost = self.open_positions[ticker].entry_cost if ticker in self.open_positions else 0.0
            remaining_budget = self.max_exposure_per_market - existing_cost
            if remaining_budget <= 0:
                return
            if cost > remaining_budget:
                contracts = max(1, int((remaining_budget - entry_fee) * 100.0 / price_cents))
                cost = contracts * price_cents / 100.0 + entry_fee

        if cost > self.bankroll:
            # Size down to what we can afford
            affordable = int((self.bankroll - entry_fee) * 100.0 / price_cents)
            if affordable <= 0:
                return
            contracts = affordable
            cost = contracts * price_cents / 100.0 + entry_fee

        self.bankroll -= cost
        self.total_entry_fees += entry_fee

        if ticker in self.open_positions:
            existing = self.open_positions[ticker]
            if existing.side != side:
                # Can't hold both sides — skip
                self.bankroll += cost
                self.total_entry_fees -= entry_fee
                return
            total_contracts = existing.contracts + contracts
            existing.avg_price_cents = (
                (existing.avg_price_cents * existing.contracts + price_cents * contracts)
                / total_contracts
            )
            existing.contracts = total_contracts
            existing.entry_cost += cost
        else:
            self.open_positions[ticker] = Position(
                ticker=ticker,
                side=side,
                contracts=contracts,
                avg_price_cents=float(price_cents),
                entry_cost=cost,
            )

    def resolve_market(self, ticker: str, result: str, settlement_fee_fn=None) -> ClosedPosition | None:
        """Resolve a market and calculate P&L."""
        if ticker not in self.open_positions:
            return None

        pos = self.open_positions.pop(ticker)
        won = pos.side == result

        # Payout: $1 per contract if won, $0 if lost
        gross_payout = pos.contracts * 1.0 if won else 0.0

        # Settlement fees
        if settlement_fee_fn and won:
            profit_cents = 100.0 - pos.avg_price_cents
            settlement_fee = settlement_fee_fn(profit_cents, pos.contracts)
        else:
            settlement_fee = 0.0

        net_payout = gross_payout - settlement_fee
        net_pnl = net_payout - pos.entry_cost

        self.bankroll += net_payout
        self.total_settlement_fees += settlement_fee

        closed = ClosedPosition(
            ticker=ticker,
            side=pos.side,
            contracts=pos.contracts,
            avg_price_cents=pos.avg_price_cents,
            entry_cost=pos.entry_cost,
            payout=gross_payout,
            settlement_fees=settlement_fee,
            net_pnl=net_pnl,
            result=result,
            won=won,
        )
        self.closed_positions.append(closed)
        return closed

    @property
    def total_pnl(self) -> float:
        return sum(p.net_pnl for p in self.closed_positions)

    @property
    def equity(self) -> float:
        """Current equity = bankroll + mark-to-market of open positions (at cost)."""
        open_value = sum(p.cost_basis for p in self.open_positions.values())
        return self.bankroll + open_value

    @property
    def num_trades(self) -> int:
        return len(self.closed_positions)
