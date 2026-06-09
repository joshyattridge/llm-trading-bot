"""Closed-trade log and win/loss streak tracking for LLM context."""

from __future__ import annotations

from pydantic import BaseModel

from llm_trading_bot.trading.models import PositionSide, PositionState


class ClosedTrade(BaseModel):
    side: str
    entry_price: float
    exit_price: float
    size: float
    realized_pnl: float
    bars_held: int
    exit_reason: str  # stop_loss | take_profit | llm_close


def realized_pnl(
    side: PositionSide,
    *,
    entry_price: float,
    exit_price: float,
    size: float,
) -> float:
    if side == PositionSide.LONG:
        return (exit_price - entry_price) * size
    if side == PositionSide.SHORT:
        return (entry_price - exit_price) * size
    return 0.0


def exit_reason_from_stop_hit(reason: str) -> str:
    lower = reason.lower()
    if "take profit" in lower:
        return "take_profit"
    if "stop loss" in lower:
        return "stop_loss"
    return "stop_loss"


class TradeHistoryTracker:
    def __init__(self, *, max_trades: int = 5) -> None:
        self._max_trades = max(1, max_trades)
        self._trades: list[ClosedTrade] = []

    def record_close(
        self,
        position: PositionState,
        *,
        exit_price: float,
        bars_held: int,
        exit_reason: str,
    ) -> None:
        if position.side == PositionSide.FLAT or position.entry_price is None:
            return
        if exit_price <= 0 or position.size <= 0:
            return

        pnl = realized_pnl(
            position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size=position.size,
        )
        self._trades.append(
            ClosedTrade(
                side=position.side.value,
                entry_price=position.entry_price,
                exit_price=exit_price,
                size=position.size,
                realized_pnl=pnl,
                bars_held=bars_held,
                exit_reason=exit_reason,
            )
        )
        if len(self._trades) > self._max_trades:
            self._trades = self._trades[-self._max_trades :]

    def to_prompt(self) -> dict:
        recent = [t.model_dump() for t in self._trades]
        wins = sum(1 for t in self._trades if t.realized_pnl > 0)
        losses = sum(1 for t in self._trades if t.realized_pnl < 0)
        return {
            "recent_trades": recent,
            "trade_count": len(self._trades),
            "wins": wins,
            "losses": losses,
            "consecutive_wins": self._consecutive_wins(),
            "consecutive_losses": self._consecutive_losses(),
            "net_pnl": sum(t.realized_pnl for t in self._trades),
        }

    def _consecutive_wins(self) -> int:
        count = 0
        for trade in reversed(self._trades):
            if trade.realized_pnl <= 0:
                break
            count += 1
        return count

    def _consecutive_losses(self) -> int:
        count = 0
        for trade in reversed(self._trades):
            if trade.realized_pnl >= 0:
                break
            count += 1
        return count
