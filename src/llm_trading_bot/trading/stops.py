"""Intrabar stop-loss / take-profit detection for backtests."""

from __future__ import annotations

from dataclasses import dataclass

from llm_trading_bot.trading.models import Candle, PositionSide, PositionState


@dataclass(frozen=True)
class StopHit:
    reason: str
    fill_price: float


def backtrader_candle_indices(history_len: int) -> range:
    """Backtrader data indices for the last `history_len` bars, including [0]."""
    if history_len < 1:
        raise ValueError("history_len must be at least 1")
    return range(-(history_len - 1), 1)


def check_stop_hit(position: PositionState, candle: Candle) -> StopHit | None:
    """Return a stop/TP hit on `candle` with the simulated fill price."""
    if position.side == PositionSide.FLAT:
        return None

    stop_loss = position.stop_loss
    take_profit = position.take_profit
    if stop_loss is None and take_profit is None:
        return None

    if position.side == PositionSide.LONG:
        if stop_loss is not None and candle.low <= stop_loss:
            return StopHit(
                reason=f"stop loss hit at {stop_loss:.2f}",
                fill_price=stop_loss,
            )
        if take_profit is not None and candle.high >= take_profit:
            return StopHit(
                reason=f"take profit hit at {take_profit:.2f}",
                fill_price=take_profit,
            )
    elif position.side == PositionSide.SHORT:
        if stop_loss is not None and candle.high >= stop_loss:
            return StopHit(
                reason=f"stop loss hit at {stop_loss:.2f}",
                fill_price=stop_loss,
            )
        if take_profit is not None and candle.low <= take_profit:
            return StopHit(
                reason=f"take profit hit at {take_profit:.2f}",
                fill_price=take_profit,
            )

    return None
