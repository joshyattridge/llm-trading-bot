"""Tests for closed-trade history tracking."""

from __future__ import annotations

from llm_trading_bot.data.serialize import state_to_prompt
from llm_trading_bot.trading.models import AccountState, PositionSide, PositionState
from llm_trading_bot.trading.trade_history import TradeHistoryTracker, exit_reason_from_stop_hit


class TestTradeHistoryTracker:
    def test_records_closed_trade_and_streaks(self):
        tracker = TradeHistoryTracker(max_trades=3)
        long_pos = PositionState(
            side=PositionSide.LONG,
            size=1.0,
            entry_price=100.0,
        )
        short_pos = PositionState(
            side=PositionSide.SHORT,
            size=1.0,
            entry_price=200.0,
        )

        tracker.record_close(
            long_pos,
            exit_price=105.0,
            bars_held=3,
            exit_reason="take_profit",
        )
        tracker.record_close(
            short_pos,
            exit_price=205.0,
            bars_held=5,
            exit_reason="stop_loss",
        )

        payload = tracker.to_prompt()
        assert payload["trade_count"] == 2
        assert payload["wins"] == 1
        assert payload["losses"] == 1
        assert payload["consecutive_wins"] == 0
        assert payload["consecutive_losses"] == 1
        assert payload["recent_trades"][-1]["exit_reason"] == "stop_loss"

    def test_exit_reason_from_stop_hit(self):
        assert exit_reason_from_stop_hit("stop loss hit at 98.00") == "stop_loss"
        assert exit_reason_from_stop_hit("take profit hit at 110.00") == "take_profit"

    def test_state_to_prompt_includes_trade_history(self):
        position = PositionState()
        account = AccountState(balance=10_000, equity=10_000, available_cash=10_000)
        history = {
            "recent_trades": [],
            "trade_count": 0,
            "wins": 0,
            "losses": 0,
            "consecutive_wins": 0,
            "consecutive_losses": 0,
            "net_pnl": 0.0,
        }
        payload = state_to_prompt(position, account, trade_history=history)
        assert payload["trade_history"] == history
