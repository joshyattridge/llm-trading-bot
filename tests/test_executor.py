"""Tests for decision execution and stop validation."""

import pytest

from llm_trading_bot.trading.executor import (
    _validate_adjust_stop_levels,
    _validate_entry_levels,
    _validate_stop_levels,
    calculate_position_size,
    execute_decision,
)
from llm_trading_bot.trading.models import (
    AccountState,
    Action,
    ExecutionOutcome,
    LLMDecision,
    PositionSide,
    PositionState,
)
from tests.helpers import make_decision
from tests.mock_broker import MockBroker


def _account(equity: float = 10_000.0, cash: float = 10_000.0) -> AccountState:
    return AccountState(
        balance=cash,
        equity=equity,
        available_cash=cash,
        currency="USDT",
    )


class TestValidateStopLevels:
    def test_long_valid_levels(self):
        assert _validate_stop_levels(PositionSide.LONG, 95.0, 110.0)

    def test_long_rejects_sl_above_tp(self):
        assert not _validate_stop_levels(PositionSide.LONG, 110.0, 95.0)

    def test_short_valid_levels(self):
        assert _validate_stop_levels(PositionSide.SHORT, 110.0, 90.0)


class TestValidateAdjustStopLevels:
    def test_long_allows_stop_below_entry(self):
        assert _validate_adjust_stop_levels(PositionSide.LONG, 100.0, 95.0, 110.0)

    def test_long_allows_breakeven(self):
        assert _validate_adjust_stop_levels(PositionSide.LONG, 100.0, 100.0, 110.0)

    def test_long_rejects_stop_above_entry(self):
        assert not _validate_adjust_stop_levels(PositionSide.LONG, 100.0, 101.0, 110.0)

    def test_short_allows_stop_above_entry(self):
        assert _validate_adjust_stop_levels(PositionSide.SHORT, 100.0, 105.0, 90.0)

    def test_short_allows_breakeven(self):
        assert _validate_adjust_stop_levels(PositionSide.SHORT, 100.0, 100.0, 90.0)

    def test_short_rejects_stop_below_entry(self):
        assert not _validate_adjust_stop_levels(PositionSide.SHORT, 100.0, 99.0, 90.0)


class TestValidateEntryLevels:
    def test_long_entry_requires_sl_below_price_below_tp(self):
        assert _validate_entry_levels(PositionSide.LONG, 100.0, 95.0, 110.0)
        assert not _validate_entry_levels(PositionSide.LONG, 100.0, 101.0, 110.0)


class TestCalculatePositionSize:
    def test_sizes_from_risk_percent(self):
        account = _account(equity=10_000.0)
        size = calculate_position_size(account, 0.02, 100.0, 95.0, leverage=1.0)
        assert size == pytest.approx(40.0)

    def test_zero_when_stop_equals_entry(self):
        account = _account()
        assert calculate_position_size(account, 0.02, 100.0, 100.0) == 0.0


class TestExecuteDecision:
    def test_adjust_stops_rejects_trailing_above_entry_for_long(self):
        broker = MockBroker()
        broker.set_position(
            side=PositionSide.LONG,
            size=1.0,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
        )
        decision = make_decision(
            Action.ADJUST_STOPS,
            stop_loss=101.0,
            take_profit=110.0,
            reasoning="invalid trailing stop",
        )
        outcome = execute_decision(broker, decision, 100.0)
        assert outcome == ExecutionOutcome.SKIPPED_INVALID_LEVELS
        assert broker.get_position().stop_loss == 95.0

    def test_adjust_stops_applies_valid_breakeven(self):
        broker = MockBroker()
        broker.set_position(
            side=PositionSide.LONG,
            size=1.0,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
        )
        decision = make_decision(
            Action.ADJUST_STOPS,
            stop_loss=100.0,
            take_profit=110.0,
            reasoning="move to breakeven",
        )
        outcome = execute_decision(broker, decision, 100.0)
        assert outcome == ExecutionOutcome.STOPS_ADJUSTED
        assert broker.get_position().stop_loss == 100.0

    def test_close_waits_for_broker_fill(self):
        broker = MockBroker()
        broker.set_position(
            side=PositionSide.LONG,
            size=0.5,
            entry_price=100.0,
        )
        decision = make_decision(Action.CLOSE, reasoning="manual close")
        outcome = execute_decision(broker, decision, 100.0)
        assert outcome == ExecutionOutcome.CLOSED
        assert broker.get_position().side == PositionSide.FLAT
