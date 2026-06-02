"""Tests for the trading engine display and stop handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from llm_trading_bot.data.market import MultiTimeframeMarket, TimeframeSeries
from llm_trading_bot.trading.engine import TradingEngine
from llm_trading_bot.trading.executor import BrokerAdapter
from llm_trading_bot.trading.models import (
    AccountState,
    Action,
    Candle,
    ExecutionOutcome,
    LLMDecision,
    PositionSide,
    PositionState,
)
from tests.helpers import make_decision
from tests.mock_broker import MockBroker


def _market(candle: Candle) -> MultiTimeframeMarket:
    return MultiTimeframeMarket(
        lower=TimeframeSeries(timeframe="1h", candles=[candle]),
    )


@dataclass
class RecordingDisplay:
    panels: list[dict[str, Any]] = field(default_factory=list)

    def print_candle(self, **kwargs: Any) -> None:
        self.panels.append(kwargs)


class DeferredFillBroker(MockBroker):
    """Simulates backtrader: close order submitted but position open until settled."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._fill_on_close = False
        self._pending_close = False

    def close_position(self) -> None:
        self.close_calls.append(None)
        self._pending_close = True

    def close_position_at_price(self, price: float) -> None:
        self.close_calls.append(price)
        self._pending_close = True

    def settle_pending_close(self) -> None:
        if not self._pending_close:
            return
        self._fill_on_close = True
        self._apply_close()
        self._fill_on_close = False
        self._pending_close = False


class TestTradingEngine:
    def test_stop_hit_defers_display_until_order_settled(self):
        broker = DeferredFillBroker(cash=8_000.0)
        broker.set_position(
            side=PositionSide.LONG,
            size=1.0,
            entry_price=100.0,
            stop_loss=98.0,
            take_profit=110.0,
        )
        display = RecordingDisplay()
        advisor = MagicMock()
        engine = TradingEngine(advisor, broker, display, symbol="BTC/USDT", timeframe="1h")

        candle = Candle(open=100.0, high=101.0, low=97.5, close=99.0, volume=1.0)
        engine.on_new_candle(_market(candle), candle, bar=10, total_bars=100)

        assert advisor.decide.call_count == 0
        assert len(display.panels) == 0
        assert engine.waiting_for_order_notify()
        assert broker.close_calls == [98.0]
        assert broker.get_position(mark_price=99.0).side == PositionSide.LONG

        broker.settle_pending_close()
        engine.on_order_settled()

        assert len(display.panels) == 1
        panel = display.panels[0]
        assert panel["decision"].action == Action.CLOSE
        assert panel["outcome"] == ExecutionOutcome.CLOSED
        assert panel["position"].side == PositionSide.FLAT
        assert panel["close_price"] == pytest.approx(99.0)

    def test_stop_hit_shows_flat_position_after_settle(self):
        broker = MockBroker(cash=9_000.0)
        broker.set_position(
            side=PositionSide.LONG,
            size=2.0,
            entry_price=100.0,
            stop_loss=98.0,
            take_profit=120.0,
        )
        display = RecordingDisplay()
        engine = TradingEngine(MagicMock(), broker, display)

        candle = Candle(open=100.0, high=100.5, low=97.0, close=97.5, volume=1.0)
        engine.on_new_candle(_market(candle), candle, bar=2, total_bars=10)
        engine.on_order_settled()

        assert len(display.panels) == 1
        panel = display.panels[0]
        assert panel["close_price"] == pytest.approx(97.5)
        assert panel["position"].side == PositionSide.FLAT
        assert broker.close_calls == [98.0]

    def test_llm_close_defers_display(self):
        broker = DeferredFillBroker()
        broker.set_position(
            side=PositionSide.LONG,
            size=1.0,
            entry_price=50.0,
        )
        display = RecordingDisplay()
        advisor = MagicMock()
        advisor.decide.return_value = make_decision(Action.CLOSE, reasoning="exit now")
        engine = TradingEngine(advisor, broker, display)

        candle = Candle(open=50.0, high=51.0, low=49.0, close=50.5, volume=1.0)
        engine.on_new_candle(_market(candle), candle, bar=1, total_bars=5)

        assert len(display.panels) == 0
        assert engine.waiting_for_order_notify()

        broker.settle_pending_close()
        engine.on_order_settled()

        assert display.panels[0]["position"].side == PositionSide.FLAT

    def test_should_wait_for_order_includes_close_and_entries(self):
        assert TradingEngine._should_wait_for_order(
            make_decision(Action.CLOSE),
            ExecutionOutcome.CLOSED,
        )
        assert TradingEngine._should_wait_for_order(
            make_decision(
                Action.ENTER_LONG,
                stop_loss=90.0,
                take_profit=110.0,
            ),
            ExecutionOutcome.ORDER_SUBMITTED,
        )
        assert not TradingEngine._should_wait_for_order(
            make_decision(Action.HOLD),
            ExecutionOutcome.NOOP,
        )
