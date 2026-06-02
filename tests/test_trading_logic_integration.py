"""
End-to-end trading logic tests without LLM integration.

Uses a ScriptAdvisor that returns predetermined decisions and exercises
the full TradingEngine + executor + stops pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import backtrader as bt
import pandas as pd
import pytest

from llm_trading_bot.brokers.backtrader_broker import BacktraderBrokerAdapter
from llm_trading_bot.data.market import MultiTimeframeMarket, TimeframeSeries
from llm_trading_bot.trading.engine import TradingEngine
from llm_trading_bot.trading.models import (
    Action,
    Candle,
    ExecutionOutcome,
    PositionSide,
)
from llm_trading_bot.trading.stops import backtrader_candle_indices
from tests.helpers import make_decision
from tests.mock_broker import MockBroker


@dataclass
class ScriptAdvisor:
    """Deterministic advisor — no LLM calls."""

    decisions: list = field(default_factory=list)
    _index: int = 0

    def decide(self, market, position, account):
        if self._index >= len(self.decisions):
            return make_decision(Action.HOLD, reasoning="script done")
        decision = self.decisions[self._index]
        self._index += 1
        return decision


def _market(candles: list[Candle]) -> MultiTimeframeMarket:
    return MultiTimeframeMarket(
        lower=TimeframeSeries(timeframe="1h", candles=candles),
    )


def _run_engine_candles(
    engine: TradingEngine,
    broker: MockBroker,
    candles: list[Candle],
) -> None:
    """Feed candles through the engine, settling orders after each bar."""
    history: list[Candle] = []
    for i, candle in enumerate(candles, start=1):
        history.append(candle)
        engine.on_new_candle(
            _market(history),
            candle,
            bar=i,
            total_bars=len(candles),
        )
        if engine.waiting_for_order_notify():
            engine.on_order_settled()
        else:
            engine.flush_display()


class TestMockBrokerEngineFlow:
    """Full engine flows on the in-memory broker (instant fills)."""

    def test_enter_long_then_take_profit(self):
        broker = MockBroker(cash=10_000.0)
        advisor = ScriptAdvisor(
            decisions=[
                make_decision(
                    Action.ENTER_LONG,
                    risk_pct=0.02,
                    stop_loss=95.0,
                    take_profit=110.0,
                    reasoning="enter",
                ),
            ]
        )
        engine = TradingEngine(advisor, broker, commission_rate=0.0, leverage=1.0)

        candles = [
            Candle(open=100, high=101, low=99, close=100, volume=1),
            Candle(open=100, high=105, low=99, close=104, volume=1),
            Candle(open=104, high=112, low=103, close=111, volume=1),
        ]
        _run_engine_candles(engine, broker, candles)

        assert broker.get_position().side == PositionSide.FLAT
        assert broker.close_calls == [110.0]
        assert broker._cash > 10_000.0

    def test_enter_long_then_stop_loss(self):
        broker = MockBroker(cash=10_000.0)
        advisor = ScriptAdvisor(
            decisions=[
                make_decision(
                    Action.ENTER_LONG,
                    risk_pct=0.02,
                    stop_loss=95.0,
                    take_profit=110.0,
                    reasoning="enter",
                ),
            ]
        )
        engine = TradingEngine(advisor, broker, commission_rate=0.0, leverage=1.0)

        candles = [
            Candle(open=100, high=101, low=99, close=100, volume=1),
            Candle(open=100, high=100, low=94, close=96, volume=1),
        ]
        _run_engine_candles(engine, broker, candles)

        assert broker.get_position().side == PositionSide.FLAT
        assert broker.close_calls == [95.0]
        assert broker._cash < 10_000.0

    def test_adjust_stops_then_breakeven_exit(self):
        broker = MockBroker(cash=10_000.0)
        advisor = ScriptAdvisor(
            decisions=[
                make_decision(
                    Action.ENTER_LONG,
                    risk_pct=0.01,
                    stop_loss=90.0,
                    take_profit=120.0,
                    reasoning="enter",
                ),
                make_decision(
                    Action.ADJUST_STOPS,
                    stop_loss=100.0,
                    take_profit=120.0,
                    reasoning="breakeven",
                ),
            ]
        )
        engine = TradingEngine(advisor, broker, commission_rate=0.0, leverage=1.0)

        candles = [
            Candle(open=100, high=101, low=99, close=100, volume=1),
            Candle(open=100, high=102, low=99, close=101, volume=1),
            Candle(open=101, high=102, low=99.5, close=100, volume=1),
        ]
        _run_engine_candles(engine, broker, candles)

        assert broker.get_position().side == PositionSide.FLAT
        assert broker.close_calls == [100.0]

    def test_rejects_invalid_trailing_stop_adjust(self):
        broker = MockBroker(cash=10_000.0)
        broker.set_position(
            side=PositionSide.LONG,
            size=1.0,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
        )
        advisor = ScriptAdvisor(
            decisions=[
                make_decision(
                    Action.ADJUST_STOPS,
                    stop_loss=101.0,
                    take_profit=110.0,
                    reasoning="bad trailing",
                ),
            ]
        )
        engine = TradingEngine(advisor, broker, commission_rate=0.0, leverage=1.0)

        candle = Candle(open=100, high=101, low=99, close=100, volume=1)
        engine.on_new_candle(_market([candle]), candle, bar=1, total_bars=1)
        engine.flush_display()

        assert broker.get_position().stop_loss == 95.0

    def test_manual_close_via_advisor(self):
        broker = MockBroker(cash=10_000.0)
        broker.set_position(
            side=PositionSide.LONG,
            size=2.0,
            entry_price=100.0,
            stop_loss=90.0,
            take_profit=120.0,
        )
        advisor = ScriptAdvisor(
            decisions=[make_decision(Action.CLOSE, reasoning="exit")]
        )
        engine = TradingEngine(advisor, broker, commission_rate=0.0, leverage=1.0)

        candle = Candle(open=105, high=106, low=104, close=105, volume=1)
        engine.on_new_candle(_market([candle]), candle, bar=1, total_bars=1)
        engine.on_order_settled()

        assert broker.get_position().side == PositionSide.FLAT
        # Mock market close fills at entry price when no explicit fill price is set.
        assert broker._cash == pytest.approx(10_000.0 + 2.0 * 100.0)

    def test_upnl_matches_candle_close_while_holding(self):
        broker = MockBroker(cash=10_000.0)
        advisor = ScriptAdvisor(
            decisions=[
                make_decision(
                    Action.ENTER_LONG,
                    risk_pct=0.01,
                    stop_loss=90.0,
                    take_profit=120.0,
                    reasoning="enter",
                ),
                make_decision(Action.HOLD, reasoning="hold"),
            ]
        )
        engine = TradingEngine(advisor, broker, commission_rate=0.0, leverage=1.0)

        c1 = Candle(open=100, high=101, low=99, close=100, volume=1)
        c2 = Candle(open=100, high=106, low=99, close=105, volume=1)

        engine.on_new_candle(_market([c1]), c1, bar=1, total_bars=2)
        engine.on_order_settled()

        pos = broker.get_position(mark_price=100.0)
        assert pos.side == PositionSide.LONG
        assert pos.unrealized_pnl == pytest.approx(0.0, abs=0.01)

        engine.on_new_candle(_market([c1, c2]), c2, bar=2, total_bars=2)
        engine.flush_display()

        pos = broker.get_position(mark_price=105.0)
        assert pos.unrealized_pnl == pytest.approx(
            (105.0 - 100.0) * pos.size,
        )


class _ScriptedEngineStrategy(bt.Strategy):
    """Backtrader strategy wired to TradingEngine + ScriptAdvisor."""

    params = (("advisor", None), ("history_len", 3))

    def __init__(self):
        self._adapter = BacktraderBrokerAdapter(self)
        self._engine = TradingEngine(
            self.p.advisor,
            self._adapter,
            commission_rate=0.0,
            leverage=1.0,
        )
        self._bar = 0
        self.events: list[str] = []

    def notify_order(self, order: bt.Order) -> None:
        self._adapter.on_order(order)
        if order.status in (
            order.Completed,
            order.Margin,
            order.Rejected,
            order.Canceled,
        ):
            self._engine.on_order_settled()
            side = "buy" if order.isbuy() else "sell"
            self.events.append(
                f"order {side} {order.executed.size}@{order.executed.price:.2f}"
            )

    def next(self):
        self._bar += 1
        n = self.p.history_len
        candles: list[Candle] = []
        for i in backtrader_candle_indices(n):
            candles.append(
                Candle(
                    open=float(self.data.open[i]),
                    high=float(self.data.high[i]),
                    low=float(self.data.low[i]),
                    close=float(self.data.close[i]),
                    volume=float(self.data.volume[i]),
                )
            )
        market = MultiTimeframeMarket(
            lower=TimeframeSeries(timeframe="1h", candles=candles),
        )
        self._engine.on_new_candle(
            market,
            candles[-1],
            bar=self._bar,
            total_bars=len(self.data),
        )
        if not self._engine.waiting_for_order_notify():
            self._engine.flush_display()

        pos = self._adapter.get_position(mark_price=candles[-1].close)
        self.events.append(
            f"bar {self._bar} close={candles[-1].close:.0f} "
            f"pos={pos.side.value} cash={self.broker.getcash():.2f}"
        )


def _run_backtrader_script(
    ohlcv: list[dict[str, float]],
    advisor: ScriptAdvisor,
    *,
    history_len: int = 3,
) -> _ScriptedEngineStrategy:
    df = pd.DataFrame(ohlcv)
    df.index = pd.date_range("2024-01-01", periods=len(df), freq="h")
    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        _ScriptedEngineStrategy,
        advisor=advisor,
        history_len=history_len,
    )
    cerebro.adddata(
        bt.feeds.PandasData(
            dataname=df,
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
        )
    )
    cerebro.broker.setcash(10_000.0)
    cerebro.broker.set_coc(True)
    cerebro.broker.setcommission(commission=0.0, leverage=1.0, automargin=True)
    cerebro.run()
    return cerebro.runstrats[0][0]  # type: ignore[attr-defined]


class TestBacktraderEngineIntegration:
    """Engine + backtrader broker on synthetic OHLCV, no LLM."""

    def test_scripted_entry_hold_exit_flat(self):
        ohlcv = [
            {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
            {"open": 100, "high": 102, "low": 99, "close": 101, "volume": 1},
            {"open": 101, "high": 103, "low": 100, "close": 102, "volume": 1},
            {"open": 102, "high": 104, "low": 101, "close": 103, "volume": 1},
            {"open": 103, "high": 105, "low": 102, "close": 104, "volume": 1},
        ]
        advisor = ScriptAdvisor(
            decisions=[
                make_decision(
                    Action.ENTER_LONG,
                    risk_pct=0.02,
                    stop_loss=95.0,
                    take_profit=120.0,
                    reasoning="enter",
                ),
                make_decision(Action.HOLD, reasoning="hold"),
                make_decision(Action.CLOSE, reasoning="manual exit"),
            ]
        )
        strat = _run_backtrader_script(ohlcv, advisor)

        assert strat._adapter.get_position().side == PositionSide.FLAT
        assert any("order buy" in e for e in strat.events)
        assert any("order sell" in e for e in strat.events)

    def test_stop_loss_fills_when_low_hits_stop(self):
        ohlcv = [
            {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
            {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
            {"open": 100, "high": 100, "low": 94, "close": 96, "volume": 1},
            {"open": 96, "high": 97, "low": 95, "close": 96, "volume": 1},
            {"open": 96, "high": 97, "low": 95, "close": 96, "volume": 1},
        ]
        advisor = ScriptAdvisor(
            decisions=[
                make_decision(
                    Action.ENTER_LONG,
                    risk_pct=0.02,
                    stop_loss=95.0,
                    take_profit=120.0,
                    reasoning="enter",
                ),
            ]
        )
        strat = _run_backtrader_script(ohlcv, advisor)

        assert strat._adapter.get_position().side == PositionSide.FLAT
        sell_events = [e for e in strat.events if "order sell" in e]
        assert sell_events
        assert strat.broker.getvalue() < 10_000.0

    def test_take_profit_fills_at_tp_price(self):
        ohlcv = [
            {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
            {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
            {"open": 100, "high": 106, "low": 99, "close": 105, "volume": 1},
            {"open": 105, "high": 112, "low": 104, "close": 111, "volume": 1},
            {"open": 111, "high": 112, "low": 110, "close": 111, "volume": 1},
        ]

        advisor = ScriptAdvisor(
            decisions=[
                make_decision(
                    Action.ENTER_LONG,
                    risk_pct=0.02,
                    stop_loss=95.0,
                    take_profit=110.0,
                    reasoning="enter",
                ),
            ]
        )
        strat = _run_backtrader_script(ohlcv, advisor)

        assert strat._adapter.get_position().side == PositionSide.FLAT
        sell_events = [e for e in strat.events if "order sell" in e]
        assert sell_events
        assert strat.broker.getvalue() > 10_000.0
