"""Edge-case tests for trading logic (no LLM)."""

from __future__ import annotations

import backtrader as bt
import pandas as pd
import pytest

from llm_trading_bot.brokers.backtrader_broker import BacktraderBrokerAdapter
from llm_trading_bot.trading.engine import TradingEngine
from llm_trading_bot.trading.executor import (
    _CASH_BUFFER,
    calculate_position_size,
    execute_decision,
)
from llm_trading_bot.trading.models import (
    AccountState,
    Action,
    Candle,
    ExecutionOutcome,
    PositionSide,
)
from tests.helpers import make_decision
from tests.mock_broker import MockBroker
from tests.trading_harness import (
    BTC_USDT_1H_FIXTURE,
    ScriptAdvisor,
    fixture_candles,
    load_btc_fixture,
    market_from_candles,
    run_backtrader_script,
    run_engine_candles,
)


class PendingEntryBroker(MockBroker):
    """Simulates an unfilled entry order sitting in the book."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._pending_entry = False
        self._queued: tuple[float, float, float, float] | None = None

    def enter_long(
        self,
        size: float,
        price: float,
        *,
        stop_loss: float,
        take_profit: float,
    ) -> bool:
        self._pending_entry = True
        self._queued = (size, price, stop_loss, take_profit)
        return True

    def enter_short(
        self,
        size: float,
        price: float,
        *,
        stop_loss: float,
        take_profit: float,
    ) -> bool:
        self._pending_entry = True
        self._queued = (size, price, stop_loss, take_profit)
        return True

    def settle_pending_entry(self) -> None:
        if not self._queued:
            return
        size, price, stop_loss, take_profit = self._queued
        self._pending_entry = False
        self._queued = None
        super().enter_long(size, price, stop_loss=stop_loss, take_profit=take_profit)


class TestHighLeverageAndMargin:
    def test_higher_leverage_increases_max_position_size(self):
        account = AccountState(
            balance=10_000.0,
            equity=10_000.0,
            available_cash=10_000.0,
        )
        unlevered = calculate_position_size(
            account, 1.0, 100.0, 90.0, commission_rate=0.0, leverage=1.0
        )
        levered = calculate_position_size(
            account, 1.0, 100.0, 90.0, commission_rate=0.0, leverage=5.0
        )
        assert levered > unlevered
        assert levered == pytest.approx(unlevered * 5, rel=0.01)

    def test_oversized_backtrader_entry_is_margin_rejected(self):
        ohlcv = [
            {"open": 70_000, "high": 70_100, "low": 69_900, "close": 70_000, "volume": 1},
            {"open": 70_000, "high": 70_100, "low": 69_900, "close": 70_000, "volume": 1},
        ]

        class _MarginProbe(bt.Strategy):
            def __init__(self):
                self.adapter = BacktraderBrokerAdapter(self)
                self.margin_rejected = False

            def next(self):
                if len(self) == 1:
                    self.adapter.enter_long(
                        10.0,
                        float(self.data.close[0]),
                        stop_loss=69_000.0,
                        take_profit=80_000.0,
                    )

            def notify_order(self, order: bt.Order) -> None:
                self.adapter.on_order(order)
                if order.status == order.Margin:
                    self.margin_rejected = True

        df = pd.DataFrame(ohlcv)
        df.index = pd.date_range("2024-01-01", periods=len(df), freq="h")
        cerebro = bt.Cerebro()
        cerebro.addstrategy(_MarginProbe)
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
        cerebro.broker.setcash(1_000.0)
        cerebro.broker.set_coc(True)
        cerebro.broker.setcommission(commission=0.0, leverage=1.0, automargin=True)
        cerebro.run()
        strat = cerebro.runstrats[0][0]  # type: ignore[attr-defined]

        assert strat.margin_rejected
        assert strat.adapter.get_position().side == PositionSide.FLAT

    def test_engine_caps_size_with_tiny_cash(self):
        """Risk-based size is capped by buying power; tiny cash cannot open BTC size."""
        ohlcv = [
            {"open": 70_000, "high": 70_100, "low": 69_900, "close": 70_000, "volume": 1},
            {"open": 70_000, "high": 70_100, "low": 69_900, "close": 70_000, "volume": 1},
            {"open": 70_000, "high": 70_100, "low": 69_900, "close": 70_000, "volume": 1},
        ]
        advisor = ScriptAdvisor(
            decisions=[
                make_decision(
                    Action.ENTER_LONG,
                    risk_pct=1.0,
                    stop_loss=69_999.0,
                    take_profit=80_000.0,
                    reasoning="max size attempt",
                ),
            ]
        )
        strat = run_backtrader_script(
            ohlcv,
            advisor,
            initial_cash=500.0,
            commission_rate=0.0,
            leverage=1.0,
        )

        pos = strat._adapter.get_position()
        if pos.side != PositionSide.FLAT:
            assert pos.size < 0.01
        assert strat.broker.getvalue() == pytest.approx(500.0, rel=0.05)


class TestCommissionRounding:
    def test_commission_reduces_size_vs_zero_commission(self):
        account = AccountState(
            balance=10_000.0,
            equity=10_000.0,
            available_cash=10_000.0,
        )
        no_comm = calculate_position_size(
            account, 0.5, 100.0, 90.0, commission_rate=0.0, leverage=1.0
        )
        with_comm = calculate_position_size(
            account, 0.5, 100.0, 90.0, commission_rate=0.001, leverage=1.0
        )
        assert with_comm < no_comm

    def test_sized_position_respects_cash_buffer_with_commission(self):
        account = AccountState(
            balance=10_000.0,
            equity=10_000.0,
            available_cash=10_000.0,
        )
        entry = 73_536.01
        stop = 72_000.0
        commission = 0.001
        leverage = 1.0
        size = calculate_position_size(
            account,
            0.02,
            entry,
            stop,
            commission_rate=commission,
            leverage=leverage,
        )
        cost = size * entry * (1.0 + commission)
        max_cost = account.available_cash * leverage * (1.0 - _CASH_BUFFER)
        assert cost <= max_cost + 1e-6

    def test_backtrader_entry_with_commission_still_fills(self):
        ohlcv = [
            {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
            {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
            {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
            {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
        ]
        advisor = ScriptAdvisor(
            decisions=[
                make_decision(
                    Action.ENTER_LONG,
                    risk_pct=0.02,
                    stop_loss=95.0,
                    take_profit=120.0,
                    reasoning="commission entry",
                ),
            ]
        )
        strat = run_backtrader_script(
            ohlcv,
            advisor,
            commission_rate=0.001,
            leverage=1.0,
        )

        assert strat._adapter.get_position().side == PositionSide.LONG
        assert any("Completed" in e and "buy" in e for e in strat.events)


class TestShortPositions:
    def test_short_stop_loss_on_mock_broker(self):
        broker = MockBroker(cash=10_000.0)
        advisor = ScriptAdvisor(
            decisions=[
                make_decision(
                    Action.ENTER_SHORT,
                    risk_pct=0.02,
                    stop_loss=105.0,
                    take_profit=85.0,
                    reasoning="short",
                ),
            ]
        )
        engine = TradingEngine(advisor, broker, commission_rate=0.0, leverage=1.0)

        candles = [
            Candle(open=100, high=101, low=99, close=100, volume=1),
            Candle(open=100, high=106, low=99, close=104, volume=1),
        ]
        run_engine_candles(engine, candles)

        assert broker.get_position().side == PositionSide.FLAT
        assert broker.close_calls == [105.0]
        assert broker._cash < 10_000.0

    def test_short_take_profit_on_mock_broker(self):
        broker = MockBroker(cash=10_000.0)
        advisor = ScriptAdvisor(
            decisions=[
                make_decision(
                    Action.ENTER_SHORT,
                    risk_pct=0.02,
                    stop_loss=105.0,
                    take_profit=90.0,
                    reasoning="short",
                ),
            ]
        )
        engine = TradingEngine(advisor, broker, commission_rate=0.0, leverage=1.0)

        candles = [
            Candle(open=100, high=101, low=99, close=100, volume=1),
            Candle(open=100, high=101, low=88, close=91, volume=1),
        ]
        run_engine_candles(engine, candles)

        assert broker.get_position().side == PositionSide.FLAT
        assert broker.close_calls == [90.0]
        assert broker._cash > 10_000.0

    def test_short_end_to_end_on_backtrader(self):
        ohlcv = [
            {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
            {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
            {"open": 100, "high": 101, "low": 88, "close": 91, "volume": 1},
            {"open": 91, "high": 92, "low": 89, "close": 90, "volume": 1},
        ]
        advisor = ScriptAdvisor(
            decisions=[
                make_decision(
                    Action.ENTER_SHORT,
                    risk_pct=0.02,
                    stop_loss=105.0,
                    take_profit=90.0,
                    reasoning="short",
                ),
            ]
        )
        strat = run_backtrader_script(ohlcv, advisor)

        assert strat._adapter.get_position().side == PositionSide.FLAT
        assert any("sell" in e for e in strat.events)
        assert any("buy" in e for e in strat.events)
        assert strat.broker.getvalue() > 10_000.0


class TestOverlappingOrders:
    def test_second_entry_skipped_while_first_pending(self):
        broker = PendingEntryBroker(cash=10_000.0)
        candle = Candle(open=100, high=101, low=99, close=100, volume=1)

        first = make_decision(
            Action.ENTER_LONG,
            risk_pct=0.02,
            stop_loss=95.0,
            take_profit=110.0,
            reasoning="first",
        )
        outcome1 = execute_decision(broker, first, 100.0)
        assert outcome1 == ExecutionOutcome.ORDER_SUBMITTED
        assert broker.has_pending_entry()

        second = make_decision(
            Action.ENTER_LONG,
            risk_pct=0.02,
            stop_loss=94.0,
            take_profit=112.0,
            reasoning="duplicate",
        )
        outcome2 = execute_decision(broker, second, 100.0)
        assert outcome2 == ExecutionOutcome.SKIPPED_PENDING_ENTRY
        assert broker._queued[2] == 95.0

    def test_engine_skips_second_entry_while_first_pending(self):
        broker = PendingEntryBroker(cash=10_000.0)
        advisor = ScriptAdvisor(
            decisions=[
                make_decision(
                    Action.ENTER_LONG,
                    risk_pct=0.02,
                    stop_loss=95.0,
                    take_profit=120.0,
                    reasoning="first",
                ),
                make_decision(
                    Action.ENTER_LONG,
                    risk_pct=0.02,
                    stop_loss=94.0,
                    take_profit=120.0,
                    reasoning="second",
                ),
            ]
        )
        engine = TradingEngine(advisor, broker, commission_rate=0.0, leverage=1.0)

        candles = [
            Candle(open=100, high=101, low=99, close=100, volume=1),
            Candle(open=100, high=101, low=99, close=100, volume=1),
        ]
        run_engine_candles(engine, candles)

        assert broker.has_pending_entry() or broker.get_position().side == PositionSide.LONG
        assert broker._queued is not None
        assert broker._queued[2] == 95.0


class TestRealOhlcvFixture:
    def test_fixture_file_exists_and_looks_like_btc(self):
        assert BTC_USDT_1H_FIXTURE.is_file()
        df = load_btc_fixture()
        assert len(df) >= 20
        assert df["close"].iloc[0] > 10_000
        assert (df["high"] >= df["low"]).all()

    def test_engine_runs_on_realistic_btc_candles(self):
        df = load_btc_fixture()
        candles = fixture_candles(df, count=8)
        broker = MockBroker(cash=10_000.0)
        first_close = candles[0].close
        advisor = ScriptAdvisor(
            decisions=[
                make_decision(
                    Action.ENTER_LONG,
                    risk_pct=0.01,
                    stop_loss=first_close * 0.98,
                    take_profit=first_close * 1.02,
                    reasoning="btc long",
                ),
            ]
        )
        engine = TradingEngine(
            advisor,
            broker,
            commission_rate=0.001,
            leverage=1.0,
        )
        run_engine_candles(engine, candles)

        assert broker.get_position().size > 0 or broker.close_calls
        assert all(c.close > 0 for c in candles)

    def test_backtrader_script_on_recorded_exchange_data(self):
        df = load_btc_fixture().iloc[:12]
        first_close = float(df["close"].iloc[2])
        advisor = ScriptAdvisor(
            decisions=[
                make_decision(
                    Action.ENTER_LONG,
                    risk_pct=0.01,
                    stop_loss=first_close * 0.97,
                    take_profit=first_close * 1.05,
                    reasoning="btc backtrader",
                ),
                make_decision(Action.CLOSE, reasoning="exit"),
            ]
        )
        strat = run_backtrader_script(
            df,
            advisor,
            history_len=3,
            commission_rate=0.001,
            leverage=1.0,
        )

        assert any("buy" in e for e in strat.events)
        assert strat._adapter.get_position().side == PositionSide.FLAT
