"""Integration tests with backtrader broker adapter."""

from __future__ import annotations

import backtrader as bt
import pandas as pd
import pytest

from llm_trading_bot.brokers.backtrader_broker import BacktraderBrokerAdapter
from llm_trading_bot.trading.models import PositionSide


class _ImmediateStrategy(bt.Strategy):
    params = (("actions", None),)

    def __init__(self):
        self.adapter = BacktraderBrokerAdapter(self)
        self._step = 0

    def next(self):
        actions = self.p.actions or []
        if self._step >= len(actions):
            return
        action = actions[self._step]
        self._step += 1
        action(self.adapter, self)


def _run_strategy(
    ohlcv: list[dict[str, float]],
    actions: list,
    *,
    initial_cash: float = 10_000.0,
) -> BacktraderBrokerAdapter:
    df = pd.DataFrame(ohlcv)
    df.index = pd.date_range("2024-01-01", periods=len(df), freq="h")

    cerebro = bt.Cerebro()
    cerebro.addstrategy(_ImmediateStrategy, actions=actions)
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
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.set_coc(True)
    cerebro.broker.setcommission(commission=0.0, leverage=1.0, automargin=True)
    cerebro.run()
    return cerebro.runstrats[0][0].adapter  # type: ignore[attr-defined]


class TestBacktraderBrokerAdapter:
    def test_get_position_uses_explicit_mark_price(self):
        ohlcv = [
            {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
            {"open": 100, "high": 105, "low": 99, "close": 104, "volume": 1},
        ]

        def enter(adapter: BacktraderBrokerAdapter, strategy: bt.Strategy) -> None:
            adapter.enter_long(
                1.0, float(strategy.data.close[0]), stop_loss=90.0, take_profit=120.0
            )

        adapter = _run_strategy(ohlcv, [enter, lambda a, s: None])

        assert adapter.get_position(mark_price=104.0).unrealized_pnl == pytest.approx(4.0)

    def test_close_position_at_price_closes_position(self):
        ohlcv = [
            {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
            {"open": 100, "high": 101, "low": 97, "close": 98, "volume": 1},
            {"open": 98, "high": 99, "low": 97, "close": 98, "volume": 1},
        ]

        def enter(adapter: BacktraderBrokerAdapter, strategy: bt.Strategy) -> None:
            adapter.enter_long(1.0, 100.0, stop_loss=95.0, take_profit=120.0)

        def close_at_stop(adapter: BacktraderBrokerAdapter, strategy: bt.Strategy) -> None:
            adapter.close_position_at_price(98.0)

        adapter = _run_strategy(ohlcv, [enter, close_at_stop, lambda a, s: None])

        assert adapter.get_position().side == PositionSide.FLAT

    def test_candle_index_zero_is_current_bar(self):
        from llm_trading_bot.trading.stops import backtrader_candle_indices

        ohlcv = [{"open": 50, "high": 55, "low": 45, "close": 52, "volume": 1}] * 3
        captured: list[float] = []

        def capture(_adapter: BacktraderBrokerAdapter, strategy: bt.Strategy) -> None:
            idx = list(backtrader_candle_indices(2))[-1]
            captured.append(float(strategy.data.close[idx]))

        _run_strategy(ohlcv, [lambda a, s: None, capture])

        assert captured[-1] == pytest.approx(52.0)
