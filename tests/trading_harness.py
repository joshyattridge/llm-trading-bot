"""Shared helpers for trading integration and edge-case tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import backtrader as bt
import pandas as pd

from llm_trading_bot.brokers.backtrader_broker import BacktraderBrokerAdapter
from llm_trading_bot.data.market import MultiTimeframeMarket, TimeframeSeries, bar_time_fields
from llm_trading_bot.trading.engine import TradingEngine
from llm_trading_bot.trading.models import Action, Candle
from llm_trading_bot.trading.stops import backtrader_candle_indices
from tests.helpers import make_decision

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
BTC_USDT_1H_FIXTURE = FIXTURES_DIR / "btc_usdt_1h_sample.csv"


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


def market_from_candles(candles: list[Candle]) -> MultiTimeframeMarket:
    return MultiTimeframeMarket(
        lower=TimeframeSeries(timeframe="1h", candles=candles),
    )


def load_btc_fixture() -> pd.DataFrame:
    df = pd.read_csv(BTC_USDT_1H_FIXTURE, parse_dates=["datetime"])
    df = df.set_index("datetime")
    return df


def fixture_candles(df: pd.DataFrame, count: int | None = None) -> list[Candle]:
    subset = df if count is None else df.iloc[:count]
    candles: list[Candle] = []
    for row in subset.itertuples():
        bar_time, day_of_week = bar_time_fields(row.Index)
        candles.append(
            Candle(
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                bar_time=bar_time,
                day_of_week=day_of_week,
            )
        )
    return candles


class ScriptedEngineStrategy(bt.Strategy):
    """Backtrader strategy wired to TradingEngine + ScriptAdvisor."""

    params = (
        ("advisor", None),
        ("history_len", 3),
        ("commission_rate", 0.0),
        ("leverage", 1.0),
    )

    def __init__(self):
        self._adapter = BacktraderBrokerAdapter(self)
        self._engine = TradingEngine(
            self.p.advisor,
            self._adapter,
            commission_rate=self.p.commission_rate,
            leverage=self.p.leverage,
        )
        self._bar = 0
        self.events: list[str] = []
        self.outcomes: list[str] = []

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
                f"order {order.getstatusname()} {side} "
                f"{order.executed.size}@{order.executed.price:.2f}"
            )

    def next(self):
        self._bar += 1
        n = self.p.history_len
        candles: list[Candle] = []
        for i in backtrader_candle_indices(n):
            bar_time, day_of_week = bar_time_fields(bt.num2date(self.data.datetime[i]))
            candles.append(
                Candle(
                    open=float(self.data.open[i]),
                    high=float(self.data.high[i]),
                    low=float(self.data.low[i]),
                    close=float(self.data.close[i]),
                    volume=float(self.data.volume[i]),
                    bar_time=bar_time,
                    day_of_week=day_of_week,
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
        if self._engine._pending_panel:
            outcome = self._engine._pending_panel.get("outcome")
            if outcome is not None:
                self.outcomes.append(str(outcome))


def run_backtrader_script(
    ohlcv: list[dict[str, float]] | pd.DataFrame,
    advisor: ScriptAdvisor,
    *,
    history_len: int = 3,
    initial_cash: float = 10_000.0,
    commission_rate: float = 0.0,
    leverage: float = 1.0,
    cheat_on_close: bool = True,
) -> ScriptedEngineStrategy:
    if isinstance(ohlcv, pd.DataFrame):
        df = ohlcv.copy()
    else:
        df = pd.DataFrame(ohlcv)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.date_range("2024-01-01", periods=len(df), freq="h")

    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        ScriptedEngineStrategy,
        advisor=advisor,
        history_len=history_len,
        commission_rate=commission_rate,
        leverage=leverage,
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
    cerebro.broker.setcash(initial_cash)
    if cheat_on_close:
        cerebro.broker.set_coc(True)
    cerebro.broker.setcommission(
        commission=commission_rate,
        leverage=leverage,
        automargin=True,
    )
    cerebro.run()
    return cerebro.runstrats[0][0]  # type: ignore[attr-defined]


def run_engine_candles(
    engine: TradingEngine,
    candles: list[Candle],
) -> None:
    history: list[Candle] = []
    for i, candle in enumerate(candles, start=1):
        history.append(candle)
        engine.on_new_candle(
            market_from_candles(history),
            candle,
            bar=i,
            total_bars=len(candles),
        )
        if engine.waiting_for_order_notify():
            engine.on_order_settled()
        else:
            engine.flush_display()
