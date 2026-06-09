import logging

import backtrader as bt
import pandas as pd

from llm_trading_bot.brokers.backtrader_broker import BacktraderBrokerAdapter
from llm_trading_bot.config import Settings
from llm_trading_bot.data.market import (
    MultiTimeframeMarket,
    TimeframeSeries,
    bar_time_fields,
    slice_candles_as_of,
    timeframe_to_timedelta,
)
from llm_trading_bot.display import TerminalDisplay
from llm_trading_bot.llm.client import LLMTradingAdvisor
from llm_trading_bot.trading.engine import TradingEngine
from llm_trading_bot.trading.models import Candle
from llm_trading_bot.trading.stops import backtrader_candle_indices

logger = logging.getLogger(__name__)


class LLMStrategy(bt.Strategy):
    params = (
        ("candle_history", 50),
        ("settings", None),
        ("total_bars", 0),
        ("display", None),
        ("higher_ohlcv", None),
    )

    def __init__(self):
        settings: Settings = self.p.settings
        display: TerminalDisplay | None = self.p.display
        self._history_len = self.p.candle_history
        self._settings = settings
        self._higher_df: pd.DataFrame | None = self.p.higher_ohlcv
        self._advisor = LLMTradingAdvisor(settings)
        self._broker_adapter = BacktraderBrokerAdapter(self)
        self._engine = TradingEngine(
            self._advisor,
            self._broker_adapter,
            display,
            symbol=settings.symbol,
            timeframe=settings.timeframe,
            commission_rate=settings.commission_rate,
            leverage=settings.leverage,
            trade_history_limit=settings.trade_history_limit,
        )
        self._bar_count = 0
        self._total_bars = self.p.total_bars

    def notify_order(self, order: bt.Order) -> None:
        self._broker_adapter.on_order(order)
        if order.status in (
            order.Completed,
            order.Margin,
            order.Rejected,
            order.Canceled,
        ):
            self._engine.on_order_settled()

    def next(self):
        self._bar_count += 1
        if self._bar_count < self._history_len:
            return

        lower_candles = self._build_lower_history()
        market = self._build_market(lower_candles)
        self._engine.on_new_candle(
            market,
            lower_candles[-1],
            bar=self._bar_count,
            total_bars=self._total_bars or None,
        )
        if not self._engine.waiting_for_order_notify():
            self._engine.flush_display()

    def _build_lower_history(self) -> list[Candle]:
        n = self._history_len
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
        return candles

    def _build_market(self, lower_candles: list[Candle]) -> MultiTimeframeMarket:
        lower = TimeframeSeries(timeframe=self._settings.timeframe, candles=lower_candles)
        higher: TimeframeSeries | None = None

        if self._settings.uses_higher_timeframe() and self._higher_df is not None:
            bar_open = bt.num2date(self.data.datetime[0])
            as_of = pd.Timestamp(bar_open, tz="UTC") + timeframe_to_timedelta(
                self._settings.timeframe
            )
            htf = self._settings.higher_timeframe.strip()
            higher_candles = slice_candles_as_of(
                self._higher_df,
                as_of,
                self._history_len,
                bar_timeframe=htf,
            )
            if higher_candles:
                higher = TimeframeSeries(
                    timeframe=self._settings.higher_timeframe.strip(),
                    candles=higher_candles,
                )

        return MultiTimeframeMarket(lower=lower, higher=higher)
