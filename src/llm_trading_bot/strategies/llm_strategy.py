import logging

import backtrader as bt

from llm_trading_bot.brokers.backtrader_broker import BacktraderBrokerAdapter
from llm_trading_bot.config import Settings
from llm_trading_bot.display import TerminalDisplay
from llm_trading_bot.llm.client import LLMTradingAdvisor
from llm_trading_bot.trading.engine import TradingEngine
from llm_trading_bot.trading.models import Candle

logger = logging.getLogger(__name__)


class LLMStrategy(bt.Strategy):
    params = (
        ("candle_history", 50),
        ("settings", None),
        ("total_bars", 0),
        ("display", None),
    )

    def __init__(self):
        settings: Settings = self.p.settings
        display: TerminalDisplay | None = self.p.display
        self._history_len = self.p.candle_history
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
            self._engine.on_entry_order_settled()

    def next(self):
        # Only act on completed bars (backtrader calls next once per bar at close)
        self._bar_count += 1
        if self._bar_count < self._history_len:
            return

        candles = self._build_history()
        self._engine.on_new_candle(
            candles,
            candles[-1],
            bar=self._bar_count,
            total_bars=self._total_bars or None,
        )
        if not self._engine.waiting_for_order_notify():
            self._engine.flush_display()

    def _build_history(self) -> list[Candle]:
        n = self._history_len
        candles: list[Candle] = []
        for i in range(-n, 0):
            candles.append(
                Candle(
                    open=float(self.data.open[i]),
                    high=float(self.data.high[i]),
                    low=float(self.data.low[i]),
                    close=float(self.data.close[i]),
                    volume=float(self.data.volume[i]),
                )
            )
        return candles
