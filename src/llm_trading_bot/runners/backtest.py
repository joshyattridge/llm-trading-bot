import logging

import backtrader as bt
import pandas as pd

from llm_trading_bot.config import Settings
from llm_trading_bot.data.historical import fetch_ohlcv_dataframe
from llm_trading_bot.display import TerminalDisplay
from llm_trading_bot.strategies.llm_strategy import LLMStrategy

logger = logging.getLogger(__name__)


def _prepare_ohlcv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df.rename(
        columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
    )
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"OHLCV data must contain columns: {required}")

    return df[list(required)]


def dataframe_to_feed(df: pd.DataFrame) -> bt.feeds.PandasData:
    return bt.feeds.PandasData(dataname=_prepare_ohlcv_dataframe(df))


def llm_decision_count(candles: int, candle_history: int) -> int:
    """Bars that trigger an LLM call after warmup."""
    if candles < candle_history:
        return 0
    return candles - candle_history + 1


def run_backtest(
    settings: Settings,
    ohlcv: pd.DataFrame,
    initial_cash: float = 10_000.0,
    display: TerminalDisplay | None = None,
) -> dict:
    prepared = _prepare_ohlcv_dataframe(ohlcv)
    total_bars = len(prepared)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        LLMStrategy,
        settings=settings,
        candle_history=settings.candle_history,
        total_bars=total_bars,
        display=display,
    )
    cerebro.adddata(dataframe_to_feed(ohlcv))
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(
        commission=settings.commission_rate,
        leverage=settings.leverage,
        automargin=True,
    )

    start_value = cerebro.broker.getvalue()
    logger.debug("Starting backtest with cash=%.2f", start_value)
    cerebro.run()
    end_value = cerebro.broker.getvalue()

    index = prepared.index
    return {
        "start_value": start_value,
        "end_value": end_value,
        "pnl": end_value - start_value,
        "return_pct": ((end_value / start_value) - 1) * 100,
        "candles": len(index),
        "llm_decisions": llm_decision_count(len(index), settings.candle_history),
        "from": index[0],
        "to": index[-1],
    }


def run_backtest_from_exchange(
    settings: Settings,
    candles: int,
    initial_cash: float = 10_000.0,
    display: TerminalDisplay | None = None,
    span_label: str = "",
) -> dict:
    if display:
        display.console.print(
            f"[dim]Fetching {candles} candles from {settings.exchange_id}…[/]"
        )
    df = fetch_ohlcv_dataframe(settings, candles)
    prepared = _prepare_ohlcv_dataframe(df)
    decisions = llm_decision_count(len(prepared), settings.candle_history)
    if display:
        display.print_backtest_header(
            settings,
            len(prepared),
            decisions,
            span_label,
            prepared.index[0],
            prepared.index[-1],
        )
    return run_backtest(settings, df, initial_cash=initial_cash, display=display)
