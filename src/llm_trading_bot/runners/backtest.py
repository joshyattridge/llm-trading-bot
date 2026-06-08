import logging

import backtrader as bt
import pandas as pd

from llm_trading_bot.config import Settings
from llm_trading_bot.data.historical import (
    fetch_backtest_dataframe,
    fetch_ohlcv_dataframe_for_range,
)
from llm_trading_bot.data.market import higher_timeframe_fetch_count, timeframe_to_timedelta
from llm_trading_bot.data.range import (
    BacktestDateRange,
    count_bars_in_window,
    slice_window,
    validate_backtest_window,
)
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


def _load_higher_timeframe_for_window(
    settings: Settings,
    window: BacktestDateRange,
    lower_df: pd.DataFrame,
) -> pd.DataFrame | None:
    if not settings.uses_higher_timeframe():
        return None

    htf = settings.higher_timeframe.strip()
    lower_bar_count = count_bars_in_window(lower_df, window)
    htf_warmup = higher_timeframe_fetch_count(settings.candle_history, lower_bar_count)
    htf_fetch_start = window.start - htf_warmup * timeframe_to_timedelta(htf)
    htf_window = BacktestDateRange(start=htf_fetch_start, end_exclusive=window.end_exclusive)
    df = fetch_ohlcv_dataframe_for_range(
        settings,
        htf_window,
        timeframe=htf,
    )
    return _prepare_ohlcv_dataframe(df)


def run_backtest(
    settings: Settings,
    ohlcv: pd.DataFrame,
    window: BacktestDateRange,
    initial_cash: float = 10_000.0,
    display: TerminalDisplay | None = None,
    *,
    higher_ohlcv: pd.DataFrame | None = None,
) -> dict:
    prepared = _prepare_ohlcv_dataframe(ohlcv)
    validate_backtest_window(prepared, window, candle_history=settings.candle_history)

    if higher_ohlcv is None:
        higher_ohlcv = _load_higher_timeframe_for_window(settings, window, prepared)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        LLMStrategy,
        settings=settings,
        candle_history=settings.candle_history,
        total_bars=len(prepared),
        display=display,
        higher_ohlcv=higher_ohlcv,
    )
    cerebro.adddata(dataframe_to_feed(ohlcv))
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.set_coc(True)
    cerebro.broker.setcommission(
        commission=settings.commission_rate,
        leverage=settings.leverage,
        automargin=True,
    )
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

    start_value = cerebro.broker.getvalue()
    logger.debug("Starting backtest with cash=%.2f", start_value)
    results = cerebro.run()
    end_value = cerebro.broker.getvalue()
    drawdown = results[0].analyzers.drawdown.get_analysis()

    decision_df = slice_window(prepared, window)
    return {
        "start_value": start_value,
        "end_value": end_value,
        "pnl": end_value - start_value,
        "return_pct": ((end_value / start_value) - 1) * 100,
        "max_drawdown_pct": float(drawdown.get("max", {}).get("drawdown", 0.0) or 0.0),
        "candles": len(decision_df),
        "llm_decisions": len(decision_df),
        "from": decision_df.index[0],
        "to": decision_df.index[-1],
        "window": window.label(),
    }


def run_backtest_for_range(
    settings: Settings,
    start: str,
    end: str,
    initial_cash: float = 10_000.0,
    display: TerminalDisplay | None = None,
) -> dict:
    window = BacktestDateRange.parse(start, end, timeframe=settings.timeframe)
    if display:
        htf = settings.higher_timeframe.strip() if settings.uses_higher_timeframe() else None
        msg = (
            f"[dim]Fetching {settings.timeframe} candles for {window.label()} "
            f"from {settings.exchange_id}"
        )
        if htf:
            msg += f" (+ {htf} higher timeframe)"
        display.console.print(msg + "…[/]")

    df = fetch_backtest_dataframe(settings, window)
    prepared = _prepare_ohlcv_dataframe(df)
    validate_backtest_window(prepared, window, candle_history=settings.candle_history)
    decision_df = slice_window(prepared, window)

    if display:
        display.print_backtest_header(
            settings,
            len(decision_df),
            len(decision_df),
            window.label(),
            decision_df.index[0],
            decision_df.index[-1],
        )

    return run_backtest(
        settings,
        df,
        window,
        initial_cash=initial_cash,
        display=display,
    )
