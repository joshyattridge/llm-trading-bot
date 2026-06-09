"""Multi-timeframe market data for LLM decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from llm_trading_bot.trading.models import Candle


def bar_time_fields(ts: datetime | pd.Timestamp) -> tuple[str, str]:
    """Candle open time (UTC) and day name — no calendar date."""
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.strftime("%H:%M:%S"), stamp.day_name()


@dataclass(frozen=True)
class TimeframeSeries:
    timeframe: str
    candles: list[Candle]


@dataclass(frozen=True)
class MultiTimeframeMarket:
    """Lower timeframe drives entries; higher timeframe provides context."""

    lower: TimeframeSeries
    higher: TimeframeSeries | None = None


def dataframe_to_candles(df: pd.DataFrame) -> list[Candle]:
    candles: list[Candle] = []
    for row in df.itertuples():
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


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    """Parse ccxt-style timeframe strings (e.g. 1h, 1d, 15m)."""
    tf = timeframe.strip().lower()
    for suffix, unit in (("m", "minutes"), ("h", "hours"), ("d", "days"), ("w", "weeks")):
        if tf.endswith(suffix) and tf[:-1].isdigit():
            return pd.Timedelta(**{unit: int(tf[:-1])})
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def slice_candles_as_of(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    count: int,
    *,
    bar_timeframe: str | None = None,
) -> list[Candle]:
    """
    Return up to `count` closed candles known at `as_of`.

    Exchange rows are indexed by candle **open** time. A bar is only
    "closed" once `open_time + bar_timeframe <= as_of`. Using `index <= as_of`
    alone would include the in-progress daily bar during an intraday 1h step.
    """
    if df.empty:
        return []
    as_of = pd.Timestamp(as_of)
    if df.index.tz is not None:
        if as_of.tzinfo is None:
            as_of = as_of.tz_localize("UTC")
        as_of = as_of.tz_convert(df.index.tz)

    if bar_timeframe:
        period = timeframe_to_timedelta(bar_timeframe)
        closed_mask = df.index + period <= as_of
        subset = df.loc[closed_mask]
    else:
        subset = df[df.index <= as_of]

    return dataframe_to_candles(subset.tail(count))


def higher_timeframe_fetch_count(candle_history: int, lower_bar_count: int) -> int:
    """Enough higher-TF bars to fill `candle_history` at the earliest lower bar."""
    return min(candle_history + max(lower_bar_count, 1), 5000)
