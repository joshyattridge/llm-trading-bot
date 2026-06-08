"""Date-range parsing and bounds for backtests."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from llm_trading_bot.data.market import timeframe_to_timedelta


def _looks_date_only(value: str) -> bool:
    raw = value.strip()
    return "T" not in raw and " " not in raw and len(raw) <= 10


def parse_range_start(value: str) -> pd.Timestamp:
    return pd.to_datetime(value.strip(), utc=True)


def parse_range_end_exclusive(value: str, *, timeframe: str) -> pd.Timestamp:
    """Return exclusive upper bound for bar open times in the user range."""
    raw = value.strip()
    if _looks_date_only(raw):
        return pd.Timestamp(raw, tz="UTC") + pd.Timedelta(days=1)
    return pd.to_datetime(raw, utc=True) + timeframe_to_timedelta(timeframe)


@dataclass(frozen=True)
class BacktestDateRange:
    """Inclusive backtest window on bar open times."""

    start: pd.Timestamp
    end_exclusive: pd.Timestamp

    @classmethod
    def parse(cls, start: str, end: str, *, timeframe: str) -> BacktestDateRange:
        start_ts = parse_range_start(start)
        end_ex = parse_range_end_exclusive(end, timeframe=timeframe)
        if start_ts >= end_ex:
            raise ValueError(f"Start ({start}) must be before end ({end})")
        return cls(start=start_ts, end_exclusive=end_ex)

    def label(self) -> str:
        end_inclusive = self.end_exclusive - pd.Timedelta(microseconds=1)
        return f"{self.start.date()} → {end_inclusive.date()}"


def slice_window(df: pd.DataFrame, window: BacktestDateRange) -> pd.DataFrame:
    return df[(df.index >= window.start) & (df.index < window.end_exclusive)]


def count_bars_in_window(df: pd.DataFrame, window: BacktestDateRange) -> int:
    return len(slice_window(df, window))


def validate_backtest_window(
    df: pd.DataFrame,
    window: BacktestDateRange,
    *,
    candle_history: int,
) -> None:
    in_window = slice_window(df, window)
    if in_window.empty:
        raise ValueError(
            f"No {window.label()} bars in fetched data "
            f"({df.index.min()} → {df.index.max()})"
        )

    warmup = df[df.index < window.start]
    needed = max(candle_history - 1, 0)
    if len(warmup) < needed:
        raise ValueError(
            f"Need {needed} warmup bars before {window.start}, "
            f"but only {len(warmup)} available — choose a later start date"
        )
