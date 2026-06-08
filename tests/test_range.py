import pandas as pd
import pytest

from llm_trading_bot.data.range import (
    BacktestDateRange,
    count_bars_in_window,
    parse_range_end_exclusive,
    parse_range_start,
    validate_backtest_window,
)


class TestBacktestDateRange:
    def test_parse_date_only_end_is_inclusive(self):
        window = BacktestDateRange.parse("2024-01-01", "2024-01-03", timeframe="1h")
        assert window.start == pd.Timestamp("2024-01-01", tz="UTC")
        assert window.end_exclusive == pd.Timestamp("2024-01-04", tz="UTC")

    def test_parse_rejects_inverted_range(self):
        with pytest.raises(ValueError, match="before end"):
            BacktestDateRange.parse("2024-01-10", "2024-01-01", timeframe="1h")

    def test_slice_and_validate_warmup(self):
        index = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        df = pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
            index=index,
        )
        window = BacktestDateRange(
            start=parse_range_start("2024-01-01T02:00:00"),
            end_exclusive=parse_range_end_exclusive("2024-01-01T04:00:00", timeframe="1h"),
        )
        assert count_bars_in_window(df, window) == 3

        short_window = BacktestDateRange.parse(
            "2024-01-01T05:00:00",
            "2024-01-01T07:00:00",
            timeframe="1h",
        )
        with pytest.raises(ValueError, match="warmup"):
            validate_backtest_window(df, short_window, candle_history=50)
