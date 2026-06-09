import pandas as pd

from llm_trading_bot.data.market import bar_time_fields, dataframe_to_candles
from llm_trading_bot.data.serialize import candles_to_prompt
from llm_trading_bot.trading.models import Candle


def test_bar_time_fields_utc_no_date():
    ts = pd.Timestamp("2024-06-03 14:30:00", tz="UTC")
    bar_time, day_of_week = bar_time_fields(ts)
    assert bar_time == "14:30:00"
    assert day_of_week == "Monday"
    assert "2024" not in bar_time
    assert "06" not in bar_time


def test_candles_to_prompt_includes_time_and_day_not_date():
    candle = Candle(
        open=1.0,
        high=2.0,
        low=0.5,
        close=1.5,
        volume=100.0,
        bar_time="09:30:00",
        day_of_week="Tuesday",
    )
    payload = candles_to_prompt([candle], timeframe="15m")
    assert payload["timeframe"] == "15m"
    assert payload["candles"][0]["bar_time"] == "09:30:00"
    assert payload["candles"][0]["day_of_week"] == "Tuesday"
    assert payload["candles"][0]["close"] == 1.5
    serialized = str(payload)
    assert "2024" not in serialized
    assert "date" not in serialized.lower()


def test_dataframe_to_candles_populates_time_fields():
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [10.0],
        },
        index=pd.DatetimeIndex(["2024-01-15 08:00:00"], tz="UTC"),
    )
    candles = dataframe_to_candles(df)
    assert len(candles) == 1
    assert candles[0].bar_time == "08:00:00"
    assert candles[0].day_of_week == "Monday"
