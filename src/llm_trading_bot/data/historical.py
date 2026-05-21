"""Fetch historical OHLCV from exchanges via ccxt."""

from __future__ import annotations

import logging

import ccxt
import pandas as pd

from llm_trading_bot.brokers.ccxt_broker import create_exchange
from llm_trading_bot.config import Settings

logger = logging.getLogger(__name__)

MAX_FETCH_LIMIT = 1000


def fetch_ohlcv_dataframe(settings: Settings, candles: int) -> pd.DataFrame:
    """Fetch the most recent `candles` closed bars for backtesting."""
    if candles < 1:
        raise ValueError("candles must be at least 1")
    if candles > MAX_FETCH_LIMIT:
        raise ValueError(
            f"candles must be <= {MAX_FETCH_LIMIT} (exchange limit per request)"
        )

    exchange = create_exchange(settings, sandbox=False)
    exchange.load_markets()

    if settings.symbol not in exchange.symbols:
        raise ValueError(
            f"{settings.symbol} is not available on {settings.exchange_id}"
        )

    logger.debug(
        "Fetching %d %s candles for %s from %s",
        candles,
        settings.timeframe,
        settings.symbol,
        settings.exchange_id,
    )
    rows = exchange.fetch_ohlcv(
        settings.symbol,
        settings.timeframe,
        limit=candles,
    )
    if len(rows) < candles:
        raise RuntimeError(
            f"Exchange returned {len(rows)} candles, expected {candles}"
        )

    df = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop(columns=["timestamp"])
    df = df.set_index("datetime")
    return df[["open", "high", "low", "close", "volume"]]
