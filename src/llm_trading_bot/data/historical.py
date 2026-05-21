"""Fetch historical OHLCV from exchanges via ccxt."""

from __future__ import annotations

import logging

import ccxt
import pandas as pd

from llm_trading_bot.brokers.ccxt_broker import create_exchange
from llm_trading_bot.config import Settings

logger = logging.getLogger(__name__)

# Most exchanges (e.g. Binance) cap a single fetch_ohlcv call at 1000 bars.
EXCHANGE_BATCH_LIMIT = 1000
# Maximum total candles for backtest (fetched via pagination when > EXCHANGE_BATCH_LIMIT).
MAX_FETCH_LIMIT = 5000


def _fetch_ohlcv_paginated(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    candles: int,
) -> list[list[float]]:
    """Fetch up to `candles` most recent closed bars, paginating if needed."""
    if candles <= EXCHANGE_BATCH_LIMIT:
        rows = exchange.fetch_ohlcv(symbol, timeframe, limit=candles)
        return rows

    collected: list[list[float]] = []
    end_time: int | None = None

    while len(collected) < candles:
        remaining = candles - len(collected)
        batch_limit = min(remaining, EXCHANGE_BATCH_LIMIT)
        kwargs: dict = {"limit": batch_limit}
        if end_time is not None:
            kwargs["params"] = {"endTime": end_time}

        batch = exchange.fetch_ohlcv(symbol, timeframe, **kwargs)
        if not batch:
            break

        if collected:
            oldest_collected = collected[0][0]
            batch = [row for row in batch if row[0] < oldest_collected]

        if not batch:
            break

        collected = batch + collected
        end_time = batch[0][0] - 1

        if len(batch) < batch_limit:
            break

    return collected[-candles:]


def fetch_ohlcv_dataframe(settings: Settings, candles: int) -> pd.DataFrame:
    """Fetch the most recent `candles` closed bars for backtesting."""
    if candles < 1:
        raise ValueError("candles must be at least 1")
    if candles > MAX_FETCH_LIMIT:
        raise ValueError(
            f"candles must be <= {MAX_FETCH_LIMIT} (paginated fetch limit)"
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
    rows = _fetch_ohlcv_paginated(
        exchange,
        settings.symbol,
        settings.timeframe,
        candles,
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
