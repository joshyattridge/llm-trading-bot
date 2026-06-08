"""Fetch historical OHLCV from exchanges via ccxt."""

from __future__ import annotations

import logging

import ccxt
import pandas as pd

from llm_trading_bot.brokers.ccxt_broker import create_exchange
from llm_trading_bot.config import Settings
from llm_trading_bot.data.range import BacktestDateRange

logger = logging.getLogger(__name__)

# Most exchanges (e.g. Binance) cap a single fetch_ohlcv call at 1000 bars.
EXCHANGE_BATCH_LIMIT = 1000
# Maximum total candles for backtest (fetched via pagination when > EXCHANGE_BATCH_LIMIT).
MAX_FETCH_LIMIT = 5000


def _rows_to_dataframe(rows: list[list[float]]) -> pd.DataFrame:
    df = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop(columns=["timestamp"])
    df = df.drop_duplicates(subset=["datetime"], keep="last")
    df = df.sort_index()
    return df.set_index("datetime")[["open", "high", "low", "close", "volume"]]


def _fetch_ohlcv_range(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
) -> list[list[float]]:
    """Fetch bars with open time in [since_ms, until_ms], paginating forward."""
    collected: list[list[float]] = []
    cursor = since_ms

    while cursor <= until_ms and len(collected) < MAX_FETCH_LIMIT:
        batch = exchange.fetch_ohlcv(
            symbol,
            timeframe,
            since=cursor,
            limit=EXCHANGE_BATCH_LIMIT,
        )
        if not batch:
            break

        for row in batch:
            ts = row[0]
            if ts < since_ms:
                continue
            if ts >= until_ms:
                break
            collected.append(row)

        if batch[-1][0] >= until_ms or len(batch) < EXCHANGE_BATCH_LIMIT:
            break

        cursor = batch[-1][0] + 1

    return collected


def fetch_ohlcv_dataframe_for_range(
    settings: Settings,
    window: BacktestDateRange,
    *,
    timeframe: str | None = None,
    warmup_bars: int = 0,
) -> pd.DataFrame:
    """Fetch OHLCV for a backtest window plus optional warmup bars before start."""
    tf = timeframe or settings.timeframe
    fetch_start = window.start
    if warmup_bars > 0:
        from llm_trading_bot.data.market import timeframe_to_timedelta

        fetch_start = fetch_start - warmup_bars * timeframe_to_timedelta(tf)

    since_ms = int(fetch_start.timestamp() * 1000)
    until_ms = int(window.end_exclusive.timestamp() * 1000)

    exchange = create_exchange(settings, sandbox=False)
    exchange.load_markets()

    if settings.symbol not in exchange.symbols:
        raise ValueError(
            f"{settings.symbol} is not available on {settings.exchange_id}"
        )

    logger.debug(
        "Fetching %s %s candles for %s from %s (%s → %s)",
        tf,
        settings.symbol,
        settings.exchange_id,
        fetch_start,
        window.end_exclusive,
    )
    rows = _fetch_ohlcv_range(
        exchange,
        settings.symbol,
        tf,
        since_ms,
        until_ms,
    )
    if not rows:
        raise RuntimeError(
            f"Exchange returned no candles for {window.label()} on {tf}"
        )

    df = _rows_to_dataframe(rows)
    if len(df) > MAX_FETCH_LIMIT:
        raise ValueError(
            f"Date range requires {len(df)} bars, exceeding limit of {MAX_FETCH_LIMIT}"
        )
    return df


def fetch_backtest_dataframe(
    settings: Settings,
    window: BacktestDateRange,
) -> pd.DataFrame:
    """Fetch lower-timeframe OHLCV with warmup for the given backtest window."""
    warmup = max(settings.candle_history - 1, 0)
    return fetch_ohlcv_dataframe_for_range(
        settings,
        window,
        warmup_bars=warmup,
    )
