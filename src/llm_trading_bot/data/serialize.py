from llm_trading_bot.data.market import MultiTimeframeMarket, TimeframeSeries
from llm_trading_bot.trading.models import AccountState, Candle, PositionState


def candles_to_prompt(candles: list[Candle], *, timeframe: str) -> dict:
    """
    Serialize candles for the LLM with bar open time and day of week (UTC).
    Calendar dates are omitted. Order is oldest → newest (index 0 is earliest).
    """
    return {
        "timeframe": timeframe,
        "candle_format": {
            "bar_time": "HH:MM:SS UTC (candle open)",
            "day_of_week": "Monday–Sunday",
            "ohlcv": "open, high, low, close, volume",
        },
        "candles": [c.to_prompt_dict() for c in candles],
    }


def timeframe_series_to_prompt(series: TimeframeSeries) -> dict:
    return candles_to_prompt(series.candles, timeframe=series.timeframe)


def market_to_prompt(market: MultiTimeframeMarket) -> dict:
    payload: dict = {
        "execution_timeframe": market.lower.timeframe,
        "lower_timeframe": timeframe_series_to_prompt(market.lower),
    }
    if market.higher is not None:
        payload["higher_timeframe"] = timeframe_series_to_prompt(market.higher)
    return payload


def state_to_prompt(
    position: PositionState,
    account: AccountState,
    *,
    include_drawdown: bool = False,
    trade_history: dict | None = None,
) -> dict:
    pos: dict = {"side": position.side.value}
    if position.pending_entry:
        pos["pending_entry"] = True
    if position.side.value != "flat":
        pos["size"] = position.size
        pos["entry_price"] = position.entry_price
        pos["unrealized_pnl"] = position.unrealized_pnl
        pos["bars_in_trade"] = position.bars_in_trade
        if position.stop_loss is not None:
            pos["stop_loss"] = position.stop_loss
        if position.take_profit is not None:
            pos["take_profit"] = position.take_profit

    account_payload: dict = {
        "balance": account.balance,
        "equity": account.equity,
        "available_cash": account.available_cash,
        "currency": account.currency,
    }
    if include_drawdown and account.peak_equity is not None and account.drawdown_pct is not None:
        account_payload["peak_equity"] = account.peak_equity
        account_payload["drawdown_pct"] = account.drawdown_pct

    payload: dict = {
        "position": pos,
        "account": account_payload,
    }
    if trade_history is not None:
        payload["trade_history"] = trade_history
    return payload
