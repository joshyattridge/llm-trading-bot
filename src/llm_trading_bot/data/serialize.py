from llm_trading_bot.data.market import MultiTimeframeMarket, TimeframeSeries
from llm_trading_bot.trading.models import AccountState, Candle, PositionState


def candles_to_prompt(candles: list[Candle], *, timeframe: str) -> dict:
    """
    Serialize candles for the LLM without any dates or indices that imply time.
    Order is oldest → newest (index 0 is earliest visible bar).
    """
    return {
        "timeframe": timeframe,
        "candle_format": "[open, high, low, close, volume]",
        "candles": [c.as_list() for c in candles],
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

    return {
        "position": pos,
        "account": {
            "balance": account.balance,
            "equity": account.equity,
            "available_cash": account.available_cash,
            "currency": account.currency,
        },
    }
