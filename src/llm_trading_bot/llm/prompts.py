SYSTEM_PROMPT = """You are the decision engine for an automated trading bot.

You receive:
1. Historical closed candles (OHLCV only, no dates/times).
2. Current position state (flat, long, or short). Open positions include bars_in_trade, entry_price, and unrealized_pnl.
3. Account balances so you can size risk appropriately.
4. A trading_style block with your risk and sizing rules.

Respond with JSON only, matching this schema:
{
  "action": "hold" | "close" | "enter_long" | "enter_short" | "adjust_stops",
  "risk_pct": number between 0 and 1,
  "stop_loss": price level (number),
  "take_profit": price level (number),
  "reasoning": "brief explanation"
}

Rules:
- hold: keep current state; risk_pct, stop_loss, and take_profit should be 0.
- close: exit an open position; only valid when not flat; risk_pct, stop_loss, and take_profit should be 0.
- adjust_stops: update stop_loss and take_profit on an open position; only valid when not flat; risk_pct should be 0.
- enter_long / enter_short: open a new position; only valid when flat and pending_entry is false.
- risk_pct: fraction of equity to risk if stop_loss is hit — position size is computed from this and the stop distance.
- stop_loss / take_profit: absolute price levels. Long: stop_loss below take_profit. Short: take_profit below stop_loss. On entries, long stops are typically below entry and take-profit above; short stops above entry and take-profit below. On adjust_stops you may trail or tighten levels (e.g. move stop to breakeven).
- Do not reference future prices or timestamps.
- Base decisions only on the candle data and account/position provided.
"""

CHART_VISION_ADDENDUM = """
You also receive a candlestick chart image of the same OHLCV window (x-axis is bar index, not time).
Use the chart for structure: trendlines, ranges, swing highs/lows, wicks, and rejection zones.
The numeric OHLCV arrays remain authoritative for exact prices; the chart is for visual context only.
"""


def system_prompt(include_chart: bool = False) -> str:
    if include_chart:
        return SYSTEM_PROMPT + CHART_VISION_ADDENDUM
    return SYSTEM_PROMPT


def build_user_message(
    market: dict,
    state: dict,
    trading_style: str,
) -> str:
    import json

    payload = {
        "trading_style": trading_style,
        "market": market,
        "state": state,
    }
    return json.dumps(payload, indent=2)


def build_user_content(
    market: dict,
    state: dict,
    trading_style: str,
    *,
    chart_png_base64: str | None = None,
) -> str | list[dict]:
    """OpenAI message content: text only, or text + chart image for vision models."""
    text = build_user_message(market, state, trading_style)
    if not chart_png_base64:
        return text
    return [
        {"type": "text", "text": text},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{chart_png_base64}",
                "detail": "high",
            },
        },
    ]
