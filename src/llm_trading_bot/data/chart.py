"""Render OHLCV history as a candlestick PNG for LLM vision input."""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from llm_trading_bot.trading.models import Candle


def candles_to_chart_png_base64(
    candles: list[Candle],
    *,
    width_px: int = 960,
    height_px: int = 540,
    symbol: str = "",
    timeframe: str = "",
) -> str:
    """
    Build a candlestick chart (bar index on x-axis, no timestamps) and return base64 PNG.
    """
    if not candles:
        raise ValueError("candles must not be empty")

    fig_w = width_px / 100
    fig_h = height_px / 100
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)

    body_width = 0.6
    for i, c in enumerate(candles):
        color = "#26a69a" if c.close >= c.open else "#ef5350"
        ax.plot([i, i], [c.low, c.high], color=color, linewidth=1.0, solid_capstyle="round")
        bottom = min(c.open, c.close)
        height = abs(c.close - c.open) or (c.high - c.low) * 0.02 or 1e-8
        ax.add_patch(
            Rectangle(
                (i - body_width / 2, bottom),
                body_width,
                height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.5,
            )
        )

    ax.set_xlim(-0.8, len(candles) - 0.2)
    lows = [c.low for c in candles]
    highs = [c.high for c in candles]
    pad = (max(highs) - min(lows)) * 0.04 or 1.0
    ax.set_ylim(min(lows) - pad, max(highs) + pad)
    ax.set_xlabel("Bar index (0 = oldest, right = newest)")
    ax.set_ylabel("Price")
    title_parts = ["OHLCV candlestick chart"]
    if symbol:
        title_parts.append(symbol)
    if timeframe:
        title_parts.append(timeframe)
    ax.set_title(" · ".join(title_parts))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.standard_b64encode(buf.read()).decode("ascii")
