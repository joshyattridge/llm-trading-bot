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
    entry_price: float | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    entry_bar_index: int | None = None,
) -> str:
    """
    Build a candlestick chart (bar index on x-axis, no timestamps) and return base64 PNG.
    When entry_price / stop_loss / take_profit are set, draw labeled horizontal levels.
    entry_bar_index marks the execution-timeframe bar where the position was opened.
    """
    if not candles:
        raise ValueError("candles must not be empty")

    fig_w = width_px / 100
    fig_h = height_px / 100
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)

    highlight_entry = (
        entry_bar_index is not None
        and 0 <= entry_bar_index < len(candles)
    )
    if highlight_entry:
        ax.axvspan(
            entry_bar_index - 0.5,
            entry_bar_index + 0.5,
            color="#1565c0",
            alpha=0.12,
            zorder=0,
        )

    body_width = 0.6
    for i, c in enumerate(candles):
        color = "#26a69a" if c.close >= c.open else "#ef5350"
        is_entry_bar = highlight_entry and i == entry_bar_index
        ax.plot([i, i], [c.low, c.high], color=color, linewidth=1.0, solid_capstyle="round")
        bottom = min(c.open, c.close)
        height = abs(c.close - c.open) or (c.high - c.low) * 0.02 or 1e-8
        ax.add_patch(
            Rectangle(
                (i - body_width / 2, bottom),
                body_width,
                height,
                facecolor=color,
                edgecolor="#1565c0" if is_entry_bar else color,
                linewidth=2.5 if is_entry_bar else 0.5,
                zorder=3 if is_entry_bar else 2,
            )
        )

    x_right = len(candles) - 0.2
    ax.set_xlim(-0.8, x_right)

    lows = [c.low for c in candles]
    highs = [c.high for c in candles]
    y_refs = lows + highs
    for level in (entry_price, stop_loss, take_profit):
        if level is not None:
            y_refs.append(level)
    y_min, y_max = min(y_refs), max(y_refs)
    pad = (y_max - y_min) * 0.04 or 1.0
    ax.set_ylim(y_min - pad, y_max + pad)

    def _draw_level(price: float, color: str, label: str) -> None:
        ax.axhline(price, color=color, linestyle="--", linewidth=1.4, alpha=0.9)
        ax.text(
            x_right,
            price,
            f" {label}",
            va="center",
            ha="left",
            color=color,
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor=color),
        )

    if entry_price is not None:
        _draw_level(entry_price, "#1565c0", f"Entry {entry_price:,.2f}")
    if stop_loss is not None:
        _draw_level(stop_loss, "#c62828", f"Stop {stop_loss:,.2f}")
    if take_profit is not None:
        _draw_level(take_profit, "#2e7d32", f"TP {take_profit:,.2f}")

    if highlight_entry:
        entry_candle = candles[entry_bar_index]
        label_y = entry_candle.low - (y_max - y_min) * 0.035
        ax.text(
            entry_bar_index,
            label_y,
            "Entry bar",
            ha="center",
            va="top",
            fontsize=9,
            color="#1565c0",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, edgecolor="#1565c0"),
        )
        ax.set_ylim(min(y_min - pad, label_y - pad * 0.25), y_max + pad)
    ax.set_xlabel("Bar index (0 = oldest, right = newest)")
    ax.set_ylabel("Price")
    bar_count = len(candles)
    if timeframe and symbol:
        title = f"{timeframe} · {symbol}"
    elif timeframe:
        title = timeframe
    elif symbol:
        title = symbol
    else:
        title = "OHLCV"
    ax.set_title(f"{title}  ·  {bar_count} bars", fontsize=14, fontweight="bold")
    if timeframe:
        ax.text(
            0.01,
            0.98,
            f"Timeframe: {timeframe}",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.standard_b64encode(buf.read()).decode("ascii")
