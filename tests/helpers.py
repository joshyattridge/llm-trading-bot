"""Test helpers."""

from __future__ import annotations

from llm_trading_bot.trading.models import Action, LLMDecision


def make_decision(
    action: Action,
    *,
    risk_pct: float = 0.0,
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
    reasoning: str = "",
) -> LLMDecision:
    return LLMDecision(
        action=action,
        risk_pct=risk_pct,
        stop_loss=stop_loss,
        take_profit=take_profit,
        reasoning=reasoning,
    )
