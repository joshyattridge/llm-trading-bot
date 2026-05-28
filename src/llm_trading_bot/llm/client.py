import logging

from openai import OpenAI

from llm_trading_bot.config import Settings
from llm_trading_bot.data.serialize import candles_to_prompt, state_to_prompt
from llm_trading_bot.llm.prompts import SYSTEM_PROMPT, build_user_message
from llm_trading_bot.trading.models import (
    AccountState,
    Action,
    Candle,
    LLMDecision,
    LLMDecisionResponse,
    PositionState,
    PositionSide,
)

logger = logging.getLogger(__name__)


class LLMTradingAdvisor:
    def __init__(self, settings: Settings):
        self.settings = settings
        kwargs: dict = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        self._client = OpenAI(**kwargs)
        self._style = settings.load_trading_style_prompt()

    def decide(
        self,
        candles: list[Candle],
        position: PositionState,
        account: AccountState,
    ) -> LLMDecision:
        market = candles_to_prompt(candles)
        state = state_to_prompt(position, account)
        user_msg = build_user_message(market, state, self._style)

        schema = LLMDecisionResponse.model_json_schema()
        schema["additionalProperties"] = False

        response = self._client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "trading_decision",
                    "strict": True,
                    "schema": schema,
                },
            },
            temperature=0.2,
        )

        raw = response.choices[0].message.content or "{}"
        parsed = LLMDecisionResponse.model_validate_json(raw)
        decision = LLMDecision(
            action=Action(parsed.action),
            risk_pct=parsed.risk_pct,
            stop_loss=parsed.stop_loss,
            take_profit=parsed.take_profit,
            reasoning=parsed.reasoning,
        )
        return self._validate_decision(decision, position)

    def _validate_decision(
        self,
        decision: LLMDecision,
        position: PositionState,
    ) -> LLMDecision:
        """Enforce position constraints the LLM might violate."""
        side = position.side
        invalid = LLMDecision(
            action=Action.HOLD,
            risk_pct=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            reasoning="invalid decision",
        )

        if decision.action == Action.CLOSE and side == PositionSide.FLAT:
            logger.warning("LLM requested close while flat; forcing hold")
            return invalid

        if decision.action in (Action.ENTER_LONG, Action.ENTER_SHORT) and side != PositionSide.FLAT:
            logger.warning("LLM requested entry while in position; forcing hold")
            return invalid

        if decision.action in (Action.ENTER_LONG, Action.ENTER_SHORT) and position.pending_entry:
            logger.warning("LLM requested entry while order pending; forcing hold")
            return invalid

        if decision.action in (Action.ENTER_LONG, Action.ENTER_SHORT):
            if decision.risk_pct <= 0:
                logger.warning("LLM requested entry with zero risk_pct; forcing hold")
                return invalid
            if decision.stop_loss <= 0 or decision.take_profit <= 0:
                logger.warning("LLM requested entry without stop_loss/take_profit; forcing hold")
                return invalid

        if decision.action == Action.ADJUST_STOPS:
            if side == PositionSide.FLAT:
                logger.warning("LLM requested adjust_stops while flat; forcing hold")
                return invalid
            if decision.stop_loss <= 0 or decision.take_profit <= 0:
                logger.warning(
                    "LLM requested adjust_stops without stop_loss/take_profit; forcing hold"
                )
                return invalid

        return decision
