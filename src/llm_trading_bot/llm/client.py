import logging

from openai import OpenAI

from llm_trading_bot.config import Settings
from llm_trading_bot.data.chart import candles_to_chart_png_base64
from llm_trading_bot.data.market import MultiTimeframeMarket
from llm_trading_bot.data.serialize import market_to_prompt, state_to_prompt
from llm_trading_bot.llm.prompts import build_user_content, system_prompt
from llm_trading_bot.trading.models import (
    AccountState,
    Action,
    LLMDecision,
    LLMDecisionResponse,
    PositionState,
    PositionSide,
)
from llm_trading_bot.trading.trade_history import TradeHistoryTracker

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
        market: MultiTimeframeMarket,
        position: PositionState,
        account: AccountState,
        *,
        trade_history: TradeHistoryTracker | None = None,
    ) -> LLMDecision:
        market_payload = market_to_prompt(market)
        history_payload = (
            trade_history.to_prompt()
            if self.settings.llm_include_trade_history and trade_history is not None
            else None
        )
        state = state_to_prompt(
            position,
            account,
            include_drawdown=self.settings.llm_include_drawdown,
            trade_history=history_payload,
        )
        chart_images = (
            self._build_chart_images(market, position) if self.settings.llm_include_chart else None
        )
        user_content = build_user_content(
            market_payload,
            state,
            self._style,
            chart_images=chart_images,
        )

        schema = LLMDecisionResponse.model_json_schema()
        schema["additionalProperties"] = False

        response = self._client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt(
                        include_chart=self.settings.llm_include_chart,
                        include_higher_timeframe=market.higher is not None,
                        include_drawdown=self.settings.llm_include_drawdown,
                        include_trade_history=self.settings.llm_include_trade_history,
                    ),
                },
                {"role": "user", "content": user_content},
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

    def _build_chart_images(
        self,
        market: MultiTimeframeMarket,
        position: PositionState,
    ) -> list[tuple[str, str]]:
        """Lower timeframe chart first, then higher."""
        images: list[tuple[str, str]] = []
        symbol = self.settings.symbol
        in_position = position.side != PositionSide.FLAT
        entry_price = position.entry_price if in_position else None
        stop_loss = position.stop_loss if in_position else None
        take_profit = position.take_profit if in_position else None
        entry_bar_index = (
            self._entry_bar_index(len(market.lower.candles), position.bars_in_trade)
            if in_position
            else None
        )

        for series in (market.lower, market.higher):
            if series is None or not series.candles:
                continue
            is_execution_tf = series is market.lower
            images.append(
                (
                    series.timeframe,
                    candles_to_chart_png_base64(
                        series.candles,
                        symbol=symbol,
                        timeframe=series.timeframe,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        entry_bar_index=entry_bar_index if is_execution_tf else None,
                    ),
                )
            )
        return images

    @staticmethod
    def _entry_bar_index(candle_count: int, bars_in_trade: int) -> int | None:
        """Map bars_in_trade to a bar index in the visible execution-TF window."""
        if candle_count <= 0:
            return None
        if bars_in_trade <= 0:
            return candle_count - 1
        idx = candle_count - bars_in_trade - 1
        if idx < 0:
            return None
        return idx

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
