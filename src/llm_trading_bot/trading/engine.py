import logging
from typing import TYPE_CHECKING

from llm_trading_bot.llm.client import LLMTradingAdvisor
from llm_trading_bot.trading.executor import BrokerAdapter, execute_decision
from llm_trading_bot.trading.models import Action, Candle, LLMDecision, PositionSide

if TYPE_CHECKING:
    from llm_trading_bot.display import TerminalDisplay

logger = logging.getLogger(__name__)


class TradingEngine:
    """On each new closed candle: check SL/TP, consult the LLM, and execute."""

    def __init__(
        self,
        advisor: LLMTradingAdvisor,
        broker: BrokerAdapter,
        display: TerminalDisplay | None = None,
        *,
        symbol: str = "",
        timeframe: str = "",
    ):
        self.advisor = advisor
        self.broker = broker
        self.display = display
        self.symbol = symbol
        self.timeframe = timeframe
        self._bars_in_trade = 0

    def on_new_candle(
        self,
        history: list[Candle],
        candle: Candle,
        *,
        bar: int | None = None,
        total_bars: int | None = None,
    ) -> None:
        if not history:
            return

        close_price = candle.close
        stop_decision = self._check_stops(candle)
        if stop_decision:
            if self.display:
                position = self.broker.get_position()
                account = self.broker.get_account(mark_price=close_price)
                self.display.print_candle(
                    bar=bar,
                    total_bars=total_bars,
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    close_price=close_price,
                    position=position,
                    account=account,
                    decision=stop_decision,
                )
            return

        position = self.broker.get_position()
        if position.side == PositionSide.FLAT:
            self._bars_in_trade = 0
        else:
            self._bars_in_trade += 1
            position = position.model_copy(
                update={"bars_in_trade": self._bars_in_trade},
            )
        account = self.broker.get_account(mark_price=close_price)

        decision = self.advisor.decide(history, position, account)
        execute_decision(self.broker, decision, close_price)

        if self.display:
            position = self.broker.get_position()
            account = self.broker.get_account(mark_price=close_price)
            self.display.print_candle(
                bar=bar,
                total_bars=total_bars,
                symbol=self.symbol,
                timeframe=self.timeframe,
                close_price=close_price,
                position=position,
                account=account,
                decision=decision,
            )

    def _check_stops(self, candle: Candle) -> LLMDecision | None:
        position = self.broker.get_position()
        if position.side == PositionSide.FLAT:
            return None

        stop_loss = position.stop_loss
        take_profit = position.take_profit
        if stop_loss is None and take_profit is None:
            return None

        hit_reason = None
        if position.side == PositionSide.LONG:
            if stop_loss is not None and candle.low <= stop_loss:
                hit_reason = f"stop loss hit at {stop_loss:.2f}"
            elif take_profit is not None and candle.high >= take_profit:
                hit_reason = f"take profit hit at {take_profit:.2f}"
        elif position.side == PositionSide.SHORT:
            if stop_loss is not None and candle.high >= stop_loss:
                hit_reason = f"stop loss hit at {stop_loss:.2f}"
            elif take_profit is not None and candle.low <= take_profit:
                hit_reason = f"take profit hit at {take_profit:.2f}"

        if not hit_reason:
            return None

        logger.debug("Closing position — %s", hit_reason)
        self.broker.close_position()
        self._bars_in_trade = 0
        return LLMDecision(
            action=Action.CLOSE,
            risk_pct=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            reasoning=hit_reason,
        )
