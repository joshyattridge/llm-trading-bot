import logging
from typing import TYPE_CHECKING, Any

from llm_trading_bot.llm.client import LLMTradingAdvisor
from llm_trading_bot.trading.executor import (
    BrokerAdapter,
    execute_decision,
    sizing_price_for_entry,
)
from llm_trading_bot.trading.models import (
    Action,
    Candle,
    ExecutionOutcome,
    LLMDecision,
    PositionSide,
)

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
        commission_rate: float = 0.001,
        leverage: float = 1.0,
    ):
        self.advisor = advisor
        self.broker = broker
        self.display = display
        self.symbol = symbol
        self.timeframe = timeframe
        self.commission_rate = commission_rate
        self.leverage = leverage
        self._bars_in_trade = 0
        self._pending_panel: dict[str, Any] | None = None
        self._display_flushed = False
        self._wait_for_order_notify = False

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

        if self._pending_panel and self._wait_for_order_notify:
            self._wait_for_order_notify = False
            self.flush_display()

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
                    outcome=ExecutionOutcome.CLOSED,
                )
            return

        if self.broker.has_pending_entry():
            self._defer_panel(
                bar=bar,
                total_bars=total_bars,
                close_price=close_price,
                decision=LLMDecision(
                    action=Action.HOLD,
                    reasoning="Waiting for pending entry order to fill.",
                ),
                outcome=ExecutionOutcome.SKIPPED_PENDING_ENTRY,
            )
            self.flush_display()
            return

        position = self._position_for_llm()
        if position.side == PositionSide.FLAT:
            self._bars_in_trade = 0
        else:
            self._bars_in_trade += 1
            position = position.model_copy(
                update={"bars_in_trade": self._bars_in_trade},
            )
        account = self.broker.get_account(mark_price=close_price)

        decision = self.advisor.decide(history, position, account)
        sizing_price = None
        if decision.action in (Action.ENTER_LONG, Action.ENTER_SHORT):
            side = (
                PositionSide.LONG
                if decision.action == Action.ENTER_LONG
                else PositionSide.SHORT
            )
            sizing_price = sizing_price_for_entry(
                side, candle.close, candle.high, candle.low
            )

        outcome = execute_decision(
            self.broker,
            decision,
            close_price,
            commission_rate=self.commission_rate,
            leverage=self.leverage,
            sizing_price=sizing_price,
        )

        position = self.broker.get_position()
        account = self.broker.get_account(mark_price=close_price)
        self._defer_panel(
            bar=bar,
            total_bars=total_bars,
            close_price=close_price,
            position=position,
            account=account,
            decision=decision,
            outcome=outcome,
        )
        self._wait_for_order_notify = decision.action in (
            Action.ENTER_LONG,
            Action.ENTER_SHORT,
        ) and outcome not in (
            ExecutionOutcome.SKIPPED_INVALID_LEVELS,
            ExecutionOutcome.SKIPPED_ZERO_SIZE,
            ExecutionOutcome.SKIPPED_PENDING_ENTRY,
            ExecutionOutcome.SKIPPED_ORDER_REJECTED,
        )

    def on_entry_order_settled(self) -> None:
        if self._pending_panel is None:
            return

        rejected = getattr(self.broker, "consume_entry_rejected", lambda: False)()
        getattr(self.broker, "consume_entry_settled", lambda: False)()

        if rejected:
            self._pending_panel["outcome"] = ExecutionOutcome.SKIPPED_ORDER_REJECTED
        else:
            position = self.broker.get_position()
            if position.side != PositionSide.FLAT:
                self._pending_panel["outcome"] = ExecutionOutcome.EXECUTED
            elif self.broker.has_pending_entry():
                self._pending_panel["outcome"] = ExecutionOutcome.ORDER_SUBMITTED

        mark = self._pending_panel["close_price"]
        self._pending_panel["position"] = self.broker.get_position()
        self._pending_panel["account"] = self.broker.get_account(mark_price=mark)
        self._wait_for_order_notify = False
        self.flush_display()

    def waiting_for_order_notify(self) -> bool:
        return self._wait_for_order_notify

    def flush_display(self) -> None:
        if not self.display or self._pending_panel is None or self._display_flushed:
            return
        if self._wait_for_order_notify:
            return
        self.display.print_candle(**self._pending_panel)
        self._display_flushed = True
        self._pending_panel = None

    def _defer_panel(
        self,
        *,
        bar: int | None,
        total_bars: int | None,
        close_price: float,
        decision: LLMDecision,
        outcome: ExecutionOutcome,
        position=None,
        account=None,
    ) -> None:
        if position is None:
            position = self.broker.get_position()
        if account is None:
            account = self.broker.get_account(mark_price=close_price)
        self._pending_panel = {
            "bar": bar,
            "total_bars": total_bars,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "close_price": close_price,
            "position": position,
            "account": account,
            "decision": decision,
            "outcome": outcome,
        }
        self._display_flushed = False

    def _position_for_llm(self):
        position = self.broker.get_position()
        if position.side != PositionSide.FLAT:
            return position
        pending = self.broker.has_pending_entry()
        if pending == position.pending_entry:
            return position
        return position.model_copy(update={"pending_entry": pending})

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
