from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from llm_trading_bot.llm.client import LLMTradingAdvisor
from llm_trading_bot.trading.executor import (
    BrokerAdapter,
    execute_decision,
    sizing_price_for_entry,
)
from llm_trading_bot.data.market import MultiTimeframeMarket
from llm_trading_bot.trading.models import (
    Action,
    Candle,
    ExecutionOutcome,
    LLMDecision,
    PositionSide,
)
from llm_trading_bot.trading.drawdown import DrawdownTracker
from llm_trading_bot.trading.stops import StopHit, check_stop_hit

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
        self._drawdown_tracker = DrawdownTracker()
        self._bars_in_trade = 0
        self._pending_panel: dict[str, Any] | None = None
        self._display_flushed = False
        self._wait_for_order_notify = False

    def on_new_candle(
        self,
        market: MultiTimeframeMarket,
        candle: Candle,
        *,
        bar: int | None = None,
        total_bars: int | None = None,
    ) -> None:
        if not market.lower.candles:
            return

        close_price = candle.close

        stop_hit = self._check_stops(candle, close_price)
        if stop_hit:
            self._defer_panel(
                bar=bar,
                total_bars=total_bars,
                close_price=close_price,
                decision=LLMDecision(
                    action=Action.CLOSE,
                    risk_pct=0.0,
                    stop_loss=0.0,
                    take_profit=0.0,
                    reasoning=stop_hit.reason,
                ),
                outcome=ExecutionOutcome.CLOSED,
            )
            self._wait_for_order_notify = True
            return

        if self.broker.has_pending_entry():
            self._defer_panel(
                bar=bar,
                total_bars=total_bars,
                close_price=close_price,
                decision=LLMDecision(
                    action=Action.HOLD,
                    risk_pct=0.0,
                    stop_loss=0.0,
                    take_profit=0.0,
                    reasoning="Waiting for pending entry order to fill.",
                ),
                outcome=ExecutionOutcome.SKIPPED_PENDING_ENTRY,
            )
            self.flush_display()
            return

        position = self._position_for_llm(close_price)
        if position.side == PositionSide.FLAT:
            self._bars_in_trade = 0
        else:
            self._bars_in_trade += 1
            position = position.model_copy(
                update={"bars_in_trade": self._bars_in_trade},
            )
        account = self.broker.get_account(mark_price=close_price)
        account = self._drawdown_tracker.enrich_account(account)

        decision = self.advisor.decide(market, position, account)
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

        position = self.broker.get_position(mark_price=close_price)
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
        self._wait_for_order_notify = self._should_wait_for_order(
            decision,
            outcome,
        )

    def on_order_settled(self) -> None:
        if self._pending_panel is None:
            return

        rejected = getattr(self.broker, "consume_entry_rejected", lambda: False)()
        getattr(self.broker, "consume_entry_settled", lambda: False)()

        mark = self._pending_panel["close_price"]
        decision = self._pending_panel["decision"]
        outcome = self._pending_panel.get("outcome")

        if rejected:
            self._pending_panel["outcome"] = ExecutionOutcome.SKIPPED_ORDER_REJECTED
        elif outcome == ExecutionOutcome.CLOSED:
            self._pending_panel["outcome"] = ExecutionOutcome.CLOSED
        else:
            position = self.broker.get_position(mark_price=mark)
            if position.side != PositionSide.FLAT:
                self._pending_panel["outcome"] = ExecutionOutcome.EXECUTED
            elif self.broker.has_pending_entry():
                self._pending_panel["outcome"] = ExecutionOutcome.ORDER_SUBMITTED

        self._pending_panel["position"] = self.broker.get_position(mark_price=mark)
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
            position = self.broker.get_position(mark_price=close_price)
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

    def _position_for_llm(self, mark_price: float):
        position = self.broker.get_position(mark_price=mark_price)
        if position.side != PositionSide.FLAT:
            return position
        pending = self.broker.has_pending_entry()
        if pending == position.pending_entry:
            return position
        return position.model_copy(update={"pending_entry": pending})

    def _check_stops(self, candle: Candle, close_price: float) -> StopHit | None:
        position = self.broker.get_position(mark_price=close_price)
        if position.side == PositionSide.FLAT:
            return None

        hit = check_stop_hit(position, candle)
        if hit is None:
            return None

        logger.debug("Closing position — %s", hit.reason)
        self.broker.close_position_at_price(hit.fill_price)
        self._bars_in_trade = 0
        return hit

    @staticmethod
    def _should_wait_for_order(
        decision: LLMDecision,
        outcome: ExecutionOutcome,
    ) -> bool:
        if outcome in (
            ExecutionOutcome.SKIPPED_INVALID_LEVELS,
            ExecutionOutcome.SKIPPED_ZERO_SIZE,
            ExecutionOutcome.SKIPPED_PENDING_ENTRY,
            ExecutionOutcome.SKIPPED_ORDER_REJECTED,
        ):
            return False

        if decision.action == Action.CLOSE and outcome == ExecutionOutcome.CLOSED:
            return True

        return decision.action in (
            Action.ENTER_LONG,
            Action.ENTER_SHORT,
        )
