import logging
from abc import ABC, abstractmethod

from llm_trading_bot.trading.models import (
    AccountState,
    Action,
    ExecutionOutcome,
    LLMDecision,
    PositionSide,
    PositionState,
)

logger = logging.getLogger(__name__)

# Small haircut so backtrader's margin check passes after commission rounding.
_CASH_BUFFER = 0.002


class BrokerAdapter(ABC):
    @abstractmethod
    def get_position(self) -> PositionState:
        ...

    @abstractmethod
    def get_account(self, mark_price: float) -> AccountState:
        ...

    @abstractmethod
    def close_position(self) -> None:
        ...

    @abstractmethod
    def enter_long(
        self,
        size: float,
        price: float,
        *,
        stop_loss: float,
        take_profit: float,
    ) -> bool:
        ...

    @abstractmethod
    def enter_short(
        self,
        size: float,
        price: float,
        *,
        stop_loss: float,
        take_profit: float,
    ) -> bool:
        ...

    @abstractmethod
    def update_stops(
        self,
        *,
        stop_loss: float,
        take_profit: float,
    ) -> None:
        ...

    def has_pending_entry(self) -> bool:
        return False


def calculate_position_size(
    account: AccountState,
    risk_pct: float,
    entry_price: float,
    stop_loss: float,
    *,
    commission_rate: float = 0.0,
    leverage: float = 1.0,
) -> float:
    """Size so that hitting stop_loss risks risk_pct of equity."""
    risk_amount = account.equity * risk_pct
    risk_per_unit = abs(entry_price - stop_loss)
    if risk_per_unit <= 0 or entry_price <= 0:
        return 0.0
    size = risk_amount / risk_per_unit
    cost_per_unit = entry_price * (1.0 + commission_rate)
    if cost_per_unit <= 0 or leverage <= 0:
        return 0.0
    buying_power = account.available_cash * leverage
    max_size = buying_power / cost_per_unit
    max_size *= 1.0 - _CASH_BUFFER
    return min(size, max_size)


def sizing_price_for_entry(side: PositionSide, candle_close: float, candle_high: float, candle_low: float) -> float:
    """Conservative fill estimate so margin checks survive intra-bar moves."""
    if side == PositionSide.LONG:
        return max(candle_close, candle_high)
    if side == PositionSide.SHORT:
        return min(candle_close, candle_low)
    return candle_close


def _validate_entry_levels(
    side: PositionSide,
    price: float,
    stop_loss: float,
    take_profit: float,
) -> bool:
    if stop_loss <= 0 or take_profit <= 0 or price <= 0:
        return False
    if side == PositionSide.LONG:
        return stop_loss < price < take_profit
    if side == PositionSide.SHORT:
        return take_profit < price < stop_loss
    return False


def _validate_stop_levels(
    side: PositionSide,
    stop_loss: float,
    take_profit: float,
) -> bool:
    if stop_loss <= 0 or take_profit <= 0:
        return False
    if side == PositionSide.LONG:
        return stop_loss < take_profit
    if side == PositionSide.SHORT:
        return take_profit < stop_loss
    return False


def execute_decision(
    broker: BrokerAdapter,
    decision: LLMDecision,
    price: float,
    *,
    commission_rate: float = 0.0,
    leverage: float = 1.0,
    sizing_price: float | None = None,
) -> ExecutionOutcome:
    position = broker.get_position()
    account = broker.get_account(mark_price=price)

    if decision.action == Action.HOLD:
        logger.debug("HOLD — %s", decision.reasoning)
        return ExecutionOutcome.NOOP

    if decision.action == Action.CLOSE:
        if position.side != PositionSide.FLAT:
            logger.debug("CLOSE — %s", decision.reasoning)
            broker.close_position()
            return ExecutionOutcome.CLOSED
        return ExecutionOutcome.NOOP

    if decision.action == Action.ADJUST_STOPS:
        if not _validate_stop_levels(
            position.side, decision.stop_loss, decision.take_profit
        ):
            logger.warning(
                "Invalid SL/TP levels for %s adjust_stops (sl=%.2f tp=%.2f); skipping",
                position.side.value,
                decision.stop_loss,
                decision.take_profit,
            )
            return ExecutionOutcome.SKIPPED_INVALID_LEVELS
        logger.debug(
            "ADJUST STOPS sl=%.2f tp=%.2f — %s",
            decision.stop_loss,
            decision.take_profit,
            decision.reasoning,
        )
        broker.update_stops(
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
        )
        return ExecutionOutcome.STOPS_ADJUSTED

    if broker.has_pending_entry():
        logger.warning("Entry order already pending; skipping duplicate entry")
        return ExecutionOutcome.SKIPPED_PENDING_ENTRY

    side = (
        PositionSide.LONG
        if decision.action == Action.ENTER_LONG
        else PositionSide.SHORT
    )
    if not _validate_entry_levels(
        side, price, decision.stop_loss, decision.take_profit
    ):
        logger.warning(
            "Invalid SL/TP levels for %s at %.2f (sl=%.2f tp=%.2f); skipping entry",
            side.value,
            price,
            decision.stop_loss,
            decision.take_profit,
        )
        return ExecutionOutcome.SKIPPED_INVALID_LEVELS

    fill_estimate = sizing_price if sizing_price is not None else price
    size = calculate_position_size(
        account,
        decision.risk_pct,
        fill_estimate,
        decision.stop_loss,
        commission_rate=commission_rate,
        leverage=leverage,
    )
    if size <= 0:
        logger.warning("Position size is zero; skipping entry")
        return ExecutionOutcome.SKIPPED_ZERO_SIZE

    if decision.action == Action.ENTER_LONG:
        logger.debug(
            "ENTER LONG size=%.6f risk=%.1f%% sl=%.2f tp=%.2f — %s",
            size,
            decision.risk_pct * 100,
            decision.stop_loss,
            decision.take_profit,
            decision.reasoning,
        )
        submitted = broker.enter_long(
            size,
            price,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
        )
    else:
        logger.debug(
            "ENTER SHORT size=%.6f risk=%.1f%% sl=%.2f tp=%.2f — %s",
            size,
            decision.risk_pct * 100,
            decision.stop_loss,
            decision.take_profit,
            decision.reasoning,
        )
        submitted = broker.enter_short(
            size,
            price,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
        )

    if not submitted:
        return ExecutionOutcome.SKIPPED_ORDER_REJECTED
    if broker.has_pending_entry():
        return ExecutionOutcome.ORDER_SUBMITTED
    return ExecutionOutcome.EXECUTED
