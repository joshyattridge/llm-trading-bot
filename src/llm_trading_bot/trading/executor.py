import logging
from abc import ABC, abstractmethod

from llm_trading_bot.trading.models import (
    AccountState,
    Action,
    LLMDecision,
    PositionSide,
    PositionState,
)

logger = logging.getLogger(__name__)


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
    ) -> None:
        ...

    @abstractmethod
    def enter_short(
        self,
        size: float,
        price: float,
        *,
        stop_loss: float,
        take_profit: float,
    ) -> None:
        ...


def calculate_position_size(
    account: AccountState,
    risk_pct: float,
    entry_price: float,
    stop_loss: float,
) -> float:
    """Size so that hitting stop_loss risks risk_pct of equity."""
    risk_amount = account.equity * risk_pct
    risk_per_unit = abs(entry_price - stop_loss)
    if risk_per_unit <= 0 or entry_price <= 0:
        return 0.0
    size = risk_amount / risk_per_unit
    max_size = account.available_cash / entry_price
    return min(size, max_size)


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


def execute_decision(
    broker: BrokerAdapter,
    decision: LLMDecision,
    price: float,
) -> None:
    position = broker.get_position()
    account = broker.get_account(mark_price=price)

    if decision.action == Action.HOLD:
        logger.debug("HOLD — %s", decision.reasoning)
        return

    if decision.action == Action.CLOSE:
        if position.side != PositionSide.FLAT:
            logger.debug("CLOSE — %s", decision.reasoning)
            broker.close_position()
        return

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
        return

    size = calculate_position_size(
        account, decision.risk_pct, price, decision.stop_loss
    )
    if size <= 0:
        logger.warning("Position size is zero; skipping entry")
        return

    if decision.action == Action.ENTER_LONG:
        logger.debug(
            "ENTER LONG size=%.6f risk=%.1f%% sl=%.2f tp=%.2f — %s",
            size,
            decision.risk_pct * 100,
            decision.stop_loss,
            decision.take_profit,
            decision.reasoning,
        )
        broker.enter_long(
            size,
            price,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
        )
    elif decision.action == Action.ENTER_SHORT:
        logger.debug(
            "ENTER SHORT size=%.6f risk=%.1f%% sl=%.2f tp=%.2f — %s",
            size,
            decision.risk_pct * 100,
            decision.stop_loss,
            decision.take_profit,
            decision.reasoning,
        )
        broker.enter_short(
            size,
            price,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
        )
