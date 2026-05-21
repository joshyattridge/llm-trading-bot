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
    def enter_long(self, stake_cash: float, price: float) -> None:
        ...

    @abstractmethod
    def enter_short(self, stake_cash: float, price: float) -> None:
        ...


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

    stake_cash = account.available_cash * decision.stake_pct
    if stake_cash <= 0:
        logger.warning("Stake cash is zero; skipping entry")
        return

    if decision.action == Action.ENTER_LONG:
        logger.debug("ENTER LONG (%.1f%%) — %s", decision.stake_pct * 100, decision.reasoning)
        broker.enter_long(stake_cash, price)
    elif decision.action == Action.ENTER_SHORT:
        logger.debug("ENTER SHORT (%.1f%%) — %s", decision.stake_pct * 100, decision.reasoning)
        broker.enter_short(stake_cash, price)
