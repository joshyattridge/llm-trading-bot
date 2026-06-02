"""In-memory broker for unit tests."""

from __future__ import annotations

from llm_trading_bot.trading.executor import BrokerAdapter
from llm_trading_bot.trading.models import (
    AccountState,
    PositionSide,
    PositionState,
)


class MockBroker(BrokerAdapter):
    """In-memory broker for unit tests."""

    def __init__(
        self,
        *,
        cash: float = 10_000.0,
        equity: float | None = None,
    ) -> None:
        self._position = PositionState()
        self._cash = cash
        self._equity = equity if equity is not None else cash
        self._pending_entry = False
        self.close_calls: list[float | None] = []
        self._fill_on_close = True

    def set_position(
        self,
        *,
        side: PositionSide,
        size: float,
        entry_price: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> None:
        self._position = PositionState(
            side=side,
            size=size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def get_position(self, mark_price: float | None = None) -> PositionState:
        if self._position.side == PositionSide.FLAT:
            return PositionState(pending_entry=self._pending_entry)

        entry = self._position.entry_price or 0.0
        price = mark_price if mark_price is not None else entry
        if self._position.side == PositionSide.LONG:
            upnl = (price - entry) * self._position.size
        else:
            upnl = (entry - price) * self._position.size

        return self._position.model_copy(update={"unrealized_pnl": upnl})

    def get_account(self, mark_price: float) -> AccountState:
        position = self.get_position(mark_price=mark_price)
        equity = self._cash + position.unrealized_pnl
        return AccountState(
            balance=self._cash,
            equity=equity,
            available_cash=self._cash,
            currency="USDT",
        )

    def close_position(self) -> None:
        self.close_calls.append(None)
        if self._fill_on_close:
            self._apply_close()

    def close_position_at_price(self, price: float) -> None:
        self.close_calls.append(price)
        if self._fill_on_close:
            self._apply_close()

    def _apply_close(self) -> None:
        position = self.get_position()
        if position.side == PositionSide.FLAT:
            return
        entry = position.entry_price or 0.0
        fill = self.close_calls[-1] if self.close_calls else None
        exit_price = fill if fill is not None else entry
        if position.side == PositionSide.LONG:
            self._cash += position.size * exit_price
        else:
            self._cash -= position.size * exit_price
        self._position = PositionState()

    def enter_long(
        self,
        size: float,
        price: float,
        *,
        stop_loss: float,
        take_profit: float,
    ) -> bool:
        self._position = PositionState(
            side=PositionSide.LONG,
            size=size,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        self._cash -= size * price
        return True

    def enter_short(
        self,
        size: float,
        price: float,
        *,
        stop_loss: float,
        take_profit: float,
    ) -> bool:
        self._position = PositionState(
            side=PositionSide.SHORT,
            size=size,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        self._cash += size * price
        return True

    def update_stops(
        self,
        *,
        stop_loss: float,
        take_profit: float,
    ) -> None:
        self._position = self._position.model_copy(
            update={"stop_loss": stop_loss, "take_profit": take_profit},
        )

    def has_pending_entry(self) -> bool:
        return self._pending_entry
