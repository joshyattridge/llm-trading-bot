import backtrader as bt

from llm_trading_bot.trading.executor import BrokerAdapter
from llm_trading_bot.trading.models import AccountState, PositionSide, PositionState


class BacktraderBrokerAdapter(BrokerAdapter):
    """Maps backtrader's broker/position API to our adapter."""

    def __init__(self, strategy: bt.Strategy, stake_currency: str = "USDT"):
        self._strategy = strategy
        self._currency = stake_currency
        self._stop_loss: float | None = None
        self._take_profit: float | None = None
        self._entry_rejected = False
        self._entry_settled = False

    def _broker(self) -> bt.brokers.BackBroker:
        return self._strategy.broker

    def on_order(self, order: bt.Order) -> None:
        if order.status in (order.Completed, order.Margin, order.Rejected, order.Canceled):
            self._entry_settled = True
            if order.status in (order.Margin, order.Rejected, order.Canceled):
                if not self._strategy.position.size:
                    self._entry_rejected = True
                    self._stop_loss = None
                    self._take_profit = None

    def consume_entry_settled(self) -> bool:
        settled = self._entry_settled
        self._entry_settled = False
        return settled

    def consume_entry_rejected(self) -> bool:
        rejected = self._entry_rejected
        self._entry_rejected = False
        return rejected

    def has_pending_entry(self) -> bool:
        if self._strategy.position.size:
            return False
        return any(order.alive() for order in self._broker().orders)

    def get_position(self) -> PositionState:
        pos = self._strategy.position
        if not pos.size:
            return PositionState(pending_entry=self.has_pending_entry())

        side = PositionSide.LONG if pos.size > 0 else PositionSide.SHORT
        price = self._strategy.data.close[0]
        entry = pos.price
        if side == PositionSide.LONG:
            upnl = (price - entry) * abs(pos.size)
        else:
            upnl = (entry - price) * abs(pos.size)

        return PositionState(
            side=side,
            size=abs(pos.size),
            entry_price=entry,
            unrealized_pnl=upnl,
            stop_loss=self._stop_loss,
            take_profit=self._take_profit,
        )

    def get_account(self, mark_price: float) -> AccountState:
        broker = self._broker()
        cash = broker.getcash()
        equity = broker.getvalue()
        return AccountState(
            balance=cash,
            equity=equity,
            available_cash=cash,
            currency=self._currency,
        )

    def close_position(self) -> None:
        self._strategy.close()
        self._stop_loss = None
        self._take_profit = None

    def enter_long(
        self,
        size: float,
        price: float,
        *,
        stop_loss: float,
        take_profit: float,
    ) -> bool:
        if price <= 0 or size <= 0:
            return False
        if self.has_pending_entry():
            return False
        self._entry_rejected = False
        self._entry_settled = False
        self._stop_loss = stop_loss
        self._take_profit = take_profit
        self._strategy.buy(size=size)
        if self._entry_rejected:
            return False
        return True

    def enter_short(
        self,
        size: float,
        price: float,
        *,
        stop_loss: float,
        take_profit: float,
    ) -> bool:
        if price <= 0 or size <= 0:
            return False
        if self.has_pending_entry():
            return False
        self._entry_rejected = False
        self._entry_settled = False
        self._stop_loss = stop_loss
        self._take_profit = take_profit
        self._strategy.sell(size=size)
        if self._entry_rejected:
            return False
        return True

    def update_stops(
        self,
        *,
        stop_loss: float,
        take_profit: float,
    ) -> None:
        self._stop_loss = stop_loss
        self._take_profit = take_profit
