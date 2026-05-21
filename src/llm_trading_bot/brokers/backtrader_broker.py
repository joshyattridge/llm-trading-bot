import backtrader as bt

from llm_trading_bot.trading.executor import BrokerAdapter
from llm_trading_bot.trading.models import AccountState, PositionSide, PositionState


class BacktraderBrokerAdapter(BrokerAdapter):
    """Maps backtrader's broker/position API to our adapter."""

    def __init__(self, strategy: bt.Strategy, stake_currency: str = "USDT"):
        self._strategy = strategy
        self._currency = stake_currency

    def _broker(self) -> bt.brokers.BackBroker:
        return self._strategy.broker

    def get_position(self) -> PositionState:
        pos = self._strategy.position
        if not pos.size:
            return PositionState()

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

    def enter_long(self, stake_cash: float, price: float) -> None:
        if price <= 0:
            return
        size = stake_cash / price
        self._strategy.buy(size=size)

    def enter_short(self, stake_cash: float, price: float) -> None:
        if price <= 0:
            return
        size = stake_cash / price
        self._strategy.sell(size=size)
