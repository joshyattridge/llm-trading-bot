import logging
from typing import Any

import ccxt

from llm_trading_bot.trading.executor import BrokerAdapter
from llm_trading_bot.trading.models import AccountState, PositionSide, PositionState

logger = logging.getLogger(__name__)


class CcxtBrokerAdapter(BrokerAdapter):
    """
    Live / paper broker via ccxt.
    Paper mode uses the exchange sandbox when supported (set CCXT_SANDBOX=true).
    """

    def __init__(
        self,
        exchange: ccxt.Exchange,
        symbol: str,
        quote_currency: str = "USDT",
    ):
        self._exchange = exchange
        self._symbol = symbol
        self._quote = quote_currency
        self._position: PositionState = PositionState()
        self._stop_order_id: str | None = None
        self._tp_order_id: str | None = None

    def refresh_position(self, mark_price: float) -> None:
        try:
            balance = self._exchange.fetch_balance()
            base = self._symbol.split("/")[0]
            base_free = float(balance.get(base, {}).get("free", 0) or 0)
            base_used = float(balance.get(base, {}).get("used", 0) or 0)
            size = base_free + base_used

            if size < 1e-8:
                self._position = PositionState()
                return

            entry = mark_price
            side = PositionSide.LONG
            upnl = (mark_price - entry) * size
            self._position = PositionState(
                side=side,
                size=size,
                entry_price=entry,
                unrealized_pnl=upnl,
                stop_loss=self._position.stop_loss,
                take_profit=self._position.take_profit,
            )
        except Exception as e:
            logger.warning("Could not refresh position: %s", e)

    def get_position(self) -> PositionState:
        return self._position

    def get_account(self, mark_price: float) -> AccountState:
        self.refresh_position(mark_price)
        raw = self._exchange.fetch_balance()
        base = self._symbol.split("/")[0]
        quote_free = float(raw.get(self._quote, {}).get("free", 0) or 0)
        quote_total = float(raw.get(self._quote, {}).get("total", 0) or 0)
        base_total = float(raw.get(base, {}).get("total", 0) or 0)
        equity = quote_total + base_total * mark_price
        return AccountState(
            balance=quote_total,
            equity=equity,
            available_cash=quote_free,
            currency=self._quote,
        )

    def _cancel_protective_orders(self) -> None:
        for order_id in (self._stop_order_id, self._tp_order_id):
            if not order_id:
                continue
            try:
                self._exchange.cancel_order(order_id, self._symbol)
            except Exception as e:
                logger.debug("Could not cancel order %s: %s", order_id, e)
        self._stop_order_id = None
        self._tp_order_id = None

    def _place_protective_orders(
        self,
        side: PositionSide,
        amount: float,
        stop_loss: float,
        take_profit: float,
    ) -> None:
        exit_side = "sell" if side == PositionSide.LONG else "buy"
        try:
            tp_order = self._exchange.create_order(
                self._symbol,
                "limit",
                exit_side,
                amount,
                take_profit,
            )
            self._tp_order_id = tp_order.get("id")
        except Exception as e:
            logger.warning("Could not place take-profit order: %s", e)

        try:
            params: dict[str, Any] = {"stopPrice": stop_loss}
            stop_order = self._exchange.create_order(
                self._symbol,
                "stop_market",
                exit_side,
                amount,
                None,
                params,
            )
            self._stop_order_id = stop_order.get("id")
        except Exception as e:
            logger.warning(
                "Could not place stop-loss order (soft stops still active): %s", e
            )

    def close_position(self) -> None:
        pos = self._position
        if pos.side == PositionSide.FLAT:
            return
        self._cancel_protective_orders()
        side = "sell" if pos.side == PositionSide.LONG else "buy"
        self._exchange.create_market_order(self._symbol, side, pos.size)
        self._position = PositionState()

    def enter_long(
        self,
        size: float,
        price: float,
        *,
        stop_loss: float,
        take_profit: float,
    ) -> None:
        self._exchange.create_market_order(self._symbol, "buy", size)
        self._position = PositionState(
            side=PositionSide.LONG,
            size=size,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        self._place_protective_orders(
            PositionSide.LONG, size, stop_loss, take_profit
        )

    def enter_short(
        self,
        size: float,
        price: float,
        *,
        stop_loss: float,
        take_profit: float,
    ) -> None:
        if not self._exchange.has.get("createOrder", False):
            raise RuntimeError("Exchange does not support orders")
        self._exchange.create_market_order(self._symbol, "sell", size)
        self._position = PositionState(
            side=PositionSide.SHORT,
            size=size,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        self._place_protective_orders(
            PositionSide.SHORT, size, stop_loss, take_profit
        )


def create_exchange(settings: Any, *, sandbox: bool | None = None) -> ccxt.Exchange:
    exchange_class = getattr(ccxt, settings.exchange_id)
    config: dict = {
        "apiKey": settings.ccxt_api_key,
        "secret": settings.ccxt_api_secret,
        "enableRateLimit": True,
    }
    exchange = exchange_class(config)
    use_sandbox = settings.ccxt_sandbox if sandbox is None else sandbox
    if use_sandbox and hasattr(exchange, "set_sandbox_mode"):
        exchange.set_sandbox_mode(True)
    return exchange
