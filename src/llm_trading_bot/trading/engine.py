import logging
from typing import TYPE_CHECKING

from llm_trading_bot.llm.client import LLMTradingAdvisor
from llm_trading_bot.trading.executor import BrokerAdapter, execute_decision
from llm_trading_bot.trading.models import Candle, PositionSide

if TYPE_CHECKING:
    from llm_trading_bot.display import TerminalDisplay

logger = logging.getLogger(__name__)


class TradingEngine:
    """On each new closed candle: consult the LLM and execute."""

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
        close_price: float,
        *,
        bar: int | None = None,
        total_bars: int | None = None,
    ) -> None:
        if not history:
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
