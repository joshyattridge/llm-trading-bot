import logging
import time

from llm_trading_bot.brokers.ccxt_broker import CcxtBrokerAdapter, create_exchange
from llm_trading_bot.config import Settings
from llm_trading_bot.display import TerminalDisplay
from llm_trading_bot.llm.client import LLMTradingAdvisor
from llm_trading_bot.trading.engine import TradingEngine
from llm_trading_bot.trading.models import Candle

logger = logging.getLogger(__name__)


def _ohlcv_to_candles(rows: list[list]) -> list[Candle]:
    """ccxt OHLCV rows include ms timestamps — never pass those to the LLM."""
    candles = []
    for row in rows:
        o, h, l, c, v = row[1], row[2], row[3], row[4], row[5]
        candles.append(Candle(open=o, high=h, low=l, close=c, volume=v))
    return candles


def run_live_loop(
    settings: Settings,
    paper: bool = False,
    display: TerminalDisplay | None = None,
) -> None:
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required")

    exchange = create_exchange(settings)
    if paper and hasattr(exchange, "set_sandbox_mode"):
        exchange.set_sandbox_mode(True)

    broker = CcxtBrokerAdapter(exchange, settings.symbol)
    advisor = LLMTradingAdvisor(settings)
    display = display or TerminalDisplay()
    engine = TradingEngine(
        advisor,
        broker,
        display,
        symbol=settings.symbol,
        timeframe=settings.timeframe,
    )

    last_ts: int | None = None
    mode = "paper" if paper or settings.ccxt_sandbox else "live"
    display.print_live_header(settings, mode)

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(
                settings.symbol,
                settings.timeframe,
                limit=settings.candle_history + 2,
            )
            closed = ohlcv[:-1]
            if not closed:
                time.sleep(10)
                continue

            latest_ts = closed[-1][0]
            if last_ts is not None and latest_ts <= last_ts:
                time.sleep(10)
                continue

            last_ts = latest_ts
            history = _ohlcv_to_candles(closed[-settings.candle_history :])
            close_price = history[-1].close
            bar = display.next_live_candle()
            engine.on_new_candle(history, close_price, bar=bar)
        except KeyboardInterrupt:
            logger.info("Stopped by user")
            break
        except Exception as e:
            logger.exception("Loop error: %s", e)
            time.sleep(30)
