import logging
import time

from llm_trading_bot.brokers.ccxt_broker import CcxtBrokerAdapter, create_exchange
from llm_trading_bot.config import Settings
from llm_trading_bot.data.market import MultiTimeframeMarket, TimeframeSeries
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


def _fetch_closed_candles(
    exchange,
    symbol: str,
    timeframe: str,
    history_len: int,
) -> list[Candle]:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=history_len + 2)
    closed = ohlcv[:-1]
    return _ohlcv_to_candles(closed[-history_len:])


def _build_market(settings: Settings, exchange) -> MultiTimeframeMarket:
    lower_candles = _fetch_closed_candles(
        exchange,
        settings.symbol,
        settings.timeframe,
        settings.candle_history,
    )
    lower = TimeframeSeries(timeframe=settings.timeframe, candles=lower_candles)
    higher: TimeframeSeries | None = None

    if settings.uses_higher_timeframe():
        htf = settings.higher_timeframe.strip()
        higher_candles = _fetch_closed_candles(
            exchange,
            settings.symbol,
            htf,
            settings.candle_history,
        )
        higher = TimeframeSeries(timeframe=htf, candles=higher_candles)

    return MultiTimeframeMarket(lower=lower, higher=higher)


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
        commission_rate=settings.commission_rate,
        leverage=settings.leverage,
        trade_history_limit=settings.trade_history_limit,
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
            market = _build_market(settings, exchange)
            bar = display.next_live_candle()
            engine.on_new_candle(market, market.lower.candles[-1], bar=bar)
        except KeyboardInterrupt:
            logger.info("Stopped by user")
            break
        except Exception as e:
            logger.exception("Loop error: %s", e)
            time.sleep(30)
