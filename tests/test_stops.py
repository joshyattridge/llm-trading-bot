"""Tests for stop-loss / take-profit detection."""

from llm_trading_bot.trading.models import Candle, PositionSide, PositionState
from llm_trading_bot.trading.stops import backtrader_candle_indices, check_stop_hit


def _candle(
    *,
    open_: float = 100.0,
    high: float = 105.0,
    low: float = 95.0,
    close: float = 102.0,
) -> Candle:
    return Candle(open=open_, high=high, low=low, close=close, volume=1.0)


def _long(sl: float, tp: float, entry: float = 100.0) -> PositionState:
    return PositionState(
        side=PositionSide.LONG,
        size=1.0,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
    )


def _short(sl: float, tp: float, entry: float = 100.0) -> PositionState:
    return PositionState(
        side=PositionSide.SHORT,
        size=1.0,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
    )


class TestBacktraderCandleIndices:
    def test_includes_current_bar_zero(self):
        assert list(backtrader_candle_indices(50)) == list(range(-49, 1))

    def test_single_bar_window(self):
        assert list(backtrader_candle_indices(1)) == [0]

    def test_rejects_zero_history(self):
        try:
            backtrader_candle_indices(0)
            assert False, "expected ValueError"
        except ValueError:
            pass


class TestCheckStopHit:
    def test_long_stop_loss_hit(self):
        hit = check_stop_hit(_long(sl=98.0, tp=110.0), _candle(low=97.5, high=101.0))
        assert hit is not None
        assert hit.fill_price == 98.0
        assert "stop loss hit at 98.00" in hit.reason

    def test_long_take_profit_hit(self):
        hit = check_stop_hit(_long(sl=90.0, tp=108.0), _candle(low=99.0, high=108.5))
        assert hit is not None
        assert hit.fill_price == 108.0
        assert "take profit hit at 108.00" in hit.reason

    def test_long_stop_takes_priority_over_tp_same_bar(self):
        hit = check_stop_hit(
            _long(sl=98.0, tp=108.0),
            _candle(low=97.0, high=109.0),
        )
        assert hit is not None
        assert hit.fill_price == 98.0

    def test_short_stop_loss_hit(self):
        hit = check_stop_hit(_short(sl=102.0, tp=90.0), _candle(low=99.0, high=102.5))
        assert hit is not None
        assert hit.fill_price == 102.0

    def test_short_take_profit_hit(self):
        hit = check_stop_hit(_short(sl=110.0, tp=92.0), _candle(low=91.5, high=100.0))
        assert hit is not None
        assert hit.fill_price == 92.0

    def test_flat_position_never_hits(self):
        flat = PositionState()
        assert check_stop_hit(flat, _candle()) is None

    def test_no_stops_configured(self):
        pos = PositionState(
            side=PositionSide.LONG,
            size=1.0,
            entry_price=100.0,
        )
        assert check_stop_hit(pos, _candle(low=50.0)) is None

    def test_no_hit_when_price_stays_in_range(self):
        assert check_stop_hit(_long(sl=95.0, tp=105.0), _candle(low=96.0, high=104.0)) is None
