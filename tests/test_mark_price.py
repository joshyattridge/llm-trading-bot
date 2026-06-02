"""Tests for consistent mark-to-market pricing."""

from __future__ import annotations

import pytest

from llm_trading_bot.trading.models import PositionSide
from tests.mock_broker import MockBroker


class TestMarkPriceConsistency:
    def test_get_position_upnl_matches_explicit_mark(self):
        broker = MockBroker()
        broker.set_position(
            side=PositionSide.LONG,
            size=2.0,
            entry_price=100.0,
        )

        mark = 105.5
        position = broker.get_position(mark_price=mark)

        assert position.unrealized_pnl == pytest.approx((mark - 100.0) * 2.0)

    def test_account_equity_includes_upnl_at_mark(self):
        broker = MockBroker(cash=5_000.0)
        broker.set_position(
            side=PositionSide.LONG,
            size=1.0,
            entry_price=100.0,
        )

        mark = 110.0
        account = broker.get_account(mark_price=mark)

        assert account.equity == pytest.approx(5_000.0 + 10.0)

    def test_short_upnl_at_mark(self):
        broker = MockBroker()
        broker.set_position(
            side=PositionSide.SHORT,
            size=1.0,
            entry_price=100.0,
        )

        position = broker.get_position(mark_price=95.0)

        assert position.unrealized_pnl == pytest.approx(5.0)
