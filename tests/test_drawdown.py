import pytest

from llm_trading_bot.data.serialize import state_to_prompt
from llm_trading_bot.trading.drawdown import DrawdownTracker
from llm_trading_bot.trading.models import AccountState, PositionState


def _account(equity: float, **extra) -> AccountState:
    return AccountState(
        balance=equity,
        equity=equity,
        available_cash=equity,
        currency="USDT",
        **extra,
    )


class TestDrawdownTracker:
    def test_tracks_peak_and_current_drawdown(self):
        tracker = DrawdownTracker()
        assert tracker.update(10_000.0) == (10_000.0, 0.0)
        assert tracker.update(10_500.0) == (10_500.0, 0.0)
        assert tracker.update(9_450.0) == (10_500.0, pytest.approx(10.0))

        enriched = tracker.enrich_account(_account(9_450.0))
        assert enriched.peak_equity == 10_500.0
        assert enriched.drawdown_pct == pytest.approx(10.0)


@pytest.mark.parametrize("include_drawdown", [False, True])
def test_state_to_prompt_drawdown(include_drawdown: bool):
    account = _account(9_000.0, peak_equity=10_000.0, drawdown_pct=10.0)
    payload = state_to_prompt(PositionState(), account, include_drawdown=include_drawdown)
    if include_drawdown:
        assert payload["account"]["peak_equity"] == 10_000.0
        assert payload["account"]["drawdown_pct"] == 10.0
    else:
        assert "peak_equity" not in payload["account"]
        assert "drawdown_pct" not in payload["account"]
