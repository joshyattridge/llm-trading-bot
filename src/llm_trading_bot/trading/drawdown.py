from __future__ import annotations

from llm_trading_bot.trading.models import AccountState


class DrawdownTracker:
    """Track peak equity and current drawdown from that peak."""

    def __init__(self) -> None:
        self._peak_equity: float | None = None

    @property
    def peak_equity(self) -> float | None:
        return self._peak_equity

    def update(self, equity: float) -> tuple[float, float]:
        """Return (peak_equity, drawdown_pct) where drawdown_pct is 0–100."""
        if self._peak_equity is None:
            self._peak_equity = equity
        else:
            self._peak_equity = max(self._peak_equity, equity)

        if self._peak_equity <= 0:
            return self._peak_equity, 0.0

        drawdown_pct = (self._peak_equity - equity) / self._peak_equity * 100.0
        return self._peak_equity, max(drawdown_pct, 0.0)

    def enrich_account(self, account: AccountState) -> AccountState:
        peak, drawdown_pct = self.update(account.equity)
        return account.model_copy(
            update={
                "peak_equity": peak,
                "drawdown_pct": drawdown_pct,
            },
        )
