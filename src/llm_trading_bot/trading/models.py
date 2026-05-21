from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Action(str, Enum):
    HOLD = "hold"
    CLOSE = "close"
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"


class PositionSide(str, Enum):
    FLAT = "flat"
    LONG = "long"
    SHORT = "short"


class Candle(BaseModel):
    """OHLCV only — no timestamp to avoid look-ahead bias in the LLM."""

    open: float
    high: float
    low: float
    close: float
    volume: float

    def as_list(self) -> list[float]:
        return [self.open, self.high, self.low, self.close, self.volume]


class PositionState(BaseModel):
    side: PositionSide = PositionSide.FLAT
    size: float = 0.0
    entry_price: float | None = None
    unrealized_pnl: float = 0.0
    bars_in_trade: int = 0


class AccountState(BaseModel):
    """Quote-currency account snapshot."""

    balance: float  # quote wallet (cash leg)
    equity: float  # total portfolio value at mark (cash + positions)
    available_cash: float  # quote free to deploy on new entries
    currency: str = "USDT"


class LLMDecision(BaseModel):
    action: Action
    stake_pct: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of available balance to allocate (new entries only).",
    )
    reasoning: str = ""


class LLMDecisionResponse(BaseModel):
    """Structured response schema sent to the model."""

    action: Literal["hold", "close", "enter_long", "enter_short"]
    stake_pct: float = Field(ge=0.0, le=1.0)
    reasoning: str
