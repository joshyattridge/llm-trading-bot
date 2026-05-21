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
    stop_loss: float | None = None
    take_profit: float | None = None


class AccountState(BaseModel):
    """Quote-currency account snapshot."""

    balance: float  # quote wallet (cash leg)
    equity: float  # total portfolio value at mark (cash + positions)
    available_cash: float  # quote free to deploy on new entries
    currency: str = "USDT"


class LLMDecision(BaseModel):
    action: Action
    risk_pct: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of equity to risk on a new entry (ignored for hold/close).",
    )
    stop_loss: float = Field(
        ge=0.0,
        description="Stop-loss price for a new entry (0 when not entering).",
    )
    take_profit: float = Field(
        ge=0.0,
        description="Take-profit price for a new entry (0 when not entering).",
    )
    reasoning: str = ""


class LLMDecisionResponse(BaseModel):
    """Structured response schema sent to the model."""

    action: Literal["hold", "close", "enter_long", "enter_short"]
    risk_pct: float = Field(ge=0.0, le=1.0)
    stop_loss: float = Field(ge=0.0)
    take_profit: float = Field(ge=0.0)
    reasoning: str
