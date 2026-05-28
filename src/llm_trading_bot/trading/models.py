from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Action(str, Enum):
    HOLD = "hold"
    CLOSE = "close"
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    ADJUST_STOPS = "adjust_stops"


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


class ExecutionOutcome(str, Enum):
    NOOP = "noop"
    CLOSED = "closed"
    STOPS_ADJUSTED = "stops_adjusted"
    ORDER_SUBMITTED = "order_submitted"
    EXECUTED = "executed"
    SKIPPED_INVALID_LEVELS = "skipped_invalid_levels"
    SKIPPED_ZERO_SIZE = "skipped_zero_size"
    SKIPPED_PENDING_ENTRY = "skipped_pending_entry"
    SKIPPED_ORDER_REJECTED = "skipped_order_rejected"


class PositionState(BaseModel):
    side: PositionSide = PositionSide.FLAT
    size: float = 0.0
    entry_price: float | None = None
    unrealized_pnl: float = 0.0
    bars_in_trade: int = 0
    stop_loss: float | None = None
    take_profit: float | None = None
    pending_entry: bool = False


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
        description="Stop-loss price for entries or adjust_stops (0 otherwise).",
    )
    take_profit: float = Field(
        ge=0.0,
        description="Take-profit price for entries or adjust_stops (0 otherwise).",
    )
    reasoning: str = ""


class LLMDecisionResponse(BaseModel):
    """Structured response schema sent to the model."""

    action: Literal["hold", "close", "enter_long", "enter_short", "adjust_stops"]
    risk_pct: float = Field(ge=0.0, le=1.0)
    stop_loss: float = Field(ge=0.0)
    take_profit: float = Field(ge=0.0)
    reasoning: str
