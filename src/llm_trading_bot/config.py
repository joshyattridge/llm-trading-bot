from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STYLE_PROMPT_PATH = PROJECT_ROOT / "prompts" / "trading_style.txt"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")

    exchange_id: str = Field(default="binance", validation_alias="EXCHANGE_ID")
    symbol: str = Field(default="BTC/USDT", validation_alias="SYMBOL")
    timeframe: str = Field(default="1h", validation_alias="TIMEFRAME")
    candle_history: int = Field(default=50, validation_alias="CANDLE_HISTORY")

    ccxt_api_key: str = Field(default="", validation_alias="CCXT_API_KEY")
    ccxt_api_secret: str = Field(default="", validation_alias="CCXT_API_SECRET")
    ccxt_sandbox: bool = Field(default=True, validation_alias="CCXT_SANDBOX")

    starting_balance: float = Field(default=10_000.0, validation_alias="STARTING_BALANCE")

    trading_style_prompt: str = Field(
        default="",
        validation_alias="TRADING_STYLE_PROMPT",
        description="Inline trading style text; overrides TRADING_STYLE_PROMPT_PATH when set.",
    )
    trading_style_prompt_path: Path = Field(
        default=DEFAULT_STYLE_PROMPT_PATH,
        validation_alias="TRADING_STYLE_PROMPT_PATH",
    )

    def resolve_trading_style_prompt_path(self) -> Path:
        path = self.trading_style_prompt_path
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    def load_trading_style_prompt(self) -> str:
        if self.trading_style_prompt.strip():
            return self.trading_style_prompt.strip()

        path = self.resolve_trading_style_prompt_path()
        if path.exists():
            return path.read_text(encoding="utf-8").strip()

        return (
            "Trade conservatively. Prefer holding in unclear markets. "
            "On entries, set risk_pct and absolute stop_loss / take_profit levels."
        )


def get_settings() -> Settings:
    return Settings()
