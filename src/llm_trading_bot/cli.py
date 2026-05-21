import typer
from rich.console import Console

from llm_trading_bot.config import get_settings
from llm_trading_bot.data.historical import MAX_FETCH_LIMIT
from llm_trading_bot.display import TerminalDisplay, configure_logging
from llm_trading_bot.runners.backtest import (
    llm_decision_count,
    run_backtest_from_exchange,
)
from llm_trading_bot.runners.live import run_live_loop

app = typer.Typer(help="LLM trading bot — backtest, paper, and live modes.")
console = Console()


def _timeframe_duration_label(timeframe: str, candles: int) -> str:
    """Human-readable span, e.g. 50 × 1h ≈ 50 hours."""
    for suffix in ("m", "h", "d", "w"):
        if timeframe.endswith(suffix) and timeframe[:-1].isdigit():
            n = int(timeframe[:-1])
            plural = {
                "m": ("minute", "minutes"),
                "h": ("hour", "hours"),
                "d": ("day", "days"),
                "w": ("week", "weeks"),
            }[suffix]
            name = plural[0] if n == 1 else plural[1]
            return f"{candles} × {timeframe} ≈ {candles * n} {name}"
    return f"{candles} × {timeframe}"


@app.command()
def backtest(
    candles: int = typer.Option(
        ...,
        "--candles",
        "-n",
        min=1,
        max=MAX_FETCH_LIMIT,
        help="Number of candles to backtest (length follows TIMEFRAME, e.g. 50 + 1h = ~50 hours).",
    ),
    cash: float | None = typer.Option(
        None,
        "--cash",
        help="Starting cash (overrides STARTING_BALANCE from .env).",
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Backtest on historical OHLCV fetched from the exchange."""
    configure_logging(verbose)
    settings = get_settings()
    if not settings.openai_api_key:
        typer.echo("Set OPENAI_API_KEY in .env before running.", err=True)
        raise typer.Exit(1)

    decisions = llm_decision_count(candles, settings.candle_history)
    if decisions == 0:
        typer.echo(
            f"--candles ({candles}) must be greater than CANDLE_HISTORY "
            f"({settings.candle_history}) to run any LLM decisions.",
            err=True,
        )
        raise typer.Exit(1)

    initial_cash = cash if cash is not None else settings.starting_balance
    span = _timeframe_duration_label(settings.timeframe, candles)
    display = TerminalDisplay()

    result = run_backtest_from_exchange(
        settings,
        candles,
        initial_cash=initial_cash,
        display=display,
        span_label=span,
    )
    display.print_backtest_summary(result, settings)


@app.command()
def paper(
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Paper trade via ccxt sandbox on each new closed candle."""
    configure_logging(verbose)
    settings = get_settings()
    run_live_loop(settings, paper=True, display=TerminalDisplay())


@app.command()
def live(
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Live trade via ccxt (requires API keys; use with caution)."""
    configure_logging(verbose)
    settings = get_settings()
    if settings.ccxt_sandbox:
        typer.confirm(
            "CCXT_SANDBOX is true. Continue anyway?",
            abort=True,
        )
    run_live_loop(settings, paper=False, display=TerminalDisplay())


if __name__ == "__main__":
    app()
