import typer
from rich.console import Console

from llm_trading_bot.config import get_settings
from llm_trading_bot.display import TerminalDisplay, configure_logging
from llm_trading_bot.runners.backtest import run_backtest_for_range
from llm_trading_bot.runners.live import run_live_loop

app = typer.Typer(help="LLM trading bot — backtest, paper, and live modes.")
console = Console()


@app.command()
def backtest(
    from_date: str = typer.Option(
        ...,
        "--from",
        help="Start of backtest window (UTC), e.g. 2025-01-01",
    ),
    to_date: str = typer.Option(
        ...,
        "--to",
        help="End of backtest window, inclusive for date-only values, e.g. 2025-01-07",
    ),
    cash: float | None = typer.Option(
        None,
        "--cash",
        help="Starting cash (overrides STARTING_BALANCE from .env).",
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    """Backtest on historical OHLCV fetched from the exchange for a date range."""
    configure_logging(verbose)
    settings = get_settings()
    if not settings.openai_api_key:
        typer.echo("Set OPENAI_API_KEY in .env before running.", err=True)
        raise typer.Exit(1)

    initial_cash = cash if cash is not None else settings.starting_balance
    display = TerminalDisplay()

    try:
        result = run_backtest_for_range(
            settings,
            from_date,
            to_date,
            initial_cash=initial_cash,
            display=display,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

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
