"""Rich terminal UI for backtest and live trading."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from llm_trading_bot.trading.models import ExecutionOutcome

if TYPE_CHECKING:
    from llm_trading_bot.config import Settings
    from llm_trading_bot.trading.models import (
        AccountState,
        Action,
        LLMDecision,
        PositionState,
    )

ACTION_STYLES = {
    "hold": ("bold yellow", "HOLD"),
    "close": ("bold blue", "CLOSE"),
    "enter_long": ("bold green", "ENTER LONG"),
    "enter_short": ("bold red", "ENTER SHORT"),
    "adjust_stops": ("bold cyan", "ADJUST STOPS"),
}

OUTCOME_STYLES = {
    ExecutionOutcome.NOOP: None,
    ExecutionOutcome.CLOSED: ("dim", "closed"),
    ExecutionOutcome.STOPS_ADJUSTED: ("green", "stops updated"),
    ExecutionOutcome.ORDER_SUBMITTED: ("green", "order submitted"),
    ExecutionOutcome.EXECUTED: ("green", "filled"),
    ExecutionOutcome.SKIPPED_INVALID_LEVELS: ("red", "skipped — invalid SL/TP"),
    ExecutionOutcome.SKIPPED_ZERO_SIZE: ("red", "skipped — zero size"),
    ExecutionOutcome.SKIPPED_PENDING_ENTRY: ("yellow", "skipped — entry pending"),
    ExecutionOutcome.SKIPPED_ORDER_REJECTED: ("red", "skipped — order rejected"),
}


def configure_logging(verbose: bool) -> None:
    """Quiet third-party noise; route user-facing output through Rich."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    for name in ("httpx", "httpcore", "openai", "urllib3", "ccxt"):
        logging.getLogger(name).setLevel(logging.WARNING)
    bot_level = logging.DEBUG if verbose else logging.WARNING
    logging.getLogger("llm_trading_bot").setLevel(bot_level)


class TerminalDisplay:
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._live_candle = 0

    def print_backtest_header(
        self,
        settings: Settings,
        candles: int,
        decisions: int,
        span_label: str,
        from_ts: object,
        to_ts: object,
    ) -> None:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="dim")
        table.add_column()
        table.add_row("Symbol", f"[bold]{settings.symbol}[/]")
        table.add_row("Timeframe", settings.timeframe)
        table.add_row("Window", span_label)
        table.add_row("Candles", str(candles))
        table.add_row("Warmup", f"{settings.candle_history} bars (no LLM)")
        table.add_row("LLM calls", str(decisions))
        table.add_row("Range", f"{from_ts} → {to_ts}")
        table.add_row("Model", settings.openai_model)
        table.add_row("Leverage", f"{settings.leverage:g}x")
        self.console.print(
            Panel(table, title="[bold]Backtest[/]", border_style="cyan", padding=(1, 2))
        )
        self.console.print()

    def print_live_header(self, settings: Settings, mode: str) -> None:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="dim")
        table.add_column()
        table.add_row("Mode", f"[bold]{mode}[/]")
        table.add_row("Symbol", settings.symbol)
        table.add_row("Timeframe", settings.timeframe)
        table.add_row("History", f"{settings.candle_history} candles to LLM")
        table.add_row("Model", settings.openai_model)
        self.console.print(
            Panel(table, title="[bold]Live loop[/]", border_style="cyan", padding=(1, 2))
        )
        self.console.print(Rule(style="dim"))

    def next_live_candle(self) -> int:
        self._live_candle += 1
        return self._live_candle

    def print_candle(
        self,
        *,
        bar: int | None,
        total_bars: int | None,
        symbol: str,
        timeframe: str,
        close_price: float,
        position: PositionState,
        account: AccountState,
        decision: LLMDecision,
        outcome: ExecutionOutcome | None = None,
    ) -> None:
        from llm_trading_bot.trading.models import Action, PositionSide

        action_key = decision.action.value
        style, action_label = ACTION_STYLES[action_key]

        if outcome is None:
            outcome = ExecutionOutcome.NOOP

        if bar is not None and total_bars:
            step = f"Candle [bold]{bar}[/] / {total_bars}"
        elif bar is not None:
            step = f"Candle [bold]{bar}[/]"
        else:
            step = "New candle"

        header = (
            f"{step}  ·  {symbol} {timeframe}  ·  "
            f"Close [bold]${close_price:,.2f}[/]"
        )

        lines: list[str] = []
        lines.append(
            f"[dim]Cash[/]  ${account.available_cash:,.2f}   "
            f"[dim]Equity[/]  ${account.equity:,.2f}"
        )
        if abs(account.equity - account.available_cash) > 0.01:
            invested = account.equity - account.available_cash
            inv_style = "green" if invested >= 0 else "red"
            lines.append(
                f"[dim]In positions[/]  [{inv_style}]${invested:+,.2f}[/]  "
                f"[dim](cash + mark-to-market)[/]"
            )
        lines.append(self._format_position(position))

        action_line = Text()
        action_line.append("Decision  ", style="dim")
        action_line.append(action_label, style=style)
        if decision.action in (
            Action.ENTER_LONG,
            Action.ENTER_SHORT,
            Action.ADJUST_STOPS,
        ):
            if decision.action in (Action.ENTER_LONG, Action.ENTER_SHORT):
                action_line.append(
                    f"  ·  risk {decision.risk_pct * 100:.1f}%  ·  ",
                    style="dim",
                )
            action_line.append(
                f"SL ${decision.stop_loss:,.2f}  ·  TP ${decision.take_profit:,.2f}",
                style="dim",
            )
        outcome_style = OUTCOME_STYLES.get(outcome)
        if outcome_style:
            o_style, o_label = outcome_style
            action_line.append("  ·  ", style="dim")
            action_line.append(o_label, style=o_style)
        lines.append(str(action_line))

        reasoning = decision.reasoning.strip() or "(no reasoning)"
        lines.append(f"[dim]Thinking[/]\n{reasoning}")

        border = {
            Action.HOLD: "yellow",
            Action.CLOSE: "blue",
            Action.ENTER_LONG: "green",
            Action.ENTER_SHORT: "red",
            Action.ADJUST_STOPS: "cyan",
        }[decision.action]

        self.console.print(Panel("\n".join(lines), title=header, border_style=border))

    def print_backtest_summary(self, result: dict, settings: Settings) -> None:
        pnl = result["pnl"]
        pnl_style = "green" if pnl >= 0 else "red"
        ret = result["return_pct"]
        ret_style = "green" if ret >= 0 else "red"

        table = Table(title="Results", show_header=False, box=None, padding=(0, 1))
        table.add_column(style="dim", min_width=14)
        table.add_column(justify="right")
        table.add_row("Symbol", settings.symbol)
        table.add_row("Candles", str(result["candles"]))
        table.add_row("LLM decisions", str(result["llm_decisions"]))
        table.add_row("Start", f"${result['start_value']:,.2f}")
        table.add_row("End", f"${result['end_value']:,.2f}")
        table.add_row("PnL", f"[{pnl_style}]${pnl:,.2f}[/]")
        table.add_row("Return", f"[{ret_style}]{ret:+.2f}%[/]")

        self.console.print()
        self.console.print(Panel(table, border_style="cyan", padding=(1, 2)))

    @staticmethod
    def _format_position(position: PositionState) -> str:
        from llm_trading_bot.trading.models import PositionSide

        if position.side == PositionSide.FLAT:
            if position.pending_entry:
                return "[dim]Position[/]  [bold]FLAT[/]  ·  [yellow]entry order pending[/]"
            return "[dim]Position[/]  [bold]FLAT[/]"

        side_style = "green" if position.side == PositionSide.LONG else "red"
        entry = position.entry_price or 0.0
        pnl = position.unrealized_pnl
        pnl_style = "green" if pnl >= 0 else "red"
        parts = [
            f"[dim]Position[/]  [{side_style} bold]{position.side.value.upper()}[/]  ·  "
            f"size {position.size:.6f}  ·  entry ${entry:,.2f}  ·  "
            f"[dim]uPnL[/]  [{pnl_style}]${pnl:+,.2f}[/]",
        ]
        if position.stop_loss is not None:
            parts.append(f"[dim]SL[/]  ${position.stop_loss:,.2f}")
        if position.take_profit is not None:
            parts.append(f"[dim]TP[/]  ${position.take_profit:,.2f}")
        return "  ·  ".join(parts)
