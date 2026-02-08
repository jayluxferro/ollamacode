"""
Optional TUI (terminal UI) for OllamaCode chat using Rich.
Install with: pip install ollamacode[tui]
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, AsyncIterator, Callable

from .agent import run_agent_loop_no_mcp_stream, run_agent_loop_stream

if TYPE_CHECKING:
    from .mcp_client import McpConnection


def _render_conversation(history: list[tuple[str, str]], current: str) -> str:
    """Build text for conversation panel: history + current reply."""
    parts = []
    for role, text in history:
        label = "You" if role == "user" else "Assistant"
        parts.append(f"[bold blue]{label}:[/]\n{text}")
    if current:
        parts.append(f"[bold green]Assistant:[/]\n{current}")
    return "\n\n".join(parts) if parts else " (no messages yet)"


async def _stream_into_live(
    stream: AsyncIterator[str],
    update_cb: Callable[[str], None],
) -> str:
    """Consume async stream, call update_cb(accumulated) on each fragment, return final text."""
    accumulated: list[str] = []
    async for frag in stream:
        accumulated.append(frag)
        update_cb("".join(accumulated))
    return "".join(accumulated)


async def run_tui(
    session: McpConnection | None,
    model: str,
    system_extra: str,
) -> None:
    """
    Run interactive TUI chat: Rich panels for history and streaming reply, prompt for input.
    Requires rich: pip install ollamacode[tui]
    """
    try:
        from rich.console import Console
        from rich.live import Live
        from rich.panel import Panel
    except ImportError as e:
        raise ImportError(
            "TUI requires rich. Install with: pip install ollamacode[tui]"
        ) from e

    console = Console()
    _SYSTEM = "You are a helpful coding assistant. Use the available tools when they would help."
    if system_extra:
        _SYSTEM = _SYSTEM + "\n\n" + system_extra

    history: list[tuple[str, str]] = []
    loop = asyncio.get_event_loop()

    def get_input() -> str:
        return input("You: ").strip()

    console.print(
        Panel(
            "[bold]OllamaCode TUI[/] – local model + MCP. Empty line or Ctrl+C to exit.",
            title="OllamaCode",
            border_style="green",
        )
    )

    while True:
        try:
            line = await loop.run_in_executor(None, get_input)
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue

        history.append(("user", line))

        def make_update(accumulated: str) -> None:
            text = _render_conversation(history, accumulated)
            live.update(Panel(text, title="Chat", border_style="blue", height=20))

        if session is not None:
            stream = run_agent_loop_stream(
                session, model, line, system_prompt=_SYSTEM
            )
        else:
            stream = run_agent_loop_no_mcp_stream(
                model, line, system_prompt=_SYSTEM
            )

        with Live(
            Panel(
                _render_conversation(history, ""),
                title="Chat",
                border_style="blue",
                height=20,
            ),
            console=console,
            refresh_per_second=8,
        ) as live:
            final = await _stream_into_live(stream, make_update)

        history.append(("assistant", final))
