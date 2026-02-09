"""
Optional TUI (terminal UI) for OllamaCode chat using Rich.
Install with: pip install ollamacode[tui]
Slash commands, Rich Markdown, multi-line input, conversation history.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

from .agent import run_agent_loop_no_mcp_stream, run_agent_loop_stream

if TYPE_CHECKING:
    from .mcp_client import McpConnection


def _conversation_to_markdown(history: list[tuple[str, str]], current: str) -> str:
    """Build markdown string for conversation panel (Rich Markdown renderable)."""
    parts = []
    for role, text in history:
        label = "**You**" if role == "user" else "**Assistant**"
        parts.append(f"{label}\n\n{text}")
    if current:
        parts.append("**Assistant** *(streaming)*\n\n" + current)
    return "\n\n---\n\n".join(parts) if parts else "*(no messages yet)*"


def _handle_tui_slash(
    line: str,
    model_ref: list[str],
    history: list[tuple[str, str]],
    message_history: list[dict[str, Any]],
    console: Any,
) -> str | None:
    """Handle slash command in TUI. Returns 'quit', 'cleared', 'help', or None."""
    line = line.strip()
    if not line.startswith("/"):
        return None
    parts = line.split(maxsplit=1)
    cmd = (parts[0] or "").lower()
    rest = (parts[1] or "").strip() if len(parts) > 1 else ""
    if cmd in ("/clear", "/new"):
        history.clear()
        message_history.clear()
        console.print("[dim]Conversation cleared.[/]")
        return "cleared"
    if cmd == "/help":
        from rich.panel import Panel as RichPanel

        help_text = """[bold]Slash commands:[/]
  /clear, /new   Clear conversation and start fresh
  /help         Show this help
  /model [name] Show or set Ollama model
  /quit, /exit  Exit (or Ctrl+C)"""
        console.print(RichPanel(help_text, title="Commands", border_style="dim"))
        return "help"
    if cmd == "/model":
        if rest:
            model_ref[0] = rest
            console.print(f"[dim]Model set to: {rest}[/]")
            return "help"
        console.print(f"[dim]Current model: {model_ref[0]}[/]")
        return "help"
    if cmd in ("/quit", "/exit"):
        return "quit"
    console.print(f"[dim]Unknown command: {cmd}. Use /help[/]")
    return "help"


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
    Run interactive TUI chat: Rich panels with Markdown, slash commands, multi-line input.
    Requires rich: pip install ollamacode[tui]
    """
    try:
        from rich.console import Console
        from rich.live import Live
        from rich.markdown import Markdown
        from rich.panel import Panel
    except ImportError as e:
        raise ImportError("TUI requires rich. Install with: pip install ollamacode[tui]") from e

    # Suppress MCP SDK INFO logs (e.g. "Processing request of type ListToolsRequest") so TUI stays clean
    for _name in ("mcp", "mcp.client", "mcp.server"):
        logging.getLogger(_name).setLevel(logging.WARNING)

    console = Console()
    _SYSTEM = "You are a helpful coding assistant. Use the available tools when they would help."
    if system_extra:
        _SYSTEM = _SYSTEM + "\n\n" + system_extra

    history: list[tuple[str, str]] = []
    message_history: list[dict[str, Any]] = []
    model_ref = [model]
    loop = asyncio.get_event_loop()

    def get_input() -> str:
        """Read a single line; Enter sends."""
        return input("You: ").strip()

    console.print(
        Panel(
            "[bold]OllamaCode TUI[/] – [dim]/help[/] for commands. Empty or Ctrl+C to exit.",
            title="OllamaCode",
            border_style="green",
        )
    )

    while True:
        try:
            line = await loop.run_in_executor(None, get_input)
        except (EOFError, KeyboardInterrupt, asyncio.CancelledError):
            break
        if not line:
            continue

        result = _handle_tui_slash(line, model_ref, history, message_history, console)
        if result == "quit":
            break
        if result is not None:
            continue

        history.append(("user", line))

        def make_update(accumulated: str) -> None:
            md = _conversation_to_markdown(history, accumulated)
            live.update(Panel(Markdown(md), title="Chat", border_style="blue"))

        msg_history = [{"role": r, "content": c} for r, c in history[:-1]]
        if session is not None:
            stream = run_agent_loop_stream(
                session,
                model_ref[0],
                line,
                system_prompt=_SYSTEM,
                message_history=msg_history,
            )
        else:
            stream = run_agent_loop_no_mcp_stream(
                model_ref[0],
                line,
                system_prompt=_SYSTEM,
                message_history=msg_history,
            )

        with Live(
            Panel(
                Markdown(_conversation_to_markdown(history, "")),
                title="Chat",
                border_style="blue",
            ),
            console=console,
            refresh_per_second=8,
            vertical_overflow="visible",
        ) as live:
            final = await _stream_into_live(stream, make_update)

        history.append(("assistant", final))
        message_history.append({"role": "user", "content": line})
        message_history.append({"role": "assistant", "content": final})
