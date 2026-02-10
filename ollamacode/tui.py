"""
Optional TUI (terminal UI) for OllamaCode chat using Rich.
Install with: pip install ollamacode[tui]
Slash commands, Rich Markdown, multi-line input, conversation history.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import subprocess
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Literal, cast

from .agent import (
    run_agent_loop,
    run_agent_loop_no_mcp,
    run_agent_loop_no_mcp_stream,
    run_agent_loop_stream,
)
from .context import expand_at_refs
from .skills import load_skills_text
from .templates import load_prompt_template

if TYPE_CHECKING:
    from .mcp_client import McpConnection

# Arrow keys and line editing: readline on Unix/macOS; on Windows use prompt_toolkit when available (pip install ollamacode[tui])
_tui_prompt_fn = None
try:
    import readline  # noqa: F401  # enables arrow keys for input() on Unix/macOS
except ImportError:
    try:
        from prompt_toolkit import prompt as _tui_prompt_fn  # pyright: ignore[reportMissingImports]
    except ImportError:
        pass

# Show only the last N exchanges in the Chat panel so tool output (stderr) stays visible
_CHAT_PANEL_LAST_N_EXCHANGES = 2

# rest of original functions unchanged...


def _conversation_to_markdown(
    history: list[tuple[str, str]],
    current: str,
    *,
    limit_exchanges: int | None = _CHAT_PANEL_LAST_N_EXCHANGES,
) -> str:
    """Build markdown string for conversation panel (Rich Markdown renderable).
    If limit_exchanges is set, only the last N user+assistant pairs are shown so
    the panel stays small and tool output remains visible in the terminal.
    """
    if (
        limit_exchanges is not None
        and limit_exchanges > 0
        and len(history) > limit_exchanges * 2
    ):
        history = history[-(limit_exchanges * 2) :]
        prefix = "*(scroll up for earlier messages)*\n\n"
    else:
        prefix = ""
    parts = []
    for role, text in history:
        label = "**You**" if role == "user" else "**Assistant**"
        parts.append(f"{label}\n\n{text}")
    if current:
        parts.append("**Assistant** *(streaming)*\n\n" + current)
    body = "\n\n---\n\n".join(parts) if parts else "*(no messages yet)*"
    return prefix + body


def _run_cmd_sync(workspace_root: str, command: str) -> str:
    """Run command in workspace_root; return combined stdout+stderr. Empty on failure."""
    try:
        parts = shlex.split(command)
        if not parts:
            return ""
        result = subprocess.run(
            parts,
            cwd=workspace_root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        return err if err else out
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return ""


def _handle_tui_slash(
    line: str,
    model_ref: list[str],
    history: list[tuple[str, str]],
    message_history: list[dict[str, Any]],
    console: Any,
    workspace_root: str = ".",
    linter_command: str | None = None,
    test_command: str | None = None,
    docs_command: str | None = None,
    profile_command: str | None = None,
) -> str | None | tuple[str, str] | tuple[Literal["run_summary"], int]:
    """Handle slash command in TUI. Returns 'quit', 'cleared', 'help', ('run_prompt', prompt), ('run_summary', n), or None."""
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
  /fix          Run linter, send errors to model
  /test         Run tests, send failures to model
  /docs         Run docs build, send output to model
  /profile      Run profiler, send summary to model
  /reset-state  Clear persistent state (recent files, preferences)
  /summary [N]  Summarize last N turns (default 5)
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
    if cmd == "/fix":
        run_cmd = linter_command or "ruff check ."
        output = _run_cmd_sync(workspace_root, run_cmd)
        if not output:
            console.print(
                "[dim]No linter output (command may have succeeded or not run).[/]"
            )
            return "help"
        return (
            "run_prompt",
            f"Fix these linter errors (from `{run_cmd}`):\n\n```\n{output}\n```",
        )
    if cmd == "/test":
        run_cmd = test_command or "pytest"
        output = _run_cmd_sync(workspace_root, run_cmd)
        if not output:
            console.print(
                "[dim]No test output (command may have succeeded or not run).[/]"
            )
            return "help"
        return (
            "run_prompt",
            f"Fix these test failures (from `{run_cmd}`):\n\n```\n{output}\n```",
        )
    if cmd == "/docs":
        run_cmd = docs_command or "mkdocs build"
        output = _run_cmd_sync(workspace_root, run_cmd)
        if not output:
            console.print("[dim]No docs output (command may have succeeded).[/]")
            return "help"
        return (
            "run_prompt",
            f"Docs build output (from `{run_cmd}`). Fix any errors or suggest improvements:\n\n```\n{output}\n```",
        )
    if cmd == "/profile":
        run_cmd = (
            profile_command
            or "python -m cProfile -s cumtime -m pytest tests/ -q --no-header 2>&1 | head -40"
        )
        output = _run_cmd_sync(workspace_root, run_cmd)
        if not output:
            console.print("[dim]No profile output.[/]")
            return "help"
        return (
            "run_prompt",
            f"Profiler output (from `{run_cmd}`). Identify hotspots and suggest optimizations:\n\n```\n{output}\n```",
        )
    if cmd == "/reset-state":
        try:
            from .state import clear_state

            msg = clear_state()
            console.print(f"[dim]{msg}[/]")
        except Exception as e:
            console.print(f"[dim]Failed to clear state: {e}[/]")
        return "help"
    if cmd == "/summary":
        try:
            n = int(rest) if rest else 5
            n = max(1, min(n, 50))
        except ValueError:
            n = 5
        return ("run_summary", n)
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
    *,
    quiet: bool = False,
    max_tool_rounds: int = 20,
    max_messages: int = 0,
    max_tool_result_chars: int = 0,
    timing: bool = False,
    workspace_root: str | None = None,
    linter_command: str | None = None,
    test_command: str | None = None,
    docs_command: str | None = None,
    profile_command: str | None = None,
    show_semantic_hint: bool = False,
    use_skills: bool = True,
    prompt_template: str | None = None,
    inject_recent_context: bool = True,
    recent_context_max_files: int = 10,
    branch_context: bool = False,
    branch_context_base: str = "main",
) -> None:
    """Run interactive TUI chat: Rich panels with Markdown, slash commands, multi-line input.
    Requires rich: pip install ollamacode[tui]
    """
    try:
        from rich.console import Console
        from rich.live import Live
        from rich.markdown import Markdown
        from rich.panel import Panel
    except ImportError as e:
        raise ImportError(
            "TUI requires rich. Install with: pip install ollamacode[tui]"
        ) from e

    # Suppress MCP SDK INFO logs (e.g. "Processing request of type ListToolsRequest") so TUI stays clean
    for _name in ("mcp", "mcp.client", "mcp.server"):
        logging.getLogger(_name).setLevel(logging.WARNING)

    console = Console()
    _SYSTEM = (
        "You are a coding assistant with full access to the workspace. You are given a list of available tools with their names "
        "and descriptions—use whichever tools fit the task. When the user asks you to run something, check something, or change "
        "something, use the appropriate tool and report the result."
    )
    if system_extra:
        _SYSTEM = _SYSTEM + "\n\n" + system_extra
    if use_skills:
        skills_text = load_skills_text(workspace_root)
        if skills_text:
            _SYSTEM = (
                _SYSTEM
                + "\n\n--- Skills (saved instructions & memory) ---\n\n"
                + skills_text
            )
    if prompt_template:
        template_text = load_prompt_template(prompt_template, workspace_root)
        if template_text:
            _SYSTEM = _SYSTEM + "\n\n--- Prompt template ---\n\n" + template_text
    if inject_recent_context and workspace_root:
        from .state import get_state, format_recent_context
        from .context import get_branch_summary_one_line

        state = get_state()
        block = format_recent_context(state, max_files=recent_context_max_files)
        if branch_context:
            branch_line = get_branch_summary_one_line(
                workspace_root, branch_context_base
            )
            if branch_line:
                block = (block + "\n\n" + branch_line) if block else branch_line
        if block:
            _SYSTEM = _SYSTEM + "\n\n--- Recent context ---\n\n" + block

    history: list[tuple[str, str]] = []
    message_history: list[dict[str, Any]] = []
    model_ref = [model]
    loop = asyncio.get_event_loop()

    def get_input() -> str:
        """Read a single line; Enter sends. Prompt shows current model. Uses prompt_toolkit on Windows when available for arrow keys."""
        if _tui_prompt_fn is not None:
            return _tui_prompt_fn(f"You [{model_ref[0]}]: ").strip()
        return input(f"You [{model_ref[0]}]: ").strip()

    console.print(
        Panel(
            "[bold]OllamaCode TUI[/] – [dim]/help[/] for commands. Empty or Ctrl+C to exit.",
            title="OllamaCode",
            border_style="green",
        )
    )
    if show_semantic_hint:
        console.print(
            "[dim]Tip: For semantic codebase search, add a semantic MCP server to config. See docs/MCP_SERVERS.md.[/]"
        )

    while True:
        try:
            line = await loop.run_in_executor(None, get_input)
        except (EOFError, KeyboardInterrupt, asyncio.CancelledError):
            break
        if not line:
            continue

        result = _handle_tui_slash(
            line,
            model_ref,
            history,
            message_history,
            console,
            workspace_root=workspace_root or os.getcwd(),
            linter_command=linter_command,
            test_command=test_command,
            docs_command=docs_command,
            profile_command=profile_command,
        )
        if result == "quit":
            break
        if result is not None:
            if isinstance(result, tuple) and result[0] == "run_prompt":
                line = cast(str, result[1])
            elif isinstance(result, tuple) and result[0] == "run_summary":
                n_turns = cast(int, result[1])
                n_msgs = min(n_turns * 2, len(message_history))
                if n_msgs == 0:
                    console.print("[dim]No conversation to summarize.[/]")
                    continue
                transcript_parts = []
                for i in range(-n_msgs, 0):
                    m = message_history[i]
                    role = "User" if m.get("role") == "user" else "Assistant"
                    transcript_parts.append(f"{role}: {m.get('content', '')}")
                transcript = "\n\n".join(transcript_parts)
                prompt = f"Summarize the following conversation in one short paragraph. Reply with only the summary, no preamble:\n\n{transcript}"
                console.print("[dim]Summarizing last", n_turns, "turn(s)...[/]")
                if session is not None:
                    summary = await run_agent_loop(
                        session,
                        model_ref[0],
                        prompt,
                        system_prompt=_SYSTEM,
                        message_history=[],
                        max_tool_rounds=0,
                    )
                else:
                    summary = await run_agent_loop_no_mcp(
                        model_ref[0],
                        prompt,
                        system_prompt=_SYSTEM,
                        message_history=[],
                    )
                summary = summary.strip()
                message_history[:] = message_history[:-n_msgs] + [
                    {"role": "user", "content": "Summary of previous conversation"},
                    {"role": "assistant", "content": summary},
                ]
                history[:] = history[:-n_msgs] + [
                    ("user", "Summary of previous conversation"),
                    ("assistant", summary),
                ]
                console.print("[dim]Replaced with summary.[/]")
                continue
            else:
                continue

        history.append(("user", line))
        root = workspace_root if workspace_root is not None else os.getcwd()
        line_expanded = expand_at_refs(line, root)

        def make_update(accumulated: str) -> None:
            md = _conversation_to_markdown(history, accumulated)
            live.update(Panel(Markdown(md), title="Chat", border_style="blue"))

        msg_history = [{"role": r, "content": c} for r, c in history[:-1]]
        if session is not None:
            stream = run_agent_loop_stream(
                session,
                model_ref[0],
                line_expanded,
                system_prompt=_SYSTEM,
                max_tool_rounds=max_tool_rounds,
                max_messages=max_messages,
                max_tool_result_chars=max_tool_result_chars,
                message_history=msg_history,
                quiet=quiet,
                timing=timing,
                tool_progress_brief=True,
            )
        else:
            stream = run_agent_loop_no_mcp_stream(
                model_ref[0],
                line_expanded,
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
