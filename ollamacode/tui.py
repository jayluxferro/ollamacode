"""
Optional TUI (terminal UI) for OllamaCode chat using Rich.
Install with: pip install ollamacode[tui]
Slash commands, Rich Markdown, multi-line input, conversation history.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Literal, cast
from collections import deque

from .agent import (
    _tool_call_one_line,
    run_agent_loop,
    run_agent_loop_no_mcp,
    run_agent_loop_no_mcp_stream,
    run_agent_loop_stream,
)
from .edits import (
    apply_edits,
    format_edits_diff,
    parse_edits,
    apply_unified_diff_filtered,
)
from .memory import build_dynamic_memory_context
from .multi_agent import run_multi_agent
from .context import expand_at_refs
from .skills import load_skills_text
from .templates import load_prompt_template

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .mcp_client import McpConnection

# Slash commands for TUI (for autocomplete)
_SLASH_COMMANDS = [
    "/clear",
    "/new",
    "/help",
    "/model",
    "/fix",
    "/test",
    "/docs",
    "/profile",
    "/plan",
    "/continue",
    "/rate",
    "/query_docs",
    "/multi",
    "/kg_add",
    "/kg_query",
    "/rag_index",
    "/rag_query",
    "/copy",
    "/trace",
    "/compact",
    "/reset-state",
    "/summary",
    "/auto",
    "/agents",
    "/agents_show",
    "/agents_summary",
    "/listen",
    "/say",
    "/commands",
    "/sessions",
    "/search",
    "/refactor",
    "/palette",
    "/resume",
    "/session",
    "/branch",
    "/checkpoints",
    "/rewind",
    "/quit",
    "/exit",
]


# Helpers
def _extract_pasted_image(line: str) -> tuple[str | None, str]:
    """Detect data URL image paste; save to disk and return (path, cleaned_line).
    Supports png, jpeg, jpg, gif, webp formats."""
    if "data:image" not in line:
        return None, line
    m = re.search(r"data:image/(png|jpeg|jpg|gif|webp);base64,([A-Za-z0-9+/=]+)", line)
    if not m:
        return None, line
    _ext_map = {"jpeg": "jpg", "jpg": "jpg", "png": "png", "gif": "gif", "webp": "webp"}
    ext = _ext_map.get(m.group(1), "png")
    b64 = m.group(2)
    try:
        raw = base64.b64decode(b64)
        tmp_dir = Path.home() / ".ollamacode" / "clipboard"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        path = tmp_dir / f"paste_{int(time.time())}.{ext}"
        path.write_bytes(raw)
        cleaned = line.replace(m.group(0), "").strip()
        return str(path), cleaned
    except Exception:
        return None, line


# Arrow keys and line editing: readline on Unix/macOS; on Windows use prompt_toolkit when available (pip install ollamacode[tui])
# Up/Down: readline.add_history_entry() for input(); PromptSession keeps history for prompt_toolkit.
# Tab: complete slash commands when line starts with /.
_tui_prompt_fn = None
_tui_prompt_session = None
_readline_available = False
_tui_voice_ptt_cb = None
_tui_ptt_key = os.environ.get("OLLAMACODE_TUI_PTT_KEY", "c-space")
try:
    import readline  # noqa: F401  # enables arrow keys and history for input() on Unix/macOS

    _readline_available = True
except ImportError:
    pass
# Readline completer for slash commands (used when prompt_toolkit not available)
_readline_slash_matches: list[str] = []


def _readline_slash_completer(text: str, state: int) -> str | None:
    """Complete slash commands when line starts with / (Tab). Used by readline."""
    global _readline_slash_matches
    if state == 0:
        prefix = (text or "").strip()
        if not prefix.startswith("/"):
            _readline_slash_matches = []
        else:
            _readline_slash_matches = [
                c for c in _SLASH_COMMANDS if c.startswith(prefix.lower())
            ]
    if state < len(_readline_slash_matches):
        return _readline_slash_matches[state]
    return None


# Initialize readline tab-completion once at import time (only when prompt_toolkit is absent).
if _readline_available and _tui_prompt_session is None:
    try:
        readline.set_completer(_readline_slash_completer)  # type: ignore[name-defined]
        readline.parse_and_bind("tab: complete")
    except Exception:
        pass

_symbol_cache: list[str] = []
_symbol_cache_ts: float = 0.0
# Mutable workspace root reference for path completion; updated by run_tui().
_workspace_root_ref: list[str] = []
_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def _reset_symbol_cache() -> None:
    global _symbol_cache, _symbol_cache_ts
    _symbol_cache = []
    _symbol_cache_ts = 0.0


# Theme support: OLLAMACODE_THEME env var (dark/light)
_THEMES: dict[str, dict[str, str]] = {
    "dark": {
        "panel_main": "green",
        "panel_chat": "blue",
        "panel_tools": "magenta",
        "panel_agents": "cyan",
        "panel_status": "green",
        "panel_timeline": "yellow",
        "panel_info": "dim",
        "status_dim": "dim",
    },
    "light": {
        "panel_main": "dark_green",
        "panel_chat": "dark_blue",
        "panel_tools": "dark_magenta",
        "panel_agents": "dark_cyan",
        "panel_status": "dark_green",
        "panel_timeline": "dark_goldenrod",
        "panel_info": "grey50",
        "status_dim": "grey50",
    },
}


def _get_theme() -> dict[str, str]:
    """Return the active theme dict based on OLLAMACODE_THEME env var."""
    name = os.environ.get("OLLAMACODE_THEME", "dark").strip().lower()
    return _THEMES.get(name, _THEMES["dark"])


def _highlight_diff(diff_text: str) -> Any:
    """Syntax-highlight a unified diff using Pygments if available; returns a Rich renderable."""
    if not diff_text:
        return diff_text
    try:
        from pygments import highlight as pyg_highlight
        from pygments.lexers import DiffLexer
        from pygments.formatters import TerminalTrueColorFormatter

        return pyg_highlight(diff_text, DiffLexer(), TerminalTrueColorFormatter())
    except ImportError:
        return diff_text


def _escape_rich_markup(text: str) -> str:
    """Escape Rich markup tags in user input so they are not interpreted as formatting."""
    if not text or "[" not in text:
        return text
    return text.replace("[", "\\[")


def _sanitize_stream_text(text: str) -> str:
    """
    Remove terminal control noise so TUI panels don't show raw escape sequences
    or carriage-return artifacts while streaming.
    """
    if not text:
        return ""
    cleaned = _ANSI_ESCAPE_RE.sub("", text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\x00", "")
    return cleaned


def _set_apple_fm_session_key(session_id: str | None) -> None:
    if not session_id:
        return
    os.environ.setdefault("OLLAMACODE_APPLE_FM_STATEFUL", "1")
    os.environ["OLLAMACODE_APPLE_FM_SESSION_KEY"] = session_id


if _tui_prompt_session is None:
    try:
        from prompt_toolkit import PromptSession  # pyright: ignore[reportMissingImports]
        from prompt_toolkit.history import InMemoryHistory  # pyright: ignore[reportMissingImports]
        from prompt_toolkit.completion import Completer, Completion  # pyright: ignore[reportMissingImports]
        from prompt_toolkit.key_binding import KeyBindings  # pyright: ignore[reportMissingImports]

        def _load_symbol_suggestions() -> list[str]:
            """Load symbol names from persistent index (best-effort)."""
            global _symbol_cache, _symbol_cache_ts
            now = time.time()
            if _symbol_cache and now - _symbol_cache_ts < 10:
                return _symbol_cache
            try:
                import sqlite3
                from .symbol_index import _db_path  # type: ignore

                db = _db_path()
                if db.exists():
                    conn = sqlite3.connect(str(db))
                    try:
                        cur = conn.execute(
                            "SELECT DISTINCT name FROM symbols LIMIT 200"
                        )
                        _symbol_cache = [r[0] for r in cur.fetchall() if r and r[0]]
                    finally:
                        conn.close()
                else:
                    _symbol_cache = []
            except Exception:
                _symbol_cache = []
            _symbol_cache_ts = now
            return _symbol_cache

        def _path_completions(prefix: str) -> list[str]:
            root = Path(
                _workspace_root_ref[0]
                if _workspace_root_ref
                else os.environ.get("OLLAMACODE_FS_ROOT") or os.getcwd()
            )
            if prefix.startswith("@"):
                prefix = prefix[1:]
            base = root / prefix
            try:
                if base.is_dir():
                    entries = [
                        p.name + ("/" if p.is_dir() else "") for p in base.iterdir()
                    ]
                    return [f"@{prefix.rstrip('/')}/{e}" for e in entries][:50]
                parent = base.parent
                if parent.exists():
                    entries = [
                        p.name + ("/" if p.is_dir() else "") for p in parent.iterdir()
                    ]
                    return [f"@{parent.relative_to(root)}/{e}" for e in entries][:50]
            except OSError:
                return []
            return []

        class _SlashCommandCompleter(Completer):
            """Complete slash commands, paths (@), and symbols."""

            def get_completions(self, document, complete_event):
                text = document.text_before_cursor
                idx = text.rfind("/")
                if idx < 0:
                    # Path or symbol completion
                    if text.strip().startswith("@"):
                        prefix = text.strip()
                        for c in _path_completions(prefix):
                            if c.startswith(prefix):
                                yield Completion(c, start_position=-len(prefix))
                        return
                    # Symbol completions (best-effort)
                    prefix = text.strip()
                    if prefix:
                        for s in _load_symbol_suggestions():
                            if s.startswith(prefix):
                                yield Completion(s, start_position=-len(prefix))
                    return
                prefix = text[idx:].lower()
                if not prefix.startswith("/"):
                    return
                for cmd in _SLASH_COMMANDS:
                    if cmd.startswith(prefix) and cmd != prefix:
                        yield Completion(cmd, start_position=-len(prefix))

        kb_main = KeyBindings()

        @kb_main.add("enter")
        def _accept(event):
            event.app.current_buffer.validate_and_handle()

        @kb_main.add("escape", "enter")
        def _newline(event):
            """Alt/Option+Enter inserts a newline (terminals cannot distinguish Shift+Enter)."""
            event.app.current_buffer.insert_text("\n")

        @kb_main.add(_tui_ptt_key)
        def _push_to_talk(event):
            if _tui_voice_ptt_cb is None:
                return
            try:
                text = _tui_voice_ptt_cb()
            except Exception:
                return
            if text:
                event.app.current_buffer.insert_text(text)
                event.app.current_buffer.validate_and_handle()

        @kb_main.add("c-l")
        def _clear_screen(event):
            """Ctrl+L clears the screen."""
            event.app.renderer.clear()

        @kb_main.add("c-v")
        def _paste_image(event):
            """Ctrl+V: attempt clipboard image paste via pngpaste (macOS), then fall back to text paste."""
            try:
                tmp_dir = Path.home() / ".ollamacode" / "clipboard"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = tmp_dir / f"clipboard_{int(time.time())}.png"
                p = subprocess.run(
                    ["pngpaste", str(tmp_path)],
                    capture_output=True,
                    text=True,
                )
                if p.returncode == 0 and tmp_path.exists():
                    event.app.current_buffer.insert_text(f"/image {tmp_path} ")
                    return
            except Exception:
                pass
            # Fall back to default paste behavior
            event.app.current_buffer.paste_clipboard_data(
                event.app.clipboard.get_data()
            )

        _tui_prompt_session = PromptSession(
            history=InMemoryHistory(),
            completer=_SlashCommandCompleter(),
            complete_while_typing=True,
            multiline=True,
            key_bindings=kb_main,
        )
    except ImportError:
        pass

# Show only the last N exchanges in the Chat panel so tool output (stderr) stays visible.
# 5 exchanges = 10 messages (user+assistant pairs), which gives useful context without
# crowding out the tool trace below.
_CHAT_PANEL_LAST_N_EXCHANGES = 5

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
        display_text = _escape_rich_markup(text) if role == "user" else text
        parts.append(f"{label}\n\n{display_text}")
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


def _copy_text_sync(text: str) -> bool:
    """Best-effort clipboard copy across platforms."""
    if not text:
        return False
    candidates: list[list[str]] = []
    if os.name == "nt":
        candidates.append(["clip"])
    else:
        candidates.extend(
            [
                ["pbcopy"],
                ["wl-copy"],
                ["xclip", "-selection", "clipboard"],
                ["xsel", "--clipboard", "--input"],
            ]
        )
    for cmd in candidates:
        try:
            p = subprocess.run(
                cmd, input=text, text=True, capture_output=True, timeout=5
            )
            if p.returncode == 0:
                return True
        except (OSError, subprocess.SubprocessError):
            continue
    return False


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
    autonomous_ref: list[bool] | None = None,
    custom_commands: list[tuple[str, str, str | None]] | None = None,
    session_ref: list[str] | None = None,
    subagents: list[dict[str, Any]] | None = None,
) -> (
    str
    | None
    | tuple[str, str]
    | tuple[Literal["run_summary"], int]
    | tuple[Literal["run_multi"], str]
    | tuple[Literal["copy_last"], str]
    | tuple[Literal["set_trace_filter"], str]
    | tuple[Literal["set_compact_mode"], str]
    | tuple[Literal["kg_add"], str]
    | tuple[Literal["kg_query"], str]
    | tuple[Literal["rag_index"], str]
    | tuple[Literal["new_session"],]
    | tuple[Literal["resume_session"], str]
    | tuple[Literal["branch_session"], str]
    | tuple[Literal["run_subagent"], str, str]
    | tuple[Literal["set_pending_image"], str, str]
    | tuple[Literal["run_agents"], int, str]
    | tuple[Literal["voice_in"], float]
    | tuple[Literal["voice_out"], str]
    | tuple[Literal["show_agents"], str]
    | tuple[Literal["agents_summary"], str]
):
    """Handle slash command in TUI. Returns 'quit', 'cleared', 'help', ('run_prompt', prompt), ('run_multi', prompt), ('run_summary', n), ('run_subagent', type, task), ('set_pending_image', path, msg), ('new_session',), ('resume_session', id), ('branch_session', new_id), or None."""
    line = line.strip()
    if not line.startswith("/"):
        return None
    # Use shlex to handle quoted arguments (e.g. /session "My Session Title")
    try:
        _shlex_parts = shlex.split(line)
    except ValueError:
        # Malformed quoting — fall back to simple split
        _shlex_parts = line.split(maxsplit=1)
    cmd = (_shlex_parts[0] if _shlex_parts else "").lower()
    rest = " ".join(_shlex_parts[1:]) if len(_shlex_parts) > 1 else ""
    # Custom commands from commands.md
    if custom_commands:
        for name, description, prompt_template in custom_commands:
            if name.lower() == cmd:
                if prompt_template:
                    prompt = prompt_template.replace("{{rest}}", rest).replace(
                        "{rest}", rest
                    )
                    return ("run_prompt", prompt)
                console.print(f"[dim]{name}: {description}[/]")
                return "help"
    if cmd in ("/clear", "/new"):
        history.clear()
        message_history.clear()
        if session_ref is not None:
            try:
                from .sessions import create_session

                session_ref[0] = create_session("")
                _set_apple_fm_session_key(session_ref[0])
                console.print("[dim]New session started.[/]")
            except Exception:
                console.print("[dim]Conversation cleared.[/]")
        else:
            console.print("[dim]Conversation cleared.[/]")
        return "cleared"
    if cmd == "/help":
        from rich.panel import Panel as RichPanel

        help_text = """[bold]Slash commands:[/]
  /clear, /new   Clear conversation and start fresh (/new starts a new persisted session)
  /sessions     List recent sessions
  /search [q]   Search sessions by title or message content
  /resume [id]  Resume a session by id
  /session [title]  Show current session or set title
  /branch       Branch current session (copy history to new session)
  /checkpoints  List recent checkpoints for this session
  /rewind [id]  Rewind code to a checkpoint (use /checkpoints to list)
  /help         Show this help
  /model [name] Show or set Ollama model
  /fix          Run linter, send errors to model
  /test         Run tests, send failures to model
  /docs         Run docs build, send output to model
  /profile      Run profiler, send summary to model
  /plan <text>  Set long-term plan (use /continue to work on it)
  /continue     Continue with current plan (next step)
  /kg_add <topic> | <summary> [| rel1,rel2]  Add/update lightweight knowledge graph node
  /kg_query <q> Search lightweight knowledge graph nodes
  /rag_index [path] Build local RAG index (default: workspace root)
  /rag_query <q> Retrieve top local snippets and ask model with that context
  /rate good|bad  Record feedback on last reply
  /query_docs [q] Search codebase for docs relevant to query
  /multi [goal] Multi-agent: planner → executor → reviewer
  /copy         Copy last assistant reply to clipboard
  /trace [q]    Filter tool trace/log by text (empty clears)
  /compact [on|off|auto] Toggle compact chat view
  /reset-state  Clear persistent state (recent files, preferences)
  /summary [N]  Summarize last N turns (default 5)
  /auto         Toggle autonomous mode (no per-tool confirm, more rounds)
  /agents <N> <task>  Run N concurrent agents (default 3)
  /agents_show <n|all|summary>  Show last agent outputs (collapsed by default)
  /agents_summary  Re-summarize last agent outputs
  /listen [s]  Record from mic for s seconds (default 5) and send as prompt
  /say <text>  Speak text using local TTS
  /commands     List built-in and custom slash commands
  /refactor     Refactor helpers: index/rename/extract/move/rollback
  /palette      Quick command picker
  /subagents    List available subagent types
  /subagent <type> <task>  Run task in a subagent (restricted tools)
  /image <path> [msg]  Attach image to next message (vision models)
  /quit, /exit  Exit (or Ctrl+C)

[dim]Input tips:[/]
  Tab to autocomplete commands (prompt_toolkit). Shift+Enter adds a newline when available.
  Set OLLAMACODE_TUI_QUEUE_INPUT=0 to disable input queue while running.
  Set OLLAMACODE_TUI_SIMPLE=1 to disable Live rendering.
  Set OLLAMACODE_TUI_AUTO_AGENTS=1 to auto-spawn /agents for larger tasks.
  Set OLLAMACODE_TUI_AUTO_AGENTS_PLAN=0 to disable planner routing.
  Set OLLAMACODE_TUI_PLANNER_MODEL=<name> to override planner model.
  Set OLLAMACODE_TUI_AGENTS_SUMMARY=0 to disable agent summary.
  Set OLLAMACODE_TUI_AGENTS_SUMMARY_MODEL=<name> to override summary model.
  Set OLLAMACODE_TUI_AGENTS_SYNTHESIS=0 to disable agent synthesis.
  Set OLLAMACODE_PLAN_EXECUTE_VERIFY=1 to enable plan/execute/verify.
  Set OLLAMACODE_TUI_VOICE_OUT=1 to speak assistant replies.
  Set OLLAMACODE_TUI_PTT_KEY=<key> to change push-to-talk hotkey (default: c-space).
  Push-to-talk hotkey: Ctrl+Space (configurable)."""
        console.print(RichPanel(help_text, title="Commands", border_style="dim"))
        return "help"
    if cmd == "/model":
        if rest:
            _model_name_re = re.compile(r"^[a-zA-Z0-9._:/@-]+$")
            if len(rest) > 200 or not _model_name_re.match(rest):
                logger.warning("Invalid model name rejected: %r", rest[:80])
                console.print(
                    "[red]Invalid model name. Use alphanumeric, dots, colons, slashes, hyphens (max 200 chars).[/]"
                )
                return "help"
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
            logger.debug("Failed to clear state: %s", e)
            console.print("[dim]Failed to clear state.[/]")
        return "help"
    if cmd == "/plan":
        from .state import update_state

        update_state(
            current_plan=rest.strip() or "No plan entered.", completed_steps=[]
        )
        console.print("[dim]Plan updated. Use /continue to work on it.[/]")
        return "help"
    if cmd == "/continue":
        from .state import get_state as _get_state

        s = _get_state()
        plan = s.get("current_plan") or ""
        steps = s.get("completed_steps") or []
        if not plan.strip():
            console.print("[dim]No plan set. Use /plan <your plan> first.[/]")
            return "help"
        if steps:
            prompt = (
                f"Current plan:\n{plan}\n\nCompleted steps so far:\n"
                + "\n".join(f"  - {x}" for x in steps[-15:] if x)
                + "\n\nWhat should we do next?"
            )
        else:
            prompt = f"Current plan:\n{plan}\n\nWhat should we do next?"
        return ("run_prompt", prompt)
    if cmd == "/rate":
        from .state import append_feedback

        r = rest.strip().lower()
        if r in ("good", "great", "yes", "+", "1"):
            append_feedback("rating", 1, "user rated last reply positively")
            console.print("[dim]Thanks, feedback recorded.[/]")
        elif r in ("bad", "no", "poor", "-", "0"):
            append_feedback("rating", -1, "user rated last reply negatively")
            console.print("[dim]Thanks, feedback recorded.[/]")
        else:
            console.print("[dim]Use /rate good or /rate bad.[/]")
        return "help"
    if cmd == "/query_docs":
        q = rest.strip() or "documentation and usage"
        prompt = f"Search the codebase for documentation relevant to: {q}. Use search_codebase or get_relevant_files (e.g. *.md, README). Summarize what you find."
        return ("run_prompt", prompt)
    if cmd == "/multi":
        goal = (
            rest.strip() or "Use the current context to decide the next concrete task."
        )
        return ("run_multi", goal)
    if cmd == "/kg_add":
        return ("kg_add", rest.strip())
    if cmd == "/kg_query":
        return ("kg_query", rest.strip())
    if cmd == "/rag_index":
        return ("rag_index", rest.strip())
    if cmd == "/rag_query":
        q = rest.strip()
        if not q:
            console.print("[dim]Usage: /rag_query <query>[/]")
            return "help"
        from .rag import query_local_rag

        rows = query_local_rag(q, max_results=6)
        if not rows:
            console.print("[dim]No RAG matches. Run /rag_index first.[/]")
            return "help"
        ctx_lines = []
        for i, r in enumerate(rows, 1):
            path = r.get("path", "")
            score = r.get("score", 0)
            snippet = str(r.get("snippet", "")).strip()
            ctx_lines.append(f"[{i}] {path} (score={score})\n{snippet}")
        context = "\n\n".join(ctx_lines)
        prompt = (
            f"Use the retrieved local context below to answer: {q}\n\n"
            f"Retrieved context:\n\n{context}\n\n"
            "If context is insufficient, say what else to index or inspect."
        )
        return ("run_prompt", prompt)
    if cmd == "/copy":
        return ("copy_last", "")
    if cmd == "/trace":
        return ("set_trace_filter", rest.strip())
    if cmd == "/compact":
        return ("set_compact_mode", rest.strip().lower())
    if cmd == "/summary":
        try:
            n = int(rest) if rest else 5
            n = max(1, min(n, 50))
        except ValueError:
            n = 5
        return ("run_summary", n)
    if cmd == "/auto":
        if autonomous_ref is not None:
            autonomous_ref[0] = not autonomous_ref[0]
            console.print(
                f"[dim]Autonomous mode: {'on' if autonomous_ref[0] else 'off'} (no per-tool confirm, more rounds)[/]"
            )
        return "help"
    if cmd == "/commands":
        from rich.panel import Panel as RichPanel
        from rich.table import Table

        table = Table(title="Slash commands")
        table.add_column("Command", style="cyan")
        table.add_column("Description", style="dim")
        for name, desc, _ in custom_commands or []:
            table.add_row(name, desc)
        if custom_commands:
            console.print(
                RichPanel(table, title="Custom (from commands.md)", border_style="dim")
            )
        console.print("[dim]Built-in: /help for full list.[/]")
        return "help"
    if cmd == "/sessions":
        try:
            from .sessions import list_sessions

            rows = list_sessions(limit=20)
            if not rows:
                console.print("[dim]No sessions yet.[/]")
                return "help"
            from rich.table import Table as T

            t = T(title="Recent sessions")
            t.add_column("Id", style="dim")
            t.add_column("Title", style="cyan")
            t.add_column("Updated", style="dim")
            t.add_column("Messages", justify="right")
            for s in rows:
                uid = (s.get("id") or "")[:8]
                title = (s.get("title") or "")[:40] or "(no title)"
                updated = (s.get("updated_at") or "")[:19]
                count = s.get("message_count", 0)
                t.add_row(uid, title, updated, str(count))
            console.print(t)
            console.print("[dim]Use /resume <id> to load a session.[/]")
        except Exception as e:
            logger.debug("Sessions listing failed: %s", e)
            console.print("[dim]Could not list sessions.[/]")
        return "help"
    if cmd == "/search":
        try:
            from .sessions import search_sessions

            rows = search_sessions(rest.strip() or "", limit=20)
            if not rows:
                console.print("[dim]No matching sessions.[/]")
                return "help"
            from rich.table import Table as T

            t = T(title="Sessions matching query")
            t.add_column("Id", style="dim")
            t.add_column("Title", style="cyan")
            t.add_column("Updated", style="dim")
            t.add_column("Messages", justify="right")
            for s in rows:
                uid = (s.get("id") or "")[:8]
                title = (s.get("title") or "")[:40] or "(no title)"
                updated = (s.get("updated_at") or "")[:19]
                count = s.get("message_count", 0)
                t.add_row(uid, title, updated, str(count))
            console.print(t)
            console.print("[dim]Use /resume <id> to load a session.[/]")
        except Exception as e:
            logger.debug("Session search failed: %s", e)
            console.print("[dim]Search failed.[/]")
        return "help"
    if cmd == "/resume":
        rid = rest.strip()
        if not rid:
            console.print("[dim]Usage: /resume <session-id> (use /sessions to list)[/]")
            return "help"
        try:
            from .sessions import load_session, list_sessions

            # Allow short id prefix
            if len(rid) < 32:
                for s in list_sessions(limit=100):
                    if (s.get("id") or "").startswith(rid) or rid in (
                        s.get("id") or ""
                    ):
                        rid = s["id"]
                        break
            msgs = load_session(rid)
            if msgs is None:
                console.print("[dim]Session not found.[/]")
                return "help"
            history.clear()
            message_history.clear()
            for m in msgs:
                role = m.get("role", "user")
                content = m.get("content", "")
                message_history.append({"role": role, "content": content})
                if role == "user":
                    history.append(("user", content))
                else:
                    history.append(("assistant", content))
            if session_ref is not None:
                session_ref[0] = rid
                _set_apple_fm_session_key(session_ref[0])
            console.print(
                f"[dim]Resumed session {rid[:8]}... ({len(msgs)} messages)[/]"
            )
            return ("resume_session", rid)
        except Exception as e:
            logger.debug("Session resume failed: %s", e)
            console.print("[dim]Resume failed.[/]")
        return "help"
    if cmd == "/palette":
        console.print("[bold]Command palette:[/]")
        console.print("  1) Refactor: rename symbol")
        console.print("  2) Build symbol index")
        console.print("  3) Build repo map")
        choice = input("Choose number (or empty to cancel): ").strip()
        if choice == "1":
            return ("run_prompt", "/refactor rename")
        if choice == "2":
            return ("run_prompt", "/refactor index")
        if choice == "3":
            return ("run_prompt", "/refactor repo-map")
        return "help"
    if cmd == "/refactor":
        sub = rest.strip().split()
        if not sub:
            console.print(
                "[dim]Usage: /refactor rename <old> <new> | /refactor index|refresh | /refactor repo-map | /refactor extract <file> <start-end> <name> | /refactor move <src> <symbol> <dst>[/]"
            )
            return "help"
        if sub[0] in ("index", "refresh"):
            try:
                from .symbol_index import build_symbol_index

                info = build_symbol_index(workspace_root or os.getcwd())
                console.print(
                    f"[dim]Indexed symbols: {info.get('symbols')} refs: {info.get('references')}[/]"
                )
                _reset_symbol_cache()
            except Exception as e:
                logger.debug("Symbol index failed: %s", e)
                console.print("[dim]Symbol index failed.[/]")
            return "help"
        if sub[0] == "repo-map":
            try:
                from .repo_map import write_repo_map

                out = write_repo_map(
                    workspace_root or os.getcwd(),
                    str(
                        Path(workspace_root or os.getcwd())
                        / ".ollamacode"
                        / "repo_map.md"
                    ),
                )
                console.print(f"[dim]Repo map written: {out}[/]")
            except Exception as e:
                logger.debug("Repo map failed: %s", e)
                console.print("[dim]Repo map failed.[/]")
            return "help"
        if sub[0] == "rename":
            if len(sub) < 3:
                old = input("Old symbol: ").strip()
                new = input("New symbol: ").strip()
            else:
                old, new = sub[1], sub[2]
            if not old or not new:
                console.print("[dim]Symbol names required.[/]")
                return "help"
            from .refactor import rename_symbol, save_last_refactor

            edits = rename_symbol(workspace_root or os.getcwd(), old, new)
            if not edits:
                console.print("[dim]No changes found.[/]")
                return "help"
            console.print(format_edits_diff(edits, workspace_root or os.getcwd()))
            ans = input("Apply these edits? [y/N]: ").strip().lower()
            if ans in ("y", "yes"):
                n = apply_edits(edits, workspace_root or os.getcwd())
                console.print(f"[dim]Applied {n} edit(s).[/]")
                # Save diff for rollback (best-effort)
                for e in edits:
                    if isinstance(e.get("newText"), str) and "\n@@ " in str(
                        e.get("newText")
                    ):
                        save_last_refactor(str(e.get("newText")))
            return "help"
        if sub[0] == "rollback":
            from .refactor import rollback_last_refactor

            n = rollback_last_refactor(workspace_root or os.getcwd())
            if n > 0:
                console.print(f"[dim]Rolled back {n} file(s).[/]")
            else:
                console.print("[dim]No rollback applied.[/]")
            return "help"
        if sub[0] == "extract":
            # Usage: /refactor extract path start-end name
            if len(sub) < 4:
                file_path = input("File path: ").strip()
                range_spec = input("Line range (start-end): ").strip()
                new_name = input("New function name: ").strip()
            else:
                file_path, range_spec, new_name = sub[1], sub[2], sub[3]
            if not file_path or not range_spec or not new_name:
                console.print("[dim]Missing args for extract.[/]")
                return "help"
            try:
                start_s, end_s = re.split(r"[-:]", range_spec)
                start_line = int(start_s.strip())
                end_line = int(end_s.strip())
            except Exception:
                console.print("[dim]Invalid range. Use START-END.[/]")
                return "help"
            from .refactor import extract_function

            target = str(Path(workspace_root or os.getcwd()) / file_path)
            out = extract_function(target, start_line, end_line, new_name)
            if not out:
                console.print("[dim]Extract failed.[/]")
                return "help"
            console.print(f"[dim]Extracted to {out}.[/]")
            return "help"
        if sub[0] == "move":
            # Usage: /refactor move src symbol dst
            if len(sub) < 4:
                src_path = input("Source file: ").strip()
                symbol = input("Symbol name: ").strip()
                dst_path = input("Target file: ").strip()
            else:
                src_path, symbol, dst_path = sub[1], sub[2], sub[3]
            if not src_path or not symbol or not dst_path:
                console.print("[dim]Missing args for move.[/]")
                return "help"
            from .refactor import move_symbol

            src = str(Path(workspace_root or os.getcwd()) / src_path)
            dst = str(Path(workspace_root or os.getcwd()) / dst_path)
            ok = move_symbol(src, symbol, dst)
            if not ok:
                console.print("[dim]Move failed.[/]")
                return "help"
            console.print(f"[dim]Moved {symbol} to {dst_path}.[/]")
            return "help"
        console.print("[dim]Unknown refactor command.[/]")
        return "help"
    if cmd == "/session":
        if session_ref is None:
            console.print("[dim]Session persistence not active.[/]")
            return "help"
        try:
            from .sessions import get_session_info, save_session

            sid = session_ref[0]
            if rest:
                new_title = rest.strip()[:500]
                if len(rest.strip()) > 500:
                    logger.warning("Session title truncated to 500 chars")
                    console.print("[dim]Title truncated to 500 characters.[/]")
                save_session(sid, new_title, message_history, workspace_root)
                console.print(f"[dim]Session title set to: {new_title[:60]}[/]")
                return "help"
            info = get_session_info(sid)
            if not info:
                console.print("[dim]Current session not in DB.[/]")
                return "help"
            console.print(
                f"[dim]Session: {sid[:8]}... | title: {info.get('title') or '(none)'} | "
                f"messages: {info.get('message_count', 0)} | updated: {info.get('updated_at', '')[:19]}[/]"
            )
        except Exception as e:
            logger.debug("Session info failed: %s", e)
            console.print("[dim]Could not retrieve session info.[/]")
        return "help"
    if cmd == "/branch":
        if session_ref is None:
            console.print("[dim]Session persistence not active.[/]")
            return "help"
        try:
            from .sessions import branch_session

            new_id = branch_session(session_ref[0], title=None)
            if new_id is None:
                console.print("[dim]Could not branch (session not found).[/]")
                return "help"
            session_ref[0] = new_id
            _set_apple_fm_session_key(session_ref[0])
            console.print(f"[dim]Branched to new session {new_id[:8]}...[/]")
            return ("branch_session", new_id)
        except Exception as e:
            logger.debug("Branch failed: %s", e)
            console.print("[dim]Branch failed.[/]")
        return "help"
    if cmd == "/checkpoints":
        if session_ref is None:
            console.print("[dim]Session persistence not active.[/]")
            return "help"
        try:
            from .checkpoints import list_checkpoints
            from datetime import datetime
            from rich.table import Table as T

            rows = list_checkpoints(session_ref[0], limit=10)
            if not rows:
                console.print("[dim]No checkpoints yet.[/]")
                return "help"
            t = T(title="Recent checkpoints")
            t.add_column("id")
            t.add_column("time")
            t.add_column("files")
            t.add_column("prompt")
            for r in rows:
                ts = datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d %H:%M")
                t.add_row(
                    r["id"][:8],
                    ts,
                    str(r.get("file_count", "")),
                    (r.get("prompt") or "")[:48],
                )
            console.print(t)
        except Exception as e:
            logger.debug("Checkpoints list failed: %s", e)
            console.print("[dim]Could not list checkpoints.[/]")
        return "help"
    if cmd == "/rewind":
        if session_ref is None:
            console.print("[dim]Session persistence not active.[/]")
            return "help"
        parts = (rest or "").strip().split()
        if not parts:
            console.print("[dim]Usage: /rewind <id|index> [code|conversation|both][/]")
            return "help"
        mode = parts[1].lower() if len(parts) > 1 else "code"
        try:
            from .checkpoints import list_checkpoints, restore_checkpoint

            rows = list_checkpoints(session_ref[0], limit=20)
            if not rows:
                console.print("[dim]No checkpoints yet.[/]")
                return "help"
            ident = parts[0]
            row = None
            if ident.isdigit():
                idx = int(ident) - 1
                if 0 <= idx < len(rows):
                    row = rows[idx]
            else:
                for r in rows:
                    if r["id"].startswith(ident):
                        row = r
                        break
            if row is None:
                console.print("[dim]Checkpoint not found.[/]")
                return "help"
            if mode not in ("code", "conversation", "both"):
                console.print("[dim]Mode must be code, conversation, or both.[/]")
                return "help"
            ans = input("Rewind now? This may overwrite files. [y/N] ").strip().lower()
            if ans not in ("y", "yes"):
                return "help"
            if mode in ("code", "both"):
                modified = restore_checkpoint(row["id"], workspace_root or os.getcwd())
                console.print(f"[dim]Rewound {len(modified)} file(s).[/]")
            if mode in ("conversation", "both"):
                idx = int(row.get("message_index") or 0)
                message_history[:] = message_history[:idx]
                history.clear()
                for m in message_history:
                    role = (m.get("role") or "").strip().lower()
                    if role in ("user", "assistant"):
                        history.append((role, m.get("content") or ""))
                console.print("[dim]Conversation rewound.[/]")
            return "help"
        except Exception as e:
            logger.debug("Rewind failed: %s", e)
            console.print("[dim]Rewind failed.[/]")
        return "help"
    if cmd == "/subagents":
        if not subagents:
            console.print(
                "[dim]No subagents configured. Add subagents to config (name, tools, model?).[/]"
            )
            return "help"
        from rich.table import Table as T

        t = T(title="Subagents")
        t.add_column("Type", style="cyan")
        t.add_column("Tools", style="dim")
        t.add_column("Model", style="dim")
        for s in subagents:
            name = (s.get("name") or "").strip() or "?"
            tools_list = s.get("tools") or []
            tools_str = (
                ", ".join(str(x) for x in tools_list)[:50] if tools_list else "(all)"
            )
            model_str = (s.get("model") or "").strip() or "(same)"
            t.add_row(name, tools_str, model_str)
        console.print(t)
        console.print(
            "[dim]Use /subagent <type> <task> to run a task with that type.[/]"
        )
        return "help"
    if cmd == "/subagent":
        if not subagents:
            console.print("[dim]No subagents configured.[/]")
            return "help"
        parts = rest.split(maxsplit=1)
        stype = (parts[0] or "").strip().lower()
        task = (parts[1] or "").strip() if len(parts) > 1 else ""
        if not stype or not task:
            console.print(
                "[dim]Usage: /subagent <type> <task> (e.g. /subagent linter fix ruff errors)[/]"
            )
            return "help"
        match = next(
            (s for s in subagents if (s.get("name") or "").strip().lower() == stype),
            None,
        )
        if not match:
            console.print(
                f"[dim]Unknown subagent type: {stype}. Use /subagents to list.[/]"
            )
            return "help"
        return ("run_subagent", stype, task)
    if cmd == "/agents":
        # /agents <N> <task> or /agents show <n|all|summary>
        parts = rest.split(maxsplit=1)
        if parts and parts[0].lower() == "show":
            arg = parts[1].strip() if len(parts) > 1 else "summary"
            return ("show_agents", arg)
        parts = rest.split(maxsplit=1)
        n = 3
        task = ""
        if parts:
            if parts[0].isdigit():
                n = int(parts[0])
                task = parts[1] if len(parts) > 1 else ""
            else:
                task = rest.strip()
        n = max(1, min(6, n))
        if not task:
            console.print(
                "[dim]Usage: /agents <N> <task> (e.g. /agents 3 analyze repo)[/]"
            )
            return "help"
        return ("run_agents", n, task)
    if cmd == "/agents_show":
        arg = rest.strip() or "summary"
        return ("show_agents", arg)
    if cmd == "/agents_summary":
        return ("agents_summary", "")
    if cmd == "/listen":
        secs = 5.0
        if rest.strip():
            try:
                secs = float(rest.strip())
            except ValueError:
                secs = 5.0
        return ("voice_in", secs)
    if cmd == "/say":
        if not rest.strip():
            console.print("[dim]Usage: /say <text>[/]")
            return "help"
        return ("voice_out", rest.strip())
    if cmd == "/image":
        # Return special tuple so main loop can set pending_image; we don't have pending_image in slash handler
        rest_stripped = rest.strip()
        path_part = rest_stripped.split(maxsplit=1)[0] if rest_stripped else ""
        msg_part = (
            rest_stripped.split(maxsplit=1)[1]
            if len(rest_stripped.split(maxsplit=1)) > 1
            else ""
        )
        if not path_part:
            # Try clipboard image via pngpaste (macOS). Optional dependency.
            tmp_dir = Path.home() / ".ollamacode" / "clipboard"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / f"clipboard_{int(time.time())}.png"
            try:
                p = subprocess.run(
                    ["pngpaste", str(tmp_path)],
                    capture_output=True,
                    text=True,
                )
                if p.returncode == 0 and tmp_path.exists():
                    console.print(f"[dim]Clipboard image saved: {tmp_path}[/]")
                    return ("set_pending_image", str(tmp_path), msg_part.strip())
            except Exception:
                pass
            console.print(
                "[dim]Usage: /image <path> [message]. Attach image to next message. Tip: install pngpaste for clipboard images.[/]"
            )
            return "help"
        return ("set_pending_image", path_part.strip(), msg_part.strip())
    if cmd in ("/quit", "/exit"):
        return "quit"
    console.print(f"[dim]Unknown command: {cmd}. Use /help[/]")
    return "help"


async def _stream_into_live(
    stream: AsyncIterator[str],
    update_cb: Callable[[str, bool], None],
) -> str:
    """Consume async stream, call update_cb(accumulated, done) on each fragment, return final text."""
    parts: list[str] = []
    async for frag in stream:
        clean_frag = _sanitize_stream_text(frag)
        parts.append(clean_frag)
        # Callback only updates status dict (no render), so a join here is fine;
        # but we defer the join to avoid O(n²) by only joining when _tick reads it.
        update_cb(clean_frag, False)
    final = "".join(parts)
    update_cb(final, True)
    return final


async def _stream_into_console(stream: AsyncIterator[str]) -> str:
    """Stream to stdout without Rich Live; returns final text."""
    chunks: list[str] = []
    spinner = ["|", "/", "-", "\\"]
    wait_started = time.perf_counter()
    showed_wait = False
    stream_iter = stream.__aiter__()
    show_wait = sys.stderr.isatty()
    while True:
        try:
            frag = await asyncio.wait_for(stream_iter.__anext__(), timeout=0.1)
        except asyncio.TimeoutError:
            if show_wait:
                elapsed = max(0.0, time.perf_counter() - wait_started)
                spin = spinner[int(time.monotonic() * 8) % len(spinner)]
                sys.stderr.write(
                    f"\r{spin} Assistant (thinking... {elapsed:.1f}s)".ljust(80)
                )
                sys.stderr.flush()
                showed_wait = True
            continue
        except StopAsyncIteration:
            break
        if showed_wait:
            sys.stderr.write("\r" + (" " * 80) + "\r")
            sys.stderr.flush()
            showed_wait = False
        clean_frag = _sanitize_stream_text(frag)
        chunks.append(clean_frag)
        print(clean_frag, end="", flush=True)
    if showed_wait:
        sys.stderr.write("\r" + (" " * 80) + "\r")
        sys.stderr.flush()
    final = "".join(chunks)
    if final and not final.endswith("\n"):
        print()
    return final


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
    use_reasoning: bool = False,
    prompt_snippets: list[str] | None = None,
    allowed_tools: list[str] | None = None,
    blocked_tools: list[str] | None = None,
    confirm_tool_calls: bool = False,
    code_style: str | None = None,
    planner_model: str | None = None,
    executor_model: str | None = None,
    reviewer_model: str | None = None,
    multi_agent_max_iterations: int = 2,
    multi_agent_require_review: bool = True,
    tui_tool_trace_max: int = 20,
    tui_tool_log_max: int = 8,
    tui_tool_log_chars: int = 160,
    tui_refresh_hz: int = 12,
    memory_auto_context: bool = True,
    memory_kg_max_results: int = 4,
    memory_rag_max_results: int = 4,
    memory_rag_snippet_chars: int = 220,
    autonomous_mode: bool = False,
    subagents: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
    session_title: str | None = None,
    session_history: list[dict[str, Any]] | None = None,
    provider: "Any" = None,
    provider_name: str = "ollama",
) -> None:
    """Run interactive TUI chat: Rich panels with Markdown, slash commands, multi-line input.
    Requires rich: pip install ollamacode[tui]
    """
    try:
        from rich.console import Console, Group
        from rich.live import Live
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.text import Text
    except ImportError as e:
        raise ImportError(
            "TUI requires rich. Install with: pip install ollamacode[tui]"
        ) from e

    # Suppress noisy logs so TUI stays clean.
    for _name in (
        "mcp",
        "mcp.client",
        "mcp.server",
        "mcp.server.lowlevel",
        "mcp.server.lowlevel.server",
        "httpx",
        "urllib3",
    ):
        logger = logging.getLogger(_name)
        logger.setLevel(logging.WARNING)
        logger.propagate = False
        logger.disabled = True

    console = Console()
    _SYSTEM = (
        "You are a coding assistant with full access to the workspace. You are given a list of available tools with their names "
        "and descriptions—use whichever tools fit the task. When the user asks you to run something, check something, or change "
        "something, use the appropriate tool and report the result. When generating code, include docstrings and brief comments where helpful."
    )
    if system_extra:
        _SYSTEM = _SYSTEM + "\n\n" + system_extra
    if use_skills:
        skills_text = load_skills_text(workspace_root, query=None)
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
    from .state import (
        format_feedback_context,
        format_knowledge_context,
        format_past_errors_context,
        format_plan_context,
        format_preferences,
        format_recent_context,
        get_state,
    )

    state = get_state()
    if inject_recent_context and workspace_root:
        from .context import get_branch_summary_one_line

        block = format_recent_context(state, max_files=recent_context_max_files)
        if branch_context:
            branch_line = get_branch_summary_one_line(
                workspace_root, branch_context_base
            )
            if branch_line:
                block = (block + "\n\n" + branch_line) if block else branch_line
        if block:
            _SYSTEM = _SYSTEM + "\n\n--- Recent context ---\n\n" + block
    prefs_block = format_preferences(state)
    if prefs_block:
        _SYSTEM = _SYSTEM + "\n\n--- User preferences ---\n\n" + prefs_block
    plan_block = format_plan_context(state)
    if plan_block:
        _SYSTEM = (
            _SYSTEM + "\n\n--- Plan (use /continue to work on it) ---\n\n" + plan_block
        )
    feedback_block = format_feedback_context(state)
    if feedback_block:
        _SYSTEM = _SYSTEM + "\n\n--- Recent feedback ---\n\n" + feedback_block
    knowledge_block = format_knowledge_context(state)
    if knowledge_block:
        _SYSTEM = _SYSTEM + "\n\n--- " + knowledge_block
    past_errors_block = format_past_errors_context(state, max_entries=5)
    if past_errors_block:
        _SYSTEM = _SYSTEM + "\n\n--- " + past_errors_block
    if use_reasoning:
        _SYSTEM = (
            _SYSTEM
            + "\n\nWhen answering, you may include a brief reasoning or rationale before your conclusion; for code changes, briefly explain the fix."
            + '\n\nOptionally output structured reasoning: <<REASONING>>\n{"steps": ["..."], "conclusion": "..."}\n<<END>> before your reply, or call record_reasoning(steps, conclusion).'
        )
    for snip in prompt_snippets or []:
        if snip and isinstance(snip, str) and snip.strip():
            _SYSTEM = _SYSTEM + "\n\n" + snip.strip()
    if code_style:
        _SYSTEM = (
            _SYSTEM
            + "\n\n--- Code style (follow when generating code) ---\n\n"
            + code_style.strip()
        )

    import fnmatch
    from .bridge import BUILTIN_SERVER_PREFIXES
    from .checkpoints import CheckpointRecorder
    from .hooks import HookManager

    root = workspace_root if workspace_root is not None else os.getcwd()
    # Update workspace root ref so path completion always uses current root.
    if _workspace_root_ref:
        _workspace_root_ref[0] = root
    else:
        _workspace_root_ref.append(root)
    history: list[tuple[str, str]] = []
    message_history: list[dict[str, Any]] = []
    try:
        from .sessions import create_session, load_session, save_session

        session_ref: list[str] = []
        if session_id:
            msgs = (
                session_history
                if session_history is not None
                else load_session(session_id)
            )
            if msgs is None:
                console.print(
                    f"[yellow]Warning:[/] Session not found ({session_id}); starting new session."
                )
            else:
                session_ref = [session_id]
                _set_apple_fm_session_key(session_ref[0])
                message_history = list(msgs)
                for m in message_history:
                    role = (m.get("role") or "").strip().lower()
                    content = m.get("content") or ""
                    if role in ("user", "assistant"):
                        history.append((role, content))
                if session_title:
                    save_session(session_id, session_title, message_history, root)
        if not session_ref:
            session_ref = [create_session(session_title or "", workspace_root=root)]
            _set_apple_fm_session_key(session_ref[0])
    except Exception:
        logger.debug("Session init failed", exc_info=True)
        console.print(
            "[yellow]Warning:[/] Session persistence unavailable — conversations will not be saved."
        )
        session_ref = []
    pending_image_ref: list[tuple[str, str] | None] = [None]
    model_ref = [model]
    autonomous_ref = [autonomous_mode]
    hook_mgr = HookManager(
        workspace_root or os.getcwd(), session_ref[0] if session_ref else None
    )
    checkpoints_enabled = os.environ.get("OLLAMACODE_CHECKPOINTS", "1") != "0"
    recorder_ref: list[CheckpointRecorder | None] = [None]
    current_prompt_ref: list[str | None] = [None]

    def _normalize_tool_name(name: str) -> str:
        n = (name or "").strip()
        if n.startswith("functions::"):
            n = n[len("functions::") :]
        for prefix in BUILTIN_SERVER_PREFIXES:
            if n.startswith(prefix):
                n = n[len(prefix) :]
        return n

    def _tool_paths_from_args(tool_name: str, arguments: dict[str, Any]) -> list[str]:
        base = _normalize_tool_name(tool_name)
        paths: list[str] = []
        if base in ("write_file", "edit_file"):
            p = arguments.get("path")
            if isinstance(p, str) and p:
                paths.append(p)
        elif base == "multi_edit":
            edits = arguments.get("edits")
            if isinstance(edits, list):
                for item in edits:
                    if isinstance(item, dict):
                        p = item.get("path")
                        if isinstance(p, str) and p:
                            paths.append(p)
        return paths

    def _is_allowed_tool(name: str) -> bool:
        if not allowed_tools:
            return False
        base = _normalize_tool_name(name)
        for pat in allowed_tools:
            try:
                if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(base, pat):
                    return True
            except Exception:
                if pat == name or pat == base:
                    return True
        return False

    def _maybe_checkpoint(prompt: str) -> None:
        if not checkpoints_enabled or not session_ref:
            recorder_ref[0] = None
            return
        recorder_ref[0] = CheckpointRecorder(
            session_ref[0],
            root,
            prompt,
            len(message_history),
        )

    # Set to prevent fire-and-forget asyncio tasks from being GC'd before completion.
    _background_tasks: set[asyncio.Task] = set()

    def _tool_start_cb(n: str, a: dict) -> None:
        r = recorder_ref[0]
        if r is not None:
            for p in _tool_paths_from_args(n, a):
                r.record_pre(p)

    def _tool_end_cb(n: str, a: dict, s: str) -> None:
        task = asyncio.create_task(
            hook_mgr.run_post_tool_use(
                n,
                a,
                s,
                bool(str(s or "").lower().startswith("tool error")),
                user_prompt=current_prompt_ref[0],
            )
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    def effective_confirm() -> bool:
        return confirm_tool_calls and not autonomous_ref[0]

    def max_tool_rounds_eff() -> int:
        return max(max_tool_rounds, 30) if autonomous_ref[0] else max_tool_rounds

    loop = asyncio.get_event_loop()

    ptt_seconds = float(os.environ.get("OLLAMACODE_TUI_PTT_SECONDS", "5"))

    def _voice_meter(level: float) -> None:
        bars = int(max(0.0, min(level, 1.0)) * 20)
        meter = "█" * bars + " " * (20 - bars)
        sys.stderr.write(f"\r[voice] [{meter}]")
        sys.stderr.flush()
        status["voice_level"] = level

    def _voice_meter_done() -> None:
        sys.stderr.write("\r[voice] done                  \n")
        sys.stderr.flush()
        status["voice_level"] = 0.0

    def _ptt_record() -> str:
        from .voice import record_and_transcribe

        sys.stderr.write("\r[voice] listening...\n")
        sys.stderr.flush()
        text = record_and_transcribe(seconds=ptt_seconds, meter_cb=_voice_meter)
        _voice_meter_done()
        return text

    global _tui_voice_ptt_cb
    _tui_voice_ptt_cb = _ptt_record

    def get_input() -> str:
        """Read a single line; Enter sends. Up/Down cycle input history. Tab completes slash commands. Prompt shows current model."""
        _provider_prefix = (
            f"{provider_name}/" if provider_name and provider_name != "ollama" else ""
        )
        _prompt = f"You [{_provider_prefix}{model_ref[0]}]: "
        if _tui_prompt_session is not None:
            return _tui_prompt_session.prompt(_prompt).strip()
        line = input(_prompt)
        if os.environ.get("OLLAMACODE_TUI_MULTILINE", "1") == "1":
            stripped = line.strip()
            if stripped == '"""':
                lines: list[str] = []
                while True:
                    cont = input("... ")
                    if cont.strip() == '"""':
                        break
                    lines.append(cont)
                return "\n".join(lines).strip()
            if line.endswith("\\"):
                lines = [line[:-1]]
                while True:
                    cont = input("... ")
                    if cont.endswith("\\"):
                        lines.append(cont[:-1])
                        continue
                    lines.append(cont)
                    break
                return "\n".join(lines).strip()
        return line.strip()

    def add_input_history(line: str) -> None:
        """Add a submitted line to input history so Up/Down can recall it (readline path only)."""
        if not line or _tui_prompt_session is not None:
            return
        if _readline_available:
            try:
                readline.add_history_entry(line)  # type: ignore[name-defined]
            except Exception:
                pass

    from .commands_loader import load_custom_commands

    _custom_commands_raw = load_custom_commands(root)
    custom_commands_list: list[tuple[str, str, str | None]] = [
        (c.name, c.description, c.prompt_template) for c in _custom_commands_raw
    ]

    def _edit_tool_args_in_editor(arguments: dict, work_root: str) -> dict | None:
        """Write tool arguments JSON to temp file, open $EDITOR, read back. Return parsed dict or None."""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(arguments, f, indent=2)
                path = f.name
        except OSError:
            return None
        try:
            editor = os.environ.get("EDITOR", "nano")
            subprocess.run(
                shlex.split(editor) + [path],
                cwd=work_root,
            )
            raw = Path(path).read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            return None
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        return data if isinstance(data, dict) else None

    def _maybe_apply_edits_tui(response: str) -> None:
        edits = parse_edits(response)
        if not edits:
            return
        console.print("\n[bold]Proposed edits:[/]")
        diff_str = format_edits_diff(edits, root)
        console.print(_highlight_diff(diff_str))
        while True:
            choice = input("Apply edits? [a]ll / [s]elect / [n]: ").strip().lower()
            if choice in ("a", "all", "y", "yes"):
                n = apply_edits(edits, root)
                console.print(f"[dim]Applied {n} edit(s).[/]")
                return
            if choice in ("n", "no", ""):
                return
            if choice in ("s", "select"):
                # If any unified diff edits exist, offer per-hunk selection.
                unified_edits = [
                    e
                    for e in edits
                    if isinstance(e.get("newText"), str)
                    and "\n@@ " in str(e.get("newText"))
                ]
                if unified_edits:
                    for e in unified_edits:
                        diff_text = str(e.get("newText") or "")
                        _interactive_hunk_selector(diff_text, root)
                    return
                selected: list[dict[str, Any]] = []
                seen: set[str] = set()
                for e in edits:
                    path = e.get("path") or ""
                    if path in seen:
                        continue
                    seen.add(path)
                    ans = input(f"Apply edits for {path}? [y/N]: ").strip().lower()
                    if ans in ("y", "yes"):
                        selected.extend([x for x in edits if x.get("path") == path])
                if not selected:
                    console.print("[dim]No edits selected.[/]")
                    return
                n = apply_edits(selected, root)
                console.print(f"[dim]Applied {n} edit(s).[/]")
                return
            console.print("[dim]Choose a/all, s/select, or n.[/]")

    def _interactive_hunk_selector(diff_text: str, workspace: str | Path) -> None:
        """Interactive per-hunk selector with optional prompt_toolkit key bindings."""
        try:
            from .edits import _parse_unified_diff  # type: ignore
        except Exception:
            return
        patches = _parse_unified_diff(diff_text)
        hunks: list[tuple[str, int, dict[str, Any]]] = []
        for p in patches:
            path = p.get("path") or ""
            for i, h in enumerate(p.get("hunks") or []):
                hunks.append((path, i, h))
        if not hunks:
            return
        selected: set[int] = set()
        idx = 0

        def render() -> None:
            console.print(f"\n[bold]Hunk {idx + 1}/{len(hunks)}[/]")
            path, i, h = hunks[idx]
            header = h.get("header") or "@@"
            console.print(f"[dim]{path} #{i} {header}[/]")
            from rich.text import Text

            for tag, payload in (h.get("lines") or [])[:60]:
                prefix = "+" if tag == "+" else "-" if tag == "-" else " "
                style = "green" if tag == "+" else "red" if tag == "-" else "dim"
                t = Text(prefix + payload, style=style)
                console.print(t)
            console.print(f"[dim]Accepted: {len(selected)}/{len(hunks)}[/]")
            console.print(
                "[dim]Commands: j/k (next/prev), y accept, n reject, space toggle, a apply, q quit[/]"
            )

        # prompt_toolkit path (arrow keys)
        try:
            from prompt_toolkit import Application
            from prompt_toolkit.key_binding import KeyBindings
            from prompt_toolkit.layout import Layout
            from prompt_toolkit.layout.controls import FormattedTextControl
            from prompt_toolkit.layout.containers import Window
            from prompt_toolkit.styles import Style

            def _render_text():
                path, i, h = hunks[idx]
                header = h.get("header") or "@@"
                body: list[tuple[str, str]] = []
                for tag, payload in (h.get("lines") or [])[:80]:
                    prefix = "+" if tag == "+" else "-" if tag == "-" else " "
                    style = (
                        "class:add"
                        if tag == "+"
                        else "class:del"
                        if tag == "-"
                        else "class:ctx"
                    )
                    body.append((style, prefix + payload + "\n"))
                header_lines: list[tuple[str, str]] = [
                    ("bold", f"Hunk {idx + 1}/{len(hunks)}\n"),
                    ("", f"{path} #{i} {header}\n\n"),
                ]
                footer = [
                    ("", "\n"),
                    ("class:info", f"Accepted: {len(selected)}/{len(hunks)}\n"),
                    (
                        "class:info",
                        "Up/Down=nav  y=accept  n=reject  Space=toggle  a=apply  q=quit",
                    ),
                ]
                return header_lines + body + footer

            kb = KeyBindings()

            @kb.add("down")
            def _down(event):
                nonlocal idx
                idx = min(len(hunks) - 1, idx + 1)
                event.app.invalidate()

            @kb.add("up")
            def _up(event):
                nonlocal idx
                idx = max(0, idx - 1)
                event.app.invalidate()

            @kb.add(" ")
            def _toggle(event):
                if idx in selected:
                    selected.remove(idx)
                else:
                    selected.add(idx)
                event.app.invalidate()

            @kb.add("y")
            def _accept_one(event):
                nonlocal idx
                if idx not in selected:
                    selected.add(idx)
                idx = min(len(hunks) - 1, idx + 1)
                event.app.invalidate()

            @kb.add("n")
            def _reject_one(event):
                nonlocal idx
                if idx in selected:
                    selected.remove(idx)
                idx = min(len(hunks) - 1, idx + 1)
                event.app.invalidate()

            @kb.add("a")
            def _apply(event):
                event.app.exit(result="apply")

            @kb.add("q")
            def _quit(event):
                event.app.exit(result="quit")

            control = FormattedTextControl(_render_text, focusable=True)
            window = Window(content=control)
            style = Style.from_dict(
                {
                    "add": "ansigreen",
                    "del": "ansired",
                    "ctx": "ansibrightblack",
                    "info": "ansicyan",
                }
            )
            app = Application(
                layout=Layout(window),
                key_bindings=kb,
                full_screen=False,
                style=style,
            )
            result = app.run()
            if result == "quit":
                return
        except Exception:
            # fallback to text prompts
            while True:
                render()
                cmd = input("> ").strip().lower()
                if cmd in ("j", "down"):
                    idx = min(len(hunks) - 1, idx + 1)
                elif cmd in ("k", "up", "p"):
                    idx = max(0, idx - 1)
                elif cmd in (" ", "t"):
                    if idx in selected:
                        selected.remove(idx)
                    else:
                        selected.add(idx)
                elif cmd in ("y", "yes"):
                    selected.add(idx)
                    idx = min(len(hunks) - 1, idx + 1)
                elif cmd in ("n", "no"):
                    if idx in selected:
                        selected.remove(idx)
                    idx = min(len(hunks) - 1, idx + 1)
                elif cmd in ("a", "apply"):
                    break
                elif cmd in ("q", "quit", ""):
                    return
        if not selected:
            console.print("[dim]No hunks selected.[/]")
            return
        want = set(selected)

        def _include(path, hidx, h):
            return hidx in want

        n = apply_unified_diff_filtered(diff_text, Path(workspace), _include)
        console.print(f"[dim]Applied {n} hunk(s).[/]")

    status: dict[str, Any] = {
        "phase": "idle",
        "has_output": False,
        "done": False,
        "waiting_since": None,
        "tool": "",
        "accumulated": "",
        "last_user": "",
        "token_count": 0,
        "agents_running": 0,
        "agents_done": 0,
        "agents": [],
        "agent_outputs": [],
        "agent_roles": [],
        "agent_task": "",
    }

    def _status_update(**kwargs: Any) -> None:
        """Batch-update status dict for consistent multi-key mutations."""
        status.update(kwargs)

    tool_trace: deque[str] = deque(maxlen=tui_tool_trace_max)
    tool_log: deque[str] = deque(maxlen=tui_tool_log_max)
    trace_filter = ""
    compact_mode = console.size.width < 110

    async def _before_tool_call_tui(tool_name: str, arguments: dict):
        """Prompt [y/N/e] per tool with Rich panel when confirm_tool_calls; e = edit args in $EDITOR."""
        modified_by_hook = False
        try:
            decision = await hook_mgr.run_pre_tool_use(
                tool_name, arguments, user_prompt=current_prompt_ref[0]
            )
            if decision and decision.behavior == "deny":
                return ("skip", decision.message or "Blocked by hook.")
            if decision and decision.behavior == "modify" and decision.updated_input:
                arguments = decision.updated_input
                modified_by_hook = True
        except Exception:
            pass
        if _is_allowed_tool(tool_name):
            return ("edit", arguments) if modified_by_hook else "run"
        _status_update(phase="tool", tool=tool_name)
        one_line = _tool_call_one_line(tool_name, arguments)
        args_preview = json.dumps(arguments, indent=2)
        if len(args_preview) > 400:
            args_preview = args_preview[:400] + "\n  ..."
        console.print(
            Panel(
                f"[bold]{tool_name}[/]\n\n[dim]{one_line}[/]\n\n{args_preview}",
                title="Confirm tool call",
                border_style="yellow",
            )
        )
        choice = await loop.run_in_executor(
            None, lambda: input("[y/N/e(dit)]? ").strip().lower() or "n"
        )
        if choice in ("y", "yes"):
            _status_update(
                phase="streaming" if status.get("has_output") else "thinking", tool=""
            )
            return ("edit", arguments) if modified_by_hook else "run"
        if choice in ("n", "no", ""):
            _status_update(
                phase="streaming" if status.get("has_output") else "thinking", tool=""
            )
            return "skip"
        if choice in ("e", "edit"):
            edited = _edit_tool_args_in_editor(arguments, root)
            if edited is not None:
                _status_update(
                    phase="streaming" if status.get("has_output") else "thinking",
                    tool="",
                )
                return ("edit", edited)
            console.print("[dim]Invalid JSON or cancel; running with original args.[/]")
            _status_update(
                phase="streaming" if status.get("has_output") else "thinking", tool=""
            )
            return "run"
        console.print("[dim]Choose y (run), N (skip), or e (edit).[/]")
        return await _before_tool_call_tui(tool_name, arguments)  # re-prompt

    queue_inputs_requested = os.environ.get("OLLAMACODE_TUI_QUEUE_INPUT", "1") != "0"
    # prompt_toolkit does not behave well with background queued reads while Rich Live is rendering.
    # Keep queueing for plain input(), but force foreground input for prompt_toolkit sessions.
    queue_inputs = queue_inputs_requested and _tui_prompt_session is None
    use_live = (
        console.is_terminal and os.environ.get("OLLAMACODE_TUI_SIMPLE", "0") != "1"
    )
    auto_agents = os.environ.get("OLLAMACODE_TUI_AUTO_AGENTS", "0") == "1"
    auto_agents_plan = os.environ.get("OLLAMACODE_TUI_AUTO_AGENTS_PLAN", "1") != "0"
    planner_model_override = os.environ.get("OLLAMACODE_TUI_PLANNER_MODEL", "").strip()
    planner_model = planner_model_override or model_ref[0]
    plan_exec_verify = os.environ.get("OLLAMACODE_PLAN_EXECUTE_VERIFY", "0") == "1"
    plan_model = os.environ.get("OLLAMACODE_PLAN_MODEL", "").strip() or model_ref[0]
    verify_model = os.environ.get("OLLAMACODE_VERIFY_MODEL", "").strip() or model_ref[0]
    voice_out_enabled = os.environ.get("OLLAMACODE_TUI_VOICE_OUT", "0") == "1"
    auto_agents_summary = os.environ.get("OLLAMACODE_TUI_AGENTS_SUMMARY", "1") != "0"
    summary_model_override = os.environ.get(
        "OLLAMACODE_TUI_AGENTS_SUMMARY_MODEL", ""
    ).strip()
    summary_model = summary_model_override or model_ref[0]
    agents_synthesis = os.environ.get("OLLAMACODE_TUI_AGENTS_SYNTHESIS", "1") != "0"
    synthesis_model = (
        os.environ.get("OLLAMACODE_TUI_AGENTS_SYNTHESIS_MODEL", "").strip()
        or model_ref[0]
    )
    agents_structured = os.environ.get("OLLAMACODE_AGENTS_STRUCTURED", "1") != "0"
    router_enabled = os.environ.get("OLLAMACODE_ROUTER", "0") == "1"
    router_fast = os.environ.get("OLLAMACODE_MODEL_FAST", "").strip() or model_ref[0]
    router_strong = (
        os.environ.get("OLLAMACODE_MODEL_STRONG", "").strip() or model_ref[0]
    )
    try:
        auto_agents_n = int(os.environ.get("OLLAMACODE_TUI_AUTO_AGENTS_N", "3"))
    except ValueError:
        auto_agents_n = 3
    auto_agents_n = max(2, min(6, auto_agents_n))
    try:
        agents_preview_lines = int(
            os.environ.get("OLLAMACODE_TUI_AGENTS_PREVIEW_LINES", "10")
        )
    except ValueError:
        agents_preview_lines = 10
    agents_preview_lines = max(3, min(30, agents_preview_lines))
    _theme = _get_theme()
    console.print(
        Panel(
            "[bold]OllamaCode TUI[/] – [dim]/help[/] for commands. Up/Down = history. Empty or Ctrl+C to exit.",
            title="OllamaCode",
            border_style=_theme["panel_main"],
        )
    )
    if show_semantic_hint:
        console.print(
            "[dim]Tip: For semantic codebase search, add a semantic MCP server to config. See docs/MCP_SERVERS.md.[/]"
        )

    # Producer: optionally reads input continuously; consumer: processes one command at a time.
    input_queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def input_producer() -> None:
        while True:
            try:
                line = await loop.run_in_executor(None, get_input)
            except (EOFError, KeyboardInterrupt, asyncio.CancelledError):
                input_queue.put_nowait(None)
                return
            if line:
                input_queue.put_nowait(line)
                # Don't show prompt again for /quit or /exit so user doesn't have to type it twice.
                if line.strip().lower() in ("/quit", "/exit"):
                    input_queue.put_nowait(None)
                    return

    producer_task: asyncio.Task[None] | None = None

    async def get_next_line() -> str | None:
        """Get next user line from queue (None = quit)."""
        if not queue_inputs:
            try:
                return await loop.run_in_executor(None, get_input)
            except (EOFError, KeyboardInterrupt, asyncio.CancelledError):
                return None
        return await input_queue.get()

    def _should_auto_agents(text: str) -> bool:
        if not text or text.strip().startswith("/"):
            return False
        words = len(text.split())
        if words >= 10:
            return True
        lowered = text.lower()
        keywords = (
            "analyze",
            "audit",
            "review",
            "regression",
            "optimize",
            "refactor",
            "plan",
            "roadmap",
            "deep",
            "thorough",
        )
        return any(k in lowered for k in keywords)

    def _should_plan_exec_verify(text: str) -> bool:
        if not text or text.strip().startswith("/"):
            return False
        words = len(text.split())
        if words >= 12:
            return True
        lowered = text.lower()
        keywords = (
            "analyze",
            "audit",
            "review",
            "regression",
            "optimize",
            "refactor",
            "plan",
            "roadmap",
            "deep",
            "thorough",
            "migrate",
            "redesign",
            "performance",
            "security",
        )
        return any(k in lowered for k in keywords)

    def _route_model(text: str, default_model: str) -> str:
        if not router_enabled:
            return default_model
        words = len(text.split())
        lowered = text.lower()
        heavy = (
            "analyze",
            "audit",
            "review",
            "refactor",
            "regression",
            "optimize",
            "roadmap",
            "security",
            "performance",
            "migrate",
        )
        if words > 40 or any(k in lowered for k in heavy):
            return router_strong
        if words <= 6:
            return router_fast
        return default_model

    async def _plan_agents(task: str) -> tuple[int, list[dict[str, str]]]:
        """Lightweight planner to decide agent count and focus areas."""
        prompt = (
            "You are a planner. Decide how many parallel agents (2-6) should be used and their focus.\n"
            "Return JSON only with keys: count (int), agents (list of {name, focus}).\n"
            f"Task: {task}"
        )
        try:
            if session is not None:
                text = await run_agent_loop(
                    session,
                    planner_model,
                    prompt,
                    system_prompt=_SYSTEM,
                    message_history=[],
                    max_tool_rounds=1,
                    confirm_tool_calls=False,
                    provider=provider,
                )
            else:
                text = await run_agent_loop_no_mcp(
                    planner_model,
                    prompt,
                    system_prompt=_SYSTEM,
                    message_history=[],
                    provider=provider,
                )
            raw = text.strip().splitlines()[-1].strip()
            data = json.loads(raw)
            count = int(data.get("count", auto_agents_n))
            agents = data.get("agents") or []
            if not isinstance(agents, list):
                agents = []
            count = max(2, min(6, count))
            roles: list[dict[str, str]] = []
            for a in agents[:count]:
                if isinstance(a, dict):
                    name = str(a.get("name", "")).strip() or "Agent"
                    focus = str(a.get("focus", "")).strip()
                    roles.append({"name": name, "focus": focus})
            if not roles:
                roles = [
                    {"name": "Agent A", "focus": "Architecture & risks"},
                    {"name": "Agent B", "focus": "Tests & regressions"},
                    {"name": "Agent C", "focus": "UX & TUI polish"},
                ][:count]
            return count, roles
        except Exception as e:
            logger.debug("Planner failed, using defaults: %s", e)
            console.print(
                "[dim]Planner returned invalid response, using default agents.[/]"
            )
            roles = [
                {"name": "Agent A", "focus": "Architecture & risks"},
                {"name": "Agent B", "focus": "Tests & regressions"},
                {"name": "Agent C", "focus": "UX & TUI polish"},
            ][:auto_agents_n]
            return auto_agents_n, roles

    try:
        if queue_inputs:
            producer_task = asyncio.create_task(input_producer())
        while True:
            line = await get_next_line()
            if line is None:
                break
            add_input_history(line)

            result = None
            if auto_agents and _should_auto_agents(line):
                if auto_agents_plan:
                    n_plan, roles_plan = await _plan_agents(line)
                    result = ("run_agents", n_plan, line, roles_plan)
                    console.print(f"[dim]Auto agents: {n_plan} (planned)[/]")
                else:
                    result = ("run_agents", auto_agents_n, line, None)
                    console.print(f"[dim]Auto agents: {auto_agents_n}[/]")
            else:
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
                    autonomous_ref=autonomous_ref,
                    custom_commands=custom_commands_list,
                    session_ref=session_ref if session_ref else None,
                    subagents=subagents,
                )
            if result == "quit":
                break
            if isinstance(result, tuple) and result[0] in (
                "resume_session",
                "branch_session",
            ):
                if session_ref:
                    hook_mgr = HookManager(root, session_ref[0])
                continue
            if (
                isinstance(result, tuple)
                and len(result) == 3
                and result[0] == "set_pending_image"
            ):
                pending_image_ref[0] = (cast(str, result[1]), cast(str, result[2]))
                console.print(
                    f"[dim]Image attached: {result[1]} (message for next turn: {result[2][:40] or '(none)'}...)[/]"
                )
                continue
            if (
                isinstance(result, tuple)
                and len(result) == 3
                and result[0] == "run_subagent"
            ):
                _, stype, task = cast(tuple[Any, str, str], result)
                match = next(
                    (
                        s
                        for s in (subagents or [])
                        if (s.get("name") or "").strip().lower() == stype
                    ),
                    None,
                )
                if not match:
                    continue
                tools = list(match.get("tools") or [])
                sub_model = (match.get("model") or "").strip() or model_ref[0]
                console.print(f"[dim]Running subagent '{stype}'...[/]")
                if session is not None:
                    out = await run_agent_loop(
                        session,
                        sub_model,
                        task,
                        system_prompt=_SYSTEM,
                        message_history=[],
                        max_tool_rounds=max_tool_rounds,
                        allowed_tools=tools if tools else None,
                        blocked_tools=None,
                        confirm_tool_calls=effective_confirm(),
                        before_tool_call=_before_tool_call_tui
                        if effective_confirm()
                        else None,
                        provider=provider,
                    )
                else:
                    out = await run_agent_loop_no_mcp(
                        sub_model,
                        task,
                        system_prompt=_SYSTEM,
                        message_history=[],
                        provider=provider,
                    )
                history.append((task, out))
                message_history.append({"role": "user", "content": task})
                message_history.append({"role": "assistant", "content": out})
                if session_ref:
                    try:
                        from .sessions import save_session

                        save_session(session_ref[0], None, message_history, root)
                    except Exception:
                        logger.debug("Session save failed", exc_info=True)
                console.print(
                    Panel(
                        Markdown(out), title=f"Subagent ({stype})", border_style="cyan"
                    )
                )
                continue
            if result is not None:
                if (
                    isinstance(result, tuple)
                    and len(result) >= 2
                    and result[0] == "run_prompt"
                ):
                    line = cast(str, result[1])
                elif (
                    isinstance(result, tuple)
                    and len(result) >= 2
                    and result[0] == "run_multi"
                ):
                    prompt = cast(str, result[1])
                    multi_memory = (
                        build_dynamic_memory_context(
                            prompt,
                            kg_max_results=memory_kg_max_results,
                            rag_max_results=memory_rag_max_results,
                            rag_snippet_chars=memory_rag_snippet_chars,
                        )
                        if memory_auto_context
                        else ""
                    )
                    multi_system = (
                        _SYSTEM
                        + "\n\n--- Retrieved memory (query-specific) ---\n\n"
                        + multi_memory
                        if multi_memory
                        else _SYSTEM
                    )
                    out = await run_multi_agent(
                        session,
                        model_ref[0],
                        prompt,
                        system_prompt=multi_system,
                        max_messages=max_messages,
                        max_tool_result_chars=max_tool_result_chars,
                        allowed_tools=allowed_tools,
                        blocked_tools=blocked_tools,
                        confirm_tool_calls=effective_confirm(),
                        before_tool_call=_before_tool_call_tui
                        if effective_confirm()
                        else None,
                        planner_model=planner_model,
                        executor_model=executor_model,
                        reviewer_model=reviewer_model,
                        max_iterations=multi_agent_max_iterations,
                        require_review=multi_agent_require_review,
                    )
                    if out.plan:
                        console.print("[dim]Plan:[/]")
                        console.print(out.plan)
                    if out.review:
                        console.print(
                            f"[dim]Review approved: {out.review.get('approved')}[/]"
                        )
                    history.append((prompt, out.content))
                    message_history.append({"role": "user", "content": prompt})
                    message_history.append(
                        {"role": "assistant", "content": out.content}
                    )
                    if session_ref:
                        try:
                            from .sessions import save_session

                            save_session(session_ref[0], None, message_history, root)
                        except Exception:
                            logger.debug("Session save failed", exc_info=True)
                    console.print(
                        Panel(
                            Markdown(out.content),
                            title="Assistant",
                            border_style="cyan",
                        )
                    )
                    continue
                elif (
                    isinstance(result, tuple)
                    and len(result) >= 3
                    and result[0] == "run_agents"
                ):
                    roles = None
                    if len(result) >= 4:
                        roles = cast(list[dict[str, str]] | None, result[3])
                    _, n_agents, task = cast(tuple[Any, int, str], result[:3])
                    n_agents = max(1, min(6, n_agents))
                    if not roles:
                        roles = [
                            {"name": "Agent A", "focus": "Architecture & risks"},
                            {"name": "Agent B", "focus": "Tests & regressions"},
                            {"name": "Agent C", "focus": "UX & TUI polish"},
                        ][:n_agents]
                    status["agents_running"] = n_agents
                    status["agents_done"] = 0
                    status["agent_task"] = task
                    status["agent_roles"] = roles
                    status["agent_outputs"] = []
                    status["agents"] = [
                        {
                            "name": r.get("name", f"Agent {i + 1}"),
                            "state": "running",
                            "note": r.get("focus", ""),
                        }
                        for i, r in enumerate(roles)
                    ]
                    console.print(f"[dim]Running {n_agents} agents...[/]")

                    async def _run_one(idx: int) -> str:
                        assert roles is not None
                        role = (
                            roles[idx]
                            if idx < len(roles)
                            else {"name": f"Agent {idx + 1}", "focus": ""}
                        )
                        label = role.get("name", f"Agent {idx + 1}")
                        focus = role.get("focus", "")
                        prompt = f"{task}\n\n(You are {label}. Focus: {focus})"
                        if agents_structured:
                            prompt += (
                                "\n\nReturn Markdown with sections:\n"
                                "## Summary\n## Findings\n## Risks\n## Next Steps"
                            )
                        if session is not None:
                            return await run_agent_loop(
                                session,
                                model_ref[0],
                                prompt,
                                system_prompt=_SYSTEM,
                                message_history=[],
                                max_tool_rounds=max_tool_rounds,
                                allowed_tools=allowed_tools,
                                blocked_tools=blocked_tools,
                                confirm_tool_calls=False,
                                provider=provider,
                            )
                        return await run_agent_loop_no_mcp(
                            model_ref[0],
                            prompt,
                            system_prompt=_SYSTEM,
                            message_history=[],
                            provider=provider,
                        )

                    async def _wrap(i: int) -> str:
                        try:
                            return await _run_one(i)
                        finally:
                            status["agents_done"] += 1
                            if i < len(status["agents"]):
                                status["agents"][i]["state"] = "done"

                    tasks = [asyncio.create_task(_wrap(i)) for i in range(n_agents)]
                    results = await asyncio.gather(*tasks)
                    status["agents_running"] = 0
                    status["agent_outputs"] = results
                    summary_text = ""
                    if auto_agents_summary:
                        try:
                            summary_prompt = (
                                "Summarize the agent outputs into a short executive summary (max 6 bullets) "
                                "and list key action items (max 5). Respond in Markdown only.\n\n"
                                f"Task: {task}\n\n"
                                + "\n\n".join(
                                    [
                                        f"[{roles[i].get('name', f'Agent {i + 1}')}] {results[i]}"
                                        for i in range(min(len(results), len(roles)))
                                    ]
                                )
                            )
                            if session is not None:
                                summary_text = await run_agent_loop(
                                    session,
                                    summary_model,
                                    summary_prompt,
                                    system_prompt=_SYSTEM,
                                    message_history=[],
                                    max_tool_rounds=1,
                                    confirm_tool_calls=False,
                                    provider=provider,
                                )
                            else:
                                summary_text = await run_agent_loop_no_mcp(
                                    summary_model,
                                    summary_prompt,
                                    system_prompt=_SYSTEM,
                                    message_history=[],
                                    provider=provider,
                                )
                            summary_text = summary_text.strip()
                        except Exception:
                            summary_text = ""
                    synthesis_text = ""
                    if agents_synthesis:
                        try:
                            synthesis_prompt = (
                                "Synthesize the agent outputs into a single final answer. "
                                "Explicitly reconcile conflicts. Respond in Markdown with sections:\n"
                                "## Final Answer\n"
                                "## Conflicts Resolved (use 'None' if no conflicts)\n\n"
                                f"Task: {task}\n\n"
                                + "\n\n".join(
                                    [
                                        f"[{roles[i].get('name', f'Agent {i + 1}')}] {results[i]}"
                                        for i in range(min(len(results), len(roles)))
                                    ]
                                )
                            )
                            if session is not None:
                                synthesis_text = await run_agent_loop(
                                    session,
                                    synthesis_model,
                                    synthesis_prompt,
                                    system_prompt=_SYSTEM,
                                    message_history=[],
                                    max_tool_rounds=1,
                                    confirm_tool_calls=False,
                                    provider=provider,
                                )
                            else:
                                synthesis_text = await run_agent_loop_no_mcp(
                                    synthesis_model,
                                    synthesis_prompt,
                                    system_prompt=_SYSTEM,
                                    message_history=[],
                                    provider=provider,
                                )
                            synthesis_text = synthesis_text.strip()
                        except Exception:
                            synthesis_text = ""
                    parts = [f"## Agents ({n_agents})"]
                    if summary_text:
                        parts.insert(0, f"## Executive Summary\n\n{summary_text}")
                    if synthesis_text:
                        parts.insert(0, f"## Final Synthesis\n\n{synthesis_text}")
                    for i, text in enumerate(results, 1):
                        name = (
                            roles[i - 1].get("name", f"Agent {i}")
                            if roles
                            else f"Agent {i}"
                        )
                        focus = roles[i - 1].get("focus", "") if roles else ""
                        header = f"### {name}"
                        if focus:
                            header += f" — {focus}"
                        preview_lines = text.strip().splitlines()
                        preview = "\n".join(preview_lines[:agents_preview_lines])
                        parts.append(f"{header}\n\n{preview}")
                        if len(preview_lines) > agents_preview_lines:
                            parts.append(
                                f"[dim]… use /agents_show {i} for full output[/]"
                            )
                    combined = "\n\n".join(parts)
                    history.append((task, combined))
                    message_history.append({"role": "user", "content": task})
                    message_history.append({"role": "assistant", "content": combined})
                    if session_ref:
                        try:
                            from .sessions import save_session

                            save_session(session_ref[0], None, message_history, root)
                        except Exception:
                            logger.debug("Session save failed", exc_info=True)
                    console.print(
                        Panel(Markdown(combined), title="Agents", border_style="cyan")
                    )
                    continue
                elif (
                    isinstance(result, tuple)
                    and len(result) >= 2
                    and result[0] == "copy_last"
                ):
                    last = ""
                    for role, text in reversed(history):
                        if role == "assistant":
                            last = text
                            break
                    if not last:
                        console.print("[dim]No assistant reply to copy yet.[/]")
                    else:
                        ok = await loop.run_in_executor(None, _copy_text_sync, last)
                        if ok:
                            console.print(
                                "[dim]Copied last assistant reply to clipboard.[/]"
                            )
                        else:
                            console.print(
                                "[dim]Clipboard unavailable. Install pbcopy/wl-copy/xclip/xsel.[/]"
                            )
                    continue
                elif (
                    isinstance(result, tuple)
                    and len(result) >= 2
                    and result[0] == "show_agents"
                ):
                    arg = cast(str, result[1]).strip().lower()
                    outputs = status.get("agent_outputs") or []
                    roles = status.get("agent_roles") or []
                    task = status.get("agent_task") or ""
                    if not outputs:
                        console.print("[dim]No agent outputs yet.[/]")
                        continue
                    if arg in ("summary", "sum", "s"):
                        lines = ["[bold]Agents (summary)[/]"]
                        if task:
                            lines.append(f"[dim]Task:[/] {task}")
                        for i, out in enumerate(outputs, 1):
                            name = (
                                roles[i - 1].get("name", f"Agent {i}")
                                if i - 1 < len(roles)
                                else f"Agent {i}"
                            )
                            focus = (
                                roles[i - 1].get("focus", "")
                                if i - 1 < len(roles)
                                else ""
                            )
                            header = f"{name}"
                            if focus:
                                header += f" — {focus}"
                            preview = "\n".join(
                                out.strip().splitlines()[:agents_preview_lines]
                            )
                            lines.append(f"[dim]{header}[/]\n{preview}")
                        console.print(
                            Panel(
                                "\n\n".join(lines), title="Agents", border_style="cyan"
                            )
                        )
                        continue
                    if arg in ("all", "*"):
                        combined = []
                        if task:
                            combined.append(f"## Task\n\n{task}")
                        for i, out in enumerate(outputs, 1):
                            name = (
                                roles[i - 1].get("name", f"Agent {i}")
                                if i - 1 < len(roles)
                                else f"Agent {i}"
                            )
                            focus = (
                                roles[i - 1].get("focus", "")
                                if i - 1 < len(roles)
                                else ""
                            )
                            header = f"### {name}"
                            if focus:
                                header += f" — {focus}"
                            combined.append(f"{header}\n\n{out.strip()}")
                        console.print(
                            Panel(
                                Markdown("\n\n".join(combined)),
                                title="Agents",
                                border_style="cyan",
                            )
                        )
                        continue
                    try:
                        idx = int(arg)
                    except ValueError:
                        console.print("[dim]Usage: /agents_show <n|all|summary>[/]")
                        continue
                    if idx < 1 or idx > len(outputs):
                        console.print("[dim]Agent index out of range.[/]")
                        continue
                    name = (
                        roles[idx - 1].get("name", f"Agent {idx}")
                        if idx - 1 < len(roles)
                        else f"Agent {idx}"
                    )
                    focus = (
                        roles[idx - 1].get("focus", "") if idx - 1 < len(roles) else ""
                    )
                    header = f"{name}"
                    if focus:
                        header += f" — {focus}"
                    body = outputs[idx - 1].strip()
                    console.print(
                        Panel(
                            Markdown(f"### {header}\n\n{body}"),
                            title="Agent",
                            border_style="cyan",
                        )
                    )
                    continue
                elif (
                    isinstance(result, tuple)
                    and len(result) >= 1
                    and result[0] == "agents_summary"
                ):
                    outputs = status.get("agent_outputs") or []
                    roles = status.get("agent_roles") or []
                    task = status.get("agent_task") or ""
                    if not outputs:
                        console.print("[dim]No agent outputs yet.[/]")
                        continue
                    try:
                        summary_prompt = (
                            "Summarize the agent outputs into a short executive summary (max 6 bullets) "
                            "and list key action items (max 5). Respond in Markdown only.\n\n"
                            f"Task: {task}\n\n"
                            + "\n\n".join(
                                [
                                    f"[{roles[i].get('name', f'Agent {i + 1}')}] {outputs[i]}"
                                    for i in range(min(len(outputs), len(roles)))
                                ]
                            )
                        )
                        if session is not None:
                            summary_text = await run_agent_loop(
                                session,
                                summary_model,
                                summary_prompt,
                                system_prompt=_SYSTEM,
                                message_history=[],
                                max_tool_rounds=1,
                                confirm_tool_calls=False,
                                provider=provider,
                            )
                        else:
                            summary_text = await run_agent_loop_no_mcp(
                                summary_model,
                                summary_prompt,
                                system_prompt=_SYSTEM,
                                message_history=[],
                                provider=provider,
                            )
                        summary_text = summary_text.strip()
                        console.print(
                            Panel(
                                Markdown(summary_text),
                                title="Executive Summary",
                                border_style="cyan",
                            )
                        )
                    except Exception as e:
                        logger.debug("Summary failed: %s", e)
                        console.print("[dim]Summary failed.[/]")
                    continue
                elif (
                    isinstance(result, tuple)
                    and len(result) >= 2
                    and result[0] == "voice_out"
                ):
                    text = cast(str, result[1])
                    try:
                        from .voice import speak_text

                        speak_text(text)
                    except Exception as e:
                        logger.debug("Voice output failed: %s", e)
                        console.print("[dim]Voice output failed.[/]")
                    continue
                elif (
                    isinstance(result, tuple)
                    and len(result) >= 2
                    and result[0] == "voice_in"
                ):
                    secs = float(result[1])
                    try:
                        from .voice import record_and_transcribe

                        console.print("[dim]Listening...[/]")
                        line = record_and_transcribe(
                            seconds=secs, meter_cb=_voice_meter
                        )
                        _voice_meter_done()
                        console.print(f"[dim]Heard:[/] {line}")
                    except Exception as e:
                        logger.debug("Voice input failed: %s", e)
                        console.print(
                            "[dim]Voice input failed. Please type your message instead.[/]"
                        )
                        continue
                elif (
                    isinstance(result, tuple)
                    and len(result) >= 2
                    and result[0] == "set_trace_filter"
                ):
                    trace_filter = cast(str, result[1]).strip()
                    if trace_filter:
                        console.print(f"[dim]Tool trace filter set: {trace_filter}[/]")
                    else:
                        console.print("[dim]Tool trace filter cleared.[/]")
                    continue
                elif (
                    isinstance(result, tuple)
                    and len(result) >= 2
                    and result[0] == "set_compact_mode"
                ):
                    arg = cast(str, result[1]).strip().lower()
                    if arg in ("", "toggle"):
                        compact_mode = not compact_mode
                    elif arg in ("on", "1", "true", "yes"):
                        compact_mode = True
                    elif arg in ("off", "0", "false", "no"):
                        compact_mode = False
                    elif arg == "auto":
                        compact_mode = console.size.width < 110
                    else:
                        console.print(
                            "[dim]Use /compact on|off|auto (or no arg to toggle).[/]"
                        )
                        continue
                    console.print(
                        f"[dim]Compact mode: {'on' if compact_mode else 'off'}[/]"
                    )
                    continue
                elif (
                    isinstance(result, tuple)
                    and len(result) >= 2
                    and result[0] == "kg_add"
                ):
                    from .state import add_knowledge_node

                    raw = cast(str, result[1]).strip()
                    if not raw:
                        console.print(
                            "[dim]Usage: /kg_add <topic> | <summary> [| rel1,rel2][/]"
                        )
                        continue
                    parts_ = [p.strip() for p in raw.split("|")]
                    topic = parts_[0] if parts_ else ""
                    summary = parts_[1] if len(parts_) > 1 else ""
                    related = (
                        [x.strip() for x in parts_[2].split(",") if x.strip()]
                        if len(parts_) > 2
                        else []
                    )
                    msg = add_knowledge_node(
                        topic, summary=summary, related=related, source="tui:/kg_add"
                    )
                    console.print(f"[dim]knowledge_graph: {msg}[/]")
                    continue
                elif (
                    isinstance(result, tuple)
                    and len(result) >= 2
                    and result[0] == "kg_query"
                ):
                    from .state import query_knowledge_graph

                    q = cast(str, result[1]).strip()
                    rows = query_knowledge_graph(q, max_results=8)
                    if not rows:
                        console.print("[dim]No knowledge graph matches.[/]")
                        continue
                    console.print("[dim]Knowledge graph matches:[/]")
                    for i, row in enumerate(rows, 1):
                        topic = row.get("topic", "")
                        summary = str(row.get("summary", "")).strip()
                        console.print(f"[dim]  {i}. {topic}[/]")
                        if summary:
                            console.print(f"[dim]     - {summary[:160]}[/]")
                    continue
                elif (
                    isinstance(result, tuple)
                    and len(result) >= 2
                    and result[0] == "rag_index"
                ):
                    from .rag import build_local_rag_index

                    target = cast(str, result[1]).strip() or (
                        workspace_root or os.getcwd()
                    )
                    try:
                        info = build_local_rag_index(target)
                        console.print(
                            f"[dim]RAG index built: files={info.get('indexed_files')} chunks={info.get('chunk_count')}[/]"
                        )
                    except Exception as e:
                        logger.debug("RAG index build failed: %s", e)
                        console.print("[dim]Failed to build RAG index.[/]")
                    continue
                elif (
                    isinstance(result, tuple)
                    and len(result) >= 2
                    and result[0] == "run_summary"
                ):
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
                            max_tool_rounds=1,
                            confirm_tool_calls=effective_confirm(),
                            before_tool_call=_before_tool_call_tui
                            if effective_confirm()
                            else None,
                            provider=provider,
                        )
                    else:
                        summary = await run_agent_loop_no_mcp(
                            model_ref[0],
                            prompt,
                            system_prompt=_SYSTEM,
                            message_history=[],
                            provider=provider,
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
            status["last_user"] = line
            root = workspace_root if workspace_root is not None else os.getcwd()
            # Handle pasted image data URLs in the input.
            img_path, cleaned = _extract_pasted_image(line)
            if img_path:
                pending_image_ref[0] = (img_path, "")
                line = cleaned or line
            line_expanded = expand_at_refs(line, root)
            current_prompt_ref[0] = str(line_expanded)
            _maybe_checkpoint(str(line_expanded))
            model_for_turn = _route_model(str(line_expanded), model_ref[0])
            image_paths_list: list[str] = []
            if pending_image_ref[0]:
                path, msg = cast(tuple[str, str], pending_image_ref[0])
                pending_image_ref[0] = None
                try:
                    resolved = Path(path) if os.path.isabs(path) else Path(root) / path
                    resolved = resolved.resolve()
                    if resolved.exists():
                        image_paths_list = [str(resolved)]
                    if msg:
                        line_expanded = (
                            (msg + "\n\n" + line_expanded) if line_expanded else msg
                        )
                except Exception:
                    logger.debug("Image attachment failed", exc_info=True)
            msg_history = [{"role": r, "content": c} for r, c in history[:-1]]
            if plan_exec_verify and _should_plan_exec_verify(str(line_expanded)):
                mem_block = (
                    build_dynamic_memory_context(
                        cast(str, line_expanded),
                        kg_max_results=memory_kg_max_results,
                        rag_max_results=memory_rag_max_results,
                        rag_snippet_chars=memory_rag_snippet_chars,
                    )
                    if memory_auto_context
                    else ""
                )
                sys_prompt = (
                    _SYSTEM
                    + "\n\n--- Retrieved memory (query-specific) ---\n\n"
                    + mem_block
                    if mem_block
                    else _SYSTEM
                )
                if not quiet:
                    console.print("[dim]Planning...[/]")
                plan = await run_agent_loop_no_mcp(
                    plan_model,
                    cast(str, line_expanded),
                    system_prompt=(
                        sys_prompt
                        + "\n\nYou are a planner. Produce a concise step-by-step plan (3-8 steps). "
                        "Do not call tools. Return the plan only."
                    ),
                    message_history=None,
                    provider=provider,
                )
                plan = (plan or "").strip()
                exec_prompt = (
                    f"Goal:\n{line_expanded}\n\nPlan:\n{plan}\n\n"
                    "Execute the plan step-by-step. Provide the final answer only."
                )
                if not quiet:
                    console.print("[dim]Executing...[/]")
                run_start = time.perf_counter()
                status["tool_calls"] = 0
                status["tool_errors"] = 0
                if session is not None:
                    final = await run_agent_loop(
                        session,
                        model_for_turn,
                        exec_prompt,
                        system_prompt=sys_prompt,
                        max_tool_rounds=max_tool_rounds_eff(),
                        image_paths=image_paths_list if image_paths_list else None,
                        max_messages=max_messages,
                        max_tool_result_chars=max_tool_result_chars,
                        message_history=msg_history if msg_history else None,
                        quiet=quiet,
                        timing=timing,
                        allowed_tools=allowed_tools,
                        blocked_tools=blocked_tools,
                        confirm_tool_calls=effective_confirm(),
                        before_tool_call=_before_tool_call_tui
                        if effective_confirm()
                        else None,
                        provider=provider,
                    )
                else:
                    final = await run_agent_loop_no_mcp(
                        model_for_turn,
                        exec_prompt,
                        system_prompt=sys_prompt,
                        message_history=msg_history if msg_history else None,
                        provider=provider,
                    )
                if not quiet:
                    console.print("[dim]Verifying...[/]")
                verify = await run_agent_loop_no_mcp(
                    verify_model,
                    "Review the following answer for correctness and completeness. Output only the revised answer, no preamble.\n\n"
                    + (final or ""),
                    system_prompt="You are a verifier. Output only the revised answer.",
                    message_history=None,
                    provider=provider,
                )
                final = (verify or "").strip()
                status["last_duration"] = time.perf_counter() - run_start
                history.append(("assistant", final))
                message_history.append({"role": "user", "content": line})
                message_history.append({"role": "assistant", "content": final})
                _maybe_apply_edits_tui(final)
                if session_ref:
                    try:
                        from .sessions import save_session

                        save_session(session_ref[0], None, message_history, root)
                    except Exception:
                        logger.debug("Session save failed", exc_info=True)
                console.print(
                    Panel(Markdown(final), title="Assistant", border_style="cyan")
                )
                continue
            # Keep Live panel to a fixed height so the prompt line stays visible below (during and after agent run).
            live_panel_max_lines = max(10, (console.size.height or 24) - 4)

            def _status_line() -> str:
                spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                # Tie spinner to configured refresh rate so animation stays in sync with redraws.
                idx = int(time.monotonic() * max(1, tui_refresh_hz)) % len(spinner)
                spin = spinner[idx]
                if status["phase"] == "tool" and status.get("tool"):
                    return f"[dim]{spin} Tool: {status['tool']} (awaiting approval)[/]"
                if status["phase"] == "tool_running" and status.get("tool"):
                    return f"[dim]{spin} Tool: {status['tool']} (running)[/]"
                if status["phase"] == "streaming" or status.get("has_output"):
                    return f"[dim]{spin} Assistant (streaming...)[/]"
                if status["phase"] == "thinking":
                    waiting_since = status.get("waiting_since")
                    if isinstance(waiting_since, (int, float)) and not status.get(
                        "has_output"
                    ):
                        elapsed = max(0.0, time.perf_counter() - waiting_since)
                        return f"[dim]{spin} Assistant (thinking... {elapsed:.1f}s)[/]"
                    return f"[dim]{spin} Assistant (thinking...)[/]"
                return f"[dim]{spin} Assistant (idle)[/]"

            def _tool_panel_text() -> str:
                if not tool_trace and not tool_log:
                    return "[dim]No tool activity yet.[/]"

                def _shorten(s: str, n: int = 140) -> str:
                    return s if len(s) <= n else (s[: n - 1] + "…")

                entries = list(tool_trace)
                if trace_filter:
                    needle = trace_filter.lower()
                    entries = [t for t in entries if needle in t.lower()]
                if compact_mode:
                    entries = entries[: min(3, len(entries))]
                trace_lines = (
                    "\n".join(f"[dim]{_shorten(t)}[/]" for t in entries)
                    if entries
                    else "[dim](no matches)[/]"
                )
                log_entries = list(tool_log)
                if trace_filter:
                    needle = trace_filter.lower()
                    log_entries = [t for t in log_entries if needle in t.lower()]
                if compact_mode:
                    log_entries = log_entries[: min(3, len(log_entries))]
                log_lines = (
                    "\n".join(f"[dim]{_shorten(t)}[/]" for t in log_entries)
                    if log_entries
                    else "[dim](no matches)[/]"
                )
                return f"[bold]Trace[/]\n{trace_lines}\n\n[bold]Log[/]\n{log_lines}"

            def _safe_single_line(s: str, n: int = 160) -> str:
                s = " ".join((s or "").split())
                return s if len(s) <= n else s[: n - 1] + "…"

            def _timeline_text() -> str:
                phase = status.get("phase", "idle")
                tool = status.get("tool") or "-"
                last_user = status.get("last_user") or "-"
                agents = status.get("agents_running", 0)
                agents_done = status.get("agents_done", 0)
                last_duration = status.get("last_duration")
                tool_calls = status.get("tool_calls", 0)
                tool_errors = status.get("tool_errors", 0)
                voice_level = float(status.get("voice_level") or 0.0)
                agent_line = (
                    f"[bold]Agents:[/] {agents_done}/{agents} running"
                    if agents
                    else "[bold]Agents:[/] -"
                )
                summary_line = (
                    f"[bold]LastRun:[/] {last_duration:.1f}s  "
                    f"[bold]Tools:[/] {tool_calls}  "
                    f"[bold]Errors:[/] {tool_errors}"
                    if isinstance(last_duration, (int, float))
                    else "[bold]LastRun:[/] -"
                )
                voice_line = ""
                if voice_level > 0:
                    bars = int(max(0.0, min(voice_level, 1.0)) * 16)
                    voice_line = "[bold]Mic:[/] " + ("█" * bars) + (" " * (16 - bars))
                return (
                    f"[bold]Phase:[/] {phase}  "
                    f"[bold]Tool:[/] {tool}  "
                    f"{agent_line}\n"
                    f"{summary_line}\n"
                    f"{voice_line}\n"
                    f"[bold]Last:[/] {_safe_single_line(last_user, 160)}"
                )

            def _sidebar_text() -> str:
                """Compact sidebar showing session metadata, token count, tool calls, duration."""
                session_short = session_ref[0][:8] if session_ref else "none"
                tokens = int(status.get("token_count", 0))
                tool_calls_count = int(status.get("tool_calls", 0))
                tool_errs = int(status.get("tool_errors", 0))
                last_dur = status.get("last_duration")
                dur_str = (
                    f"{last_dur:.1f}s" if isinstance(last_dur, (int, float)) else "-"
                )
                mode = (
                    "auto"
                    if autonomous_ref[0]
                    else "confirm"
                    if effective_confirm()
                    else "open"
                )
                lines = [
                    f"[bold]Session:[/] {session_short}",
                    f"[bold]Tokens:[/]  {tokens}",
                    f"[bold]Tools:[/]   {tool_calls_count} ({tool_errs} err)",
                    f"[bold]Duration:[/]{dur_str}",
                    f"[bold]Mode:[/]    {mode}",
                ]
                return "\n".join(lines)

            def _agents_panel_text() -> str:
                agents = status.get("agents") or []
                if not agents:
                    return "[dim]No agents active.[/]"
                lines = ["[bold]Agents[/]"]
                for a in agents:
                    name = a.get("name", "Agent")
                    state = a.get("state", "idle")
                    note = a.get("note", "")
                    lines.append(f"[dim]{name}[/] [{state}] {note}")
                lines.append("[dim]Tip: /agents_show <n|all|summary>[/]")
                return "\n".join(lines)

            def _status_bar_text() -> str:
                sandbox_level = os.environ.get("OLLAMACODE_SANDBOX_LEVEL", "supervised")
                session_short = session_ref[0][:8] if session_ref else "none"
                mode = (
                    "auto"
                    if autonomous_ref[0]
                    else "confirm"
                    if effective_confirm()
                    else "manual"
                )
                tool_state = status.get("tool") or "-"
                queue_info = ""
                if queue_inputs:
                    queue_info = f"  [bold]Queue:[/] {input_queue.qsize()}"
                budget_info = ""
                run_budget = os.environ.get("OLLAMACODE_RUN_BUDGET_SECONDS", "").strip()
                tool_budget = os.environ.get(
                    "OLLAMACODE_TOOL_BUDGET_SECONDS", ""
                ).strip()
                tool_timeout = os.environ.get(
                    "OLLAMACODE_TOOL_TIMEOUT_SECONDS", ""
                ).strip()
                if run_budget or tool_budget or tool_timeout:
                    budget_info = (
                        f"  [bold]Budget:[/] run {run_budget or '-'}s "
                        f"tool {tool_budget or '-'}s "
                        f"tmo {tool_timeout or '-'}s"
                    )
                return (
                    f"[bold]Model:[/] {model_ref[0]}  "
                    f"[bold]Provider:[/] {provider_name}  "
                    f"[bold]Sandbox:[/] {sandbox_level}  "
                    f"[bold]Mode:[/] {mode}  "
                    f"[bold]Session:[/] {session_short}  "
                    f"[bold]Tool:[/] {tool_state}"
                    f"{queue_info}"
                    f"{budget_info}"
                )

            # Cache history markdown: key = (len(history), limit); rebuilt only when history changes.
            _history_md_cache: dict[str, Any] = {
                "len": -1,
                "limit": -1,
                "prefix": "",
                "body": "",
            }

            def _build_chat_md_str(accumulated: str) -> str:
                """Build the chat markdown string (without status line)."""
                limit = 1 if compact_mode else _CHAT_PANEL_LAST_N_EXCHANGES
                cache = _history_md_cache
                hlen = len(history)
                if cache["len"] != hlen or cache["limit"] != limit:
                    # Rebuild history portion only when conversation changes.
                    cache["len"] = hlen
                    cache["limit"] = limit
                    hist = history
                    if limit is not None and limit > 0 and hlen > limit * 2:
                        hist = history[-(limit * 2) :]
                        cache["prefix"] = "*(scroll up for earlier messages)*\n\n"
                    else:
                        cache["prefix"] = ""
                    parts = []
                    for role, text in hist:
                        label = "**You**" if role == "user" else "**Assistant**"
                        display_text = (
                            _escape_rich_markup(text) if role == "user" else text
                        )
                        parts.append(f"{label}\n\n{display_text}")
                    cache["body"] = "\n\n---\n\n".join(parts) if parts else ""
                # Build final markdown: cached history + streaming suffix.
                parts_list = []
                if cache["body"]:
                    parts_list.append(cache["body"])
                if accumulated:
                    parts_list.append("**Assistant** *(streaming)*\n\n" + accumulated)
                body = (
                    "\n\n---\n\n".join(parts_list)
                    if parts_list
                    else "*(no messages yet)*"
                )
                md = cache["prefix"] + body
                # Truncate so Live only redraws this many lines; prompt stays visible below.
                lines = md.split("\n")
                if len(lines) > live_panel_max_lines:
                    md = "\n".join(lines[-live_panel_max_lines:])
                # Hard char limit to prevent Markdown parser from choking on very long content.
                _MAX_CHAT_MD_CHARS = 50_000
                if len(md) > _MAX_CHAT_MD_CHARS:
                    md = "*(content truncated)*\n\n" + md[-_MAX_CHAT_MD_CHARS:]
                return md

            # Cache parsed Markdown object — only re-parse when chat content changes.
            _md_obj_cache: dict[str, Any] = {"src": None, "obj": None}

            def _render_live(accumulated: str, done: bool):
                chat_src = _build_chat_md_str(accumulated)
                if _md_obj_cache["src"] != chat_src:
                    _md_obj_cache["src"] = chat_src
                    _md_obj_cache["obj"] = Markdown(chat_src)
                chat_renderable = _md_obj_cache["obj"]
                # Status line changes every frame (spinner/timer) — use lightweight Text.
                if not done:
                    chat_panel_content = Group(
                        chat_renderable, Text.from_markup(_status_line())
                    )
                else:
                    chat_panel_content = chat_renderable
                panels = [
                    Panel(
                        _timeline_text(),
                        title="Timeline",
                        border_style=_theme["panel_timeline"],
                    ),
                    Panel(
                        chat_panel_content,
                        title="Chat",
                        border_style=_theme["panel_chat"],
                    ),
                ]
                if not compact_mode:
                    panels.append(
                        Panel(
                            _tool_panel_text(),
                            title="Tools",
                            border_style=_theme["panel_tools"],
                        )
                    )
                    panels.append(
                        Panel(
                            _agents_panel_text(),
                            title="Agents",
                            border_style=_theme["panel_agents"],
                        )
                    )
                panels.append(
                    Panel(
                        _sidebar_text(), title="Info", border_style=_theme["panel_info"]
                    )
                )
                panels.append(
                    Panel(
                        _status_bar_text(),
                        title="Status",
                        border_style=_theme["panel_status"],
                    )
                )
                return Group(*panels)

            def make_update(text: str, done: bool) -> None:
                if done:
                    # Final call: text is the complete response.
                    status["accumulated"] = text
                else:
                    # Intermediate: text is a single fragment — append.
                    status["accumulated"] = status.get("accumulated", "") + text
                status["has_output"] = bool(status["accumulated"])
                status["phase"] = "streaming" if status["has_output"] else "thinking"
                if status["has_output"]:
                    status["waiting_since"] = None
                # Estimate token count (~4 chars per token)
                status["token_count"] = len(status["accumulated"]) // 4
                # Only render on final frame; _tick drives all intermediate renders.
                if done:
                    live.update(
                        _render_live(status["accumulated"], done=True), refresh=True
                    )

            def _on_tool_start(name: str, arguments: dict) -> None:
                status["phase"] = "tool_running"
                status["tool"] = name
                status["tool_calls"] = int(status.get("tool_calls", 0)) + 1
                ts = datetime.now().strftime("%H:%M:%S")
                tool_trace.appendleft(
                    f"{ts} → {name} {_tool_call_one_line(name, arguments)}"
                )
                _tool_start_cb(name, arguments)

            def _on_tool_end(name: str, arguments: dict, summary: str) -> None:
                status["phase"] = (
                    "streaming" if status.get("has_output") else "thinking"
                )
                status["tool"] = ""
                if "error" in summary.lower():
                    status["tool_errors"] = int(status.get("tool_errors", 0)) + 1
                ts = datetime.now().strftime("%H:%M:%S")
                tool_trace.appendleft(f"{ts} ✓ {name} {summary}")
                trimmed = summary[:tui_tool_log_chars] + (
                    "…" if len(summary) > tui_tool_log_chars else ""
                )
                tool_log.appendleft(f"{ts} {name}: {trimmed}")
                _tool_end_cb(name, arguments, summary)

            if session is not None:
                if (
                    provider is not None
                    and hasattr(provider, "set_tool_executor")
                    and os.environ.get("OLLAMACODE_APPLE_FM_NATIVE_TOOLS", "0") == "1"
                ):
                    from .mcp_client import call_tool, tool_result_to_content

                    async def _exec_tool(name: str, args: dict):
                        try:
                            decision = await hook_mgr.run_pre_tool_use(
                                name, args, user_prompt=current_prompt_ref[0]
                            )
                            if decision and decision.behavior == "deny":
                                raise RuntimeError(
                                    decision.message or "Blocked by hook."
                                )
                            if (
                                decision
                                and decision.behavior == "modify"
                                and decision.updated_input
                            ):
                                args = decision.updated_input
                        except Exception:
                            pass
                        r = recorder_ref[0]
                        if r is not None:
                            for p in _tool_paths_from_args(name, args):
                                r.record_pre(p)
                        result = await call_tool(session, name, args)
                        out = tool_result_to_content(result)
                        await hook_mgr.run_post_tool_use(
                            name,
                            args,
                            out,
                            getattr(result, "isError", False),
                            user_prompt=current_prompt_ref[0],
                        )
                        return out

                    try:
                        provider.set_tool_executor(_exec_tool)
                    except Exception:
                        pass
                mem_block = (
                    build_dynamic_memory_context(
                        cast(str, line_expanded),
                        kg_max_results=memory_kg_max_results,
                        rag_max_results=memory_rag_max_results,
                        rag_snippet_chars=memory_rag_snippet_chars,
                    )
                    if memory_auto_context
                    else ""
                )
                sys_prompt = (
                    _SYSTEM
                    + "\n\n--- Retrieved memory (query-specific) ---\n\n"
                    + mem_block
                    if mem_block
                    else _SYSTEM
                )
                stream = run_agent_loop_stream(
                    session,
                    model_for_turn,
                    cast(str, line_expanded),
                    system_prompt=sys_prompt,
                    max_tool_rounds=max_tool_rounds_eff(),
                    image_paths=image_paths_list if image_paths_list else None,
                    max_messages=max_messages,
                    max_tool_result_chars=max_tool_result_chars,
                    message_history=msg_history,
                    quiet=quiet,
                    timing=timing,
                    tool_progress_brief=True,
                    allowed_tools=allowed_tools,
                    blocked_tools=blocked_tools,
                    confirm_tool_calls=effective_confirm(),
                    before_tool_call=_before_tool_call_tui
                    if effective_confirm()
                    else None,
                    on_tool_start=_on_tool_start,
                    on_tool_end=_on_tool_end,
                    provider=provider,
                )
            else:
                mem_block = (
                    build_dynamic_memory_context(
                        cast(str, line_expanded),
                        kg_max_results=memory_kg_max_results,
                        rag_max_results=memory_rag_max_results,
                        rag_snippet_chars=memory_rag_snippet_chars,
                    )
                    if memory_auto_context
                    else ""
                )
                sys_prompt = (
                    _SYSTEM
                    + "\n\n--- Retrieved memory (query-specific) ---\n\n"
                    + mem_block
                    if mem_block
                    else _SYSTEM
                )
                stream = run_agent_loop_no_mcp_stream(
                    model_for_turn,
                    cast(str, line_expanded),
                    system_prompt=sys_prompt,
                    message_history=msg_history,
                    provider=provider,
                )

            async def _tick():
                # Small initial sleep lets the Live constructor's first render
                # display before we start overwriting it — avoids startup flicker.
                await asyncio.sleep(max(0.05, 1.0 / max(1, tui_refresh_hz)))
                while not status["done"]:
                    live.update(
                        _render_live(status.get("accumulated", ""), done=False),
                        refresh=True,
                    )
                    await asyncio.sleep(max(0.05, 1.0 / max(1, tui_refresh_hz)))

            if use_live:
                with Live(
                    _render_live("", done=False),
                    console=console,
                    auto_refresh=False,
                    transient=True,
                    vertical_overflow="crop",
                ) as live:
                    run_start = time.perf_counter()
                    _status_update(
                        done=False,
                        phase="thinking",
                        accumulated="",
                        has_output=False,
                        tool_calls=0,
                        tool_errors=0,
                        waiting_since=run_start,
                    )
                    ticker = asyncio.create_task(_tick())
                    stream_error = None
                    try:
                        final = await _stream_into_live(stream, make_update)
                    except Exception as e:  # noqa: BLE001
                        logger.debug("Stream error", exc_info=True)
                        final = "**Error:** Something went wrong while streaming the response."
                        stream_error = e
                    finally:
                        _status_update(
                            done=True,
                            waiting_since=None,
                            last_duration=time.perf_counter() - run_start,
                        )
                        ticker.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await ticker
                        # Final render to ensure the completed state is displayed.
                        live.update(
                            _render_live(status.get("accumulated", ""), done=True),
                            refresh=True,
                        )
                # Print error outside Live context to avoid visual glitches.
                if stream_error is not None and not quiet:
                    console.print("[red]Stream error.[/]")
                # transient=True erases the Live display; print the response
                # into the permanent scrollback so the user can review it.
                if final and not quiet:
                    console.print(
                        Panel(Markdown(final), title="Assistant", border_style="blue")
                    )
            else:
                try:
                    if not quiet:
                        console.print("[dim]Assistant (thinking...)[/]")
                    status["has_output"] = False
                    status["tool_calls"] = 0
                    status["tool_errors"] = 0
                    run_start = time.perf_counter()
                    status["waiting_since"] = run_start
                    final = await _stream_into_console(stream)
                except Exception:  # noqa: BLE001
                    logger.debug("Stream error (non-live)", exc_info=True)
                    final = "Error: Something went wrong while streaming the response."
                    if not quiet:
                        console.print("[red]Stream error.[/]")
                finally:
                    status["waiting_since"] = None
                    status["last_duration"] = time.perf_counter() - run_start

            final = _sanitize_stream_text(final)
            if recorder_ref[0] is not None:
                recorder_ref[0].finalize()
                recorder_ref[0] = None
            history.append(("assistant", final))
            message_history.append({"role": "user", "content": line})
            message_history.append({"role": "assistant", "content": final})
            _maybe_apply_edits_tui(final)
            if voice_out_enabled:
                try:
                    from .voice import speak_text

                    speak_text(final)
                except Exception:
                    logger.debug("Voice output failed", exc_info=True)
            if session_ref:
                try:
                    from .sessions import save_session

                    save_session(session_ref[0], None, message_history, root)
                except Exception:
                    logger.debug("Session save failed", exc_info=True)
    finally:
        if producer_task is not None:
            producer_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            if producer_task is not None:
                await producer_task
