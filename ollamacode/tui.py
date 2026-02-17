"""
Optional TUI (terminal UI) for OllamaCode chat using Rich.
Install with: pip install ollamacode[tui]
Slash commands, Rich Markdown, multi-line input, conversation history.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shlex
import subprocess
import tempfile
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
from .memory import build_dynamic_memory_context
from .multi_agent import run_multi_agent
from .context import expand_at_refs
from .skills import load_skills_text
from .templates import load_prompt_template

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
    "/commands",
    "/sessions",
    "/search",
    "/resume",
    "/session",
    "/branch",
    "/quit",
    "/exit",
]

# Arrow keys and line editing: readline on Unix/macOS; on Windows use prompt_toolkit when available (pip install ollamacode[tui])
# Up/Down: readline.add_history_entry() for input(); PromptSession keeps history for prompt_toolkit.
# Tab: complete slash commands when line starts with /.
_tui_prompt_fn = None
_tui_prompt_session = None
_readline_available = False
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


if _tui_prompt_session is None:
    try:
        from prompt_toolkit import PromptSession  # pyright: ignore[reportMissingImports]
        from prompt_toolkit.history import InMemoryHistory  # pyright: ignore[reportMissingImports]
        from prompt_toolkit.completion import Completer, Completion  # pyright: ignore[reportMissingImports]

        class _SlashCommandCompleter(Completer):
            """Complete slash commands when the line contains / (Tab to complete)."""

            def get_completions(self, document, complete_event):
                text = document.text_before_cursor
                idx = text.rfind("/")
                if idx < 0:
                    return
                prefix = text[idx:].lower()
                if not prefix.startswith("/"):
                    return
                for cmd in _SLASH_COMMANDS:
                    if cmd.startswith(prefix) and cmd != prefix:
                        yield Completion(cmd, start_position=-len(prefix))

        _tui_prompt_session = PromptSession(
            history=InMemoryHistory(),
            completer=_SlashCommandCompleter(),
        )
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
):
    """Handle slash command in TUI. Returns 'quit', 'cleared', 'help', ('run_prompt', prompt), ('run_multi', prompt), ('run_summary', n), ('run_subagent', type, task), ('set_pending_image', path, msg), ('new_session',), ('resume_session', id), ('branch_session', new_id), or None."""
    line = line.strip()
    if not line.startswith("/"):
        return None
    parts = line.split(maxsplit=1)
    cmd = (parts[0] or "").lower()
    rest = (parts[1] or "").strip() if len(parts) > 1 else ""
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
  /commands     List built-in and custom slash commands
  /subagents    List available subagent types
  /subagent <type> <task>  Run task in a subagent (restricted tools)
  /image <path> [msg]  Attach image to next message (vision models)
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
            console.print(f"[dim]Sessions: {e}[/]")
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
            console.print(f"[dim]Search failed: {e}[/]")
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
            console.print(
                f"[dim]Resumed session {rid[:8]}... ({len(msgs)} messages)[/]"
            )
            return ("resume_session", rid)
        except Exception as e:
            console.print(f"[dim]Resume failed: {e}[/]")
        return "help"
    if cmd == "/session":
        if session_ref is None:
            console.print("[dim]Session persistence not active.[/]")
            return "help"
        try:
            from .sessions import get_session_info, save_session

            sid = session_ref[0]
            if rest:
                save_session(sid, rest.strip(), message_history)
                console.print(f"[dim]Session title set to: {rest.strip()[:60]}[/]")
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
            console.print(f"[dim]Session: {e}[/]")
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
            console.print(f"[dim]Branched to new session {new_id[:8]}...[/]")
            return ("branch_session", new_id)
        except Exception as e:
            console.print(f"[dim]Branch failed: {e}[/]")
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
            console.print(
                "[dim]Usage: /image <path> [message]. Attach image to next message.[/]"
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
    accumulated: list[str] = []
    async for frag in stream:
        accumulated.append(frag)
        update_cb("".join(accumulated), False)
    final = "".join(accumulated)
    update_cb(final, True)
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
    tui_tool_trace_max: int = 5,
    tui_tool_log_max: int = 8,
    tui_tool_log_chars: int = 160,
    tui_refresh_hz: int = 5,
    memory_auto_context: bool = True,
    memory_kg_max_results: int = 4,
    memory_rag_max_results: int = 4,
    memory_rag_snippet_chars: int = 220,
    autonomous_mode: bool = False,
    subagents: list[dict[str, Any]] | None = None,
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

    history: list[tuple[str, str]] = []
    message_history: list[dict[str, Any]] = []
    try:
        from .sessions import create_session

        session_ref: list[str] = [create_session("")]
    except Exception:
        session_ref = []
    pending_image_ref: list[tuple[str, str] | None] = [None]
    model_ref = [model]
    autonomous_ref = [autonomous_mode]

    def effective_confirm() -> bool:
        return confirm_tool_calls and not autonomous_ref[0]

    def max_tool_rounds_eff() -> int:
        return max(max_tool_rounds, 30) if autonomous_ref[0] else max_tool_rounds

    loop = asyncio.get_event_loop()

    def get_input() -> str:
        """Read a single line; Enter sends. Up/Down cycle input history. Tab completes slash commands. Prompt shows current model."""
        if _tui_prompt_session is not None:
            return _tui_prompt_session.prompt(f"You [{model_ref[0]}]: ").strip()
        if _readline_available:
            readline.set_completer(_readline_slash_completer)  # type: ignore[name-defined]
            readline.parse_and_bind("tab: complete")
        return input(f"You [{model_ref[0]}]: ").strip()

    def add_input_history(line: str) -> None:
        """Add a submitted line to input history so Up/Down can recall it (readline path only)."""
        if not line or _tui_prompt_session is not None:
            return
        if _readline_available:
            try:
                readline.add_history_entry(line)  # type: ignore[name-defined]
            except Exception:
                pass

    root = workspace_root if workspace_root is not None else os.getcwd()
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

    status: dict[str, Any] = {
        "phase": "idle",
        "has_output": False,
        "done": False,
        "tool": "",
        "accumulated": "",
    }
    tool_trace: deque[str] = deque(maxlen=tui_tool_trace_max)
    tool_log: deque[str] = deque(maxlen=tui_tool_log_max)
    trace_filter = ""
    compact_mode = console.size.width < 110

    async def _before_tool_call_tui(tool_name: str, arguments: dict):
        """Prompt [y/N/e] per tool with Rich panel when confirm_tool_calls; e = edit args in $EDITOR."""
        status["phase"] = "tool"
        status["tool"] = tool_name
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
            status["phase"] = "streaming" if status.get("has_output") else "thinking"
            status["tool"] = ""
            return "run"
        if choice in ("n", "no", ""):
            status["phase"] = "streaming" if status.get("has_output") else "thinking"
            status["tool"] = ""
            return "skip"
        if choice in ("e", "edit"):
            edited = _edit_tool_args_in_editor(arguments, root)
            if edited is not None:
                status["phase"] = (
                    "streaming" if status.get("has_output") else "thinking"
                )
                status["tool"] = ""
                return ("edit", edited)
            console.print("[dim]Invalid JSON or cancel; running with original args.[/]")
            status["phase"] = "streaming" if status.get("has_output") else "thinking"
            status["tool"] = ""
            return "run"
        console.print("[dim]Choose y (run), N (skip), or e (edit).[/]")
        return await _before_tool_call_tui(tool_name, arguments)  # re-prompt

    console.print(
        Panel(
            "[bold]OllamaCode TUI[/] – [dim]/help[/] for commands. Up/Down = history. Prompt stays active; commands queue. Empty or Ctrl+C to exit.",
            title="OllamaCode",
            border_style="green",
        )
    )
    if show_semantic_hint:
        console.print(
            "[dim]Tip: For semantic codebase search, add a semantic MCP server to config. See docs/MCP_SERVERS.md.[/]"
        )

    # Producer: always reading input; consumer: processes one command at a time (like cursor-cli).
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
                    return

    producer_task: asyncio.Task[None] | None = None

    async def get_next_line() -> str | None:
        """Get next user line from queue (None = quit)."""
        return await input_queue.get()

    try:
        producer_task = asyncio.create_task(input_producer())
        while True:
            line = await get_next_line()
            if line is None:
                break
            add_input_history(line)

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
                continue
            if isinstance(result, tuple) and result[0] == "set_pending_image":
                pending_image_ref[0] = (result[1], result[2])
                console.print(
                    f"[dim]Image attached: {result[1]} (message for next turn: {result[2][:40] or '(none)'}...)[/]"
                )
                continue
            if isinstance(result, tuple) and result[0] == "run_subagent":
                _, stype, task = result
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
                    )
                else:
                    out = await run_agent_loop_no_mcp(
                        sub_model, task, system_prompt=_SYSTEM, message_history=[]
                    )
                history.append((task, out))
                message_history.append({"role": "user", "content": task})
                message_history.append({"role": "assistant", "content": out})
                if session_ref:
                    try:
                        from .sessions import save_session

                        save_session(session_ref[0], None, message_history)
                    except Exception:
                        pass
                console.print(
                    Panel(
                        Markdown(out), title=f"Subagent ({stype})", border_style="cyan"
                    )
                )
                continue
            if result is not None:
                if isinstance(result, tuple) and result[0] == "run_prompt":
                    line = cast(str, result[1])
                elif isinstance(result, tuple) and result[0] == "run_multi":
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

                            save_session(session_ref[0], None, message_history)
                        except Exception:
                            pass
                    console.print(
                        Panel(
                            Markdown(out.content),
                            title="Assistant",
                            border_style="cyan",
                        )
                    )
                    continue
                elif isinstance(result, tuple) and result[0] == "copy_last":
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
                elif isinstance(result, tuple) and result[0] == "set_trace_filter":
                    trace_filter = cast(str, result[1]).strip()
                    if trace_filter:
                        console.print(f"[dim]Tool trace filter set: {trace_filter}[/]")
                    else:
                        console.print("[dim]Tool trace filter cleared.[/]")
                    continue
                elif isinstance(result, tuple) and result[0] == "set_compact_mode":
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
                elif isinstance(result, tuple) and result[0] == "kg_add":
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
                elif isinstance(result, tuple) and result[0] == "kg_query":
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
                elif isinstance(result, tuple) and result[0] == "rag_index":
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
                        console.print(f"[dim]Failed to build RAG index: {e}[/]")
                    continue
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
                            confirm_tool_calls=effective_confirm(),
                            before_tool_call=_before_tool_call_tui
                            if effective_confirm()
                            else None,
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
            image_paths_list: list[str] = []
            if pending_image_ref[0]:
                path, msg = pending_image_ref[0]
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
                    pass
            # Keep Live panel to a fixed height so the prompt line stays visible below (during and after agent run).
            live_panel_max_lines = max(10, (console.size.height or 24) - 4)

            def _status_line() -> str:
                spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                idx = status.get("spinner_i", 0) % len(spinner)
                spin = spinner[idx]
                if status["phase"] == "tool" and status.get("tool"):
                    return f"[dim]{spin} Tool: {status['tool']} (awaiting approval)[/]"
                if status["phase"] == "tool_running" and status.get("tool"):
                    return f"[dim]{spin} Tool: {status['tool']} (running)[/]"
                if status["phase"] == "streaming" or status.get("has_output"):
                    return f"[dim]{spin} Assistant (streaming...)[/]"
                if status["phase"] == "thinking":
                    return f"[dim]{spin} Assistant (thinking...)[/]"
                return f"[dim]{spin} Assistant (idle)[/]"

            def _tool_trace_block() -> str:
                if not tool_trace:
                    return ""
                entries = list(tool_trace)
                if trace_filter:
                    needle = trace_filter.lower()
                    entries = [t for t in entries if needle in t.lower()]
                if not entries:
                    return "\n\n[bold]Tool trace[/]\n[dim](no matches)[/]"
                if compact_mode:
                    entries = entries[: min(3, len(entries))]
                lines = "\n".join(f"[dim]{t}[/]" for t in entries)
                return f"\n\n[bold]Tool trace[/]\n{lines}"

            def _tool_log_block() -> str:
                if not tool_log:
                    return ""
                entries = list(tool_log)
                if trace_filter:
                    needle = trace_filter.lower()
                    entries = [t for t in entries if needle in t.lower()]
                if not entries:
                    return "\n\n[bold]Tool log[/]\n[dim](no matches)[/]"
                if compact_mode:
                    entries = entries[: min(3, len(entries))]
                lines = "\n".join(f"[dim]{t}[/]" for t in entries)
                return f"\n\n[bold]Tool log[/]\n{lines}"

            def _render_chat_markdown(accumulated: str, *, done: bool) -> str:
                limit = 1 if compact_mode else _CHAT_PANEL_LAST_N_EXCHANGES
                md = _conversation_to_markdown(
                    history, accumulated, limit_exchanges=limit
                )
                if done:
                    pass
                else:
                    blocks = _status_line() + _tool_trace_block()
                    if not compact_mode:
                        blocks += _tool_log_block()
                    md = md + "\n\n" + blocks
                # Truncate so Live only redraws this many lines; prompt stays visible below.
                lines = md.split("\n")
                if len(lines) > live_panel_max_lines:
                    md = "\n".join(lines[-live_panel_max_lines:])
                return md

            def make_update(accumulated: str, done: bool) -> None:
                status["accumulated"] = accumulated
                status["has_output"] = bool(accumulated)
                status["phase"] = "streaming" if status["has_output"] else "thinking"
                md = _render_chat_markdown(accumulated, done=done)
                live.update(Panel(Markdown(md), title="Chat", border_style="blue"))

            msg_history = [{"role": r, "content": c} for r, c in history[:-1]]

            def _on_tool_start(name: str, arguments: dict) -> None:
                status["phase"] = "tool_running"
                status["tool"] = name
                ts = datetime.now().strftime("%H:%M:%S")
                tool_trace.appendleft(
                    f"{ts} → {name} {_tool_call_one_line(name, arguments)}"
                )

            def _on_tool_end(name: str, arguments: dict, summary: str) -> None:
                status["phase"] = (
                    "streaming" if status.get("has_output") else "thinking"
                )
                status["tool"] = ""
                ts = datetime.now().strftime("%H:%M:%S")
                tool_trace.appendleft(f"{ts} ✓ {name} {summary}")
                trimmed = summary[:tui_tool_log_chars] + (
                    "…" if len(summary) > tui_tool_log_chars else ""
                )
                tool_log.appendleft(f"{ts} {name}: {trimmed}")

            if session is not None:
                mem_block = (
                    build_dynamic_memory_context(
                        line_expanded,
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
                    model_ref[0],
                    line_expanded,
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
                )
            else:
                mem_block = (
                    build_dynamic_memory_context(
                        line_expanded,
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
                    model_ref[0],
                    line_expanded,
                    system_prompt=sys_prompt,
                    message_history=msg_history,
                )

            async def _tick():
                while not status["done"]:
                    status["spinner_i"] = status.get("spinner_i", 0) + 1
                    md = _render_chat_markdown(
                        status.get("accumulated", ""), done=False
                    )
                    live.update(Panel(Markdown(md), title="Chat", border_style="blue"))
                    await asyncio.sleep(max(0.05, 1.0 / max(1, tui_refresh_hz)))

            with Live(
                Panel(
                    Markdown(_render_chat_markdown("", done=True)),
                    title="Chat",
                    border_style="blue",
                ),
                console=console,
                refresh_per_second=max(1, tui_refresh_hz),
                vertical_overflow="visible",
            ) as live:
                status["done"] = False
                status["phase"] = "thinking"
                status["accumulated"] = ""
                # Show prompt below Live so user can type next message while agent runs (queued).
                console.print(f"\nYou [{model_ref[0]}]: ", end="")
                ticker = asyncio.create_task(_tick())
                try:
                    final = await _stream_into_live(stream, make_update)
                except Exception as e:  # noqa: BLE001
                    final = f"**Error:** {e}"
                    if not quiet:
                        console.print(f"[red]Stream error:[/] {e}")
                finally:
                    status["done"] = True
                    ticker.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await ticker

            history.append(("assistant", final))
            message_history.append({"role": "user", "content": line})
            message_history.append({"role": "assistant", "content": final})
            if session_ref:
                try:
                    from .sessions import save_session

                    save_session(session_ref[0], None, message_history)
                except Exception:
                    pass
            # Show prompt immediately when agent is done (no need to press Enter first).
            console.print(f"\nYou [{model_ref[0]}]: ", end="")
    finally:
        if producer_task is not None:
            producer_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            if producer_task is not None:
                await producer_task
