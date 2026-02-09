"""
CLI for OllamaCode: chat with local Ollama + MCP tools.
"""

from __future__ import annotations

import argparse
from typing import Literal, cast
import asyncio
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from .agent import (
    run_agent_loop,
    run_agent_loop_no_mcp,
    run_agent_loop_no_mcp_stream,
    run_agent_loop_stream,
)
from .config import find_config_file, load_config, merge_config_with_env
from .context import expand_at_refs, get_branch_context, prepend_file_context
from .edits import apply_edits, format_edits_diff, parse_edits
from .mcp_client import McpConnection, connect_mcp_servers, connect_mcp_stdio


def _check_ollama_and_model(model: str, quiet: bool) -> None:
    """Fail fast with clear message if Ollama is unreachable or model is missing."""
    if os.environ.get("OLLAMACODE_SKIP_MODEL_CHECK"):
        return
    try:
        import ollama

        listed = ollama.list()
    except Exception as e:
        msg = str(e).lower() if e else ""
        if "connection" in msg or "refused" in msg or "connect" in msg:
            print(
                "Ollama is not running or not reachable. Start it with: ollama serve",
                file=sys.stderr,
            )
        else:
            print(f"Ollama error: {e}", file=sys.stderr)
        raise SystemExit(1) from e
    models_list = (
        getattr(listed, "models", None)
        or (listed.get("models") if isinstance(listed, dict) else None)
        or []
    )
    names = []
    for m in models_list:
        n = (
            getattr(m, "name", None)
            or (m.get("name") if isinstance(m, dict) else None)
            or getattr(m, "model", None)
            or (m.get("model") if isinstance(m, dict) else None)
        )
        if n:
            names.append(n)

    # Match exact, or list name with digest (e.g. gpt-oss:20b-abc), or base name when we ask for tag (e.g. list has "gpt-oss" for gpt-oss:20b)
    def matches(a: str, b: str) -> bool:
        if a == b:
            return True
        if a.startswith(b + ":") or a.startswith(b + "-"):
            return True
        if b.startswith(a + ":") or b.startswith(a + "-"):
            return True
        return False

    if not any(matches(n, model) for n in names):
        print(
            f"Model '{model}' not found. Pull it with: ollama pull {model}",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OllamaCode: coding assistant using local Ollama and MCP tools.",
    )
    p.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to config file (default: ollamacode.yaml or .ollamacode/config.yaml in cwd)",
    )
    p.add_argument(
        "--model",
        "-m",
        default=os.environ.get("OLLAMACODE_MODEL", "gpt-oss:20b"),
        help="Ollama model name (overrides config and OLLAMACODE_MODEL)",
    )
    p.add_argument(
        "--mcp-command",
        default=os.environ.get("OLLAMACODE_MCP_COMMAND", "python"),
        help="MCP server command for legacy single stdio (default: python)",
    )
    p.add_argument(
        "--mcp-args",
        nargs="*",
        default=[],
        help="MCP server args for legacy single stdio. Override with OLLAMACODE_MCP_ARGS (space-separated).",
    )
    p.add_argument(
        "--stream",
        "-s",
        action="store_true",
        default=True,
        help="Stream response tokens to stdout (default).",
    )
    p.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming; print full response when done.",
    )
    p.add_argument(
        "--tui",
        action="store_true",
        help="Run interactive terminal UI (Rich). Requires: pip install ollamacode[tui].",
    )
    p.add_argument(
        "--max-messages",
        type=int,
        default=None,
        metavar="N",
        help="Cap message history sent to Ollama (0 = no limit). For long chats; also in config as max_messages.",
    )
    p.add_argument(
        "--history-file",
        default=None,
        metavar="PATH",
        help="Append each interactive turn to this file (user + assistant). Disabled if not set.",
    )
    p.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress [OllamaCode] progress lines (e.g. for scripts).",
    )
    p.add_argument(
        "--max-tool-rounds",
        type=int,
        default=None,
        metavar="N",
        help="Max tool-call rounds per turn (default 20; also in config as max_tool_rounds).",
    )
    p.add_argument(
        "--timing",
        action="store_true",
        help="Log per-step durations (Ollama call, each tool call, turn total) to stderr.",
    )
    p.add_argument(
        "--no-mcp",
        action="store_true",
        help="Skip starting MCP servers for this run (faster when you don't need tools).",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8000,
        metavar="N",
        help="Port for 'serve' command (default 8000).",
    )
    p.add_argument(
        "--file",
        "-f",
        default=None,
        metavar="PATH",
        help="Prepend this file's contents (or --lines range) to the prompt (chat with selection).",
    )
    p.add_argument(
        "--lines",
        default=None,
        metavar="START-END",
        help="With --file: only include lines START-END (e.g. 10-30). Inclusive 1-based.",
    )
    p.add_argument(
        "--apply-edits",
        action="store_true",
        help="Parse <<EDITS>> JSON from model output; show diff and prompt to apply.",
    )
    p.add_argument(
        "--apply-edits-dry-run",
        action="store_true",
        help="With --apply-edits: show diff only, do not prompt or apply.",
    )
    p.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Single query to run, 'serve' for HTTP API, 'protocol' for stdio JSON-RPC, or 'convert-mcp' to convert MCP config.",
    )
    p.add_argument(
        "convert_mcp_input",
        nargs="?",
        default=None,
        metavar="INPUT_JSON",
        help="For convert-mcp: input JSON file (Cursor/Claude format); stdin if omitted.",
    )
    p.add_argument(
        "--output",
        "-o",
        default=None,
        dest="convert_mcp_output",
        metavar="OUTPUT_YAML",
        help="For convert-mcp: output YAML file; stdout if omitted.",
    )
    return p.parse_args()


def _resolve_mcp_servers(
    config_path: str | None,
    mcp_command: str,
    mcp_args: list[str],
) -> tuple[list[dict], bool, bool]:
    """Resolve mcp_servers from config and CLI. Returns (server_configs, use_mcp, using_builtin)."""
    config = load_config(config_path)
    merged = merge_config_with_env(
        config,
        model_env=os.environ.get("OLLAMACODE_MODEL"),
        mcp_args_env=os.environ.get("OLLAMACODE_MCP_ARGS"),
        system_extra_env=os.environ.get("OLLAMACODE_SYSTEM_EXTRA"),
    )
    servers = merged.get("mcp_servers") or []
    cli_args = (
        mcp_args if mcp_args else (os.environ.get("OLLAMACODE_MCP_ARGS") or "").split()
    )
    if cli_args:
        servers = [{"type": "stdio", "command": mcp_command, "args": cli_args}]
    has_config_file = find_config_file(config_path) is not None
    using_builtin = not has_config_file and not cli_args
    return servers, bool(servers), using_builtin


def _append_history(path: str, user: str, assistant: str) -> None:
    """Append one turn to history file."""
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"---\nuser: {user}\nassistant: {assistant}\n")
    except OSError:
        pass


def _slash_help() -> str:
    """Help text for slash commands."""
    return """Slash commands:
  /clear, /new   Clear conversation history and start fresh
  /help          Show this help
  /model [name]  Show or set Ollama model (e.g. /model llama3.2)
  /fix           Run linter, send errors to model for suggested fix
  /test          Run tests, send failures to model
  /summary [N]   Summarize last N turns and replace with one summary (default 5)
  /quit, /exit   Exit (or use Ctrl+C, empty line)
"""


def _run_command_sync(workspace_root: str, command: str) -> str:
    """Run command in workspace_root; return combined stdout + stderr. Empty on failure."""
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


def _handle_slash(
    line: str,
    model_ref: list[str],
    message_history: list[dict],
    workspace_root: str = ".",
    linter_command: str | None = None,
    test_command: str | None = None,
) -> str | None | tuple[str, str] | tuple[Literal["run_summary"], int]:
    """
    Handle slash command. model_ref is [current_model]; message_history is the conversation list to clear.
    Returns new model if /model was used, "cleared" if /clear or /new, "help" if /help, "quit" if /quit,
    ("run_prompt", prompt) for /fix or /test, ("run_summary", n) for /summary, else None.
    """
    line = line.strip()
    if not line.startswith("/"):
        return None
    parts = line.split(maxsplit=1)
    cmd = (parts[0] or "").lower()
    rest = (parts[1] or "").strip() if len(parts) > 1 else ""
    if cmd in ("/clear", "/new"):
        message_history.clear()
        print("[OllamaCode] Conversation cleared.", flush=True)
        return "cleared"
    if cmd == "/help":
        print(_slash_help(), flush=True)
        return "help"
    if cmd == "/model":
        if rest:
            model_ref[0] = rest
            print(f"[OllamaCode] Model set to: {rest}", flush=True)
            return rest
        print(f"[OllamaCode] Current model: {model_ref[0]}", flush=True)
        return "help"
    if cmd == "/fix":
        run_cmd = linter_command or "ruff check ."
        output = _run_command_sync(workspace_root, run_cmd)
        if not output:
            print(
                "[OllamaCode] No linter output (command may have succeeded or not run).",
                flush=True,
            )
            return "help"
        prompt = f"Fix these linter errors (from `{run_cmd}`):\n\n```\n{output}\n```"
        return ("run_prompt", prompt)
    if cmd == "/test":
        run_cmd = test_command or "pytest"
        output = _run_command_sync(workspace_root, run_cmd)
        if not output:
            print(
                "[OllamaCode] No test output (command may have succeeded or not run).",
                flush=True,
            )
            return "help"
        prompt = f"Fix these test failures (from `{run_cmd}`):\n\n```\n{output}\n```"
        return ("run_prompt", prompt)
    if cmd == "/summary":
        try:
            n = int(rest) if rest else 5
            n = max(1, min(n, 50))
        except ValueError:
            n = 5
        return ("run_summary", n)
    if cmd in ("/quit", "/exit"):
        return "quit"
    print(f"[OllamaCode] Unknown command: {cmd}. Use /help for commands.", flush=True)
    return "help"


async def _run(
    model: str,
    mcp_servers: list[dict],
    system_extra: str,
    query: str | None,
    stream: bool,
    tui: bool = False,
    max_messages: int = 0,
    history_file: str | None = None,
    max_tool_rounds: int = 20,
    max_tool_result_chars: int = 0,
    quiet: bool = False,
    timing: bool = False,
    rules_file: str | None = None,
    file_path: str | None = None,
    lines_spec: str | None = None,
    apply_edits_flag: bool = False,
    max_edits_per_request: int = 0,
    apply_edits_dry_run: bool = False,
    linter_command: str | None = None,
    test_command: str | None = None,
    semantic_codebase_hint: bool = True,
    auto_summarize_after_turns: int = 0,
    branch_context: bool = False,
    branch_context_base: str = "main",
    pr_description_file: str | None = None,
) -> None:
    use_mcp = bool(mcp_servers)

    # Inject workspace root so MCP servers (fs, terminal, codebase) run in the directory from which the CLI was started
    workspace_root = os.getcwd()
    mcp_servers = [
        {
            **entry,
            "env": {**(entry.get("env") or {}), "OLLAMACODE_FS_ROOT": workspace_root},
        }
        if (entry.get("type") or "stdio").lower() == "stdio"
        else entry
        for entry in mcp_servers
    ]

    if tui and not query:
        try:
            from .tui import run_tui
        except ImportError as e:
            print(
                "TUI requires rich. Install with: pip install ollamacode[tui]",
                file=sys.stderr,
            )
            raise SystemExit(1) from e
        try:
            if use_mcp:
                if len(mcp_servers) == 1 and mcp_servers[0].get("type") == "stdio":
                    cmd = mcp_servers[0].get("command", "python")
                    args = mcp_servers[0].get("args") or []
                    env = mcp_servers[0].get("env")
                    session_ctx = connect_mcp_stdio(cmd, args, env=env)
                else:
                    session_ctx = connect_mcp_servers(mcp_servers)
                async with session_ctx as session:
                    show_semantic_hint = semantic_codebase_hint and not any(
                        "semantic" in (s.get("name") or "").lower() for s in mcp_servers
                    )
                    await run_tui(
                        session,
                        model,
                        system_extra,
                        quiet=quiet,
                        max_tool_rounds=max_tool_rounds,
                        max_messages=max_messages,
                        max_tool_result_chars=max_tool_result_chars,
                        timing=timing,
                        workspace_root=workspace_root,
                        linter_command=linter_command,
                        test_command=test_command,
                        show_semantic_hint=show_semantic_hint,
                    )
            else:
                await run_tui(
                    None,
                    model,
                    system_extra,
                    quiet=quiet,
                    max_tool_rounds=max_tool_rounds,
                    max_messages=max_messages,
                    max_tool_result_chars=max_tool_result_chars,
                    timing=timing,
                    workspace_root=workspace_root,
                    linter_command=linter_command,
                    test_command=test_command,
                    show_semantic_hint=False,
                )
        except ImportError as e:
            print(
                "TUI requires rich. Install with: pip install ollamacode[tui]",
                file=sys.stderr,
            )
            raise SystemExit(1) from e
        return
    _SYSTEM = (
        "You are a coding assistant with full access to the workspace. You are given a list of available tools with their names "
        "and descriptions—use whichever tools fit the task. When the user asks you to run something, check something, or change "
        "something, use the appropriate tool and report the result."
    )
    if system_extra:
        _SYSTEM = _SYSTEM + "\n\n" + system_extra
    if apply_edits_flag:
        _SYSTEM = (
            _SYSTEM
            + '\n\nWhen you change files, output edits in this format so the user can apply or reject: <<EDITS>>\n[JSON array of { "path": "file path", "oldText": "optional exact old content", "newText": "new content" }]\n<<END>>\nIf replacing the whole file, omit oldText. Paths are relative to the workspace.'
        )
    if rules_file:
        rules_path = (
            Path(rules_file)
            if Path(rules_file).is_absolute()
            else Path(workspace_root) / rules_file
        )
        if rules_path.exists():
            _SYSTEM = _SYSTEM + "\n\n--- Project rules ---\n\n" + rules_path.read_text()
    if branch_context or pr_description_file:
        branch_ctx = get_branch_context(
            workspace_root, branch_context_base, pr_description_file
        )
        if branch_ctx:
            _SYSTEM = (
                _SYSTEM
                + "\n\n--- Branch/PR context (what we're working on) ---"
                + branch_ctx
            )

    async def _do_chat(
        conn: McpConnection | None,
        q: str,
        current_model: str,
        message_history: list[dict],
    ) -> str:
        q = expand_at_refs(q, workspace_root)
        if conn is not None:
            out = await run_agent_loop(
                conn,
                current_model,
                q,
                system_prompt=_SYSTEM,
                max_messages=max_messages,
                max_tool_rounds=max_tool_rounds,
                max_tool_result_chars=max_tool_result_chars,
                message_history=message_history if message_history else None,
                quiet=quiet,
                timing=timing,
            )
        else:
            out = await run_agent_loop_no_mcp(
                current_model,
                q,
                system_prompt=_SYSTEM,
                message_history=message_history if message_history else None,
            )
        if history_file:
            _append_history(history_file, q, out)
        return out

    async def _do_chat_stream(
        conn: McpConnection | None,
        q: str,
        current_model: str,
        message_history: list[dict],
    ) -> str:
        buf: list[str] = []
        q_expanded = expand_at_refs(q, workspace_root)
        if conn is not None:
            async for frag in run_agent_loop_stream(
                conn,
                current_model,
                q_expanded,
                system_prompt=_SYSTEM,
                max_messages=max_messages,
                max_tool_rounds=max_tool_rounds,
                max_tool_result_chars=max_tool_result_chars,
                message_history=message_history if message_history else None,
                quiet=quiet,
                timing=timing,
            ):
                print(frag, end="", flush=True)
                buf.append(frag)
            print(flush=True)
        else:
            async for frag in run_agent_loop_no_mcp_stream(
                current_model,
                q,
                system_prompt=_SYSTEM,
                message_history=message_history if message_history else None,
            ):
                print(frag, end="", flush=True)
                buf.append(frag)
            print(flush=True)
        return "".join(buf)

    def _maybe_apply_edits(response: str) -> None:
        if not apply_edits_flag:
            return
        edits = parse_edits(response)
        if not edits:
            return
        if max_edits_per_request > 0 and len(edits) > max_edits_per_request:
            print(
                f"[OllamaCode] Too many edits ({len(edits)} > max_edits_per_request={max_edits_per_request}). Skipping apply.",
                file=sys.stderr,
            )
            return
        print(file=sys.stderr)
        print(format_edits_diff(edits, workspace_root), file=sys.stderr)
        if apply_edits_dry_run:
            print("[OllamaCode] Dry run: no edits applied.", file=sys.stderr)
            return
        try:
            ans = input("Apply these edits? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        if ans in ("y", "yes"):
            n = apply_edits(edits, workspace_root)
            print(f"[OllamaCode] Applied {n} edit(s).", file=sys.stderr)

    async def _do_summary(
        conn: McpConnection | None,
        current_model: str,
        history: list[dict],
        n_turns: int,
    ) -> None:
        """Summarize last n_turns user+assistant pairs and replace them with one summary pair."""
        n_msgs = min(n_turns * 2, len(history))
        if n_msgs == 0:
            print("[OllamaCode] No conversation to summarize.", flush=True)
            return
        transcript_parts = []
        for i in range(-n_msgs, 0):
            m = history[i]
            role = "User" if m.get("role") == "user" else "Assistant"
            transcript_parts.append(f"{role}: {m.get('content', '')}")
        transcript = "\n\n".join(transcript_parts)
        prompt = f"Summarize the following conversation in one short paragraph. Reply with only the summary, no preamble:\n\n{transcript}"
        if not quiet:
            print("[OllamaCode] Summarizing last", n_turns, "turn(s)...", flush=True)
        summary = await _do_chat(conn, prompt, current_model, [])
        history[:] = history[:-n_msgs] + [
            {"role": "user", "content": "Summary of previous conversation"},
            {"role": "assistant", "content": summary.strip()},
        ]
        print(
            "[OllamaCode] Replaced last", n_turns, "turn(s) with summary.", flush=True
        )

    async def _do_auto_summarize_oldest(
        conn: McpConnection | None,
        current_model: str,
        history: list[dict],
        after_turns: int,
    ) -> None:
        """When history has >= after_turns turns, summarize the oldest few turns into one pair."""
        if after_turns <= 0 or len(history) < after_turns * 2:
            return
        n_to_summarize = min(4, max(1, after_turns // 2))
        n_msgs = n_to_summarize * 2
        transcript_parts = []
        for i in range(n_msgs):
            m = history[i]
            role = "User" if m.get("role") == "user" else "Assistant"
            transcript_parts.append(f"{role}: {m.get('content', '')}")
        transcript = "\n\n".join(transcript_parts)
        prompt = f"Summarize the following conversation in one short paragraph. Reply with only the summary, no preamble:\n\n{transcript}"
        if not quiet:
            print(
                "[OllamaCode] Auto-summarizing oldest",
                n_to_summarize,
                "turn(s)...",
                flush=True,
            )
        summary = await _do_chat(conn, prompt, current_model, [])
        history[:] = [
            {"role": "user", "content": "Summary of earlier conversation"},
            {"role": "assistant", "content": summary.strip()},
        ] + history[n_msgs:]
        if not quiet:
            print(
                "[OllamaCode] Replaced oldest",
                n_to_summarize,
                "turn(s) with summary.",
                flush=True,
            )

    if not use_mcp:
        if query:
            q = (
                prepend_file_context(query, file_path, workspace_root, lines_spec)
                if file_path
                else query
            )
            if stream:
                out = await _do_chat_stream(None, q, model, [])
                _maybe_apply_edits(out)
            else:
                out = await _do_chat(None, q, model, [])
                print(out)
                _maybe_apply_edits(out)
        else:
            print(
                "OllamaCode (Ollama only, no MCP tools). /help for commands. Empty line or Ctrl+C to exit."
            )
            message_history: list[dict] = []
            model_ref = [model]
            while True:
                try:
                    line = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not line:
                    continue
                result = _handle_slash(
                    line,
                    model_ref,
                    message_history,
                    workspace_root=workspace_root,
                    linter_command=linter_command,
                    test_command=test_command,
                )
                if result == "quit":
                    break
                if result is not None:
                    if isinstance(result, tuple) and result[0] == "run_prompt":
                        line = cast(str, result[1])
                    elif isinstance(result, tuple) and result[0] == "run_summary":
                        await _do_summary(
                            None, model_ref[0], message_history, cast(int, result[1])
                        )
                        continue
                    else:
                        continue
                await _do_auto_summarize_oldest(
                    None, model_ref[0], message_history, auto_summarize_after_turns
                )
                if stream:
                    out = await _do_chat_stream(
                        None, line, model_ref[0], message_history
                    )
                    message_history.append({"role": "user", "content": line})
                    message_history.append({"role": "assistant", "content": out})
                    if history_file:
                        _append_history(history_file, line, out)
                else:
                    out = await _do_chat(None, line, model_ref[0], message_history)
                    message_history.append({"role": "user", "content": line})
                    message_history.append({"role": "assistant", "content": out})
                    print("Assistant:", out, sep="\n")
        return

    if len(mcp_servers) == 1 and mcp_servers[0].get("type") == "stdio":
        cmd = mcp_servers[0].get("command", "python")
        args = mcp_servers[0].get("args") or []
        env = mcp_servers[0].get("env")
        session_ctx = connect_mcp_stdio(cmd, args, env=env)
    else:
        session_ctx = connect_mcp_servers(mcp_servers)

    if query:
        async with session_ctx as session:
            q = (
                prepend_file_context(query, file_path, workspace_root, lines_spec)
                if file_path
                else query
            )
            if stream:
                out = await _do_chat_stream(session, q, model, [])
                _maybe_apply_edits(out)
            else:
                out = await _do_chat(session, q, model, [])
                print(out)
                _maybe_apply_edits(out)
        return

    # Interactive: connect MCP on first message that needs tools (lazy)
    print(
        "OllamaCode (local model + MCP tools). /help for commands. Empty line or Ctrl+C to exit."
    )
    if semantic_codebase_hint and not any(
        "semantic" in (s.get("name") or "").lower() for s in mcp_servers
    ):
        print(
            "[OllamaCode] Tip: For semantic codebase search, add a semantic MCP server to config. See docs/MCP_SERVERS.md.",
            file=sys.stderr,
        )
    message_history_mcp: list[dict] = []
    model_ref = [model]
    session: McpConnection | None = None
    try:
        while True:
            try:
                line = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            result = _handle_slash(
                line,
                model_ref,
                message_history_mcp,
                workspace_root=workspace_root,
                linter_command=linter_command,
                test_command=test_command,
            )
            if result == "quit":
                break
            if result is not None:
                if isinstance(result, tuple) and result[0] == "run_prompt":
                    line = cast(str, result[1])
                elif isinstance(result, tuple) and result[0] == "run_summary":
                    if session is None:
                        session = await session_ctx.__aenter__()
                    await _do_summary(
                        session, model_ref[0], message_history_mcp, cast(int, result[1])
                    )
                    continue
                else:
                    continue
            if session is None:
                session = await session_ctx.__aenter__()
            await _do_auto_summarize_oldest(
                session, model_ref[0], message_history_mcp, auto_summarize_after_turns
            )
            if stream:
                out = await _do_chat_stream(
                    session, line, model_ref[0], message_history_mcp
                )
                message_history_mcp.append({"role": "user", "content": line})
                message_history_mcp.append({"role": "assistant", "content": out})
                if history_file:
                    _append_history(history_file, line, out)
            else:
                out = await _do_chat(session, line, model_ref[0], message_history_mcp)
                message_history_mcp.append({"role": "user", "content": line})
                message_history_mcp.append({"role": "assistant", "content": out})
                print("Assistant:", out, sep="\n")
    finally:
        if session is not None:
            await session_ctx.__aexit__(None, None, None)


def main() -> None:
    args = _parse_args()
    if getattr(args, "no_stream", False):
        args.stream = False
    if args.query == "convert-mcp":
        try:
            from .convert_mcp import run_convert
        except ImportError as e:
            print(
                "convert-mcp requires PyYAML. Install with: pip install pyyaml",
                file=sys.stderr,
            )
            raise SystemExit(1) from e
        try:
            run_convert(
                getattr(args, "convert_mcp_input", None),
                getattr(args, "convert_mcp_output", None),
            )
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error: {e}", file=sys.stderr)
            raise SystemExit(1) from e
        return
    if args.query == "serve":
        try:
            from .serve import run_serve
        except ImportError as e:
            print(
                "Serve requires uvicorn and starlette. Install with: pip install ollamacode[server]",
                file=sys.stderr,
            )
            raise SystemExit(1) from e
        port = getattr(args, "port", 8000)
        run_serve(port=port, config_path=args.config)
        return
    if args.query == "protocol":
        import contextlib

        from .protocol_server import run_protocol_stdio

        mcp_servers, _, _ = _resolve_mcp_servers(
            args.config, args.mcp_command, args.mcp_args
        )
        if getattr(args, "no_mcp", False):
            mcp_servers = []
        config = load_config(args.config)
        merged = merge_config_with_env(
            config,
            model_env=os.environ.get("OLLAMACODE_MODEL"),
            mcp_args_env=os.environ.get("OLLAMACODE_MCP_ARGS"),
            system_extra_env=os.environ.get("OLLAMACODE_SYSTEM_EXTRA"),
        )
        model = (
            args.model
            or merged.get("model")
            or os.environ.get("OLLAMACODE_MODEL", "gpt-oss:20b")
        )
        system_extra = (merged.get("system_prompt_extra") or "").strip()
        max_messages = merged.get("max_messages", 0)
        max_tool_result_chars = merged.get("max_tool_result_chars", 0)
        workspace_root = os.getcwd()
        mcp_servers = [
            {
                **entry,
                "env": {**(entry.get("env") or {}), "OLLAMACODE_FS_ROOT": workspace_root},
            }
            if (entry.get("type") or "stdio").lower() == "stdio"
            else entry
            for entry in mcp_servers
        ]
        _check_ollama_and_model(model, getattr(args, "quiet", False))

        @contextlib.asynccontextmanager
        async def _session_ctx():
            if not mcp_servers:
                yield None
                return
            if len(mcp_servers) == 1 and mcp_servers[0].get("type") == "stdio":
                cmd = mcp_servers[0].get("command", "python")
                args_list = mcp_servers[0].get("args") or []
                env = mcp_servers[0].get("env")
                ctx = connect_mcp_stdio(cmd, args_list, env=env)
            else:
                ctx = connect_mcp_servers(mcp_servers)
            async with ctx as session:
                yield session

        async def _protocol_main() -> None:
            async with _session_ctx() as session:
                await run_protocol_stdio(
                    session,
                    model,
                    system_extra,
                    max_messages=max_messages,
                    max_tool_result_chars=max_tool_result_chars,
                    workspace_root=workspace_root,
                )

        asyncio.run(_protocol_main())
        return
    mcp_servers, _, using_builtin = _resolve_mcp_servers(
        args.config, args.mcp_command, args.mcp_args
    )
    if getattr(args, "no_mcp", False):
        mcp_servers = []
        using_builtin = False
    config = load_config(args.config)
    merged = merge_config_with_env(
        config,
        model_env=os.environ.get("OLLAMACODE_MODEL"),
        mcp_args_env=os.environ.get("OLLAMACODE_MCP_ARGS"),
        system_extra_env=os.environ.get("OLLAMACODE_SYSTEM_EXTRA"),
    )
    model = (
        args.model
        or merged.get("model")
        or os.environ.get("OLLAMACODE_MODEL", "gpt-oss:20b")
    )
    system_extra = (merged.get("system_prompt_extra") or "").strip()
    max_messages = (
        args.max_messages
        if args.max_messages is not None
        else merged.get("max_messages", 0)
    )
    max_tool_rounds = (
        args.max_tool_rounds
        if args.max_tool_rounds is not None
        else merged.get("max_tool_rounds", 20)
    )
    max_tool_result_chars = merged.get("max_tool_result_chars", 0)
    quiet = getattr(args, "quiet", False)
    timing = getattr(args, "timing", False) or merged.get("timing", False)
    if using_builtin and not quiet:
        print(
            "[OllamaCode] Using built-in MCP (fs, terminal, codebase, tools, git). Use ollamacode.yaml to customize.",
            file=sys.stderr,
            flush=True,
        )
    try:
        _check_ollama_and_model(model, quiet)
    except SystemExit:
        raise
    try:
        rules_file = merged.get("rules_file")
        asyncio.run(
            _run(
                model,
                mcp_servers,
                system_extra,
                args.query,
                args.stream,
                args.tui,
                max_messages,
                args.history_file,
                max_tool_rounds=max_tool_rounds,
                max_tool_result_chars=max_tool_result_chars,
                quiet=quiet,
                timing=timing,
                rules_file=rules_file,
                file_path=args.file,
                lines_spec=args.lines,
                apply_edits_flag=args.apply_edits,
                max_edits_per_request=merged.get("max_edits_per_request", 0),
                apply_edits_dry_run=getattr(args, "apply_edits_dry_run", False),
                linter_command=merged.get("linter_command"),
                test_command=merged.get("test_command"),
                semantic_codebase_hint=merged.get("semantic_codebase_hint", True),
                auto_summarize_after_turns=merged.get("auto_summarize_after_turns", 0),
                branch_context=merged.get("branch_context", False),
                branch_context_base=merged.get("branch_context_base", "main"),
                pr_description_file=merged.get("pr_description_file"),
            )
        )
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
