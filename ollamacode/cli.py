"""
CLI for OllamaCode: chat with local Ollama + MCP tools.
"""

from __future__ import annotations

import argparse
import tempfile
from typing import Any, Literal
import asyncio
import json
from datetime import datetime, timezone
import os
import shlex
import subprocess
import sys
from pathlib import Path

from .agent import (
    _tool_call_one_line,
    run_agent_loop,
    run_agent_loop_no_mcp,
    run_agent_loop_no_mcp_stream,
    run_agent_loop_stream,
)
from .config import (
    find_config_file,
    get_env_config_overrides,
    get_resolved_config,
    load_config,
    merge_config_with_env,
)
from .context import (
    expand_at_refs,
    get_branch_context,
    load_ollama_md_context,
    prepend_file_context,
)
from .skills import load_skills_text
from .templates import load_prompt_template
from .edits import (
    apply_edits,
    format_edits_diff,
    parse_edits,
    parse_reasoning,
    parse_review,
)
from .memory import build_dynamic_memory_context
from .mcp_client import (
    McpConnection,
    connect_mcp_servers,
    connect_mcp_stdio,
    list_tools,
)


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
        formatter_class=argparse.RawTextHelpFormatter,
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
        "--python",
        default=os.environ.get("OLLAMACODE_PYTHON"),
        metavar="PATH",
        help="Python interpreter for built-in MCP servers (default: current interpreter). Override with OLLAMACODE_PYTHON.",
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
        "--verbose",
        "-v",
        action="store_true",
        help="Set log level to DEBUG (more verbose output).",
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
        "--json",
        action="store_true",
        dest="json_output",
        help="Print machine-readable JSON output for single-query mode.",
    )
    p.add_argument(
        "--pipe",
        action="store_true",
        dest="pipe_mode",
        help="Read prompt from stdin and print one response (single-query mode).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Timeout for each Ollama request in seconds.",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="CI/non-interactive mode: no TUI, explicit exit codes (0=success, 1=failure, 2=needs human). Implies --quiet.",
    )
    p.add_argument(
        "--run-timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Wall-clock timeout for the whole run (single-query/headless). Exit 1 on timeout.",
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Read-only run: block write_file and git add/commit/push. Use for safe CI or review.",
    )
    p.add_argument(
        "--max-tools",
        type=int,
        default=None,
        metavar="N",
        dest="max_tools",
        help="Max tool-call rounds for this run (overrides config max_tool_rounds). Alias for CI.",
    )
    p.add_argument(
        "--auto",
        action="store_true",
        dest="autonomous_mode",
        help="Autonomous mode: no per-tool confirmation, higher max_tool_rounds. Use for multi-step tasks without prompts.",
    )
    p.add_argument(
        "--tool-trace-json",
        action="store_true",
        dest="tool_trace_json",
        help="Emit structured tool trace events as JSON lines to stderr.",
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
        "--rlm",
        action="store_true",
        help="Run in RLM mode (recursive language model): context is not sent to the model, only metadata; model uses REPL and llm_query().",
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
        help="Single query to run, 'serve' for HTTP API, 'protocol' for stdio JSON-RPC, 'health' to check Ollama, 'init' to scaffold, 'tutorial' for interactive tutorial, 'install-browsers' to install Playwright Chromium (for screenshot tool), 'discover-tools' to scan PATH and emit MCP config snippet, 'list-tools' to list MCP tools, 'list-mcp' to list configured MCP servers, or 'convert-mcp' to convert MCP config. Use --rlm for RLM mode.",
    )
    p.add_argument(
        "--template",
        "-t",
        default=None,
        metavar="NAME",
        help="For 'init': template name (e.g. python-cli, python-lib). Use 'init' without --template to list.",
    )
    p.add_argument(
        "--dest",
        default=None,
        metavar="DIR",
        help="For 'init': destination directory (default: current directory).",
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


def _run_discover_tools() -> None:
    """Scan PATH for common dev executables and print a minimal MCP/config snippet."""
    import shutil

    common = (
        "python",
        "python3",
        "pytest",
        "ruff",
        "git",
        "node",
        "npm",
        "npx",
        "cargo",
        "go",
        "uv",
        "pip",
        "pip3",
        "make",
        "cmake",
        "npm",
        "yarn",
    )
    found: list[tuple[str, str]] = []
    for name in sorted(set(common)):
        path = shutil.which(name)
        if path:
            found.append((name, path))
    print("# Discovered executables (ollamacode discover-tools)", file=sys.stderr)
    print(
        "# Use run_command tool (built-in terminal MCP) to run these.", file=sys.stderr
    )
    if found:
        print(
            "\n# Add to ollamacode.yaml prompt_snippets or use as allowed_tools:",
            file=sys.stderr,
        )
        print("prompt_snippets:", file=sys.stderr)
        for name, path in found[:10]:
            print(f"  # - {name}: {path}", file=sys.stderr)
    print(
        "\n# Minimal MCP config (built-in already includes terminal with run_command):",
        file=sys.stderr,
    )
    print(
        "mcp_servers: []  # default includes fs, terminal, codebase, tools, git",
        file=sys.stderr,
    )
    for name, path in found:
        print(f"{name}: {path}", file=sys.stdout)


def _run_list_mcp(
    config_path: str | None,
    mcp_command: str,
    mcp_args: list[str],
    python_executable: str | None = None,
) -> None:
    """Print configured MCP servers (name, type, command/url)."""
    mcp_servers, _, _ = _resolve_mcp_servers(
        config_path, mcp_command, mcp_args, python_executable=python_executable
    )
    print("# MCP servers (ollamacode list-mcp)", file=sys.stderr)
    if not mcp_servers:
        print("(none)", file=sys.stdout)
        return
    for i, entry in enumerate(mcp_servers, 1):
        name = entry.get("name") or f"server-{i}"
        kind = (entry.get("type") or "stdio").lower()
        if kind == "stdio":
            cmd = entry.get("command", mcp_command)
            args = entry.get("args") or []
            print(f"{name}\tstdio\t{cmd} {' '.join(args)}".strip(), file=sys.stdout)
        else:
            url = entry.get("url", "")
            print(f"{name}\t{kind}\t{url}", file=sys.stdout)


async def _run_list_tools(
    config_path: str | None,
    mcp_command: str,
    mcp_args: list[str],
    python_executable: str | None = None,
) -> None:
    """Connect to MCP and print tool names (and short description)."""
    mcp_servers, use_mcp, _ = _resolve_mcp_servers(
        config_path, mcp_command, mcp_args, python_executable=python_executable
    )
    if not use_mcp or not mcp_servers:
        print(
            "[OllamaCode] No MCP servers configured. Add config or use default.",
            file=sys.stderr,
        )
        return
    workspace_root = os.path.abspath(os.getcwd())
    _env_base = dict(os.environ)
    servers_with_env = [
        {
            **entry,
            "env": {
                **_env_base,
                **(entry.get("env") or {}),
                "OLLAMACODE_FS_ROOT": workspace_root,
            },
        }
        if (entry.get("type") or "stdio").lower() == "stdio"
        else entry
        for entry in mcp_servers
    ]
    try:
        if len(servers_with_env) == 1 and servers_with_env[0].get("type") == "stdio":
            cmd = servers_with_env[0].get("command", "python")
            args = servers_with_env[0].get("args") or []
            env = servers_with_env[0].get("env")
            ctx = connect_mcp_stdio(cmd, args, env=env)
        else:
            ctx = connect_mcp_servers(servers_with_env)
        async with ctx as session:
            result = await list_tools(session)
            print("# MCP tools (ollamacode list-tools)", file=sys.stderr)
            for t in sorted(result.tools, key=lambda x: x.name):
                desc = (t.description or "").strip()
                first = desc.splitlines()[0][:80] if desc else ""
                print(f"{t.name}\t{first}", file=sys.stdout)
    except Exception as e:
        print(f"[OllamaCode] Failed to list tools: {e}", file=sys.stderr)
        raise SystemExit(1) from e


def _resolve_mcp_servers(
    config_path: str | None,
    mcp_command: str,
    mcp_args: list[str],
    python_executable: str | None = None,
) -> tuple[list[dict], bool, bool]:
    """Resolve mcp_servers from config and CLI. Returns (server_configs, use_mcp, using_builtin)."""
    overrides = get_env_config_overrides()
    overrides["python_executable"] = python_executable
    merged = get_resolved_config(config_path, **overrides)
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


async def _run(
    model: str,
    mcp_servers: list[dict],
    system_extra: str,
    query: str | None,
    stream: bool,
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
    docs_command: str | None = None,
    profile_command: str | None = None,
    semantic_codebase_hint: bool = True,
    auto_summarize_after_turns: int = 0,
    branch_context: bool = False,
    branch_context_base: str = "main",
    pr_description_file: str | None = None,
    use_skills: bool = True,
    prompt_template: str | None = None,
    inject_recent_context: bool = True,
    recent_context_max_files: int = 10,
    use_reasoning: bool = False,
    use_meta_reflection: bool = False,
    allowed_tools: list[str] | None = None,
    blocked_tools: list[str] | None = None,
    prompt_snippets: list[str] | None = None,
    confirm_tool_calls: bool = False,
    code_style: str | None = None,
    safety_output_patterns: list[str] | None = None,
    planner_model: str | None = None,
    executor_model: str | None = None,
    reviewer_model: str | None = None,
    multi_agent_max_iterations: int = 2,
    multi_agent_require_review: bool = True,
    tui_tool_trace_max: int = 5,
    tui_tool_log_max: int = 8,
    tui_tool_log_chars: int = 160,
    tui_refresh_hz: int = 5,
    json_output: bool = False,
    pipe_mode: bool = False,
    timeout_seconds: float | None = None,
    tool_trace_json: bool = False,
    memory_auto_context: bool = True,
    memory_kg_max_results: int = 4,
    memory_rag_max_results: int = 4,
    memory_rag_snippet_chars: int = 220,
    headless: bool = False,
    run_timeout: float | None = None,
    autonomous_mode: bool = False,
    subagents: list[dict[str, Any]] | None = None,
) -> int | None:
    use_mcp = bool(mcp_servers)
    # Autonomous: no confirm_tool_calls, allow more tool rounds.
    if autonomous_mode:
        confirm_tool_calls = False
        max_tool_rounds = max(max_tool_rounds, 30)
    # Headless/single-query exit code: 0 success, 1 failure, 2 needs human (e.g. apply-edits pending).
    exit_code_holder: list[int] = [0]

    if pipe_mode:
        try:
            piped = sys.stdin.read()
        except Exception:
            piped = ""
        query = (piped or "").strip()

    # Inject workspace root so MCP servers (fs, terminal, codebase) run in the directory from which the CLI was started.
    # Merge with os.environ so the subprocess has PATH etc. and reliably sees OLLAMACODE_FS_ROOT.
    workspace_root = os.path.abspath(os.getcwd())
    _env_base = dict(os.environ)
    mcp_servers = [
        {
            **entry,
            "env": {
                **_env_base,
                **(entry.get("env") or {}),
                "OLLAMACODE_FS_ROOT": workspace_root,
            },
        }
        if (entry.get("type") or "stdio").lower() == "stdio"
        else entry
        for entry in mcp_servers
    ]

    if not query and (json_output or pipe_mode):
        err = "No input provided. Use query arg or --pipe with stdin."
        if json_output:
            print(json.dumps({"error": err}, ensure_ascii=False))
        else:
            print(f"[OllamaCode] {err}", file=sys.stderr)
        return

    # Interactive mode (no query) always uses the TUI.
    if not query:
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
                        docs_command=docs_command,
                        profile_command=profile_command,
                        show_semantic_hint=show_semantic_hint,
                        use_skills=use_skills,
                        prompt_template=prompt_template,
                        inject_recent_context=inject_recent_context,
                        recent_context_max_files=recent_context_max_files,
                        branch_context=branch_context,
                        branch_context_base=branch_context_base,
                        use_reasoning=use_reasoning,
                        prompt_snippets=prompt_snippets,
                        allowed_tools=allowed_tools,
                        blocked_tools=blocked_tools,
                        confirm_tool_calls=confirm_tool_calls,
                        code_style=code_style,
                        planner_model=planner_model,
                        executor_model=executor_model,
                        reviewer_model=reviewer_model,
                        multi_agent_max_iterations=multi_agent_max_iterations,
                        multi_agent_require_review=multi_agent_require_review,
                        tui_tool_trace_max=tui_tool_trace_max,
                        tui_tool_log_max=tui_tool_log_max,
                        tui_tool_log_chars=tui_tool_log_chars,
                        tui_refresh_hz=tui_refresh_hz,
                        memory_auto_context=memory_auto_context,
                        memory_kg_max_results=memory_kg_max_results,
                        memory_rag_max_results=memory_rag_max_results,
                        memory_rag_snippet_chars=memory_rag_snippet_chars,
                        autonomous_mode=autonomous_mode,
                        subagents=subagents or [],
                    )
            else:
                await run_tui(
                    None,
                    model,
                    system_extra,
                    quiet=quiet,
                    max_tool_rounds=max_tool_rounds,
                    use_skills=use_skills,
                    prompt_template=prompt_template,
                    max_messages=max_messages,
                    max_tool_result_chars=max_tool_result_chars,
                    timing=timing,
                    workspace_root=workspace_root,
                    linter_command=linter_command,
                    test_command=test_command,
                    docs_command=docs_command,
                    profile_command=profile_command,
                    confirm_tool_calls=confirm_tool_calls,
                    code_style=code_style,
                    planner_model=planner_model,
                    executor_model=executor_model,
                    reviewer_model=reviewer_model,
                    multi_agent_max_iterations=multi_agent_max_iterations,
                    multi_agent_require_review=multi_agent_require_review,
                    tui_tool_trace_max=tui_tool_trace_max,
                    tui_tool_log_max=tui_tool_log_max,
                    tui_tool_log_chars=tui_tool_log_chars,
                    tui_refresh_hz=tui_refresh_hz,
                    memory_auto_context=memory_auto_context,
                    memory_kg_max_results=memory_kg_max_results,
                    memory_rag_max_results=memory_rag_max_results,
                    memory_rag_snippet_chars=memory_rag_snippet_chars,
                    show_semantic_hint=False,
                    autonomous_mode=autonomous_mode,
                    inject_recent_context=inject_recent_context,
                    recent_context_max_files=recent_context_max_files,
                    branch_context=branch_context,
                    branch_context_base=branch_context_base,
                    use_reasoning=use_reasoning,
                    prompt_snippets=prompt_snippets or [],
                    allowed_tools=allowed_tools,
                    blocked_tools=blocked_tools,
                    subagents=subagents or [],
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
        "something, use the appropriate tool and report the result. When generating code, include docstrings and brief comments where helpful."
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
    ollama_md = load_ollama_md_context(workspace_root)
    if ollama_md:
        _SYSTEM = _SYSTEM + "\n\n" + ollama_md
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
    from .state import (
        append_feedback,
        format_feedback_context,
        format_knowledge_context,
        format_past_errors_context,
        format_plan_context,
        format_preferences,
        format_recent_context,
        get_state,
    )

    state = get_state()
    if inject_recent_context:
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
            + '\n\nOptionally output structured reasoning for the user: <<REASONING>>\n{"steps": ["step1", "step2"], "conclusion": "..."}\n<<END>> before your reply, or call record_reasoning(steps, conclusion) to record your reasoning.'
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

    # Task-specific snippets for /fix, /test, /test-and-fix, /generate_docs, /review (adaptive prompt)
    _TASK_SNIPPETS: dict[str, str] = {
        "fix": "\n\n--- Task: fix linter errors ---\n\nFocus on the linter output; suggest minimal fixes. Output <<EDITS>> when changing files.",
        "test": "\n\n--- Task: fix test failures ---\n\nFocus on the test output; fix failing tests. Output <<EDITS>> when changing files.",
        "test-and-fix": "\n\n--- Task: self-test and self-repair ---\n\nFix the test failures. Output <<EDITS>> when changing files. After the user applies edits, they can run /test again; iterate until tests pass.",
        "generate_docs": "\n\n--- Task: generate or update documentation ---\n\nInspect existing docs and code with read_file/list_dir. Create or update README.md, docstrings, or mkdocs/sphinx structure as appropriate. Output <<EDITS>> for file changes.",
        "review": '\n\n--- Task: code review (explainable) ---\n\nReview the code. For each suggestion output structured feedback so the user sees suggestion and rationale. Output <<REVIEW>>\n{"suggestions": [{"location": "file or file:line", "suggestion": "what to change", "rationale": "why"}]}\n<<END>> You may also include a short prose summary before or after the block.',
        "refactor": "\n\n--- Task: refactor ---\n\nRefactor the code for clarity, performance, or structure. Preserve behavior. Output <<EDITS>> when changing files.",
        "generate_guide": "\n\n--- Task: generate user guide ---\n\nCreate or update a user guide (e.g. USER_GUIDE.md, docs/USAGE.md). Use read_file/list_dir to inspect the project, then write clear usage and examples. Output <<EDITS>> for file changes.",
        "add_tests": "\n\n--- Task: add or generate tests ---\n\nAdd unit or integration tests for the given code or path. Use read_file to inspect code, then write tests (e.g. pytest). Output <<EDITS>> for new or updated test files.",
    }

    def _edit_tool_args_in_editor(arguments: dict, root: str) -> dict | None:
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
                cwd=root,
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

    async def _before_tool_call_cli(tool_name: str, arguments: dict):
        """Prompt [y/N/e] per tool when confirm_tool_calls; e = edit args in $EDITOR. Returns run | skip | (edit, dict)."""
        if not quiet:
            print(
                f"[OllamaCode] Tool: {_tool_call_one_line(tool_name, arguments)}",
                file=sys.stderr,
                flush=True,
            )
        loop = asyncio.get_event_loop()
        while True:
            choice = await loop.run_in_executor(
                None, lambda: input("[y/N/e(dit)]? ").strip().lower() or "n"
            )
            if choice in ("y", "yes"):
                return "run"
            if choice in ("n", "no", ""):
                return "skip"
            if choice in ("e", "edit"):
                edited = _edit_tool_args_in_editor(arguments, workspace_root)
                if edited is not None:
                    return ("edit", edited)
                if not quiet:
                    print(
                        "[OllamaCode] Invalid JSON or cancel; running with original args.",
                        file=sys.stderr,
                    )
                return "run"
            if not quiet:
                print(
                    "[OllamaCode] Choose y (run), N (skip), or e (edit).",
                    file=sys.stderr,
                )

    def _emit_tool_trace(
        event: Literal["tool_start", "tool_end"],
        tool_name: str,
        arguments: dict,
        summary: str = "",
    ) -> None:
        if not tool_trace_json:
            return
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "tool": tool_name,
            "arguments": arguments,
        }
        if summary:
            payload["summary"] = summary
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)

    async def _do_chat(
        conn: McpConnection | None,
        q: str,
        current_model: str,
        message_history: list[dict],
        system_prompt_override: str | None = None,
    ) -> str:
        q = expand_at_refs(q, workspace_root)
        sys_prompt = system_prompt_override or _SYSTEM
        memory_block = (
            build_dynamic_memory_context(
                q,
                kg_max_results=memory_kg_max_results,
                rag_max_results=memory_rag_max_results,
                rag_snippet_chars=memory_rag_snippet_chars,
            )
            if memory_auto_context
            else ""
        )
        if memory_block:
            sys_prompt = (
                sys_prompt
                + "\n\n--- Retrieved memory (query-specific) ---\n\n"
                + memory_block
            )
        if conn is not None:
            run_coro = run_agent_loop(
                conn,
                current_model,
                q,
                system_prompt=sys_prompt,
                max_messages=max_messages,
                max_tool_rounds=max_tool_rounds,
                max_tool_result_chars=max_tool_result_chars,
                message_history=message_history if message_history else None,
                quiet=quiet,
                timing=timing,
                allowed_tools=allowed_tools,
                blocked_tools=blocked_tools,
                confirm_tool_calls=confirm_tool_calls,
                before_tool_call=_before_tool_call_cli if confirm_tool_calls else None,
                on_tool_start=lambda n, a: _emit_tool_trace("tool_start", n, a),
                on_tool_end=lambda n, a, s: _emit_tool_trace("tool_end", n, a, s),
            )
            out = (
                await asyncio.wait_for(run_coro, timeout=timeout_seconds)
                if timeout_seconds and timeout_seconds > 0
                else await run_coro
            )
        else:
            run_coro = run_agent_loop_no_mcp(
                current_model,
                q,
                system_prompt=sys_prompt,
                message_history=message_history if message_history else None,
            )
            out = (
                await asyncio.wait_for(run_coro, timeout=timeout_seconds)
                if timeout_seconds and timeout_seconds > 0
                else await run_coro
            )
        if use_meta_reflection and out and out.strip():
            if not quiet:
                print(
                    "[OllamaCode] Meta-reflection: reviewing reply...",
                    file=sys.stderr,
                    flush=True,
                )
            reflect_coro = run_agent_loop_no_mcp(
                current_model,
                "Review the following assistant reply for consistency, clarity, and errors. Output only the revised reply, no preamble or explanation.\n\n"
                + out,
                system_prompt="You are a reviewer. Output only the revised reply.",
                message_history=None,
            )
            out = (
                await asyncio.wait_for(reflect_coro, timeout=timeout_seconds)
                if timeout_seconds and timeout_seconds > 0
                else await reflect_coro
            )
            out = (out or "").strip()
        if history_file:
            _append_history(history_file, q, out)
        return out

    async def _do_chat_stream(
        conn: McpConnection | None,
        q: str,
        current_model: str,
        message_history: list[dict],
        system_prompt_override: str | None = None,
    ) -> str:
        buf: list[str] = []
        q_expanded = expand_at_refs(q, workspace_root)
        sys_prompt = system_prompt_override or _SYSTEM
        memory_block = (
            build_dynamic_memory_context(
                q_expanded,
                kg_max_results=memory_kg_max_results,
                rag_max_results=memory_rag_max_results,
                rag_snippet_chars=memory_rag_snippet_chars,
            )
            if memory_auto_context
            else ""
        )
        if memory_block:
            sys_prompt = (
                sys_prompt
                + "\n\n--- Retrieved memory (query-specific) ---\n\n"
                + memory_block
            )
        if conn is not None:
            if timeout_seconds and timeout_seconds > 0:
                async with asyncio.timeout(timeout_seconds):
                    async for frag in run_agent_loop_stream(
                        conn,
                        current_model,
                        q_expanded,
                        system_prompt=sys_prompt,
                        max_messages=max_messages,
                        max_tool_rounds=max_tool_rounds,
                        max_tool_result_chars=max_tool_result_chars,
                        message_history=message_history if message_history else None,
                        quiet=quiet,
                        timing=timing,
                        tool_progress_brief=True,
                        allowed_tools=allowed_tools,
                        blocked_tools=blocked_tools,
                        confirm_tool_calls=confirm_tool_calls,
                        before_tool_call=_before_tool_call_cli
                        if confirm_tool_calls
                        else None,
                        on_tool_start=lambda n, a: _emit_tool_trace("tool_start", n, a),
                        on_tool_end=lambda n, a, s: _emit_tool_trace(
                            "tool_end", n, a, s
                        ),
                    ):
                        print(frag, end="", flush=True)
                        buf.append(frag)
            else:
                async for frag in run_agent_loop_stream(
                    conn,
                    current_model,
                    q_expanded,
                    system_prompt=sys_prompt,
                    max_messages=max_messages,
                    max_tool_rounds=max_tool_rounds,
                    max_tool_result_chars=max_tool_result_chars,
                    message_history=message_history if message_history else None,
                    quiet=quiet,
                    timing=timing,
                    tool_progress_brief=True,
                    allowed_tools=allowed_tools,
                    blocked_tools=blocked_tools,
                    confirm_tool_calls=confirm_tool_calls,
                    before_tool_call=_before_tool_call_cli
                    if confirm_tool_calls
                    else None,
                    on_tool_start=lambda n, a: _emit_tool_trace("tool_start", n, a),
                    on_tool_end=lambda n, a, s: _emit_tool_trace("tool_end", n, a, s),
                ):
                    print(frag, end="", flush=True)
                    buf.append(frag)
            print(flush=True)
        else:
            if timeout_seconds and timeout_seconds > 0:
                async with asyncio.timeout(timeout_seconds):
                    async for frag in run_agent_loop_no_mcp_stream(
                        current_model,
                        q,
                        system_prompt=sys_prompt,
                        message_history=message_history if message_history else None,
                    ):
                        print(frag, end="", flush=True)
                        buf.append(frag)
            else:
                async for frag in run_agent_loop_no_mcp_stream(
                    current_model,
                    q,
                    system_prompt=sys_prompt,
                    message_history=message_history if message_history else None,
                ):
                    print(frag, end="", flush=True)
                    buf.append(frag)
            print(flush=True)
        return "".join(buf)

    def _emit_json_response(out_text: str) -> None:
        payload = {
            "content": out_text,
            "edits": parse_edits(out_text) or [],
        }
        if headless:
            payload["exit_code"] = exit_code_holder[0]
            payload["exit_reason"] = (
                "success" if exit_code_holder[0] == 0 else "needs_apply_edits"
            )
        print(json.dumps(payload, ensure_ascii=False))

    async def _do_single_query(
        conn: McpConnection | None,
        q: str,
        current_model: str,
    ) -> None:
        try:
            if stream and not json_output:
                out = await _do_chat_stream(conn, q, current_model, [])
                out = _strip_and_show_reasoning(out)
                _maybe_apply_edits(out)
                return
            out = await _do_chat(conn, q, current_model, [])
            out = _strip_and_show_reasoning(out)
            if json_output:
                _emit_json_response(out)
            else:
                print(out)
            _maybe_apply_edits(out)
        except asyncio.TimeoutError:
            exit_code_holder[0] = 1
            msg = (
                f"Request timed out after {timeout_seconds} seconds."
                if timeout_seconds and timeout_seconds > 0
                else "Request timed out."
            )
            if json_output:
                print(json.dumps({"error": msg, "exit_code": 1}, ensure_ascii=False))
            else:
                print(f"[OllamaCode] {msg}", file=sys.stderr)
        except Exception:
            exit_code_holder[0] = 1
            raise

    def _edit_edits_in_editor(edits: list[dict]) -> list[dict] | None:
        """Write edits JSON to a temp file, run $EDITOR, read back and return parsed list or None."""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(edits, f, indent=2)
                path = f.name
        except OSError:
            return None
        try:
            editor = os.environ.get("EDITOR", "nano")
            subprocess.run(
                shlex.split(editor) + [path],
                cwd=workspace_root,
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
        if not isinstance(data, list):
            return None
        out: list[dict] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            path_val = item.get("path")
            new_text = item.get("newText")
            if path_val is None or new_text is None:
                continue
            out.append(
                {
                    "path": str(path_val),
                    "oldText": item.get("oldText"),
                    "newText": new_text if isinstance(new_text, str) else str(new_text),
                }
            )
        return out if out else None

    def _strip_and_show_reasoning(raw: str) -> str:
        """Parse <<REASONING>> and <<REVIEW>> from raw reply; print to stderr if present; return content without blocks."""
        reasoning, content = parse_reasoning(raw)
        if reasoning:
            steps = reasoning.get("steps") or []
            conclusion = (reasoning.get("conclusion") or "").strip()
            if steps or conclusion:
                print("[OllamaCode] Reasoning:", file=sys.stderr)
                for i, s in enumerate(steps, 1):
                    print(f"  {i}. {s}", file=sys.stderr)
                if conclusion:
                    print(f"  Conclusion: {conclusion}", file=sys.stderr)
        review_sugs, content = parse_review(content)
        if review_sugs:
            print("[OllamaCode] Review suggestions:", file=sys.stderr)
            for i, s in enumerate(review_sugs, 1):
                loc = s.get("location") or ""
                sug = s.get("suggestion") or ""
                rat = s.get("rationale") or ""
                print(f"  {i}. [{loc}] {sug}", file=sys.stderr)
                if rat:
                    print(f"      Rationale: {rat}", file=sys.stderr)
        for pat in safety_output_patterns or []:
            if pat and pat in content:
                print(
                    "[OllamaCode] Output matched safety filter pattern; review before applying.",
                    file=sys.stderr,
                )
                break
        return content

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
        if headless:
            exit_code_holder[0] = 2
            if json_output:
                payload = {
                    "content": response,
                    "edits": edits,
                    "exit_reason": "needs_apply_edits",
                    "exit_code": 2,
                }
                print(json.dumps(payload, ensure_ascii=False))
            else:
                print(format_edits_diff(edits, workspace_root), file=sys.stderr)
                print(
                    "[OllamaCode] Headless: edits not applied (exit 2). Use non-headless or --apply-edits to apply.",
                    file=sys.stderr,
                )
            return
        print(file=sys.stderr)
        while True:
            print(format_edits_diff(edits, workspace_root), file=sys.stderr)
            if apply_edits_dry_run:
                print("[OllamaCode] Dry run: no edits applied.", file=sys.stderr)
                return
            try:
                ans = input("Apply these edits? [y/N/e(dit)]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return
            if ans in ("y", "yes"):
                n = apply_edits(edits, workspace_root)
                print(f"[OllamaCode] Applied {n} edit(s).", file=sys.stderr)
                append_feedback("edit_accepted", True, "user applied suggested edits")
                return
            if ans in ("n", "no", ""):
                return
            if ans in ("e", "edit"):
                edited = _edit_edits_in_editor(edits)
                if edited is not None:
                    edits = edited
                    print(file=sys.stderr)
                else:
                    print(
                        "[OllamaCode] Could not parse edited file; using previous edits.",
                        file=sys.stderr,
                    )
                continue
            print("[OllamaCode] Choose y, N, or e.", file=sys.stderr)

    if not use_mcp:
        if query:
            q = (
                prepend_file_context(query, file_path, workspace_root, lines_spec)
                if file_path
                else query
            )
            await _do_single_query(None, q, model)
            return exit_code_holder[0]
        return None

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
            await _do_single_query(session, q, model)
        return exit_code_holder[0]
    return None


def main() -> None:
    import logging

    args = _parse_args()
    log_level = (
        logging.DEBUG
        if getattr(args, "verbose", False)
        else (logging.WARNING if getattr(args, "quiet", False) else logging.INFO)
    )
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        stream=sys.stderr,
        force=True,
    )
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
    if args.query == "init":
        from .init_templates import run_init

        dest = getattr(args, "dest", None) or os.getcwd()
        template = getattr(args, "template", None)
        msg = run_init(template, dest)
        print(msg, file=sys.stderr if template else sys.stdout)
        return
    if args.query == "tutorial":
        from .tutorial import run_tutorial

        run_tutorial()
        return
    if args.query == "install-browsers":
        try:
            from .servers.screenshot_mcp import _install_chromium
        except ImportError:
            print(
                "[OllamaCode] Playwright not installed. Install with: pip install playwright",
                file=sys.stderr,
            )
            sys.exit(1)
        ok, msg = _install_chromium()
        if ok:
            print(f"[OllamaCode] {msg}", file=sys.stderr)
        else:
            print(f"[OllamaCode] Install failed: {msg}", file=sys.stderr)
            sys.exit(1)
        return
    if args.query == "health":
        from .health import check_ollama, check_toolchain_versions

        ok, msg = check_ollama()
        print(msg, file=sys.stderr)
        config = load_config(args.config)
        merged = merge_config_with_env(config)
        checks = merged.get("toolchain_version_checks") or []
        if checks:
            results = check_toolchain_versions(checks, cwd=os.getcwd())
            for r in results:
                status = "ok" if r.get("ok") else "mismatch"
                print(f"  {r.get('name', '?')}: {status}", file=sys.stderr)
                if not r.get("ok") and r.get("expect_contains"):
                    print(
                        f"    expected to contain: {r.get('expect_contains')!r}",
                        file=sys.stderr,
                    )
                    print(f"    actual: {r.get('actual', '')[:200]!r}", file=sys.stderr)
        raise SystemExit(0 if ok else 1)
    if args.query == "discover-tools":
        _run_discover_tools()
        return
    if args.query == "list-mcp":
        _run_list_mcp(
            args.config,
            args.mcp_command,
            args.mcp_args,
            python_executable=getattr(args, "python", None),
        )
        return
    if args.query == "list-tools":
        asyncio.run(
            _run_list_tools(
                args.config,
                args.mcp_command,
                args.mcp_args,
                python_executable=getattr(args, "python", None),
            )
        )
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
    # RLM reads prompt from query or stdin; interactive (no query) uses TUI only.
    use_rlm = getattr(args, "rlm", False)
    if use_rlm:
        from .context import prepend_file_context
        from .rlm import run_rlm_loop

        config = load_config(args.config)
        merged = merge_config_with_env(config, **get_env_config_overrides())
        model = (
            args.model
            or merged.get("model")
            or os.environ.get("OLLAMACODE_MODEL", "gpt-oss:20b")
        )
        # Prompt: from query if given, else stdin; then optional --file prepend
        context = args.query or ""
        if not context:
            try:
                context = sys.stdin.read()
            except Exception:
                context = ""
        workspace_root = os.getcwd()
        if args.file:
            context = prepend_file_context(
                context or "User request below.",
                args.file,
                workspace_root,
                args.lines,
            )
        if not (context or "").strip():
            print(
                "No prompt or file for RLM. Use: ollamacode --rlm 'prompt' or echo '...' | ollamacode --rlm",
                file=sys.stderr,
            )
            raise SystemExit(1)
        _check_ollama_and_model(model, getattr(args, "quiet", False))
        if getattr(args, "stream", False):
            from .rlm import run_rlm_loop_stream

            for event in run_rlm_loop_stream(
                context,
                model=model,
                sub_model=merged.get("rlm_sub_model"),
                max_iterations=merged.get("rlm_max_iterations", 20),
                stdout_max_chars=merged.get("rlm_stdout_max_chars", 2000),
                prefix_chars=merged.get("rlm_prefix_chars", 500),
                snippet_timeout_seconds=merged.get("rlm_snippet_timeout_seconds"),
                use_subprocess=merged.get("rlm_use_subprocess", False),
                subprocess_max_memory_mb=merged.get(
                    "rlm_subprocess_max_memory_mb", 512
                ),
                subprocess_max_cpu_seconds=merged.get(
                    "rlm_subprocess_max_cpu_seconds", 60
                ),
                quiet=getattr(args, "quiet", False),
            ):
                t = event.get("type")
                if t == "step" and not getattr(args, "quiet", False):
                    logging.getLogger("ollamacode.rlm").info(
                        "RLM step %s/%s...", event.get("step"), event.get("max")
                    )
                elif t == "repl_output" and not getattr(args, "quiet", False):
                    print(event.get("text", ""), file=sys.stderr, flush=True)
                elif t == "done":
                    print(event.get("content", ""))
                    return
                elif t == "error":
                    print(event.get("message", ""), file=sys.stderr)
                    print(event.get("message", ""))
                    raise SystemExit(1)
            return
        result = run_rlm_loop(
            context,
            model=model,
            sub_model=merged.get("rlm_sub_model"),
            max_iterations=merged.get("rlm_max_iterations", 20),
            stdout_max_chars=merged.get("rlm_stdout_max_chars", 2000),
            prefix_chars=merged.get("rlm_prefix_chars", 500),
            snippet_timeout_seconds=merged.get("rlm_snippet_timeout_seconds"),
            use_subprocess=merged.get("rlm_use_subprocess", False),
            subprocess_max_memory_mb=merged.get("rlm_subprocess_max_memory_mb", 512),
            subprocess_max_cpu_seconds=merged.get("rlm_subprocess_max_cpu_seconds", 60),
            quiet=getattr(args, "quiet", False),
        )
        print(result)
        return
    if args.query == "protocol":
        import contextlib

        from .protocol_server import run_protocol_stdio

        mcp_servers, _, _ = _resolve_mcp_servers(
            args.config,
            args.mcp_command,
            args.mcp_args,
            python_executable=getattr(args, "python", None),
        )
        if getattr(args, "no_mcp", False):
            mcp_servers = []
        config = load_config(args.config)
        merged = merge_config_with_env(config, **get_env_config_overrides())
        model = (
            args.model
            or merged.get("model")
            or os.environ.get("OLLAMACODE_MODEL", "gpt-oss:20b")
        )
        system_extra = (merged.get("system_prompt_extra") or "").strip()
        max_messages = merged.get("max_messages", 0)
        max_tool_result_chars = merged.get("max_tool_result_chars", 0)
        workspace_root = os.path.abspath(os.getcwd())
        _env_base = dict(os.environ)
        mcp_servers = [
            {
                **entry,
                "env": {
                    **_env_base,
                    **(entry.get("env") or {}),
                    "OLLAMACODE_FS_ROOT": workspace_root,
                },
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
                    use_skills=merged.get("use_skills", True),
                    prompt_template=merged.get("prompt_template"),
                    inject_recent_context=merged.get("inject_recent_context", True),
                    recent_context_max_files=merged.get("recent_context_max_files", 10),
                    use_reasoning=merged.get("use_reasoning", True),
                    prompt_snippets=merged.get("prompt_snippets") or [],
                    allowed_tools=merged.get("allowed_tools"),
                    blocked_tools=merged.get("blocked_tools"),
                    code_style=merged.get("code_style"),
                    confirm_tool_calls=merged.get("confirm_tool_calls", False),
                    memory_auto_context=merged.get("memory_auto_context", True),
                    memory_kg_max_results=merged.get("memory_kg_max_results", 4),
                    memory_rag_max_results=merged.get("memory_rag_max_results", 4),
                    memory_rag_snippet_chars=merged.get(
                        "memory_rag_snippet_chars", 220
                    ),
                )

        asyncio.run(_protocol_main())
        return
    mcp_servers, _, using_builtin = _resolve_mcp_servers(
        args.config,
        args.mcp_command,
        args.mcp_args,
        python_executable=getattr(args, "python", None),
    )
    if getattr(args, "no_mcp", False):
        mcp_servers = []
        using_builtin = False
    config = load_config(args.config)
    merged = merge_config_with_env(config, **get_env_config_overrides())
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
    max_tool_rounds = merged.get("max_tool_rounds", 20)
    if args.max_tool_rounds is not None:
        max_tool_rounds = args.max_tool_rounds
    if getattr(args, "max_tools", None) is not None:
        max_tool_rounds = args.max_tools
    max_tool_result_chars = merged.get("max_tool_result_chars", 0)
    quiet = getattr(args, "quiet", False) or getattr(args, "headless", False)
    blocked_tools_merged = list(merged.get("blocked_tools") or [])
    if getattr(args, "no_write", False):
        for t in ("write_file", "git_add", "git_commit", "git_push"):
            if t not in blocked_tools_merged:
                blocked_tools_merged.append(t)
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
        run_timeout = getattr(args, "run_timeout", None)
        coro = _run(
            model,
            mcp_servers,
            system_extra,
            args.query,
            args.stream,
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
            docs_command=merged.get("docs_command", "mkdocs build"),
            profile_command=merged.get("profile_command"),
            semantic_codebase_hint=merged.get("semantic_codebase_hint", True),
            auto_summarize_after_turns=merged.get("auto_summarize_after_turns", 0),
            branch_context=merged.get("branch_context", True),
            branch_context_base=merged.get("branch_context_base", "main"),
            pr_description_file=merged.get("pr_description_file"),
            use_skills=merged.get("use_skills", True),
            prompt_template=merged.get("prompt_template"),
            inject_recent_context=merged.get("inject_recent_context", True),
            recent_context_max_files=merged.get("recent_context_max_files", 10),
            use_reasoning=merged.get("use_reasoning", True),
            use_meta_reflection=merged.get("use_meta_reflection", True),
            allowed_tools=merged.get("allowed_tools"),
            blocked_tools=blocked_tools_merged if blocked_tools_merged else None,
            prompt_snippets=merged.get("prompt_snippets") or [],
            confirm_tool_calls=merged.get("confirm_tool_calls", False),
            code_style=merged.get("code_style"),
            safety_output_patterns=merged.get("safety_output_patterns") or [],
            planner_model=merged.get("planner_model"),
            executor_model=merged.get("executor_model"),
            reviewer_model=merged.get("reviewer_model"),
            multi_agent_max_iterations=merged.get("multi_agent_max_iterations", 2),
            multi_agent_require_review=merged.get("multi_agent_require_review", True),
            tui_tool_trace_max=merged.get("tui_tool_trace_max", 5),
            tui_tool_log_max=merged.get("tui_tool_log_max", 8),
            tui_tool_log_chars=merged.get("tui_tool_log_chars", 160),
            tui_refresh_hz=merged.get("tui_refresh_hz", 5),
            json_output=getattr(args, "json_output", False),
            pipe_mode=getattr(args, "pipe_mode", False),
            timeout_seconds=getattr(args, "timeout", None),
            tool_trace_json=getattr(args, "tool_trace_json", False),
            memory_auto_context=merged.get("memory_auto_context", True),
            memory_kg_max_results=merged.get("memory_kg_max_results", 4),
            memory_rag_max_results=merged.get("memory_rag_max_results", 4),
            memory_rag_snippet_chars=merged.get("memory_rag_snippet_chars", 220),
            headless=getattr(args, "headless", False),
            run_timeout=run_timeout,
            autonomous_mode=getattr(args, "autonomous_mode", False),
            subagents=merged.get("subagents") or [],
        )
        if run_timeout and args.query:

            async def _run_with_timeout() -> int | None:
                return await asyncio.wait_for(coro, timeout=run_timeout)

            try:
                result = asyncio.run(_run_with_timeout())
            except asyncio.TimeoutError:
                if getattr(args, "json_output", False):
                    print(
                        json.dumps(
                            {
                                "error": f"Run timed out after {run_timeout}s",
                                "exit_code": 1,
                            },
                            ensure_ascii=False,
                        )
                    )
                else:
                    print(
                        f"[OllamaCode] Run timed out after {run_timeout} seconds.",
                        file=sys.stderr,
                    )
                sys.exit(1)
        else:
            result = asyncio.run(coro)
        if result is not None:
            sys.exit(result)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
