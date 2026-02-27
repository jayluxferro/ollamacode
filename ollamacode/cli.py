"""
CLI for OllamaCode: chat with local Ollama + MCP tools.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from typing import Any, Literal
import asyncio
import json
from datetime import datetime, timezone
import os
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .agent import (
    _tool_call_one_line,
    run_agent_loop,
    run_agent_loop_no_mcp,
    run_agent_loop_no_mcp_stream,
    run_agent_loop_stream,
)
from .providers.base import BaseProvider
from .providers import get_provider
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

logger = logging.getLogger(__name__)


def _check_provider_connectivity(
    provider: "BaseProvider | None",
    model: str,
    quiet: bool,
    provider_name: str = "ollama",
) -> None:
    """Fail fast if the configured provider is unreachable or the Ollama model is missing."""
    if os.environ.get("OLLAMACODE_SKIP_MODEL_CHECK"):
        return
    if provider is not None and provider_name != "ollama":
        ok, msg = provider.health_check()
        if not ok:
            print(f"Provider check failed: {msg}", file=sys.stderr)
            raise SystemExit(1)
        if not quiet:
            print(f"[OllamaCode] Provider: {provider_name} ({msg})", file=sys.stderr)
        return
    _check_ollama_and_model(model, quiet)


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
            logger.debug("Ollama error: %s", e)
            print("Ollama error: could not connect.", file=sys.stderr)
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
        help="Model name (overrides config and OLLAMACODE_MODEL)",
    )
    p.add_argument(
        "--provider",
        default=os.environ.get("OLLAMACODE_PROVIDER"),
        metavar="NAME",
        help=(
            "AI provider name (default: ollama). "
            "Options: ollama, openai, groq, deepseek, anthropic, openrouter, mistral, "
            "xai, together, fireworks, perplexity, venice, cohere, cloudflare_ai, custom. "
            "Override with OLLAMACODE_PROVIDER env var."
        ),
    )
    p.add_argument(
        "--base-url",
        default=os.environ.get("OLLAMACODE_BASE_URL"),
        metavar="URL",
        help="Override base URL for the provider (e.g. http://localhost:11434 for Ollama, or any OpenAI-compat endpoint). Override with OLLAMACODE_BASE_URL.",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("OLLAMACODE_API_KEY"),
        metavar="KEY",
        help="API key for the provider. Override with OLLAMACODE_API_KEY (or provider-specific: GROQ_API_KEY, OPENAI_API_KEY, etc.).",
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
        "--voice-in",
        action="store_true",
        help="Record from microphone and use transcription as the prompt (local).",
    )
    p.add_argument(
        "--voice-out",
        action="store_true",
        help="Speak the final response using local TTS.",
    )
    p.add_argument(
        "--voice-seconds",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Voice input record duration (default 5s).",
    )
    p.add_argument(
        "--voice-model",
        type=str,
        default="base",
        metavar="NAME",
        help="Whisper model for voice input (default: base).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Timeout for each Ollama request in seconds.",
    )
    p.add_argument(
        "--tool-timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Timeout for each tool call in seconds (sets OLLAMACODE_TOOL_TIMEOUT_SECONDS).",
    )
    p.add_argument(
        "--tool-budget",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Per-turn tool budget in seconds (sets OLLAMACODE_TOOL_BUDGET_SECONDS).",
    )
    p.add_argument(
        "--run-budget",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Total run budget in seconds (sets OLLAMACODE_RUN_BUDGET_SECONDS).",
    )
    p.add_argument(
        "--eval-file",
        type=str,
        default=None,
        metavar="PATH",
        help="Eval cases JSON file (for query=evals). Default: evals/basic.json",
    )
    p.add_argument(
        "--eval-json",
        action="store_true",
        help="Emit eval summary as JSON to stdout.",
    )
    p.add_argument(
        "--stream-with-tools",
        action="store_true",
        help="Enable streaming even when tool calls are available (less reliable).",
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
        "--sandbox",
        choices=["readonly", "supervised", "full"],
        default=None,
        metavar="LEVEL",
        help="Sandbox level for MCP servers: readonly (no writes/commands), supervised (default, workspace-scoped), full (unrestricted).",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8000,
        metavar="N",
        help="Port for 'serve' command (default 8000).",
    )
    p.add_argument(
        "--no-tunnel",
        action="store_true",
        dest="no_tunnel",
        help="Disable tunnel for 'serve' command even if tunnel is configured in ollamacode.yaml.",
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
        help="Single query to run, 'serve' for HTTP API, 'protocol' for stdio JSON-RPC, 'health' to check provider, 'init' to scaffold, 'tutorial' for interactive tutorial, 'setup' for interactive setup wizard, 'secrets' to manage encrypted secrets (--secret-action set/get/list/delete), 'reindex' to rebuild vector memory index, 'repo-map' to generate a repo map, 'cron' for scheduled tasks (cron list / cron run <name>), 'install-browsers' to install Playwright Chromium, 'discover-tools' to scan PATH, 'list-tools' to list MCP tools, 'list-mcp' to list MCP servers, or 'convert-mcp' to convert MCP config. Use --rlm for RLM mode.",
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
    # --- secrets subcommand args ---
    p.add_argument(
        "--secret-action",
        choices=["set", "get", "list", "delete"],
        default=None,
        metavar="ACTION",
        help="For 'secrets': action to perform (set, get, list, delete).",
    )
    p.add_argument(
        "--secret-name",
        default=None,
        metavar="NAME",
        help="For 'secrets set/get/delete': the secret name.",
    )
    p.add_argument(
        "--secret-value",
        default=None,
        metavar="VALUE",
        help="For 'secrets set': the plaintext secret value.",
    )
    # --- reindex (vector memory) ---
    p.add_argument(
        "--reindex-workspace",
        default=None,
        metavar="DIR",
        help="For 'reindex': workspace root to index (default: current directory).",
    )
    p.add_argument(
        "--repo-map-output",
        default=None,
        metavar="PATH",
        help="For 'repo-map': output path (default: .ollamacode/repo_map.md).",
    )
    p.add_argument(
        "--repo-map-max-files",
        default=None,
        type=int,
        metavar="N",
        help="For 'repo-map': max files to include (default: 200).",
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
        logger.debug("Failed to list tools: %s", e)
        print("[OllamaCode] Failed to list tools.", file=sys.stderr)
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
    auto_check_after_edits: bool = False,
    auto_check_mode: str = "test",
    auto_check_fix_on_fail: bool = False,
    auto_check_fix_max_rounds: int = 1,
    headless: bool = False,
    run_timeout: float | None = None,
    autonomous_mode: bool = False,
    subagents: list[dict[str, Any]] | None = None,
    provider: "BaseProvider | None" = None,
    voice_in: bool = False,
    voice_out: bool = False,
    voice_seconds: float = 5.0,
    voice_model: str = "base",
    eval_file: str = "evals/basic.json",
    eval_json: bool = False,
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
    if voice_in:
        try:
            from .voice import record_and_transcribe

            query = record_and_transcribe(seconds=voice_seconds, model=voice_model)
            if not query:
                print("[OllamaCode] Voice input produced empty text.", file=sys.stderr)
                return 1
        except Exception as e:
            logger.debug("Voice input failed: %s", e)
            print("[OllamaCode] Voice input failed.", file=sys.stderr)
            return 1

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
    if query == "evals":
        pass
    elif not query:
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
                        provider=provider,
                        provider_name=provider.name
                        if provider is not None
                        else "ollama",
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
                    provider=provider,
                    provider_name=provider.name if provider is not None else "ollama",
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

    def _should_plan_exec_verify(text: str) -> bool:
        if not text:
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
        if os.environ.get("OLLAMACODE_ROUTER", "0") != "1":
            return default_model
        fast = os.environ.get("OLLAMACODE_MODEL_FAST", "").strip() or default_model
        strong = os.environ.get("OLLAMACODE_MODEL_STRONG", "").strip() or default_model
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
            return strong
        if words <= 6:
            return fast
        return default_model

    def _local_math_answer(text: str) -> str | None:
        import re

        s = text.strip().lower()
        m = re.match(r"^(what is\s+)?(\d+)\s*([+\-*/])\s*(\d+)\s*\??$", s)
        if not m:
            return None
        a = int(m.group(2))
        b = int(m.group(4))
        op = m.group(3)
        if op == "+":
            return str(a + b)
        if op == "-":
            return str(a - b)
        if op == "*":
            return str(a * b)
        if op == "/" and b != 0:
            return str(a / b)
        return None

    async def _do_chat(
        conn: McpConnection | None,
        q: str,
        current_model: str,
        message_history: list[dict],
        system_prompt_override: str | None = None,
    ) -> str:
        local = _local_math_answer(q)
        if local is not None:
            return local
        request_id = uuid.uuid4().hex
        q = expand_at_refs(q, workspace_root)
        sys_prompt = system_prompt_override or _SYSTEM
        last_system_prompt[0] = sys_prompt
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
        plan_exec_verify = os.environ.get("OLLAMACODE_PLAN_EXECUTE_VERIFY", "0") == "1"
        if plan_exec_verify and _should_plan_exec_verify(q):
            plan_model = (
                os.environ.get("OLLAMACODE_PLAN_MODEL", "").strip() or current_model
            )
            verify_model = (
                os.environ.get("OLLAMACODE_VERIFY_MODEL", "").strip() or current_model
            )
            if not quiet:
                print("[OllamaCode] Planning...", file=sys.stderr, flush=True)
            plan = await run_agent_loop_no_mcp(
                plan_model,
                q,
                system_prompt=(
                    sys_prompt
                    + "\n\nYou are a planner. Produce a concise step-by-step plan (3-8 steps). "
                    "Do not call tools. Return the plan only."
                ),
                message_history=None,
                provider=provider,
                timing=timing,
                request_id=request_id,
            )
            plan = (plan or "").strip()
            exec_prompt = (
                f"Goal:\n{q}\n\nPlan:\n{plan}\n\n"
                "Execute the plan step-by-step. Provide the final answer only."
            )
            if conn is not None:
                run_coro = run_agent_loop(
                    conn,
                    current_model,
                    exec_prompt,
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
                    before_tool_call=_before_tool_call_cli
                    if confirm_tool_calls
                    else None,
                    on_tool_start=lambda n, a: _emit_tool_trace("tool_start", n, a),
                    on_tool_end=lambda n, a, s: _emit_tool_trace("tool_end", n, a, s),
                    provider=provider,
                    request_id=request_id,
                )
                out = (
                    await asyncio.wait_for(run_coro, timeout=timeout_seconds)
                    if timeout_seconds and timeout_seconds > 0
                    else await run_coro
                )
            else:
                run_coro = run_agent_loop_no_mcp(
                    current_model,
                    exec_prompt,
                    system_prompt=sys_prompt,
                    message_history=message_history if message_history else None,
                    provider=provider,
                    timing=timing,
                    request_id=request_id,
                )
                out = (
                    await asyncio.wait_for(run_coro, timeout=timeout_seconds)
                    if timeout_seconds and timeout_seconds > 0
                    else await run_coro
                )
            if not quiet:
                print("[OllamaCode] Verifying...", file=sys.stderr, flush=True)
            verify = await run_agent_loop_no_mcp(
                verify_model,
                "Review the following answer for correctness and completeness. "
                "Output only the revised answer, no preamble.\n\n" + (out or ""),
                system_prompt="You are a verifier. Output only the revised answer.",
                message_history=None,
                provider=provider,
                timing=timing,
                request_id=request_id,
            )
            out = (verify or "").strip()
            if history_file:
                _append_history(history_file, q, out)
            return out
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
                provider=provider,
                request_id=request_id,
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
                provider=provider,
                timing=timing,
                request_id=request_id,
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
                provider=provider,
                timing=timing,
                request_id=request_id,
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
        local = _local_math_answer(q)
        if local is not None:
            print(local, flush=True)
            return local
        plan_exec_verify = os.environ.get("OLLAMACODE_PLAN_EXECUTE_VERIFY", "0") == "1"
        if plan_exec_verify and _should_plan_exec_verify(q):
            if not quiet:
                print(
                    "[OllamaCode] Plan/execute/verify enabled; running non-streamed.",
                    file=sys.stderr,
                    flush=True,
                )
            out = await _do_chat(
                conn, q, current_model, message_history, system_prompt_override
            )
            print(out, flush=True)
            return out
        request_id = uuid.uuid4().hex
        buf: list[str] = []
        q_expanded = expand_at_refs(q, workspace_root)
        sys_prompt = system_prompt_override or _SYSTEM
        last_system_prompt[0] = sys_prompt
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
                        provider=provider,
                        request_id=request_id,
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
                    provider=provider,
                    request_id=request_id,
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
                        provider=provider,
                        timing=timing,
                        request_id=request_id,
                    ):
                        print(frag, end="", flush=True)
                        buf.append(frag)
            else:
                async for frag in run_agent_loop_no_mcp_stream(
                    current_model,
                    q,
                    system_prompt=sys_prompt,
                    message_history=message_history if message_history else None,
                    provider=provider,
                    timing=timing,
                    request_id=request_id,
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
            model_for = _route_model(q, current_model)
            if stream and not json_output:
                out = await _do_chat_stream(conn, q, model_for, [])
                out = _strip_and_show_reasoning(out)
                await _maybe_apply_edits(out, conn)
                if voice_out:
                    try:
                        from .voice import speak_text

                        speak_text(out)
                    except Exception as e:
                        logger.debug("Voice output failed: %s", e)
                        print("[OllamaCode] Voice output failed.", file=sys.stderr)
                return
            out = await _do_chat(conn, q, model_for, [])
            out = _strip_and_show_reasoning(out)
            if json_output:
                _emit_json_response(out)
            else:
                print(out)
            if voice_out:
                try:
                    from .voice import speak_text

                    speak_text(out)
                except Exception as e:
                    logger.debug("Voice output failed: %s", e)
                    print("[OllamaCode] Voice output failed.", file=sys.stderr)
            await _maybe_apply_edits(out, conn)
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

    async def _run_evals(
        conn: McpConnection | None,
        current_model: str,
        eval_file: str = "evals/basic.json",
        eval_json: bool = False,
    ) -> int:
        eval_path = eval_file
        path = Path(eval_path)
        if not path.exists():
            print(f"[OllamaCode] Eval file not found: {eval_path}", file=sys.stderr)
            return 1
        try:
            cases = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.debug("Failed to parse eval file: %s", e)
            print("[OllamaCode] Failed to parse eval file.", file=sys.stderr)
            return 1
        if not isinstance(cases, list):
            print("[OllamaCode] Eval file must be a JSON list.", file=sys.stderr)
            return 1
        failures = 0
        total = 0
        total_time = 0.0
        failed_names: list[str] = []
        case_stats: list[dict[str, Any]] = []
        for i, case in enumerate(cases, 1):
            if not isinstance(case, dict):
                failures += 1
                continue
            name = case.get("name", f"case_{i}")
            prompt = case.get("prompt") or ""
            expect_contains = case.get("expect_contains") or []
            forbid = case.get("forbid") or []
            if not prompt:
                failures += 1
                print(f"[eval] {name}: missing prompt", file=sys.stderr)
                continue
            total += 1
            t0 = time.perf_counter()
            try:
                out = await _do_chat(conn, prompt, current_model, [])
            except Exception as exc:
                dur = time.perf_counter() - t0
                total_time += dur
                failures += 1
                failed_names.append(name)
                print(f"[eval] {name}: ERROR ({exc})", file=sys.stderr)
                case_stats.append(
                    {"name": name, "duration_s": round(dur, 3), "ok": False}
                )
                continue
            dur = time.perf_counter() - t0
            total_time += dur
            ok = True
            for s in expect_contains:
                if s and s not in out:
                    ok = False
            for s in forbid:
                if s and s in out:
                    ok = False
            if ok:
                print(f"[eval] {name}: PASS")
            else:
                failures += 1
                failed_names.append(name)
                print(f"[eval] {name}: FAIL", file=sys.stderr)
            case_stats.append(
                {
                    "name": name,
                    "duration_s": round(dur, 3),
                    "ok": ok,
                }
            )
        passed = total - failures
        pass_rate = (passed / total * 100.0) if total else 0.0
        avg = (total_time / total) if total else 0.0
        summary = {
            "total": total,
            "passed": passed,
            "failed": failures,
            "pass_rate": round(pass_rate, 2),
            "avg_seconds": round(avg, 3),
            "failed_cases": failed_names,
            "cases": case_stats,
        }
        if eval_json:
            print(json.dumps(summary, ensure_ascii=False))
        else:
            print(
                f"[eval] summary: {passed}/{total} passed ({pass_rate:.1f}%), avg {avg:.2f}s",
                file=sys.stderr,
            )
            slowest = sorted(case_stats, key=lambda c: c["duration_s"], reverse=True)[
                :3
            ]
            if slowest:
                lines = ", ".join(f"{c['name']}={c['duration_s']}s" for c in slowest)
                print(f"[eval] slowest: {lines}", file=sys.stderr)
        return 0 if failures == 0 else 1

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

    last_system_prompt: list[str] = [""]

    async def _maybe_auto_fix_after_checks(
        conn: McpConnection | None,
        check_label: str,
        output: str,
        *,
        max_rounds: int,
    ) -> None:
        if not auto_check_fix_on_fail or max_rounds <= 0:
            return
        for _ in range(max_rounds):
            prompt = (
                f"{check_label} failed. Fix the issues. "
                "Return updates using <<EDITS>> JSON. "
                "Failure output:\n\n" + output[:6000]
            )
            sys_prompt = last_system_prompt[0] or _SYSTEM
            if conn is not None:
                fix = await run_agent_loop(
                    conn,
                    model,
                    prompt,
                    system_prompt=sys_prompt,
                    max_messages=max_messages,
                    max_tool_rounds=max_tool_rounds,
                    max_tool_result_chars=max_tool_result_chars,
                    message_history=None,
                    quiet=quiet,
                    timing=timing,
                    allowed_tools=allowed_tools,
                    blocked_tools=blocked_tools,
                    confirm_tool_calls=confirm_tool_calls,
                    before_tool_call=_before_tool_call_cli
                    if confirm_tool_calls
                    else None,
                    provider=provider,
                )
            else:
                fix = await run_agent_loop_no_mcp(
                    model,
                    prompt,
                    system_prompt=sys_prompt,
                    message_history=None,
                    provider=provider,
                    timing=timing,
                )
            await _maybe_apply_edits(fix, conn, allow_fix=False)
            ok, _, output = await _run_post_checks(conn)
            if ok:
                break

    async def _run_post_checks(conn: McpConnection | None) -> tuple[bool, str, str]:
        mode = (auto_check_mode or "test").strip().lower()
        commands: list[tuple[str, str | None]] = []
        if mode in ("lint", "linter", "both"):
            commands.append(("lint", linter_command))
        if mode in ("test", "tests", "both"):
            commands.append(("test", test_command))
        if not commands:
            return True, "", ""
        for label, cmd in commands:
            if not cmd:
                continue
            print(f"[OllamaCode] Auto-check: running {label} -> {cmd}", file=sys.stderr)
            if conn is not None:
                try:
                    from .mcp_client import call_tool, tool_result_to_content

                    tool_name = "run_tests" if label == "test" else "run_linter"
                    result = await call_tool(conn, tool_name, {"command": cmd})
                    content = tool_result_to_content(result)
                    if content:
                        print(content, file=sys.stderr)
                    if getattr(result, "isError", False):
                        return False, label, content
                except Exception:
                    pass
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=workspace_root,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                out = (result.stdout or "").strip()
                err = (result.stderr or "").strip()
                combined = "\n".join([out, err]).strip()
                if out:
                    print(out, file=sys.stderr)
                if err:
                    print(err, file=sys.stderr)
                print(
                    f"[OllamaCode] Auto-check {label} exit {result.returncode}",
                    file=sys.stderr,
                )
                if result.returncode != 0:
                    return False, label, combined
            except Exception as e:
                return False, label, str(e)
        return True, "", ""

    async def _maybe_apply_edits(
        response: str,
        conn: McpConnection | None,
        *,
        allow_fix: bool = True,
    ) -> None:
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
                if n > 0 and auto_check_after_edits:
                    ok, label, output = await _run_post_checks(conn)
                    if not ok and allow_fix:
                        await _maybe_auto_fix_after_checks(
                            conn, label, output, max_rounds=auto_check_fix_max_rounds
                        )
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
        if query == "evals":
            return await _run_evals(
                None, model, eval_file=eval_file, eval_json=eval_json
            )
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

    if query == "evals":
        async with session_ctx as session:
            return await _run_evals(
                session, model, eval_file=eval_file, eval_json=eval_json
            )
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
    if args.query in ("setup", "wizard"):
        from .setup_wizard import run_setup_wizard

        workspace = getattr(args, "dest", None) or os.getcwd()
        run_setup_wizard(workspace_override=workspace)
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
    if args.query == "secrets":
        from .secrets import delete_secret, get_secret, list_secrets, set_secret

        action = getattr(args, "secret_action", None) or "list"
        name = getattr(args, "secret_name", None) or ""
        value = getattr(args, "secret_value", None) or ""

        if action == "list":
            names = list_secrets()
            if names:
                print("Stored secrets:")
                for n in names:
                    print(f"  {n}")
            else:
                print("No secrets stored.")
        elif action == "set":
            if not name:
                print(
                    "Error: --secret-name is required for 'secrets set'",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            if not value:
                # Prompt interactively if value not given
                import getpass

                value = getpass.getpass(f"Value for secret '{name}': ")
            if not value:
                print("Error: secret value cannot be empty", file=sys.stderr)
                raise SystemExit(1)
            try:
                set_secret(name, value)
                print(f"Secret '{name}' stored.")
            except Exception as e:
                logger.debug("Error storing secret: %s", e)
                print("Error storing secret.", file=sys.stderr)
                raise SystemExit(1)
        elif action == "get":
            if not name:
                print(
                    "Error: --secret-name is required for 'secrets get'",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            try:
                result = get_secret(name)
            except Exception as e:
                logger.debug("Error reading secret: %s", e)
                print("Error reading secret.", file=sys.stderr)
                raise SystemExit(1)
            if result is None:
                print(f"Secret '{name}' not found.", file=sys.stderr)
                raise SystemExit(1)
            print(result)
        elif action == "delete":
            if not name:
                print(
                    "Error: --secret-name is required for 'secrets delete'",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            try:
                removed = delete_secret(name)
            except Exception as e:
                logger.debug("Error deleting secret: %s", e)
                print("Error deleting secret.", file=sys.stderr)
                raise SystemExit(1)
            if removed:
                print(f"Secret '{name}' deleted.")
            else:
                print(f"Secret '{name}' not found.", file=sys.stderr)
                raise SystemExit(1)
        return

    if args.query == "repo-map":
        from .repo_map import write_repo_map

        workspace = os.getcwd()
        out_path = getattr(args, "repo_map_output", None) or str(
            Path(workspace) / ".ollamacode" / "repo_map.md"
        )
        max_files = getattr(args, "repo_map_max_files", None) or 200
        dest = write_repo_map(workspace, out_path, max_files=max_files)
        print(f"[OllamaCode] Repo map written to {dest}", file=sys.stderr)
        return

    if args.query == "reindex":
        workspace = getattr(args, "reindex_workspace", None) or os.getcwd()
        print(f"Building vector memory index for: {workspace}", file=sys.stderr)
        try:
            from .vector_memory import build_vector_index

            result = build_vector_index(workspace)
            print(
                f"Indexed {result['indexed_files']} files, {result['chunk_count']} chunks → {result['db_path']}",
                file=sys.stderr,
            )
        except Exception as e:
            logger.debug("Reindex failed: %s", e)
            print("Reindex failed.", file=sys.stderr)
            raise SystemExit(1)
        return

    if args.query == "cron":
        from .scheduler import load_scheduled_tasks, run_task_now

        config = load_config(args.config)
        merged = merge_config_with_env(config, **get_env_config_overrides())
        workspace = os.getcwd()
        tasks = load_scheduled_tasks(merged, workspace)

        # Sub-action from --secret-action style, or deduce from remaining argv
        # Usage: ollamacode cron list | ollamacode cron run <name>
        # We detect via sys.argv manually since argparse has already consumed "cron".
        try:
            cron_idx = next(i for i, a in enumerate(sys.argv) if a == "cron")
            cron_rest = sys.argv[cron_idx + 1 :]
        except StopIteration:
            cron_rest = []

        sub = cron_rest[0] if cron_rest else "list"

        if sub == "list":
            if not tasks:
                print("No scheduled tasks configured.")
                print(
                    "Add tasks to ollamacode.yaml (scheduled_tasks:) or HEARTBEAT.md."
                )
            else:
                print(f"Scheduled tasks ({len(tasks)}):")
                for t in tasks:
                    name = t.get("name", "?")
                    desc = t.get("description", "")
                    interval = t.get("interval")
                    cron_expr = t.get("cron")
                    obs = t.get("observability", "noop")
                    sched = (
                        f"every {interval}s"
                        if interval
                        else (f"cron: {cron_expr}" if cron_expr else "no schedule")
                    )
                    line = f"  {name:<30} [{sched}] obs={obs}"
                    if desc:
                        line += f"\n    {desc}"
                    print(line)
        elif sub == "run":
            task_name = cron_rest[1] if len(cron_rest) > 1 else ""
            if not task_name:
                print("Usage: ollamacode cron run <task-name>", file=sys.stderr)
                raise SystemExit(1)
            matching = [t for t in tasks if t.get("name") == task_name]
            if not matching:
                print(
                    f"Task '{task_name}' not found. Use 'ollamacode cron list' to see available tasks.",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            task = matching[0]
            model_val = (
                args.model
                or merged.get("model")
                or os.environ.get("OLLAMACODE_MODEL", "gpt-oss:20b")
            )
            print(f"Running task '{task_name}'...", file=sys.stderr)
            try:
                output = run_task_now(task, model=model_val, config=merged)
                print(output)
            except Exception as e:
                logger.debug("Task failed: %s", e)
                print("Task failed.", file=sys.stderr)
                raise SystemExit(1)
        else:
            print(
                f"Unknown cron subcommand: {sub!r}. Use 'list' or 'run <name>'.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        return

    if args.query == "health":
        from .health import check_provider, check_toolchain_versions

        config = load_config(args.config)
        env_overrides = get_env_config_overrides()
        if getattr(args, "provider", None):
            env_overrides["provider_env"] = args.provider
        if getattr(args, "base_url", None):
            env_overrides["base_url_env"] = args.base_url
        if getattr(args, "api_key", None):
            env_overrides["api_key_env"] = args.api_key
        merged = merge_config_with_env(config, **env_overrides)
        ok, msg = check_provider(merged)
        print(msg, file=sys.stderr)
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
        no_tunnel = getattr(args, "no_tunnel", False)
        run_serve(port=port, config_path=args.config, no_tunnel=no_tunnel)
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
        _rlm_provider = (
            get_provider(merged)
            if merged.get("provider", "ollama") != "ollama"
            else None
        )
        _check_provider_connectivity(
            _rlm_provider,
            model,
            getattr(args, "quiet", False),
            merged.get("provider", "ollama"),
        )
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
        _protocol_provider = (
            get_provider(merged)
            if merged.get("provider", "ollama") != "ollama"
            else None
        )
        _check_provider_connectivity(
            _protocol_provider,
            model,
            getattr(args, "quiet", False),
            merged.get("provider", "ollama"),
        )

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
    _env_overrides = get_env_config_overrides()
    # CLI flags for provider/base-url/api-key override env vars
    if getattr(args, "provider", None):
        _env_overrides["provider_env"] = args.provider
    if getattr(args, "base_url", None):
        _env_overrides["base_url_env"] = args.base_url
    if getattr(args, "api_key", None):
        _env_overrides["api_key_env"] = args.api_key
    merged = merge_config_with_env(config, **_env_overrides)
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
    # Apply sandbox level: --sandbox CLI flag > ollamacode.yaml sandbox_level > env var (default supervised).
    _sandbox_level = (
        getattr(args, "sandbox", None)
        or merged.get("sandbox_level")
        or os.environ.get("OLLAMACODE_SANDBOX_LEVEL")
    )
    if _sandbox_level:
        os.environ["OLLAMACODE_SANDBOX_LEVEL"] = _sandbox_level

    # Build provider from merged config (Ollama is the default; returns None to preserve old path)
    _provider_name = merged.get("provider", "ollama")
    _main_provider: BaseProvider | None = (
        get_provider(merged) if _provider_name != "ollama" else None
    )
    # Commands that don't need model connectivity at startup
    _skip_conn_check = args.query in (
        "evals",
        "health",
        "init",
        "setup",
        "wizard",
        "secrets",
        "reindex",
        "repo-map",
        "cron",
        "convert-mcp",
        "tutorial",
    )
    if not _skip_conn_check:
        try:
            _check_provider_connectivity(_main_provider, model, quiet, _provider_name)
        except SystemExit:
            raise
    try:
        # Model-aware timeout defaults for Ollama (unless explicitly set via --timeout).
        if getattr(args, "timeout", None) is None:
            model_name = (
                args.model
                or merged.get("model")
                or os.environ.get("OLLAMACODE_MODEL", "")
            ).lower()
            if any(k in model_name for k in ("70b", "65b", "405b", "120b")):
                os.environ.setdefault("OLLAMACODE_OLLAMA_TIMEOUT_SECONDS", "180")
            elif any(k in model_name for k in ("13b", "8b", "7b")):
                os.environ.setdefault("OLLAMACODE_OLLAMA_TIMEOUT_SECONDS", "120")
            else:
                os.environ.setdefault("OLLAMACODE_OLLAMA_TIMEOUT_SECONDS", "150")
        if getattr(args, "stream_with_tools", False):
            os.environ["OLLAMACODE_STREAM_WITH_TOOLS"] = "1"
        if getattr(args, "tool_timeout", None) is not None:
            os.environ["OLLAMACODE_TOOL_TIMEOUT_SECONDS"] = str(args.tool_timeout)
        if getattr(args, "tool_budget", None) is not None:
            os.environ["OLLAMACODE_TOOL_BUDGET_SECONDS"] = str(args.tool_budget)
        if getattr(args, "run_budget", None) is not None:
            os.environ["OLLAMACODE_RUN_BUDGET_SECONDS"] = str(args.run_budget)

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
            auto_check_after_edits=merged.get("auto_check_after_edits", False),
            auto_check_mode=merged.get("auto_check_mode", "test"),
            headless=getattr(args, "headless", False),
            run_timeout=run_timeout,
            autonomous_mode=getattr(args, "autonomous_mode", False),
            subagents=merged.get("subagents") or [],
            provider=_main_provider,
            voice_in=getattr(args, "voice_in", False),
            voice_out=getattr(args, "voice_out", False),
            voice_seconds=getattr(args, "voice_seconds", 5.0),
            voice_model=getattr(args, "voice_model", "base"),
            eval_file=getattr(args, "eval_file", None) or "evals/basic.json",
            eval_json=getattr(args, "eval_json", False),
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
