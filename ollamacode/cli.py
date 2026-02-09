"""
CLI for OllamaCode: chat with local Ollama + MCP tools.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from .agent import (
    run_agent_loop,
    run_agent_loop_no_mcp,
    run_agent_loop_no_mcp_stream,
    run_agent_loop_stream,
)
from .config import load_config, merge_config_with_env
from .mcp_client import McpConnection, connect_mcp_servers, connect_mcp_stdio


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
        "--port",
        type=int,
        default=8000,
        metavar="N",
        help="Port for 'serve' command (default 8000).",
    )
    p.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Single query to run, or 'serve' to start the local HTTP API.",
    )
    return p.parse_args()


def _resolve_mcp_servers(
    config_path: str | None,
    mcp_command: str,
    mcp_args: list[str],
) -> tuple[list[dict], bool]:
    """Resolve mcp_servers from config and CLI. Returns (server_configs, use_mcp)."""
    config = load_config(config_path)
    merged = merge_config_with_env(
        config,
        model_env=os.environ.get("OLLAMACODE_MODEL"),
        mcp_args_env=os.environ.get("OLLAMACODE_MCP_ARGS"),
        system_extra_env=os.environ.get("OLLAMACODE_SYSTEM_EXTRA"),
    )
    servers = merged.get("mcp_servers") or []
    # CLI legacy: --mcp-args (or env) overrides config when non-empty
    cli_args = mcp_args if mcp_args else (os.environ.get("OLLAMACODE_MCP_ARGS") or "").split()
    if cli_args:
        servers = [{"type": "stdio", "command": mcp_command, "args": cli_args}]
    return servers, bool(servers)


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
  /quit, /exit   Exit (or use Ctrl+C, empty line)
"""


def _handle_slash(
    line: str,
    model_ref: list[str],
    message_history: list[dict],
) -> str | None:
    """
    Handle slash command. model_ref is [current_model]; message_history is the conversation list to clear.
    Returns new model if /model was used, "cleared" if /clear or /new, "help" if /help, "quit" if /quit, else None.
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
) -> None:
    use_mcp = bool(mcp_servers)

    if tui and not query:
        try:
            from .tui import run_tui
        except ImportError as e:
            print("TUI requires rich. Install with: pip install ollamacode[tui]", file=sys.stderr)
            raise SystemExit(1) from e
        try:
            if use_mcp:
                if len(mcp_servers) == 1 and mcp_servers[0].get("type") == "stdio":
                    cmd = mcp_servers[0].get("command", "python")
                    args = mcp_servers[0].get("args") or []
                    session_ctx = connect_mcp_stdio(cmd, args)
                else:
                    session_ctx = connect_mcp_servers(mcp_servers)
                async with session_ctx as session:
                    await run_tui(session, model, system_extra)
            else:
                await run_tui(None, model, system_extra)
        except ImportError as e:
            print("TUI requires rich. Install with: pip install ollamacode[tui]", file=sys.stderr)
            raise SystemExit(1) from e
        return
    _SYSTEM = "You are a helpful coding assistant. Use the available tools when they would help."
    if system_extra:
        _SYSTEM = _SYSTEM + "\n\n" + system_extra

    async def _do_chat(
        conn: McpConnection | None,
        q: str,
        current_model: str,
        message_history: list[dict],
    ) -> str:
        if conn is not None:
            out = await run_agent_loop(
                conn,
                current_model,
                q,
                system_prompt=_SYSTEM,
                max_messages=max_messages,
                message_history=message_history if message_history else None,
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
        if conn is not None:
            async for frag in run_agent_loop_stream(
                conn,
                current_model,
                q,
                system_prompt=_SYSTEM,
                max_messages=max_messages,
                message_history=message_history if message_history else None,
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

    if not use_mcp:
        if query:
            if stream:
                await _do_chat_stream(None, query, model, [])
            else:
                print(await _do_chat(None, query, model, []))
        else:
            print("OllamaCode (Ollama only, no MCP tools). /help for commands. Empty line or Ctrl+C to exit.")
            message_history: list[dict] = []
            model_ref = [model]
            while True:
                try:
                    line = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not line:
                    continue
                result = _handle_slash(line, model_ref, message_history)
                if result == "quit":
                    break
                if result is not None:
                    continue
                if stream:
                    out = await _do_chat_stream(None, line, model_ref[0], message_history)
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
        session_ctx = connect_mcp_stdio(cmd, args)
    else:
        session_ctx = connect_mcp_servers(mcp_servers)

    async with session_ctx as session:
        if query:
            if stream:
                await _do_chat_stream(session, query, model, [])
            else:
                print(await _do_chat(session, query, model, []))
            return
        print("OllamaCode (local model + MCP tools). /help for commands. Empty line or Ctrl+C to exit.")
        message_history_mcp: list[dict] = []
        model_ref = [model]
        while True:
            try:
                line = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            result = _handle_slash(line, model_ref, message_history_mcp)
            if result == "quit":
                break
            if result is not None:
                continue
            if stream:
                out = await _do_chat_stream(
                    session, line, model_ref[0], message_history_mcp
                )
                message_history_mcp.append({"role": "user", "content": line})
                message_history_mcp.append({"role": "assistant", "content": out})
                if history_file:
                    _append_history(history_file, line, out)
            else:
                out = await _do_chat(
                    session, line, model_ref[0], message_history_mcp
                )
                message_history_mcp.append({"role": "user", "content": line})
                message_history_mcp.append({"role": "assistant", "content": out})
                print("Assistant:", out, sep="\n")


def main() -> None:
    args = _parse_args()
    if getattr(args, "no_stream", False):
        args.stream = False
    if args.query == "serve":
        try:
            from .serve import run_serve
        except ImportError as e:
            print("Serve requires uvicorn and starlette. Install with: pip install ollamacode[server]", file=sys.stderr)
            raise SystemExit(1) from e
        port = getattr(args, "port", 8000)
        run_serve(port=port, config_path=args.config)
        return
    mcp_servers, _ = _resolve_mcp_servers(args.config, args.mcp_command, args.mcp_args)
    config = load_config(args.config)
    merged = merge_config_with_env(
        config,
        model_env=os.environ.get("OLLAMACODE_MODEL"),
        mcp_args_env=os.environ.get("OLLAMACODE_MCP_ARGS"),
        system_extra_env=os.environ.get("OLLAMACODE_SYSTEM_EXTRA"),
    )
    model = args.model or merged.get("model") or os.environ.get("OLLAMACODE_MODEL", "gpt-oss:20b")
    system_extra = (merged.get("system_prompt_extra") or "").strip()
    max_messages = args.max_messages if args.max_messages is not None else merged.get("max_messages", 0)
    try:
        asyncio.run(_run(model, mcp_servers, system_extra, args.query, args.stream, args.tui, max_messages, args.history_file))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
