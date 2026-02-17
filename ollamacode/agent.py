"""
Agent loop: user message + MCP tools → Ollama chat → tool_calls → MCP call_tool → append results → repeat.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import sys
import threading
import time
from collections.abc import AsyncIterator
from typing import Any, Awaitable, Callable, Literal

import ollama

from .ollama_client import (
    chat_async as ollama_chat_async,
    chat_stream_sync as ollama_chat_stream_with_fallback_sync,
    wrap_ollama_template_error as _wrap_ollama_template_error,
)
from .bridge import (
    add_harmony_function_aliases,
    add_tool_aliases_for_ollama,
    mcp_tools_to_ollama,
)
from .mcp_client import (
    TOOL_NAME_ALIASES,
    McpConnection,
    call_tool,
    list_tools,
    tool_result_to_content,
)
from .state import append_past_error

logger = logging.getLogger(__name__)

# When confirm_tool_calls is True, optional callback before each tool: return "run" | "skip" | ("edit", new_args).
ToolCallDecision = (
    Literal["run"] | Literal["skip"] | tuple[Literal["edit"], dict[str, Any]]
)
BeforeToolCallCB = Callable[[str, dict[str, Any]], Awaitable[ToolCallDecision]]
ToolStartCB = Callable[[str, dict[str, Any]], None]
ToolEndCB = Callable[[str, dict[str, Any], str], None]


def _get(o: Any, key: str, default: Any = None) -> Any:
    if isinstance(o, dict):
        return o.get(key, default)
    return getattr(o, key, default)


def _parse_tool_args(raw_args: str | dict | None) -> tuple[dict[str, Any], str | None]:
    """
    Parse tool-call arguments; tolerate common model JSON errors.
    Returns (parsed_dict, None) on success, or ({}, error_message) on failure so the caller can
    surface a clear error to the model.
    """
    if raw_args is None:
        return {}, None
    if isinstance(raw_args, dict):
        return raw_args, None
    s = (raw_args or "").strip()
    if not s:
        return {}, None

    last_error: str | None = None

    def try_parse(t: str) -> dict[str, Any] | None:
        nonlocal last_error
        try:
            return json.loads(t)
        except json.JSONDecodeError as e:
            last_error = f"Invalid JSON at position {e.pos}: {e.msg}"
            return None

    out = try_parse(s)
    if out is not None:
        return out, None
    # Extra '}' at end (e.g. '{"a":1}}')
    if s.endswith("}}"):
        out = try_parse(s[:-1])
        if out is not None:
            return out, None
    # Extra ']' before '}' (e.g. '{"key": "val"]}')
    if "]" in s and s.endswith("]}"):
        out = try_parse(s[:-2] + "}")
        if out is not None:
            return out, None
    # Optional: try json5 for trailing commas, comments, etc.
    try:
        import json5  # type: ignore[import-untyped]

        out = json5.loads(s)
        if isinstance(out, dict):
            return out, None
    except ImportError:
        pass
    except Exception as e:
        last_error = f"JSON5 parse failed: {e}"
    # Unescaped newlines inside quoted strings: replace literal newlines within "..." with space
    in_string = False
    escaped = False
    result: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if escaped:
            result.append(c)
            escaped = False
        elif c == "\\" and in_string:
            result.append(c)
            escaped = True
        elif c == '"':
            result.append(c)
            in_string = not in_string
        elif in_string and c == "\n":
            result.append(" ")
        else:
            result.append(c)
        i += 1
    collapsed = "".join(result)
    if collapsed != s:
        out = try_parse(collapsed)
        if out is not None:
            return out, None
    err = (
        last_error
        or 'Could not parse as JSON. Use a single JSON object, e.g. {"path": "file.txt"}.'
    )
    return {}, err


def _filter_tools_by_policy(
    ollama_tools: list[dict[str, Any]],
    allowed_tools: list[str] | None,
    blocked_tools: list[str] | None,
) -> list[dict[str, Any]]:
    """Restrict tools by allowlist or blocklist (tool names as seen by the model, e.g. read_file, run_tests)."""
    if not allowed_tools and not blocked_tools:
        return ollama_tools
    allowed_set = set(allowed_tools) if allowed_tools else None
    blocked_set = set(blocked_tools) if blocked_tools else None
    out = []
    for t in ollama_tools:
        name = (t.get("function") or {}).get("name") or ""
        if allowed_set is not None and name not in allowed_set:
            continue
        if blocked_set is not None and name in blocked_set:
            continue
        out.append(t)
    return out


def _json_log_event(**kwargs: Any) -> None:
    """If OLLAMACODE_JSON_LOGS=1, print one JSON object per line to stderr (for profiling/dashboards)."""
    if os.environ.get("OLLAMACODE_JSON_LOGS", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return
    print(json.dumps(kwargs), file=sys.stderr, flush=True)


def _log_tool_event(
    name: str,
    event: str,
    *,
    arguments: dict[str, Any] | None = None,
    content: str | None = None,
    is_error: bool = False,
) -> None:
    """Log a single tool event at DEBUG. event is 'call' or 'result'; pass arguments= or content= accordingly."""
    if event == "call":
        args_str = json.dumps(arguments or {}, indent=2)
        if len(args_str) > TOOL_LOG_ARGS_MAX_CHARS:
            args_str = args_str[:TOOL_LOG_ARGS_MAX_CHARS] + "\n  ..."
        logger.debug("Calling %s: %s", name, args_str)
    elif event == "result":
        label = "Error" if is_error else "Result"
        if is_error and content:
            hint = _format_tool_error_hint(content)
            if hint:
                logger.debug("[%s] %s: %s", name, label, hint)
        body = (content or "").strip()
        if len(body) > TOOL_LOG_RESULT_MAX_CHARS:
            body = body[:TOOL_LOG_RESULT_MAX_CHARS] + "\n  ... (truncated)"
        if body:
            logger.debug("[%s] %s: %s", name, label, body)
        else:
            logger.debug("[%s] %s: (empty)", name, label)


def _log_tool_call(name: str, arguments: dict[str, Any]) -> None:
    """Log tool name and args at DEBUG level."""
    _log_tool_event(name, "call", arguments=arguments)


# Lookup table: (matcher, hint). Matcher is a list of substrings (any in content.lower()) or a callable (content_lower: str) -> bool.
# Order matters: more specific patterns must appear before generic ones.
_TOOL_ERROR_HINTS: list[tuple[list[str] | Any, str]] = [
    (
        ["filenotfounderror", "no such file"],
        "What failed: File or path not found.\nNext step: Check path exists and is readable (use list_dir to inspect).",
    ),
    (
        ["permission denied", "eacces", "eperm"],
        "What failed: Permission denied.\nNext step: Check file/dir permissions or run from a directory you can write to.",
    ),
    (
        ["timeout", "timed out"],
        "What failed: Command or operation timed out.\nNext step: Retry or use a longer timeout / smaller input.",
    ),
    (
        ["command not found"],
        "What failed: Command or executable not found.\nNext step: Install the tool or use full path (e.g. uv run pytest).",
    ),
    (
        lambda c: "not found" in c and ("exec" in c or "path" in c or "binary" in c),
        "What failed: Command or executable not found.\nNext step: Install the tool or use full path (e.g. uv run pytest).",
    ),
    (
        lambda c: "not found" in c and "no module" not in c,
        "What failed: File or path not found.\nNext step: Check path exists and is readable (use list_dir to inspect).",
    ),
    (
        ["syntaxerror", "syntax error"],
        "What failed: Syntax error in code or command.\nNext step: Fix the reported line/expression and re-run.",
    ),
    (
        ["modulenotfounderror", "no module named", "import error"],
        "What failed: Python module not found.\nNext step: Install dependency (e.g. pip install <module> or uv sync).",
    ),
    (
        ["indentationerror", "indentation error"],
        "What failed: Indentation error.\nNext step: Fix tabs/spaces and alignment on the reported line.",
    ),
    (
        lambda c: (
            "typeerror" in c
            and ("positional" in c or "argument" in c or "required" in c)
        ),
        "What failed: Wrong number or type of arguments.\nNext step: Check function signature and call site.",
    ),
    (
        ["connection refused", "econnrefused"],
        "What failed: Connection refused (service not listening or not reachable).\nNext step: Start the service or check host/port.",
    ),
    (
        ["address already in use", "eaddrinuse"],
        "What failed: Port already in use.\nNext step: Use another port or stop the process using it.",
    ),
    (
        ["out of memory", "memoryerror"],
        "What failed: Out of memory or process killed.\nNext step: Reduce input size or increase available memory.",
    ),
    (
        lambda c: "killed" in c and "signal" in c,
        "What failed: Out of memory or process killed.\nNext step: Reduce input size or increase available memory.",
    ),
    (
        ["json.decoder", "json decode", "expecting value"],
        "What failed: Invalid JSON.\nNext step: Fix the JSON (quotes, commas, brackets) at the reported position.",
    ),
    (
        ["is a directory", "eisdir"],
        "What failed: Path is a directory but a file was expected.\nNext step: Use a file path or list_dir to pick a file.",
    ),
]

# Max lengths for tool event logging (DEBUG). Exposed as constants for tests and optional config later.
TOOL_LOG_ARGS_MAX_CHARS = 400
TOOL_LOG_RESULT_MAX_CHARS = 1200


def _format_tool_error_hint(content: str) -> str | None:
    """If content matches common failure patterns, return a short 'What failed' + 'Next step' hint."""
    if not content or len(content) > 2000:
        return None
    c = content.lower().strip()
    for matcher, hint in _TOOL_ERROR_HINTS:
        if callable(matcher):
            if matcher(c):
                return hint
        else:
            if any(s in c for s in matcher):
                return hint
    return None


def _log_tool_result(name: str, content: str, is_error: bool = False) -> None:
    """Log tool result at DEBUG. On error, include a short 'what failed + next step' if recognized."""
    _log_tool_event(name, "result", content=content, is_error=is_error)


def _tool_call_one_line(name: str, arguments: dict[str, Any]) -> str:
    """One-line summary for tool call (reduces flicker when streaming)."""
    if name in ("read_file", "write_file") and isinstance(arguments.get("path"), str):
        return f"{name}({arguments.get('path', '')})"
    if name == "run_command" and isinstance(arguments.get("command"), str):
        cmd = arguments["command"].strip()
        return f"run_command({cmd[:50] + '...' if len(cmd) > 50 else cmd})"
    if name == "list_dir" and isinstance(arguments.get("path"), str):
        return f"list_dir({arguments.get('path', '')})"
    if arguments:
        # First key or a short repr
        first = next(iter(arguments.items()))
        return (
            f"{name}({first[0]}={str(first[1])[:30]}...)"
            if len(str(first[1])) > 30
            else f"{name}({first[0]}={first[1]})"
        )
    return name


def _tool_result_one_line(name: str, content: str, is_error: bool) -> str:
    """One-line summary for tool result (Cursor-style: compact and readable)."""
    if is_error:
        first_line = (content.splitlines() or [""])[0][:60]
        return f"→ error: {first_line}{'...' if len((content.splitlines() or [''])[0]) > 60 else ''}"
    # run_command often returns JSON {stdout, stderr, return_code}
    if name == "run_command" and content.strip().startswith("{"):
        try:
            data = json.loads(content)
            code = data.get("return_code", "?")
            stdout = (data.get("stdout") or "").strip()
            stderr = (data.get("stderr") or "").strip()
            if code == 0:
                summary = (
                    (stdout.splitlines()[0][:50] + "…")
                    if stdout and len(stdout.splitlines()[0]) > 50
                    else (stdout or "ok")
                )
                return f"→ return_code={code} {summary}"
            err_preview = (stderr.splitlines()[0][:45] + "…") if stderr else "failed"
            return f"→ return_code={code} {err_preview}"
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    n = len(content)
    if n == 0:
        return "→ (empty)"
    return f"→ {n} char(s)"


def _tool_result_summary(name: str, content: str, is_error: bool) -> str:
    """One-line summary plus first-line preview for UI tool traces."""
    base = _tool_result_one_line(name, content, is_error)
    preview = (content.splitlines() or [""])[0].strip()[:120] if content else ""
    if preview and preview not in base:
        return f"{base} — {preview}"
    return base


def _truncate_messages(
    messages: list[dict[str, Any]], max_messages: int
) -> list[dict[str, Any]]:
    """Keep system (if first) + last (max_messages - 1) messages to fit context window."""
    if max_messages <= 0 or len(messages) <= max_messages:
        return messages
    keep = max_messages
    system: list[dict[str, Any]] = []
    rest = messages
    if messages and _get(messages[0], "role") == "system":
        system = [messages[0]]
        rest = messages[1:]
        keep = max_messages - 1
    if len(rest) <= keep:
        return system + rest
    return system + rest[-keep:]


def _truncate_tool_result(content: str, max_chars: int) -> str:
    """If max_chars > 0 and content is longer, truncate and append a note."""
    if max_chars <= 0 or len(content) <= max_chars:
        return content
    return content[:max_chars] + f"\n\n... (truncated from {len(content)} chars)"


# --- Tool runner: MCP execution only (no logging, no callbacks) ---

# Item: (tool_name, arguments, parse_error_message | None). Result: (tool_name, arguments, content, is_error, hint).
ToolItem = tuple[str, dict[str, Any], str | None]
ToolResult = tuple[str, dict[str, Any], str, bool, str | None]


async def run_one_tool(
    session: McpConnection,
    name: str,
    arguments: dict[str, Any],
    max_tool_result_chars: int = 0,
) -> tuple[str, bool, str | None]:
    """
    Execute a single MCP tool call and return (content, is_error, hint).
    Content is truncated and includes error hint when is_error.
    """
    try:
        result = await call_tool(session, name, arguments)
    except BaseException as e:
        content = f"Tool error: {e}"
        is_error = True
    else:
        content = tool_result_to_content(result)
        is_error = getattr(result, "isError", False)
    hint = _format_tool_error_hint(content) if is_error else None
    if hint:
        content = content + "\n\n" + hint
    content = _truncate_tool_result(content, max_tool_result_chars)
    return content, is_error, hint


async def run_tools(
    session: McpConnection,
    items: list[ToolItem],
    max_tool_result_chars: int = 0,
) -> list[ToolResult]:
    """
    Execute MCP tool calls in parallel. Items are (name, arguments, parse_err).
    Returns (name, arguments, content, is_error, hint) in same order. Parse errors become synthetic error content.
    """
    successful = [(n, a) for n, a, e in items if e is None]
    results = await asyncio.gather(
        *[call_tool(session, n, a) for n, a in successful],
        return_exceptions=True,
    )
    res_iter = iter(results)
    out: list[ToolResult] = []
    for name, arguments, parse_err in items:
        if parse_err is not None:
            content = (
                f"Tool arguments could not be parsed: {parse_err}. "
                "Please output valid JSON for the arguments."
            )
            content = _truncate_tool_result(content, max_tool_result_chars)
            out.append((name, arguments, content, True, None))
            continue
        result = next(res_iter)
        if isinstance(result, BaseException):
            content = f"Tool error: {result}"
            is_error = True
        else:
            content = tool_result_to_content(result)
            is_error = getattr(result, "isError", False)
        hint = _format_tool_error_hint(content) if is_error else None
        if hint:
            content = content + "\n\n" + hint
        content = _truncate_tool_result(content, max_tool_result_chars)
        out.append((name, arguments, content, is_error, hint))
    return out


async def run_agent_loop(
    session: McpConnection,
    model: str,
    user_message: str,
    *,
    system_prompt: str | None = None,
    max_tool_rounds: int = 20,
    max_messages: int = 0,
    max_tool_result_chars: int = 0,
    message_history: list[dict[str, Any]] | None = None,
    quiet: bool = False,
    timing: bool = False,
    tool_progress_brief: bool = False,
    allowed_tools: list[str] | None = None,
    blocked_tools: list[str] | None = None,
    tool_errors_out: list[dict[str, Any]] | None = None,
    confirm_tool_calls: bool = False,
    before_tool_call: BeforeToolCallCB | None = None,
    on_tool_start: ToolStartCB | None = None,
    on_tool_end: ToolEndCB | None = None,
    image_paths: list[str] | None = None,
) -> str:
    """
    Run one user turn: send message to Ollama with MCP tools; on tool_calls,
    execute via MCP and re-call Ollama until the model returns text only.
    message_history: optional prior turns for multi-turn conversation.
    tool_progress_brief: when True, print one line per tool to reduce terminal flicker (e.g. when streaming).
    tool_errors_out: if provided (e.g. []), append one dict per tool error: {tool, arguments_summary, error, hint}.
    confirm_tool_calls: when True and before_tool_call is set, run tools sequentially and call before_tool_call before each.
    before_tool_call: async (tool_name, arguments) -> "run" | "skip" | ("edit", new_args). Skip injects "Skipped by user."; edit uses new_args.
    """
    # 1. Get MCP tools and convert to Ollama format; keep MCP server prefix in names (e.g. ollamacode-fs_read_file).
    list_result = await list_tools(session)
    ollama_tools = mcp_tools_to_ollama(list_result.tools)
    ollama_tools = _filter_tools_by_policy(ollama_tools, allowed_tools, blocked_tools)
    ollama_tools = add_tool_aliases_for_ollama(ollama_tools, TOOL_NAME_ALIASES)
    ollama_tools = add_harmony_function_aliases(ollama_tools)
    if not ollama_tools:
        ollama_tools = []  # Ollama may require None for no tools; check docs

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if message_history:
        messages.extend(message_history)
    user_msg: dict[str, Any] = {"role": "user", "content": user_message}
    if image_paths:
        user_msg["images"] = [str(p) for p in image_paths]
    messages.append(user_msg)

    turn_start = time.perf_counter() if timing else None
    for round_num in range(max_tool_rounds):
        # 2. Call Ollama (sync in thread); truncate if over max_messages
        if not quiet and not tool_progress_brief:
            logger.info("Sending to model (turn %s)...", round_num + 1)
        to_send = (
            _truncate_messages(messages, max_messages) if max_messages > 0 else messages
        )
        t0 = time.perf_counter() if timing else None
        try:
            response = await ollama_chat_async(model, to_send, ollama_tools)
        except Exception as e:  # noqa: BLE001
            raise _wrap_ollama_template_error(e)
        if timing and t0 is not None:
            elapsed = time.perf_counter() - t0
            logger.info("Ollama call: %.2fs", elapsed)
            _json_log_event(event="ollama", duration_s=round(elapsed, 3))

        msg = (
            response.get("message")
            if isinstance(response, dict)
            else getattr(response, "message", None)
        )
        if msg is None:
            return "No response from model."

        content = _get(msg, "content") or ""
        tool_calls = _get(msg, "tool_calls") or []

        # Build assistant message for history (Ollama format: role, content, tool_calls?)
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            if timing and turn_start is not None:
                logger.info("Turn total: %.2fs", time.perf_counter() - turn_start)
            return (content or "").strip()

        # 3. Execute tool calls via MCP (in parallel when multiple)
        items: list[tuple[str, dict[str, Any], str | None]] = []
        for tc in tool_calls:
            fn = (
                tc.get("function")
                if isinstance(tc, dict)
                else getattr(tc, "function", None)
            )
            if fn is None:
                continue
            name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
            if not name:
                continue
            raw_args = (
                fn.get("arguments")
                if isinstance(fn, dict)
                else getattr(fn, "arguments", None)
            )
            arguments, parse_err = _parse_tool_args(raw_args)
            items.append((name, arguments, parse_err))

        if items and not quiet:
            names = [x[0] for x in items]
            logger.info(
                "Running %s tool(s): %s",
                len(names),
                ", ".join(names),
            )

        if items:
            t0_parallel = time.perf_counter() if timing else None
            use_confirm = confirm_tool_calls and before_tool_call is not None
            if use_confirm:
                # Sequential: prompt before each tool (run / skip / edit)
                for name, arguments, parse_err in items:
                    if on_tool_start is not None:
                        try:
                            on_tool_start(name, arguments)
                        except Exception:
                            pass
                    if parse_err is not None:
                        content = f"Tool arguments could not be parsed: {parse_err}. Please output valid JSON for the arguments."
                        content = _truncate_tool_result(content, max_tool_result_chars)
                        messages.append(
                            {"role": "tool", "tool_name": name, "content": content}
                        )
                        if on_tool_end is not None:
                            try:
                                on_tool_end(name, arguments, content)
                            except Exception:
                                pass
                        _json_log_event(
                            event="tool",
                            tool=name,
                            result_chars=len(content),
                            is_error=True,
                        )
                        continue
                    assert before_tool_call is not None
                    decision = await before_tool_call(name, arguments)
                    if decision == "skip":
                        content = "Skipped by user."
                        if not quiet:
                            if tool_progress_brief:
                                logger.info("  %s (skipped)", name)
                            else:
                                _log_tool_result(name, content, is_error=False)
                        content = _truncate_tool_result(content, max_tool_result_chars)
                        messages.append(
                            {"role": "tool", "tool_name": name, "content": content}
                        )
                        if on_tool_end is not None:
                            try:
                                on_tool_end(name, arguments, "skipped")
                            except Exception:
                                pass
                        _json_log_event(
                            event="tool",
                            tool=name,
                            result_chars=len(content),
                            is_error=False,
                        )
                        continue
                    if isinstance(decision, tuple) and decision[0] == "edit":
                        _, arguments = decision
                    # "run" or edited args
                    if not quiet:
                        if tool_progress_brief:
                            logger.info("  %s", _tool_call_one_line(name, arguments))
                        else:
                            _log_tool_call(name, arguments)
                    content, is_error, hint = await run_one_tool(
                        session, name, arguments, max_tool_result_chars
                    )
                    if not quiet and not tool_progress_brief:
                        _log_tool_result(name, content, is_error=is_error)
                    if not quiet and tool_progress_brief:
                        logger.info(
                            "  %s %s",
                            name,
                            _tool_result_one_line(name, content, is_error),
                        )
                    if on_tool_end is not None:
                        try:
                            on_tool_end(
                                name,
                                arguments,
                                _tool_result_summary(name, content, is_error),
                            )
                        except Exception:
                            pass
                    if tool_errors_out is not None and is_error:
                        args_summary = _tool_call_one_line(name, arguments)
                        tool_errors_out.append(
                            {
                                "tool": name,
                                "arguments_summary": args_summary,
                                "error": content[:2000]
                                if len(content) > 2000
                                else content,
                                "hint": hint,
                            }
                        )
                        try:
                            append_past_error(name, content[:500], hint or "")
                        except Exception:
                            pass
                    messages.append(
                        {"role": "tool", "tool_name": name, "content": content}
                    )
                    if on_tool_end is not None:
                        try:
                            on_tool_end(
                                name,
                                arguments,
                                _tool_result_summary(name, content, is_error),
                            )
                        except Exception:
                            pass
                    _json_log_event(
                        event="tool",
                        tool=name,
                        result_chars=len(content),
                        is_error=is_error,
                        **({"error_hint": hint} if hint else {}),
                    )
            else:
                for name, arguments, _ in items:
                    if on_tool_start is not None:
                        try:
                            on_tool_start(name, arguments)
                        except Exception:
                            pass
                tool_results = await run_tools(session, items, max_tool_result_chars)
                for name, arguments, content, is_error, hint in tool_results:
                    if not quiet:
                        if tool_progress_brief:
                            logger.info("  %s", _tool_call_one_line(name, arguments))
                        else:
                            _log_tool_call(name, arguments)
                        if tool_progress_brief:
                            logger.info(
                                "  %s %s",
                                name,
                                _tool_result_one_line(name, content, is_error),
                            )
                        else:
                            _log_tool_result(name, content, is_error=is_error)
                    if on_tool_end is not None:
                        try:
                            on_tool_end(
                                name,
                                arguments,
                                _tool_result_summary(name, content, is_error),
                            )
                        except Exception:
                            pass
                    if tool_errors_out is not None and is_error:
                        args_summary = _tool_call_one_line(name, arguments)
                        tool_errors_out.append(
                            {
                                "tool": name,
                                "arguments_summary": args_summary,
                                "error": content[:2000]
                                if len(content) > 2000
                                else content,
                                "hint": hint,
                            }
                        )
                        try:
                            append_past_error(name, content[:500], hint or "")
                        except Exception:
                            pass
                    messages.append(
                        {"role": "tool", "tool_name": name, "content": content}
                    )
                    if on_tool_end is not None:
                        try:
                            on_tool_end(
                                name,
                                arguments,
                                _tool_result_summary(name, content, is_error),
                            )
                        except Exception:
                            pass
                    _json_log_event(
                        event="tool",
                        tool=name,
                        result_chars=len(content),
                        is_error=is_error,
                        **({"error_hint": hint} if hint else {}),
                    )
            if timing and t0_parallel is not None:
                elapsed = time.perf_counter() - t0_parallel
                logger.info("Tools: %.2fs", elapsed)
                _json_log_event(
                    event="tools", duration_s=round(elapsed, 3), count=len(items)
                )

    if timing and turn_start is not None:
        turn_elapsed = time.perf_counter() - turn_start
        logger.info("Turn total: %.2fs", turn_elapsed)
        _json_log_event(event="turn", duration_s=round(turn_elapsed, 3))
    return "(Max tool rounds reached; stopping.)"


def _is_tool_call_parse_error(exc: BaseException) -> bool:
    """True if Ollama failed to parse tool-call JSON (retry often fixes it)."""
    s = str(exc).lower()
    return "error parsing tool call" in s or "invalid character" in s


def _stream_into_queue(
    q: queue.Queue,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> None:
    """Run Ollama chat with stream=True and put (content_fragment, done, message) into q. Puts None at end.
    On tool-call parse error, retries once automatically (model often succeeds on retry).
    On other errors, puts the exception then None so the caller can re-raise.
    """
    try:
        for attempt in range(2):
            try:
                stream = ollama.chat(
                    model=model, messages=messages, tools=tools, stream=True
                )
                for chunk in stream:
                    msg = (
                        chunk.get("message")
                        if isinstance(chunk, dict)
                        else getattr(chunk, "message", None)
                    )
                    content = _get(msg, "content", "") or "" if msg else ""
                    done = (
                        chunk.get("done")
                        if isinstance(chunk, dict)
                        else getattr(chunk, "done", False)
                    )
                    q.put((content, done, msg))
                break
            except Exception as e:
                if attempt == 0 and _is_tool_call_parse_error(e):
                    continue  # retry once
                if _is_tool_call_parse_error(e):
                    err = ValueError(
                        "Model produced invalid tool-call JSON (e.g. extra '}' or ']', or unescaped newlines in strings). "
                        "Multi-line arguments must use \\n, not literal line breaks. Try again or use a model with stricter JSON."
                    )
                    err.__cause__ = e
                    q.put(err)
                else:
                    q.put(_wrap_ollama_template_error(e))
                break
    finally:
        q.put(None)


async def run_agent_loop_stream(
    session: McpConnection,
    model: str,
    user_message: str,
    *,
    system_prompt: str | None = None,
    max_tool_rounds: int = 20,
    max_messages: int = 0,
    max_tool_result_chars: int = 0,
    message_history: list[dict[str, Any]] | None = None,
    quiet: bool = False,
    timing: bool = False,
    tool_progress_brief: bool = False,
    allowed_tools: list[str] | None = None,
    blocked_tools: list[str] | None = None,
    confirm_tool_calls: bool = False,
    before_tool_call: BeforeToolCallCB | None = None,
    on_tool_start: ToolStartCB | None = None,
    on_tool_end: ToolEndCB | None = None,
    image_paths: list[str] | None = None,
) -> AsyncIterator[str]:
    """
    Like run_agent_loop but streams content tokens as they arrive.
    Yields content fragments (str). When tool_calls occur, runs them and continues streaming the next turn.
    message_history: optional prior turns [{"role":"user","content":...},{"role":"assistant","content":...}, ...].
    tool_progress_brief: when True, print one line per tool to reduce terminal flicker during streaming.
    """
    list_result = await list_tools(session)
    ollama_tools = mcp_tools_to_ollama(list_result.tools)
    ollama_tools = _filter_tools_by_policy(ollama_tools, allowed_tools, blocked_tools)
    ollama_tools = add_tool_aliases_for_ollama(ollama_tools, TOOL_NAME_ALIASES)
    ollama_tools = add_harmony_function_aliases(ollama_tools) or []

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if message_history:
        messages.extend(message_history)
    user_msg_stream: dict[str, Any] = {"role": "user", "content": user_message}
    if image_paths:
        user_msg_stream["images"] = [str(p) for p in image_paths]
    messages.append(user_msg_stream)

    loop = asyncio.get_event_loop()
    turn_start = time.perf_counter() if timing else None
    for round_num in range(max_tool_rounds):
        if not quiet and not tool_progress_brief:
            logger.info("Sending to model (turn %s)...", round_num + 1)
        q: queue.Queue = queue.Queue()
        to_send = (
            _truncate_messages(messages, max_messages) if max_messages > 0 else messages
        )
        t0 = time.perf_counter() if timing else None
        thread = threading.Thread(
            target=_stream_into_queue,
            args=(q, model, to_send, ollama_tools),
        )
        thread.start()

        last_msg = None
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            content_frag, done, msg = item
            if content_frag:
                yield content_frag
            last_msg = msg
        thread.join()
        if timing and t0 is not None:
            elapsed = time.perf_counter() - t0
            logger.info("Ollama call: %.2fs", elapsed)
            _json_log_event(event="ollama", duration_s=round(elapsed, 3))

        if last_msg is None:
            return
        content = _get(last_msg, "content") or ""
        tool_calls = _get(last_msg, "tool_calls") or []
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            if timing and turn_start is not None:
                logger.info("Turn total: %.2fs", time.perf_counter() - turn_start)
            return

        items_stream: list[tuple[str, dict[str, Any], str | None]] = []
        for tc in tool_calls:
            fn = (
                tc.get("function")
                if isinstance(tc, dict)
                else getattr(tc, "function", None)
            )
            if fn is None:
                continue
            name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
            if not name:
                continue
            raw_args = (
                fn.get("arguments")
                if isinstance(fn, dict)
                else getattr(fn, "arguments", None)
            )
            arguments, parse_err = _parse_tool_args(raw_args)
            items_stream.append((name, arguments, parse_err))
        if items_stream and not quiet:
            names = [x[0] for x in items_stream]
            logger.info("Running %s tool(s): %s", len(names), ", ".join(names))
        if items_stream:
            use_confirm_stream = confirm_tool_calls and before_tool_call is not None
            if use_confirm_stream:
                for name, arguments, parse_err in items_stream:
                    if on_tool_start is not None:
                        try:
                            on_tool_start(name, arguments)
                        except Exception:
                            pass
                    if parse_err is not None:
                        content = f"Tool arguments could not be parsed: {parse_err}. Please output valid JSON for the arguments."
                        content = _truncate_tool_result(content, max_tool_result_chars)
                        messages.append(
                            {"role": "tool", "tool_name": name, "content": content}
                        )
                        if on_tool_end is not None:
                            try:
                                on_tool_end(name, arguments, content)
                            except Exception:
                                pass
                        _json_log_event(
                            event="tool",
                            tool=name,
                            result_chars=len(content),
                            is_error=True,
                        )
                        continue
                    assert before_tool_call is not None
                    decision = await before_tool_call(name, arguments)
                    if decision == "skip":
                        content = "Skipped by user."
                        if not quiet:
                            if tool_progress_brief:
                                logger.info("  %s (skipped)", name)
                            else:
                                _log_tool_result(name, content, is_error=False)
                        content = _truncate_tool_result(content, max_tool_result_chars)
                        messages.append(
                            {"role": "tool", "tool_name": name, "content": content}
                        )
                        if on_tool_end is not None:
                            try:
                                on_tool_end(name, arguments, "skipped")
                            except Exception:
                                pass
                        _json_log_event(
                            event="tool",
                            tool=name,
                            result_chars=len(content),
                            is_error=False,
                        )
                        continue
                    if isinstance(decision, tuple) and decision[0] == "edit":
                        _, arguments = decision
                    if not quiet:
                        if tool_progress_brief:
                            logger.info("  %s", _tool_call_one_line(name, arguments))
                        else:
                            _log_tool_call(name, arguments)
                    content, is_error, hint = await run_one_tool(
                        session, name, arguments, max_tool_result_chars
                    )
                    if not quiet and not tool_progress_brief:
                        _log_tool_result(name, content, is_error=is_error)
                    if not quiet and tool_progress_brief:
                        logger.info(
                            "  %s %s",
                            name,
                            _tool_result_one_line(name, content, is_error),
                        )
                    if on_tool_end is not None:
                        try:
                            on_tool_end(
                                name,
                                arguments,
                                _tool_result_summary(name, content, is_error),
                            )
                        except Exception:
                            pass
                    messages.append(
                        {"role": "tool", "tool_name": name, "content": content}
                    )
                    _json_log_event(
                        event="tool",
                        tool=name,
                        result_chars=len(content),
                        is_error=is_error,
                        **({"error_hint": hint} if hint else {}),
                    )
            else:
                for name, arguments, _ in items_stream:
                    if on_tool_start is not None:
                        try:
                            on_tool_start(name, arguments)
                        except Exception:
                            pass
                tool_results_stream = await run_tools(
                    session, items_stream, max_tool_result_chars
                )
                for name, arguments, content, is_error, hint in tool_results_stream:
                    if not quiet:
                        if tool_progress_brief:
                            logger.info("  %s", _tool_call_one_line(name, arguments))
                        else:
                            _log_tool_call(name, arguments)
                        if tool_progress_brief:
                            logger.info(
                                "  %s %s",
                                name,
                                _tool_result_one_line(name, content, is_error),
                            )
                        else:
                            _log_tool_result(name, content, is_error=is_error)
                    if on_tool_end is not None:
                        try:
                            on_tool_end(
                                name,
                                arguments,
                                _tool_result_summary(name, content, is_error),
                            )
                        except Exception:
                            pass
                    messages.append(
                        {"role": "tool", "tool_name": name, "content": content}
                    )
                    _json_log_event(
                        event="tool",
                        tool=name,
                        result_chars=len(content),
                        is_error=is_error,
                        **({"error_hint": hint} if hint else {}),
                    )


async def run_agent_loop_no_mcp(
    model: str,
    user_message: str,
    *,
    system_prompt: str | None = None,
    message_history: list[dict[str, Any]] | None = None,
) -> str:
    """Run one turn with Ollama only (no MCP tools). Returns assistant text.
    Uses centralised chat-with-fallback (generate API on template error)."""
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if message_history:
        messages.extend(message_history)
    messages.append({"role": "user", "content": user_message})

    try:
        response = await ollama_chat_async(model, messages, [])
    except Exception as e:  # noqa: BLE001
        raise _wrap_ollama_template_error(e)
    msg = (
        response.get("message")
        if isinstance(response, dict)
        else getattr(response, "message", None)
    )
    if msg is None:
        return "No response from model."
    content = (
        msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
    )
    return (content or "").strip()


def _stream_no_mcp_into_queue(
    q: queue.Queue,
    model: str,
    messages: list[dict[str, Any]],
) -> None:
    """Stream Ollama response into q. Uses centralised chat_stream_sync (generate fallback on template error)."""
    try:
        for content_tuple in ollama_chat_stream_with_fallback_sync(model, messages):
            q.put(content_tuple)
    except Exception as e:  # noqa: BLE001
        q.put(_wrap_ollama_template_error(e))
    finally:
        q.put(None)


async def run_agent_loop_no_mcp_stream(
    model: str,
    user_message: str,
    *,
    system_prompt: str | None = None,
    message_history: list[dict[str, Any]] | None = None,
) -> AsyncIterator[str]:
    """Stream one turn with Ollama only (no MCP). Yields content fragments."""
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if message_history:
        messages.extend(message_history)
    messages.append({"role": "user", "content": user_message})

    q: queue.Queue = queue.Queue()
    thread = threading.Thread(
        target=_stream_no_mcp_into_queue, args=(q, model, messages)
    )
    thread.start()
    loop = asyncio.get_event_loop()
    try:
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            if item and item[0]:
                yield item[0]
    finally:
        thread.join()
