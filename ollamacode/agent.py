"""
Agent loop: user message + MCP tools → LLM chat → tool_calls → MCP call_tool → append results → repeat.
Supports multiple AI providers via the providers/ package (Ollama, OpenAI-compat, Anthropic, ...).
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
import json as _json
import difflib as _difflib
from pathlib import Path
from collections.abc import AsyncIterator
from typing import Any, Awaitable, Callable, Literal

import ollama

from .ollama_client import (
    chat_async as ollama_chat_async,
    chat_stream_sync as ollama_chat_stream_with_fallback_sync,
    wrap_ollama_template_error as _wrap_ollama_template_error,
)
from .providers.base import BaseProvider
from .bridge import (
    add_harmony_function_aliases,
    add_tool_aliases_for_ollama,
    mcp_tools_to_ollama,
    use_short_names_for_builtin_tools,
)
from .mcp_client import (
    TOOL_NAME_ALIASES,
    McpConnection,
    call_tool,
    disable_tool_calls,
    list_tools,
    reset_tool_calls,
    tool_result_to_content,
)
from .state import append_past_error

logger = logging.getLogger(__name__)

# Tool replay/record (deterministic runs)
_REPLAY_PATH = os.environ.get("OLLAMACODE_TOOL_REPLAY_PATH", "").strip()
_RECORD_PATH = os.environ.get("OLLAMACODE_TOOL_RECORD_PATH", "").strip()
_REPLAY_ENTRIES: list[dict[str, Any]] | None = None
_REPLAY_INDEX = 0
_REPLAY_LOCK = threading.Lock()


def _load_replay_entries() -> list[dict[str, Any]]:
    global _REPLAY_ENTRIES
    if _REPLAY_ENTRIES is not None:
        return _REPLAY_ENTRIES
    entries: list[dict[str, Any]] = []
    if _REPLAY_PATH and Path(_REPLAY_PATH).exists():
        for line in Path(_REPLAY_PATH).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json.loads(line)
                if isinstance(obj, dict):
                    entries.append(obj)
            except Exception:
                continue
    _REPLAY_ENTRIES = entries
    return entries


def _next_replay_entry() -> dict[str, Any] | None:
    global _REPLAY_INDEX
    if not _REPLAY_PATH:
        return None
    entries = _load_replay_entries()
    with _REPLAY_LOCK:
        if _REPLAY_INDEX >= len(entries):
            return None
        entry = entries[_REPLAY_INDEX]
        _REPLAY_INDEX += 1
        return entry


def _record_tool_event(
    name: str,
    arguments: dict[str, Any],
    content: str,
    is_error: bool,
    hint: str | None,
    duration_s: float | None = None,
) -> None:
    if not _RECORD_PATH:
        return
    payload = {
        "tool": name,
        "arguments": arguments,
        "content": content,
        "is_error": is_error,
        "hint": hint,
    }
    if duration_s is not None:
        payload["duration_s"] = duration_s
    try:
        with open(_RECORD_PATH, "a", encoding="utf-8") as f:
            f.write(_json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _emit_replay_report() -> None:
    if not (_REPLAY_PATH and _RECORD_PATH):
        return
    if os.environ.get("OLLAMACODE_REPLAY_REPORT", "0") != "1":
        return
    try:
        replay = [
            json.loads(line)
            for line in Path(_REPLAY_PATH).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        live = [
            json.loads(line)
            for line in Path(_RECORD_PATH).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except Exception:
        return
    lines: list[str] = ["[replay] report"]
    for i, (replay_entry, live_entry) in enumerate(zip(replay, live), 1):
        tool = replay_entry.get("tool")
        if tool != live_entry.get("tool"):
            lines.append(
                f"  {i}. tool mismatch: replay={tool} live={live_entry.get('tool')}"
            )
            continue
        rd = replay_entry.get("duration_s")
        ld = live_entry.get("duration_s")
        if rd is not None and ld is not None:
            lines.append(
                f"  {i}. {tool} duration: replay={rd:.3f}s live={ld:.3f}s Δ={ld - rd:.3f}s"
            )
        if replay_entry.get("content") != live_entry.get("content"):
            diff = "\n".join(
                _difflib.unified_diff(
                    str(replay_entry.get("content", "")).splitlines(),
                    str(live_entry.get("content", "")).splitlines(),
                    fromfile="replay",
                    tofile="live",
                    lineterm="",
                )
            )
            if diff:
                lines.append("  diff:\n" + "\n".join(diff.splitlines()[:40]))
    print("\n".join(lines), file=sys.stderr)


# When confirm_tool_calls is True, optional callback before each tool:
# return "run" | "skip" | ("skip", reason) | ("edit", new_args).
ToolCallDecision = (
    Literal["run"]
    | Literal["skip"]
    | tuple[Literal["skip"], str]
    | tuple[Literal["edit"], dict[str, Any]]
)
BeforeToolCallCB = Callable[[str, dict[str, Any]], Awaitable[ToolCallDecision]]
ToolStartCB = Callable[[str, dict[str, Any]], None]
ToolEndCB = Callable[[str, dict[str, Any], str], None]


def _get(o: Any, key: str, default: Any = None) -> Any:
    if isinstance(o, dict):
        return o.get(key, default)
    return getattr(o, key, default)


def _chunk_text(text: str, chunk_size: int) -> list[str]:
    if not text:
        return []
    if chunk_size <= 0:
        return [text]
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


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
    """Emit one JSON log line (stderr and/or JSONL file)."""
    payload = json.dumps(kwargs, ensure_ascii=False)
    if os.environ.get("OLLAMACODE_JSON_LOGS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        print(payload, file=sys.stderr, flush=True)
    if (
        kwargs.get("event") == "trace_summary"
        and os.environ.get("OLLAMACODE_TRACE_SUMMARY_STDOUT", "0") == "1"
    ):
        tool_calls = kwargs.get("tool_calls", 0)
        tool_errors = kwargs.get("tool_errors", 0)
        duration_s = kwargs.get("duration_s", 0)
        print(
            f"[trace] duration={duration_s}s tools={tool_calls} errors={tool_errors}",
            file=sys.stdout,
            flush=True,
        )
    jsonl_path = os.environ.get("OLLAMACODE_JSONL_PATH") or os.environ.get(
        "OLLAMACODE_LOG_JSONL"
    )
    if jsonl_path:
        try:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(payload + "\n")
        except Exception:
            pass


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


# ---------------------------------------------------------------------------
# Doom loop / infinite loop detection (task 133)
# ---------------------------------------------------------------------------

_DOOM_LOOP_WINDOW = int(os.environ.get("OLLAMACODE_DOOM_LOOP_WINDOW", "6"))
_DOOM_LOOP_THRESHOLD = int(os.environ.get("OLLAMACODE_DOOM_LOOP_THRESHOLD", "3"))


class _DoomLoopDetector:
    """Track recent tool calls and detect when the LLM calls the same tool
    with identical arguments repeatedly (doom loop)."""

    def __init__(
        self,
        window: int = _DOOM_LOOP_WINDOW,
        threshold: int = _DOOM_LOOP_THRESHOLD,
    ):
        self._window = max(3, window)
        self._threshold = max(2, threshold)
        self._history: list[tuple[str, str]] = []  # (name, args_hash)

    @staticmethod
    def _hash_args(args: dict[str, Any]) -> str:
        try:
            return json.dumps(args, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(args)

    def record(self, name: str, arguments: dict[str, Any]) -> None:
        key = (name, self._hash_args(arguments))
        self._history.append(key)
        if len(self._history) > self._window:
            self._history = self._history[-self._window :]

    def is_looping(self) -> tuple[bool, str | None]:
        """Check if the last N calls are dominated by a single (tool, args) pair.

        Returns (True, warning_message) if a doom loop is detected.
        """
        if len(self._history) < self._threshold:
            return False, None
        # Check the last `threshold` entries for consecutive repetition.
        tail = self._history[-self._threshold :]
        if len(set(tail)) == 1:
            name, _ = tail[0]
            msg = (
                f"Doom loop detected: tool '{name}' called {self._threshold} times "
                f"consecutively with identical arguments. Breaking the loop."
            )
            logger.warning(msg)
            _json_log_event(event="doom_loop", tool=name, consecutive=self._threshold)
            return True, msg
        return False, None


def _active_tool_names(tools: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or "").strip()
        if name:
            names.append(name)
    return sorted(set(names))


def _tools_system_prompt(
    tools: list[dict[str, Any]], *, max_chars: int = 2500
) -> str | None:
    names = _active_tool_names(tools)
    if not names:
        return None
    joined = ", ".join(names)
    if len(joined) > max_chars:
        joined = joined[:max_chars] + "..."
    return (
        "Tool availability:\n"
        "Default built-in capabilities are usually available via MCP: "
        "filesystem (read/write/edit/list), terminal/commands/tests/lint, "
        "codebase search/indexing, git, and optional skills/state/reasoning/screenshot/web.\n"
        "Use only tools listed in this turn. Active tools:\n"
        f"{joined}"
    )


# --- Tool runner: MCP execution only (no logging, no callbacks) ---

# Item: (tool_name, arguments, parse_error_message | None). Result: (tool_name, arguments, content, is_error, hint).
ToolItem = tuple[str, dict[str, Any], str | None]
ToolResult = tuple[str, dict[str, Any], str, bool, str | None]


def _tool_retry_attempts() -> int:
    raw = os.environ.get("OLLAMACODE_TOOL_RETRY_ATTEMPTS", "").strip()
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = 1
    return max(1, min(n, 3))


def _tool_timeout_seconds() -> float | None:
    raw = os.environ.get("OLLAMACODE_TOOL_TIMEOUT_SECONDS", "").strip()
    try:
        val = float(raw) if raw else 60.0
    except (TypeError, ValueError):
        val = 60.0
    return val if val > 0 else None


def _tool_budget_seconds() -> float | None:
    raw = os.environ.get("OLLAMACODE_TOOL_BUDGET_SECONDS", "").strip()
    try:
        val = float(raw) if raw else 0.0
    except (TypeError, ValueError):
        val = 0.0
    return val if val > 0 else None


def _run_budget_seconds() -> float | None:
    raw = os.environ.get("OLLAMACODE_RUN_BUDGET_SECONDS", "").strip()
    try:
        val = float(raw) if raw else 0.0
    except (TypeError, ValueError):
        val = 0.0
    return val if val > 0 else None


def _should_retry_tool_error(content: str) -> bool:
    c = (content or "").lower()
    return any(
        k in c
        for k in (
            "timeout",
            "timed out",
            "temporarily unavailable",
            "connection reset",
            "connection refused",
            "broken pipe",
            "try again",
        )
    )


async def _maybe_recovery_hint(
    session: McpConnection,
    name: str,
    arguments: dict[str, Any],
    content: str,
) -> str | None:
    if os.environ.get("OLLAMACODE_TOOL_RECOVERY", "0") != "1":
        return None
    if _REPLAY_PATH:
        return None
    path = arguments.get("path") if isinstance(arguments, dict) else None
    if not path or not isinstance(path, str):
        return None
    lowered = (content or "").lower()
    if "file or path not found" not in lowered and "no such file" not in lowered:
        return None
    parent = os.path.dirname(path) or "."
    try:
        recovery = await call_tool(session, "list_dir", {"path": parent})
        recovery_text = tool_result_to_content(recovery)
        return f"Recovery hint: list_dir({parent}) =>\n{recovery_text}"
    except Exception:
        return None


def _is_safe_tool_name(name: str) -> bool:
    """Heuristic: treat read/search/list tools as safe for parallel execution."""
    n = name.lower()
    prefixes = ("read_", "list_", "search_", "grep", "glob", "get_", "diff_", "status")
    return n.startswith(prefixes) or n in ("run_linter", "run_tests")


async def _call_tool_with_retry(
    session: McpConnection,
    name: str,
    arguments: dict[str, Any],
    attempts: int,
    *,
    timeout_s: float | None = None,
    deadline: float | None = None,
) -> Any:
    delay = 0.4
    for i in range(attempts):
        try:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("tool budget exceeded")
                if timeout_s is None:
                    timeout = remaining
                else:
                    timeout = min(timeout_s, remaining)
            else:
                timeout = timeout_s
            if timeout is not None:
                return await asyncio.wait_for(
                    call_tool(session, name, arguments), timeout
                )
            return await call_tool(session, name, arguments)
        except BaseException as e:
            if i >= attempts - 1:
                return e
            msg = str(e)
            if not _should_retry_tool_error(msg):
                return e
        await asyncio.sleep(delay)
        delay = min(2.0, delay * 2)
    return RuntimeError("tool retry attempts exceeded")


async def _call_tool_with_retry_timed(
    session: McpConnection,
    name: str,
    arguments: dict[str, Any],
    attempts: int,
    *,
    timeout_s: float | None = None,
    deadline: float | None = None,
) -> tuple[Any, float]:
    t0 = time.perf_counter()
    result = await _call_tool_with_retry(
        session, name, arguments, attempts, timeout_s=timeout_s, deadline=deadline
    )
    return result, time.perf_counter() - t0


async def run_one_tool(
    session: McpConnection,
    name: str,
    arguments: dict[str, Any],
    max_tool_result_chars: int = 0,
    *,
    deadline: float | None = None,
    timeout_s: float | None = None,
    request_id: str | None = None,
) -> tuple[str, bool, str | None]:
    """
    Execute a single MCP tool call and return (content, is_error, hint).
    Content is truncated and includes error hint when is_error.
    """
    # Replay mode: return recorded tool output deterministically.
    if _REPLAY_PATH:
        entry = _next_replay_entry()
        if entry is None:
            content = "Tool error: replay exhausted"
            return content, True, _format_tool_error_hint(content)
        if entry.get("tool") != name:
            content = f"Tool error: replay mismatch (expected {entry.get('tool')}, got {name})"
            return content, True, _format_tool_error_hint(content)
        content = str(entry.get("content", ""))
        is_error = bool(entry.get("is_error", False))
        hint = entry.get("hint")
        return content, is_error, hint
    attempts = _tool_retry_attempts()
    if deadline is not None and time.monotonic() >= deadline:
        _json_log_event(event="tool_budget_exceeded", tool=name, request_id=request_id)
        content = "Tool error: tool budget exceeded"
        is_error = True
        hint = _format_tool_error_hint(content)
        if hint:
            content = content + "\n\n" + hint
        content = _truncate_tool_result(content, max_tool_result_chars)
        return content, is_error, hint
    t0 = time.perf_counter()
    result = await _call_tool_with_retry(
        session,
        name,
        arguments,
        attempts,
        timeout_s=timeout_s,
        deadline=deadline,
    )
    duration_s = time.perf_counter() - t0
    if isinstance(result, BaseException):
        content = f"Tool error: {result}"
        is_error = True
    else:
        content = tool_result_to_content(result)
        is_error = getattr(result, "isError", False)
        if is_error and _should_retry_tool_error(content) and attempts > 1:
            # One more retry if tool returned an explicit transient error.
            retry_result = await _call_tool_with_retry(session, name, arguments, 1)
            if not isinstance(retry_result, BaseException):
                content = tool_result_to_content(retry_result)
                is_error = getattr(retry_result, "isError", False)
    hint = _format_tool_error_hint(content) if is_error else None
    if hint:
        content = content + "\n\n" + hint
    if is_error:
        recovery = await _maybe_recovery_hint(session, name, arguments, content)
        if recovery:
            content = content + "\n\n" + recovery
    content = _truncate_tool_result(content, max_tool_result_chars)
    _record_tool_event(name, arguments, content, is_error, hint, duration_s=duration_s)
    return content, is_error, hint


async def run_tools(
    session: McpConnection,
    items: list[ToolItem],
    max_tool_result_chars: int = 0,
    *,
    request_id: str | None = None,
) -> list[ToolResult]:
    """
    Execute MCP tool calls in parallel. Items are (name, arguments, parse_err).
    Returns (name, arguments, content, is_error, hint) in same order. Parse errors become synthetic error content.
    """
    attempts = _tool_retry_attempts()
    tool_timeout_s = _tool_timeout_seconds()
    tool_budget_s = _tool_budget_seconds()
    deadline = time.monotonic() + tool_budget_s if tool_budget_s else None
    if _REPLAY_PATH:
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
            entry = _next_replay_entry()
            if entry is None:
                content = "Tool error: replay exhausted"
                out.append(
                    (name, arguments, content, True, _format_tool_error_hint(content))
                )
                continue
            if entry.get("tool") != name:
                content = f"Tool error: replay mismatch (expected {entry.get('tool')}, got {name})"
                out.append(
                    (name, arguments, content, True, _format_tool_error_hint(content))
                )
                continue
            content = str(entry.get("content", ""))
            is_error = bool(entry.get("is_error", False))
            hint = entry.get("hint")
            out.append((name, arguments, content, is_error, hint))
        return out
    successful = [(n, a) for n, a, e in items if e is None]
    safe_parallel = all(_is_safe_tool_name(n) for n, _, e in items if e is None)
    results: list[Any] = []
    durations: list[float] = []
    if deadline is not None and time.monotonic() >= deadline:
        _json_log_event(
            event="tool_budget_exceeded",
            tool_count=len(items),
            request_id=request_id,
        )
        out: list[ToolResult] = []
        for name, arguments, parse_err in items:
            if parse_err is not None:
                content = (
                    f"Tool arguments could not be parsed: {parse_err}. "
                    "Please output valid JSON for the arguments."
                )
                content = _truncate_tool_result(content, max_tool_result_chars)
                out.append((name, arguments, content, True, None))
            else:
                content = "Tool error: tool budget exceeded"
                hint = _format_tool_error_hint(content)
                if hint:
                    content = content + "\n\n" + hint
                content = _truncate_tool_result(content, max_tool_result_chars)
                out.append((name, arguments, content, True, hint))
        return out
    if safe_parallel:
        timed = await asyncio.gather(
            *[
                _call_tool_with_retry_timed(
                    session,
                    n,
                    a,
                    attempts,
                    timeout_s=tool_timeout_s,
                    deadline=deadline,
                )
                for n, a in successful
            ],
            return_exceptions=True,
        )
        for item in timed:
            if isinstance(item, BaseException):
                results.append(item)
                durations.append(0.0)
            else:
                res, dur = item
                results.append(res)
                durations.append(dur)
    else:
        for n, a in successful:
            if deadline is not None and time.monotonic() >= deadline:
                results.append(TimeoutError("tool budget exceeded"))
                durations.append(0.0)
                continue
            res, dur = await _call_tool_with_retry_timed(
                session,
                n,
                a,
                attempts,
                timeout_s=tool_timeout_s,
                deadline=deadline,
            )
            results.append(res)
            durations.append(dur)
    res_iter = iter(results)
    dur_iter = iter(durations)
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
        dur = next(dur_iter, None)
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
        _record_tool_event(name, arguments, content, is_error, hint, duration_s=dur)
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
    provider: BaseProvider | None = None,
    request_id: str | None = None,
    disallow_tools: bool = False,
    timeout_seconds: float | None = None,
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
    provider_tools = mcp_tools_to_ollama(list_result.tools)
    run_start = time.monotonic()
    run_budget_s = _run_budget_seconds()
    overall_deadline = (run_start + timeout_seconds) if timeout_seconds else None
    provider_tools = _filter_tools_by_policy(
        provider_tools, allowed_tools, blocked_tools
    )
    provider_tools = use_short_names_for_builtin_tools(provider_tools)
    ollama_tools = add_tool_aliases_for_ollama(provider_tools, TOOL_NAME_ALIASES)
    ollama_tools = add_harmony_function_aliases(ollama_tools)
    if not provider_tools:
        provider_tools = []
        ollama_tools = []  # Ollama may require None for no tools; check docs
    valid_tool_names = set(_active_tool_names(ollama_tools))

    messages: list[dict[str, Any]] = []
    if disallow_tools:
        tool_block = "Tool use is disabled for this phase. Do not call tools."
        if system_prompt:
            system_prompt = system_prompt + "\n\n" + tool_block
        else:
            system_prompt = tool_block
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    tools_hint = _tools_system_prompt(provider_tools)
    if tools_hint:
        messages.append({"role": "system", "content": tools_hint})
    if message_history:
        messages.extend(message_history)
    user_msg: dict[str, Any] = {"role": "user", "content": user_message}
    if image_paths:
        user_msg["images"] = [str(p) for p in image_paths]
    messages.append(user_msg)

    turn_start = time.perf_counter() if timing else None
    tool_calls_total = 0
    tool_errors_total = 0
    disallow_tools_warned = False
    disable_token = disable_tool_calls() if disallow_tools else None
    doom_detector = _DoomLoopDetector()

    def _finish(out: str) -> str:
        if disable_token is not None:
            reset_tool_calls(disable_token)
        if turn_start is not None:
            _json_log_event(
                event="trace_summary",
                request_id=request_id,
                tool_calls=tool_calls_total,
                tool_errors=tool_errors_total,
                duration_s=round(time.perf_counter() - turn_start, 3),
            )
        _emit_replay_report()
        return out

    for round_num in range(max_tool_rounds):
        if overall_deadline and time.monotonic() > overall_deadline:
            _json_log_event(
                event="overall_timeout",
                request_id=request_id,
                round=round_num + 1,
                timeout_seconds=timeout_seconds,
            )
            return _finish(
                "Agent loop timed out. Reduce scope or increase timeout_seconds."
            )
        if run_budget_s and (time.monotonic() - run_start) > run_budget_s:
            _json_log_event(
                event="run_budget_exceeded", request_id=request_id, round=round_num + 1
            )
            return _finish(
                "Run budget exceeded. Reduce scope or increase OLLAMACODE_RUN_BUDGET_SECONDS."
            )
        # 2. Call Ollama (sync in thread); truncate if over max_messages
        if not quiet and not tool_progress_brief:
            logger.info("Sending to model (turn %s)...", round_num + 1)
        to_send = (
            _truncate_messages(messages, max_messages) if max_messages > 0 else messages
        )
        t0 = time.perf_counter() if timing else None
        try:
            if provider is not None:
                response = await provider.chat_async(model, to_send, provider_tools)
            else:
                response = await ollama_chat_async(model, to_send, ollama_tools)
        except Exception as e:  # noqa: BLE001
            raise _wrap_ollama_template_error(e)
        if timing and t0 is not None:
            elapsed = time.perf_counter() - t0
            logger.info("LLM call: %.2fs", elapsed)
            _json_log_event(
                event="llm", duration_s=round(elapsed, 3), request_id=request_id
            )

        msg = _get(response, "message")
        if msg is None:
            return _finish("No response from model.")

        content = _get(msg, "content") or ""
        tool_calls = _get(msg, "tool_calls") or []
        if tool_calls:
            tool_calls_total += len(tool_calls)

        if disallow_tools and tool_calls:
            if not disallow_tools_warned:
                disallow_tools_warned = True
                messages.append(
                    {
                        "role": "user",
                        "content": "Tool calls are disabled for this phase. Reply with plain text only.",
                    }
                )
                continue
            return _finish(
                (content or "").strip() or "Tool calls are disabled for this phase."
            )

        # Build assistant message for history (Ollama format: role, content, tool_calls?)
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            if timing and turn_start is not None:
                logger.info("Turn total: %.2fs", time.perf_counter() - turn_start)
            return _finish((content or "").strip())

        # 3. Execute tool calls via MCP (in parallel when multiple)
        tool_timeout_s = _tool_timeout_seconds()
        tool_budget_s = _tool_budget_seconds()
        tool_deadline = time.monotonic() + tool_budget_s if tool_budget_s else None
        items: list[tuple[str, dict[str, Any], str | None]] = []
        for tc in tool_calls:
            fn = _get(tc, "function")
            if fn is None:
                continue
            name = _get(fn, "name")
            if not name:
                continue
            # Validate tool name against available tools
            if valid_tool_names and name not in valid_tool_names:
                err_content = (
                    f"Unknown tool '{name}'. Available tools: "
                    f"{', '.join(sorted(valid_tool_names)[:20])}"
                )
                messages.append(
                    {"role": "tool", "tool_name": name, "content": err_content}
                )
                tool_errors_total += 1
                logger.debug("Rejected unknown tool call: %s", name)
                continue
            raw_args = _get(fn, "arguments")
            arguments, parse_err = _parse_tool_args(raw_args)
            items.append((name, arguments, parse_err))

        # Doom loop detection: record and check for repetitive tool calls
        for name, arguments, _pe in items:
            doom_detector.record(name, arguments)
        looping, doom_msg = doom_detector.is_looping()
        if looping:
            messages.append(
                {"role": "user", "content": doom_msg or "Breaking doom loop."}
            )
            return _finish(doom_msg or "Doom loop detected. Stopping agent.")

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
                        except Exception as _exc:
                            logger.debug("on_tool_start callback error: %s", _exc)
                    if parse_err is not None:
                        content = f"Tool arguments could not be parsed: {parse_err}. Please output valid JSON for the arguments."
                        content = _truncate_tool_result(content, max_tool_result_chars)
                        messages.append(
                            {"role": "tool", "tool_name": name, "content": content}
                        )
                        tool_errors_total += 1
                        if on_tool_end is not None:
                            try:
                                on_tool_end(name, arguments, content)
                            except Exception as _exc:
                                logger.debug("on_tool_end callback error: %s", _exc)
                        _json_log_event(
                            event="tool",
                            tool=name,
                            result_chars=len(content),
                            is_error=True,
                            request_id=request_id,
                        )
                        continue
                    if before_tool_call is None:
                        raise RuntimeError(
                            "before_tool_call callback is None but confirm_tool_calls is True"
                        )
                    decision = await before_tool_call(name, arguments)
                    if decision == "skip" or (
                        isinstance(decision, tuple) and decision[0] == "skip"
                    ):
                        content = (
                            decision[1]
                            if isinstance(decision, tuple) and len(decision) > 1
                            else "Skipped by user."
                        )
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
                            except Exception as _exc:
                                logger.debug("on_tool_end callback error: %s", _exc)
                        _json_log_event(
                            event="tool",
                            tool=name,
                            result_chars=len(content),
                            is_error=False,
                            request_id=request_id,
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
                        session,
                        name,
                        arguments,
                        max_tool_result_chars,
                        deadline=tool_deadline,
                        timeout_s=tool_timeout_s,
                        request_id=request_id,
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
                        except Exception as _exc:
                            logger.debug("on_tool_end callback error: %s", _exc)
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
                        except Exception as _exc:
                            logger.debug("append_past_error failed: %s", _exc)
                    if is_error:
                        tool_errors_total += 1
                    messages.append(
                        {"role": "tool", "tool_name": name, "content": content}
                    )
                    _json_log_event(
                        event="tool",
                        tool=name,
                        result_chars=len(content),
                        is_error=is_error,
                        **({"error_hint": hint} if hint else {}),
                        request_id=request_id,
                    )
            else:
                for name, arguments, _ in items:
                    if on_tool_start is not None:
                        try:
                            on_tool_start(name, arguments)
                        except Exception as _exc:
                            logger.debug("on_tool_start callback error: %s", _exc)
                tool_results = await run_tools(
                    session, items, max_tool_result_chars, request_id=request_id
                )
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
                        except Exception as _exc:
                            logger.debug("on_tool_end callback error: %s", _exc)
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
                        except Exception as _exc:
                            logger.debug("append_past_error failed: %s", _exc)
                    if is_error:
                        tool_errors_total += 1
                    messages.append(
                        {"role": "tool", "tool_name": name, "content": content}
                    )
                    _json_log_event(
                        event="tool",
                        tool=name,
                        result_chars=len(content),
                        is_error=is_error,
                        **({"error_hint": hint} if hint else {}),
                        request_id=request_id,
                    )
            if timing and t0_parallel is not None:
                elapsed = time.perf_counter() - t0_parallel
                logger.info("Tools: %.2fs", elapsed)
                _json_log_event(
                    event="tools",
                    duration_s=round(elapsed, 3),
                    count=len(items),
                    request_id=request_id,
                )

    if timing and turn_start is not None:
        turn_elapsed = time.perf_counter() - turn_start
        logger.info("Turn total: %.2fs", turn_elapsed)
        _json_log_event(
            event="turn", duration_s=round(turn_elapsed, 3), request_id=request_id
        )
    return _finish("(Max tool rounds reached; stopping.)")


def _is_tool_call_parse_error(exc: BaseException) -> bool:
    """True if Ollama failed to parse tool-call JSON (retry often fixes it)."""
    s = str(exc).lower()
    return "error parsing tool call" in s or "invalid character" in s


def _is_stream_parse_error(exc: BaseException) -> bool:
    """True if Ollama stream returned malformed JSON chunk."""
    s = str(exc).lower()
    return "jsondecodeerror" in s or "expecting value" in s


def _is_json_decode_error(exc: BaseException) -> bool:
    return (
        isinstance(exc, json.JSONDecodeError) or "expecting value" in str(exc).lower()
    )


def _ollama_chat_nonstream_with_retry(
    model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> Any:
    """Non-stream Ollama chat with retry/timeout and fallback to no-tools."""
    timeout_s = float(os.environ.get("OLLAMACODE_OLLAMA_TIMEOUT_SECONDS", "150"))

    def _call(tools_arg: list[dict[str, Any]]) -> Any:
        result: dict[str, Any] = {"resp": None, "error": None}

        def _run() -> None:
            try:
                result["resp"] = ollama.chat(
                    model=model, messages=messages, tools=tools_arg, stream=False
                )
            except Exception as e:  # noqa: BLE001
                result["error"] = e

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=timeout_s)
        if t.is_alive():
            raise TimeoutError("Ollama chat timed out")
        if result["error"] is not None:
            raise result["error"]
        return result["resp"]

    last_err: Exception | None = None
    tool_sets = [tools]
    if tools:
        tool_sets.append([])  # fallback: retry without tools
    for tools_arg in tool_sets:
        for _ in range(2):
            try:
                return _call(tools_arg)
            except Exception as e:  # noqa: BLE001
                last_err = e
                if isinstance(e, TimeoutError):
                    break
                if _is_json_decode_error(e):
                    continue
                break
    if last_err is not None:
        raise last_err
    raise RuntimeError("Ollama chat failed")


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
                    msg = _get(chunk, "message")
                    content = _get(msg, "content", "") or "" if msg else ""
                    done = _get(chunk, "done", False)
                    q.put((content, done, msg))
                break
            except Exception as e:
                if attempt == 0 and (
                    _is_tool_call_parse_error(e) or _is_stream_parse_error(e)
                ):
                    continue  # retry once
                if _is_stream_parse_error(e):
                    # Fallback to non-stream response for this turn.
                    try:
                        resp = _ollama_chat_nonstream_with_retry(
                            model=model, messages=messages, tools=tools
                        )
                        msg = _get(resp, "message")
                        content = _get(msg, "content", "") or "" if msg else ""
                        q.put((content, True, msg))
                    except Exception as e2:
                        q.put(_wrap_ollama_template_error(e2))
                    break
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
    provider: BaseProvider | None = None,
    request_id: str | None = None,
    disallow_tools: bool = False,
) -> AsyncIterator[str]:
    """
    Like run_agent_loop but streams content tokens as they arrive.
    Yields content fragments (str). When tool_calls occur, runs them and continues streaming the next turn.
    message_history: optional prior turns [{"role":"user","content":...},{"role":"assistant","content":...}, ...].
    tool_progress_brief: when True, print one line per tool to reduce terminal flicker during streaming.
    """
    list_result = await list_tools(session)
    provider_tools = mcp_tools_to_ollama(list_result.tools)
    run_start = time.monotonic()
    run_budget_s = _run_budget_seconds()
    provider_tools = _filter_tools_by_policy(
        provider_tools, allowed_tools, blocked_tools
    )
    provider_tools = use_short_names_for_builtin_tools(provider_tools)
    ollama_tools = add_tool_aliases_for_ollama(provider_tools, TOOL_NAME_ALIASES)
    ollama_tools = add_harmony_function_aliases(ollama_tools) or []
    valid_tool_names = set(_active_tool_names(ollama_tools))

    messages: list[dict[str, Any]] = []
    if disallow_tools:
        tool_block = "Tool use is disabled for this phase. Do not call tools."
        if system_prompt:
            system_prompt = system_prompt + "\n\n" + tool_block
        else:
            system_prompt = tool_block
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    tools_hint = _tools_system_prompt(provider_tools)
    if tools_hint:
        messages.append({"role": "system", "content": tools_hint})
    if message_history:
        messages.extend(message_history)
    user_msg_stream: dict[str, Any] = {"role": "user", "content": user_message}
    if image_paths:
        user_msg_stream["images"] = [str(p) for p in image_paths]
    messages.append(user_msg_stream)

    loop = asyncio.get_event_loop()
    turn_start = time.perf_counter() if timing else None
    tool_calls_total = 0
    tool_errors_total = 0
    disallow_tools_warned = False
    disable_token = disable_tool_calls() if disallow_tools else None
    doom_detector = _DoomLoopDetector()

    def _finish() -> None:
        if disable_token is not None:
            reset_tool_calls(disable_token)
        if turn_start is not None:
            _json_log_event(
                event="trace_summary",
                request_id=request_id,
                tool_calls=tool_calls_total,
                tool_errors=tool_errors_total,
                duration_s=round(time.perf_counter() - turn_start, 3),
            )
        _emit_replay_report()

    for round_num in range(max_tool_rounds):
        if run_budget_s and (time.monotonic() - run_start) > run_budget_s:
            _json_log_event(
                event="run_budget_exceeded", request_id=request_id, round=round_num + 1
            )
            yield "Run budget exceeded. Reduce scope or increase OLLAMACODE_RUN_BUDGET_SECONDS.\n"
            _finish()
            return
        if not quiet and not tool_progress_brief:
            logger.info("Sending to model (turn %s)...", round_num + 1)
        q: queue.Queue = queue.Queue()
        to_send = (
            _truncate_messages(messages, max_messages) if max_messages > 0 else messages
        )
        t0 = time.perf_counter() if timing else None

        if provider is not None:
            q2: queue.Queue = queue.Queue()

            # For provider-backed tool loops, prefer non-stream chat so we can
            # preserve tool-calls from the model response.
            if ollama_tools:
                try:
                    response = await provider.chat_async(model, to_send, provider_tools)
                    msg = _get(response, "message")
                    content = _get(msg, "content", "") or "" if msg else ""
                    if content:
                        if os.environ.get("OLLAMACODE_STREAM_WITH_TOOLS", "0") == "1":
                            chunk_size = int(
                                os.environ.get("OLLAMACODE_STREAM_CHUNK_CHARS", "240")
                            )
                            for frag in _chunk_text(content, chunk_size):
                                yield frag
                        else:
                            yield content
                    last_msg = msg or {}
                except Exception as e:
                    raise _wrap_ollama_template_error(e)
            else:

                def _stream_provider() -> None:
                    try:
                        for (frag,) in provider.chat_stream_sync(model, to_send):
                            q2.put(frag)
                    except Exception as e:  # noqa: BLE001
                        q2.put(e)
                    finally:
                        q2.put(None)

                thread2 = threading.Thread(target=_stream_provider, daemon=True)
                thread2.start()
                try:
                    buf: list[str] = []
                    stream_err: BaseException | None = None
                    while True:
                        item = await loop.run_in_executor(None, q2.get)
                        if item is None:
                            break
                        if isinstance(item, BaseException):
                            stream_err = item
                            break
                        if item:
                            buf.append(item)
                            yield item
                    if stream_err is not None:
                        # Fallback to non-stream chat if provider streaming fails.
                        response = await provider.chat_async(model, to_send, [])
                        msg = _get(response, "message")
                        content = _get(msg, "content", "") or "" if msg else ""
                        if content:
                            yield content
                        last_msg = msg or {"content": ""}
                    else:
                        last_msg = {"content": "".join(buf)}
                except Exception as e:
                    raise _wrap_ollama_template_error(e)
                finally:
                    thread2.join()
        else:
            # Ollama: real token-by-token streaming via background thread + queue
            stream_with_tools = (
                os.environ.get("OLLAMACODE_STREAM_WITH_TOOLS", "0") == "1"
            )
            if ollama_tools and not stream_with_tools:
                # Non-stream path for tools: more reliable than streaming with tool calls.
                try:
                    resp = _ollama_chat_nonstream_with_retry(
                        model=model, messages=to_send, tools=ollama_tools
                    )
                    msg = _get(resp, "message")
                    content = _get(msg, "content", "") or "" if msg else ""
                    if content:
                        yield content
                    last_msg = msg or {}
                except Exception as e:
                    raise _wrap_ollama_template_error(e)
            else:
                thread = threading.Thread(
                    target=_stream_into_queue,
                    args=(q, model, to_send, ollama_tools),
                )
                thread.daemon = True
                thread.start()
                last_msg = None
                stream_timeout = float(
                    os.environ.get("OLLAMACODE_STREAM_TIMEOUT_SECONDS", "60")
                )
                stream_total_timeout = float(
                    os.environ.get("OLLAMACODE_STREAM_TOTAL_TIMEOUT_SECONDS", "90")
                )
                stream_start = time.monotonic()
                while True:
                    if time.monotonic() - stream_start > stream_total_timeout:
                        # Hard stop: fall back to non-stream response.
                        try:
                            resp = _ollama_chat_nonstream_with_retry(
                                model=model, messages=to_send, tools=ollama_tools
                            )
                            msg = _get(resp, "message")
                            content = _get(msg, "content", "") or "" if msg else ""
                            if content:
                                yield content
                            last_msg = msg
                        except Exception as e:
                            raise _wrap_ollama_template_error(e)
                        break
                    try:
                        item = await loop.run_in_executor(
                            None, lambda: q.get(timeout=stream_timeout)
                        )
                    except Exception:
                        # Streaming stalled; fall back to non-stream response.
                        try:
                            resp = _ollama_chat_nonstream_with_retry(
                                model=model, messages=to_send, tools=ollama_tools
                            )
                            msg = _get(resp, "message")
                            content = _get(msg, "content", "") or "" if msg else ""
                            if content:
                                yield content
                            last_msg = msg
                        except Exception as e:
                            raise _wrap_ollama_template_error(e)
                        break
                    if item is None:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    content_frag, done, msg = item
                    if content_frag:
                        yield content_frag
                    last_msg = msg
                    if done:
                        break
                thread.join(timeout=60)
                last_msg = last_msg or {}

        if timing and t0 is not None:
            elapsed = time.perf_counter() - t0
            logger.info("LLM call: %.2fs", elapsed)
            _json_log_event(
                event="llm", duration_s=round(elapsed, 3), request_id=request_id
            )

        if not last_msg:
            _finish()
            return
        content = _get(last_msg, "content") or ""
        tool_calls = _get(last_msg, "tool_calls") or []
        if tool_calls:
            tool_calls_total += len(tool_calls)
        if disallow_tools and tool_calls:
            if not disallow_tools_warned:
                disallow_tools_warned = True
                messages.append(
                    {
                        "role": "user",
                        "content": "Tool calls are disabled for this phase. Reply with plain text only.",
                    }
                )
                continue
            yield "Tool calls are disabled for this phase.\n"
            _finish()
            return
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            if timing and turn_start is not None:
                logger.info("Turn total: %.2fs", time.perf_counter() - turn_start)
                _json_log_event(
                    event="turn",
                    duration_s=round(time.perf_counter() - turn_start, 3),
                    request_id=request_id,
                )
            _finish()
            return

        tool_timeout_s = _tool_timeout_seconds()
        tool_budget_s = _tool_budget_seconds()
        tool_deadline = time.monotonic() + tool_budget_s if tool_budget_s else None
        items_stream: list[tuple[str, dict[str, Any], str | None]] = []
        for tc in tool_calls:
            fn = _get(tc, "function")
            if fn is None:
                continue
            name = _get(fn, "name")
            if not name:
                continue
            # Validate tool name against available tools
            if valid_tool_names and name not in valid_tool_names:
                err_content = (
                    f"Unknown tool '{name}'. Available tools: "
                    f"{', '.join(sorted(valid_tool_names)[:20])}"
                )
                messages.append(
                    {"role": "tool", "tool_name": name, "content": err_content}
                )
                tool_errors_total += 1
                logger.debug("Rejected unknown tool call: %s", name)
                continue
            raw_args = _get(fn, "arguments")
            arguments, parse_err = _parse_tool_args(raw_args)
            items_stream.append((name, arguments, parse_err))
        # Doom loop detection for streaming loop
        for name, arguments, _pe in items_stream:
            doom_detector.record(name, arguments)
        looping, doom_msg = doom_detector.is_looping()
        if looping:
            yield doom_msg or "Doom loop detected. Stopping agent."
            _finish()
            return
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
                        except Exception as _exc:
                            logger.debug("on_tool_start callback error: %s", _exc)
                    if parse_err is not None:
                        content = f"Tool arguments could not be parsed: {parse_err}. Please output valid JSON for the arguments."
                        content = _truncate_tool_result(content, max_tool_result_chars)
                        messages.append(
                            {"role": "tool", "tool_name": name, "content": content}
                        )
                        tool_errors_total += 1
                        if on_tool_end is not None:
                            try:
                                on_tool_end(name, arguments, content)
                            except Exception as _exc:
                                logger.debug("on_tool_end callback error: %s", _exc)
                        _json_log_event(
                            event="tool",
                            tool=name,
                            result_chars=len(content),
                            is_error=True,
                            request_id=request_id,
                        )
                        continue
                    if before_tool_call is None:
                        raise RuntimeError(
                            "before_tool_call callback is None but confirm_tool_calls is True"
                        )
                    decision = await before_tool_call(name, arguments)
                    if decision == "skip" or (
                        isinstance(decision, tuple) and decision[0] == "skip"
                    ):
                        content = (
                            decision[1]
                            if isinstance(decision, tuple) and len(decision) > 1
                            else "Skipped by user."
                        )
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
                            except Exception as _exc:
                                logger.debug("on_tool_end callback error: %s", _exc)
                        _json_log_event(
                            event="tool",
                            tool=name,
                            result_chars=len(content),
                            is_error=False,
                            request_id=request_id,
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
                        session,
                        name,
                        arguments,
                        max_tool_result_chars,
                        deadline=tool_deadline,
                        timeout_s=tool_timeout_s,
                        request_id=request_id,
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
                        except Exception as _exc:
                            logger.debug("on_tool_end callback error: %s", _exc)
                    messages.append(
                        {"role": "tool", "tool_name": name, "content": content}
                    )
                    _json_log_event(
                        event="tool",
                        tool=name,
                        result_chars=len(content),
                        is_error=is_error,
                        **({"error_hint": hint} if hint else {}),
                        request_id=request_id,
                    )
                    if is_error:
                        tool_errors_total += 1
            else:
                for name, arguments, _ in items_stream:
                    if on_tool_start is not None:
                        try:
                            on_tool_start(name, arguments)
                        except Exception as _exc:
                            logger.debug("on_tool_start callback error: %s", _exc)
                tool_results_stream = await run_tools(
                    session, items_stream, max_tool_result_chars, request_id=request_id
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
                        except Exception as _exc:
                            logger.debug("on_tool_end callback error: %s", _exc)
                    messages.append(
                        {"role": "tool", "tool_name": name, "content": content}
                    )
                    _json_log_event(
                        event="tool",
                        tool=name,
                        result_chars=len(content),
                        is_error=is_error,
                        **({"error_hint": hint} if hint else {}),
                        request_id=request_id,
                    )
                    if is_error:
                        tool_errors_total += 1

    _finish()
    return


async def run_agent_loop_no_mcp(
    model: str,
    user_message: str,
    *,
    system_prompt: str | None = None,
    message_history: list[dict[str, Any]] | None = None,
    provider: BaseProvider | None = None,
    timing: bool = False,
    request_id: str | None = None,
) -> str:
    """Run one turn with no MCP tools. Returns assistant text."""
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if message_history:
        messages.extend(message_history)
    messages.append({"role": "user", "content": user_message})

    t0 = time.perf_counter() if timing else None
    try:
        if provider is not None:
            response = await provider.chat_async(model, messages, [])
        else:
            response = await ollama_chat_async(model, messages, [])
    except Exception as e:  # noqa: BLE001
        raise _wrap_ollama_template_error(e)
    if timing and t0 is not None:
        elapsed = time.perf_counter() - t0
        _json_log_event(
            event="llm", duration_s=round(elapsed, 3), request_id=request_id
        )
    msg = _get(response, "message")
    if msg is None:
        return "No response from model."
    content = _get(msg, "content")
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
    provider: BaseProvider | None = None,
    timing: bool = False,
    request_id: str | None = None,
) -> AsyncIterator[str]:
    """Stream one turn with no MCP tools. Yields content fragments."""
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if message_history:
        messages.extend(message_history)
    messages.append({"role": "user", "content": user_message})

    t0 = time.perf_counter() if timing else None
    if provider is not None:
        # Non-Ollama provider: use chat_stream_sync in a background thread
        q2: queue.Queue = queue.Queue()

        def _stream_provider() -> None:
            try:
                for (frag,) in provider.chat_stream_sync(model, messages):
                    q2.put(frag)
            except Exception as e:  # noqa: BLE001
                q2.put(e)
            finally:
                q2.put(None)

        thread2 = threading.Thread(target=_stream_provider)
        thread2.start()
        loop2 = asyncio.get_event_loop()
        try:
            stream_err: BaseException | None = None
            collected: list[str] = []
            while True:
                item = await loop2.run_in_executor(None, q2.get)
                if item is None:
                    break
                if isinstance(item, BaseException):
                    stream_err = item
                    break
                if item:
                    collected.append(item)
                    yield item
            if stream_err is not None:
                # Fallback to non-stream chat when provider streaming fails.
                response = await provider.chat_async(model, messages, [])
                msg = _get(response, "message")
                content = _get(msg, "content") or "" if msg else ""
                if content:
                    yield str(content)
        finally:
            thread2.join()
        if timing and t0 is not None:
            elapsed = time.perf_counter() - t0
            _json_log_event(
                event="llm", duration_s=round(elapsed, 3), request_id=request_id
            )
        return

    # Ollama path: use existing thread queue
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
        thread.join(timeout=60)
        if timing and t0 is not None:
            elapsed = time.perf_counter() - t0
            _json_log_event(
                event="llm", duration_s=round(elapsed, 3), request_id=request_id
            )
