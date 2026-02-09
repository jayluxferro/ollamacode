"""
Agent loop: user message + MCP tools → Ollama chat → tool_calls → MCP call_tool → append results → repeat.
"""

from __future__ import annotations

import asyncio
import json
import queue
import sys
import threading
import time
from collections.abc import AsyncIterator
from typing import Any

import ollama

from .bridge import add_tool_aliases_for_ollama, mcp_tools_to_ollama, use_short_names_for_builtin_tools
from .mcp_client import (
    TOOL_NAME_ALIASES,
    McpConnection,
    call_tool,
    list_tools,
    tool_result_to_content,
)


def _get(o: Any, key: str, default: Any = None) -> Any:
    if isinstance(o, dict):
        return o.get(key, default)
    return getattr(o, key, default)


def _parse_tool_args(raw_args: str | dict | None) -> dict[str, Any]:
    """Parse tool-call arguments; tolerate common model JSON errors (e.g. extra trailing '}')."""
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    s = (raw_args or "").strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Common LLM mistake: extra '}' at end (e.g. '{"a":1}}')
    if s.endswith("}}"):
        try:
            return json.loads(s[:-1])
        except json.JSONDecodeError:
            pass
    return {}


def _log_tool_call(name: str, arguments: dict[str, Any]) -> None:
    """Print tool name and full args to stderr."""
    args_str = json.dumps(arguments, indent=2) if arguments else "{}"
    if len(args_str) > 400:
        args_str = args_str[:400] + "\n  ..."
    print(f"[OllamaCode] Calling {name}:", file=sys.stderr, flush=True)
    for line in args_str.splitlines():
        print(f"  {line}", file=sys.stderr, flush=True)


def _log_tool_result(name: str, content: str, is_error: bool = False) -> None:
    """Print tool result to stderr (truncated)."""
    label = "Error" if is_error else "Result"
    max_len = 1200
    if len(content) > max_len:
        content = content[:max_len] + "\n  ... (truncated)"
    for line in content.splitlines():
        print(f"  [{name}] {label}: {line}", file=sys.stderr, flush=True)
    if not content.strip():
        print(f"  [{name}] {label}: (empty)", file=sys.stderr, flush=True)


def _truncate_messages(messages: list[dict[str, Any]], max_messages: int) -> list[dict[str, Any]]:
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


def _ollama_chat_sync(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    stream: bool = False,
) -> Any:
    """Sync Ollama chat call (run in thread to avoid blocking)."""
    return ollama.chat(model=model, messages=messages, tools=tools, stream=stream)


async def run_agent_loop(
    session: McpConnection,
    model: str,
    user_message: str,
    *,
    system_prompt: str | None = None,
    max_tool_rounds: int = 20,
    max_messages: int = 0,
    message_history: list[dict[str, Any]] | None = None,
    quiet: bool = False,
    timing: bool = False,
) -> str:
    """
    Run one user turn: send message to Ollama with MCP tools; on tool_calls,
    execute via MCP and re-call Ollama until the model returns text only.
    message_history: optional prior turns for multi-turn conversation.
    """
    # 1. Get MCP tools and convert to Ollama format; use short names for built-in tools so the model sees read_file, run_command, etc.
    list_result = await list_tools(session)
    ollama_tools = mcp_tools_to_ollama(list_result.tools)
    ollama_tools = use_short_names_for_builtin_tools(ollama_tools)
    ollama_tools = add_tool_aliases_for_ollama(ollama_tools, TOOL_NAME_ALIASES)
    if not ollama_tools:
        ollama_tools = []  # Ollama may require None for no tools; check docs

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if message_history:
        messages.extend(message_history)
    messages.append({"role": "user", "content": user_message})

    turn_start = time.perf_counter() if timing else None
    for round_num in range(max_tool_rounds):
        # 2. Call Ollama (sync in thread); truncate if over max_messages
        if not quiet:
            print(
                f"[OllamaCode] Sending to model (turn {round_num + 1})...",
                file=sys.stderr,
                flush=True,
            )
        to_send = _truncate_messages(messages, max_messages) if max_messages > 0 else messages
        loop = asyncio.get_event_loop()
        t0 = time.perf_counter() if timing else None
        response = await loop.run_in_executor(
            None,
            lambda m=model, msgs=to_send, t=ollama_tools: _ollama_chat_sync(m, msgs, t),
        )
        if timing and t0 is not None:
            print(
                f"[OllamaCode] Ollama call: {time.perf_counter() - t0:.2f}s",
                file=sys.stderr,
                flush=True,
            )

        msg = response.get("message") if isinstance(response, dict) else getattr(response, "message", None)
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
                print(
                    f"[OllamaCode] Turn total: {time.perf_counter() - turn_start:.2f}s",
                    file=sys.stderr,
                    flush=True,
                )
            return (content or "").strip()

        # 3. Execute each tool call via MCP
        names = []
        for tc in tool_calls:
            fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
            if fn is not None:
                n = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
                if n:
                    names.append(n)
        if names and not quiet:
            print(
                f"[OllamaCode] Model requested {len(names)} tool(s): {', '.join(names)}",
                file=sys.stderr,
                flush=True,
            )

        for tc in tool_calls:
            fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
            if fn is None:
                continue
            name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
            if not name:
                continue
            raw_args = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", None)
            arguments = _parse_tool_args(raw_args)

            if not quiet:
                _log_tool_call(name, arguments)
            t0 = time.perf_counter() if timing else None
            result = await call_tool(session, name, arguments)
            if timing and t0 is not None:
                print(
                    f"[OllamaCode] {name}: {time.perf_counter() - t0:.2f}s",
                    file=sys.stderr,
                    flush=True,
                )
            content = tool_result_to_content(result)
            if not quiet:
                _log_tool_result(name, content, is_error=getattr(result, "isError", False))
            messages.append(
                {
                    "role": "tool",
                    "tool_name": name,
                    "content": content,
                }
            )

    if timing and turn_start is not None:
        print(
            f"[OllamaCode] Turn total: {time.perf_counter() - turn_start:.2f}s",
            file=sys.stderr,
            flush=True,
        )
    return "(Max tool rounds reached; stopping.)"


def _stream_into_queue(
    q: queue.Queue,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> None:
    """Run Ollama chat with stream=True and put (content_fragment, done, message) into q. Puts None at end.
    On error (e.g. model produced invalid tool-call JSON), puts the exception then None so the caller can re-raise.
    """
    try:
        stream = ollama.chat(model=model, messages=messages, tools=tools, stream=True)
        for chunk in stream:
            msg = chunk.get("message") if isinstance(chunk, dict) else getattr(chunk, "message", None)
            content = _get(msg, "content", "") or "" if msg else ""
            done = chunk.get("done") if isinstance(chunk, dict) else getattr(chunk, "done", False)
            q.put((content, done, msg))
    except Exception as e:  # e.g. ollama.ResponseError when server fails to parse tool-call JSON
        if "error parsing tool call" in str(e).lower() or "invalid character" in str(e).lower():
            err = ValueError(
                "Model produced invalid tool-call JSON (e.g. extra '}' or ']', or unescaped newlines in strings). "
                "Multi-line arguments must use \\n, not literal line breaks. Try again or use a model with stricter JSON."
            )
            err.__cause__ = e
            q.put(err)
        else:
            q.put(e)
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
    message_history: list[dict[str, Any]] | None = None,
    quiet: bool = False,
    timing: bool = False,
) -> AsyncIterator[str]:
    """
    Like run_agent_loop but streams content tokens as they arrive.
    Yields content fragments (str). When tool_calls occur, runs them and continues streaming the next turn.
    message_history: optional prior turns [{"role":"user","content":...},{"role":"assistant","content":...}, ...].
    """
    list_result = await list_tools(session)
    ollama_tools = use_short_names_for_builtin_tools(mcp_tools_to_ollama(list_result.tools))
    ollama_tools = add_tool_aliases_for_ollama(ollama_tools, TOOL_NAME_ALIASES) or []

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if message_history:
        messages.extend(message_history)
    messages.append({"role": "user", "content": user_message})

    loop = asyncio.get_event_loop()
    turn_start = time.perf_counter() if timing else None
    for round_num in range(max_tool_rounds):
        if not quiet:
            print(
                f"[OllamaCode] Sending to model (turn {round_num + 1})...",
                file=sys.stderr,
                flush=True,
            )
        q: queue.Queue = queue.Queue()
        to_send = _truncate_messages(messages, max_messages) if max_messages > 0 else messages
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
            print(
                f"[OllamaCode] Ollama call: {time.perf_counter() - t0:.2f}s",
                file=sys.stderr,
                flush=True,
            )

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
                print(
                    f"[OllamaCode] Turn total: {time.perf_counter() - turn_start:.2f}s",
                    file=sys.stderr,
                    flush=True,
                )
            return

        names = []
        for tc in tool_calls:
            fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
            if fn is not None:
                n = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
                if n:
                    names.append(n)
        if names and not quiet:
            print(
                f"[OllamaCode] Model requested {len(names)} tool(s): {', '.join(names)}",
                file=sys.stderr,
                flush=True,
            )

        for tc in tool_calls:
            fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
            if fn is None:
                continue
            name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
            if not name:
                continue
            raw_args = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", None)
            arguments = _parse_tool_args(raw_args)
            if not quiet:
                _log_tool_call(name, arguments)
            t0 = time.perf_counter() if timing else None
            result = await call_tool(session, name, arguments)
            if timing and t0 is not None:
                print(
                    f"[OllamaCode] {name}: {time.perf_counter() - t0:.2f}s",
                    file=sys.stderr,
                    flush=True,
                )
            content = tool_result_to_content(result)
            if not quiet:
                _log_tool_result(name, content, is_error=getattr(result, "isError", False))
            messages.append({"role": "tool", "tool_name": name, "content": content})


async def run_agent_loop_no_mcp(
    model: str,
    user_message: str,
    *,
    system_prompt: str | None = None,
    message_history: list[dict[str, Any]] | None = None,
) -> str:
    """Run one turn with Ollama only (no MCP tools). Returns assistant text."""
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if message_history:
        messages.extend(message_history)
    messages.append({"role": "user", "content": user_message})

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: _ollama_chat_sync(model, messages, []),
    )
    msg = response.get("message") if isinstance(response, dict) else getattr(response, "message", None)
    if msg is None:
        return "No response from model."
    content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
    return (content or "").strip()


def _stream_no_mcp_into_queue(
    q: queue.Queue,
    model: str,
    messages: list[dict[str, Any]],
) -> None:
    try:
        stream = ollama.chat(model=model, messages=messages, tools=[], stream=True)
        for chunk in stream:
            msg = chunk.get("message") if isinstance(chunk, dict) else getattr(chunk, "message", None)
            content = _get(msg, "content", "") or "" if msg else ""
            q.put((content,))
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
    thread = threading.Thread(target=_stream_no_mcp_into_queue, args=(q, model, messages))
    thread.start()
    loop = asyncio.get_event_loop()
    try:
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is None:
                break
            if item[0]:
                yield item[0]
    finally:
        thread.join()
