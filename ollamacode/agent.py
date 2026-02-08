"""
Agent loop: user message + MCP tools → Ollama chat → tool_calls → MCP call_tool → append results → repeat.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from collections.abc import AsyncIterator
from typing import Any

import ollama

from .bridge import mcp_tools_to_ollama
from .mcp_client import McpConnection, call_tool, list_tools, tool_result_to_content


def _get(o: Any, key: str, default: Any = None) -> Any:
    if isinstance(o, dict):
        return o.get(key, default)
    return getattr(o, key, default)


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
) -> str:
    """
    Run one user turn: send message to Ollama with MCP tools; on tool_calls,
    execute via MCP and re-call Ollama until the model returns text only.
    """
    # 1. Get MCP tools and convert to Ollama format
    list_result = await list_tools(session)
    ollama_tools = mcp_tools_to_ollama(list_result.tools)
    if not ollama_tools:
        ollama_tools = []  # Ollama may require None for no tools; check docs

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    for _ in range(max_tool_rounds):
        # 2. Call Ollama (sync in thread); truncate if over max_messages
        to_send = _truncate_messages(messages, max_messages) if max_messages > 0 else messages
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda m=model, msgs=to_send, t=ollama_tools: _ollama_chat_sync(m, msgs, t),
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
            return (content or "").strip()

        # 3. Execute each tool call via MCP
        for tc in tool_calls:
            fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
            if fn is None:
                continue
            name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
            if not name:
                continue
            raw_args = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", None)
            if isinstance(raw_args, str):
                try:
                    arguments = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    arguments = {}
            else:
                arguments = raw_args or {}

            result = await call_tool(session, name, arguments)
            content = tool_result_to_content(result)
            messages.append({
                "role": "tool",
                "tool_name": name,
                "content": content,
            })

    return "(Max tool rounds reached; stopping.)"


def _stream_into_queue(
    q: queue.Queue,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> None:
    """Run Ollama chat with stream=True and put (content_fragment, done, message) into q. Puts None at end."""
    try:
        stream = ollama.chat(model=model, messages=messages, tools=tools, stream=True)
        for chunk in stream:
            msg = chunk.get("message") if isinstance(chunk, dict) else getattr(chunk, "message", None)
            content = _get(msg, "content", "") or "" if msg else ""
            done = chunk.get("done") if isinstance(chunk, dict) else getattr(chunk, "done", False)
            q.put((content, done, msg))
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
) -> AsyncIterator[str]:
    """
    Like run_agent_loop but streams content tokens as they arrive.
    Yields content fragments (str). When tool_calls occur, runs them and continues streaming the next turn.
    """
    list_result = await list_tools(session)
    ollama_tools = mcp_tools_to_ollama(list_result.tools) or []

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    loop = asyncio.get_event_loop()
    for _ in range(max_tool_rounds):
        q: queue.Queue = queue.Queue()
        to_send = _truncate_messages(messages, max_messages) if max_messages > 0 else messages
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
            content_frag, done, msg = item
            if content_frag:
                yield content_frag
            last_msg = msg
        thread.join()

        if last_msg is None:
            return
        content = _get(last_msg, "content") or ""
        tool_calls = _get(last_msg, "tool_calls") or []
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            return

        for tc in tool_calls:
            fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
            if fn is None:
                continue
            name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
            if not name:
                continue
            raw_args = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", None)
            if isinstance(raw_args, str):
                try:
                    arguments = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    arguments = {}
            else:
                arguments = raw_args or {}
            result = await call_tool(session, name, arguments)
            content = tool_result_to_content(result)
            messages.append({"role": "tool", "tool_name": name, "content": content})


async def run_agent_loop_no_mcp(
    model: str,
    user_message: str,
    *,
    system_prompt: str | None = None,
) -> str:
    """Run one turn with Ollama only (no MCP tools). Returns assistant text."""
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
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
) -> AsyncIterator[str]:
    """Stream one turn with Ollama only (no MCP). Yields content fragments."""
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
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
