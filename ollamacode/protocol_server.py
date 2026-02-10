"""
Stdio JSON-RPC server for the editor protocol. Run with: ollamacode protocol

Reads JSON-RPC 2.0 requests from stdin (one request per line), dispatches
ollamacode/chat, ollamacode/chatStream (streaming), and ollamacode/applyEdits.
Writes one or more JSON-RPC response lines to stdout (streaming = multiple lines per request).
See docs/STRUCTURED_PROTOCOL.md.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator
from typing import Any

from .agent import (
    run_agent_loop,
    run_agent_loop_no_mcp,
    run_agent_loop_no_mcp_stream,
    run_agent_loop_stream,
)
from .context import prepend_file_context
from .edits import apply_edits, parse_edits
from .mcp_client import McpConnection
from .completions import get_completion
from .diagnostics import get_diagnostics
from .protocol import normalize_chat_body
from .skills import load_skills_text
from .state import get_state, format_recent_context
from .templates import load_prompt_template


def _system_prompt(
    system_extra: str,
    workspace_root: str | None = None,
    use_skills: bool = True,
    prompt_template: str | None = None,
    inject_recent_context: bool = True,
    recent_context_max_files: int = 10,
) -> str:
    base = (
        "You are a coding assistant with full access to the workspace. You are given a list of available tools with their names "
        "and descriptions—use whichever tools fit the task. When the user asks you to run something, check something, or change "
        "something, use the appropriate tool and report the result."
    )
    out = base + ("\n\n" + system_extra) if system_extra else base
    if use_skills and workspace_root:
        skills_text = load_skills_text(workspace_root)
        if skills_text:
            out = (
                out
                + "\n\n--- Skills (saved instructions & memory) ---\n\n"
                + skills_text
            )
    if prompt_template and workspace_root:
        template_text = load_prompt_template(prompt_template, workspace_root)
        if template_text:
            out = out + "\n\n--- Prompt template ---\n\n" + template_text
    if inject_recent_context:
        state = get_state()
        block = format_recent_context(state, max_files=recent_context_max_files)
        if block:
            out = out + "\n\n--- Recent context ---\n\n" + block
    return out


async def _handle_chat(
    session: McpConnection | None,
    model: str,
    system_extra: str,
    params: dict[str, Any],
    max_messages: int,
    max_tool_result_chars: int,
    workspace_root: str,
    use_skills: bool = True,
    prompt_template: str | None = None,
    inject_recent_context: bool = True,
    recent_context_max_files: int = 10,
) -> dict[str, Any]:
    """Handle chat params; return { content, edits?, error? }."""
    message, file_path, lines_spec = normalize_chat_body(params)
    if not message:
        return {"content": "", "error": "message required"}
    if file_path:
        message = prepend_file_context(
            message, str(file_path), workspace_root, lines_spec
        )
    use_model = params.get("model") or model
    system = _system_prompt(
        system_extra,
        workspace_root,
        use_skills,
        prompt_template,
        inject_recent_context=inject_recent_context,
        recent_context_max_files=recent_context_max_files,
    )
    try:
        if session is not None:
            out = await run_agent_loop(
                session,
                use_model,
                message,
                system_prompt=system,
                max_messages=max_messages,
                max_tool_result_chars=max_tool_result_chars,
            )
        else:
            out = await run_agent_loop_no_mcp(use_model, message, system_prompt=system)
        result: dict[str, Any] = {"content": out}
        edits_list = parse_edits(out)
        if edits_list:
            result["edits"] = edits_list
        return result
    except Exception as e:
        return {"content": "", "error": str(e)}


async def _handle_chat_stream(
    session: McpConnection | None,
    model: str,
    system_extra: str,
    params: dict[str, Any],
    max_messages: int,
    max_tool_result_chars: int,
    workspace_root: str,
    req_id: Any,
    use_skills: bool = True,
    prompt_template: str | None = None,
    inject_recent_context: bool = True,
    recent_context_max_files: int = 10,
) -> AsyncIterator[dict[str, Any]]:
    """Stream chat; yield JSON-RPC response objects: N× { result: { type: 'chunk', content } }, then { result: { type: 'done', content, edits? } } or error."""
    message, file_path, lines_spec = normalize_chat_body(params)
    if not message:
        yield {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"type": "error", "error": "message required"},
        }
        return
    if file_path:
        message = prepend_file_context(
            message, str(file_path), workspace_root, lines_spec
        )
    use_model = params.get("model") or model
    system = _system_prompt(
        system_extra,
        workspace_root,
        use_skills,
        prompt_template,
        inject_recent_context=inject_recent_context,
        recent_context_max_files=recent_context_max_files,
    )
    try:
        if session is not None:
            stream = run_agent_loop_stream(
                session,
                use_model,
                message,
                system_prompt=system,
                max_messages=max_messages,
                max_tool_result_chars=max_tool_result_chars,
            )
        else:
            stream = run_agent_loop_no_mcp_stream(
                use_model,
                message,
                system_prompt=system,
                message_history=[],
            )
        accumulated: list[str] = []
        async for chunk in stream:
            accumulated.append(chunk)
            yield {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"type": "chunk", "content": chunk},
            }
        full = "".join(accumulated)
        edits_list = parse_edits(full)
        result: dict[str, Any] = {"type": "done", "content": full}
        if edits_list:
            result["edits"] = edits_list
        yield {"jsonrpc": "2.0", "id": req_id, "result": result}
    except Exception as e:
        yield {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"type": "error", "error": str(e)},
        }


def _handle_apply_edits(params: dict[str, Any], workspace_root: str) -> dict[str, Any]:
    """Handle applyEdits params; return { applied } or { applied, error }."""
    edits_raw = params.get("edits")
    if not isinstance(edits_raw, list):
        return {"applied": 0, "error": "edits required (array)"}
    root = workspace_root
    override = params.get("workspaceRoot")
    if isinstance(override, str) and override.strip():
        root = os.path.abspath(override)
    edits: list[dict[str, Any]] = []
    for item in edits_raw:
        if not isinstance(item, dict):
            continue
        path_val = item.get("path")
        new_text = item.get("newText")
        if path_val is None or new_text is None:
            continue
        edits.append(
            {
                "path": str(path_val),
                "oldText": item.get("oldText"),
                "newText": new_text if isinstance(new_text, str) else str(new_text),
            }
        )
    if not edits:
        return {"applied": 0, "error": "no valid edits (path and newText required)"}
    try:
        n = apply_edits(edits, root)
        return {"applied": n}
    except Exception as e:
        return {"applied": 0, "error": str(e)}


async def _handle_request(
    request: dict[str, Any],
    session: McpConnection | None,
    model: str,
    system_extra: str,
    max_messages: int,
    max_tool_result_chars: int,
    workspace_root: str,
    use_skills: bool = True,
    prompt_template: str | None = None,
    inject_recent_context: bool = True,
    recent_context_max_files: int = 10,
) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
    """Dispatch JSON-RPC request; return one response dict or an async iterator of response dicts (for chatStream)."""
    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params")
    if not isinstance(params, dict):
        params = {}

    if method == "ollamacode/chat":
        result = await _handle_chat(
            session,
            model,
            system_extra,
            params,
            max_messages,
            max_tool_result_chars,
            workspace_root,
            use_skills=use_skills,
            prompt_template=prompt_template,
            inject_recent_context=inject_recent_context,
            recent_context_max_files=recent_context_max_files,
        )
        return {"jsonrpc": "2.0", "id": req_id, "result": result}
    if method == "ollamacode/chatStream":
        return _handle_chat_stream(
            session,
            model,
            system_extra,
            params,
            max_messages,
            max_tool_result_chars,
            workspace_root,
            req_id,
            use_skills=use_skills,
            prompt_template=prompt_template,
            inject_recent_context=inject_recent_context,
            recent_context_max_files=recent_context_max_files,
        )
    if method == "ollamacode/applyEdits":
        result = _handle_apply_edits(params, workspace_root)
        return {"jsonrpc": "2.0", "id": req_id, "result": result}
    if method == "ollamacode/diagnostics":
        root = (params.get("workspaceRoot") or workspace_root).strip() or workspace_root
        path = params.get("path")
        linter = (
            params.get("linterCommand") or "ruff check ."
        ).strip() or "ruff check ."
        diag = get_diagnostics(root, path=path, linter_command=linter)
        return {"jsonrpc": "2.0", "id": req_id, "result": {"diagnostics": diag}}
    if method == "ollamacode/complete":
        prefix = (params.get("prefix") or "").strip()
        use_model = params.get("model") or model
        loop = asyncio.get_event_loop()
        completion = await loop.run_in_executor(
            None, lambda: get_completion(prefix, use_model)
        )
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"completions": [completion] if completion else []},
        }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method!r}"},
    }


async def run_protocol_stdio(
    session: McpConnection | None,
    model: str,
    system_extra: str,
    max_messages: int = 0,
    max_tool_result_chars: int = 0,
    workspace_root: str | None = None,
    use_skills: bool = True,
    prompt_template: str | None = None,
    inject_recent_context: bool = True,
    recent_context_max_files: int = 10,
) -> None:
    """
    Run the JSON-RPC protocol over stdio. Reads one JSON-RPC request per line from stdin,
    writes one or more JSON-RPC response lines to stdout. Methods: ollamacode/chat,
    ollamacode/chatStream (streaming), ollamacode/applyEdits.
    """
    root = workspace_root or os.getcwd()
    loop = asyncio.get_event_loop()

    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        line = line.rstrip("\n\r")
        if not line.strip():
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            out = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {e}"},
            }
            print(json.dumps(out), flush=True)
            continue
        if not isinstance(req, dict):
            out = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Invalid request"},
            }
            print(json.dumps(out), flush=True)
            continue
        try:
            response = await _handle_request(
                req,
                session,
                model,
                system_extra,
                max_messages,
                max_tool_result_chars,
                root,
                use_skills=use_skills,
                prompt_template=prompt_template,
                inject_recent_context=inject_recent_context,
                recent_context_max_files=recent_context_max_files,
            )
            if hasattr(response, "__aiter__") and not isinstance(response, dict):
                async for part in response:
                    print(json.dumps(part), flush=True)
            else:
                print(json.dumps(response), flush=True)
        except Exception as e:
            out = {
                "jsonrpc": "2.0",
                "id": req.get("id"),
                "error": {"code": -32603, "message": str(e)},
            }
            print(json.dumps(out), flush=True)
