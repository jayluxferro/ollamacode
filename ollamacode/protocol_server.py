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
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from .agent import (
    run_agent_loop,
    run_agent_loop_no_mcp,
    run_agent_loop_no_mcp_stream,
    run_agent_loop_stream,
)
from ._chat_helpers import (
    append_dynamic_memory as _append_dynamic_memory,
    build_system_prompt as _build_system_prompt,
    resolve_memory_request_settings as _resolve_memory_request_settings,
)
from .context import prepend_file_context
from .edits import apply_edits, parse_edits
from .multi_agent import run_multi_agent
from .mcp_client import McpConnection
from .completions import get_completion
from .diagnostics import get_diagnostics
from .protocol import normalize_chat_body
from .rag import build_local_rag_index, query_local_rag

# In-memory store for tool-approval continuation (stdio: one client at a time)
_protocol_approval_pending: dict[str, dict[str, Any]] = {}
_PROTOCOL_APPROVAL_TTL_SECONDS = 300  # Tokens expire after 5 minutes


def _cleanup_stale_protocol_approvals() -> None:
    """Remove approval tokens older than _PROTOCOL_APPROVAL_TTL_SECONDS."""
    now = time.monotonic()
    expired = [
        k
        for k, v in _protocol_approval_pending.items()
        if now - v.get("created_at", 0) > _PROTOCOL_APPROVAL_TTL_SECONDS
    ]
    for k in expired:
        _protocol_approval_pending.pop(k, None)


def _set_apple_fm_session_key(key: str) -> None:
    if not key:
        return
    os.environ.setdefault("OLLAMACODE_APPLE_FM_STATEFUL", "1")
    os.environ["OLLAMACODE_APPLE_FM_SESSION_KEY"] = key


def _system_prompt(
    system_extra: str,
    workspace_root: str | None = None,
    use_skills: bool = True,
    prompt_template: str | None = None,
    inject_recent_context: bool = True,
    recent_context_max_files: int = 10,
    use_reasoning: bool = False,
    prompt_snippets: list[str] | None = None,
    code_style: str | None = None,
) -> str:
    return _build_system_prompt(
        system_extra,
        workspace_root=workspace_root,
        use_skills=use_skills,
        prompt_template=prompt_template,
        inject_recent_context=inject_recent_context,
        recent_context_max_files=recent_context_max_files,
        use_reasoning=use_reasoning,
        prompt_snippets=prompt_snippets,
        code_style=code_style,
    )


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
    use_reasoning: bool = False,
    prompt_snippets: list[str] | None = None,
    allowed_tools: list[str] | None = None,
    blocked_tools: list[str] | None = None,
    confirm_tool_calls: bool = False,
    code_style: str | None = None,
    memory_auto_context: bool = True,
    memory_kg_max_results: int = 4,
    memory_rag_max_results: int = 4,
    memory_rag_snippet_chars: int = 220,
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
        use_reasoning=use_reasoning,
        prompt_snippets=prompt_snippets,
        code_style=code_style,
    )
    system = _append_dynamic_memory(
        system,
        message,
        memory_auto_context=memory_auto_context,
        memory_kg_max_results=memory_kg_max_results,
        memory_rag_max_results=memory_rag_max_results,
        memory_rag_snippet_chars=memory_rag_snippet_chars,
    )
    tool_errors: list[dict[str, Any]] = []
    try:
        if session is not None:
            out = await run_agent_loop(
                session,
                use_model,
                message,
                system_prompt=system,
                max_messages=max_messages,
                max_tool_result_chars=max_tool_result_chars,
                allowed_tools=allowed_tools,
                blocked_tools=blocked_tools,
                tool_errors_out=tool_errors,
                confirm_tool_calls=confirm_tool_calls,
            )
        else:
            out = await run_agent_loop_no_mcp(use_model, message, system_prompt=system)
        result: dict[str, Any] = {"content": out}
        edits_list = parse_edits(out)
        if edits_list:
            result["edits"] = edits_list
        if tool_errors:
            result["tool_errors"] = tool_errors
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
    use_reasoning: bool = False,
    prompt_snippets: list[str] | None = None,
    allowed_tools: list[str] | None = None,
    blocked_tools: list[str] | None = None,
    confirm_tool_calls: bool = False,
    code_style: str | None = None,
    memory_auto_context: bool = True,
    memory_kg_max_results: int = 4,
    memory_rag_max_results: int = 4,
    memory_rag_snippet_chars: int = 220,
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
        use_reasoning=use_reasoning,
        prompt_snippets=prompt_snippets,
        code_style=code_style,
    )
    system = _append_dynamic_memory(
        system,
        message,
        memory_auto_context=memory_auto_context,
        memory_kg_max_results=memory_kg_max_results,
        memory_rag_max_results=memory_rag_max_results,
        memory_rag_snippet_chars=memory_rag_snippet_chars,
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
                allowed_tools=allowed_tools,
                blocked_tools=blocked_tools,
                confirm_tool_calls=confirm_tool_calls,
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
    use_reasoning: bool = False,
    prompt_snippets: list[str] | None = None,
    allowed_tools: list[str] | None = None,
    blocked_tools: list[str] | None = None,
    confirm_tool_calls: bool = False,
    code_style: str | None = None,
    memory_auto_context: bool = True,
    memory_kg_max_results: int = 4,
    memory_rag_max_results: int = 4,
    memory_rag_snippet_chars: int = 220,
) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
    """Dispatch JSON-RPC request; return one response dict or an async iterator of response dicts (for chatStream)."""
    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params")
    if not isinstance(params, dict):
        params = {}
    _set_apple_fm_session_key(str(req_id) if req_id is not None else uuid.uuid4().hex)
    root = workspace_root
    req_mem_auto, req_mem_kg, req_mem_rag, req_mem_chars = (
        _resolve_memory_request_settings(
            params,
            default_auto=memory_auto_context,
            default_kg_max=memory_kg_max_results,
            default_rag_max=memory_rag_max_results,
            default_rag_chars=memory_rag_snippet_chars,
        )
    )

    if method == "ollamacode/chatContinue":
        token = params.get("approvalToken")
        _cleanup_stale_protocol_approvals()
        if not token or token not in _protocol_approval_pending:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32602,
                    "message": "invalid or expired approvalToken",
                },
            }
        entry = _protocol_approval_pending.pop(token)
        decision = params.get("decision", "run")
        edited = params.get("editedArguments") if decision == "edit" else None
        if decision == "edit" and isinstance(edited, dict):
            decision_value: Any = ("edit", edited)
        elif decision == "skip":
            decision_value = "skip"
        else:
            decision_value = "run"
        task = entry["task"]
        event = entry["event"]
        pending = entry["pending"]
        mode = entry.get("mode", "chat")
        pending["future"].set_result(decision_value)
        event.clear()
        tool_errors = entry.get("tool_errors", [])
        while True:
            done, _ = await asyncio.wait(
                [task, asyncio.create_task(event.wait())],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if task in done:
                try:
                    out = task.result()
                except Exception as e:
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {"content": "", "error": str(e)},
                    }
                if mode == "multi":
                    result = {
                        "content": out.content,
                        "plan": out.plan,
                        "review": out.review,
                    }
                    edits_list = parse_edits(out.content)
                    if edits_list:
                        result["edits"] = edits_list
                    return {"jsonrpc": "2.0", "id": req_id, "result": result}
                result = {"content": out}
                edits_list = parse_edits(out)
                if edits_list:
                    result["edits"] = edits_list
                if tool_errors:
                    result["tool_errors"] = tool_errors
                return {"jsonrpc": "2.0", "id": req_id, "result": result}
            new_token = uuid.uuid4().hex
            _protocol_approval_pending[new_token] = {
                "task": task,
                "event": event,
                "pending": pending,
                "tool_errors": tool_errors,
                "mode": mode,
                "created_at": time.monotonic(),
            }
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "toolApprovalRequired": {
                        "tool": pending["tool"],
                        "arguments": pending["arguments"],
                    },
                    "approvalToken": new_token,
                },
            }

    if (
        method == "ollamacode/chat"
        and params.get("multiAgent")
        and confirm_tool_calls
        and session is not None
    ):
        message, file_path, lines_spec = normalize_chat_body(params)
        if not message:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": "", "error": "message required"},
            }
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
            use_reasoning=use_reasoning,
            prompt_snippets=prompt_snippets,
            code_style=code_style,
        )
        system = _append_dynamic_memory(
            system,
            message,
            memory_auto_context=req_mem_auto,
            memory_kg_max_results=req_mem_kg,
            memory_rag_max_results=req_mem_rag,
            memory_rag_snippet_chars=req_mem_chars,
        )
        event = asyncio.Event()
        pending: dict[str, Any] = {}
        loop = asyncio.get_event_loop()
        from .hooks import HookManager

        hook_mgr = HookManager(root, None)

        async def before_tool_call(name: str, arguments: dict):
            try:
                decision = await hook_mgr.run_pre_tool_use(
                    name, arguments, user_prompt=message
                )
                if decision and decision.behavior == "deny":
                    return ("skip", decision.message or "Blocked by hook.")
                if (
                    decision
                    and decision.behavior == "modify"
                    and decision.updated_input
                ):
                    arguments = decision.updated_input
                    return ("edit", arguments)
                if decision and decision.behavior == "allow":
                    return "run"
            except Exception:
                pass
            pending["tool"] = name
            pending["arguments"] = arguments
            pending["future"] = loop.create_future()
            event.set()
            return await pending["future"]

        task = asyncio.create_task(
            run_multi_agent(
                session,
                use_model,
                message,
                system_prompt=system,
                max_messages=max_messages,
                max_tool_result_chars=max_tool_result_chars,
                allowed_tools=allowed_tools,
                blocked_tools=blocked_tools,
                confirm_tool_calls=True,
                before_tool_call=before_tool_call,
                planner_model=params.get("plannerModel"),
                executor_model=params.get("executorModel"),
                reviewer_model=params.get("reviewerModel"),
                max_iterations=int(params.get("multiAgentMaxIterations") or 2),
                require_review=bool(params.get("multiAgentRequireReview", True)),
            )
        )
        done, _ = await asyncio.wait(
            [task, asyncio.create_task(event.wait())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if task in done:
            try:
                multi_result = task.result()
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": "", "error": str(e)},
                }
            out = {
                "content": multi_result.content,
                "plan": multi_result.plan,
                "review": multi_result.review,
            }
            edits_list = parse_edits(multi_result.content)
            if edits_list:
                out["edits"] = edits_list
            return {"jsonrpc": "2.0", "id": req_id, "result": out}
        token = uuid.uuid4().hex
        _protocol_approval_pending[token] = {
            "task": task,
            "event": event,
            "pending": pending,
            "mode": "multi",
            "created_at": time.monotonic(),
        }
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "toolApprovalRequired": {
                    "tool": pending["tool"],
                    "arguments": pending["arguments"],
                },
                "approvalToken": token,
            },
        }

    if method == "ollamacode/chat" and confirm_tool_calls and session is not None:
        message, file_path, lines_spec = normalize_chat_body(params)
        if not message:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": "", "error": "message required"},
            }
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
            use_reasoning=use_reasoning,
            prompt_snippets=prompt_snippets,
            code_style=code_style,
        )
        system = _append_dynamic_memory(
            system,
            message,
            memory_auto_context=req_mem_auto,
            memory_kg_max_results=req_mem_kg,
            memory_rag_max_results=req_mem_rag,
            memory_rag_snippet_chars=req_mem_chars,
        )
        event = asyncio.Event()
        pending: dict[str, Any] = {}
        loop = asyncio.get_event_loop()
        from .hooks import HookManager

        hook_mgr = HookManager(root, None)

        async def before_tool_call(name: str, arguments: dict):
            try:
                decision = await hook_mgr.run_pre_tool_use(
                    name, arguments, user_prompt=message
                )
                if decision and decision.behavior == "deny":
                    return ("skip", decision.message or "Blocked by hook.")
                if (
                    decision
                    and decision.behavior == "modify"
                    and decision.updated_input
                ):
                    arguments = decision.updated_input
                    return ("edit", arguments)
                if decision and decision.behavior == "allow":
                    return "run"
            except Exception:
                pass
            pending["tool"] = name
            pending["arguments"] = arguments
            pending["future"] = loop.create_future()
            event.set()
            return await pending["future"]

        tool_errors: list[dict[str, Any]] = []
        task = asyncio.create_task(
            run_agent_loop(
                session,
                use_model,
                message,
                system_prompt=system,
                max_messages=max_messages,
                max_tool_result_chars=max_tool_result_chars,
                allowed_tools=allowed_tools,
                blocked_tools=blocked_tools,
                tool_errors_out=tool_errors,
                confirm_tool_calls=True,
                before_tool_call=before_tool_call,
            )
        )
        done, _ = await asyncio.wait(
            [task, asyncio.create_task(event.wait())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if task in done:
            try:
                out = task.result()
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": "", "error": str(e)},
                }
            result: dict[str, Any] = {"content": out}
            edits_list = parse_edits(out)
            if edits_list:
                result["edits"] = edits_list
            if tool_errors:
                result["tool_errors"] = tool_errors
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        token = uuid.uuid4().hex
        _protocol_approval_pending[token] = {
            "task": task,
            "event": event,
            "pending": pending,
            "tool_errors": tool_errors,
            "created_at": time.monotonic(),
        }
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "toolApprovalRequired": {
                    "tool": pending["tool"],
                    "arguments": pending["arguments"],
                },
                "approvalToken": token,
            },
        }

    if method == "ollamacode/chat" and params.get("multiAgent"):
        message, file_path, lines_spec = normalize_chat_body(params)
        if not message:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": "", "error": "message required"},
            }
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
            use_reasoning=use_reasoning,
            prompt_snippets=prompt_snippets,
            code_style=code_style,
        )
        system = _append_dynamic_memory(
            system,
            message,
            memory_auto_context=req_mem_auto,
            memory_kg_max_results=req_mem_kg,
            memory_rag_max_results=req_mem_rag,
            memory_rag_snippet_chars=req_mem_chars,
        )
        multi_result = await run_multi_agent(
            session,
            use_model,
            message,
            system_prompt=system,
            max_messages=max_messages,
            max_tool_result_chars=max_tool_result_chars,
            allowed_tools=allowed_tools,
            blocked_tools=blocked_tools,
            confirm_tool_calls=confirm_tool_calls,
            before_tool_call=None,
            planner_model=params.get("plannerModel"),
            executor_model=params.get("executorModel"),
            reviewer_model=params.get("reviewerModel"),
            max_iterations=int(params.get("multiAgentMaxIterations") or 2),
            require_review=bool(params.get("multiAgentRequireReview", True)),
        )
        out = {
            "content": multi_result.content,
            "plan": multi_result.plan,
            "review": multi_result.review,
        }
        edits_list = parse_edits(multi_result.content)
        if edits_list:
            out["edits"] = edits_list
        return {"jsonrpc": "2.0", "id": req_id, "result": out}

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
            use_reasoning=use_reasoning,
            prompt_snippets=prompt_snippets,
            allowed_tools=allowed_tools,
            blocked_tools=blocked_tools,
            confirm_tool_calls=confirm_tool_calls,
            code_style=code_style,
            memory_auto_context=req_mem_auto,
            memory_kg_max_results=req_mem_kg,
            memory_rag_max_results=req_mem_rag,
            memory_rag_snippet_chars=req_mem_chars,
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
            use_reasoning=use_reasoning,
            prompt_snippets=prompt_snippets,
            allowed_tools=allowed_tools,
            blocked_tools=blocked_tools,
            confirm_tool_calls=confirm_tool_calls,
            code_style=code_style,
            memory_auto_context=req_mem_auto,
            memory_kg_max_results=req_mem_kg,
            memory_rag_max_results=req_mem_rag,
            memory_rag_snippet_chars=req_mem_chars,
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
    if method == "ollamacode/ragIndex":
        target_root = (
            params.get("workspaceRoot")
            if isinstance(params.get("workspaceRoot"), str)
            else workspace_root
        ) or "."
        max_files = params.get("maxFiles")
        max_chars_per_file = params.get("maxCharsPerFile")
        try:
            info = build_local_rag_index(
                target_root,
                max_files=int(max_files) if isinstance(max_files, int) else 400,
                max_chars_per_file=int(max_chars_per_file)
                if isinstance(max_chars_per_file, int)
                else 20000,
            )
            return {"jsonrpc": "2.0", "id": req_id, "result": info}
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"error": str(e)},
            }
    if method == "ollamacode/ragQuery":
        query = (
            (params.get("query") or "").strip()
            if isinstance(params.get("query"), str)
            else ""
        )
        if not query:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"results": [], "error": "query required"},
            }
        max_results = params.get("maxResults")
        try:
            rows = query_local_rag(
                query,
                max_results=int(max_results) if isinstance(max_results, int) else 5,
            )
            return {"jsonrpc": "2.0", "id": req_id, "result": {"results": rows}}
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"results": [], "error": str(e)},
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
    use_reasoning: bool = False,
    prompt_snippets: list[str] | None = None,
    allowed_tools: list[str] | None = None,
    blocked_tools: list[str] | None = None,
    confirm_tool_calls: bool = False,
    code_style: str | None = None,
    memory_auto_context: bool = True,
    memory_kg_max_results: int = 4,
    memory_rag_max_results: int = 4,
    memory_rag_snippet_chars: int = 220,
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
                use_reasoning=use_reasoning,
                prompt_snippets=prompt_snippets,
                allowed_tools=allowed_tools,
                blocked_tools=blocked_tools,
                confirm_tool_calls=confirm_tool_calls,
                code_style=code_style,
                memory_auto_context=memory_auto_context,
                memory_kg_max_results=memory_kg_max_results,
                memory_rag_max_results=memory_rag_max_results,
                memory_rag_snippet_chars=memory_rag_snippet_chars,
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
