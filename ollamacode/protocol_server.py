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
import uuid
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
from .memory import build_dynamic_memory_context
from .multi_agent import run_multi_agent
from .mcp_client import McpConnection
from .completions import get_completion
from .diagnostics import get_diagnostics
from .protocol import normalize_chat_body
from .rag import build_local_rag_index, query_local_rag
from .skills import load_skills_text
from .state import (
    format_feedback_context,
    format_knowledge_context,
    format_past_errors_context,
    format_plan_context,
    format_preferences,
    format_recent_context,
    get_state,
)
from .templates import load_prompt_template

# In-memory store for tool-approval continuation (stdio: one client at a time)
_protocol_approval_pending: dict[str, dict[str, Any]] = {}


def _coerce_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(n, max_value))


def _resolve_memory_request_settings(
    params: dict[str, Any],
    *,
    default_auto: bool,
    default_kg_max: int,
    default_rag_max: int,
    default_rag_chars: int,
) -> tuple[bool, int, int, int]:
    auto_raw = params.get("memoryAutoContext", params.get("memory_auto_context"))
    kg_raw = params.get("memoryKgMaxResults", params.get("memory_kg_max_results"))
    rag_raw = params.get("memoryRagMaxResults", params.get("memory_rag_max_results"))
    chars_raw = params.get(
        "memoryRagSnippetChars", params.get("memory_rag_snippet_chars")
    )
    auto = bool(auto_raw) if isinstance(auto_raw, bool) else default_auto
    kg_max = _coerce_int(kg_raw, default_kg_max, 0, 20)
    rag_max = _coerce_int(rag_raw, default_rag_max, 0, 20)
    rag_chars = _coerce_int(chars_raw, default_rag_chars, 40, 2000)
    return auto, kg_max, rag_max, rag_chars


def _append_dynamic_memory(
    system: str,
    query: str,
    *,
    memory_auto_context: bool = True,
    memory_kg_max_results: int = 4,
    memory_rag_max_results: int = 4,
    memory_rag_snippet_chars: int = 220,
) -> str:
    """Append per-query retrieved memory context to the system prompt."""
    if not memory_auto_context:
        return system
    block = build_dynamic_memory_context(
        query,
        kg_max_results=memory_kg_max_results,
        rag_max_results=memory_rag_max_results,
        rag_snippet_chars=memory_rag_snippet_chars,
    )
    if not block:
        return system
    return system + "\n\n--- Retrieved memory (query-specific) ---\n\n" + block


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
    base = (
        "You are a coding assistant with full access to the workspace. You are given a list of available tools with their names "
        "and descriptions—use whichever tools fit the task. When the user asks you to run something, check something, or change "
        "something, use the appropriate tool and report the result. When generating code, include docstrings and brief comments where helpful."
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
    state = get_state()
    if inject_recent_context:
        block = format_recent_context(state, max_files=recent_context_max_files)
        if block:
            out = out + "\n\n--- Recent context ---\n\n" + block
    prefs_block = format_preferences(state)
    if prefs_block:
        out = out + "\n\n--- User preferences ---\n\n" + prefs_block
    plan_block = format_plan_context(state)
    if plan_block:
        out = out + "\n\n--- Plan (use /continue to work on it) ---\n\n" + plan_block
    feedback_block = format_feedback_context(state)
    if feedback_block:
        out = out + "\n\n--- Recent feedback ---\n\n" + feedback_block
    knowledge_block = format_knowledge_context(state)
    if knowledge_block:
        out = out + "\n\n--- " + knowledge_block
    past_errors_block = format_past_errors_context(state, max_entries=5)
    if past_errors_block:
        out = out + "\n\n--- " + past_errors_block
    if use_reasoning:
        out = (
            out
            + "\n\nWhen answering, you may include a brief reasoning or rationale before your conclusion; for code changes, briefly explain the fix."
            + '\n\nOptionally output <<REASONING>>\n{"steps": ["..."], "conclusion": "..."}\n<<END>> or call record_reasoning(steps, conclusion).'
        )
    for snip in prompt_snippets or []:
        if snip and isinstance(snip, str) and snip.strip():
            out = out + "\n\n" + snip.strip()
    if code_style:
        out = (
            out
            + "\n\n--- Code style (follow when generating code) ---\n\n"
            + code_style.strip()
        )
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
        if not token or token not in _protocol_approval_pending:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32602,
                    "message": "invalid or expired approvalToken",
                },
            }
        decision = params.get("decision", "run")
        edited = params.get("editedArguments") if decision == "edit" else None
        if decision == "edit" and isinstance(edited, dict):
            decision_value: Any = ("edit", edited)
        elif decision == "skip":
            decision_value = "skip"
        else:
            decision_value = "run"
        entry = _protocol_approval_pending.pop(token)
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

        async def before_tool_call(name: str, arguments: dict):
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
                result = task.result()
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": "", "error": str(e)},
                }
            out = {
                "content": result.content,
                "plan": result.plan,
                "review": result.review,
            }
            edits_list = parse_edits(result.content)
            if edits_list:
                out["edits"] = edits_list
            return {"jsonrpc": "2.0", "id": req_id, "result": out}
        token = uuid.uuid4().hex
        _protocol_approval_pending[token] = {
            "task": task,
            "event": event,
            "pending": pending,
            "mode": "multi",
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

        async def before_tool_call(name: str, arguments: dict):
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
            result = {"content": out}
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
        result = await run_multi_agent(
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
        out = {"content": result.content, "plan": result.plan, "review": result.review}
        edits_list = parse_edits(result.content)
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
        )
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
