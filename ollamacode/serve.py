"""
Optional local HTTP API for OllamaCode. Run: ollamacode serve --port 8000
Requires: pip install ollamacode[server]
Endpoints: POST /chat, POST /chat/stream (see docs/STRUCTURED_PROTOCOL.md), POST /apply-edits.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from starlette.responses import JSONResponse

from .agent import (
    run_agent_loop,
    run_agent_loop_no_mcp,
    run_agent_loop_no_mcp_stream,
    run_agent_loop_stream,
)
from ._chat_helpers import (
    append_dynamic_memory as _append_dynamic_memory,
    build_system_prompt as _build_system_prompt_shared,
    resolve_memory_request_settings as _resolve_memory_request_settings,
)
from .config import get_env_config_overrides, load_config, merge_config_with_env
from .context import prepend_file_context
from .edits import apply_edits, parse_edits
from .multi_agent import run_multi_agent
from .mcp_client import McpConnection, connect_mcp_servers, connect_mcp_stdio
from .completions import get_completion
from .diagnostics import get_diagnostics
from .health import check_ollama
from .protocol import normalize_chat_body
from .rag import build_local_rag_index, query_local_rag

# In-memory store for tool-approval continuation: approval_token -> { task, event, pending, created_at }
_approval_pending: dict[str, dict[str, Any]] = {}
_approval_lock = asyncio.Lock()
_APPROVAL_TTL_SECONDS = 300  # Tokens expire after 5 minutes


async def _cleanup_stale_approvals() -> None:
    """Remove approval tokens older than _APPROVAL_TTL_SECONDS."""
    now = time.monotonic()
    async with _approval_lock:
        expired = [
            k
            for k, v in _approval_pending.items()
            if now - v.get("created_at", 0) > _APPROVAL_TTL_SECONDS
        ]
        for k in expired:
            _approval_pending.pop(k, None)


def _verify_webhook_hmac(body_bytes: bytes, signature_header: str, secret: str) -> bool:
    """Verify an HMAC-SHA256 webhook signature.

    Expects ``signature_header`` in the form ``sha256=<hex_digest>`` (GitHub style).
    """
    import hashlib
    import hmac

    if not signature_header.startswith("sha256="):
        return False
    expected = (
        "sha256="
        + hmac.new(secret.encode("utf-8"), body_bytes, hashlib.sha256).hexdigest()
    )
    return hmac.compare_digest(expected, signature_header)


def _set_apple_fm_session_key(key: str) -> None:
    if not key:
        return
    os.environ.setdefault("OLLAMACODE_APPLE_FM_STATEFUL", "1")
    os.environ["OLLAMACODE_APPLE_FM_SESSION_KEY"] = key


def _build_chat_system(
    body: dict,
    model: str,
    workspace_root: str,
    system_extra: str,
    use_skills: bool = True,
    prompt_template: str | None = None,
    inject_recent_context: bool = True,
    recent_context_max_files: int = 10,
    use_reasoning: bool = False,
    prompt_snippets: list[str] | None = None,
    code_style: str | None = None,
    memory_auto_context: bool = True,
    memory_kg_max_results: int = 4,
    memory_rag_max_results: int = 4,
    memory_rag_snippet_chars: int = 220,
) -> tuple[str | None, str, str]:
    """Build (message, use_model, system) for chat. Returns (None, model, '') if message required."""
    message, file_path, lines_spec = normalize_chat_body(body)
    if not message:
        return (None, model, "")
    if file_path:
        message = prepend_file_context(
            message, str(file_path), workspace_root, lines_spec
        )
    use_model = body.get("model") or model
    system = _build_system_prompt_shared(
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
    system = _append_dynamic_memory(
        system,
        message,
        memory_auto_context=memory_auto_context,
        memory_kg_max_results=memory_kg_max_results,
        memory_rag_max_results=memory_rag_max_results,
        memory_rag_snippet_chars=memory_rag_snippet_chars,
    )
    return (message, use_model, system)


async def _handle_chat(
    session: McpConnection | None,
    model: str,
    system_extra: str,
    body: dict,
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
    before_tool_call: Any = None,
    code_style: str | None = None,
    memory_auto_context: bool = True,
    memory_kg_max_results: int = 4,
    memory_rag_max_results: int = 4,
    memory_rag_snippet_chars: int = 220,
) -> dict:
    message, use_model, system = _build_chat_system(
        body,
        model,
        workspace_root,
        system_extra,
        use_skills=use_skills,
        prompt_template=prompt_template,
        inject_recent_context=inject_recent_context,
        recent_context_max_files=recent_context_max_files,
        use_reasoning=use_reasoning,
        prompt_snippets=prompt_snippets,
        code_style=code_style,
        memory_auto_context=memory_auto_context,
        memory_kg_max_results=memory_kg_max_results,
        memory_rag_max_results=memory_rag_max_results,
        memory_rag_snippet_chars=memory_rag_snippet_chars,
    )
    if message is None:
        return {"content": "", "error": "message required"}
    request_id = body.get("requestId") or uuid.uuid4().hex
    _set_apple_fm_session_key(str(request_id))
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
                before_tool_call=before_tool_call,
                request_id=request_id,
            )
        else:
            out = await run_agent_loop_no_mcp(
                use_model,
                message,
                system_prompt=system,
                request_id=request_id,
            )
        result: dict[str, Any] = {"content": out}
        edits = parse_edits(out)
        if edits:
            result["edits"] = edits
        if tool_errors:
            result["tool_errors"] = tool_errors
        return result
    except Exception as e:
        return {"content": "", "error": str(e)}


# Update type hint to Any


def _check_api_key(request: Any, api_key: str) -> JSONResponse | None:
    """If api_key is set, require Authorization: Bearer <key> or X-API-Key: <key>. Return 401 response if invalid, else None."""
    auth = request.headers.get("Authorization") or ""
    token = request.headers.get("X-API-Key") or ""
    if auth.startswith("Bearer "):
        token = auth[7:]
    if not token or token != api_key:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return None


_MAX_REQUEST_BODY_BYTES = 1_048_576  # 1 MB


def _check_body_size(request: Any) -> JSONResponse | None:
    """Reject requests whose Content-Length exceeds _MAX_REQUEST_BODY_BYTES. Return 413 response if too large, else None."""
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > _MAX_REQUEST_BODY_BYTES:
                import logging as _log

                _log.getLogger(__name__).warning(
                    "Request body too large: %s bytes (max %s)",
                    cl,
                    _MAX_REQUEST_BODY_BYTES,
                )
                return JSONResponse(
                    {
                        "error": f"request body too large (max {_MAX_REQUEST_BODY_BYTES} bytes)"
                    },
                    status_code=413,
                )
        except (ValueError, TypeError):
            pass
    return None


# create_app remains largely unchanged, but Request type hint will be Any


def create_app(
    model: str,
    mcp_servers: list[dict],
    system_extra: str,
    max_messages: int = 0,
    max_tool_result_chars: int = 0,
    workspace_root: str | None = None,
    api_key: str | None = None,
    scheduled_tasks: list[dict] | None = None,
    merged_config: dict | None = None,
    enable_channels: bool = True,
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
    planner_model: str | None = None,
    executor_model: str | None = None,
    reviewer_model: str | None = None,
    multi_agent_max_iterations: int = 2,
    multi_agent_require_review: bool = True,
    memory_auto_context: bool = True,
    memory_kg_max_results: int = 4,
    memory_rag_max_results: int = 4,
    memory_rag_snippet_chars: int = 220,
    rate_limit_rpm: int = 0,
    rate_limit_tpd: int = 0,
    webhook_secret: str | None = None,
    max_concurrent_requests: int = 4,
):
    """Create ASGI app (Starlette) with MCP session in lifespan.

    Authentication: if api_key is set, requests must send
    ``Authorization: Bearer <key>`` or ``X-API-Key: <key>``.

    Rate limiting: if rate_limit_rpm > 0, at most that many requests per
    client IP per minute are allowed; 429 with Retry-After on excess.
    If rate_limit_tpd > 0, a daily token budget is enforced per IP.

    Webhook HMAC: if webhook_secret is set, POST /chat must include
    ``X-Hub-Signature-256: sha256=<digest>`` computed over the raw body.
    """
    root = workspace_root or os.getcwd()
    try:
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse, StreamingResponse
        from starlette.routing import Route
    except ImportError as e:
        raise ImportError(
            "Server requires starlette. Install with: pip install ollamacode[server]"
        ) from e

    from .rate_limit import RateLimiter

    _concurrency_sem = asyncio.Semaphore(max(1, max_concurrent_requests))

    _limiter = RateLimiter(
        requests_per_minute=rate_limit_rpm,
        tokens_per_day=rate_limit_tpd,
    )

    def _check_rate_limit(request: Request) -> "JSONResponse | None":
        """Return a 429 JSONResponse if the client is rate-limited, else None."""
        if not _limiter.is_active():
            return None
        client_ip = (request.client.host if request.client else None) or "unknown"
        allowed, retry_after = _limiter.check(client_ip)
        if not allowed:
            return JSONResponse(
                {"error": "rate limit exceeded", "retry_after": retry_after},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )
        return None

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        app.state.session = None
        ctx = None
        if mcp_servers:
            if len(mcp_servers) == 1 and mcp_servers[0].get("type") == "stdio":
                cmd = mcp_servers[0].get("command", "python")
                args = mcp_servers[0].get("args") or []
                ctx = connect_mcp_stdio(cmd, args)
            else:
                ctx = connect_mcp_servers(mcp_servers)
            app.state.session = await ctx.__aenter__()
        # Start background scheduler if tasks are configured.
        _scheduler = None
        if scheduled_tasks:
            try:
                from .scheduler import Scheduler

                _scheduler = Scheduler(
                    scheduled_tasks, model=model, config=merged_config or {}
                )
                _scheduler.start()
            except Exception as exc:
                import logging as _log

                _log.getLogger(__name__).warning("Failed to start scheduler: %s", exc)
        # Start channel adapters (Telegram, Discord) if configured.
        _channel_handles: list = []
        if enable_channels and merged_config:
            try:
                from .channels import start_channels

                _channel_handles = start_channels(merged_config, model, merged_config)
            except Exception as exc:
                import logging as _log

                _log.getLogger(__name__).warning("Failed to start channels: %s", exc)
        try:
            yield
        finally:
            if _channel_handles:
                try:
                    from .channels import stop_channels

                    stop_channels(_channel_handles)
                except Exception:
                    pass
            if _scheduler is not None:
                _scheduler.stop()
            if ctx is not None:
                await ctx.__aexit__(None, None, None)

    async def health_handler(request: Request) -> JSONResponse:
        """GET /health: verify Ollama (and optionally MCP). No auth required."""
        ok, msg = check_ollama()
        return JSONResponse({"ollama": ok, "message": msg})

    async def chat(request: Request) -> JSONResponse:
        if request.method != "POST":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        size_err = _check_body_size(request)
        if size_err is not None:
            return size_err
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        rl_err = _check_rate_limit(request)
        if rl_err is not None:
            return rl_err
        # Concurrency limit: reject if too many in-flight requests
        if _concurrency_sem.locked():
            return JSONResponse(
                {"error": "too many concurrent requests", "retry_after": 5},
                status_code=429,
                headers={"Retry-After": "5"},
            )
        # Webhook HMAC verification (optional; for channel integrations).
        if webhook_secret:
            raw_body = await request.body()
            sig = request.headers.get("X-Hub-Signature-256", "")
            if not _verify_webhook_hmac(raw_body, sig, webhook_secret):
                return JSONResponse(
                    {"error": "invalid webhook signature"}, status_code=403
                )
            try:
                body = json.loads(raw_body)
            except Exception:
                return JSONResponse({"error": "invalid json"}, status_code=400)
        else:
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "invalid json"}, status_code=400)
        # Acquire concurrency permit (non-blocking check already done above)
        await _concurrency_sem.acquire()
        try:
            return await _chat_inner(request, body)
        finally:
            _concurrency_sem.release()

    async def _chat_inner(request: Any, body: dict) -> JSONResponse:
        session: McpConnection | None = getattr(request.app.state, "session", None)
        use_confirm = (
            body.get("confirmToolCalls") or confirm_tool_calls
        ) and session is not None
        from .hooks import HookManager

        hook_mgr = HookManager(root, None)
        req_mem_auto, req_mem_kg, req_mem_rag, req_mem_chars = (
            _resolve_memory_request_settings(
                body,
                default_auto=memory_auto_context,
                default_kg_max=memory_kg_max_results,
                default_rag_max=memory_rag_max_results,
                default_rag_chars=memory_rag_snippet_chars,
            )
        )
        if body.get("multiAgent") and use_confirm:
            message, use_model, system = _build_chat_system(
                body,
                model,
                root,
                system_extra,
                use_skills=use_skills,
                prompt_template=prompt_template,
                inject_recent_context=inject_recent_context,
                recent_context_max_files=recent_context_max_files,
                use_reasoning=use_reasoning,
                prompt_snippets=prompt_snippets,
                code_style=code_style,
                memory_auto_context=req_mem_auto,
                memory_kg_max_results=req_mem_kg,
                memory_rag_max_results=req_mem_rag,
                memory_rag_snippet_chars=req_mem_chars,
            )
            if message is None:
                return JSONResponse({"content": "", "error": "message required"})
            event = asyncio.Event()
            pending: dict[str, Any] = {}
            loop = asyncio.get_event_loop()

            async def before_tool_call(name: str, arguments: dict):
                try:
                    decision = await hook_mgr.run_pre_tool_use(
                        name, arguments, user_prompt=body.get("message")
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
                decision = await pending["future"]
                return decision

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
                    planner_model=planner_model,
                    executor_model=executor_model,
                    reviewer_model=reviewer_model,
                    max_iterations=multi_agent_max_iterations,
                    require_review=multi_agent_require_review,
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
                    return JSONResponse({"content": "", "error": str(e)})
                out = {
                    "content": result.content,
                    "plan": result.plan,
                    "review": result.review,
                }
                edits = parse_edits(result.content)
                if edits:
                    out["edits"] = edits
                return JSONResponse(out)
            token = uuid.uuid4().hex
            _approval_pending[token] = {
                "task": task,
                "event": event,
                "pending": pending,
                "mode": "multi",
                "created_at": time.monotonic(),
            }
            return JSONResponse(
                {
                    "toolApprovalRequired": {
                        "tool": pending["tool"],
                        "arguments": pending["arguments"],
                    },
                    "approvalToken": token,
                }
            )
        if body.get("multiAgent"):
            message, use_model, system = _build_chat_system(
                body,
                model,
                root,
                system_extra,
                use_skills=use_skills,
                prompt_template=prompt_template,
                inject_recent_context=inject_recent_context,
                recent_context_max_files=recent_context_max_files,
                use_reasoning=use_reasoning,
                prompt_snippets=prompt_snippets,
                code_style=code_style,
                memory_auto_context=req_mem_auto,
                memory_kg_max_results=req_mem_kg,
                memory_rag_max_results=req_mem_rag,
                memory_rag_snippet_chars=req_mem_chars,
            )
            if message is None:
                return JSONResponse({"content": "", "error": "message required"})
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
                planner_model=planner_model,
                executor_model=executor_model,
                reviewer_model=reviewer_model,
                max_iterations=multi_agent_max_iterations,
                require_review=multi_agent_require_review,
            )
            out = {
                "content": result.content,
                "plan": result.plan,
                "review": result.review,
            }
            edits = parse_edits(result.content)
            if edits:
                out["edits"] = edits
            return JSONResponse(out)
        if use_confirm:
            request_id = body.get("requestId") or uuid.uuid4().hex
            _set_apple_fm_session_key(str(request_id))
            message, use_model, system = _build_chat_system(
                body,
                model,
                root,
                system_extra,
                use_skills=use_skills,
                prompt_template=prompt_template,
                inject_recent_context=inject_recent_context,
                recent_context_max_files=recent_context_max_files,
                use_reasoning=use_reasoning,
                prompt_snippets=prompt_snippets,
                code_style=code_style,
                memory_auto_context=req_mem_auto,
                memory_kg_max_results=req_mem_kg,
                memory_rag_max_results=req_mem_rag,
                memory_rag_snippet_chars=req_mem_chars,
            )
            if message is None:
                return JSONResponse({"content": "", "error": "message required"})
            event = asyncio.Event()
            pending: dict[str, Any] = {}
            loop = asyncio.get_event_loop()

            async def before_tool_call(name: str, arguments: dict):
                try:
                    decision = await hook_mgr.run_pre_tool_use(
                        name, arguments, user_prompt=body.get("message")
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
                decision = await pending["future"]
                return decision

            tool_errors: list[dict[str, Any]] = []
            if session is None:
                return JSONResponse(
                    {"content": "", "error": "Tool approval requires MCP session"}
                )
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
                    request_id=request_id,
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
                    return JSONResponse({"content": "", "error": str(e)})
                result_out: dict[str, Any] = {"content": out}
                edits = parse_edits(out)
                if edits:
                    result_out["edits"] = edits
                if tool_errors:
                    result_out["tool_errors"] = tool_errors
                return JSONResponse(result_out)
            token = uuid.uuid4().hex
            _approval_pending[token] = {
                "task": task,
                "event": event,
                "pending": pending,
                "tool_errors": tool_errors,
                "created_at": time.monotonic(),
            }
            return JSONResponse(
                {
                    "toolApprovalRequired": {
                        "tool": pending["tool"],
                        "arguments": pending["arguments"],
                    },
                    "approvalToken": token,
                }
            )
        result = await _handle_chat(
            session,
            model,
            system_extra,
            body,
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
            memory_auto_context=req_mem_auto,
            memory_kg_max_results=req_mem_kg,
            memory_rag_max_results=req_mem_rag,
            memory_rag_snippet_chars=req_mem_chars,
        )
        return JSONResponse(result)

    async def chat_continue(request: Request) -> JSONResponse:
        """POST /chat/continue: send tool approval decision (run/skip/edit) and get next result or final content."""
        if request.method != "POST":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        size_err = _check_body_size(request)
        if size_err is not None:
            return size_err
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        rl_err = _check_rate_limit(request)
        if rl_err is not None:
            return rl_err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid json"}, status_code=400)
        token = body.get("approvalToken")
        await _cleanup_stale_approvals()
        async with _approval_lock:
            if not token or token not in _approval_pending:
                return JSONResponse(
                    {"error": "invalid or expired approvalToken"}, status_code=400
                )
            entry = _approval_pending.pop(token)
        decision = body.get("decision", "run")
        edited = body.get("editedArguments") if decision == "edit" else None
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
        tool_errors = entry.get("tool_errors", [])  # same list the task appends to
        while True:
            done, _ = await asyncio.wait(
                [task, asyncio.create_task(event.wait())],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if task in done:
                try:
                    out = task.result()
                except Exception as e:
                    return JSONResponse({"content": "", "error": str(e)})
                if mode == "multi":
                    result = {
                        "content": out.content,
                        "plan": out.plan,
                        "review": out.review,
                    }
                    edits = parse_edits(out.content)
                    if edits:
                        result["edits"] = edits
                    return JSONResponse(result)
                result = {"content": out}
                edits = parse_edits(out)
                if edits:
                    result["edits"] = edits
                if tool_errors:
                    result["tool_errors"] = tool_errors
                return JSONResponse(result)
            new_token = uuid.uuid4().hex
            async with _approval_lock:
                _approval_pending[new_token] = {
                    "task": task,
                    "event": event,
                    "pending": pending,
                    "tool_errors": entry.get("tool_errors", []),
                    "mode": mode,
                    "created_at": time.monotonic(),
                }
            return JSONResponse(
                {
                    "toolApprovalRequired": {
                        "tool": pending["tool"],
                        "arguments": pending["arguments"],
                    },
                    "approvalToken": new_token,
                }
            )

    async def chat_stream(request: Request):
        if request.method != "POST":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        size_err = _check_body_size(request)
        if size_err is not None:
            return size_err
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        rl_err = _check_rate_limit(request)
        if rl_err is not None:
            return rl_err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid json"}, status_code=400)
        req_mem_auto, req_mem_kg, req_mem_rag, req_mem_chars = (
            _resolve_memory_request_settings(
                body,
                default_auto=memory_auto_context,
                default_kg_max=memory_kg_max_results,
                default_rag_max=memory_rag_max_results,
                default_rag_chars=memory_rag_snippet_chars,
            )
        )
        message, use_model, system = _build_chat_system(
            body,
            model,
            root,
            system_extra,
            use_skills=use_skills,
            prompt_template=prompt_template,
            inject_recent_context=inject_recent_context,
            recent_context_max_files=recent_context_max_files,
            use_reasoning=use_reasoning,
            prompt_snippets=prompt_snippets,
            code_style=code_style,
            memory_auto_context=req_mem_auto,
            memory_kg_max_results=req_mem_kg,
            memory_rag_max_results=req_mem_rag,
            memory_rag_snippet_chars=req_mem_chars,
        )
        if message is None:
            return JSONResponse({"error": "message required"}, status_code=400)
        request_id = body.get("requestId") or uuid.uuid4().hex
        _set_apple_fm_session_key(str(request_id))

        _stream_timeout = float(
            os.environ.get("OLLAMACODE_SERVE_STREAM_TIMEOUT", "300")
        )  # 5 min default

        async def generate() -> AsyncIterator[str]:
            session: McpConnection | None = getattr(request.app.state, "session", None)
            accumulated: list[str] = []
            stream_start = time.monotonic()
            try:
                if session is not None:
                    stream = run_agent_loop_stream(
                        session,
                        use_model,
                        message,
                        system_prompt=system,
                        max_messages=max_messages,
                        max_tool_result_chars=max_tool_result_chars,
                        quiet=True,
                        allowed_tools=allowed_tools,
                        blocked_tools=blocked_tools,
                        confirm_tool_calls=confirm_tool_calls,
                        request_id=request_id,
                    )
                else:
                    stream = run_agent_loop_no_mcp_stream(
                        use_model,
                        message,
                        system_prompt=system,
                        message_history=[],
                        request_id=request_id,
                    )
                async for chunk in stream:
                    if time.monotonic() - stream_start > _stream_timeout:
                        yield f"data: {json.dumps({'type': 'error', 'error': 'Stream timeout exceeded'})}\n\n"
                        return
                    accumulated.append(chunk)
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                full = "".join(accumulated)
                edits = parse_edits(full)
                yield f"data: {json.dumps({'type': 'done', 'content': full, 'edits': edits})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def apply_edits_handler(request: Request):
        """POST /apply-edits: apply protocol edits server-side. Body: { "edits": [...], "workspaceRoot"?: "..." }."""
        if request.method != "POST":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        size_err = _check_body_size(request)
        if size_err is not None:
            return size_err
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid json"}, status_code=400)
        edits_raw = body.get("edits")
        if not isinstance(edits_raw, list):
            return JSONResponse(
                {"applied": 0, "error": "edits required (array)"}, status_code=400
            )
        workspace_override = body.get("workspaceRoot")
        work_root = (
            os.path.abspath(workspace_override)
            if isinstance(workspace_override, str) and workspace_override.strip()
            else root
        )
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
            return JSONResponse(
                {"applied": 0, "error": "no valid edits (path and newText required)"},
                status_code=400,
            )
        try:
            n = apply_edits(edits, work_root)
            return JSONResponse({"applied": n})
        except Exception as e:
            return JSONResponse({"applied": 0, "error": str(e)}, status_code=500)

    async def diagnostics_handler(request: Request):
        """POST /diagnostics: run linter, return LSP-like diagnostics. Body: { workspaceRoot?, path?, linterCommand? }."""
        if request.method != "POST":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        size_err = _check_body_size(request)
        if size_err is not None:
            return size_err
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        try:
            body = await request.json()
        except Exception:
            body = {}
        work_root = (body.get("workspaceRoot") or root).strip() or root
        path = body.get("path")
        linter = (body.get("linterCommand") or "ruff check .").strip() or "ruff check ."
        diag = get_diagnostics(work_root, path=path, linter_command=linter)
        return JSONResponse({"diagnostics": diag})

    async def complete_handler(request: Request):
        """POST /complete: inline completion for prefix. Body: { prefix, model? }."""
        if request.method != "POST":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        size_err = _check_body_size(request)
        if size_err is not None:
            return size_err
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        try:
            body = await request.json()
        except Exception:
            body = {}
        prefix = (body.get("prefix") or "").strip()
        use_model = body.get("model") or model
        completion = get_completion(prefix, use_model)
        return JSONResponse({"completions": [completion] if completion else []})

    async def rag_index_handler(request: Request):
        """POST /rag/index: build local RAG index from workspace files."""
        if request.method != "POST":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        size_err = _check_body_size(request)
        if size_err is not None:
            return size_err
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        try:
            body = await request.json()
        except Exception:
            body = {}
        target_root = (
            (body.get("workspaceRoot") or root).strip()
            if isinstance(body.get("workspaceRoot"), str)
            else root
        ) or root
        max_files = body.get("maxFiles")
        max_chars_per_file = body.get("maxCharsPerFile")
        try:
            info = build_local_rag_index(
                target_root,
                max_files=int(max_files) if isinstance(max_files, int) else 400,
                max_chars_per_file=int(max_chars_per_file)
                if isinstance(max_chars_per_file, int)
                else 20000,
            )
            return JSONResponse(info)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    async def rag_query_handler(request: Request):
        """POST /rag/query: retrieve top snippets from local RAG index."""
        if request.method != "POST":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        size_err = _check_body_size(request)
        if size_err is not None:
            return size_err
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        try:
            body = await request.json()
        except Exception:
            body = {}
        query = (
            (body.get("query") or "").strip()
            if isinstance(body.get("query"), str)
            else ""
        )
        if not query:
            return JSONResponse(
                {"results": [], "error": "query required"}, status_code=400
            )
        max_results = body.get("maxResults")
        try:
            rows = query_local_rag(
                query,
                max_results=int(max_results) if isinstance(max_results, int) else 5,
            )
            return JSONResponse({"results": rows})
        except Exception as e:
            return JSONResponse({"results": [], "error": str(e)}, status_code=500)

    app = Starlette(
        routes=[
            Route("/health", health_handler, methods=["GET"]),
            Route("/chat", chat, methods=["POST"]),
            Route("/chat/continue", chat_continue, methods=["POST"]),
            Route("/chat/stream", chat_stream, methods=["POST"]),
            Route("/apply-edits", apply_edits_handler, methods=["POST"]),
            Route("/diagnostics", diagnostics_handler, methods=["POST"]),
            Route("/complete", complete_handler, methods=["POST"]),
            Route("/rag/index", rag_index_handler, methods=["POST"]),
            Route("/rag/query", rag_query_handler, methods=["POST"]),
        ],
        lifespan=lifespan,
    )
    return app


# run_serve unchanged except for imports


def run_serve(
    port: int = 8000, config_path: str | None = None, no_tunnel: bool = False
) -> None:
    """Load config, create app, run uvicorn."""
    try:
        import uvicorn
    except ImportError as e:
        raise SystemExit(
            "Server requires uvicorn. Install with: pip install ollamacode[server]"
        ) from e

    config = load_config(config_path)
    merged = merge_config_with_env(config, **get_env_config_overrides())
    model = merged.get("model") or os.environ.get("OLLAMACODE_MODEL", "gpt-oss:20b")
    system_extra = (merged.get("system_prompt_extra") or "").strip()
    mcp_servers = merged.get("mcp_servers") or []
    max_messages = merged.get("max_messages", 0)
    max_tool_result_chars = merged.get("max_tool_result_chars", 0)
    workspace_root = os.getcwd()
    serve_config = merged.get("serve") or {}
    api_key = (
        serve_config.get("api_key")
        or os.environ.get("OLLAMACODE_SERVE_API_KEY")
        or os.environ.get("OLLAMACODE_API_KEY")
        or ""
    ).strip() or None
    rate_limit_rpm = int(
        serve_config.get("rate_limit_rpm")
        or os.environ.get("OLLAMACODE_RATE_LIMIT_RPM")
        or 0
    )
    rate_limit_tpd = int(
        serve_config.get("rate_limit_tpd")
        or os.environ.get("OLLAMACODE_RATE_LIMIT_TPD")
        or 0
    )
    webhook_secret = (
        serve_config.get("webhook_secret")
        or os.environ.get("OLLAMACODE_WEBHOOK_SECRET")
        or ""
    ).strip() or None

    from .scheduler import load_scheduled_tasks

    scheduled = load_scheduled_tasks(merged, workspace_root)

    app = create_app(
        model,
        mcp_servers,
        system_extra,
        max_messages,
        max_tool_result_chars,
        workspace_root,
        api_key=api_key,
        scheduled_tasks=scheduled or None,
        merged_config=merged,
        use_skills=merged.get("use_skills", True),
        prompt_template=merged.get("prompt_template"),
        inject_recent_context=merged.get("inject_recent_context", True),
        recent_context_max_files=merged.get("recent_context_max_files", 10),
        use_reasoning=merged.get("use_reasoning", True),
        prompt_snippets=merged.get("prompt_snippets") or [],
        allowed_tools=merged.get("allowed_tools"),
        blocked_tools=merged.get("blocked_tools"),
        confirm_tool_calls=merged.get("confirm_tool_calls", False),
        code_style=merged.get("code_style"),
        planner_model=merged.get("planner_model"),
        executor_model=merged.get("executor_model"),
        reviewer_model=merged.get("reviewer_model"),
        multi_agent_max_iterations=merged.get("multi_agent_max_iterations", 2),
        multi_agent_require_review=merged.get("multi_agent_require_review", True),
        memory_auto_context=merged.get("memory_auto_context", True),
        memory_kg_max_results=merged.get("memory_kg_max_results", 4),
        memory_rag_max_results=merged.get("memory_rag_max_results", 4),
        memory_rag_snippet_chars=merged.get("memory_rag_snippet_chars", 220),
        rate_limit_rpm=rate_limit_rpm,
        rate_limit_tpd=rate_limit_tpd,
        webhook_secret=webhook_secret,
        enable_channels=True,
    )
    # Start tunnel if configured and not disabled.
    _tunnel_proc = None
    if not no_tunnel:
        try:
            from .tunnel import start_tunnel_from_config

            tunnel_url, _tunnel_proc = start_tunnel_from_config(merged, port)
            if tunnel_url:
                print(f"[OllamaCode] Tunnel active: {tunnel_url}", flush=True)
        except Exception as exc:
            import logging as _log

            _log.getLogger(__name__).warning("Tunnel error: %s", exc)
    try:
        uvicorn.run(app, host="127.0.0.1", port=port)
    finally:
        if _tunnel_proc is not None:
            try:
                from .tunnel import stop_tunnel

                stop_tunnel(_tunnel_proc)
            except Exception:
                pass
