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
from pathlib import Path

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
from .permission_runtime import SessionApprovalStore, evaluate_permission
from .question_runtime import format_question_answers, normalize_question_list
from .task_runtime import run_task_delegation
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


def _webapp_asset(name: str) -> tuple[str, str]:
    base = Path(__file__).resolve().parent / "webapp"
    path = (base / name).resolve()
    if not path.is_file() or not path.is_relative_to(base):
        raise FileNotFoundError(name)
    content_type = {
        ".html": "text/html; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
    }.get(path.suffix, "text/plain; charset=utf-8")
    return path.read_text(encoding="utf-8"), content_type


async def _proxy_remote_workspace_request(
    method: str,
    url: str,
    *,
    body_bytes: bytes | None,
    api_key: str = "",
) -> tuple[int, str, Any]:
    """Forward a request to a remote workspace."""
    import httpx

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.request(
            method,
            url,
            content=body_bytes,
            headers=headers,
        )
    content_type = resp.headers.get("content-type", "application/json")
    try:
        payload = resp.json()
    except Exception:
        payload = {"content": resp.text, "contentType": content_type}
    return resp.status_code, content_type, payload


async def _proxy_remote_workspace_sse(
    method: str,
    url: str,
    *,
    body_bytes: bytes | None,
    api_key: str = "",
) -> AsyncIterator[bytes]:
    """Forward an SSE request to a remote workspace."""
    import httpx

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream(
            method,
            url,
            content=body_bytes,
            headers=headers,
        ) as resp:
            async for chunk in resp.aiter_bytes():
                yield chunk


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
    """Validate bearer/API key against configured key or principal registry."""
    auth = request.headers.get("Authorization") or ""
    token = request.headers.get("X-API-Key") or ""
    if auth.startswith("Bearer "):
        token = auth[7:]
    principal = None
    if token:
        try:
            from .auth_registry import find_principal_by_token

            principal = find_principal_by_token(token)
        except Exception:
            principal = None
    if token and principal is not None:
        try:
            request.state.principal = principal
        except Exception:
            pass
        return None
    if not token or token != api_key:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        request.state.principal = {
            "id": "legacy",
            "name": "Legacy API Key",
            "role": "admin",
            "workspace_ids": [],
        }
    except Exception:
        pass
    return None


def _principal(request: Any) -> dict[str, Any] | None:
    try:
        return getattr(request.state, "principal", None)
    except Exception:
        return None


def _principal_role(principal: dict[str, Any] | None) -> str:
    return str((principal or {}).get("role") or "").strip().lower()


def _principal_name(principal: dict[str, Any] | None) -> str:
    return str((principal or {}).get("name") or "").strip()


def _principal_is_admin(principal: dict[str, Any] | None) -> bool:
    return _principal_role(principal) == "admin"


def _principal_can_write(principal: dict[str, Any] | None) -> bool:
    return _principal_role(principal) in {"admin", "editor"}


def _principal_has_workspace(principal: dict[str, Any] | None, workspace_id: str) -> bool:
    if not principal or not workspace_id:
        return False
    return workspace_id in list(principal.get("workspace_ids") or [])


def _forbidden() -> JSONResponse:
    return JSONResponse({"error": "forbidden"}, status_code=403)


def _authorize_admin(request: Any) -> JSONResponse | None:
    principal = _principal(request)
    if principal is None or _principal_is_admin(principal):
        return None
    return _forbidden()


def _authorize_workspace_access(
    request: Any,
    workspace: dict[str, Any] | None,
    *,
    write: bool = False,
) -> JSONResponse | None:
    principal = _principal(request)
    if principal is None or _principal_is_admin(principal):
        return None
    if workspace is None:
        return _forbidden()
    owner = str(workspace.get("owner") or "").strip()
    if owner and owner == _principal_name(principal):
        return None if (not write or _principal_can_write(principal)) else _forbidden()
    if _principal_has_workspace(principal, str(workspace.get("id") or "")):
        return None if (not write or _principal_can_write(principal)) else _forbidden()
    return _forbidden()


def _authorize_session_access(
    request: Any,
    session_info: dict[str, Any] | None,
    *,
    write: bool = False,
) -> JSONResponse | None:
    principal = _principal(request)
    if principal is None or _principal_is_admin(principal):
        return None
    if session_info is None:
        return _forbidden()
    owner = str(session_info.get("owner") or "").strip()
    if not owner:
        return None if not write else _forbidden()
    if owner == _principal_name(principal):
        return None if (not write or _principal_can_write(principal)) else _forbidden()
    return _forbidden()


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
    from .permissions import PermissionManager, ToolPermission

    _concurrency_sem = asyncio.Semaphore(max(1, max_concurrent_requests))
    _serve_session_approvals = SessionApprovalStore()
    permission_manager = PermissionManager.from_config(merged_config)

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

    async def web_app_handler(request: Request):
        """GET /app: serve the built-in browser dashboard."""
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        try:
            text, content_type = _webapp_asset("index.html")
        except FileNotFoundError:
            return JSONResponse({"error": "web app not found"}, status_code=404)
        from starlette.responses import Response

        return Response(text, media_type=content_type)

    async def web_asset_handler(request: Request):
        """Serve a specific browser app asset."""
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        name = request.url.path.rsplit("/", 1)[-1]
        try:
            text, content_type = _webapp_asset(name)
        except FileNotFoundError:
            return JSONResponse({"error": "asset not found"}, status_code=404)
        from starlette.responses import Response

        return Response(text, media_type=content_type)

    async def events_handler(request: Request):
        """GET /events: lightweight control-plane SSE stream."""
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        from starlette.responses import StreamingResponse
        from .control_plane import subscribe, unsubscribe

        async def generate():
            queue = subscribe()
            try:
                yield f"data: {json.dumps({'type': 'ready'})}\n\n"
                while True:
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                        continue
                    yield f"data: {json.dumps(item)}\n\n"
            finally:
                unsubscribe(queue)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def recent_events_handler(request: Request):
        """GET /events/recent: return a bounded recent control-plane event list."""
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        from .control_plane import list_recent_events

        try:
            limit = int(request.query_params.get("limit") or 50)
        except ValueError:
            limit = 50
        return JSONResponse({"events": list_recent_events(limit=limit)})

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
        session_key = str(body.get("sessionID") or root)
        req_mem_auto, req_mem_kg, req_mem_rag, req_mem_chars = (
            _resolve_memory_request_settings(
                body,
                default_auto=memory_auto_context,
                default_kg_max=memory_kg_max_results,
                default_rag_max=memory_rag_max_results,
                default_rag_chars=memory_rag_snippet_chars,
            )
        )
        if body.get("multiAgent") and session is not None:
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
                normalized_name = name.removeprefix("functions::")
                if name.endswith("question") or name == "question":
                    questions = normalize_question_list(arguments)
                    if questions:
                        pending["kind"] = "question"
                        pending["questions"] = questions
                        pending["future"] = loop.create_future()
                        event.set()
                        return await pending["future"]
                    return ("skip", "Question tool called without valid questions.")
                if name.endswith("task") or name == "task":
                    result = await run_task_delegation(
                        session=session,
                        session_id=body.get("sessionID"),
                        workspace_root=root,
                        subagents=(merged_config or {}).get("subagents") or [],
                        arguments=arguments,
                        default_model=use_model,
                        system_prompt=system,
                        max_messages=max_messages,
                        max_tool_rounds=20,
                        max_tool_result_chars=max_tool_result_chars,
                        before_tool_call=before_tool_call,
                    )
                    return ("skip", result)
                permission = evaluate_permission(
                    permission_manager,
                    _serve_session_approvals,
                    session_key,
                    [name, normalized_name],
                )
                if permission is ToolPermission.DENY:
                    _serve_session_approvals.record_deny(session_key)
                    return ("skip", f"Blocked by permission rule for tool: {normalized_name}")
                if permission is ToolPermission.ALLOW:
                    _serve_session_approvals.record_grant(session_key)
                    return "run"
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
                if not use_confirm:
                    return "run"
                pending["kind"] = "approval"
                pending["tool"] = name
                pending["arguments"] = arguments
                pending["patterns"] = [name, normalized_name]
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
                "session_key": session_key,
                "created_at": time.monotonic(),
            }
            return JSONResponse(
                (
                    {
                        "questionRequired": {"questions": pending["questions"]},
                        "approvalToken": token,
                        "continuationToken": token,
                    }
                    if pending.get("kind") == "question"
                    else {
                        "toolApprovalRequired": {
                            "tool": pending["tool"],
                            "arguments": pending["arguments"],
                        },
                        "approvalToken": token,
                        "continuationToken": token,
                    }
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
                normalized_name = name.removeprefix("functions::")
                permission = evaluate_permission(
                    permission_manager,
                    _serve_session_approvals,
                    session_key,
                    [name, normalized_name],
                )
                if permission is ToolPermission.DENY:
                    _serve_session_approvals.record_deny(session_key)
                    return ("skip", f"Blocked by permission rule for tool: {normalized_name}")
                if permission is ToolPermission.ALLOW:
                    _serve_session_approvals.record_grant(session_key)
                    return "run"
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
                pending["patterns"] = [name, normalized_name]
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
                "session_key": session_key,
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
                normalized_name = name.removeprefix("functions::")
                permission = evaluate_permission(
                    permission_manager,
                    _serve_session_approvals,
                    session_key,
                    [name, normalized_name],
                )
                if permission is ToolPermission.DENY:
                    _serve_session_approvals.record_deny(session_key)
                    return ("skip", f"Blocked by permission rule for tool: {normalized_name}")
                if permission is ToolPermission.ALLOW:
                    _serve_session_approvals.record_grant(session_key)
                    return "run"
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
                pending["patterns"] = [name, normalized_name]
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
                "session_key": session_key,
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
        if session is not None:
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
                normalized_name = name.removeprefix("functions::")
                if name.endswith("question") or name == "question":
                    questions = normalize_question_list(arguments)
                    if questions:
                        pending["kind"] = "question"
                        pending["questions"] = questions
                        pending["future"] = loop.create_future()
                        event.set()
                        return await pending["future"]
                    return ("skip", "Question tool called without valid questions.")
                if name.endswith("task") or name == "task":
                    result = await run_task_delegation(
                        session=session,
                        session_id=body.get("sessionID"),
                        workspace_root=root,
                        subagents=(merged_config or {}).get("subagents") or [],
                        arguments=arguments,
                        default_model=use_model,
                        system_prompt=system,
                        max_messages=max_messages,
                        max_tool_rounds=20,
                        max_tool_result_chars=max_tool_result_chars,
                        before_tool_call=before_tool_call,
                    )
                    return ("skip", result)
                permission = evaluate_permission(
                    permission_manager,
                    _serve_session_approvals,
                    session_key,
                    [name, normalized_name],
                )
                if permission is ToolPermission.DENY:
                    _serve_session_approvals.record_deny(session_key)
                    return ("skip", f"Blocked by permission rule for tool: {normalized_name}")
                if permission is ToolPermission.ALLOW:
                    _serve_session_approvals.record_grant(session_key)
                    return "run"
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
                if not use_confirm:
                    return "run"
                pending["kind"] = "approval"
                pending["tool"] = name
                pending["arguments"] = arguments
                pending["patterns"] = [name, normalized_name]
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
                "session_key": session_key,
                "created_at": time.monotonic(),
            }
            return JSONResponse(
                (
                    {
                        "questionRequired": {"questions": pending["questions"]},
                        "approvalToken": token,
                        "continuationToken": token,
                    }
                    if pending.get("kind") == "question"
                    else {
                        "toolApprovalRequired": {
                            "tool": pending["tool"],
                            "arguments": pending["arguments"],
                        },
                        "approvalToken": token,
                        "continuationToken": token,
                    }
                )
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
        token = body.get("approvalToken") or body.get("continuationToken")
        await _cleanup_stale_approvals()
        async with _approval_lock:
            if not token or token not in _approval_pending:
                return JSONResponse(
                    {"error": "invalid or expired approvalToken"}, status_code=400
                )
            entry = _approval_pending.pop(token)
        if entry.get("pending", {}).get("kind") == "question":
            answers_raw = body.get("answers")
            if not isinstance(answers_raw, list):
                single = body.get("answer")
                answers_raw = [single] if single is not None else []
            answers = [str(item or "") for item in answers_raw]
            decision_value: Any = (
                "skip",
                format_question_answers(entry.get("pending", {}).get("questions") or [], answers),
            )
        else:
            decision = body.get("decision", "run")
            edited = body.get("editedArguments") if decision == "edit" else None
            if decision == "edit" and isinstance(edited, dict):
                decision_value = ("edit", edited)
            elif decision == "always":
                _serve_session_approvals.allow(
                    entry.get("session_key"),
                    entry.get("pending", {}).get("patterns") or [],
                )
                _serve_session_approvals.record_grant(entry.get("session_key"))
                decision_value = "run"
            elif decision == "skip":
                _serve_session_approvals.record_deny(entry.get("session_key"))
                decision_value = "skip"
            else:
                _serve_session_approvals.record_grant(entry.get("session_key"))
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
                    "session_key": entry.get("session_key"),
                    "created_at": time.monotonic(),
                }
            return JSONResponse(
                (
                    {
                        "questionRequired": {"questions": pending["questions"]},
                        "approvalToken": new_token,
                        "continuationToken": new_token,
                    }
                    if pending.get("kind") == "question"
                    else {
                        "toolApprovalRequired": {
                            "tool": pending["tool"],
                            "arguments": pending["arguments"],
                        },
                        "approvalToken": new_token,
                        "continuationToken": new_token,
                    }
                )
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
        session_key = str(body.get("sessionID") or root)

        _stream_timeout = float(
            os.environ.get("OLLAMACODE_SERVE_STREAM_TIMEOUT", "300")
        )  # 5 min default

        async def generate() -> AsyncIterator[str]:
            session: McpConnection | None = getattr(request.app.state, "session", None)
            accumulated: list[str] = []
            stream_start = time.monotonic()
            try:
                if session is not None:
                    event = asyncio.Event()
                    pending: dict[str, Any] = {}
                    loop = asyncio.get_event_loop()
                    from .hooks import HookManager
                    from .permissions import ToolPermission

                    hook_mgr = HookManager(root, None)
                    use_confirm = (
                        body.get("confirmToolCalls") or confirm_tool_calls
                    ) and session is not None

                    async def before_tool_call(name: str, arguments: dict):
                        normalized_name = name.removeprefix("functions::")
                        if name.endswith("question") or name == "question":
                            questions = normalize_question_list(arguments)
                            if questions:
                                pending["kind"] = "question"
                                pending["questions"] = questions
                                pending["future"] = loop.create_future()
                                event.set()
                                return await pending["future"]
                            return ("skip", "Question tool called without valid questions.")
                        if name.endswith("task") or name == "task":
                            result = await run_task_delegation(
                                session=session,
                                session_id=body.get("sessionID"),
                                workspace_root=root,
                                subagents=(merged_config or {}).get("subagents") or [],
                                arguments=arguments,
                                default_model=use_model,
                                system_prompt=system,
                                max_messages=max_messages,
                                max_tool_rounds=20,
                                max_tool_result_chars=max_tool_result_chars,
                                before_tool_call=before_tool_call,
                            )
                            return ("skip", result)
                        permission = evaluate_permission(
                            permission_manager,
                            _serve_session_approvals,
                            session_key,
                            [name, normalized_name],
                        )
                        if permission is ToolPermission.DENY:
                            _serve_session_approvals.record_deny(session_key)
                            return ("skip", f"Blocked by permission rule for tool: {normalized_name}")
                        if permission is ToolPermission.ALLOW:
                            _serve_session_approvals.record_grant(session_key)
                            return "run"
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
                        if not use_confirm:
                            return "run"
                        pending["kind"] = "approval"
                        pending["tool"] = name
                        pending["arguments"] = arguments
                        pending["patterns"] = [name, normalized_name]
                        pending["future"] = loop.create_future()
                        event.set()
                        return await pending["future"]

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
                        confirm_tool_calls=True,
                        before_tool_call=before_tool_call,
                        request_id=request_id,
                    )

                    queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()

                    async def _consume_stream() -> str:
                        async for chunk in stream:
                            await queue.put(("chunk", chunk))
                        return "".join(accumulated)

                    task = asyncio.create_task(_consume_stream())
                    while True:
                        queue_task = asyncio.create_task(queue.get())
                        event_task = asyncio.create_task(event.wait())
                        done, _ = await asyncio.wait(
                            [task, queue_task, event_task],
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if queue_task in done:
                            _kind, chunk = queue_task.result()
                            if time.monotonic() - stream_start > _stream_timeout:
                                yield f"data: {json.dumps({'type': 'error', 'error': 'Stream timeout exceeded'})}\n\n"
                                return
                            accumulated.append(chunk)
                            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                        elif task in done:
                            full = task.result()
                            edits = parse_edits(full)
                            yield f"data: {json.dumps({'type': 'done', 'content': full, 'edits': edits})}\n\n"
                            return
                        else:
                            token = uuid.uuid4().hex
                            _approval_pending[token] = {
                                "task": task,
                                "event": event,
                                "pending": pending,
                                "session_key": session_key,
                                "created_at": time.monotonic(),
                            }
                            payload = (
                                {
                                    "type": "question",
                                    "questionRequired": {"questions": pending["questions"]},
                                    "approvalToken": token,
                                    "continuationToken": token,
                                }
                                if pending.get("kind") == "question"
                                else {
                                    "type": "toolApproval",
                                    "toolApprovalRequired": {
                                        "tool": pending["tool"],
                                        "arguments": pending["arguments"],
                                    },
                                    "approvalToken": token,
                                    "continuationToken": token,
                                }
                            )
                            yield f"data: {json.dumps(payload)}\n\n"
                            return
                        for pending_task in (queue_task, event_task):
                            if pending_task not in done:
                                pending_task.cancel()
                    return
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

    async def sessions_list_handler(request: Request):
        """GET /sessions: list sessions, optionally filtered by workspace/search."""
        if request.method != "GET":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        from .sessions import list_sessions, search_sessions

        query = request.query_params
        workspace = (query.get("workspaceRoot") or "").strip() or None
        search = (query.get("search") or "").strip()
        try:
            limit = int(query.get("limit") or 50)
        except ValueError:
            limit = 50
        if search:
            rows = search_sessions(search, limit=limit)
            if workspace:
                rows = [row for row in rows if row.get("workspace_root") == workspace]
        else:
            rows = list_sessions(limit=limit, workspace_root=workspace)
        principal = _principal(request)
        if principal is not None and not _principal_is_admin(principal):
            name = _principal_name(principal)
            rows = [row for row in rows if not row.get("owner") or row.get("owner") == name]
        return JSONResponse({"sessions": rows})

    async def session_get_handler(request: Request):
        """GET /sessions/{id}: return session metadata and messages."""
        if request.method != "GET":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        from .sessions import get_session_info, load_session

        session_id = request.path_params.get("session_id", "")
        info = get_session_info(session_id)
        if info is None:
            return JSONResponse({"error": "session not found"}, status_code=404)
        denied = _authorize_session_access(request, info)
        if denied is not None:
            return denied
        return JSONResponse({"session": info, "messages": load_session(session_id) or []})

    async def session_messages_handler(request: Request):
        """GET /sessions/{id}/messages: return just session messages."""
        if request.method != "GET":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        from .sessions import load_session
        from .sessions import get_session_info

        session_id = request.path_params.get("session_id", "")
        denied = _authorize_session_access(request, get_session_info(session_id))
        if denied is not None:
            return denied
        messages = load_session(session_id)
        if messages is None:
            return JSONResponse({"error": "session not found"}, status_code=404)
        return JSONResponse({"messages": messages})

    async def session_children_handler(request: Request):
        """GET /sessions/{id}/children: return child sessions."""
        if request.method != "GET":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        from .sessions import list_child_sessions
        from .sessions import get_session_info

        session_id = request.path_params.get("session_id", "")
        denied = _authorize_session_access(request, get_session_info(session_id))
        if denied is not None:
            return denied
        return JSONResponse({"sessions": list_child_sessions(session_id)})

    async def session_ancestors_handler(request: Request):
        """GET /sessions/{id}/ancestors: return ancestor chain."""
        if request.method != "GET":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        from .sessions import list_session_ancestors
        from .sessions import get_session_info

        session_id = request.path_params.get("session_id", "")
        denied = _authorize_session_access(request, get_session_info(session_id))
        if denied is not None:
            return denied
        return JSONResponse({"sessions": list_session_ancestors(session_id)})

    async def session_timeline_handler(request: Request):
        """GET /sessions/{id}/timeline: return combined timeline view."""
        if request.method != "GET":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        from .sessions import get_session_timeline

        session_id = request.path_params.get("session_id", "")
        timeline = get_session_timeline(session_id)
        if timeline is None:
            return JSONResponse({"error": "session not found"}, status_code=404)
        denied = _authorize_session_access(request, timeline.get("session"))
        if denied is not None:
            return denied
        return JSONResponse({"timeline": timeline})

    async def session_create_handler(request: Request):
        """POST /sessions: create a new session."""
        if request.method != "POST":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        try:
            body = await request.json()
        except Exception:
            body = {}
        from .sessions import create_session, get_session_info

        title = str(body.get("title") or "").strip()
        workspace = str(body.get("workspaceRoot") or "").strip() or root
        session_id = create_session(
            title=title,
            workspace_root=workspace,
            owner=str(body.get("owner") or ""),
            role=str(body.get("role") or "owner"),
        )
        return JSONResponse({"session": get_session_info(session_id)})

    async def session_update_handler(request: Request):
        """PATCH /sessions/{id}: update session metadata."""
        if request.method != "PATCH":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid json"}, status_code=400)
        from .sessions import update_session

        session_id = request.path_params.get("session_id", "")
        from .sessions import get_session_info
        denied = _authorize_session_access(request, get_session_info(session_id), write=True)
        if denied is not None:
            return denied
        session = update_session(
            session_id,
            title=body.get("title"),
            workspace_root=body.get("workspaceRoot"),
            owner=body.get("owner"),
            role=body.get("role"),
        )
        if session is None:
            return JSONResponse({"error": "session not found"}, status_code=404)
        return JSONResponse({"session": session})

    async def session_export_handler(request: Request):
        """GET /sessions/{id}/export: export session JSON."""
        if request.method != "GET":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        from .sessions import export_session

        session_id = request.path_params.get("session_id", "")
        from .sessions import get_session_info
        denied = _authorize_session_access(request, get_session_info(session_id))
        if denied is not None:
            return denied
        data = export_session(session_id)
        if data is None:
            return JSONResponse({"error": "session not found"}, status_code=404)
        return JSONResponse({"data": data})

    async def session_import_handler(request: Request):
        """POST /sessions/import: import session JSON and return new session info."""
        if request.method != "POST":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid json"}, status_code=400)
        from .sessions import get_session_info, import_session

        data = body.get("data")
        if not isinstance(data, str) or not data.strip():
            return JSONResponse({"error": "data required"}, status_code=400)
        try:
            session_id = import_session(data, title=body.get("title"))
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse({"session": get_session_info(session_id)})

    async def session_todo_handler(request: Request):
        """GET /sessions/{id}/todos: return persisted session todos."""
        if request.method != "GET":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        from .sessions import load_session_todos

        session_id = request.path_params.get("session_id", "")
        from .sessions import get_session_info
        denied = _authorize_session_access(request, get_session_info(session_id))
        if denied is not None:
            return denied
        todos = load_session_todos(session_id)
        if todos is None:
            return JSONResponse({"error": "session not found"}, status_code=404)
        return JSONResponse({"todos": todos})

    async def session_delete_handler(request: Request):
        """DELETE /sessions/{id}: delete a session."""
        if request.method != "DELETE":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        from .sessions import delete_session

        session_id = request.path_params.get("session_id", "")
        from .sessions import get_session_info
        denied = _authorize_session_access(request, get_session_info(session_id), write=True)
        if denied is not None:
            return denied
        deleted = delete_session(session_id)
        if not deleted:
            return JSONResponse({"error": "session not found"}, status_code=404)
        return JSONResponse({"deleted": True})

    async def session_branch_handler(request: Request):
        """POST /sessions/{id}/branch: branch a session."""
        if request.method != "POST":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        try:
            body = await request.json()
        except Exception:
            body = {}
        from .sessions import branch_session, get_session_info

        session_id = request.path_params.get("session_id", "")
        denied = _authorize_session_access(request, get_session_info(session_id), write=True)
        if denied is not None:
            return denied
        new_id = branch_session(session_id, title=body.get("title"))
        if new_id is None:
            return JSONResponse({"error": "session not found"}, status_code=404)
        return JSONResponse({"session": get_session_info(new_id)})

    async def session_fork_handler(request: Request):
        """POST /sessions/{id}/fork: fork a session at a message index."""
        if request.method != "POST":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        try:
            body = await request.json()
        except Exception:
            body = {}
        from .sessions import fork_session, get_session_info

        session_id = request.path_params.get("session_id", "")
        denied = _authorize_session_access(request, get_session_info(session_id), write=True)
        if denied is not None:
            return denied
        message_index = body.get("messageIndex")
        if not isinstance(message_index, int):
            return JSONResponse({"error": "messageIndex required"}, status_code=400)
        new_id = fork_session(session_id, message_index, title=body.get("title"))
        if new_id is None:
            return JSONResponse(
                {"error": "session not found or invalid messageIndex"}, status_code=404
            )
        return JSONResponse({"session": get_session_info(new_id)})

    async def workspaces_list_handler(request: Request):
        """GET /workspaces: list registered workspaces."""
        if request.method != "GET":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        from .workspaces import list_workspaces

        principal = _principal(request)
        rows = list_workspaces()
        if principal is not None and not _principal_is_admin(principal):
            name = _principal_name(principal)
            rows = [
                row
                for row in rows
                if row.get("owner") == name
                or _principal_has_workspace(principal, str(row.get("id") or ""))
            ]
        return JSONResponse({"workspaces": rows})

    async def workspace_get_handler(request: Request):
        """GET /workspaces/{id}: get a registered workspace."""
        if request.method != "GET":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        from .workspaces import get_workspace

        workspace_id = request.path_params.get("workspace_id", "")
        workspace = get_workspace(workspace_id)
        if workspace is None:
            return JSONResponse({"error": "workspace not found"}, status_code=404)
        denied = _authorize_workspace_access(request, workspace)
        if denied is not None:
            return denied
        return JSONResponse({"workspace": workspace})

    async def workspace_update_handler(request: Request):
        """PATCH /workspaces/{id}: update a workspace entry."""
        if request.method != "PATCH":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid json"}, status_code=400)
        from .workspaces import update_workspace
        from .workspaces import get_workspace

        workspace_id = request.path_params.get("workspace_id", "")
        denied = _authorize_workspace_access(request, get_workspace(workspace_id), write=True)
        if denied is not None:
            return denied
        workspace = update_workspace(
            workspace_id,
            name=body.get("name"),
            kind=body.get("type"),
            workspace_root=body.get("workspaceRoot"),
            base_url=body.get("baseUrl"),
            api_key=body.get("apiKey"),
            owner=body.get("owner"),
            role=body.get("role"),
        )
        if workspace is None:
            return JSONResponse({"error": "workspace not found"}, status_code=404)
        return JSONResponse({"workspace": workspace})

    async def workspace_create_handler(request: Request):
        """POST /workspaces: create a local or remote workspace entry."""
        if request.method != "POST":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid json"}, status_code=400)
        from .workspaces import create_workspace

        name = str(body.get("name") or "").strip()
        if not name:
            return JSONResponse({"error": "name required"}, status_code=400)
        workspace = create_workspace(
            name=name,
            kind=str(body.get("type") or "local"),
            workspace_root=str(body.get("workspaceRoot") or ""),
            base_url=str(body.get("baseUrl") or ""),
            api_key=str(body.get("apiKey") or ""),
            owner=str(body.get("owner") or ""),
            role=str(body.get("role") or "owner"),
        )
        return JSONResponse({"workspace": workspace})

    async def workspace_delete_handler(request: Request):
        """DELETE /workspaces/{id}: delete a workspace entry."""
        if request.method != "DELETE":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        from .workspaces import delete_workspace
        from .workspaces import get_workspace

        workspace_id = request.path_params.get("workspace_id", "")
        denied = _authorize_workspace_access(request, get_workspace(workspace_id), write=True)
        if denied is not None:
            return denied
        deleted = delete_workspace(workspace_id)
        if not deleted:
            return JSONResponse({"error": "workspace not found"}, status_code=404)
        return JSONResponse({"deleted": True})

    async def workspace_health_handler(request: Request):
        """GET /workspaces/{id}/health: check remote workspace health."""
        if request.method != "GET":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        from .workspaces import get_workspace, update_workspace
        import httpx

        workspace_id = request.path_params.get("workspace_id", "")
        workspace = get_workspace(workspace_id)
        if workspace is None:
            return JSONResponse({"error": "workspace not found"}, status_code=404)
        denied = _authorize_workspace_access(request, workspace)
        if denied is not None:
            return denied
        if workspace.get("type") != "remote" or not workspace.get("base_url"):
            return JSONResponse({"ok": True, "workspace": workspace})
        try:
            headers = {}
            if workspace.get("api_key"):
                headers["Authorization"] = f"Bearer {workspace['api_key']}"
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(str(workspace["base_url"]).rstrip("/") + "/health", headers=headers)
            workspace = update_workspace(
                workspace_id,
                last_status="ok" if resp.status_code == 200 else "error",
                last_error="" if resp.status_code == 200 else f"HTTP {resp.status_code}",
            ) or workspace
            return JSONResponse({"ok": resp.status_code == 200, "statusCode": resp.status_code, "workspace": workspace})
        except Exception as e:
            workspace = update_workspace(
                workspace_id,
                last_status="error",
                last_error=str(e),
            ) or workspace
            return JSONResponse({"ok": False, "error": str(e), "workspace": workspace}, status_code=502)

    async def workspace_proxy_handler(request: Request):
        """Proxy selected routes to a remote registered workspace."""
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        from .workspaces import get_workspace

        workspace_id = request.path_params.get("workspace_id", "")
        target = request.path_params.get("target", "")
        workspace = get_workspace(workspace_id)
        if workspace is None:
            return JSONResponse({"error": "workspace not found"}, status_code=404)
        denied = _authorize_workspace_access(request, workspace)
        if denied is not None:
            return denied
        if workspace.get("type") != "remote" or not workspace.get("base_url"):
            return JSONResponse({"error": "workspace is not remote"}, status_code=400)
        base_url = str(workspace.get("base_url")).rstrip("/")
        url = f"{base_url}/{target.lstrip('/')}"
        if request.url.query:
            url += f"?{request.url.query}"
        body_bytes = None if request.method == "GET" else await request.body()
        if target.endswith("chat/stream") or target == "events":
            from starlette.responses import StreamingResponse

            return StreamingResponse(
                _proxy_remote_workspace_sse(
                    request.method,
                    url,
                    body_bytes=body_bytes,
                    api_key=str(workspace.get("api_key") or ""),
                ),
                media_type="text/event-stream",
            )
        status_code, _content_type, payload = await _proxy_remote_workspace_request(
            request.method,
            url,
            body_bytes=body_bytes,
            api_key=str(workspace.get("api_key") or ""),
        )
        return JSONResponse(payload, status_code=status_code)

    async def session_checkpoints_handler(request: Request):
        """GET /sessions/{id}/checkpoints: list checkpoints."""
        if request.method != "GET":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        from .checkpoints import list_checkpoints
        from .sessions import get_session_info

        session_id = request.path_params.get("session_id", "")
        denied = _authorize_session_access(request, get_session_info(session_id))
        if denied is not None:
            return denied
        try:
            limit = int(request.query_params.get("limit") or 20)
        except ValueError:
            limit = 20
        return JSONResponse({"checkpoints": list_checkpoints(session_id, limit=limit)})

    async def session_restore_checkpoint_handler(request: Request):
        """POST /sessions/{id}/rewind: restore a checkpoint."""
        if request.method != "POST":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid json"}, status_code=400)
        from .checkpoints import restore_checkpoint

        checkpoint_id = str(body.get("checkpointID") or "").strip()
        if not checkpoint_id:
            return JSONResponse({"error": "checkpointID required"}, status_code=400)
        try:
            workspace_override = body.get("workspaceRoot")
            modified = restore_checkpoint(
                checkpoint_id,
                str(workspace_override).strip()
                if isinstance(workspace_override, str) and workspace_override.strip()
                else None,
            )
            return JSONResponse({"modified": modified})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    async def checkpoint_files_handler(request: Request):
        """GET /checkpoints/{id}/files: inspect checkpoint file snapshots."""
        if request.method != "GET":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        from .checkpoints import get_checkpoint_files

        checkpoint_id = request.path_params.get("checkpoint_id", "")
        return JSONResponse({"files": get_checkpoint_files(checkpoint_id)})

    async def checkpoint_get_handler(request: Request):
        """GET /checkpoints/{id}: return checkpoint metadata."""
        if request.method != "GET":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        from .checkpoints import get_checkpoint_info

        checkpoint_id = request.path_params.get("checkpoint_id", "")
        info = get_checkpoint_info(checkpoint_id)
        if info is None:
            return JSONResponse({"error": "checkpoint not found"}, status_code=404)
        return JSONResponse({"checkpoint": info})

    async def checkpoint_diff_handler(request: Request):
        """GET /checkpoints/{id}/diff: inspect checkpoint diff preview."""
        if request.method != "GET":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        from .checkpoints import get_checkpoint_diff

        checkpoint_id = request.path_params.get("checkpoint_id", "")
        return JSONResponse({"diff": get_checkpoint_diff(checkpoint_id)})

    async def workspace_info_handler(request: Request):
        """GET /workspace: return workspace metadata."""
        if request.method != "GET":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        from .sessions import list_sessions

        rows = list_sessions(limit=1000, workspace_root=root)
        return JSONResponse(
            {
                "workspaceRoot": root,
                "sessionCount": len(rows),
                "hasMcpSession": getattr(request.app.state, "session", None) is not None,
                "subagents": [
                    item.get("name")
                    for item in ((merged_config or {}).get("subagents") or [])
                    if isinstance(item, dict)
                ],
            }
        )

    async def principals_list_handler(request: Request):
        """GET /principals: list control-plane principals."""
        if request.method != "GET":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        denied = _authorize_admin(request)
        if denied is not None:
            return denied
        from .auth_registry import list_principals

        return JSONResponse({"principals": list_principals()})

    async def principal_get_handler(request: Request):
        """GET /principals/{id}: fetch a principal."""
        if request.method != "GET":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        denied = _authorize_admin(request)
        if denied is not None:
            return denied
        from .auth_registry import get_principal

        principal_id = request.path_params.get("principal_id", "")
        principal = get_principal(principal_id)
        if principal is None:
            return JSONResponse({"error": "principal not found"}, status_code=404)
        return JSONResponse({"principal": principal})

    async def principal_create_handler(request: Request):
        """POST /principals: create a principal/token."""
        if request.method != "POST":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        denied = _authorize_admin(request)
        if denied is not None:
            return denied
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid json"}, status_code=400)
        from .auth_registry import create_principal

        name = str(body.get("name") or "").strip()
        if not name:
            return JSONResponse({"error": "name required"}, status_code=400)
        principal = create_principal(
            name=name,
            role=str(body.get("role") or "admin"),
            api_key=str(body.get("apiKey") or ""),
            workspace_ids=body.get("workspaceIDs") if isinstance(body.get("workspaceIDs"), list) else None,
        )
        return JSONResponse({"principal": principal})

    async def principal_update_handler(request: Request):
        """PATCH /principals/{id}: update a principal."""
        if request.method != "PATCH":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        denied = _authorize_admin(request)
        if denied is not None:
            return denied
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid json"}, status_code=400)
        from .auth_registry import update_principal

        principal_id = request.path_params.get("principal_id", "")
        principal = update_principal(
            principal_id,
            name=body.get("name"),
            role=body.get("role"),
            api_key=body.get("apiKey"),
            workspace_ids=body.get("workspaceIDs") if isinstance(body.get("workspaceIDs"), list) else None,
        )
        if principal is None:
            return JSONResponse({"error": "principal not found"}, status_code=404)
        return JSONResponse({"principal": principal})

    async def principal_delete_handler(request: Request):
        """DELETE /principals/{id}: delete a principal."""
        if request.method != "DELETE":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        denied = _authorize_admin(request)
        if denied is not None:
            return denied
        from .auth_registry import delete_principal

        principal_id = request.path_params.get("principal_id", "")
        deleted = delete_principal(principal_id)
        if not deleted:
            return JSONResponse({"error": "principal not found"}, status_code=404)
        return JSONResponse({"deleted": True})

    async def fleet_summary_handler(request: Request):
        """GET /fleet: summarize local control-plane fleet state."""
        if request.method != "GET":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        if api_key:
            err = _check_api_key(request, api_key)
            if err is not None:
                return err
        from .workspaces import list_workspaces
        from .fleet import collect_fleet_snapshot

        snapshot = await collect_fleet_snapshot(list_workspaces())
        return JSONResponse(snapshot)

    app = Starlette(
        routes=[
            Route("/health", health_handler, methods=["GET"]),
            Route("/events", events_handler, methods=["GET"]),
            Route("/events/recent", recent_events_handler, methods=["GET"]),
            Route("/app", web_app_handler, methods=["GET"]),
            Route("/app.css", web_asset_handler, methods=["GET"], name="app_css"),
            Route("/app.js", web_asset_handler, methods=["GET"], name="app_js"),
            Route("/chat", chat, methods=["POST"]),
            Route("/chat/continue", chat_continue, methods=["POST"]),
            Route("/chat/stream", chat_stream, methods=["POST"]),
            Route("/sessions", sessions_list_handler, methods=["GET"]),
            Route("/sessions", session_create_handler, methods=["POST"]),
            Route("/sessions/import", session_import_handler, methods=["POST"]),
            Route("/sessions/{session_id:str}", session_get_handler, methods=["GET"]),
            Route("/sessions/{session_id:str}", session_update_handler, methods=["PATCH"]),
            Route("/sessions/{session_id:str}/messages", session_messages_handler, methods=["GET"]),
            Route("/sessions/{session_id:str}/children", session_children_handler, methods=["GET"]),
            Route("/sessions/{session_id:str}/ancestors", session_ancestors_handler, methods=["GET"]),
            Route("/sessions/{session_id:str}/timeline", session_timeline_handler, methods=["GET"]),
            Route("/sessions/{session_id:str}", session_delete_handler, methods=["DELETE"]),
            Route("/sessions/{session_id:str}/export", session_export_handler, methods=["GET"]),
            Route("/sessions/{session_id:str}/todos", session_todo_handler, methods=["GET"]),
            Route("/sessions/{session_id:str}/branch", session_branch_handler, methods=["POST"]),
            Route("/sessions/{session_id:str}/fork", session_fork_handler, methods=["POST"]),
            Route("/sessions/{session_id:str}/checkpoints", session_checkpoints_handler, methods=["GET"]),
            Route("/sessions/{session_id:str}/rewind", session_restore_checkpoint_handler, methods=["POST"]),
            Route("/checkpoints/{checkpoint_id:str}", checkpoint_get_handler, methods=["GET"]),
            Route("/checkpoints/{checkpoint_id:str}/files", checkpoint_files_handler, methods=["GET"]),
            Route("/checkpoints/{checkpoint_id:str}/diff", checkpoint_diff_handler, methods=["GET"]),
            Route("/workspace", workspace_info_handler, methods=["GET"]),
            Route("/fleet", fleet_summary_handler, methods=["GET"]),
            Route("/principals", principals_list_handler, methods=["GET"]),
            Route("/principals", principal_create_handler, methods=["POST"]),
            Route("/principals/{principal_id:str}", principal_get_handler, methods=["GET"]),
            Route("/principals/{principal_id:str}", principal_update_handler, methods=["PATCH"]),
            Route("/principals/{principal_id:str}", principal_delete_handler, methods=["DELETE"]),
            Route("/workspaces", workspaces_list_handler, methods=["GET"]),
            Route("/workspaces", workspace_create_handler, methods=["POST"]),
            Route("/workspaces/{workspace_id:str}", workspace_get_handler, methods=["GET"]),
            Route("/workspaces/{workspace_id:str}", workspace_update_handler, methods=["PATCH"]),
            Route("/workspaces/{workspace_id:str}", workspace_delete_handler, methods=["DELETE"]),
            Route("/workspaces/{workspace_id:str}/health", workspace_health_handler, methods=["GET"]),
            Route("/workspaces/{workspace_id:str}/proxy/{target:path}", workspace_proxy_handler, methods=["GET", "POST", "DELETE", "PATCH", "PUT"]),
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
