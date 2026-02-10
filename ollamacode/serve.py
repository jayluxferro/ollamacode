"""
Optional local HTTP API for OllamaCode. Run: ollamacode serve --port 8000
Requires: pip install ollamacode[server]
Endpoints: POST /chat, POST /chat/stream (see docs/STRUCTURED_PROTOCOL.md), POST /apply-edits.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import AsyncIterator
from typing import Any

# Removed unused import of Request
from starlette.responses import JSONResponse

from .agent import (
    run_agent_loop,
    run_agent_loop_no_mcp,
    run_agent_loop_no_mcp_stream,
    run_agent_loop_stream,
)
from .config import load_config, merge_config_with_env
from .context import prepend_file_context
from .edits import apply_edits, parse_edits
from .mcp_client import McpConnection, connect_mcp_servers, connect_mcp_stdio
from .completions import get_completion
from .diagnostics import get_diagnostics
from .health import check_ollama
from .protocol import normalize_chat_body
from .skills import load_skills_text
from .state import get_state, format_recent_context
from .templates import load_prompt_template


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
) -> dict:
    message, file_path, lines_spec = normalize_chat_body(body)
    if not message:
        return {"content": "", "error": "message required"}
    if file_path:
        message = prepend_file_context(
            message, str(file_path), workspace_root, lines_spec
        )
    model_override = body.get("model")
    use_model = model_override or model
    system = (
        "You are a coding assistant with full access to the workspace. You are given a list of available tools with their names "
        "and descriptions—use whichever tools fit the task. When the user asks you to run something, check something, or change "
        "something, use the appropriate tool and report the result."
    )
    if system_extra:
        system = system + "\n\n" + system_extra
    if use_skills:
        skills_text = load_skills_text(workspace_root)
        if skills_text:
            system = (
                system
                + "\n\n--- Skills (saved instructions & memory) ---\n\n"
                + skills_text
            )
    if prompt_template:
        template_text = load_prompt_template(prompt_template, workspace_root)
        if template_text:
            system = system + "\n\n--- Prompt template ---\n\n" + template_text
    if inject_recent_context:
        state = get_state()
        block = format_recent_context(state, max_files=recent_context_max_files)
        if block:
            system = system + "\n\n--- Recent context ---\n\n" + block
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
        edits = parse_edits(out)
        if edits:
            result["edits"] = edits
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


# create_app remains largely unchanged, but Request type hint will be Any


def create_app(
    model: str,
    mcp_servers: list[dict],
    system_extra: str,
    max_messages: int = 0,
    max_tool_result_chars: int = 0,
    workspace_root: str | None = None,
    api_key: str | None = None,
    use_skills: bool = True,
    prompt_template: str | None = None,
    inject_recent_context: bool = True,
    recent_context_max_files: int = 10,
):
    """Create ASGI app (Starlette) with MCP session in lifespan. If api_key is set, requests must send Authorization: Bearer <key> or X-API-Key: <key>."""
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
        try:
            yield
        finally:
            if ctx is not None:
                await ctx.__aexit__(None, None, None)

    async def health_handler(request: Request) -> JSONResponse:
        """GET /health: verify Ollama (and optionally MCP). No auth required."""
        ok, msg = check_ollama()
        return JSONResponse({"ollama": ok, "message": msg})

    async def chat(request: Request) -> JSONResponse:
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
        session: McpConnection | None = getattr(request.app.state, "session", None)
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
        )
        return JSONResponse(result)

    async def chat_stream(request: Request):
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
        message, file_path, lines_spec = normalize_chat_body(body)
        if not message:
            return JSONResponse({"error": "message required"}, status_code=400)
        if file_path:
            message = prepend_file_context(message, str(file_path), root, lines_spec)
        model_override = body.get("model")
        use_model = model_override or model
        system = (
            "You are a coding assistant with full access to the workspace. You are given a list of available tools with their names "
            "and descriptions—use whichever tools fit the task. When the user asks you to run something, check something, or change "
            "something, use the appropriate tool and report the result."
        )
        if system_extra:
            system = system + "\n\n" + system_extra
        if use_skills:
            skills_text = load_skills_text(root)
            if skills_text:
                system = (
                    system
                    + "\n\n--- Skills (saved instructions & memory) ---\n\n"
                    + skills_text
                )
        if prompt_template:
            template_text = load_prompt_template(prompt_template, root)
            if template_text:
                system = system + "\n\n--- Prompt template ---\n\n" + template_text
        if inject_recent_context:
            state = get_state()
            block = format_recent_context(state, max_files=recent_context_max_files)
            if block:
                system = system + "\n\n--- Recent context ---\n\n" + block

        async def generate() -> AsyncIterator[str]:
            session: McpConnection | None = getattr(request.app.state, "session", None)
            accumulated: list[str] = []
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
                    )
                else:
                    stream = run_agent_loop_no_mcp_stream(
                        use_model,
                        message,
                        system_prompt=system,
                        message_history=[],
                    )
                async for chunk in stream:
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

    app = Starlette(
        routes=[
            Route("/health", health_handler, methods=["GET"]),
            Route("/chat", chat, methods=["POST"]),
            Route("/chat/stream", chat_stream, methods=["POST"]),
            Route("/apply-edits", apply_edits_handler, methods=["POST"]),
            Route("/diagnostics", diagnostics_handler, methods=["POST"]),
            Route("/complete", complete_handler, methods=["POST"]),
        ],
        lifespan=lifespan,
    )
    return app


# run_serve unchanged except for imports


def run_serve(port: int = 8000, config_path: str | None = None) -> None:
    """Load config, create app, run uvicorn."""
    try:
        import uvicorn
    except ImportError as e:
        raise SystemExit(
            "Server requires uvicorn. Install with: pip install ollamacode[server]"
        ) from e

    config = load_config(config_path)
    merged = merge_config_with_env(
        config,
        model_env=os.environ.get("OLLAMACODE_MODEL"),
        mcp_args_env=os.environ.get("OLLAMACODE_MCP_ARGS"),
        system_extra_env=os.environ.get("OLLAMACODE_SYSTEM_EXTRA"),
    )
    model = merged.get("model") or os.environ.get("OLLAMACODE_MODEL", "gpt-oss:20b")
    system_extra = (merged.get("system_prompt_extra") or "").strip()
    mcp_servers = merged.get("mcp_servers") or []
    max_messages = merged.get("max_messages", 0)
    max_tool_result_chars = merged.get("max_tool_result_chars", 0)
    workspace_root = os.getcwd()
    serve_config = merged.get("serve") or {}
    api_key = (
        serve_config.get("api_key") or os.environ.get("OLLAMACODE_SERVE_API_KEY") or ""
    ).strip() or None

    app = create_app(
        model,
        mcp_servers,
        system_extra,
        max_messages,
        max_tool_result_chars,
        workspace_root,
        api_key=api_key,
        use_skills=merged.get("use_skills", True),
        prompt_template=merged.get("prompt_template"),
        inject_recent_context=merged.get("inject_recent_context", True),
        recent_context_max_files=merged.get("recent_context_max_files", 10),
    )
    uvicorn.run(app, host="127.0.0.1", port=port)
