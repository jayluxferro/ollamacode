"""
Optional local HTTP API for OllamaCode. Run: ollamacode serve --port 8000
Requires: pip install ollamacode[server]
Endpoints: POST /chat with JSON {"message": "..."} returns {"content": "..."}.
"""

from __future__ import annotations

import contextlib
import os

from .agent import run_agent_loop, run_agent_loop_no_mcp
from .config import load_config, merge_config_with_env
from .mcp_client import McpConnection, connect_mcp_servers, connect_mcp_stdio


async def _handle_chat(
    session: McpConnection | None,
    model: str,
    system_extra: str,
    body: dict,
    max_messages: int,
) -> dict:
    message = (body.get("message") or "").strip()
    if not message:
        return {"content": "", "error": "message required"}
    model_override = body.get("model")
    use_model = model_override or model
    system = "You are a helpful coding assistant. Use the available tools when they would help."
    if system_extra:
        system = system + "\n\n" + system_extra
    try:
        if session is not None:
            out = await run_agent_loop(
                session, use_model, message, system_prompt=system, max_messages=max_messages
            )
        else:
            out = await run_agent_loop_no_mcp(use_model, message, system_prompt=system)
        return {"content": out}
    except Exception as e:
        return {"content": "", "error": str(e)}


def create_app(
    model: str,
    mcp_servers: list[dict],
    system_extra: str,
    max_messages: int = 0,
):
    """Create ASGI app (Starlette) with MCP session in lifespan."""
    try:
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse
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

    async def chat(request: Request) -> JSONResponse:
        if request.method != "POST":
            return JSONResponse({"error": "method not allowed"}, status_code=405)
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid json"}, status_code=400)
        session: McpConnection | None = getattr(request.app.state, "session", None)
        result = await _handle_chat(
            session, model, system_extra, body, max_messages
        )
        return JSONResponse(result)

    app = Starlette(
        routes=[Route("/chat", chat, methods=["POST"])],
        lifespan=lifespan,
    )
    return app


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
    model = merged.get("model") or os.environ.get(
        "OLLAMACODE_MODEL", "gpt-oss:20b"
    )
    system_extra = (merged.get("system_prompt_extra") or "").strip()
    mcp_servers = merged.get("mcp_servers") or []
    max_messages = merged.get("max_messages", 0)

    app = create_app(model, mcp_servers, system_extra, max_messages)
    uvicorn.run(app, host="127.0.0.1", port=port)
