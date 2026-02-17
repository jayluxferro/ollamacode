"""Unit tests for serve HTTP API (Starlette)."""

import pytest

from ollamacode.multi_agent import MultiAgentResult

starlette = pytest.importorskip("starlette")
httpx = pytest.importorskip("httpx")
from httpx import ASGITransport  # type: ignore  # noqa: E402

import ollamacode.serve as serve  # noqa: E402


@pytest.mark.asyncio
async def test_multi_agent_confirm_flow_http(monkeypatch):
    """multiAgent + confirmToolCalls returns approval token, then completes on /chat/continue."""

    async def fake_run_multi_agent(*args, before_tool_call=None, **kwargs):
        assert before_tool_call is not None
        decision = await before_tool_call("run_command", {"command": "echo hi"})
        return MultiAgentResult(
            content=f"done {decision}",
            plan="plan",
            review={"approved": True},
        )

    monkeypatch.setattr(serve, "run_multi_agent", fake_run_multi_agent)

    app = serve.create_app(
        model="model",
        mcp_servers=[],
        system_extra="",
        confirm_tool_calls=True,
    )
    # Force a session so confirm flow is active
    app.state.session = object()

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/chat",
            json={"message": "do it", "multiAgent": True, "confirmToolCalls": True},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["toolApprovalRequired"]["tool"] == "run_command"
        token = data["approvalToken"]

        res2 = await client.post(
            "/chat/continue", json={"approvalToken": token, "decision": "run"}
        )
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["content"] == "done run"
        assert data2["plan"] == "plan"
        assert data2["review"]["approved"] is True


@pytest.mark.asyncio
async def test_multi_agent_confirm_flow_http_edit(monkeypatch):
    """multiAgent confirm flow supports editedArguments (decision=edit)."""

    async def fake_run_multi_agent(*args, before_tool_call=None, **kwargs):
        assert before_tool_call is not None
        decision = await before_tool_call("run_command", {"command": "echo hi"})
        return MultiAgentResult(
            content=f"done {decision}",
            plan="plan",
            review={"approved": True},
        )

    monkeypatch.setattr(serve, "run_multi_agent", fake_run_multi_agent)

    app = serve.create_app(
        model="model",
        mcp_servers=[],
        system_extra="",
        confirm_tool_calls=True,
    )
    app.state.session = object()

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/chat",
            json={"message": "do it", "multiAgent": True, "confirmToolCalls": True},
        )
        assert res.status_code == 200
        data = res.json()
        token = data["approvalToken"]

        res2 = await client.post(
            "/chat/continue",
            json={
                "approvalToken": token,
                "decision": "edit",
                "editedArguments": {"command": "echo edited"},
            },
        )
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["content"] == "done ('edit', {'command': 'echo edited'})"


@pytest.mark.asyncio
async def test_rag_endpoints_http(monkeypatch):
    """HTTP /rag/index and /rag/query return indexed and query results."""

    def fake_build_local_rag_index(
        workspace_root, max_files=400, max_chars_per_file=20000
    ):
        assert workspace_root
        return {
            "index_path": "/tmp/rag_index.json",
            "workspace_root": workspace_root,
            "indexed_files": 2,
            "chunk_count": 3,
        }

    def fake_query_local_rag(query, max_results=5):
        assert query == "jwt"
        return [
            {
                "path": "docs/AUTH.md",
                "chunk_index": 0,
                "score": 5.0,
                "snippet": "JWT tokens",
            }
        ]

    monkeypatch.setattr(serve, "build_local_rag_index", fake_build_local_rag_index)
    monkeypatch.setattr(serve, "query_local_rag", fake_query_local_rag)

    app = serve.create_app(model="model", mcp_servers=[], system_extra="")
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        idx = await client.post("/rag/index", json={"workspaceRoot": "/tmp/work"})
        assert idx.status_code == 200
        idx_data = idx.json()
        assert idx_data["indexed_files"] == 2
        q = await client.post("/rag/query", json={"query": "jwt", "maxResults": 3})
        assert q.status_code == 200
        q_data = q.json()
        assert q_data["results"][0]["path"] == "docs/AUTH.md"
