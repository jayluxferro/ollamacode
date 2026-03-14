"""Unit tests for serve HTTP API (Starlette)."""

import asyncio
import json

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


@pytest.mark.asyncio
async def test_http_continue_always_persists_tool_approval(monkeypatch):
    """HTTP `always` approval should auto-allow the same tool on the next request."""
    calls = {"count": 0}

    async def fake_run_agent_loop(*args, before_tool_call=None, **kwargs):
        assert before_tool_call is not None
        calls["count"] += 1
        decision = await before_tool_call("run_command", {"command": "echo hi"})
        return f"done {decision}"

    monkeypatch.setattr(serve, "run_agent_loop", fake_run_agent_loop)

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
            json={"message": "first", "confirmToolCalls": True, "sessionID": "serve-session"},
        )
        token = res.json()["approvalToken"]
        res2 = await client.post(
            "/chat/continue",
            json={"approvalToken": token, "decision": "always"},
        )
        assert res2.json()["content"] == "done run"

        res3 = await client.post(
            "/chat",
            json={"message": "second", "confirmToolCalls": True, "sessionID": "serve-session"},
        )
        assert res3.json()["content"] == "done run"


@pytest.mark.asyncio
async def test_http_session_routes(tmp_path, monkeypatch):
    """HTTP session/workspace routes should expose the local session store."""
    monkeypatch.setattr("ollamacode.sessions._DB_PATH", tmp_path / "sessions.db")
    monkeypatch.setattr("ollamacode.workspaces._WORKSPACES_PATH", tmp_path / "workspaces.json")

    app = serve.create_app(
        model="model",
        mcp_servers=[],
        system_extra="",
        workspace_root=str(tmp_path),
        merged_config={"subagents": [{"name": "reviewer"}]},
    )
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/sessions",
            json={"title": "HTTP Session", "workspaceRoot": str(tmp_path), "owner": "alice", "role": "editor"},
        )
        session = created.json()["session"]
        session_id = session["id"]
        assert session["owner"] == "alice"

        renamed = await client.patch(
            f"/sessions/{session_id}",
            json={"title": "HTTP Session Renamed"},
        )
        assert renamed.json()["session"]["title"] == "HTTP Session Renamed"

        listed = await client.get("/sessions", params={"workspaceRoot": str(tmp_path)})
        assert any(row["id"] == session_id for row in listed.json()["sessions"])

        messages = await client.get(f"/sessions/{session_id}/messages")
        assert isinstance(messages.json()["messages"], list)

        exported = await client.get(f"/sessions/{session_id}/export")
        assert json.loads(exported.json()["data"])["session"]["title"] == "HTTP Session Renamed"

        branched = await client.post(f"/sessions/{session_id}/branch", json={})
        assert branched.json()["session"]["id"] != session_id
        children = await client.get(f"/sessions/{session_id}/children")
        assert any(
            row["id"] == branched.json()["session"]["id"]
            for row in children.json()["sessions"]
        )
        ancestors = await client.get(
            f"/sessions/{branched.json()['session']['id']}/ancestors"
        )
        assert ancestors.json()["sessions"][0]["id"] == session_id
        timeline = await client.get(
            f"/sessions/{branched.json()['session']['id']}/timeline"
        )
        assert timeline.json()["timeline"]["session"]["id"] == branched.json()["session"]["id"]

        from ollamacode.checkpoints import CheckpointRecorder

        note = tmp_path / "note.txt"
        note.write_text("before")
        recorder = CheckpointRecorder(
            session_id=branched.json()["session"]["id"],
            workspace_root=str(tmp_path),
            prompt="edit",
            message_index=0,
        )
        recorder.record_pre("note.txt")
        note.write_text("after")
        checkpoint_id = recorder.finalize()
        checkpoint_files = await client.get(f"/checkpoints/{checkpoint_id}/files")
        assert checkpoint_files.json()["files"][0]["path"] == "note.txt"
        checkpoint_info = await client.get(f"/checkpoints/{checkpoint_id}")
        assert checkpoint_info.json()["checkpoint"]["id"] == checkpoint_id
        checkpoint_diff = await client.get(f"/checkpoints/{checkpoint_id}/diff")
        assert "--- a/note.txt" in checkpoint_diff.json()["diff"]

        deleted = await client.delete(f"/sessions/{session_id}")
        assert deleted.json()["deleted"] is True

        workspace_created = await client.post(
            "/workspaces",
            json={"name": "Remote", "type": "remote", "baseUrl": "http://localhost:9000"},
        )
        workspace_id = workspace_created.json()["workspace"]["id"]
        workspace_list = await client.get("/workspaces")
        assert any(item["id"] == workspace_id for item in workspace_list.json()["workspaces"])
        workspace_update = await client.patch(
            f"/workspaces/{workspace_id}",
            json={"name": "Remote-2"},
        )
        assert workspace_update.json()["workspace"]["name"] == "Remote-2"

        workspace = await client.get("/workspace")
        assert workspace.json()["workspaceRoot"] == str(tmp_path)
        assert "reviewer" in workspace.json()["subagents"]


@pytest.mark.asyncio
async def test_http_question_continue_flow(monkeypatch):
    """HTTP /chat should surface interactive question prompts and resume via /chat/continue."""

    async def fake_run_agent_loop(*args, before_tool_call=None, **kwargs):
        assert before_tool_call is not None
        result = await before_tool_call(
            "question",
            {
                "questions": [
                    {
                        "header": "Scope",
                        "question": "Which file should I edit?",
                        "options": [{"label": "app.py"}, {"label": "cli.py"}],
                    }
                ]
            },
        )
        return f"done {result[1]}"

    monkeypatch.setattr(serve, "run_agent_loop", fake_run_agent_loop)

    app = serve.create_app(model="model", mcp_servers=[], system_extra="")
    app.state.session = object()
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post("/chat", json={"message": "ask first"})
        assert first.status_code == 200
        token = first.json()["approvalToken"]
        assert first.json()["questionRequired"]["questions"][0]["question"] == "Which file should I edit?"
        second = await client.post(
            "/chat/continue",
            json={"approvalToken": token, "answers": ["cli.py"]},
        )
        assert "cli.py" in second.json()["content"]


@pytest.mark.asyncio
async def test_http_task_delegation(monkeypatch):
    """HTTP /chat should delegate `task` tool calls through the shared task runtime."""

    async def fake_run_agent_loop(*args, before_tool_call=None, **kwargs):
        assert before_tool_call is not None
        result = await before_tool_call(
            "task",
            {
                "description": "Review code",
                "prompt": "Review the latest patch.",
                "subagent_type": "reviewer",
            },
        )
        return f"done {result[1]}"

    async def fake_run_task_delegation(**kwargs):
        return "task_id: child-http\n\n<task_result>\nhttp subagent output\n</task_result>"

    monkeypatch.setattr(serve, "run_agent_loop", fake_run_agent_loop)
    monkeypatch.setattr(serve, "run_task_delegation", fake_run_task_delegation)

    app = serve.create_app(
        model="model",
        mcp_servers=[],
        system_extra="",
        merged_config={"subagents": [{"name": "reviewer", "tools": ["read_file"]}]},
    )
    app.state.session = object()
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/chat", json={"message": "delegate this"})
        assert "http subagent output" in res.json()["content"]


@pytest.mark.asyncio
async def test_workspace_proxy_route(tmp_path, monkeypatch):
    monkeypatch.setattr("ollamacode.workspaces._WORKSPACES_PATH", tmp_path / "workspaces.json")
    async def fake_proxy(method, url, *, body_bytes, api_key=""):
        assert method == "GET"
        assert url == "http://localhost:9000/workspace"
        return 200, "application/json", {"ok": True}

    monkeypatch.setattr(serve, "_proxy_remote_workspace_request", fake_proxy)

    app = serve.create_app(model="model", mcp_servers=[], system_extra="")
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/workspaces",
            json={"name": "Remote", "type": "remote", "baseUrl": "http://localhost:9000"},
        )
        workspace_id = created.json()["workspace"]["id"]
        proxied = await client.get(f"/workspaces/{workspace_id}/proxy/workspace")
        assert proxied.json()["ok"] is True


@pytest.mark.asyncio
async def test_browser_app_routes():
    app = serve.create_app(model="model", mcp_servers=[], system_extra="")
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        page = await client.get("/app")
        assert page.status_code == 200
        assert "OllamaCode Control Plane" in page.text
        js = await client.get("/app.js")
        assert js.status_code == 200
        assert "loadWorkspaces" in js.text
        css = await client.get("/app.css")
        assert css.status_code == 200
        assert "--accent" in css.text


@pytest.mark.asyncio
async def test_http_chat_stream_endpoint(monkeypatch):
    async def fake_stream(*args, **kwargs):
        yield "hel"
        yield "lo"

    monkeypatch.setattr(serve, "run_agent_loop_stream", fake_stream)

    app = serve.create_app(model="model", mcp_servers=[], system_extra="")
    app.state.session = object()
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/chat/stream", json={"message": "hi"})
        assert res.status_code == 200
        assert '"type": "chunk"' in res.text
        assert '"type": "done"' in res.text


@pytest.mark.asyncio
async def test_http_principals_and_fleet_routes(tmp_path, monkeypatch):
    monkeypatch.setattr("ollamacode.auth_registry._AUTH_PATH", tmp_path / "principals.json")
    monkeypatch.setattr("ollamacode.workspaces._WORKSPACES_PATH", tmp_path / "workspaces.json")
    async def fake_snapshot(workspaces):
        return {"total": len(workspaces), "remote": 0, "healthy": len(workspaces), "unhealthy": 0, "workspaces": workspaces}
    monkeypatch.setattr("ollamacode.fleet.collect_fleet_snapshot", fake_snapshot)
    app = serve.create_app(model="model", mcp_servers=[], system_extra="")
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post("/principals", json={"name": "Alice", "role": "admin"})
        assert created.json()["principal"]["name"] == "Alice"
        principal_id = created.json()["principal"]["id"]
        updated = await client.patch(
            f"/principals/{principal_id}",
            json={"workspaceIDs": ["w1", "w2"]},
        )
        assert updated.json()["principal"]["workspace_ids"] == ["w1", "w2"]
        fetched = await client.get(f"/principals/{principal_id}")
        assert fetched.json()["principal"]["workspace_ids"] == ["w1", "w2"]
        principals = await client.get("/principals")
        assert any(item["name"] == "Alice" for item in principals.json()["principals"])
        fleet = await client.get("/fleet")
        assert "total" in fleet.json()


@pytest.mark.asyncio
async def test_http_authz_principal_admin_and_scope(tmp_path, monkeypatch):
    monkeypatch.setattr("ollamacode.auth_registry._AUTH_PATH", tmp_path / "principals.json")
    monkeypatch.setattr("ollamacode.workspaces._WORKSPACES_PATH", tmp_path / "workspaces.json")
    monkeypatch.setattr("ollamacode.sessions._DB_PATH", tmp_path / "sessions.db")

    from ollamacode.auth_registry import create_principal
    from ollamacode.workspaces import create_workspace
    from ollamacode.sessions import create_session

    admin = create_principal(name="Admin", role="admin", api_key="admin-token")
    viewer = create_principal(name="Viewer", role="viewer", api_key="viewer-token", workspace_ids=["allowed-workspace"])
    create_workspace(name="Owned", kind="remote", base_url="http://x", owner="Viewer", role="owner")
    create_workspace(name="Other", kind="remote", base_url="http://y", owner="SomeoneElse", role="owner")
    create_session("Owned Session", workspace_root=str(tmp_path), owner="Viewer", role="owner")
    create_session("Other Session", workspace_root=str(tmp_path), owner="SomeoneElse", role="owner")

    app = serve.create_app(model="model", mcp_servers=[], system_extra="", api_key="legacy")
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        forbidden = await client.get("/principals", headers={"Authorization": "Bearer viewer-token"})
        assert forbidden.status_code == 403

        allowed = await client.get("/principals", headers={"Authorization": "Bearer admin-token"})
        assert allowed.status_code == 200
        assert any(item["name"] == "Viewer" for item in allowed.json()["principals"])

        workspaces = await client.get("/workspaces", headers={"Authorization": "Bearer viewer-token"})
        assert workspaces.status_code == 200
        assert all(item["owner"] == "Viewer" for item in workspaces.json()["workspaces"])

        sessions = await client.get("/sessions", headers={"Authorization": "Bearer viewer-token"})
        assert sessions.status_code == 200
        assert all((item.get("owner") or "") in ("", "Viewer") for item in sessions.json()["sessions"])


@pytest.mark.asyncio
async def test_http_chat_stream_question_event(monkeypatch):
    async def fake_stream(*args, before_tool_call=None, **kwargs):
        assert before_tool_call is not None
        decision = await before_tool_call(
            "question",
            {
                "questions": [
                    {
                        "header": "Scope",
                        "question": "Which file should I edit?",
                        "options": [{"label": "app.py"}, {"label": "cli.py"}],
                    }
                ]
            },
        )
        yield f"done {decision[1]}"

    monkeypatch.setattr(serve, "run_agent_loop_stream", fake_stream)

    app = serve.create_app(model="model", mcp_servers=[], system_extra="")
    app.state.session = object()
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/chat/stream", json={"message": "hi"})
        assert res.status_code == 200
        assert '"type": "question"' in res.text
        assert '"approvalToken"' in res.text


def test_http_events_route_registered():
    app = serve.create_app(model="model", mcp_servers=[], system_extra="")
    paths = {route.path for route in app.routes}
    assert "/events" in paths
    assert "/events/recent" in paths
