"""E2E-style HTTP control-plane flow against the in-process Starlette app."""

from __future__ import annotations

import json

import pytest

starlette = pytest.importorskip("starlette")
httpx = pytest.importorskip("httpx")
from httpx import ASGITransport  # type: ignore  # noqa: E402

import ollamacode.serve as serve  # noqa: E402


@pytest.mark.asyncio
async def test_http_control_plane_flow(tmp_path, monkeypatch):
    """Create workspace and session, branch/export, then run an interactive question flow."""
    monkeypatch.setattr("ollamacode.sessions._DB_PATH", tmp_path / "sessions.db")
    monkeypatch.setattr("ollamacode.workspaces._WORKSPACES_PATH", tmp_path / "workspaces.json")

    async def fake_run_agent_loop(*args, before_tool_call=None, **kwargs):
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
        return f"done {decision[1]}"

    monkeypatch.setattr(serve, "run_agent_loop", fake_run_agent_loop)

    app = serve.create_app(
        model="model",
        mcp_servers=[],
        system_extra="",
        workspace_root=str(tmp_path),
        merged_config={"subagents": [{"name": "reviewer"}]},
    )
    app.state.session = object()
    transport = ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        workspace = await client.post(
            "/workspaces",
            json={"name": "Remote Dev", "type": "remote", "baseUrl": "http://localhost:9000"},
        )
        assert workspace.status_code == 200
        workspace_id = workspace.json()["workspace"]["id"]

        session = await client.post(
            "/sessions",
            json={"title": "Flow Session", "workspaceRoot": str(tmp_path)},
        )
        assert session.status_code == 200
        session_id = session.json()["session"]["id"]

        branched = await client.post(f"/sessions/{session_id}/branch", json={})
        assert branched.status_code == 200
        branched_id = branched.json()["session"]["id"]
        assert branched_id != session_id

        exported = await client.get(f"/sessions/{branched_id}/export")
        assert exported.status_code == 200
        assert json.loads(exported.json()["data"])["session"]["title"]

        chat = await client.post("/chat", json={"message": "ask first", "sessionID": branched_id})
        assert chat.status_code == 200
        token = chat.json()["approvalToken"]
        assert chat.json()["questionRequired"]["questions"][0]["question"] == "Which file should I edit?"

        continued = await client.post(
            "/chat/continue",
            json={"approvalToken": token, "answers": ["cli.py"]},
        )
        assert continued.status_code == 200
        assert "cli.py" in continued.json()["content"]

        proxied = await client.get("/workspace")
        assert proxied.status_code == 200
        assert proxied.json()["workspaceRoot"] == str(tmp_path)

        workspaces = await client.get("/workspaces")
        assert any(item["id"] == workspace_id for item in workspaces.json()["workspaces"])
