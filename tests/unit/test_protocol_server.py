"""Unit tests for stdio JSON-RPC protocol server."""

from pathlib import Path
import json

import pytest

from ollamacode.protocol_server import (
    _handle_apply_edits,
    _handle_chat_stream,
    _handle_request,
    _resolve_memory_request_settings,
)
from ollamacode.multi_agent import MultiAgentResult


def test_handle_apply_edits_missing_edits():
    """_handle_apply_edits returns error when edits is not an array."""
    out = _handle_apply_edits({}, "/tmp")
    assert out["applied"] == 0
    assert "error" in out

    out2 = _handle_apply_edits({"edits": "not array"}, "/tmp")
    assert out2["applied"] == 0
    assert "error" in out2


def test_handle_apply_edits_empty_or_invalid():
    """_handle_apply_edits returns error when no valid edits."""
    out = _handle_apply_edits({"edits": []}, "/tmp")
    assert out["applied"] == 0
    assert "error" in out

    out2 = _handle_apply_edits({"edits": [{}]}, "/tmp")
    assert out2["applied"] == 0
    assert "error" in out2


def test_handle_apply_edits_applies_file(tmp_path: Path):
    """_handle_apply_edits applies a valid edit and returns applied count."""
    f = tmp_path / "hello.txt"
    f.write_text("hello\n")
    out = _handle_apply_edits(
        {
            "edits": [
                {"path": "hello.txt", "newText": "world\n"},
            ],
        },
        str(tmp_path),
    )
    assert out["applied"] == 1
    assert f.read_text() == "world\n"


@pytest.mark.asyncio
async def test_handle_request_unknown_method():
    """_handle_request returns Method not found for unknown method."""
    req = {"jsonrpc": "2.0", "id": 42, "method": "unknown/method", "params": {}}
    res = await _handle_request(req, None, "model", "", 0, 0, "/tmp")
    assert res["jsonrpc"] == "2.0"
    assert res["id"] == 42
    assert "error" in res
    assert res["error"]["code"] == -32601
    assert "not found" in res["error"]["message"].lower()


@pytest.mark.asyncio
async def test_chat_stream_empty_message_yields_one_error():
    """_handle_chat_stream with empty message yields a single error result."""
    chunks = []
    async for r in _handle_chat_stream(
        None, "model", "", {"message": ""}, 0, 0, "/tmp", req_id=99
    ):
        chunks.append(r)
    assert len(chunks) == 1
    assert chunks[0]["jsonrpc"] == "2.0"
    assert chunks[0]["id"] == 99
    assert chunks[0]["result"]["type"] == "error"
    assert "message required" in chunks[0]["result"]["error"]


@pytest.mark.asyncio
async def test_handle_request_chat_stream_empty_message():
    """_handle_request(ollamacode/chatStream) with empty message returns stream of one error."""
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "ollamacode/chatStream",
        "params": {"message": ""},
    }
    response = await _handle_request(req, None, "model", "", 0, 0, "/tmp")
    assert hasattr(response, "__aiter__")
    parts = []
    async for p in response:
        parts.append(p)
    assert len(parts) == 1
    assert parts[0]["result"]["type"] == "error"


@pytest.mark.asyncio
async def test_multi_agent_confirm_flow(monkeypatch):
    """multiAgent with confirmToolCalls returns approval token and completes on continue."""

    async def fake_run_multi_agent(*args, before_tool_call=None, **kwargs):
        assert before_tool_call is not None
        decision = await before_tool_call("run_command", {"command": "echo hi"})
        return MultiAgentResult(
            content=f"done {decision}",
            plan="plan",
            review={"approved": True},
        )

    monkeypatch.setattr(
        "ollamacode.protocol_server.run_multi_agent", fake_run_multi_agent
    )

    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "ollamacode/chat",
        "params": {"message": "do it", "multiAgent": True},
    }
    res = await _handle_request(
        req,
        object(),
        "model",
        "",
        0,
        0,
        "/tmp",
        confirm_tool_calls=True,
    )
    assert res["result"]["toolApprovalRequired"]["tool"] == "run_command"
    token = res["result"]["approvalToken"]

    res2 = await _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "ollamacode/chatContinue",
            "params": {"approvalToken": token, "decision": "run"},
        },
        object(),
        "model",
        "",
        0,
        0,
        "/tmp",
        confirm_tool_calls=True,
    )
    assert res2["result"]["content"] == "done run"
    assert res2["result"]["plan"] == "plan"
    assert res2["result"]["review"]["approved"] is True


@pytest.mark.asyncio
async def test_question_flow_uses_chat_continue(monkeypatch):
    """Interactive question tool requests answers and resumes through chatContinue."""

    async def fake_run_agent_loop(*args, before_tool_call=None, **kwargs):
        assert before_tool_call is not None
        answer_result = await before_tool_call(
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
        return f"done {answer_result[1]}"

    monkeypatch.setattr("ollamacode.protocol_server.run_agent_loop", fake_run_agent_loop)

    req = {
        "jsonrpc": "2.0",
        "id": 21,
        "method": "ollamacode/chat",
        "params": {"message": "ask me first"},
    }
    res = await _handle_request(
        req,
        object(),
        "model",
        "",
        0,
        0,
        "/tmp",
        confirm_tool_calls=False,
    )
    assert res["result"]["questionRequired"]["questions"][0]["question"] == "Which file should I edit?"
    token = res["result"]["approvalToken"]

    res2 = await _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 22,
            "method": "ollamacode/chatContinue",
            "params": {"approvalToken": token, "answers": ["cli.py"]},
        },
        object(),
        "model",
        "",
        0,
        0,
        "/tmp",
        confirm_tool_calls=False,
    )
    assert "cli.py" in res2["result"]["content"]


@pytest.mark.asyncio
async def test_task_flow_delegates_to_subagent_runtime(monkeypatch):
    """Task tool should delegate through the shared task runtime."""

    async def fake_run_agent_loop(*args, before_tool_call=None, **kwargs):
        assert before_tool_call is not None
        task_result = await before_tool_call(
            "task",
            {
                "description": "Review code",
                "prompt": "Review the patch for issues.",
                "subagent_type": "reviewer",
            },
        )
        return f"done {task_result[1]}"

    async def fake_run_task_delegation(**kwargs):
        return "task_id: child-1\n\n<task_result>\nsubagent output\n</task_result>"

    monkeypatch.setattr("ollamacode.protocol_server.run_agent_loop", fake_run_agent_loop)
    monkeypatch.setattr("ollamacode.protocol_server.run_task_delegation", fake_run_task_delegation)

    req = {
        "jsonrpc": "2.0",
        "id": 31,
        "method": "ollamacode/chat",
        "params": {"message": "delegate this"},
    }
    res = await _handle_request(
        req,
        object(),
        "model",
        "",
        0,
        0,
        "/tmp",
        subagents=[{"name": "reviewer", "tools": ["read_file"]}],
    )
    assert "subagent output" in res["result"]["content"]


@pytest.mark.asyncio
async def test_protocol_continue_always_persists_tool_approval(monkeypatch):
    """`always` approval should auto-allow the same tool on the next turn."""
    calls = {"count": 0}

    async def fake_run_agent_loop(*args, before_tool_call=None, **kwargs):
        assert before_tool_call is not None
        calls["count"] += 1
        decision = await before_tool_call("run_command", {"command": "echo hi"})
        return f"done {decision}"

    monkeypatch.setattr("ollamacode.protocol_server.run_agent_loop", fake_run_agent_loop)

    req = {
        "jsonrpc": "2.0",
        "id": 41,
        "method": "ollamacode/chat",
        "params": {"message": "first", "sessionID": "protocol-session"},
    }
    res = await _handle_request(
        req, object(), "model", "", 0, 0, "/tmp", confirm_tool_calls=True
    )
    token = res["result"]["approvalToken"]
    res2 = await _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 42,
            "method": "ollamacode/chatContinue",
            "params": {"approvalToken": token, "decision": "always"},
        },
        object(),
        "model",
        "",
        0,
        0,
        "/tmp",
        confirm_tool_calls=True,
    )
    assert res2["result"]["content"] == "done run"

    res3 = await _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 43,
            "method": "ollamacode/chat",
            "params": {"message": "second", "sessionID": "protocol-session"},
        },
        object(),
        "model",
        "",
        0,
        0,
        "/tmp",
        confirm_tool_calls=True,
    )
    assert res3["result"]["content"] == "done run"


@pytest.mark.asyncio
async def test_protocol_rag_methods(monkeypatch):
    """ollamacode/ragIndex and ollamacode/ragQuery return expected payloads."""

    def fake_build_local_rag_index(
        workspace_root, max_files=400, max_chars_per_file=20000
    ):
        assert workspace_root == "/tmp/work"
        return {
            "index_path": "/tmp/rag_index.json",
            "workspace_root": workspace_root,
            "indexed_files": 4,
            "chunk_count": 9,
        }

    def fake_query_local_rag(query, max_results=5):
        assert query == "jwt"
        return [
            {"path": "docs/AUTH.md", "chunk_index": 0, "score": 4.2, "snippet": "JWT"}
        ]

    monkeypatch.setattr(
        "ollamacode.protocol_server.build_local_rag_index", fake_build_local_rag_index
    )
    monkeypatch.setattr(
        "ollamacode.protocol_server.query_local_rag", fake_query_local_rag
    )

    res1 = await _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "ollamacode/ragIndex",
            "params": {"workspaceRoot": "/tmp/work"},
        },
        None,
        "model",
        "",
        0,
        0,
        "/tmp",
    )
    assert res1["result"]["indexed_files"] == 4

    res2 = await _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "ollamacode/ragQuery",
            "params": {"query": "jwt", "maxResults": 3},
        },
        None,
        "model",
        "",
        0,
        0,
        "/tmp",
    )
    assert res2["result"]["results"][0]["path"] == "docs/AUTH.md"


def test_resolve_memory_request_settings_protocol():
    """Per-request memory settings accept camelCase/snake_case and clamp values."""
    auto, kg, rag, chars = _resolve_memory_request_settings(
        {
            "memoryAutoContext": False,
            "memory_kg_max_results": -2,
            "memoryRagMaxResults": 999,
            "memory_rag_snippet_chars": 12,
        },
        default_auto=True,
        default_kg_max=4,
        default_rag_max=4,
        default_rag_chars=220,
    )
    assert auto is False
    assert kg == 0
    assert rag == 20
    assert chars == 40


@pytest.mark.asyncio
async def test_protocol_session_methods(tmp_path, monkeypatch):
    """Session management protocol methods should expose the local session store."""
    monkeypatch.setattr("ollamacode.sessions._DB_PATH", tmp_path / "sessions.db")

    create_res = await _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 51,
            "method": "ollamacode/sessionCreate",
            "params": {"title": "Protocol Session", "workspaceRoot": str(tmp_path), "owner": "alice", "role": "editor"},
        },
        None,
        "model",
        "",
        0,
        0,
        str(tmp_path),
    )
    session = create_res["result"]["session"]
    session_id = session["id"]
    assert session["owner"] == "alice"

    update_res = await _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 51_1,
            "method": "ollamacode/sessionUpdate",
            "params": {"sessionID": session_id, "title": "Renamed Session", "owner": "alice", "role": "editor"},
        },
        None,
        "model",
        "",
        0,
        0,
        str(tmp_path),
    )
    assert update_res["result"]["session"]["title"] == "Renamed Session"
    assert update_res["result"]["session"]["owner"] == "alice"

    list_res = await _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 52,
            "method": "ollamacode/sessionList",
            "params": {"workspaceRoot": str(tmp_path)},
        },
        None,
        "model",
        "",
        0,
        0,
        str(tmp_path),
    )
    assert any(row["id"] == session_id for row in list_res["result"]["sessions"])

    export_res = await _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 53,
            "method": "ollamacode/sessionExport",
            "params": {"sessionID": session_id},
        },
        None,
        "model",
        "",
        0,
        0,
        str(tmp_path),
    )
    exported = export_res["result"]["data"]
    assert json.loads(exported)["session"]["title"] == "Renamed Session"

    workspace_res = await _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 54,
            "method": "ollamacode/workspaceInfo",
            "params": {},
        },
        None,
        "model",
        "",
        0,
        0,
        str(tmp_path),
        subagents=[{"name": "reviewer"}],
    )
    assert workspace_res["result"]["workspaceRoot"] == str(tmp_path)
    assert "reviewer" in workspace_res["result"]["subagents"]

    branch_res = await _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 55,
            "method": "ollamacode/sessionBranch",
            "params": {"sessionID": session_id},
        },
        None,
        "model",
        "",
        0,
        0,
        str(tmp_path),
    )
    assert branch_res["result"]["session"]["id"] != session_id

    children_res = await _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 56,
            "method": "ollamacode/sessionChildren",
            "params": {"sessionID": session_id},
        },
        None,
        "model",
        "",
        0,
        0,
        str(tmp_path),
    )
    assert any(
        row["id"] == branch_res["result"]["session"]["id"]
        for row in children_res["result"]["sessions"]
    )

    ancestors_res = await _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 56_1,
            "method": "ollamacode/sessionAncestors",
            "params": {"sessionID": branch_res["result"]["session"]["id"]},
        },
        None,
        "model",
        "",
        0,
        0,
        str(tmp_path),
    )
    assert ancestors_res["result"]["sessions"][0]["id"] == session_id

    timeline_res = await _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 56_2,
            "method": "ollamacode/sessionTimeline",
            "params": {"sessionID": branch_res["result"]["session"]["id"]},
        },
        None,
        "model",
        "",
        0,
        0,
        str(tmp_path),
    )
    assert timeline_res["result"]["timeline"]["session"]["id"] == branch_res["result"]["session"]["id"]

    from ollamacode.checkpoints import CheckpointRecorder

    file_path = tmp_path / "note.txt"
    file_path.write_text("before")
    recorder = CheckpointRecorder(
        session_id=branch_res["result"]["session"]["id"],
        workspace_root=str(tmp_path),
        prompt="edit",
        message_index=0,
    )
    recorder.record_pre("note.txt")
    file_path.write_text("after")
    checkpoint_id = recorder.finalize()

    checkpoint_files = await _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 56_3,
            "method": "ollamacode/checkpointFiles",
            "params": {"checkpointID": checkpoint_id},
        },
        None,
        "model",
        "",
        0,
        0,
        str(tmp_path),
    )
    assert checkpoint_files["result"]["files"][0]["path"] == "note.txt"

    checkpoint_info = await _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 56_35,
            "method": "ollamacode/checkpointGet",
            "params": {"checkpointID": checkpoint_id},
        },
        None,
        "model",
        "",
        0,
        0,
        str(tmp_path),
    )
    assert checkpoint_info["result"]["checkpoint"]["id"] == checkpoint_id

    checkpoint_diff = await _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 56_4,
            "method": "ollamacode/checkpointDiff",
            "params": {"checkpointID": checkpoint_id},
        },
        None,
        "model",
        "",
        0,
        0,
        str(tmp_path),
    )
    assert "--- a/note.txt" in checkpoint_diff["result"]["diff"]

    messages_res = await _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 57,
            "method": "ollamacode/sessionMessages",
            "params": {"sessionID": branch_res["result"]["session"]["id"]},
        },
        None,
        "model",
        "",
        0,
        0,
        str(tmp_path),
    )
    assert isinstance(messages_res["result"]["messages"], list)

    delete_res = await _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 56,
            "method": "ollamacode/sessionDelete",
            "params": {"sessionID": session_id},
        },
        None,
        "model",
        "",
        0,
        0,
        str(tmp_path),
    )
    assert delete_res["result"]["deleted"] is True


@pytest.mark.asyncio
async def test_protocol_workspace_registry_methods(tmp_path, monkeypatch):
    monkeypatch.setattr("ollamacode.workspaces._WORKSPACES_PATH", tmp_path / "workspaces.json")
    created = await _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 61,
            "method": "ollamacode/workspaceCreate",
            "params": {"name": "Remote", "type": "remote", "baseUrl": "http://localhost:9000"},
        },
        None,
        "model",
        "",
        0,
        0,
        str(tmp_path),
    )
    workspace_id = created["result"]["workspace"]["id"]
    listed = await _handle_request(
        {"jsonrpc": "2.0", "id": 62, "method": "ollamacode/workspaceList", "params": {}},
        None,
        "model",
        "",
        0,
        0,
        str(tmp_path),
    )
    assert any(item["id"] == workspace_id for item in listed["result"]["workspaces"])
    fetched = await _handle_request(
        {"jsonrpc": "2.0", "id": 63, "method": "ollamacode/workspaceGet", "params": {"workspaceID": workspace_id}},
        None,
        "model",
        "",
        0,
        0,
        str(tmp_path),
    )
    assert fetched["result"]["workspace"]["name"] == "Remote"
    updated = await _handle_request(
        {"jsonrpc": "2.0", "id": 64, "method": "ollamacode/workspaceUpdate", "params": {"workspaceID": workspace_id, "name": "Remote-2"}},
        None,
        "model",
        "",
        0,
        0,
        str(tmp_path),
    )
    assert updated["result"]["workspace"]["name"] == "Remote-2"


@pytest.mark.asyncio
async def test_protocol_workspace_proxy_method(tmp_path, monkeypatch):
    monkeypatch.setattr("ollamacode.workspaces._WORKSPACES_PATH", tmp_path / "workspaces.json")

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"ok": True}

        @property
        def text(self):
            return '{"ok": true}'

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def request(self, method, url, headers=None, json=None):
            assert method == "GET"
            assert url == "http://localhost:9000/workspace"
            return FakeResponse()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    created = await _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 71,
            "method": "ollamacode/workspaceCreate",
            "params": {"name": "Remote", "type": "remote", "baseUrl": "http://localhost:9000"},
        },
        None,
        "model",
        "",
        0,
        0,
        str(tmp_path),
    )
    workspace_id = created["result"]["workspace"]["id"]

    proxied = await _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 72,
            "method": "ollamacode/workspaceProxy",
            "params": {"workspaceID": workspace_id, "target": "workspace"},
        },
        None,
        "model",
        "",
        0,
        0,
        str(tmp_path),
    )
    assert proxied["result"]["statusCode"] == 200
    assert proxied["result"]["payload"]["ok"] is True


@pytest.mark.asyncio
async def test_protocol_control_plane_events():
    from ollamacode.control_plane import publish_event

    publish_event("session.created", {"session_id": "event-session"})
    result = await _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 73,
            "method": "ollamacode/controlPlaneEvents",
            "params": {"limit": 10},
        },
        None,
        "model",
        "",
        0,
        0,
        "/tmp",
    )
    assert any(
        event["type"] == "session.created"
        for event in result["result"]["events"]
    )


@pytest.mark.asyncio
async def test_protocol_principals_and_fleet(tmp_path, monkeypatch):
    monkeypatch.setattr("ollamacode.auth_registry._AUTH_PATH", tmp_path / "principals.json")
    monkeypatch.setattr("ollamacode.workspaces._WORKSPACES_PATH", tmp_path / "workspaces.json")
    async def fake_snapshot(workspaces):
        return {"total": len(workspaces), "remote": 0, "healthy": len(workspaces), "unhealthy": 0, "workspaces": workspaces}
    monkeypatch.setattr("ollamacode.fleet.collect_fleet_snapshot", fake_snapshot)
    created = await _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 74,
            "method": "ollamacode/principalCreate",
            "params": {"name": "Alice", "role": "admin"},
        },
        None,
        "model",
        "",
        0,
        0,
        str(tmp_path),
    )
    assert created["result"]["principal"]["name"] == "Alice"
    principals = await _handle_request(
        {"jsonrpc": "2.0", "id": 75, "method": "ollamacode/principalList", "params": {}},
        None,
        "model",
        "",
        0,
        0,
        str(tmp_path),
    )
    assert any(item["name"] == "Alice" for item in principals["result"]["principals"])
    principal_id = created["result"]["principal"]["id"]
    updated = await _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 75_1,
            "method": "ollamacode/principalUpdate",
            "params": {"principalID": principal_id, "workspaceIDs": ["w1", "w2"]},
        },
        None,
        "model",
        "",
        0,
        0,
        str(tmp_path),
    )
    assert updated["result"]["principal"]["workspace_ids"] == ["w1", "w2"]
    fetched = await _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 75_2,
            "method": "ollamacode/principalGet",
            "params": {"principalID": principal_id},
        },
        None,
        "model",
        "",
        0,
        0,
        str(tmp_path),
    )
    assert fetched["result"]["principal"]["workspace_ids"] == ["w1", "w2"]

    fleet = await _handle_request(
        {"jsonrpc": "2.0", "id": 76, "method": "ollamacode/fleetSummary", "params": {}},
        None,
        "model",
        "",
        0,
        0,
        str(tmp_path),
    )
    assert "total" in fleet["result"]
