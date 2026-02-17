"""Unit tests for stdio JSON-RPC protocol server."""

from pathlib import Path

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
