"""Unit tests for stdio JSON-RPC protocol server."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from ollamacode.protocol_server import (
    _handle_apply_edits,
    _handle_chat_stream,
    _handle_request,
)


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
    res = await _handle_request(
        req, None, "model", "", 0, 0, "/tmp"
    )
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
    req = {"jsonrpc": "2.0", "id": 1, "method": "ollamacode/chatStream", "params": {"message": ""}}
    response = await _handle_request(req, None, "model", "", 0, 0, "/tmp")
    assert hasattr(response, "__aiter__")
    parts = []
    async for p in response:
        parts.append(p)
    assert len(parts) == 1
    assert parts[0]["result"]["type"] == "error"
