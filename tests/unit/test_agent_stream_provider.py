"""Unit tests for provider-backed streaming behavior in agent loop."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import CallToolResult, TextContent, Tool

from ollamacode.agent import run_agent_loop_stream
from tests.helpers.mock_mcp import make_mock_mcp_session


@pytest.mark.asyncio
async def test_run_agent_loop_stream_provider_with_tools_uses_chat_async() -> None:
    tool = Tool(
        name="add",
        description="Add numbers",
        inputSchema={"type": "object", "properties": {"a": {}, "b": {}}},
    )
    session = make_mock_mcp_session(
        tools=[tool],
        call_tool_return=CallToolResult(
            content=[TextContent(type="text", text="5")],
            isError=False,
        ),
    )

    provider = MagicMock()
    provider.chat_async = AsyncMock(
        side_effect=[
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "add", "arguments": {"a": 2, "b": 3}}}
                    ],
                }
            },
            {"message": {"content": "The result is 5."}},
        ]
    )
    provider.chat_stream_sync = MagicMock()

    chunks: list[str] = []
    async for frag in run_agent_loop_stream(
        session,
        "test-model",
        "What is 2+3?",
        provider=provider,
        max_tool_rounds=3,
    ):
        chunks.append(frag)

    assert "".join(chunks).strip() == "The result is 5."
    assert provider.chat_async.await_count == 2
    provider.chat_stream_sync.assert_not_called()
    session.call_tool.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_agent_loop_stream_provider_falls_back_on_stream_error() -> None:
    session = make_mock_mcp_session(tools=[])

    provider = MagicMock()
    provider.chat_stream_sync = MagicMock(side_effect=RuntimeError("stream broke"))
    provider.chat_async = AsyncMock(
        return_value={"message": {"content": "fallback ok"}}
    )

    chunks: list[str] = []
    async for frag in run_agent_loop_stream(
        session,
        "test-model",
        "hello",
        provider=provider,
        max_tool_rounds=1,
    ):
        chunks.append(frag)

    assert "fallback ok" in "".join(chunks)
    provider.chat_async.assert_awaited_once()
