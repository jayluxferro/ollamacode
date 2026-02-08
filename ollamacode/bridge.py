"""
Bridge: map MCP tools to Ollama (OpenAI-style) tool format.

MCP Tool: name, description, inputSchema (JSON Schema)
Ollama/OpenAI: type="function", function={ name, description, parameters }
where parameters = inputSchema.
"""

from __future__ import annotations

from typing import Any


def mcp_tool_to_ollama(mcp_tool: Any) -> dict[str, Any]:
    """Convert one MCP Tool to Ollama chat API tool format."""
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description or "",
            "parameters": getattr(mcp_tool, "inputSchema", {}) or {},
        },
    }


def mcp_tools_to_ollama(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    """Convert a list of MCP tools to Ollama chat API tools format."""
    return [mcp_tool_to_ollama(t) for t in mcp_tools]
