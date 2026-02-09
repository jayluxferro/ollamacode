"""
Bridge: map MCP tools to Ollama (OpenAI-style) tool format.

MCP Tool: name, description, inputSchema (JSON Schema)
Ollama/OpenAI: type="function", function={ name, description, parameters }
where parameters = inputSchema.

When multiple servers are used, MCP prefixes tool names (e.g. ollamacode-fs_read_file).
We rewrite built-in server tools to short names (read_file, run_command, etc.) so the
model sees one clear tool per capability and does not confuse tool names with search queries.
"""

from __future__ import annotations

from typing import Any

BUILTIN_SERVER_PREFIXES = (
    "ollamacode-fs_",
    "ollamacode-terminal_",
    "ollamacode-codebase_",
    "ollamacode-tools_",
    "ollamacode-git_",
)


def _short_name_for_builtin(name: str) -> str | None:
    """If name is from a built-in server, return the short name (e.g. read_file); else None."""
    for prefix in BUILTIN_SERVER_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return None


def use_short_names_for_builtin_tools(
    ollama_tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rewrite built-in server tool names to short form so the model sees read_file, run_command, etc."""
    result = []
    for t in ollama_tools:
        t = dict(t)
        fn = (t.get("function") or {}).copy()
        name = fn.get("name") or ""
        short = _short_name_for_builtin(name)
        if short is not None:
            fn["name"] = short
            t["function"] = fn
        result.append(t)
    return result


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


def add_tool_aliases_for_ollama(
    ollama_tools: list[dict[str, Any]],
    alias_map: dict[str, str],
) -> list[dict[str, Any]]:
    """
    Append synthetic tool entries for aliases (e.g. open_file -> read_file) so Ollama
    has a mapping and does not warn "no reverse mapping found for function name".
    """
    result = list(ollama_tools)
    for alias_name, target_name in alias_map.items():
        for t in ollama_tools:
            fn = t.get("function") or {}
            name = fn.get("name") or ""
            if name == target_name or name.endswith("_" + target_name):
                result.append(
                    {
                        "type": "function",
                        "function": {
                            "name": alias_name,
                            "description": (fn.get("description") or "")
                            + f" (Alias for {target_name}.)",
                            "parameters": fn.get("parameters") or {},
                        },
                    }
                )
                break
    return result
