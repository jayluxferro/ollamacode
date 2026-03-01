"""
Built-in state MCP server: get_state, update_state, clear_state, append_recent_file.

State is stored in ~/.ollamacode/state.json (recent files, preferences). Lets the assistant remember context across sessions.
"""

import os

from mcp.server.fastmcp import FastMCP
from . import configure_server_logging

from ..rag import build_local_rag_index as _build_local_rag_index
from ..rag import query_local_rag as _query_local_rag
from ..state import append_recent_file as _append_recent_file
from ..state import add_knowledge_node as _add_knowledge_node
from ..state import clear_state as _clear_state
from ..state import get_state as _get_state
from ..state import query_knowledge_graph as _query_knowledge_graph
from ..state import record_reasoning as _record_reasoning
from ..state import update_state as _update_state

configure_server_logging()

mcp = FastMCP("ollamacode-state")


@mcp.tool()
def get_state() -> dict:
    """Return persistent state (recent_files, preferences, etc.) from ~/.ollamacode/state.json."""
    return _get_state()


@mcp.tool()
def update_state(
    recent_files: list[str] | None = None,
    preferences: dict | None = None,
    current_plan: str | None = None,
    completed_steps: list[str] | None = None,
    knowledge_index: list[dict | str] | None = None,
    feedback_append: dict | None = None,
) -> str:
    """Update state. recent_files: list of paths. preferences: dict to merge. current_plan: plan text (long-term). completed_steps: list of done steps. knowledge_index: list of {topic, summary} or strings. feedback_append: append one feedback entry {type, value, context}."""
    kwargs = {}
    if recent_files is not None:
        kwargs["recent_files"] = recent_files
    if preferences is not None:
        kwargs["preferences"] = preferences
    if current_plan is not None:
        kwargs["current_plan"] = current_plan
    if completed_steps is not None:
        kwargs["completed_steps"] = completed_steps
    if knowledge_index is not None:
        kwargs["knowledge_index"] = knowledge_index
    if feedback_append is not None:
        kwargs["feedback_append"] = feedback_append
    return _update_state(**kwargs)


@mcp.tool()
def append_recent_file(path: str) -> str:
    """Record a file path as recently used (for context across sessions)."""
    return _append_recent_file(path)


@mcp.tool()
def clear_state() -> str:
    """Clear all persistent state (recent files, preferences)."""
    return _clear_state()


@mcp.tool()
def record_reasoning(steps: list[str], conclusion: str = "") -> str:
    """Record reasoning for explainability: steps (list of strings) and conclusion (string). User can see this in the UI."""
    return _record_reasoning(steps, conclusion)


@mcp.tool()
def add_knowledge_node(
    topic: str,
    summary: str = "",
    related: list[str] | None = None,
    source: str = "",
) -> str:
    """Upsert a lightweight knowledge node in persistent state with optional related topics and source."""
    return _add_knowledge_node(topic, summary=summary, related=related, source=source)


@mcp.tool()
def query_knowledge_graph(query: str, max_results: int = 5) -> list[dict]:
    """Search persistent lightweight knowledge graph by keyword query."""
    return _query_knowledge_graph(query, max_results=max_results)


@mcp.tool()
def build_rag_index(
    workspace_root: str | None = None,
    max_files: int = 400,
    max_chars_per_file: int = 20000,
) -> dict:
    """Build local RAG chunk index from workspace docs/code into ~/.ollamacode/rag_index.json."""
    root = workspace_root or os.environ.get("OLLAMACODE_FS_ROOT") or os.getcwd()
    return _build_local_rag_index(
        root,
        max_files=max_files,
        max_chars_per_file=max_chars_per_file,
    )


@mcp.tool()
def rag_query(query: str, max_results: int = 5) -> list[dict]:
    """Query local RAG chunk index and return top matching snippets."""
    return _query_local_rag(query, max_results=max_results)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
