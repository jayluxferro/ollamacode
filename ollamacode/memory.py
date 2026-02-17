"""Helpers for automatic query-specific memory retrieval."""

from __future__ import annotations

from .rag import query_local_rag
from .state import query_knowledge_graph


def build_dynamic_memory_context(
    query: str,
    *,
    kg_max_results: int = 4,
    rag_max_results: int = 4,
    rag_snippet_chars: int = 220,
) -> str:
    """Build a compact query-specific memory block from KG + local RAG."""
    q = (query or "").strip()
    if not q:
        return ""

    parts: list[str] = []

    try:
        kg_rows = query_knowledge_graph(q, max_results=kg_max_results)
    except Exception:
        kg_rows = []
    if kg_rows:
        lines = []
        for r in kg_rows:
            if not isinstance(r, dict):
                continue
            topic = str(r.get("topic", "")).strip()
            summary = str(r.get("summary", "")).strip()
            related = r.get("related")
            rel = (
                [str(x).strip() for x in related if str(x).strip()]
                if isinstance(related, list)
                else []
            )
            if not topic:
                continue
            line = f"- {topic}"
            if summary:
                line += f": {summary[:160]}"
            if rel:
                line += f" (related: {', '.join(rel[:4])})"
            lines.append(line)
        if lines:
            parts.append("Knowledge graph matches:\n" + "\n".join(lines))

    try:
        rag_rows = query_local_rag(q, max_results=rag_max_results)
    except Exception:
        rag_rows = []
    if rag_rows:
        lines = []
        for i, r in enumerate(rag_rows, 1):
            if not isinstance(r, dict):
                continue
            path = str(r.get("path", "")).strip()
            snippet = str(r.get("snippet", "")).strip().replace("\n", " ")
            score = r.get("score", 0)
            if not path:
                continue
            lines.append(f"[{i}] {path} (score={score}): {snippet[:rag_snippet_chars]}")
        if lines:
            parts.append("Local retrieval matches:\n" + "\n".join(lines))

    return "\n\n".join(parts)
