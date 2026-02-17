"""Unit tests for lightweight local RAG index/query."""

from unittest.mock import patch

from ollamacode.rag import build_local_rag_index, query_local_rag


def test_build_and_query_local_rag(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "AUTH.md").write_text("Auth uses JWT tokens and refresh tokens.")
    (docs / "API.md").write_text("API uses REST endpoints under /v1.")

    idx_path = tmp_path / "rag_index.json"
    with patch("ollamacode.rag._RAG_INDEX_PATH", idx_path):
        info = build_local_rag_index(
            str(tmp_path), max_files=20, max_chars_per_file=5000
        )
        assert info["indexed_files"] >= 2
        assert info["chunk_count"] >= 2
        rows = query_local_rag("jwt refresh", max_results=3)
        assert rows
        assert any("AUTH.md" in str(r.get("path", "")) for r in rows)


def test_query_local_rag_no_index(tmp_path):
    idx_path = tmp_path / "no_index.json"
    with patch("ollamacode.rag._RAG_INDEX_PATH", idx_path):
        rows = query_local_rag("anything", max_results=3)
        assert rows == []
