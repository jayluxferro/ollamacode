from ollamacode.memory import build_dynamic_memory_context


def test_build_dynamic_memory_context_empty_query() -> None:
    assert build_dynamic_memory_context("   ") == ""


def test_build_dynamic_memory_context_includes_kg_and_rag(monkeypatch) -> None:
    def _fake_kg(query: str, max_results: int = 4):
        assert query == "auth bug"
        assert max_results == 4
        return [
            {
                "topic": "auth token refresh",
                "summary": "Refresh flow fails when clock skew is large.",
                "related": ["jwt", "session"],
            }
        ]

    def _fake_rag(query: str, max_results: int = 4):
        assert query == "auth bug"
        assert max_results == 4
        return [
            {
                "path": "src/auth/refresh.py",
                "score": 5,
                "snippet": "if now - issued_at > ttl: raise TokenExpired()",
            }
        ]

    monkeypatch.setattr("ollamacode.memory.query_knowledge_graph", _fake_kg)
    monkeypatch.setattr("ollamacode.memory.query_local_rag", _fake_rag)

    out = build_dynamic_memory_context("auth bug")
    assert "Knowledge graph matches:" in out
    assert "auth token refresh" in out
    assert "Local retrieval matches:" in out
    assert "src/auth/refresh.py" in out
