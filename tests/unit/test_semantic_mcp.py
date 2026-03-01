"""Unit tests for the semantic search MCP server (semantic_mcp.py).

Covers: model name validation, cache versioning, embedding timeout wrapper,
cosine similarity, index/search flow with mocked ollama.
"""

import concurrent.futures
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ollamacode.servers import semantic_mcp


# ---------------------------------------------------------------------------
# Model name validation
# ---------------------------------------------------------------------------


class TestModelNameValidation:
    def test_valid_names(self):
        """Standard model names pass validation."""
        assert semantic_mcp._validate_model_name("nomic-embed-text") is None
        assert semantic_mcp._validate_model_name("mxbai-embed-large:latest") is None
        assert semantic_mcp._validate_model_name("all-minilm") is None
        assert semantic_mcp._validate_model_name("user/model:v1.0") is None

    def test_too_long_name_rejected(self):
        """Model names exceeding 200 chars are rejected."""
        err = semantic_mcp._validate_model_name("a" * 201)
        assert err is not None
        assert "too long" in err.lower()

    def test_invalid_chars_rejected(self):
        """Model names with shell metacharacters are rejected."""
        err = semantic_mcp._validate_model_name("model; rm -rf /")
        assert err is not None
        assert "invalid characters" in err.lower()

    def test_empty_name_rejected(self):
        """Empty model name is rejected."""
        err = semantic_mcp._validate_model_name("")
        assert err is not None


# ---------------------------------------------------------------------------
# _embed_model default and env override
# ---------------------------------------------------------------------------


class TestEmbedModel:
    def test_default_model(self, monkeypatch):
        """Default model is nomic-embed-text."""
        monkeypatch.delenv("OLLAMACODE_EMBED_MODEL", raising=False)
        assert semantic_mcp._embed_model() == "nomic-embed-text"

    def test_env_override(self, monkeypatch):
        """OLLAMACODE_EMBED_MODEL overrides the default."""
        monkeypatch.setenv("OLLAMACODE_EMBED_MODEL", "mxbai-embed-large")
        assert semantic_mcp._embed_model() == "mxbai-embed-large"

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        """Invalid model name in env falls back to default."""
        monkeypatch.setenv("OLLAMACODE_EMBED_MODEL", "bad; name")
        assert semantic_mcp._embed_model() == "nomic-embed-text"


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


class TestCosine:
    def test_identical_vectors(self):
        """Identical vectors have cosine similarity of 1.0."""
        assert abs(semantic_mcp._cosine([1, 0, 0], [1, 0, 0]) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have cosine similarity of 0.0."""
        assert abs(semantic_mcp._cosine([1, 0], [0, 1]) - 0.0) < 1e-6

    def test_opposite_vectors(self):
        """Opposite vectors have cosine similarity of -1.0."""
        assert abs(semantic_mcp._cosine([1, 0], [-1, 0]) - (-1.0)) < 1e-6

    def test_empty_vectors_return_zero(self):
        """Empty or mismatched vectors return 0.0."""
        assert semantic_mcp._cosine([], []) == 0.0
        assert semantic_mcp._cosine([1, 2], [1]) == 0.0

    def test_zero_vectors_return_zero(self):
        """Zero-magnitude vectors return 0.0 (no division by zero)."""
        assert semantic_mcp._cosine([0, 0], [0, 0]) == 0.0


# ---------------------------------------------------------------------------
# Cache versioning
# ---------------------------------------------------------------------------


class TestCacheVersioning:
    def test_version_mismatch_detected(self, mock_fs: Path):
        """semantic_search_codebase detects stale cache versions."""
        cache_dir = mock_fs / ".ollamacode"
        cache_dir.mkdir(parents=True)
        cache_file = cache_dir / "embeddings.json"
        cache_file.write_text(
            json.dumps(
                {
                    "version": 1,  # Old version; current is 2
                    "model": "nomic-embed-text",
                    "entries": [{"path": "a.py", "text": "hello", "embedding": [0.1]}],
                }
            )
        )
        result = semantic_mcp.semantic_search_codebase("hello")
        assert "version mismatch" in result.lower()

    def test_model_mismatch_detected(self, mock_fs: Path, monkeypatch):
        """semantic_search_codebase detects model changes since indexing."""
        monkeypatch.delenv("OLLAMACODE_EMBED_MODEL", raising=False)
        cache_dir = mock_fs / ".ollamacode"
        cache_dir.mkdir(parents=True)
        cache_file = cache_dir / "embeddings.json"
        cache_file.write_text(
            json.dumps(
                {
                    "version": semantic_mcp._CACHE_VERSION,
                    "model": "different-model",
                    "entries": [{"path": "a.py", "text": "hello", "embedding": [0.1]}],
                }
            )
        )
        result = semantic_mcp.semantic_search_codebase("hello")
        assert "model" in result.lower() and "rebuild" in result.lower()


# ---------------------------------------------------------------------------
# Embedding timeout wrapper
# ---------------------------------------------------------------------------


class TestEmbedWithTimeout:
    def test_timeout_raises(self):
        """_embed_with_timeout raises TimeoutError when embedding hangs."""
        import time

        slow_mod = MagicMock()

        def slow_embed(**kwargs):
            time.sleep(10)

        slow_mod.embed.side_effect = slow_embed

        with pytest.raises(concurrent.futures.TimeoutError):
            semantic_mcp._embed_with_timeout(slow_mod, "model", ["text"], timeout=1)

    def test_normal_call_succeeds(self, mock_ollama):
        """_embed_with_timeout returns result for fast calls."""
        result = semantic_mcp._embed_with_timeout(
            mock_ollama, "model", ["text"], timeout=5
        )
        assert result is not None


# ---------------------------------------------------------------------------
# index_codebase with mock ollama
# ---------------------------------------------------------------------------


class TestIndexCodebase:
    def test_no_files_returns_message(self, mock_fs: Path):
        """index_codebase returns 'No files matched' for empty workspace."""
        with patch.dict("sys.modules", {"ollama": MagicMock()}):
            result = semantic_mcp.index_codebase()
        assert "No files matched" in result

    def test_ollama_not_installed(self, mock_fs: Path, monkeypatch):
        """index_codebase returns clear error when ollama is not installed."""
        (mock_fs / "test.py").write_text("hello")
        # Simulate ImportError for ollama
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "ollama":
                raise ImportError("No module named 'ollama'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        result = semantic_mcp.index_codebase()
        assert "not installed" in result.lower()
