"""Vector memory engine for OllamaCode.

Replaces the JSON-based rag.py with a SQLite-backed index that combines:
  - FTS5 virtual table for fast keyword / BM25 search
  - A ``chunks`` table with BLOB embedding columns for cosine similarity
  - Hybrid scoring: weighted fusion of keyword and vector scores
  - Markdown-aware chunking (preserves heading context)
  - LRU in-memory embedding cache
  - Safe atomic reindex (rebuilds inside a transaction; safe concurrent reads)

If FTS5 is not available in the SQLite build, keyword search falls back to
a LIKE scan so the module always works.

Embeddings are generated via the configured provider or Ollama's
``/api/embeddings`` endpoint.  If no embedding backend is reachable the module
degrades gracefully to keyword-only search.

Usage:
    from ollamacode.vector_memory import build_vector_index, query_vector_memory
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import struct
import threading
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VM_DIR = Path(os.path.expanduser("~")) / ".ollamacode"
_VM_DB_PATH = _VM_DIR / "vector_memory.db"

_IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".cursor",
}
_TEXT_EXTS = {
    ".md",
    ".txt",
    ".rst",
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".sh",
    ".sql",
}

# Embedding dimension cache (set when first embedding is fetched)
_EMBEDDING_DIM: int | None = None
_DB_LOCK = threading.Lock()
_EMBED_CONFIGS: dict[str, dict[str, Any]] = {}
_SQLITE_VEC_AVAILABLE = False


def _sqlite_vec_available() -> bool:
    """Return True if sqlite-vec has been loaded in this process."""
    return _SQLITE_VEC_AVAILABLE


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _markdown_heading_context(text: str, pos: int) -> str:
    """Return the most recent heading before *pos* in *text*, or ''."""
    last_heading = ""
    for m in re.finditer(r"^#{1,6}\s+(.+)$", text[:pos], re.MULTILINE):
        last_heading = m.group(1).strip()
    return last_heading


def _chunk_text_markdown(
    text: str,
    *,
    max_chars: int = 1200,
    overlap: int = 200,
) -> list[dict[str, str]]:
    """Chunk *text* into overlapping windows, attaching heading context."""
    t = text.strip()
    if not t:
        return []

    step = max(1, max_chars - overlap)
    chunks: list[dict[str, str]] = []
    i = 0
    n = len(t)
    while i < n:
        chunk = t[i : i + max_chars].strip()
        if chunk:
            heading = _markdown_heading_context(t, i)
            chunks.append({"text": chunk, "heading": heading})
        i += step
    return chunks


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def _embed_text(text: str, config: dict[str, Any] | None = None) -> list[float] | None:
    """Return an embedding vector for *text*, or None if backend unavailable."""
    try:
        return _embed_via_ollama(text)
    except Exception:
        pass
    if config:
        try:
            return _embed_via_provider(text, config)
        except Exception:
            pass
    return None


def _embed_via_ollama(text: str, model: str = "nomic-embed-text") -> list[float]:
    """Fetch embedding from Ollama /api/embeddings."""
    import urllib.request

    payload = json.dumps({"model": model, "prompt": text}).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    return data["embedding"]


def _embed_via_provider(text: str, config: dict[str, Any]) -> list[float]:
    """Fetch embedding from the configured OpenAI-compatible provider."""
    from .providers import get_provider

    provider = get_provider(config)
    # Use the provider's underlying openai client if available
    client = getattr(provider, "_client", None)
    if client is None:
        raise RuntimeError("Provider does not expose _client for embeddings")
    model = config.get("embedding_model") or "text-embedding-3-small"
    resp = client.embeddings.create(input=text, model=model)
    return resp.data[0].embedding


def _embed_via_provider_many(
    texts: list[str], config: dict[str, Any]
) -> list[list[float] | None]:
    """Batch embeddings via OpenAI-compatible client (if supported)."""
    from .providers import get_provider

    provider = get_provider(config)
    client = getattr(provider, "_client", None)
    if client is None:
        raise RuntimeError("Provider does not expose _client for embeddings")
    model = config.get("embedding_model") or "text-embedding-3-small"
    resp = client.embeddings.create(input=texts, model=model)
    data = getattr(resp, "data", None) or []
    out: list[list[float] | None] = [None] * len(texts)
    for i, item in enumerate(data):
        emb = getattr(item, "embedding", None)
        if emb is not None:
            out[i] = emb
    return out


# Memoize embeddings for the current process to avoid redundant API calls.
def _embedding_config_key(config: dict[str, Any] | None) -> str:
    """Return a stable cache key for embedding configuration."""
    if not config:
        return "ollama:nomic-embed-text"
    provider = str(config.get("provider") or "ollama").lower()
    base_url = str(config.get("base_url") or "")
    model = str(config.get("embedding_model") or "text-embedding-3-small")
    return f"{provider}|{base_url}|{model}"


@lru_cache(maxsize=512)
def _cached_embed(text: str, config_key: str) -> tuple[float, ...] | None:
    """LRU-cached wrapper; config_key selects embedding backend."""
    config = _EMBED_CONFIGS.get(config_key)
    result = _embed_text(text, config)
    return tuple(result) if result is not None else None


def _get_embedding(
    text: str, config: dict[str, Any] | None = None
) -> list[float] | None:
    """Fetch an embedding with LRU caching."""
    key = _embedding_config_key(config)
    if key not in _EMBED_CONFIGS:
        _EMBED_CONFIGS[key] = dict(config or {})
    cached = _cached_embed(text, key)
    return list(cached) if cached is not None else None


def _embed_many(
    texts: list[str], config: dict[str, Any] | None = None
) -> list[list[float] | None]:
    """Embed multiple texts; uses provider batch if configured, else per-text with caching."""
    if not texts:
        return []
    if config and str(config.get("embedding_backend") or "").lower() == "provider":
        try:
            return _embed_via_provider_many(texts, config)
        except Exception:
            pass
    return [_get_embedding(t, config) for t in texts]


def _try_enable_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Attempt to enable sqlite-vec extension if installed."""
    global _SQLITE_VEC_AVAILABLE
    if _SQLITE_VEC_AVAILABLE:
        return
    try:
        import sqlite_vec  # type: ignore[import-not-found]

        sqlite_vec.load(conn)
        _SQLITE_VEC_AVAILABLE = True
    except Exception:
        _SQLITE_VEC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Cosine similarity (pure Python + optional numpy fast path)
# ---------------------------------------------------------------------------


def _cosine_sim(
    a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]
) -> float:
    try:
        import numpy as np  # type: ignore[import]

        av, bv = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
        na, nb = np.linalg.norm(av), np.linalg.norm(bv)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(av, bv) / (na * nb))
    except ImportError:
        # Pure Python fallback
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


# ---------------------------------------------------------------------------
# Blob serialisation for float32 vectors
# ---------------------------------------------------------------------------


def _vec_to_blob(vec: list[float] | tuple[float, ...]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _blob_to_vec(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace   TEXT NOT NULL,
    path        TEXT NOT NULL,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    heading     TEXT NOT NULL DEFAULT '',
    text        TEXT NOT NULL,
    embedding   BLOB,          -- float32 little-endian packed, NULL if not embedded
    indexed_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_workspace ON chunks(workspace);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(workspace, path);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

_FTS5_TABLE_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    heading,
    path UNINDEXED,
    workspace UNINDEXED,
    content=chunks,
    content_rowid=id
);
"""

_FTS5_TRIGGER_SQL = """
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, heading, path, workspace)
    VALUES (new.id, new.text, new.heading, new.path, new.workspace);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, heading, path, workspace)
    VALUES ('delete', old.id, old.text, old.heading, old.path, old.workspace);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, heading, path, workspace)
    VALUES ('delete', old.id, old.text, old.heading, old.path, old.workspace);
    INSERT INTO chunks_fts(rowid, text, heading, path, workspace)
    VALUES (new.id, new.text, new.heading, new.path, new.workspace);
END;
"""


def _ensure_vec_table(conn: sqlite3.Connection, dim: int) -> None:
    """Create sqlite-vec table if available; no-op on errors."""
    if not _SQLITE_VEC_AVAILABLE:
        return
    try:
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(embedding float[{dim}])"
        )
    except sqlite3.OperationalError:
        pass


def _insert_vec(conn: sqlite3.Connection, rowid: int, vec: list[float]) -> None:
    """Insert embedding into sqlite-vec table if available."""
    if not _SQLITE_VEC_AVAILABLE:
        return
    try:
        conn.execute(
            "INSERT OR REPLACE INTO chunks_vec(rowid, embedding) VALUES (?, ?)",
            (rowid, json.dumps(vec)),
        )
    except sqlite3.OperationalError:
        pass


def _has_fts5(conn: sqlite3.Connection) -> bool:
    """Return True if this SQLite build includes FTS5."""
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE IF EXISTS _fts5_probe")
        return True
    except sqlite3.OperationalError:
        return False


def _open_db(db_path: Path = _VM_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA_SQL)
    _try_enable_sqlite_vec(conn)
    if _has_fts5(conn):
        try:
            conn.executescript(_FTS5_TABLE_SQL + _FTS5_TRIGGER_SQL)
        except sqlite3.OperationalError:
            pass  # Triggers may already exist
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# File iteration
# ---------------------------------------------------------------------------


def _iter_files(root: Path, max_files: int) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if len(out) >= max_files:
            break
        if not p.is_file():
            continue
        if any(part in _IGNORE_DIRS for part in p.parts):
            continue
        if p.suffix.lower() not in _TEXT_EXTS:
            continue
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# Build / reindex
# ---------------------------------------------------------------------------


def build_vector_index(
    workspace_root: str,
    *,
    max_files: int = 400,
    max_chars_per_file: int = 20_000,
    embed: bool = True,
    config: dict[str, Any] | None = None,
    db_path: Path = _VM_DB_PATH,
) -> dict[str, Any]:
    """Build (or rebuild) the vector memory index for *workspace_root*.

    Returns a summary dict with ``indexed_files``, ``chunk_count``, ``db_path``.
    """
    global _EMBEDDING_DIM
    root = Path(workspace_root).resolve()
    workspace_key = str(root)
    files = _iter_files(root, max_files)
    now = datetime.now(timezone.utc).isoformat()

    conn = _open_db(db_path)
    chunk_count = 0

    with _DB_LOCK:
        try:
            # Atomically replace all chunks for this workspace
            conn.execute("DELETE FROM chunks WHERE workspace = ?", (workspace_key,))

            for path in files:
                try:
                    raw = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if not raw.strip():
                    continue
                raw = raw[:max_chars_per_file]
                rel = str(path.relative_to(root)).replace("\\", "/")
                chunks = _chunk_text_markdown(raw)
                texts = [c["text"] for c in chunks]
                headings = [c["heading"] for c in chunks]
                vectors: list[list[float] | None] = []
                if embed:
                    vectors = _embed_many(texts, config)
                else:
                    vectors = [None] * len(texts)
                for i, (text, heading, vec) in enumerate(zip(texts, headings, vectors)):
                    blob: bytes | None = None
                    if vec:
                        blob = _vec_to_blob(vec)
                        if _EMBEDDING_DIM is None:
                            _EMBEDDING_DIM = len(vec)
                            _ensure_vec_table(conn, _EMBEDDING_DIM)
                    conn.execute(
                        "INSERT INTO chunks (workspace, path, chunk_index, heading, text, embedding, indexed_at)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (workspace_key, rel, i, heading, text, blob, now),
                    )
                    if vec:
                        try:
                            rowid = conn.execute(
                                "SELECT last_insert_rowid()"
                            ).fetchone()[0]
                            _insert_vec(conn, int(rowid), vec)
                        except Exception:
                            pass
                    chunk_count += 1

            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (f"last_indexed:{workspace_key}", now),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    return {
        "workspace_root": workspace_key,
        "indexed_files": len(files),
        "chunk_count": chunk_count,
        "db_path": str(db_path),
    }


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def query_vector_memory(
    query: str,
    workspace_root: str | None = None,
    *,
    max_results: int = 5,
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4,
    config: dict[str, Any] | None = None,
    db_path: Path = _VM_DB_PATH,
) -> list[dict[str, Any]]:
    """Hybrid (vector + keyword) search over the vector memory index.

    Returns up to *max_results* dicts with keys:
      path, chunk_index, heading, score, snippet
    """
    q = (query or "").strip()
    if not q:
        return []

    if not db_path.exists():
        return []

    conn = _open_db(db_path)
    workspace_key = str(Path(workspace_root).resolve()) if workspace_root else None

    # 1. Keyword search via FTS5 (or LIKE fallback)
    kw_scores: dict[int, float] = {}
    try:
        if workspace_key:
            rows = conn.execute(
                "SELECT rowid, bm25(chunks_fts) AS score FROM chunks_fts"
                " WHERE chunks_fts MATCH ? AND workspace = ? ORDER BY score LIMIT ?",
                (q, workspace_key, max_results * 4),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT rowid, bm25(chunks_fts) AS score FROM chunks_fts"
                " WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
                (q, max_results * 4),
            ).fetchall()
        # BM25 returns negative scores in SQLite FTS5; negate so higher = better.
        for row in rows:
            kw_scores[row["rowid"]] = -float(row["score"])
    except sqlite3.OperationalError:
        # FTS5 unavailable — fallback to LIKE scan
        like_q = f"%{q}%"
        if workspace_key:
            rows = conn.execute(
                "SELECT id FROM chunks WHERE text LIKE ? AND workspace = ? LIMIT ?",
                (like_q, workspace_key, max_results * 4),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id FROM chunks WHERE text LIKE ? LIMIT ?",
                (like_q, max_results * 4),
            ).fetchall()
        for row in rows:
            kw_scores[row["id"]] = 1.0

    # Normalise keyword scores to [0, 1]
    if kw_scores:
        max_kw = max(kw_scores.values())
        if max_kw > 0:
            kw_scores = {k: v / max_kw for k, v in kw_scores.items()}

    # 2. Vector search
    vec_scores: dict[int, float] = {}
    q_vec = _get_embedding(q, config)
    if q_vec:
        if _SQLITE_VEC_AVAILABLE:
            try:
                if workspace_key:
                    rows = conn.execute(
                        "SELECT v.rowid AS id, v.distance AS distance "
                        "FROM chunks_vec v JOIN chunks c ON c.id = v.rowid "
                        "WHERE c.workspace = ? AND v.embedding MATCH ? "
                        "ORDER BY v.distance LIMIT ?",
                        (workspace_key, json.dumps(q_vec), max_results * 4),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT rowid AS id, distance FROM chunks_vec "
                        "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                        (json.dumps(q_vec), max_results * 4),
                    ).fetchall()
                for row in rows:
                    dist = float(row["distance"])
                    vec_scores[row["id"]] = 1.0 / (1.0 + dist)
            except sqlite3.OperationalError:
                pass
        if vec_scores:
            # Already have candidates from sqlite-vec; skip full scan.
            pass
        else:
            # Prefer a smaller candidate set (FTS top matches) to avoid full scans.
            candidate_ids: list[int] | None = None
            if kw_scores:
                candidate_ids = [
                    rid
                    for rid, _ in sorted(
                        kw_scores.items(), key=lambda x: x[1], reverse=True
                    )
                ][: max_results * 40]
            rows = []
            if candidate_ids:
                placeholders = ",".join("?" * len(candidate_ids))
                rows = conn.execute(
                    f"SELECT id, embedding FROM chunks WHERE id IN ({placeholders}) AND embedding IS NOT NULL",
                    candidate_ids,
                ).fetchall()
            else:
                # Fallback: bounded scan to avoid loading entire DB.
                limit = max(200, max_results * 80)
                if workspace_key:
                    rows = conn.execute(
                        "SELECT id, embedding FROM chunks WHERE workspace = ? AND embedding IS NOT NULL LIMIT ?",
                        (workspace_key, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL LIMIT ?",
                        (limit,),
                    ).fetchall()
            sims: list[tuple[int, float]] = []
            for row in rows:
                blob = row["embedding"]
                if blob:
                    vec = _blob_to_vec(blob)
                    sim = _cosine_sim(q_vec, vec)
                    sims.append((row["id"], sim))
            # Keep top candidates
            sims.sort(key=lambda x: x[1], reverse=True)
            for rid, sim in sims[: max_results * 4]:
                vec_scores[rid] = sim

    # 3. Hybrid fusion
    all_ids = set(kw_scores) | set(vec_scores)
    fused: list[tuple[float, int]] = []
    for rid in all_ids:
        score = keyword_weight * kw_scores.get(
            rid, 0.0
        ) + vector_weight * vec_scores.get(rid, 0.0)
        fused.append((score, rid))
    fused.sort(reverse=True)
    top_ids = [rid for _, rid in fused[:max_results]]

    if not top_ids:
        conn.close()
        return []

    # 4. Fetch chunk details
    placeholders = ",".join("?" * len(top_ids))
    chunk_rows = conn.execute(
        f"SELECT id, path, chunk_index, heading, text FROM chunks WHERE id IN ({placeholders})",
        top_ids,
    ).fetchall()
    conn.close()

    id_to_row = {row["id"]: row for row in chunk_rows}
    results: list[dict[str, Any]] = []
    for score, rid in fused[:max_results]:
        row = id_to_row.get(rid)
        if row is None:
            continue
        results.append(
            {
                "path": row["path"],
                "chunk_index": row["chunk_index"],
                "heading": row["heading"],
                "score": round(score, 4),
                "snippet": row["text"][:500],
            }
        )
    return results


# ---------------------------------------------------------------------------
# Backwards-compatible shim (drop-in replacement for rag.py public API)
# ---------------------------------------------------------------------------


def build_local_rag_index(
    workspace_root: str,
    *,
    max_files: int = 400,
    max_chars_per_file: int = 20_000,
) -> dict[str, Any]:
    """Drop-in replacement for ``rag.build_local_rag_index``.

    Delegates to :func:`build_vector_index` but skips embeddings so it is
    fast and dependency-free for callers that only use keyword search.
    """
    return build_vector_index(
        workspace_root,
        max_files=max_files,
        max_chars_per_file=max_chars_per_file,
        embed=False,
    )


def query_local_rag(query: str, *, max_results: int = 5) -> list[dict[str, Any]]:
    """Drop-in replacement for ``rag.query_local_rag``."""
    return query_vector_memory(query, max_results=max_results)
