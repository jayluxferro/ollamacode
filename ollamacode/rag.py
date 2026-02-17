"""Lightweight local RAG index/query helpers.

Stores a simple chunk index in ~/.ollamacode/rag_index.json.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

_RAG_INDEX_PATH = Path(os.path.expanduser("~")) / ".ollamacode" / "rag_index.json"
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


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]{2,}", text.lower())


def _chunk_text(text: str, *, max_chars: int = 1200, overlap: int = 200) -> list[str]:
    t = text.strip()
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]
    chunks: list[str] = []
    i = 0
    n = len(t)
    step = max(1, max_chars - overlap)
    while i < n:
        chunk = t[i : i + max_chars].strip()
        if chunk:
            chunks.append(chunk)
        i += step
    return chunks


def _iter_candidate_files(root: Path, max_files: int) -> list[Path]:
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


def build_local_rag_index(
    workspace_root: str,
    *,
    max_files: int = 400,
    max_chars_per_file: int = 20000,
) -> dict:
    """Build a local chunked text index for retrieval."""
    root = Path(workspace_root).resolve()
    files = _iter_candidate_files(root, max_files=max_files)
    chunks: list[dict] = []

    for path in files:
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not raw.strip():
            continue
        raw = raw[:max_chars_per_file]
        rel = str(path.relative_to(root)).replace("\\", "/")
        for i, ch in enumerate(_chunk_text(raw)):
            toks = _tokenize(ch)
            chunks.append(
                {
                    "path": rel,
                    "chunk_index": i,
                    "text": ch,
                    "tokens": toks[:200],
                }
            )

    data = {
        "workspace_root": str(root),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
    _RAG_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RAG_INDEX_PATH.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return {
        "index_path": str(_RAG_INDEX_PATH),
        "workspace_root": str(root),
        "indexed_files": len(files),
        "chunk_count": len(chunks),
    }


def _load_index() -> dict:
    if not _RAG_INDEX_PATH.exists():
        return {}
    try:
        return json.loads(_RAG_INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def query_local_rag(
    query: str,
    *,
    max_results: int = 5,
) -> list[dict]:
    """Query local chunk index using lightweight keyword scoring."""
    q = (query or "").strip().lower()
    if not q:
        return []
    data = _load_index()
    chunks = data.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        return []
    q_tokens = _tokenize(q)
    if not q_tokens:
        return []

    scored: list[tuple[float, dict]] = []
    for item in chunks:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", ""))
        text_l = text.lower()
        toks = item.get("tokens")
        token_set = set(toks) if isinstance(toks, list) else set(_tokenize(text_l))
        score = 0.0
        if q in text_l:
            score += 3.0
        for t in q_tokens:
            if t in token_set:
                score += 1.0
                score += min(0.5, text_l.count(t) * 0.05)
        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict] = []
    for score, item in scored[: max(1, max_results)]:
        out.append(
            {
                "path": item.get("path", ""),
                "chunk_index": item.get("chunk_index", 0),
                "score": round(score, 3),
                "snippet": str(item.get("text", ""))[:500],
            }
        )
    return out
