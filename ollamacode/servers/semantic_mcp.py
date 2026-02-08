"""
Optional semantic codebase search MCP: index_codebase, semantic_search_codebase.

Uses Ollama embeddings (e.g. nomic-embed-text). Requires Ollama running with an embedding model.
Add to config when you want semantic (meaning-based) search; not in default servers.
Root directory: OLLAMACODE_FS_ROOT env var, or current working directory.
Cache: root/.ollamacode/embeddings.json
"""

import json
import math
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ollamacode-semantic")

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"}
MAX_FILE_BYTES = 300_000
CHUNK_LINES = 25
CHUNK_CHARS = 600
BATCH_SIZE = 10  # embed in small batches to avoid timeouts


def _root() -> Path:
    root = os.environ.get("OLLAMACODE_FS_ROOT")
    return Path(root).resolve() if root else Path.cwd().resolve()


def _cache_path() -> Path:
    return _root() / ".ollamacode" / "embeddings.json"


def _embed_model() -> str:
    return os.environ.get("OLLAMACODE_EMBED_MODEL", "nomic-embed-text")


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _chunk_text(text: str, path: str) -> list[tuple[int, str]]:
    """Return list of (start_line_1based, chunk_text)."""
    lines = text.splitlines()
    chunks: list[tuple[int, int, str]] = []
    for i in range(0, len(lines), CHUNK_LINES):
        block = lines[i : i + CHUNK_LINES]
        chunk = "\n".join(block)
        if len(chunk) > CHUNK_CHARS:
            chunk = chunk[: CHUNK_CHARS] + "\n..."
        if chunk.strip():
            chunks.append((i + 1, i + len(block), chunk))
    return [(start, c) for start, _, c in chunks]


@mcp.tool()
def index_codebase(file_pattern: str = "*") -> str:
    """
    Index the workspace for semantic search using Ollama embeddings.
    Scans files matching file_pattern (default '*' = all), chunks them, embeds via Ollama, and caches to .ollamacode/embeddings.json.
    Requires Ollama running with an embedding model (set OLLAMACODE_EMBED_MODEL, default nomic-embed-text). Run 'ollama pull nomic-embed-text' if needed.
    file_pattern: glob for files to index (e.g. '*.py', '*.ts'). Default '*' indexes all supported files.
    """
    try:
        import ollama
    except ImportError:
        return "Ollama Python client required. Install with: pip install ollama."

    root = _root()
    model = _embed_model()
    cache: dict[str, list[dict]] = {}

    # Collect chunks: path -> [(line_start, text), ...]
    chunks_by_path: dict[str, list[tuple[int, str]]] = {}
    try:
        pattern = file_pattern.strip() or "*"
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            if any(skip in path.parts for skip in SKIP_DIRS):
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except (OSError, UnicodeDecodeError):
                continue
            rel = str(path.relative_to(root))
            for line_start, chunk in _chunk_text(text, rel):
                chunks_by_path.setdefault(rel, []).append((line_start, chunk))
    except Exception as e:
        return f"Scan error: {e}"

    if not chunks_by_path:
        return "No files to index (or file_pattern matched nothing)."

    # Embed in batches and build cache
    all_texts: list[str] = []
    all_meta: list[tuple[str, int, str]] = []  # (path, line_start, text)
    for path, chunk_list in chunks_by_path.items():
        for line_start, text in chunk_list:
            all_texts.append(text)
            all_meta.append((path, line_start, text))

    try:
        for i in range(0, len(all_texts), BATCH_SIZE):
            batch = all_texts[i : i + BATCH_SIZE]
            resp = ollama.embed(model=model, input=batch)
            # resp.embeddings is list of lists (one per input)
            embs = getattr(resp, "embeddings", None) or []
            for j, (path, line_start, text) in enumerate(all_meta[i : i + len(batch)]):
                if j < len(embs):
                    cache.setdefault(path, []).append(
                        {"line": line_start, "text": text[:300], "embedding": embs[j]}
                    )
    except Exception as e:
        return f"Embed error: {e}. Ensure Ollama is running and you have an embedding model (e.g. ollama pull {model})."

    cache_dir = root / ".ollamacode"
    cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_path().write_text(json.dumps(cache, indent=0), encoding="utf-8")
    total = sum(len(v) for v in cache.values())
    return f"Indexed {len(cache)} files, {total} chunks. Cache: {_cache_path()}"


@mcp.tool()
def semantic_search_codebase(query: str, max_results: int = 20) -> str:
    """
    Semantic (meaning-based) search over the indexed codebase.
    Embeds the query via Ollama and returns the most similar chunks by cosine similarity.
    Run index_codebase first. Requires Ollama with an embedding model (OLLAMACODE_EMBED_MODEL, default nomic-embed-text).
    query: Natural language or code phrase (e.g. 'where is auth validated', 'config loading').
    max_results: Max number of results (default 20).
    """
    try:
        import ollama
    except ImportError:
        return "Ollama Python client required. Install with: pip install ollama."

    cache_file = _cache_path()
    if not cache_file.exists():
        return "No index found. Run index_codebase first (e.g. index_codebase with default file_pattern)."

    try:
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception as e:
        return f"Cache read error: {e}"

    model = _embed_model()
    try:
        resp = ollama.embed(model=model, input=query)
        q_emb = (getattr(resp, "embeddings", None) or [[]])[0]
    except Exception as e:
        return f"Embed error: {e}. Ensure Ollama is running with an embedding model (e.g. ollama pull {model})."

    if not q_emb:
        return "Empty query embedding."

    results: list[tuple[float, str, int, str]] = []
    for path, entries in cache.items():
        for ent in entries:
            emb = ent.get("embedding") or []
            if len(emb) != len(q_emb):
                continue
            score = _cosine(q_emb, emb)
            line = ent.get("line", 0)
            text = (ent.get("text") or "")[:200]
            results.append((score, path, line, text))

    results.sort(key=lambda x: -x[0])
    top = results[: max(1, max_results)]
    if not top:
        return "No matching chunks."
    lines = [f"{path}:{line}: (score {s:.3f}) {text}" for s, path, line, text in top]
    return "\n".join(lines)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
