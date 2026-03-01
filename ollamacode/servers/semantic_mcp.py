"""
Optional semantic codebase MCP server: index_codebase, semantic_search_codebase.

Uses Ollama embeddings (e.g. nomic-embed-text). Cache: .ollamacode/embeddings.json in workspace root.
Root directory: OLLAMACODE_FS_ROOT env var, or current working directory.
Add via config; not in default built-in list. Requires: ollama pull nomic-embed-text (or OLLAMACODE_EMBED_MODEL).
"""

import concurrent.futures
import json
import logging
import os
import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from . import configure_server_logging

configure_server_logging()

logger = logging.getLogger(__name__)

_MAX_MODEL_NAME_LEN = 200
_MODEL_NAME_RE = re.compile(r"^[a-zA-Z0-9._:/@-]+$")

mcp = FastMCP("ollamacode-semantic")

SKIP_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    ".ollamacode",
}
MAX_FILE_BYTES = 100_000
MAX_CHARS_PER_ENTRY = 8_000
EMBED_BATCH = 10
EMBED_TIMEOUT_SECONDS = 120
DEFAULT_EMBED_MODEL = "nomic-embed-text"
_CACHE_VERSION = 2  # Increment when cache format changes


def _root() -> Path:
    root = os.environ.get("OLLAMACODE_FS_ROOT")
    return Path(root).resolve() if root else Path.cwd().resolve()


def _cache_path() -> Path:
    return _root() / ".ollamacode" / "embeddings.json"


def _validate_model_name(name: str) -> str | None:
    """Return an error message if the model name is invalid, else None."""
    if len(name) > _MAX_MODEL_NAME_LEN:
        return f"Model name too long ({len(name)} chars, max {_MAX_MODEL_NAME_LEN})"
    if not _MODEL_NAME_RE.match(name):
        return f"Model name contains invalid characters: {name!r}"
    return None


def _embed_model() -> str:
    raw = os.environ.get("OLLAMACODE_EMBED_MODEL", "").strip()
    if not raw:
        return DEFAULT_EMBED_MODEL
    err = _validate_model_name(raw)
    if err:
        logger.warning("Invalid embed model name, using default: %s", err)
        return DEFAULT_EMBED_MODEL
    return raw


def _embed_with_timeout(
    ollama_mod: object,
    model: str,
    input_data: object,
    timeout: int = EMBED_TIMEOUT_SECONDS,
) -> object:
    """Call ollama.embed with a timeout to prevent hangs."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(ollama_mod.embed, model=model, input=input_data)  # type: ignore[union-attr]
        return future.result(timeout=timeout)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@mcp.tool()
def index_codebase(file_pattern: str = "*") -> str:
    """
    Index the workspace with Ollama embeddings for semantic search.
    file_pattern: glob for files to index (default '*' = all). Examples: '*.py', '*.ts'.
    Reads files under workspace root, computes embeddings, and caches them in .ollamacode/embeddings.json.
    Run this before using semantic_search_codebase. Requires Ollama with an embedding model (e.g. nomic-embed-text).
    """
    root = _root()
    pattern = (file_pattern or "*").strip()
    model = _embed_model()
    entries: list[dict] = []

    try:
        import ollama
    except ImportError:
        return "Ollama Python package not installed. pip install ollama."

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
        text = text.strip()[:MAX_CHARS_PER_ENTRY]
        if not text:
            continue
        entries.append({"path": rel, "text": text})

    if not entries:
        return f"No files matched {pattern!r} under {root}."

    texts = [e["text"] for e in entries]
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        try:
            r = _embed_with_timeout(ollama, model, batch)
            embs = (
                r.get("embeddings")
                if isinstance(r, dict)
                else getattr(r, "embeddings", None)
            )
            if embs:
                all_embeddings.extend(embs)
            else:
                all_embeddings.extend([[]] * len(batch))
        except concurrent.futures.TimeoutError:
            return f"Embedding timed out after {EMBED_TIMEOUT_SECONDS}s (batch {i // EMBED_BATCH + 1}). Is Ollama responsive?"
        except Exception as e:
            return f"Embedding error (is {model} pulled?): {e}"

    for e, emb in zip(entries, all_embeddings):
        e["embedding"] = emb

    cache_file = _cache_path()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(
            {"version": _CACHE_VERSION, "model": model, "entries": entries}, f, indent=0
        )

    return f"Indexed {len(entries)} file(s) with {model}. Cache: {cache_file}"


@mcp.tool()
def semantic_search_codebase(query: str, max_results: int = 10) -> str:
    """
    Search the workspace by meaning using the embedding cache.
    query: natural language question (e.g. 'where is config loaded', 'run tests or pytest').
    max_results: max number of results to return (default 10).
    Requires index_codebase to have been run first. Uses Ollama embedding model (e.g. nomic-embed-text).
    """
    cache_file = _cache_path()
    if not cache_file.is_file():
        return "No index found. Run index_codebase first (e.g. 'Index the codebase' or 'Run index_codebase for this project')."

    model = _embed_model()
    try:
        with open(cache_file, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return f"Failed to load cache: {e}"

    # Verify cache version and model match
    cache_version = data.get("version", 1)
    cache_model = data.get("model", "")
    if cache_version != _CACHE_VERSION:
        return (
            f"Cache version mismatch (cache={cache_version}, expected={_CACHE_VERSION}). "
            "Run index_codebase again to rebuild."
        )
    if cache_model and cache_model != model:
        return (
            f"Cache was built with model '{cache_model}' but current model is '{model}'. "
            "Run index_codebase again to rebuild with the current model."
        )

    entries = data.get("entries") or []
    if not entries:
        return "Index is empty. Run index_codebase again."

    try:
        import ollama
    except ImportError:
        return "Ollama Python package not installed. pip install ollama."

    try:
        r = _embed_with_timeout(ollama, model, query.strip() or " ")
        qemb = (
            r.get("embeddings")
            if isinstance(r, dict)
            else getattr(r, "embeddings", None)
        ) or []
        qvec = qemb[0] if qemb else []
    except concurrent.futures.TimeoutError:
        return (
            f"Embedding timed out after {EMBED_TIMEOUT_SECONDS}s. Is Ollama responsive?"
        )
    except Exception as e:
        return f"Embedding error (is {model} pulled?): {e}"

    if not qvec:
        return "Could not embed query."

    scored = []
    for e in entries:
        emb = e.get("embedding") or []
        if not emb:
            continue
        score = _cosine(qvec, emb)
        scored.append((score, e["path"], (e.get("text") or "")[:500]))

    scored.sort(key=lambda x: -x[0])
    top = scored[:max_results]
    if not top:
        return "No results."
    lines = [
        f"{path} (score={s:.3f}):\n{snippet[:300]}..."
        if len(snippet) > 300
        else f"{path} (score={s:.3f}):\n{snippet}"
        for s, path, snippet in top
    ]
    return "\n\n".join(lines)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
