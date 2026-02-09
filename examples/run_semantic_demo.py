#!/usr/bin/env python3
"""
Demo: run semantic codebase index + search using nomic-embed-text.
From repo root: uv run python examples/run_semantic_demo.py
Requires: ollama pull nomic-embed-text
"""

# ruff: noqa

import sys
from pathlib import Path

# Ensure the repo root is on the import path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

# Import after path manipulation; ignore E402 for this file
from ollamacode.servers.semantic_mcp import (
    index_codebase,
    semantic_search_codebase,
)  # noqa: E402


def main():
    print("=== 1. Indexing Python files (ollamacode/*.py) ===\n")
    out = index_codebase("*.py")
    print(out)
    print()

    print("=== 2. Semantic search: 'where is config loaded' ===\n")
    out = semantic_search_codebase("where is config file loaded", max_results=8)
    print(out)
    print()

    print("=== 3. Semantic search: 'run tests or pytest' ===\n")
    out = semantic_search_codebase("run tests or pytest", max_results=5)
    print(out)


if __name__ == "__main__":
    main()
