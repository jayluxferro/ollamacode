"""Integration test: CLI + demo MCP. Requires Ollama + tool-capable model (e.g. gpt-oss:20b). Can be slow (1–2 min)."""

import subprocess
import sys
from pathlib import Path

import pytest

# Repo root (parent of tests/)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _run_cli_with_demo_mcp(query: str, timeout: int = 150) -> tuple[str, str, int]:
    """Run ollamacode CLI with demo MCP server; return (stdout, stderr, returncode)."""
    demo_path = str(REPO_ROOT / "examples" / "demo_server.py")
    cmd = [
        sys.executable,
        "-m",
        "ollamacode",
        "--model",
        "gpt-oss:20b",
        "--mcp-command",
        sys.executable,
        "--mcp-args",
        demo_path,
        "--",
        query,
    ]
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout, result.stderr, result.returncode


def _ollama_available() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/version", timeout=2)
        return True
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
def test_cli_with_demo_mcp_what_is_2_plus_3():
    """CLI + demo MCP: query 'What is 2+3?' yields response containing '5' (model uses add tool)."""
    stdout, stderr, code = _run_cli_with_demo_mcp("What is 2+3?")
    assert code == 0, f"CLI failed: stderr={stderr!r}"
    assert "5" in stdout, f"Expected '5' in response. out={stdout!r} err={stderr!r}"
