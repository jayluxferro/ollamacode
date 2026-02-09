"""E2E: CLI with demo MCP → query → expect response. Requires Ollama + tool-capable model. Run with -m e2e or integration."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _ollama_available() -> bool:
    try:
        import urllib.request

        urllib.request.urlopen("http://localhost:11434/api/version", timeout=2)
        return True
    except Exception:
        return False


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
def test_cli_demo_mcp_full_flow():
    """E2E: CLI + demo MCP; query 'What is 2+3?'; expect response containing '5' (model uses add tool)."""
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
        "What is 2+3?",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=150,
    )
    assert result.returncode == 0, f"CLI failed: stderr={result.stderr!r}"
    assert (
        "5" in result.stdout
    ), f"Expected '5' in response. out={result.stdout!r} err={result.stderr!r}"
