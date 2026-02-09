"""Integration: CLI with config and multiple MCP servers. Requires Ollama + tool-capable model."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _ollama_available() -> bool:
    try:
        import urllib.request

        urllib.request.urlopen("http://localhost:11434/api/version", timeout=2)
        return True
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
def test_cli_with_config_two_mcp_servers(tmp_path):
    """CLI with config listing two stdio MCP servers; query uses first server's add tool."""
    config_path = tmp_path / "ollamacode.yaml"
    demo_path = REPO_ROOT / "examples" / "demo_server.py"
    fs_path = REPO_ROOT / "examples" / "fs_mcp.py"
    cfg = {
        "model": "gpt-oss:20b",
        "include_builtin_servers": False,  # test only demo + fs; built-in fs would duplicate examples/fs_mcp (same FastMCP name)
        "mcp_servers": [
            {"type": "stdio", "command": sys.executable, "args": [str(demo_path)]},
            {"type": "stdio", "command": sys.executable, "args": [str(fs_path)]},
        ],
    }
    config_path.write_text(yaml.safe_dump(cfg))
    cmd = [
        sys.executable,
        "-m",
        "ollamacode",
        "--config",
        str(config_path),
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
    assert "5" in result.stdout, f"Expected '5' in response. out={result.stdout!r} err={result.stderr!r}"
