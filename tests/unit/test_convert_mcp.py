"""Unit tests for MCP config converter (Cursor/Claude JSON -> OllamaCode YAML)."""

from ollamacode.convert_mcp import (
    convert_to_ollamacode_servers,
    load_json,
    run_convert,
)


def test_convert_cursor_stdio():
    """Cursor mcpServers with command+args -> OllamaCode stdio list with name from key."""
    data = {
        "mcpServers": {
            "git": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-git"]},
            "fs": {"command": "python", "args": ["-m", "ollamacode.servers.fs_mcp"]},
        }
    }
    out = convert_to_ollamacode_servers(data)
    assert len(out) == 2
    assert out[0]["name"] == "git"
    assert out[0]["type"] == "stdio"
    assert out[0]["command"] == "npx"
    assert out[0]["args"] == ["-y", "@modelcontextprotocol/server-git"]
    assert out[1]["name"] == "fs"
    assert out[1]["type"] == "stdio"
    assert out[1]["command"] == "python"
    assert out[1]["args"] == ["-m", "ollamacode.servers.fs_mcp"]


def test_convert_claude_mcp_servers():
    """Claude mcp_servers (object) -> OllamaCode list with name from key."""
    data = {
        "mcp_servers": {
            "git": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-git"]},
        }
    }
    out = convert_to_ollamacode_servers(data)
    assert len(out) == 1
    assert out[0]["name"] == "git"
    assert out[0]["type"] == "stdio"
    assert out[0]["command"] == "npx"


def test_convert_url_sse():
    """URL with /sse -> type sse, name from key."""
    data = {
        "mcpServers": {
            "remote": {"url": "http://localhost:8000/sse"},
        }
    }
    out = convert_to_ollamacode_servers(data)
    assert len(out) == 1
    assert out[0]["name"] == "remote"
    assert out[0]["type"] == "sse"
    assert out[0]["url"] == "http://localhost:8000/sse"


def test_convert_url_streamable_http():
    """URL without /sse -> type streamable_http, name from key."""
    data = {
        "mcpServers": {
            "remote": {"url": "http://localhost:8000/mcp"},
        }
    }
    out = convert_to_ollamacode_servers(data)
    assert len(out) == 1
    assert out[0]["name"] == "remote"
    assert out[0]["type"] == "streamable_http"
    assert out[0]["url"] == "http://localhost:8000/mcp"


def test_convert_env_preserved():
    """Stdio server with env -> env preserved, name from key."""
    data = {
        "mcpServers": {
            "fs": {
                "command": "python",
                "args": ["-m", "ollamacode.servers.fs_mcp"],
                "env": {"OLLAMACODE_FS_ROOT": "/path"},
            },
        }
    }
    out = convert_to_ollamacode_servers(data)
    assert out[0]["name"] == "fs"
    assert out[0].get("env") == {"OLLAMACODE_FS_ROOT": "/path"}


def test_convert_empty_returns_empty():
    """No mcpServers/mcp_servers -> empty list."""
    assert convert_to_ollamacode_servers({}) == []
    assert convert_to_ollamacode_servers({"other": 1}) == []


def test_load_json_from_path(tmp_path):
    """load_json reads file when path given."""
    f = tmp_path / "mcp.json"
    f.write_text('{"mcpServers": {"a": {"command": "npx", "args": ["x"]}}}')
    data = load_json(str(f))
    assert data["mcpServers"]["a"]["command"] == "npx"


def test_run_convert_writes_yaml(tmp_path):
    """run_convert reads JSON and writes YAML with name from key."""
    inp = tmp_path / "in.json"
    out = tmp_path / "out.yaml"
    inp.write_text('{"mcpServers": {"git": {"command": "npx", "args": ["-y", "server-git"]}}}')
    run_convert(str(inp), str(out))
    text = out.read_text()
    assert "mcp_servers:" in text
    assert "name:" in text or "git" in text
    assert "npx" in text
    assert "server-git" in text
