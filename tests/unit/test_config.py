"""Unit tests for config loading and merging."""

from unittest.mock import patch

from ollamacode.config import DEFAULT_MCP_SERVERS, load_config, merge_config_with_env


def test_load_config_no_file():
    """load_config returns {} when no config file exists."""
    with patch("ollamacode.config._find_config_file", return_value=None):
        assert load_config() == {}
    assert load_config("/nonexistent/path.yaml") == {}


def test_load_config_no_yaml(monkeypatch):
    """load_config returns {} when yaml is not available."""
    monkeypatch.setattr("ollamacode.config.yaml", None)
    assert load_config() == {}


def test_load_config_parses_yaml(tmp_path):
    """load_config parses YAML and returns dict."""
    config_file = tmp_path / "ollamacode.yaml"
    config_file.write_text("model: llama3.2\nmcp_servers:\n  - type: stdio\n    command: python\n    args: [a.py]\n")
    result = load_config(config_path=str(config_file))
    assert result.get("model") == "llama3.2"
    assert result.get("mcp_servers") == [{"type": "stdio", "command": "python", "args": ["a.py"]}]


def test_merge_config_with_env_empty():
    """merge_config_with_env with empty config uses env when provided."""
    out = merge_config_with_env(
        {},
        model_env="custom-model",
        mcp_args_env="python server.py",
        system_extra_env="Extra.",
    )
    assert out["model"] == "custom-model"
    assert out["system_prompt_extra"] == "Extra."
    assert out["mcp_servers"] == [{"type": "stdio", "command": "python", "args": ["server.py"]}]


def test_merge_config_with_env_config_wins_when_env_empty():
    """merge_config_with_env uses config when env not set."""
    config = {"model": "from-config", "mcp_servers": [{"type": "stdio", "command": "npx", "args": ["mcp"]}]}
    out = merge_config_with_env(config, model_env=None, mcp_args_env=None, system_extra_env=None)
    assert out["model"] == "from-config"
    assert out["mcp_servers"] == [{"type": "stdio", "command": "npx", "args": ["mcp"]}]


def test_merge_config_mcp_args_env_overrides_config():
    """When mcp_args_env is set, mcp_servers come from env (legacy single stdio)."""
    config = {"mcp_servers": [{"type": "stdio", "command": "x", "args": []}]}
    out = merge_config_with_env(config, mcp_args_env="python demo.py")
    assert out["mcp_servers"] == [{"type": "stdio", "command": "python", "args": ["demo.py"]}]


def test_merge_config_empty_config_no_env_uses_default_servers():
    """When config has no mcp_servers and env has no MCP args, built-in servers are used."""
    out = merge_config_with_env({}, model_env=None, mcp_args_env=None, system_extra_env=None)
    assert out["mcp_servers"] == DEFAULT_MCP_SERVERS
    assert len(out["mcp_servers"]) == 5
    mods = [s["args"][1] for s in out["mcp_servers"] if s["args"]]
    assert "ollamacode.servers.fs_mcp" in mods
    assert "ollamacode.servers.terminal_mcp" in mods
    assert "ollamacode.servers.codebase_mcp" in mods
    assert "ollamacode.servers.git_mcp" in mods
    assert "ollamacode.servers.tools_mcp" in mods


def test_merge_config_explicit_empty_mcp_servers():
    """When config has mcp_servers: [], no default servers; MCP disabled."""
    out = merge_config_with_env({"mcp_servers": []}, mcp_args_env=None)
    assert out["mcp_servers"] == []
