"""Unit tests for config loading and merging."""

import sys
from unittest.mock import patch

from ollamacode.config import (
    DEFAULT_MCP_SERVERS,
    find_config_file,
    load_config,
    merge_config_with_env,
)


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
    config_file.write_text(
        "model: llama3.2\nmcp_servers:\n  - type: stdio\n    command: python\n    args: [a.py]\n"
    )
    result = load_config(config_path=str(config_file))
    assert result.get("model") == "llama3.2"
    assert result.get("mcp_servers") == [
        {"type": "stdio", "command": "python", "args": ["a.py"]}
    ]


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
    assert out["mcp_servers"] == [
        {"type": "stdio", "command": "python", "args": ["server.py"]}
    ]


def test_merge_config_with_env_config_wins_when_env_empty():
    """merge_config_with_env uses config when env not set; custom mcp_servers get built-in prepended by default."""
    config = {
        "model": "from-config",
        "mcp_servers": [{"type": "stdio", "command": "npx", "args": ["mcp"]}],
    }
    out = merge_config_with_env(
        config, model_env=None, mcp_args_env=None, system_extra_env=None
    )
    assert out["model"] == "from-config"
    # By default include_builtin_servers is True: built-in (fs, terminal, codebase, tools, git) + custom
    assert len(out["mcp_servers"]) == len(DEFAULT_MCP_SERVERS) + 1
    assert out["mcp_servers"][: len(DEFAULT_MCP_SERVERS)] == DEFAULT_MCP_SERVERS
    assert out["mcp_servers"][-1] == {
        "type": "stdio",
        "command": "npx",
        "args": ["mcp"],
    }


def test_merge_config_include_builtin_servers_false():
    """When include_builtin_servers: false, only config mcp_servers are used (no built-in)."""
    config = {
        "mcp_servers": [{"type": "stdio", "command": "npx", "args": ["mcp"]}],
        "include_builtin_servers": False,
    }
    out = merge_config_with_env(
        config, model_env=None, mcp_args_env=None, system_extra_env=None
    )
    assert out["mcp_servers"] == [{"type": "stdio", "command": "npx", "args": ["mcp"]}]


def test_merge_config_deduplicate_builtin_when_in_custom():
    """When custom mcp_servers includes a server equivalent to a built-in, that built-in is not prepended (no duplicate tools)."""
    # Custom list includes fs_mcp (same as first built-in) plus a demo server
    config = {
        "mcp_servers": [
            {
                "type": "stdio",
                "command": sys.executable,
                "args": ["-m", "ollamacode.servers.fs_mcp"],
            },
            {"type": "stdio", "command": "python", "args": ["examples/demo_server.py"]},
        ],
    }
    out = merge_config_with_env(
        config, model_env=None, mcp_args_env=None, system_extra_env=None
    )
    # Built-in list has 5: fs, terminal, codebase, tools, git. We skip fs because it's in custom.
    assert len(out["mcp_servers"]) == 4 + 2  # 4 built-in (no fs) + 2 custom
    mods = [s.get("args", [])[-1] if s.get("args") else "" for s in out["mcp_servers"]]
    # First 4 should be terminal, codebase, tools, git (no fs)
    assert "ollamacode.servers.terminal_mcp" in mods
    assert "ollamacode.servers.codebase_mcp" in mods
    assert "ollamacode.servers.tools_mcp" in mods
    assert "ollamacode.servers.git_mcp" in mods
    assert mods.count("ollamacode.servers.fs_mcp") == 1  # only from custom
    assert "examples/demo_server.py" in mods


def test_merge_config_mcp_args_env_overrides_config():
    """When mcp_args_env is set, mcp_servers come from env (legacy single stdio)."""
    config = {"mcp_servers": [{"type": "stdio", "command": "x", "args": []}]}
    out = merge_config_with_env(config, mcp_args_env="python demo.py")
    assert out["mcp_servers"] == [
        {"type": "stdio", "command": "python", "args": ["demo.py"]}
    ]


def test_merge_config_empty_config_no_env_uses_default_servers():
    """When config has no mcp_servers and env has no MCP args, built-in servers are used."""
    out = merge_config_with_env(
        {}, model_env=None, mcp_args_env=None, system_extra_env=None
    )
    assert out["mcp_servers"] == DEFAULT_MCP_SERVERS
    assert len(out["mcp_servers"]) == 5
    mods = [s["args"][1] for s in out["mcp_servers"] if s["args"]]
    assert "ollamacode.servers.fs_mcp" in mods
    assert "ollamacode.servers.terminal_mcp" in mods
    assert "ollamacode.servers.codebase_mcp" in mods
    assert "ollamacode.servers.tools_mcp" in mods
    assert "ollamacode.servers.git_mcp" in mods


def test_merge_config_explicit_empty_mcp_servers():
    """When config has mcp_servers: [], no default servers; MCP disabled."""
    out = merge_config_with_env({"mcp_servers": []}, mcp_args_env=None)
    assert out["mcp_servers"] == []


def test_merge_config_max_tool_rounds():
    """merge_config_with_env returns max_tool_rounds from config or default 20."""
    out = merge_config_with_env(
        {}, model_env=None, mcp_args_env=None, system_extra_env=None
    )
    assert out["max_tool_rounds"] == 20
    out2 = merge_config_with_env({"max_tool_rounds": 10}, mcp_args_env=None)
    assert out2["max_tool_rounds"] == 10


def test_find_config_file_lookup_parent_dirs(tmp_path):
    """find_config_file with lookup_parent_dirs=True finds config in parent dir."""
    subdir = tmp_path / "sub" / "deep"
    subdir.mkdir(parents=True)
    config_in_parent = tmp_path / "ollamacode.yaml"
    config_in_parent.write_text("model: from-parent\n")
    found = find_config_file(None, cwd=subdir, lookup_parent_dirs=True)
    assert found is not None
    assert found == config_in_parent.resolve()
    found_none = find_config_file(None, cwd=subdir, lookup_parent_dirs=False)
    assert found_none is None
