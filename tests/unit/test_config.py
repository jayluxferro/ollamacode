"""Unit tests for config loading and merging."""

import sys
from unittest.mock import patch

import pytest

from ollamacode.config import (
    DEFAULT_MCP_SERVERS,
    ENV_CONFIG_SCHEMA,
    ConfigValidationError,
    find_config_file,
    get_env_config_overrides,
    get_resolved_config,
    load_config,
    merge_config_with_env,
    validate_config,
)
from ollamacode.config import _deep_merge


def test_load_config_no_file():
    """load_config returns {} when no config file exists."""
    with patch("ollamacode.config._find_config_file", return_value=None):
        assert load_config() == {}
    assert load_config("/nonexistent/path.yaml") == {}


def test_get_resolved_config_always_returns_full_dict():
    """get_resolved_config returns a dict with all keys even when no config file."""
    with patch("ollamacode.config._find_config_file", return_value=None):
        out = get_resolved_config()
    assert isinstance(out, dict)
    assert "max_tool_rounds" in out
    assert "mcp_servers" in out
    assert out["max_tool_rounds"] == 20


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
    assert out["inject_recent_context"] is True
    assert out["recent_context_max_files"] == 10
    assert out["memory_auto_context"] is True
    assert out["memory_kg_max_results"] == 4
    assert out["memory_rag_max_results"] == 4
    assert out["memory_rag_snippet_chars"] == 220


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
    # Built-in list has fs, terminal, codebase, tools, git, skills, state, reasoning, screenshot. We skip fs because it's in custom.
    assert (
        len(out["mcp_servers"]) == (len(DEFAULT_MCP_SERVERS) - 1) + 2
    )  # built-in minus fs + 2 custom
    mods = [s.get("args", [])[-1] if s.get("args") else "" for s in out["mcp_servers"]]
    # First 5 should be terminal, codebase, tools, git, skills (no fs)
    assert "ollamacode.servers.terminal_mcp" in mods
    assert "ollamacode.servers.codebase_mcp" in mods
    assert "ollamacode.servers.tools_mcp" in mods
    assert "ollamacode.servers.git_mcp" in mods
    assert "ollamacode.servers.skills_mcp" in mods
    assert "ollamacode.servers.state_mcp" in mods
    assert mods.count("ollamacode.servers.fs_mcp") == 1  # only from custom
    assert "examples/demo_server.py" in mods


def test_merge_config_mcp_args_env_overrides_config():
    """When mcp_args_env is set, mcp_servers come from env (legacy single stdio)."""
    config = {"mcp_servers": [{"type": "stdio", "command": "x", "args": []}]}
    out = merge_config_with_env(config, mcp_args_env="python demo.py")
    assert out["mcp_servers"] == [
        {"type": "stdio", "command": "python", "args": ["demo.py"]}
    ]


def test_merge_config_python_executable_override():
    """merge_config_with_env uses python_executable for built-in stdio servers when provided."""
    out = merge_config_with_env(
        {},
        model_env=None,
        mcp_args_env=None,
        system_extra_env=None,
        python_executable="/opt/python3.11",
    )
    assert out["mcp_servers"]
    for entry in out["mcp_servers"]:
        if (entry.get("type") or "stdio").lower() == "stdio":
            assert entry.get("command") == "/opt/python3.11"


def test_merge_config_empty_config_no_env_uses_default_servers():
    """When config has no mcp_servers and env has no MCP args, built-in servers are used."""
    out = merge_config_with_env(
        {}, model_env=None, mcp_args_env=None, system_extra_env=None
    )
    assert out["mcp_servers"] == DEFAULT_MCP_SERVERS
    assert len(out["mcp_servers"]) == len(DEFAULT_MCP_SERVERS)
    mods = [s["args"][1] for s in out["mcp_servers"] if s["args"]]
    assert "ollamacode.servers.fs_mcp" in mods
    assert "ollamacode.servers.terminal_mcp" in mods
    assert "ollamacode.servers.codebase_mcp" in mods
    assert "ollamacode.servers.tools_mcp" in mods
    assert "ollamacode.servers.git_mcp" in mods
    assert "ollamacode.servers.skills_mcp" in mods
    assert "ollamacode.servers.state_mcp" in mods


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


def test_merge_config_rlm_clamping():
    """merge_config_with_env clamps invalid rlm_* values to safe ranges."""
    out = merge_config_with_env(
        {
            "rlm_max_iterations": -1,
            "rlm_stdout_max_chars": -100,
            "rlm_prefix_chars": 100000,
        },
        model_env=None,
        mcp_args_env=None,
        system_extra_env=None,
    )
    assert out["rlm_max_iterations"] == 1
    assert out["rlm_stdout_max_chars"] == 0
    assert out["rlm_prefix_chars"] == 50_000  # max clamp
    out2 = merge_config_with_env(
        {"rlm_use_subprocess": True, "rlm_subprocess_max_memory_mb": 10},
        model_env=None,
        mcp_args_env=None,
        system_extra_env=None,
    )
    assert out2["rlm_use_subprocess"] is True
    assert out2["rlm_subprocess_max_memory_mb"] == 64  # clamped from 10 to min 64


def test_merge_config_memory_clamping():
    """merge_config_with_env clamps memory retrieval knobs to safe ranges."""
    out = merge_config_with_env(
        {
            "memory_auto_context": False,
            "memory_kg_max_results": -1,
            "memory_rag_max_results": 999,
            "memory_rag_snippet_chars": 10,
        },
        model_env=None,
        mcp_args_env=None,
        system_extra_env=None,
    )
    assert out["memory_auto_context"] is False
    assert out["memory_kg_max_results"] == 0
    assert out["memory_rag_max_results"] == 20
    assert out["memory_rag_snippet_chars"] == 40


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


def test_get_resolved_config_with_config_path(tmp_path):
    """get_resolved_config(config_path=...) loads file and returns merged config with defaults."""
    config_file = tmp_path / "ollamacode.yaml"
    config_file.write_text(
        "model: my-model\nmax_tool_rounds: 5\nmcp_servers:\n  - type: stdio\n    command: python\n    args: [a.py]\n"
    )
    out = get_resolved_config(config_path=str(config_file))
    assert out["model"] == "my-model"
    assert out["max_tool_rounds"] == 5
    custom = {"type": "stdio", "command": "python", "args": ["a.py"]}
    assert custom in out["mcp_servers"]
    assert "timing" in out
    assert out["timing"] is False


def test_default_mcp_servers_structure():
    """DEFAULT_MCP_SERVERS entries have type stdio, command, and args (built-in MCP)."""
    assert len(DEFAULT_MCP_SERVERS) >= 5
    for entry in DEFAULT_MCP_SERVERS:
        assert entry.get("type") == "stdio"
        assert "command" in entry
        assert "args" in entry
        assert isinstance(entry["args"], list)


def test_validate_config_empty_passes():
    """validate_config(empty dict) does not raise."""
    validate_config({})


def test_validate_config_mcp_servers_not_list_raises():
    """validate_config raises ConfigValidationError when mcp_servers is not a list."""
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config({"mcp_servers": "not a list"})
    assert "mcp_servers must be a list" in str(exc_info.value)
    assert exc_info.value.errors


def test_validate_config_mcp_servers_stdio_missing_command_raises():
    """validate_config raises when stdio entry has no command."""
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config({"mcp_servers": [{"type": "stdio", "args": ["x"]}]})
    assert "command" in str(exc_info.value).lower()


def test_validate_config_mcp_servers_sse_missing_url_raises():
    """validate_config raises when sse entry has no url."""
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config({"mcp_servers": [{"type": "sse"}]})
    assert "url" in str(exc_info.value).lower()


def test_validate_config_mcp_servers_valid_passes():
    """validate_config passes for valid mcp_servers."""
    validate_config(
        {"mcp_servers": [{"type": "stdio", "command": "python", "args": ["a.py"]}]}
    )
    validate_config({"mcp_servers": [{"type": "sse", "url": "http://localhost/sse"}]})


def test_merge_config_with_env_invalid_mcp_servers_raises():
    """merge_config_with_env raises ConfigValidationError when config has invalid mcp_servers."""
    with pytest.raises(ConfigValidationError):
        merge_config_with_env({"mcp_servers": "invalid"}, mcp_args_env=None)


def test_env_config_schema_declares_merge_params():
    """ENV_CONFIG_SCHEMA lists env var -> merge param mapping."""
    assert len(ENV_CONFIG_SCHEMA) >= 4
    env_vars = [ev for ev, _ in ENV_CONFIG_SCHEMA]
    assert "OLLAMACODE_MODEL" in env_vars
    assert "OLLAMACODE_MCP_ARGS" in env_vars
    assert "OLLAMACODE_SYSTEM_EXTRA" in env_vars
    assert "OLLAMACODE_PYTHON" in env_vars


def test_get_env_config_overrides_returns_merge_kwargs():
    """get_env_config_overrides(env) returns dict suitable for merge_config_with_env(**overrides)."""
    env = {
        "OLLAMACODE_MODEL": "my-model",
        "OLLAMACODE_MCP_ARGS": "python server.py",
        "OLLAMACODE_SYSTEM_EXTRA": "Extra.",
        "OLLAMACODE_PYTHON": "/usr/bin/python3",
    }
    overrides = get_env_config_overrides(env)
    assert overrides["model_env"] == "my-model"
    assert overrides["mcp_args_env"] == "python server.py"
    assert overrides["system_extra_env"] == "Extra."
    assert overrides["python_executable"] == "/usr/bin/python3"
    merged = merge_config_with_env({}, **overrides)
    assert merged["model"] == "my-model"
    assert merged["system_prompt_extra"] == "Extra."
    assert merged["mcp_servers"] == [
        {"type": "stdio", "command": "python", "args": ["server.py"]}
    ]


def test_get_env_config_overrides_empty_env():
    """get_env_config_overrides with empty env returns None for all params."""
    overrides = get_env_config_overrides({})
    assert overrides["model_env"] is None
    assert overrides["mcp_args_env"] is None


def test_deep_merge():
    """_deep_merge overlays override onto base; nested dicts merged recursively."""
    base = {"a": 1, "b": {"x": 1, "y": 2}}
    override = {"b": {"y": 20, "z": 3}, "c": 4}
    out = _deep_merge(base, override)
    assert out["a"] == 1
    assert out["b"]["x"] == 1
    assert out["b"]["y"] == 20
    assert out["b"]["z"] == 3
    assert out["c"] == 4


def test_load_config_user_then_project_merge(tmp_path):
    """load_config merges user config as base, then project overrides (deep-merge)."""
    user_file = tmp_path / "user.yaml"
    user_file.write_text("model: user-model\nmax_tool_rounds: 10\n")
    proj_file = tmp_path / "proj.yaml"
    proj_file.write_text("model: proj-model\n")  # override model only
    with patch("ollamacode.config._user_config_path", return_value=user_file):
        with patch("ollamacode.config._find_config_file", return_value=proj_file):
            result = load_config(config_path=str(proj_file))
    assert result.get("model") == "proj-model"
    assert result.get("max_tool_rounds") == 10  # from user (not overridden)
