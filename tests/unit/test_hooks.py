"""Unit tests for hooks.py — command injection prevention, config loading, decisions."""

import asyncio
import json
from unittest.mock import patch


from ollamacode.hooks import (
    _DANGEROUS_SHELL_PATTERNS,
    _load_hook_config,
    _match,
    _parse_decision,
    _run_command_hook,
)


class TestDangerousShellPatterns:
    """Test that dangerous shell metacharacters are detected."""

    def test_semicolon_detected(self):
        assert _DANGEROUS_SHELL_PATTERNS.search("echo; rm -rf /")

    def test_pipe_detected(self):
        assert _DANGEROUS_SHELL_PATTERNS.search("cat file | bash")

    def test_dollar_detected(self):
        assert _DANGEROUS_SHELL_PATTERNS.search("echo $HOME")

    def test_backtick_detected(self):
        assert _DANGEROUS_SHELL_PATTERNS.search("echo `whoami`")

    def test_subshell_detected(self):
        assert _DANGEROUS_SHELL_PATTERNS.search("echo $(whoami)")

    def test_safe_command_not_flagged(self):
        assert _DANGEROUS_SHELL_PATTERNS.search("python -m pytest") is None
        assert _DANGEROUS_SHELL_PATTERNS.search("ruff check .") is None


class TestMatch:
    """Test hook matcher pattern."""

    def test_exact_match(self):
        assert _match("write_file", "write_file") is True
        assert _match("write_file", "edit_file") is False

    def test_regex_match(self):
        assert _match("write_file|edit_file", "write_file") is True
        assert _match("write_file|edit_file", "edit_file") is True
        assert _match("write_file|edit_file", "read_file") is False

    def test_wildcard_match(self):
        assert _match(".*", "anything") is True

    def test_empty_matcher(self):
        assert _match("", "write_file") is False


class TestParseDecision:
    """Test hook decision parsing."""

    def test_allow_decision(self):
        d = _parse_decision({"behavior": "allow"})
        assert d is not None
        assert d.behavior == "allow"

    def test_deny_decision(self):
        d = _parse_decision({"behavior": "deny", "message": "Blocked by policy"})
        assert d is not None
        assert d.behavior == "deny"
        assert d.message == "Blocked by policy"

    def test_modify_decision(self):
        d = _parse_decision(
            {
                "behavior": "modify",
                "updatedInput": {"path": "/safe/path"},
            }
        )
        assert d is not None
        assert d.behavior == "modify"
        assert d.updated_input == {"path": "/safe/path"}

    def test_nested_decision(self):
        d = _parse_decision({"decision": {"behavior": "deny"}})
        assert d is not None
        assert d.behavior == "deny"

    def test_non_dict_returns_none(self):
        assert _parse_decision("invalid") is None
        assert _parse_decision(42) is None
        assert _parse_decision(None) is None


class TestLoadHookConfig:
    """Test hook config loading."""

    def test_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMACODE_DISABLE_HOOKS", "1")
        config = _load_hook_config(None)
        assert config == {"hooks": {}}

    def test_loads_from_workspace(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OLLAMACODE_DISABLE_HOOKS", raising=False)
        hooks_dir = tmp_path / ".ollamacode"
        hooks_dir.mkdir()
        hooks_file = hooks_dir / "hooks.json"
        hooks_file.write_text(
            json.dumps(
                {
                    "hooks": {
                        "PreToolUse": [
                            {
                                "matcher": "write_file",
                                "hooks": [{"type": "command", "command": "echo ok"}],
                            }
                        ]
                    }
                }
            )
        )
        # Patch home to avoid loading real user hooks
        with patch("ollamacode.hooks.Path.home", return_value=tmp_path / "fakehome"):
            config = _load_hook_config(str(tmp_path))
        assert "PreToolUse" in config["hooks"]
        assert len(config["hooks"]["PreToolUse"]) == 1

    def test_missing_file_ok(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OLLAMACODE_DISABLE_HOOKS", raising=False)
        with patch("ollamacode.hooks.Path.home", return_value=tmp_path / "fakehome"):
            config = _load_hook_config(str(tmp_path))
        assert config == {"hooks": {}}


class TestRunCommandHook:
    """Test command hook execution."""

    def test_dangerous_command_rejected(self):
        result = asyncio.get_event_loop().run_until_complete(
            _run_command_hook("echo; rm -rf /", {}, 10.0, None)
        )
        assert result is None

    def test_empty_command_returns_none(self):
        result = asyncio.get_event_loop().run_until_complete(
            _run_command_hook("", {}, 10.0, None)
        )
        assert result is None

    def test_safe_command_runs(self, tmp_path):
        result = asyncio.get_event_loop().run_until_complete(
            _run_command_hook(
                "echo ok",
                {"tool": "test"},
                10.0,
                str(tmp_path),
            )
        )
        # echo doesn't output valid JSON decision, so returns None
        assert result is None
