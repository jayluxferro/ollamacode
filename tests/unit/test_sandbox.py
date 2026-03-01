"""Unit tests for sandbox.py — all levels, path checks, command checks."""

import pytest

from ollamacode.sandbox import (
    SandboxLevel,
    check_fs_path,
    check_terminal_command,
    get_sandbox_level,
)


# ---------------------------------------------------------------------------
# SandboxLevel enum
# ---------------------------------------------------------------------------


class TestSandboxLevelEnum:
    def test_values(self):
        assert SandboxLevel.READONLY.value == "readonly"
        assert SandboxLevel.SUPERVISED.value == "supervised"
        assert SandboxLevel.FULL.value == "full"

    def test_is_str_enum(self):
        assert isinstance(SandboxLevel.READONLY, str)
        assert SandboxLevel.FULL == "full"


# ---------------------------------------------------------------------------
# get_sandbox_level
# ---------------------------------------------------------------------------


class TestGetSandboxLevel:
    def test_default_is_supervised(self, monkeypatch):
        monkeypatch.delenv("OLLAMACODE_SANDBOX_LEVEL", raising=False)
        assert get_sandbox_level() == SandboxLevel.SUPERVISED

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMACODE_SANDBOX_LEVEL", "readonly")
        assert get_sandbox_level() == SandboxLevel.READONLY

    def test_full_from_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMACODE_SANDBOX_LEVEL", "full")
        assert get_sandbox_level() == SandboxLevel.FULL

    def test_invalid_falls_back_to_supervised(self, monkeypatch):
        monkeypatch.setenv("OLLAMACODE_SANDBOX_LEVEL", "invalid_level")
        assert get_sandbox_level() == SandboxLevel.SUPERVISED

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("OLLAMACODE_SANDBOX_LEVEL", "  READONLY  ")
        assert get_sandbox_level() == SandboxLevel.READONLY


# ---------------------------------------------------------------------------
# check_fs_path
# ---------------------------------------------------------------------------


class TestCheckFsPath:
    def test_null_byte_rejected_at_all_levels(self, tmp_path, monkeypatch):
        for level in ("readonly", "supervised", "full"):
            monkeypatch.setenv("OLLAMACODE_SANDBOX_LEVEL", level)
            with pytest.raises(PermissionError, match="null byte"):
                check_fs_path("foo\x00bar", tmp_path)

    def test_path_escape_blocked_in_supervised(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OLLAMACODE_SANDBOX_LEVEL", "supervised")
        with pytest.raises(PermissionError, match="escapes workspace"):
            check_fs_path("../../etc/passwd", tmp_path)

    def test_path_escape_blocked_in_readonly(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OLLAMACODE_SANDBOX_LEVEL", "readonly")
        with pytest.raises(PermissionError, match="escapes workspace"):
            check_fs_path("../../etc/passwd", tmp_path)

    def test_path_escape_allowed_in_full(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OLLAMACODE_SANDBOX_LEVEL", "full")
        # Should not raise even for paths outside workspace
        check_fs_path("../../etc/passwd", tmp_path)

    def test_sensitive_dotdir_blocked_in_supervised(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OLLAMACODE_SANDBOX_LEVEL", "supervised")
        # Create the .ssh dir inside workspace so path resolves within workspace
        ssh_dir = tmp_path / ".ssh"
        ssh_dir.mkdir()
        with pytest.raises(PermissionError, match=".ssh"):
            check_fs_path(".ssh/id_rsa", tmp_path)

    def test_sensitive_dotdir_allowed_in_full(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OLLAMACODE_SANDBOX_LEVEL", "full")
        # Should not raise
        check_fs_path(".ssh/id_rsa", tmp_path)

    def test_write_in_readonly_blocked(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OLLAMACODE_SANDBOX_LEVEL", "readonly")
        with pytest.raises(PermissionError, match="readonly"):
            check_fs_path("test.txt", tmp_path, allow_write=True)

    def test_write_in_supervised_allowed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OLLAMACODE_SANDBOX_LEVEL", "supervised")
        # Should not raise for write within workspace
        check_fs_path("test.txt", tmp_path, allow_write=True)

    def test_read_in_readonly_within_workspace_allowed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OLLAMACODE_SANDBOX_LEVEL", "readonly")
        # Read-only access within workspace should be fine
        check_fs_path("test.txt", tmp_path, allow_write=False)


# ---------------------------------------------------------------------------
# check_terminal_command
# ---------------------------------------------------------------------------


class TestCheckTerminalCommand:
    def test_null_byte_rejected_at_all_levels(self, monkeypatch):
        for level in ("readonly", "supervised", "full"):
            monkeypatch.setenv("OLLAMACODE_SANDBOX_LEVEL", level)
            with pytest.raises(PermissionError, match="null byte"):
                check_terminal_command("ls\x00-la")

    def test_readonly_blocks_all_commands(self, monkeypatch):
        monkeypatch.setenv("OLLAMACODE_SANDBOX_LEVEL", "readonly")
        with pytest.raises(PermissionError, match="readonly"):
            check_terminal_command("ls -la")

    def test_supervised_allows_commands(self, monkeypatch):
        monkeypatch.setenv("OLLAMACODE_SANDBOX_LEVEL", "supervised")
        # Should not raise
        check_terminal_command("ls -la")

    def test_full_allows_commands(self, monkeypatch):
        monkeypatch.setenv("OLLAMACODE_SANDBOX_LEVEL", "full")
        # Should not raise
        check_terminal_command("rm -rf /tmp/test")
