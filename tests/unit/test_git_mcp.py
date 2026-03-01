"""Unit tests for the git MCP server (git_mcp.py).

Covers: _validate_ref, git_status, git_diff, git_log, git_show,
write operations (git_add, git_commit, git_checkout), max_count clamping,
structured return values, error cases.
"""

import subprocess
from pathlib import Path

import pytest

from ollamacode.servers import git_mcp


# ---------------------------------------------------------------------------
# _validate_ref
# ---------------------------------------------------------------------------


class TestValidateRef:
    def test_normal_refs_pass(self):
        """Standard branch names and rev specs are accepted."""
        assert git_mcp._validate_ref("main", "branch") == "main"
        assert git_mcp._validate_ref("HEAD", "ref") == "HEAD"
        assert (
            git_mcp._validate_ref("feature/my-branch", "branch") == "feature/my-branch"
        )
        assert git_mcp._validate_ref("HEAD~3", "ref") == "HEAD~3"
        assert git_mcp._validate_ref("v1.0.0", "tag") == "v1.0.0"
        assert git_mcp._validate_ref("HEAD^2", "ref") == "HEAD^2"

    def test_empty_ref_rejected(self):
        """Empty or whitespace-only refs are rejected."""
        with pytest.raises(ValueError, match="required"):
            git_mcp._validate_ref("", "branch")
        with pytest.raises(ValueError, match="required"):
            git_mcp._validate_ref("   ", "branch")

    def test_null_byte_rejected(self):
        """Refs containing null bytes are rejected."""
        with pytest.raises(ValueError, match="null"):
            git_mcp._validate_ref("main\0evil", "branch")

    def test_shell_metacharacters_rejected(self):
        """Refs with shell metacharacters are rejected."""
        for bad in [
            "; rm -rf /",
            "HEAD | cat",
            "`id`",
            "$(evil)",
            "a && b",
            "a{b}",
            "a!b",
        ]:
            with pytest.raises(ValueError, match="disallowed"):
                git_mcp._validate_ref(bad, "ref")

    def test_dotdot_range_allowed(self):
        """Legitimate range specs like main..develop are allowed."""
        assert git_mcp._validate_ref("main..develop", "ref") == "main..develop"

    def test_dotdot_with_metachar_rejected(self):
        """Dot-dot ranges with invalid characters are rejected."""
        with pytest.raises(ValueError, match="disallowed|invalid"):
            git_mcp._validate_ref("main..$(id)", "ref")


# ---------------------------------------------------------------------------
# git_status
# ---------------------------------------------------------------------------


class TestGitStatus:
    def test_git_status_on_non_git_dir(self, mock_fs: Path):
        """git_status returns an error message when run outside a git repo."""
        result = git_mcp.git_status()
        # It should return something (error from git or "not a git repo")
        assert isinstance(result, str)

    def test_git_status_on_git_repo(self, mock_fs: Path):
        """git_status returns short-format status on a real git repo."""
        subprocess.run(["git", "init"], cwd=mock_fs, capture_output=True)
        (mock_fs / "file.txt").write_text("hello")
        result = git_mcp.git_status()
        assert isinstance(result, str)
        # Untracked file should show up
        assert "file.txt" in result


# ---------------------------------------------------------------------------
# git_log — max_count clamping
# ---------------------------------------------------------------------------


class TestGitLog:
    def test_max_count_clamped_to_positive(self, mock_fs: Path):
        """git_log with negative max_count is clamped to 1 (no crash)."""
        result = git_mcp.git_log(max_count=-5)
        assert isinstance(result, str)

    def test_max_count_clamped_to_500(self, mock_fs: Path):
        """git_log with huge max_count is clamped to 500 (no crash)."""
        result = git_mcp.git_log(max_count=999999)
        assert isinstance(result, str)

    def test_git_log_on_real_repo(self, mock_fs: Path):
        """git_log returns commit history on a real git repo."""
        subprocess.run(["git", "init"], cwd=mock_fs, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=mock_fs,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"], cwd=mock_fs, capture_output=True
        )
        (mock_fs / "f.txt").write_text("v1")
        subprocess.run(["git", "add", "."], cwd=mock_fs, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"], cwd=mock_fs, capture_output=True
        )
        result = git_mcp.git_log(max_count=5)
        assert "Initial commit" in result


# ---------------------------------------------------------------------------
# git_show — revision validation
# ---------------------------------------------------------------------------


class TestGitShow:
    def test_show_validates_revision(self, mock_fs: Path):
        """git_show rejects dangerous revision strings."""
        result = git_mcp.git_show(revision="; rm -rf /")
        assert "disallowed" in result.lower()

    def test_show_with_valid_ref(self, mock_fs: Path):
        """git_show accepts HEAD on a real repo."""
        subprocess.run(["git", "init"], cwd=mock_fs, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=mock_fs,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"], cwd=mock_fs, capture_output=True
        )
        (mock_fs / "f.txt").write_text("v1")
        subprocess.run(["git", "add", "."], cwd=mock_fs, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "first"], cwd=mock_fs, capture_output=True
        )
        result = git_mcp.git_show(revision="HEAD")
        assert "first" in result


# ---------------------------------------------------------------------------
# Structured write operations
# ---------------------------------------------------------------------------


class TestGitWriteOps:
    def test_git_checkout_validates_branch(self, mock_fs: Path):
        """git_checkout returns structured error for invalid branch names."""
        result = git_mcp.git_checkout(branch="$(evil)")
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "disallowed" in result["error"].lower()

    def test_git_commit_empty_message_rejected(self, mock_fs: Path):
        """git_commit rejects empty commit messages."""
        result = git_mcp.git_commit(message="")
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "required" in result["error"].lower()

    def test_git_push_validates_remote(self, mock_fs: Path):
        """git_push rejects malicious remote names."""
        result = git_mcp.git_push(remote="; rm -rf /")
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "disallowed" in result["error"].lower()

    def test_git_merge_validates_branch(self, mock_fs: Path):
        """git_merge rejects malicious branch names."""
        result = git_mcp.git_merge(branch="`id`")
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "disallowed" in result["error"].lower()

    def test_git_branch_delete_validates(self, mock_fs: Path):
        """git_branch_delete rejects malicious branch names."""
        result = git_mcp.git_branch_delete(branch="$(whoami)")
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "disallowed" in result["error"].lower()

    def test_git_add_structured_return(self, mock_fs: Path):
        """git_add returns a dict with success/output/error keys."""
        result = git_mcp.git_add()
        assert isinstance(result, dict)
        assert "success" in result
        assert "output" in result
        assert "error" in result


# ---------------------------------------------------------------------------
# git_stash
# ---------------------------------------------------------------------------


class TestGitStash:
    def test_stash_unknown_action(self, mock_fs: Path):
        """git_stash with unknown action returns error dict."""
        result = git_mcp.git_stash(action="unknown_action")
        assert isinstance(result, dict)
        assert result["success"] is False
        assert "unknown" in result["error"].lower()

    def test_stash_list_returns_string(self, mock_fs: Path):
        """git_stash list returns a string (not dict)."""
        subprocess.run(["git", "init"], cwd=mock_fs, capture_output=True)
        result = git_mcp.git_stash(action="list")
        assert isinstance(result, str)
