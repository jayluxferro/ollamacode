"""Unit tests for the filesystem MCP server (fs_mcp.py).

Covers: read_file, write_file, edit_file, multi_edit, list_dir,
path traversal prevention, symlinks, size limits, atomic writes.
"""

from pathlib import Path

import pytest

from ollamacode.servers import fs_mcp


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_read_existing_file(self, mock_fs: Path):
        """read_file returns file contents for a normal file inside workspace."""
        (mock_fs / "hello.txt").write_text("Hello, world!")
        assert fs_mcp.read_file("hello.txt") == "Hello, world!"

    def test_read_nested_file(self, mock_fs: Path):
        """read_file works for files in subdirectories."""
        sub = mock_fs / "sub" / "dir"
        sub.mkdir(parents=True)
        (sub / "nested.py").write_text("x = 1")
        assert fs_mcp.read_file("sub/dir/nested.py") == "x = 1"

    def test_read_nonexistent_file_raises(self, mock_fs: Path):
        """read_file raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            fs_mcp.read_file("does_not_exist.txt")

    def test_read_directory_raises(self, mock_fs: Path):
        """read_file raises FileNotFoundError when path is a directory."""
        (mock_fs / "adir").mkdir()
        with pytest.raises(FileNotFoundError):
            fs_mcp.read_file("adir")

    def test_read_oversized_file_rejected(self, mock_fs: Path):
        """read_file raises ValueError for files exceeding 10 MB."""
        big = mock_fs / "big.txt"
        big.write_text("x")
        real_stat = big.stat()

        class FakeStat:
            st_size = 11 * 1024 * 1024
            st_nlink = 1
            st_mode = real_stat.st_mode
            st_ino = real_stat.st_ino
            st_dev = real_stat.st_dev
            st_uid = real_stat.st_uid
            st_gid = real_stat.st_gid
            st_atime = real_stat.st_atime
            st_mtime = real_stat.st_mtime
            st_ctime = real_stat.st_ctime

        original_stat = Path.stat

        def patched_stat(self_path, *a, **kw):
            if self_path.name == "big.txt":
                return FakeStat()
            return original_stat(self_path, *a, **kw)

        from unittest.mock import patch

        with patch.object(Path, "stat", patched_stat):
            with pytest.raises(ValueError, match="too large"):
                fs_mcp.read_file("big.txt")


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


class TestWriteFile:
    def test_write_new_file(self, mock_fs: Path):
        """write_file creates a new file and returns a success message."""
        result = fs_mcp.write_file("new.txt", "content here")
        assert "Wrote" in result
        assert (mock_fs / "new.txt").read_text() == "content here"

    def test_write_creates_parent_dirs(self, mock_fs: Path):
        """write_file creates intermediate directories automatically."""
        result = fs_mcp.write_file("a/b/c.txt", "deep")
        assert "Wrote" in result
        assert (mock_fs / "a" / "b" / "c.txt").read_text() == "deep"

    def test_write_oversized_content_rejected(self, mock_fs: Path):
        """write_file raises ValueError when content exceeds 10 MB."""
        huge = "x" * (11 * 1024 * 1024)
        with pytest.raises(ValueError, match="too large"):
            fs_mcp.write_file("huge.txt", huge)

    def test_write_atomic_via_rename(self, mock_fs: Path):
        """write_file uses atomic write (temp + rename); the final file exists with correct content."""
        fs_mcp.write_file("atomic.txt", "safe content")
        assert (mock_fs / "atomic.txt").read_text() == "safe content"
        # No leftover .tmp files
        tmps = list(mock_fs.glob("*.tmp"))
        assert len(tmps) == 0

    def test_write_dry_run_does_not_write(self, mock_fs: Path, monkeypatch):
        """write_file in dry-run mode returns a diff but does not modify the file."""
        (mock_fs / "existing.txt").write_text("old")
        monkeypatch.setenv("OLLAMACODE_DRY_RUN_DIFF", "1")
        result = fs_mcp.write_file("existing.txt", "new")
        assert "Dry run" in result
        assert (mock_fs / "existing.txt").read_text() == "old"


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------


class TestEditFile:
    def test_edit_replaces_first_occurrence(self, mock_fs: Path):
        """edit_file replaces first match only by default."""
        (mock_fs / "f.txt").write_text("aaa bbb aaa")
        result = fs_mcp.edit_file("f.txt", "aaa", "ccc")
        assert "Edited" in result
        assert (mock_fs / "f.txt").read_text() == "ccc bbb aaa"

    def test_edit_replace_all(self, mock_fs: Path):
        """edit_file with replace_all=True replaces every occurrence."""
        (mock_fs / "f.txt").write_text("aaa bbb aaa")
        result = fs_mcp.edit_file("f.txt", "aaa", "ccc", replace_all=True)
        assert "all replacements" in result
        assert (mock_fs / "f.txt").read_text() == "ccc bbb ccc"

    def test_edit_no_match_returns_message(self, mock_fs: Path):
        """edit_file returns a message when old_string is not found."""
        (mock_fs / "f.txt").write_text("hello")
        result = fs_mcp.edit_file("f.txt", "not_here", "xxx")
        assert "No occurrence" in result

    def test_edit_nonexistent_file_raises(self, mock_fs: Path):
        """edit_file raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            fs_mcp.edit_file("nope.txt", "a", "b")


# ---------------------------------------------------------------------------
# multi_edit
# ---------------------------------------------------------------------------


class TestMultiEdit:
    def test_multi_edit_basic(self, mock_fs: Path):
        """multi_edit applies multiple edits across files."""
        (mock_fs / "a.txt").write_text("foo bar")
        (mock_fs / "b.txt").write_text("baz qux")
        result = fs_mcp.multi_edit(
            [
                {"path": "a.txt", "old_string": "foo", "new_string": "FOO"},
                {"path": "b.txt", "old_string": "baz", "new_string": "BAZ"},
            ]
        )
        assert "replaced" in result
        assert (mock_fs / "a.txt").read_text() == "FOO bar"
        assert (mock_fs / "b.txt").read_text() == "BAZ qux"

    def test_multi_edit_rollback_on_failure(self, mock_fs: Path):
        """multi_edit rolls back all changes if any edit fails."""
        (mock_fs / "a.txt").write_text("original_a")
        (mock_fs / "b.txt").write_text("original_b")
        result = fs_mcp.multi_edit(
            [
                {
                    "path": "a.txt",
                    "old_string": "original_a",
                    "new_string": "changed_a",
                },
                {"path": "b.txt", "old_string": "NOT_PRESENT", "new_string": "xxx"},
            ]
        )
        assert "rollback" in result
        # a.txt should be reverted
        assert (mock_fs / "a.txt").read_text() == "original_a"

    def test_multi_edit_overwrite_mode(self, mock_fs: Path):
        """multi_edit overwrites entire file when old_string is omitted."""
        (mock_fs / "c.txt").write_text("old content")
        result = fs_mcp.multi_edit(
            [
                {"path": "c.txt", "new_string": "entirely new"},
            ]
        )
        assert "overwrote" in result
        assert (mock_fs / "c.txt").read_text() == "entirely new"

    def test_multi_edit_skips_non_dict(self, mock_fs: Path):
        """multi_edit skips entries that are not dicts."""
        result = fs_mcp.multi_edit(["not a dict"])
        assert "skip" in result


# ---------------------------------------------------------------------------
# Path traversal / symlinks
# ---------------------------------------------------------------------------


class TestPathTraversal:
    def test_dotdot_traversal_rejected(self, mock_fs: Path):
        """Paths with '..' that escape workspace are rejected."""
        with pytest.raises((ValueError, PermissionError)):
            fs_mcp._resolve("../../etc/passwd")

    def test_symlink_outside_workspace_rejected(self, mock_fs: Path, monkeypatch):
        """Symlinks pointing outside the workspace are rejected."""
        secret = mock_fs.parent / "secret.txt"
        secret.write_text("secret data")
        link = mock_fs / "escape"
        link.symlink_to(secret)

        with pytest.raises(ValueError, match="outside workspace"):
            fs_mcp._resolve("escape")

    def test_symlink_inside_workspace_allowed(self, mock_fs: Path):
        """Symlinks pointing within the workspace are allowed."""
        target = mock_fs / "real.txt"
        target.write_text("ok")
        link = mock_fs / "link.txt"
        link.symlink_to(target)
        result = fs_mcp._resolve("link.txt")
        assert result == target.resolve()

    def test_long_path_rejected(self, mock_fs: Path):
        """Paths exceeding _MAX_PATH_LENGTH are rejected."""
        long_path = "a" * 5000
        with pytest.raises(ValueError, match="too long"):
            fs_mcp._resolve(long_path)


# ---------------------------------------------------------------------------
# list_dir
# ---------------------------------------------------------------------------


class TestListDir:
    def test_list_dir_basic(self, mock_fs: Path):
        """list_dir returns sorted directory entries."""
        (mock_fs / "b.txt").write_text("b")
        (mock_fs / "a.txt").write_text("a")
        (mock_fs / "subdir").mkdir()
        entries = fs_mcp.list_dir(".")
        assert entries == ["a.txt", "b.txt", "subdir"]

    def test_list_dir_not_a_directory(self, mock_fs: Path):
        """list_dir raises NotADirectoryError for a file."""
        (mock_fs / "file.txt").write_text("x")
        with pytest.raises(NotADirectoryError):
            fs_mcp.list_dir("file.txt")
