"""Unit tests for @-style context expansion."""

from ollamacode.context import expand_at_refs, get_branch_context, prepend_file_context


def test_expand_at_refs_no_refs():
    """Message without @ refs is unchanged."""
    assert expand_at_refs("hello world", "/tmp") == "hello world"
    assert expand_at_refs("refactor this", "/tmp") == "refactor this"


def test_expand_at_refs_file(tmp_path):
    """@path to a file injects file contents."""
    f = tmp_path / "main.py"
    f.write_text("print(1)\n")
    # Use str(workspace_root) so path resolution matches CLI (cwd string)
    out = expand_at_refs("refactor @main.py", str(tmp_path))
    assert "Contents of main.py:" in out
    assert "print(1)" in out
    assert "refactor" in out
    assert "@main.py" not in out


def test_expand_at_refs_folder(tmp_path):
    """@path/ to a folder injects file list."""
    (tmp_path / "a").write_text("1")
    (tmp_path / "b").write_text("2")
    out = expand_at_refs("list @.", str(tmp_path))
    assert "Files in ." in out
    assert "a" in out
    assert "b" in out


def test_expand_at_refs_outside_workspace(tmp_path):
    """@path outside workspace is skipped."""
    out = expand_at_refs("see @/etc/passwd", str(tmp_path))
    assert out == "see @/etc/passwd"
    assert "Contents" not in out


def test_prepend_file_context_full_file(tmp_path):
    """--file without --lines prepends full file."""
    (tmp_path / "main.py").write_text("a = 1\nb = 2\n")
    out = prepend_file_context("explain this", "main.py", str(tmp_path), None)
    assert "Contents of main.py:" in out
    assert "a = 1" in out
    assert "b = 2" in out
    assert "explain this" in out


def test_prepend_file_context_lines(tmp_path):
    """--file with --lines prepends only that range."""
    (tmp_path / "main.py").write_text("line1\nline2\nline3\nline4\n")
    out = prepend_file_context("refactor", "main.py", str(tmp_path), "2-3")
    assert "lines 2-3" in out
    assert "line2" in out
    assert "line3" in out
    assert "line1" not in out
    assert "line4" not in out


def test_get_branch_context_no_git(tmp_path):
    """get_branch_context returns '' when not a git repo."""
    out = get_branch_context(tmp_path, "main")
    assert out == ""
