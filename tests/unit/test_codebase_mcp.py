"""Unit tests for the codebase search MCP server (codebase_mcp.py).

Covers: search_codebase, grep, glob, get_relevant_files,
max_results clamping, regex DoS protection, invalid regex handling.
"""

from pathlib import Path


from ollamacode.servers import codebase_mcp


# ---------------------------------------------------------------------------
# search_codebase
# ---------------------------------------------------------------------------


class TestSearchCodebase:
    def test_basic_search(self, mock_fs: Path):
        """search_codebase finds matching lines."""
        (mock_fs / "hello.py").write_text("print('hello world')\nprint('goodbye')\n")
        result = codebase_mcp.search_codebase("hello")
        assert "hello.py" in result
        assert "hello world" in result

    def test_case_insensitive(self, mock_fs: Path):
        """search_codebase performs case-insensitive matching."""
        (mock_fs / "test.txt").write_text("Hello WORLD\n")
        result = codebase_mcp.search_codebase("hello world")
        assert "test.txt" in result

    def test_no_matches(self, mock_fs: Path):
        """search_codebase returns 'No matches' when nothing found."""
        (mock_fs / "test.txt").write_text("nothing relevant\n")
        result = codebase_mcp.search_codebase("xyz_not_found_42")
        assert "No matches" in result

    def test_max_results_clamped_negative(self, mock_fs: Path):
        """Negative max_results is clamped to 1."""
        (mock_fs / "test.txt").write_text("hello\n" * 100)
        result = codebase_mcp.search_codebase("hello", max_results=-5)
        assert isinstance(result, str)
        # Should still return at least one result
        lines = [line for line in result.strip().split("\n") if line.strip()]
        assert len(lines) >= 1

    def test_max_results_clamped_huge(self, mock_fs: Path):
        """Huge max_results is clamped to MAX_RESULTS."""
        (mock_fs / "test.txt").write_text("hello\n" * 200)
        result = codebase_mcp.search_codebase("hello", max_results=999999)
        lines = [line for line in result.strip().split("\n") if line.strip()]
        assert len(lines) <= codebase_mcp.MAX_RESULTS

    def test_file_pattern_filter(self, mock_fs: Path):
        """search_codebase respects file_pattern glob."""
        (mock_fs / "code.py").write_text("target line\n")
        (mock_fs / "readme.md").write_text("target line\n")
        result = codebase_mcp.search_codebase("target", file_pattern="*.py")
        assert "code.py" in result
        assert "readme.md" not in result

    def test_skips_git_dir(self, mock_fs: Path):
        """search_codebase skips .git and other special directories."""
        git_dir = mock_fs / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("target\n")
        (mock_fs / "real.txt").write_text("no match\n")
        result = codebase_mcp.search_codebase("target")
        assert ".git" not in result or "No matches" in result


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------


class TestGrep:
    def test_basic_regex(self, mock_fs: Path):
        """grep finds regex matches."""
        (mock_fs / "code.py").write_text(
            "def foo():\n    pass\ndef bar():\n    return 1\n"
        )
        result = codebase_mcp.grep(pattern=r"def \w+")
        assert "code.py" in result
        assert "def foo" in result
        assert "def bar" in result

    def test_regex_too_long(self, mock_fs: Path):
        """grep rejects patterns exceeding 500 chars."""
        long_pattern = "a" * 600
        result = codebase_mcp.grep(pattern=long_pattern)
        assert "too long" in result.lower()

    def test_invalid_regex(self, mock_fs: Path):
        """grep returns error for invalid regex."""
        result = codebase_mcp.grep(pattern="[invalid")
        assert "invalid regex" in result.lower()

    def test_context_lines(self, mock_fs: Path):
        """grep includes context lines when requested."""
        (mock_fs / "data.txt").write_text("line1\nline2\nTARGET\nline4\nline5\n")
        result = codebase_mcp.grep(pattern="TARGET", context_lines=1)
        assert "line2" in result
        assert "TARGET" in result
        assert "line4" in result

    def test_no_matches(self, mock_fs: Path):
        """grep returns 'No matches' when pattern is not found."""
        (mock_fs / "empty.txt").write_text("nothing here\n")
        result = codebase_mcp.grep(pattern="xyzzy_42")
        assert "No matches" in result

    def test_grep_single_file(self, mock_fs: Path):
        """grep can target a single file."""
        (mock_fs / "a.txt").write_text("match here\n")
        (mock_fs / "b.txt").write_text("match here too\n")
        result = codebase_mcp.grep(pattern="match", path="a.txt")
        assert "a.txt" in result
        assert "b.txt" not in result


# ---------------------------------------------------------------------------
# glob
# ---------------------------------------------------------------------------


class TestGlob:
    def test_glob_basic(self, mock_fs: Path):
        """glob lists files matching a pattern."""
        (mock_fs / "a.py").write_text("")
        (mock_fs / "b.py").write_text("")
        (mock_fs / "c.txt").write_text("")
        result = codebase_mcp.glob("*.py")
        assert "a.py" in result
        assert "b.py" in result
        assert "c.txt" not in result

    def test_glob_no_matches(self, mock_fs: Path):
        """glob returns 'No files matching' when nothing matches."""
        result = codebase_mcp.glob("*.rs")
        assert "No files matching" in result

    def test_glob_empty_pattern_rejected(self, mock_fs: Path):
        """glob rejects empty patterns."""
        result = codebase_mcp.glob("")
        assert "Provide a glob pattern" in result


# ---------------------------------------------------------------------------
# get_relevant_files
# ---------------------------------------------------------------------------


class TestGetRelevantFiles:
    def test_basic_relevance(self, mock_fs: Path):
        """get_relevant_files matches file paths against description words."""
        sub = mock_fs / "auth"
        sub.mkdir()
        (sub / "login.py").write_text("")
        (mock_fs / "readme.md").write_text("")
        result = codebase_mcp.get_relevant_files("auth login")
        assert "auth" in result and "login" in result

    def test_empty_description(self, mock_fs: Path):
        """get_relevant_files returns prompt when given empty/short description."""
        result = codebase_mcp.get_relevant_files("")
        assert "Provide" in result
