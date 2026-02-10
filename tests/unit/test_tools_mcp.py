"""Unit tests for tools MCP (run_code_quality, run_coverage, _parse_coverage_term_missing)."""

from unittest.mock import patch


from ollamacode.servers import tools_mcp


def test_default_code_quality_commands(monkeypatch):
    """_default_code_quality_commands returns ruff, black, isort, mypy when env unset."""
    monkeypatch.delenv("OLLAMACODE_CODE_QUALITY_COMMANDS", raising=False)
    cmds = tools_mcp._default_code_quality_commands()
    assert "ruff check ." in cmds
    assert "black --check ." in cmds
    assert "isort --check-only ." in cmds
    assert "mypy ." in cmds


def test_default_code_quality_commands_from_env(monkeypatch):
    """_default_code_quality_commands uses OLLAMACODE_CODE_QUALITY_COMMANDS when set."""
    monkeypatch.setenv("OLLAMACODE_CODE_QUALITY_COMMANDS", "ruff check .,mypy .")
    cmds = tools_mcp._default_code_quality_commands()
    assert cmds == ["ruff check .", "mypy ."]


def test_run_code_quality_empty_commands_list():
    """run_code_quality with commands=[] returns all_passed True and empty report."""
    with patch.object(tools_mcp, "_run") as m:
        out = tools_mcp.run_code_quality(commands=[])
    assert out["all_passed"] is True
    assert out["report"] == ""
    assert out["results"] == []
    m.assert_not_called()


def test_parse_coverage_term_missing_empty():
    """_parse_coverage_term_missing returns empty lists for empty or header-only input."""
    u, s = tools_mcp._parse_coverage_term_missing("")
    assert u == [] and s == []
    u, s = tools_mcp._parse_coverage_term_missing("Name   Stmts   Miss  Cover\n------")
    assert u == [] and s == []


def test_parse_coverage_term_missing_parses_table():
    """_parse_coverage_term_missing extracts uncovered .py files and suggests tests."""
    out = """
Name                     Stmts   Miss  Cover   Missing
------------------------------------------------------
ollamacode/foo.py           10      2    80%   5, 6
ollamacode/bar.py            5      0   100%
tests/test_foo.py            8      0   100%
TOTAL                      23      2    91%
"""
    uncovered, suggested = tools_mcp._parse_coverage_term_missing(out)
    assert "ollamacode/foo.py" in uncovered
    assert "ollamacode/bar.py" not in uncovered  # 0 miss
    assert "tests/test_foo.py" not in uncovered  # 0 miss
    assert any("foo.py" in x for x in suggested)
    assert any("test_foo" in x or "foo" in x for x in suggested)
