"""Unit tests for the skills MCP server (skills_mcp.py).

Covers: list_skills, read_skill, write_skill, save_memory,
file operations, invalid names.
"""

from pathlib import Path

import pytest

from ollamacode.servers import skills_mcp


# ---------------------------------------------------------------------------
# Helpers: redirect workspace root to temp dirs
# ---------------------------------------------------------------------------


@pytest.fixture()
def skills_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set up a temp workspace for skills operations."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("OLLAMACODE_FS_ROOT", str(workspace))
    return workspace


# ---------------------------------------------------------------------------
# list_skills
# ---------------------------------------------------------------------------


class TestListSkills:
    def test_empty_workspace(self, skills_workspace: Path):
        """list_skills returns empty list when no skills exist."""
        result = skills_mcp.list_skills()
        # May include global skills from ~/.ollamacode/skills, but workspace is empty
        assert isinstance(result, list)

    def test_lists_created_skills(self, skills_workspace: Path):
        """list_skills includes skills created via write_skill."""
        skills_mcp.write_skill("my_skill", "content here")
        result = skills_mcp.list_skills()
        assert "my_skill" in result

    def test_lists_multiple_skills(self, skills_workspace: Path):
        """list_skills returns all workspace skills."""
        skills_mcp.write_skill("alpha", "A")
        skills_mcp.write_skill("beta", "B")
        result = skills_mcp.list_skills()
        assert "alpha" in result
        assert "beta" in result


# ---------------------------------------------------------------------------
# read_skill
# ---------------------------------------------------------------------------


class TestReadSkill:
    def test_read_existing_skill(self, skills_workspace: Path):
        """read_skill returns the content of a skill."""
        skills_mcp.write_skill("greet", "Hello from skill.")
        result = skills_mcp.read_skill("greet")
        assert "Hello from skill." in result

    def test_read_nonexistent_skill(self, skills_workspace: Path):
        """read_skill returns 'not found' for missing skills."""
        result = skills_mcp.read_skill("no_such_skill")
        assert "not found" in result.lower()

    def test_read_invalid_name(self, skills_workspace: Path):
        """read_skill returns error for path-traversal names."""
        result = skills_mcp.read_skill("../../etc/passwd")
        assert "not found" in result.lower() or "invalid" in result.lower()


# ---------------------------------------------------------------------------
# write_skill
# ---------------------------------------------------------------------------


class TestWriteSkill:
    def test_write_creates_file(self, skills_workspace: Path):
        """write_skill creates a .md file in the skills directory."""
        result = skills_mcp.write_skill("new_skill", "# New Skill\n\nContent.")
        assert "Wrote" in result or "wrote" in result.lower()
        skills_dir = skills_workspace / ".ollamacode" / "skills"
        assert (skills_dir / "new_skill.md").exists()

    def test_write_overwrites_existing(self, skills_workspace: Path):
        """write_skill overwrites an existing skill."""
        skills_mcp.write_skill("overme", "version 1")
        skills_mcp.write_skill("overme", "version 2")
        content = skills_mcp.read_skill("overme")
        assert "version 2" in content
        assert "version 1" not in content

    def test_write_invalid_name_rejected(self, skills_workspace: Path):
        """write_skill returns error for invalid names."""
        result = skills_mcp.write_skill("bad name with spaces", "content")
        assert "Error" in result or "invalid" in result.lower()

    def test_write_empty_name_rejected(self, skills_workspace: Path):
        """write_skill returns error for empty name."""
        result = skills_mcp.write_skill("", "content")
        assert "Error" in result or "invalid" in result.lower()


# ---------------------------------------------------------------------------
# save_memory
# ---------------------------------------------------------------------------


class TestSaveMemory:
    def test_save_creates_memory_skill(self, skills_workspace: Path):
        """save_memory creates the memory skill if it doesn't exist."""
        result = skills_mcp.save_memory("key1", "value1")
        assert "Wrote" in result or "wrote" in result.lower()
        content = skills_mcp.read_skill("memory")
        assert "key1" in content
        assert "value1" in content

    def test_save_appends_entries(self, skills_workspace: Path):
        """save_memory appends multiple entries to the memory skill."""
        skills_mcp.save_memory("first", "A")
        skills_mcp.save_memory("second", "B")
        content = skills_mcp.read_skill("memory")
        assert "first" in content and "A" in content
        assert "second" in content and "B" in content
