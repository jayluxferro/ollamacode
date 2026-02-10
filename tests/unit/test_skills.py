"""Unit tests for skills and memory."""

import tempfile
from pathlib import Path


from ollamacode.skills import (
    get_skills_dirs,
    list_skills,
    load_skills_text,
    read_skill,
    save_memory,
    write_skill,
)


def test_safe_skill_name_rejected():
    """Invalid names return None from read_skill / empty from write_skill."""
    with tempfile.TemporaryDirectory() as d:
        assert read_skill("../../../etc/passwd", d) is None
        assert "Error" in write_skill("bad name", "x", d)
        assert "Error" in write_skill("", "x", d)


def test_write_and_read_skill_workspace():
    """Write and read a skill in workspace .ollamacode/skills."""
    with tempfile.TemporaryDirectory() as d:
        out = write_skill("my_skill", "Content here.", d)
        assert "Wrote" in out
        skills_dir = Path(d) / ".ollamacode" / "skills"
        assert (skills_dir / "my_skill.md").exists()
        assert read_skill("my_skill", d) == "Content here.\n"


def test_save_memory_appends():
    """save_memory creates or appends to the memory skill."""
    with tempfile.TemporaryDirectory() as d:
        save_memory("key1", "value1", d)
        save_memory("key2", "value2", d)
        content = read_skill("memory", d)
        assert "key1" in content and "value1" in content
        assert "key2" in content and "value2" in content


def test_list_skills_deduplicated():
    """list_skills returns names from both dirs without duplicate."""
    with tempfile.TemporaryDirectory() as d:
        write_skill("a", "x", d)
        write_skill("b", "y", d)
        names = list_skills(d)
        assert "a" in names and "b" in names
        assert len(names) >= 2


def test_load_skills_text_includes_workspace():
    """load_skills_text returns concatenated skill sections."""
    with tempfile.TemporaryDirectory() as d:
        write_skill("s1", "First skill.", d)
        write_skill("s2", "Second.", d)
        text = load_skills_text(d)
        assert "Skill: s1" in text and "First skill." in text
        assert "Skill: s2" in text and "Second." in text


def test_get_skills_dirs_workspace_only_when_exists():
    """get_skills_dirs includes workspace only if .ollamacode/skills exists."""
    with tempfile.TemporaryDirectory() as d:
        dirs_before = get_skills_dirs(d)
        write_skill("x", "y", d)
        dirs_after = get_skills_dirs(d)
        assert len(dirs_after) >= len(dirs_before)
        ws_skills = Path(d).resolve() / ".ollamacode" / "skills"
        assert any(p.resolve() == ws_skills for p in dirs_after)
