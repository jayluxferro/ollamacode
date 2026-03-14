from __future__ import annotations

from ollamacode._chat_helpers import build_system_prompt


def test_build_system_prompt_includes_todo_and_patch_guidance():
    """The shared system prompt should encourage todo tracking and patch-based edits."""
    prompt = build_system_prompt(system_extra="")
    assert "todowrite" in prompt
    assert "todoread" in prompt
    assert "apply_patch" in prompt
