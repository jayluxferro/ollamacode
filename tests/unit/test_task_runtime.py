from __future__ import annotations

import pytest

from ollamacode.task_runtime import find_subagent, run_task_delegation


def test_find_subagent_matches_by_name():
    subagent = find_subagent(
        [{"name": "reviewer", "tools": ["read_file"]}],
        "reviewer",
    )
    assert subagent is not None
    assert subagent["name"] == "reviewer"


@pytest.mark.asyncio
async def test_run_task_delegation_uses_restricted_tools(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    async def fake_run_agent_loop(session, model, prompt, **kwargs):
        captured["model"] = model
        captured["prompt"] = prompt
        captured["allowed_tools"] = kwargs.get("allowed_tools")
        captured["blocked_tools"] = kwargs.get("blocked_tools")
        return "subagent result"

    monkeypatch.setattr("ollamacode.task_runtime.run_agent_loop", fake_run_agent_loop)
    monkeypatch.setattr("ollamacode.sessions._DB_PATH", tmp_path / "sessions.db")

    result = await run_task_delegation(
        session=object(),
        session_id="parent-session",
        workspace_root=str(tmp_path),
        subagents=[{"name": "reviewer", "tools": ["read_file"], "model": "mini"}],
        arguments={
            "description": "Review changed files",
            "prompt": "Inspect the latest edits for bugs.",
            "subagent_type": "reviewer",
        },
        default_model="default-model",
        system_prompt="system",
        max_messages=0,
        max_tool_rounds=5,
        max_tool_result_chars=0,
    )

    assert captured["model"] == "mini"
    assert captured["allowed_tools"] == ["read_file"]
    assert "task" in captured["blocked_tools"]
    assert "<task_result>" in result
    assert "subagent result" in result
