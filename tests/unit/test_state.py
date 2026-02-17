"""Unit tests for persistent state and format_recent_context (incl. Phase 5 helpers)."""

from unittest.mock import patch

from ollamacode.state import (
    add_knowledge_node,
    append_feedback,
    append_past_error,
    format_feedback_context,
    format_knowledge_graph_context,
    format_knowledge_context,
    format_past_errors_context,
    format_plan_context,
    format_preferences,
    format_recent_context,
    get_state,
    query_knowledge_graph,
    record_reasoning,
)


def test_format_recent_context_empty():
    """format_recent_context returns '' for empty or no recent_files."""
    assert format_recent_context({}) == ""
    assert format_recent_context({"recent_files": []}) == ""
    assert format_recent_context({"recent_files": []}, max_files=0) == ""


def test_format_recent_context_respects_max_files():
    """format_recent_context returns at most max_files paths (last N)."""
    state = {"recent_files": ["a", "b", "c", "d", "e"]}
    assert format_recent_context(state, max_files=2) == "Recent files: d, e"
    assert format_recent_context(state, max_files=10) == "Recent files: a, b, c, d, e"


def test_format_recent_context_single():
    """format_recent_context with one file."""
    assert (
        format_recent_context({"recent_files": ["foo/bar.py"]})
        == "Recent files: foo/bar.py"
    )


def test_format_preferences_empty():
    """format_preferences returns '' for no or empty preferences."""
    assert format_preferences({}) == ""
    assert format_preferences({"preferences": {}}) == ""
    assert format_preferences({"preferences": None}) == ""


def test_format_preferences_formats():
    """format_preferences returns 'User preferences: k: v; ...' for non-empty prefs."""
    assert (
        format_preferences({"preferences": {"coding_style": "type hints"}})
        == "User preferences: coding_style: type hints"
    )
    state = {"preferences": {"a": "1", "b": "2"}}
    out = format_preferences(state)
    assert out.startswith("User preferences: ")
    assert "a: 1" in out and "b: 2" in out


def test_get_state_empty(tmp_path):
    """get_state returns {} when state file does not exist or is empty."""
    with patch("ollamacode.state._STATE_PATH", tmp_path / "state.json"):
        assert get_state() == {}
    (tmp_path / "state.json").write_text("{}")
    with patch("ollamacode.state._STATE_PATH", tmp_path / "state.json"):
        assert get_state() == {}


# --- Phase 5: plan, feedback, knowledge ---


def test_format_plan_context_empty():
    """format_plan_context returns '' when no plan or empty plan."""
    assert format_plan_context({}) == ""
    assert format_plan_context({"current_plan": ""}) == ""
    assert format_plan_context({"current_plan": "   \n  "}) == ""


def test_format_plan_context_plan_only():
    """format_plan_context returns 'Current plan: ...' when no steps."""
    out = format_plan_context({"current_plan": "Add tests for API"})
    assert "Current plan: Add tests for API" in out
    assert "Completed steps" not in out


def test_format_plan_context_with_steps():
    """format_plan_context includes completed steps list."""
    state = {
        "current_plan": "Refactor auth",
        "completed_steps": ["Step one", "Step two"],
    }
    out = format_plan_context(state)
    assert "Current plan: Refactor auth" in out
    assert "Completed steps so far:" in out
    assert "  - Step one" in out
    assert "  - Step two" in out


def test_format_feedback_context_empty():
    """format_feedback_context returns '' for no or empty feedback."""
    assert format_feedback_context({}) == ""
    assert format_feedback_context({"feedback": []}) == ""
    assert format_feedback_context({"feedback": None}) == ""


def test_format_feedback_context_ratings():
    """format_feedback_context formats rating and edit_accepted."""
    state = {
        "feedback": [
            {"type": "rating", "value": 1, "context": "positive"},
            {"type": "rating", "value": -1, "context": "negative"},
            {"type": "edit_accepted", "value": True, "context": "user applied edits"},
        ]
    }
    out = format_feedback_context(state, max_entries=5)
    assert "Recent feedback:" in out
    assert "User rated a reply positively" in out
    assert "User rated a reply negatively" in out
    assert "User applied suggested edits" in out


def test_format_knowledge_context_empty():
    """format_knowledge_context returns '' for no or empty knowledge_index."""
    assert format_knowledge_context({}) == ""
    assert format_knowledge_context({"knowledge_index": []}) == ""


def test_format_knowledge_context_dict_entries():
    """format_knowledge_context formats topic/summary dicts."""
    state = {
        "knowledge_index": [
            {"topic": "Auth", "summary": "Uses JWT in header"},
            {"topic": "API", "summary": "REST under /v1"},
        ]
    }
    out = format_knowledge_context(state, max_entries=15)
    assert "Knowledge index" in out
    assert "Auth" in out and "JWT" in out
    assert "API" in out and "REST" in out


def test_format_knowledge_context_string_entries():
    """format_knowledge_context formats string entries."""
    state = {"knowledge_index": ["Doc string one", "Doc string two"]}
    out = format_knowledge_context(state, max_entries=15)
    assert "Knowledge index" in out
    assert "Doc string one" in out
    assert "Doc string two" in out


def test_append_feedback_persists(tmp_path):
    """append_feedback appends to state feedback and saves."""
    with patch("ollamacode.state._STATE_PATH", tmp_path / "state.json"):
        (tmp_path / "state.json").write_text("{}")
        append_feedback("rating", 1, "user liked the reply")
        state = get_state()
        assert "feedback" in state
        assert len(state["feedback"]) == 1
        assert state["feedback"][0]["type"] == "rating"
        assert state["feedback"][0]["value"] == 1
        assert "user liked" in state["feedback"][0]["context"]


def test_format_past_errors_context_empty():
    """format_past_errors_context returns '' for no or empty past_errors."""
    assert format_past_errors_context({}) == ""
    assert format_past_errors_context({"past_errors": []}) == ""


def test_format_past_errors_context_formats():
    """format_past_errors_context returns block with tool, error, hint."""
    state = {
        "past_errors": [
            {
                "tool": "read_file",
                "error_summary": "FileNotFoundError",
                "hint": "check path",
            },
        ],
    }
    out = format_past_errors_context(state, max_entries=5)
    assert "Similar past errors" in out
    assert "read_file" in out
    assert "FileNotFoundError" in out
    assert "check path" in out


def test_append_past_error_persists(tmp_path):
    """append_past_error appends to state past_errors and saves."""
    with patch("ollamacode.state._STATE_PATH", tmp_path / "state.json"):
        (tmp_path / "state.json").write_text("{}")
        append_past_error("run_command", "exit code 1", "check command")
        state = get_state()
        assert "past_errors" in state
        assert len(state["past_errors"]) == 1
        assert state["past_errors"][0]["tool"] == "run_command"
        assert "exit code" in state["past_errors"][0]["error_summary"]
        assert "check command" in state["past_errors"][0]["hint"]


def test_record_reasoning_persists(tmp_path):
    """record_reasoning stores steps and conclusion in state."""
    with patch("ollamacode.state._STATE_PATH", tmp_path / "state.json"):
        (tmp_path / "state.json").write_text("{}")
        record_reasoning(["step1", "step2"], "conclusion text")
        state = get_state()
        assert "last_reasoning" in state
        assert state["last_reasoning"]["steps"] == ["step1", "step2"]
        assert state["last_reasoning"]["conclusion"] == "conclusion text"


def test_add_and_query_knowledge_graph(tmp_path):
    """add_knowledge_node upserts nodes and query_knowledge_graph returns matches."""
    with patch("ollamacode.state._STATE_PATH", tmp_path / "state.json"):
        (tmp_path / "state.json").write_text("{}")
        add_knowledge_node("Auth", "JWT tokens", ["security"], "docs")
        add_knowledge_node("API", "REST endpoints", ["http"], "readme")
        rows = query_knowledge_graph("jwt", max_results=5)
        assert rows
        assert rows[0]["topic"] == "Auth"
        # upsert same topic
        add_knowledge_node("auth", "JWT + refresh", ["oauth"], "guide")
        rows2 = query_knowledge_graph("refresh", max_results=5)
        assert rows2
        assert rows2[0]["topic"].lower() == "auth"


def test_format_knowledge_graph_context():
    """format_knowledge_graph_context renders graph nodes with related topics."""
    state = {
        "knowledge_graph": {
            "nodes": [
                {"topic": "Auth", "summary": "JWT", "related": ["security", "tokens"]},
                {"topic": "API", "summary": "REST"},
            ]
        }
    }
    out = format_knowledge_graph_context(state, max_nodes=8)
    assert "Knowledge graph" in out
    assert "Auth" in out
    assert "related:" in out
