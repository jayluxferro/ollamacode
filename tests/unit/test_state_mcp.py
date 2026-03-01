"""Unit tests for the state MCP server (state_mcp.py).

Covers: get_state, update_state, clear_state, append_recent_file,
record_reasoning, add_knowledge_node, query_knowledge_graph,
feedback operations.
"""

from pathlib import Path


from ollamacode.servers import state_mcp


# ---------------------------------------------------------------------------
# get_state
# ---------------------------------------------------------------------------


class TestGetState:
    def test_empty_state(self, temp_config: Path):
        """get_state returns {} when no state file exists."""
        result = state_mcp.get_state()
        assert result == {}

    def test_state_after_update(self, temp_config: Path):
        """get_state returns data previously written by update_state."""
        state_mcp.update_state(recent_files=["a.py", "b.py"])
        state = state_mcp.get_state()
        assert state["recent_files"] == ["a.py", "b.py"]


# ---------------------------------------------------------------------------
# update_state
# ---------------------------------------------------------------------------


class TestUpdateState:
    def test_update_recent_files(self, temp_config: Path):
        """update_state replaces recent_files."""
        result = state_mcp.update_state(recent_files=["x.py"])
        assert "Updated" in result
        state = state_mcp.get_state()
        assert state["recent_files"] == ["x.py"]

    def test_update_preferences_merges(self, temp_config: Path):
        """update_state merges preferences rather than replacing."""
        state_mcp.update_state(preferences={"lang": "python"})
        state_mcp.update_state(preferences={"style": "pep8"})
        state = state_mcp.get_state()
        prefs = state.get("preferences", {})
        assert prefs.get("lang") == "python"
        assert prefs.get("style") == "pep8"

    def test_update_current_plan(self, temp_config: Path):
        """update_state stores a plan string."""
        state_mcp.update_state(current_plan="Add tests for MCP servers")
        state = state_mcp.get_state()
        assert state["current_plan"] == "Add tests for MCP servers"

    def test_update_completed_steps(self, temp_config: Path):
        """update_state stores completed steps list."""
        state_mcp.update_state(completed_steps=["Step 1", "Step 2"])
        state = state_mcp.get_state()
        assert state["completed_steps"] == ["Step 1", "Step 2"]

    def test_update_feedback_append(self, temp_config: Path):
        """update_state with feedback_append adds to feedback list."""
        state_mcp.update_state(
            feedback_append={"type": "rating", "value": 1, "context": "good reply"}
        )
        state = state_mcp.get_state()
        assert "feedback" in state
        assert len(state["feedback"]) == 1
        assert state["feedback"][0]["type"] == "rating"


# ---------------------------------------------------------------------------
# clear_state
# ---------------------------------------------------------------------------


class TestClearState:
    def test_clear_empties_state(self, temp_config: Path):
        """clear_state removes all state."""
        state_mcp.update_state(recent_files=["a.py"])
        result = state_mcp.clear_state()
        assert "Cleared" in result
        state = state_mcp.get_state()
        assert state == {}


# ---------------------------------------------------------------------------
# append_recent_file
# ---------------------------------------------------------------------------


class TestAppendRecentFile:
    def test_append_adds_file(self, temp_config: Path):
        """append_recent_file adds a path to recent_files."""
        state_mcp.append_recent_file("src/main.py")
        state = state_mcp.get_state()
        assert "src/main.py" in state.get("recent_files", [])

    def test_append_deduplicates(self, temp_config: Path):
        """append_recent_file moves duplicates to end."""
        state_mcp.append_recent_file("a.py")
        state_mcp.append_recent_file("b.py")
        state_mcp.append_recent_file("a.py")
        state = state_mcp.get_state()
        recent = state.get("recent_files", [])
        assert recent.count("a.py") == 1
        assert recent[-1] == "a.py"


# ---------------------------------------------------------------------------
# record_reasoning
# ---------------------------------------------------------------------------


class TestRecordReasoning:
    def test_reasoning_persisted(self, temp_config: Path):
        """record_reasoning stores steps and conclusion."""
        result = state_mcp.record_reasoning(
            ["analyze", "decide"], conclusion="go ahead"
        )
        assert result == "ok"
        state = state_mcp.get_state()
        assert state["last_reasoning"]["steps"] == ["analyze", "decide"]
        assert state["last_reasoning"]["conclusion"] == "go ahead"


# ---------------------------------------------------------------------------
# Knowledge graph operations
# ---------------------------------------------------------------------------


class TestKnowledgeGraph:
    def test_add_and_query(self, temp_config: Path):
        """add_knowledge_node upserts a node; query_knowledge_graph finds it."""
        state_mcp.add_knowledge_node("Auth", summary="JWT tokens", related=["security"])
        results = state_mcp.query_knowledge_graph("JWT")
        assert len(results) > 0
        assert results[0]["topic"] == "Auth"

    def test_query_no_match(self, temp_config: Path):
        """query_knowledge_graph returns [] for no matches."""
        results = state_mcp.query_knowledge_graph("nonexistent_topic_xyz")
        assert results == []

    def test_upsert_updates_existing(self, temp_config: Path):
        """add_knowledge_node updates an existing node with the same topic."""
        state_mcp.add_knowledge_node("API", summary="REST v1")
        state_mcp.add_knowledge_node("api", summary="REST v2 with GraphQL")
        results = state_mcp.query_knowledge_graph("REST v2")
        assert len(results) > 0
        assert "v2" in results[0].get("summary", "").lower()
