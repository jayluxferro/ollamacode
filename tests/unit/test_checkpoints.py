"""Unit tests for checkpoints.py — create, restore, list, cleanup."""

import pytest

from ollamacode.checkpoints import (
    CheckpointRecorder,
    get_checkpoint_info,
    get_checkpoint_files,
    get_checkpoint_diff,
    list_checkpoints,
    restore_checkpoint,
)


@pytest.fixture
def checkpoint_db(tmp_path, monkeypatch):
    """Use a temp DB path for checkpoints."""
    db_path = tmp_path / "checkpoints.db"
    monkeypatch.setattr("ollamacode.checkpoints._DB_PATH", db_path)
    return db_path


class TestCheckpointRecorder:
    """Test checkpoint recording of file changes."""

    def test_record_and_finalize(self, tmp_path, checkpoint_db):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        test_file = workspace / "test.txt"
        test_file.write_text("original content")

        recorder = CheckpointRecorder(
            session_id="test-session",
            workspace_root=str(workspace),
            prompt="test prompt",
            message_index=0,
        )
        recorder.record_pre("test.txt")

        # Simulate file change
        test_file.write_text("modified content")

        cp_id = recorder.finalize()
        assert cp_id is not None

        # Verify checkpoint stored
        files = get_checkpoint_files(cp_id)
        assert len(files) == 1
        assert files[0]["path"] == "test.txt"
        assert files[0]["before_content"] == "original content"
        assert files[0]["after_content"] == "modified content"
        assert files[0]["before_exists"] is True
        assert files[0]["after_exists"] is True

    def test_record_new_file(self, tmp_path, checkpoint_db):
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        recorder = CheckpointRecorder(
            session_id="test-session",
            workspace_root=str(workspace),
            prompt="create file",
            message_index=1,
        )
        recorder.record_pre("new.txt")

        # Create file after recording pre-state
        (workspace / "new.txt").write_text("new content")

        cp_id = recorder.finalize()
        files = get_checkpoint_files(cp_id)
        assert len(files) == 1
        assert files[0]["before_exists"] is False
        assert files[0]["after_exists"] is True

    def test_empty_recorder_returns_none(self, tmp_path, checkpoint_db):
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        recorder = CheckpointRecorder(
            session_id="test-session",
            workspace_root=str(workspace),
            prompt="nothing",
            message_index=0,
        )
        assert recorder.finalize() is None

    def test_path_traversal_blocked(self, tmp_path, checkpoint_db):
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        recorder = CheckpointRecorder(
            session_id="test-session",
            workspace_root=str(workspace),
            prompt="test",
            message_index=0,
        )
        recorder.record_pre("../../etc/passwd")
        # Path outside workspace should be rejected
        assert len(recorder._before) == 0 or recorder.finalize() is None


class TestListCheckpoints:
    """Test checkpoint listing."""

    def test_list_empty(self, checkpoint_db):
        result = list_checkpoints("nonexistent-session")
        assert result == []

    def test_list_returns_checkpoints(self, tmp_path, checkpoint_db):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        test_file = workspace / "a.txt"
        test_file.write_text("content")

        recorder = CheckpointRecorder(
            session_id="sess-1",
            workspace_root=str(workspace),
            prompt="first",
            message_index=0,
        )
        recorder.record_pre("a.txt")
        test_file.write_text("changed")
        cp_id = recorder.finalize()

        checkpoints = list_checkpoints("sess-1")
        assert len(checkpoints) == 1
        assert checkpoints[0]["id"] == cp_id
        assert checkpoints[0]["prompt"] == "first"
        assert checkpoints[0]["file_count"] == 1


class TestRestoreCheckpoint:
    """Test checkpoint restoration."""

    def test_restore_reverts_file(self, tmp_path, checkpoint_db):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        test_file = workspace / "test.txt"
        test_file.write_text("original")

        recorder = CheckpointRecorder(
            session_id="sess-1",
            workspace_root=str(workspace),
            prompt="test",
            message_index=0,
        )
        recorder.record_pre("test.txt")
        test_file.write_text("modified")
        cp_id = recorder.finalize()

        # Restore
        modified = restore_checkpoint(cp_id, str(workspace))
        assert "test.txt" in modified
        assert test_file.read_text() == "original"

    def test_restore_deletes_new_file(self, tmp_path, checkpoint_db):
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        recorder = CheckpointRecorder(
            session_id="sess-1",
            workspace_root=str(workspace),
            prompt="test",
            message_index=0,
        )
        recorder.record_pre("new.txt")
        new_file = workspace / "new.txt"
        new_file.write_text("created")
        cp_id = recorder.finalize()

        # Restore should delete the newly created file
        modified = restore_checkpoint(cp_id, str(workspace))
        assert "new.txt" in modified
        assert not new_file.exists()

    def test_restore_uses_stored_workspace_root(self, tmp_path, checkpoint_db):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        test_file = workspace / "test.txt"
        test_file.write_text("original")

        recorder = CheckpointRecorder(
            session_id="sess-1",
            workspace_root=str(workspace),
            prompt="test",
            message_index=0,
        )
        recorder.record_pre("test.txt")
        test_file.write_text("modified")
        cp_id = recorder.finalize()

        modified = restore_checkpoint(cp_id)
        assert "test.txt" in modified
        assert test_file.read_text() == "original"


def test_checkpoint_info_and_diff(tmp_path, checkpoint_db):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    test_file = workspace / "demo.txt"
    test_file.write_text("before\n")

    recorder = CheckpointRecorder(
        session_id="sess-1",
        workspace_root=str(workspace),
        prompt="demo",
        message_index=0,
    )
    recorder.record_pre("demo.txt")
    test_file.write_text("after\n")
    cp_id = recorder.finalize()

    info = get_checkpoint_info(cp_id)
    assert info is not None
    assert info["workspace_root"] == str(workspace)
    diff = get_checkpoint_diff(cp_id)
    assert "--- a/demo.txt" in diff
