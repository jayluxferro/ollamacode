"""Unit tests for session persistence (SQLite)."""

from ollamacode.sessions import (
    branch_session,
    create_session,
    list_child_sessions,
    list_session_ancestors,
    export_session,
    import_session,
    load_session_todos,
    get_session_info,
    list_sessions,
    load_session,
    update_session,
    save_session_todos,
    save_session,
    search_sessions,
)


def test_create_session(tmp_path, monkeypatch):
    """create_session creates a new session and returns a UUID."""
    monkeypatch.setattr("ollamacode.sessions._DB_PATH", tmp_path / "sessions.db")
    sid = create_session("My title")
    assert sid
    assert len(sid) == 36
    info = get_session_info(sid)
    assert info is not None
    assert info["title"] == "My title"
    assert info["message_count"] == 0


def test_save_and_load_session(tmp_path, monkeypatch):
    """save_session and load_session round-trip messages."""
    monkeypatch.setattr("ollamacode.sessions._DB_PATH", tmp_path / "sessions.db")
    sid = create_session("Test")
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    save_session(sid, "Updated title", messages)
    loaded = load_session(sid)
    assert loaded == messages
    info = get_session_info(sid)
    assert info["title"] == "Updated title"
    assert info["message_count"] == 2


def test_update_session(tmp_path, monkeypatch):
    monkeypatch.setattr("ollamacode.sessions._DB_PATH", tmp_path / "sessions.db")
    sid = create_session("Old title", workspace_root="/tmp/a", owner="alice", role="owner")
    updated = update_session(sid, title="New title", workspace_root="/tmp/b", owner="bob", role="editor")
    assert updated is not None
    assert updated["title"] == "New title"
    assert updated["workspace_root"] == "/tmp/b"
    assert updated["owner"] == "bob"
    assert updated["role"] == "editor"


def test_load_session_not_found(tmp_path, monkeypatch):
    """load_session returns None for unknown id."""
    monkeypatch.setattr("ollamacode.sessions._DB_PATH", tmp_path / "sessions.db")
    assert load_session("nonexistent-id-12345") is None


def test_list_sessions(tmp_path, monkeypatch):
    """list_sessions returns sessions by updated_at desc."""
    monkeypatch.setattr("ollamacode.sessions._DB_PATH", tmp_path / "sessions.db")
    s1 = create_session("First")
    s2 = create_session("Second")
    rows = list_sessions(limit=10)
    assert len(rows) >= 2
    ids = [r["id"] for r in rows]
    assert s1 in ids
    assert s2 in ids


def test_search_sessions(tmp_path, monkeypatch):
    """search_sessions filters by title and message content."""
    monkeypatch.setattr("ollamacode.sessions._DB_PATH", tmp_path / "sessions.db")
    sid = create_session("Python refactor")
    save_session(
        sid,
        "Python refactor",
        [
            {"role": "user", "content": "Add type hints"},
            {"role": "assistant", "content": "Here are the type hints."},
        ],
    )
    # Search by title
    rows = search_sessions("Python", limit=10)
    assert any(r["id"] == sid for r in rows)
    # Search by content
    rows2 = search_sessions("type hints", limit=10)
    assert any(r["id"] == sid for r in rows2)
    # No match
    rows3 = search_sessions("xyznonexistent", limit=10)
    assert not any(r["id"] == sid for r in rows3)


def test_branch_session(tmp_path, monkeypatch):
    """branch_session copies messages to a new session."""
    monkeypatch.setattr("ollamacode.sessions._DB_PATH", tmp_path / "sessions.db")
    sid = create_session("Original")
    save_session(
        sid,
        "Original",
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
    )
    new_id = branch_session(sid)
    assert new_id
    assert new_id != sid
    loaded = load_session(new_id)
    assert loaded == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    info = get_session_info(new_id)
    assert "Branch" in (info["title"] or "")
    assert info["parent_session_id"] == sid


def test_branch_session_not_found(tmp_path, monkeypatch):
    """branch_session returns None when source does not exist."""
    monkeypatch.setattr("ollamacode.sessions._DB_PATH", tmp_path / "sessions.db")
    assert branch_session("nonexistent-id") is None


def test_save_and_load_session_todos(tmp_path, monkeypatch):
    """Session todos round-trip with normalized fields."""
    monkeypatch.setattr("ollamacode.sessions._DB_PATH", tmp_path / "sessions.db")
    sid = create_session("Todo session")
    save_session_todos(
        sid,
        [
            {"content": "Inspect sidebar sync", "status": "in_progress"},
            {"text": "Write tests", "done": True, "priority": "high"},
        ],
    )
    assert load_session_todos(sid) == [
        {
            "content": "Inspect sidebar sync",
            "status": "in_progress",
            "priority": "medium",
        },
        {
            "content": "Write tests",
            "status": "completed",
            "priority": "high",
        },
    ]


def test_branch_session_copies_todos(tmp_path, monkeypatch):
    """branch_session copies session todos to the branched session."""
    monkeypatch.setattr("ollamacode.sessions._DB_PATH", tmp_path / "sessions.db")
    sid = create_session("Original")
    save_session_todos(sid, [{"content": "Keep todo state", "status": "pending"}])
    new_id = branch_session(sid)
    assert new_id is not None
    assert load_session_todos(new_id) == [
        {
            "content": "Keep todo state",
            "status": "pending",
            "priority": "medium",
        }
    ]


def test_list_child_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr("ollamacode.sessions._DB_PATH", tmp_path / "sessions.db")
    parent = create_session("Parent")
    child = branch_session(parent)
    rows = list_child_sessions(parent)
    assert any(row["id"] == child for row in rows)


def test_list_session_ancestors(tmp_path, monkeypatch):
    monkeypatch.setattr("ollamacode.sessions._DB_PATH", tmp_path / "sessions.db")
    root = create_session("Root")
    child = branch_session(root)
    grandchild = branch_session(child)
    rows = list_session_ancestors(grandchild)
    assert [row["id"] for row in rows[:2]] == [child, root]


def test_export_import_session_preserves_todos(tmp_path, monkeypatch):
    """Session export/import includes per-session todos."""
    monkeypatch.setattr("ollamacode.sessions._DB_PATH", tmp_path / "sessions.db")
    sid = create_session("Exportable")
    save_session(
        sid,
        "Exportable",
        [{"role": "user", "content": "hello"}],
    )
    save_session_todos(
        sid,
        [{"content": "Port todo support", "status": "completed", "priority": "high"}],
    )

    exported = export_session(sid)
    assert exported is not None

    imported_id = import_session(exported, title="Imported")
    assert load_session_todos(imported_id) == [
        {
            "content": "Port todo support",
            "status": "completed",
            "priority": "high",
        }
    ]
