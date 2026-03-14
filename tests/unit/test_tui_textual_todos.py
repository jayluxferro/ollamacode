from __future__ import annotations

import pytest

from ollamacode.sessions import create_session, load_session_todos, save_session_todos


@pytest.mark.asyncio
async def test_session_screen_loads_persisted_todos(tmp_path, monkeypatch):
    """The Textual session screen should load persisted session todos."""
    monkeypatch.setattr("ollamacode.sessions._DB_PATH", tmp_path / "sessions.db")
    session_id = create_session("Todo session", workspace_root=str(tmp_path))
    save_session_todos(
        session_id,
        [{"content": "Persisted task", "status": "in_progress", "priority": "high"}],
    )

    from ollamacode._tui_textual.app import OllamaCodeApp
    from ollamacode._tui_textual.widgets.sidebar import Sidebar

    monkeypatch.setattr(OllamaCodeApp, "CSS_PATH", None)
    app = OllamaCodeApp(
        model="test-model",
        system_extra="",
        workspace_root=str(tmp_path),
        session_id=session_id,
        session_title="Todo session",
    )

    async with app.run_test(size=(120, 40)):
        sidebar = app.screen.query_one(Sidebar)
        assert app.app_state.todos == [
            {
                "content": "Persisted task",
                "status": "in_progress",
                "priority": "high",
            }
        ]
        assert sidebar.todos == app.app_state.todos


@pytest.mark.asyncio
async def test_todo_slash_commands_persist_changes(tmp_path, monkeypatch):
    """Todo slash commands should persist changes to the session DB."""
    monkeypatch.setattr("ollamacode.sessions._DB_PATH", tmp_path / "sessions.db")
    session_id = create_session("Todo commands", workspace_root=str(tmp_path))

    from ollamacode._tui_textual.app import OllamaCodeApp
    from ollamacode._tui_textual.screens.session import SessionScreen

    monkeypatch.setattr(OllamaCodeApp, "CSS_PATH", None)
    app = OllamaCodeApp(
        model="test-model",
        system_extra="",
        workspace_root=str(tmp_path),
        session_id=session_id,
        session_title="Todo commands",
    )

    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        assert isinstance(screen, SessionScreen)

        screen._handle_todo_command("add Finish the sidebar wiring")
        await pilot.pause()
        assert load_session_todos(session_id) == [
            {
                "content": "Finish the sidebar wiring",
                "status": "pending",
                "priority": "medium",
            }
        ]

        screen._handle_todo_command("done 1")
        await pilot.pause()
        assert load_session_todos(session_id) == [
            {
                "content": "Finish the sidebar wiring",
                "status": "completed",
                "priority": "medium",
            }
        ]
