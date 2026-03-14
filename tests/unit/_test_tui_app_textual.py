"""Unit tests for TUI app and screens using Textual's async test harness."""

from __future__ import annotations

import pytest



# ---------------------------------------------------------------------------
# OllamaCodeApp construction
# ---------------------------------------------------------------------------


class TestOllamaCodeApp:
    """Tests for OllamaCodeApp initialization."""

    def test_app_creation(self) -> None:
        from ollamacode.tui.app import OllamaCodeApp

        app = OllamaCodeApp(model="test-model", system_extra="")
        assert app.model == "test-model"
        assert app.provider_name == "ollama"
        assert app.confirm_tool_calls is False
        assert app.autonomous_mode is False

    def test_app_state_initialized(self) -> None:
        from ollamacode.tui.app import OllamaCodeApp

        app = OllamaCodeApp(
            model="test",
            system_extra="",
            workspace_root="/tmp/test",
            provider_name="openai",
        )
        assert app.app_state.workspace_root == "/tmp/test"
        assert app.session_state.model == "test"
        assert app.session_state.provider_name == "openai"

    def test_app_with_session_id(self) -> None:
        from ollamacode.tui.app import OllamaCodeApp

        app = OllamaCodeApp(
            model="test",
            system_extra="",
            session_id="abc-123",
            session_title="My Session",
        )
        assert app.session_state.session_id == "abc-123"
        assert app.session_state.title == "My Session"

    def test_app_with_history(self) -> None:
        from ollamacode.tui.app import OllamaCodeApp

        history = [{"role": "user", "content": "hello"}]
        app = OllamaCodeApp(
            model="test",
            system_extra="",
            session_history=history,
        )
        assert app.session_history == history

    def test_app_autonomous_mode(self) -> None:
        from ollamacode.tui.app import OllamaCodeApp

        app = OllamaCodeApp(
            model="test",
            system_extra="",
            autonomous_mode=True,
        )
        assert app.autonomous_mode is True
        assert app.session_state.autonomous is True

    def test_app_default_workspace(self) -> None:
        """When no workspace_root given, default to cwd."""
        import os
        from ollamacode.tui.app import OllamaCodeApp

        app = OllamaCodeApp(model="test", system_extra="")
        assert app.app_state.workspace_root == os.getcwd()

    def test_app_bindings_defined(self) -> None:
        from ollamacode.tui.app import OllamaCodeApp

        binding_keys = [b.key for b in OllamaCodeApp.BINDINGS]
        assert "ctrl+n" in binding_keys
        assert "ctrl+p" in binding_keys
        assert "ctrl+backslash" in binding_keys
        assert "escape" in binding_keys
        assert "ctrl+c" in binding_keys


# ---------------------------------------------------------------------------
# Async app tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOllamaCodeAppAsync:
    """Async tests that start/mount the app."""

    async def test_app_mounts_home_screen(self) -> None:
        """App should push HomeScreen on mount when no session_id."""
        from ollamacode.tui.app import OllamaCodeApp
        from ollamacode.tui.screens.home import HomeScreen

        app = OllamaCodeApp(model="test", system_extra="")

        async with app.run_test(size=(120, 40)) as pilot:
            assert isinstance(app.screen, HomeScreen)

    async def test_app_mounts_session_screen_with_id(self) -> None:
        """App should push SessionScreen when session_id is given."""
        from ollamacode.tui.app import OllamaCodeApp
        from ollamacode.tui.screens.session import SessionScreen

        app = OllamaCodeApp(
            model="test",
            system_extra="",
            session_id="test-session-id",
        )

        async with app.run_test(size=(120, 40)) as pilot:
            assert isinstance(app.screen, SessionScreen)

    async def test_home_screen_has_logo(self) -> None:
        """HomeScreen should contain a Logo widget."""
        from ollamacode.tui.app import OllamaCodeApp
        from ollamacode.tui.widgets.logo import Logo

        app = OllamaCodeApp(model="test", system_extra="")

        async with app.run_test(size=(120, 40)) as pilot:
            logos = app.screen.query(Logo)
            assert len(logos) >= 1

    async def test_home_screen_has_prompt(self) -> None:
        """HomeScreen should contain a PromptInput widget."""
        from ollamacode.tui.app import OllamaCodeApp
        from ollamacode.tui.widgets.prompt import PromptInput

        app = OllamaCodeApp(model="test", system_extra="")

        async with app.run_test(size=(120, 40)) as pilot:
            prompts = app.screen.query(PromptInput)
            assert len(prompts) >= 1

    async def test_session_screen_has_header(self) -> None:
        """SessionScreen should have a SessionHeader."""
        from ollamacode.tui.app import OllamaCodeApp
        from ollamacode.tui.widgets.header import SessionHeader

        app = OllamaCodeApp(
            model="test-model",
            system_extra="",
            session_id="test-session",
        )

        async with app.run_test(size=(120, 40)) as pilot:
            headers = app.screen.query(SessionHeader)
            assert len(headers) >= 1

    async def test_session_screen_has_sidebar(self) -> None:
        """SessionScreen should have a Sidebar."""
        from ollamacode.tui.app import OllamaCodeApp
        from ollamacode.tui.widgets.sidebar import Sidebar

        app = OllamaCodeApp(
            model="test",
            system_extra="",
            session_id="test-session",
        )

        async with app.run_test(size=(120, 40)) as pilot:
            sidebars = app.screen.query(Sidebar)
            assert len(sidebars) >= 1

    async def test_session_screen_has_footer(self) -> None:
        """SessionScreen should have a SessionFooter."""
        from ollamacode.tui.app import OllamaCodeApp
        from ollamacode.tui.widgets.footer import SessionFooter

        app = OllamaCodeApp(
            model="test",
            system_extra="",
            session_id="test-session",
        )

        async with app.run_test(size=(120, 40)) as pilot:
            footers = app.screen.query(SessionFooter)
            assert len(footers) >= 1

    async def test_new_session_action_returns_to_home(self) -> None:
        """action_new_session should switch to HomeScreen."""
        from ollamacode.tui.app import OllamaCodeApp
        from ollamacode.tui.screens.home import HomeScreen

        app = OllamaCodeApp(
            model="test",
            system_extra="",
            session_id="test-session",
        )

        async with app.run_test(size=(120, 40)) as pilot:
            app.action_new_session()
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)
            assert app.session_state.session_id == ""
