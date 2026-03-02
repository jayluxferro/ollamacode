"""Unit tests for TUI widget logic that can be tested without a running app."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Theme helpers
# ---------------------------------------------------------------------------


class TestTheme:
    """Tests for the theme system."""

    def test_list_themes_returns_sorted(self) -> None:
        from ollamacode.tui.context.theme import list_themes

        names = list_themes()
        assert names == sorted(names)
        assert len(names) >= 10

    def test_get_theme_known(self) -> None:
        from ollamacode.tui.context.theme import get_theme

        theme = get_theme("dracula")
        assert theme["primary"] == "#bd93f9"
        assert "background" in theme
        assert "text" in theme

    def test_get_theme_unknown_returns_default(self) -> None:
        from ollamacode.tui.context.theme import DEFAULT_THEME, get_theme

        default = get_theme(DEFAULT_THEME)
        fallback = get_theme("nonexistent-theme")
        assert fallback == default

    def test_generate_css(self) -> None:
        from ollamacode.tui.context.theme import generate_css, get_theme

        theme = get_theme("opencode")
        css = generate_css(theme)
        assert "$primary:" in css
        assert "$background:" in css
        assert "$text:" in css
        # Check CSS variable naming convention (hyphens, no prefix)
        assert "$background-panel:" in css
        assert "$border-active:" in css

    def test_theme_has_all_keys(self) -> None:
        from ollamacode.tui.context.theme import THEMES, Theme

        required = set(Theme.__annotations__.keys())
        for name, theme in THEMES.items():
            assert set(theme.keys()) == required, f"Theme {name!r} has wrong keys"


# ---------------------------------------------------------------------------
# Colour math helpers
# ---------------------------------------------------------------------------


class TestColourHelpers:
    """Tests for the internal colour manipulation functions."""

    def test_hex_to_rgb(self) -> None:
        from ollamacode.tui.context.theme import _hex_to_rgb

        assert _hex_to_rgb("#000000") == (0, 0, 0)
        assert _hex_to_rgb("#ffffff") == (255, 255, 255)
        assert _hex_to_rgb("#1e1e2e") == (30, 30, 46)

    def test_lighten_hex(self) -> None:
        from ollamacode.tui.context.theme import _lighten_hex

        result = _lighten_hex("#000000", 0.5)
        # Black lightened by 50% should be ~#808080
        assert result.startswith("#")
        r, g, b = int(result[1:3], 16), int(result[3:5], 16), int(result[5:7], 16)
        assert 120 <= r <= 135  # Allow rounding margin
        assert r == g == b

    def test_darken_hex(self) -> None:
        from ollamacode.tui.context.theme import _darken_hex

        result = _darken_hex("#ffffff", 0.5)
        assert result.startswith("#")
        r = int(result[1:3], 16)
        assert 120 <= r <= 135

    def test_clamp(self) -> None:
        from ollamacode.tui.context.theme import _clamp

        assert _clamp(-10) == 0
        assert _clamp(300) == 255
        assert _clamp(128.6) == 129


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------


class TestState:
    """Tests for the state dataclasses."""

    def test_session_state_defaults(self) -> None:
        from ollamacode.tui.context.state import SessionState

        state = SessionState()
        assert state.session_id == ""
        assert state.model == ""
        assert state.agent_mode == "build"
        assert state.token_count == 0
        assert state.cost == 0.0
        assert state.is_busy is False
        assert state.is_streaming is False
        assert state.autonomous is False

    def test_session_state_custom(self) -> None:
        from ollamacode.tui.context.state import SessionState

        state = SessionState(
            session_id="abc123",
            model="llama3",
            agent_mode="plan",
            token_count=5000,
            cost=0.0123,
        )
        assert state.session_id == "abc123"
        assert state.model == "llama3"
        assert state.agent_mode == "plan"
        assert state.token_count == 5000

    def test_app_state_defaults(self) -> None:
        from ollamacode.tui.context.state import AppState

        state = AppState()
        assert state.workspace_root == "."
        assert state.mcp_servers == []
        assert state.todos == []
        assert state.modified_files == []

    def test_app_state_mutable_defaults_independent(self) -> None:
        from ollamacode.tui.context.state import AppState

        a = AppState()
        b = AppState()
        a.mcp_servers.append({"name": "test"})
        assert b.mcp_servers == []


# ---------------------------------------------------------------------------
# Keybinds
# ---------------------------------------------------------------------------


class TestKeybinds:
    """Tests for keybinding definitions."""

    def test_default_keybinds_has_required_keys(self) -> None:
        from ollamacode.tui.context.keybinds import DEFAULT_KEYBINDS

        required = {
            "submit",
            "newline",
            "new_session",
            "clear",
            "command_palette",
            "toggle_sidebar",
            "cancel",
            "quit",
        }
        assert required.issubset(set(DEFAULT_KEYBINDS.keys()))

    def test_keybind_values_are_strings(self) -> None:
        from ollamacode.tui.context.keybinds import DEFAULT_KEYBINDS

        for key, value in DEFAULT_KEYBINDS.items():
            assert isinstance(value, str), f"Keybind {key!r} has non-string value"


# ---------------------------------------------------------------------------
# Logo widget
# ---------------------------------------------------------------------------


class TestLogo:
    """Tests for the logo widget."""

    def test_logo_art_not_empty(self) -> None:
        from ollamacode.tui.widgets.logo import LOGO_ART

        assert len(LOGO_ART) > 100
        # ASCII art renders letters across multiple lines
        assert "____" in LOGO_ART  # Box-drawing chars in ASCII art


# ---------------------------------------------------------------------------
# Spinner
# ---------------------------------------------------------------------------


class TestSpinner:
    """Tests for the braille spinner."""

    def test_spinner_frames(self) -> None:
        from ollamacode.tui.widgets.spinner import BrailleSpinner

        assert len(BrailleSpinner.FRAMES) == 10
        # All frames should be single braille chars
        for frame in BrailleSpinner.FRAMES:
            assert len(frame) == 1
            assert 0x2800 <= ord(frame) <= 0x28FF


# ---------------------------------------------------------------------------
# Tips
# ---------------------------------------------------------------------------


class TestTips:
    """Tests for the tips module."""

    def test_tips_list_nonempty(self) -> None:
        from ollamacode.tui.widgets.tips import TIPS

        assert len(TIPS) >= 10

    def test_tips_are_markdown(self) -> None:
        from ollamacode.tui.widgets.tips import TIPS

        for tip in TIPS:
            assert isinstance(tip, str)
            assert len(tip) > 5


# ---------------------------------------------------------------------------
# Tool display
# ---------------------------------------------------------------------------


class TestToolDisplay:
    """Tests for tool display widgets."""

    def test_tool_icons_defined(self) -> None:
        from ollamacode.tui.widgets.tool_display import TOOL_ICONS

        assert "bash" in TOOL_ICONS
        assert "read_file" in TOOL_ICONS
        assert "edit_file" in TOOL_ICONS
        assert "grep" in TOOL_ICONS

    def test_make_tool_widget_inline(self) -> None:
        from ollamacode.tui.widgets.tool_display import InlineToolCall, make_tool_widget

        w = make_tool_widget("read_file", {"path": "/foo/bar.py"})
        assert isinstance(w, InlineToolCall)
        assert w.tool_name == "read_file"
        rendered = w.render()
        assert "/foo/bar.py" in rendered

    def test_make_tool_widget_block(self) -> None:
        from ollamacode.tui.widgets.tool_display import BlockToolCall, make_tool_widget

        w = make_tool_widget("bash", {"command": "ls -la"})
        assert isinstance(w, BlockToolCall)
        assert w.tool_name == "bash"
        assert w.tool_args["command"] == "ls -la"

    def test_make_tool_widget_glob(self) -> None:
        from ollamacode.tui.widgets.tool_display import InlineToolCall, make_tool_widget

        w = make_tool_widget("glob", {"pattern": "**/*.py"})
        assert isinstance(w, InlineToolCall)
        rendered = w.render()
        assert "**/*.py" in rendered

    def test_make_tool_widget_unknown(self) -> None:
        from ollamacode.tui.widgets.tool_display import BlockToolCall, make_tool_widget

        w = make_tool_widget("custom_tool", {"arg": "val"})
        assert isinstance(w, BlockToolCall)


# ---------------------------------------------------------------------------
# Tool confirm dialog icons
# ---------------------------------------------------------------------------


class TestToolConfirmIcons:
    """Tests for tool confirmation dialog icon map."""

    def test_tool_icons_subset(self) -> None:
        from ollamacode.tui.dialogs.tool_confirm import TOOL_ICONS

        assert "run_command" in TOOL_ICONS
        assert "bash" in TOOL_ICONS
        assert "read_file" in TOOL_ICONS


# ---------------------------------------------------------------------------
# Command palette slash commands
# ---------------------------------------------------------------------------


class TestSlashCommands:
    """Tests for slash command definitions."""

    def test_slash_commands_defined(self) -> None:
        from ollamacode.tui.dialogs.command_palette import SlashCommands

        assert len(SlashCommands.COMMANDS) >= 10
        # Each should be a tuple of (command, description)
        for cmd, desc in SlashCommands.COMMANDS:
            assert cmd.startswith("/")
            assert isinstance(desc, str)
            assert len(desc) > 0


# ---------------------------------------------------------------------------
# Sanitize stream text (backward compat import)
# ---------------------------------------------------------------------------


class TestSanitizeStreamText:
    """Tests for _sanitize_stream_text exported from tui package."""

    def test_import_from_package(self) -> None:
        from ollamacode.tui import _sanitize_stream_text

        assert callable(_sanitize_stream_text)

    def test_empty_string(self) -> None:
        from ollamacode.tui import _sanitize_stream_text

        assert _sanitize_stream_text("") == ""

    def test_strips_ansi(self) -> None:
        from ollamacode.tui import _sanitize_stream_text

        result = _sanitize_stream_text("\x1b[31mred\x1b[0m")
        assert result == "red"
        assert "\x1b" not in result

    def test_normalizes_crlf(self) -> None:
        from ollamacode.tui import _sanitize_stream_text

        result = _sanitize_stream_text("a\r\nb\rc")
        assert result == "a\nb\nc"

    def test_strips_null(self) -> None:
        from ollamacode.tui import _sanitize_stream_text

        result = _sanitize_stream_text("hello\x00world")
        assert result == "helloworld"

    def test_combined(self) -> None:
        from ollamacode.tui import _sanitize_stream_text

        raw = "\x1b[1mHello\x1b[0m\r\nWorld\x00!"
        result = _sanitize_stream_text(raw)
        assert result == "Hello\nWorld!"
