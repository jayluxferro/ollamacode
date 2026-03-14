"""Unit tests for TUI widget logic that can be tested without a running app."""

from __future__ import annotations


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
    """Tests for the knight-rider scanner spinner."""

    def test_spinner_alias(self) -> None:
        from ollamacode._tui_textual.widgets.spinner import (
            BrailleSpinner,
            KnightRiderSpinner,
        )

        # BrailleSpinner is an alias for KnightRiderSpinner
        assert BrailleSpinner is KnightRiderSpinner

    def test_spinner_has_label(self) -> None:
        from ollamacode._tui_textual.widgets.spinner import KnightRiderSpinner

        spinner = KnightRiderSpinner()
        # label is a reactive property
        assert hasattr(spinner, "label")


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
        from ollamacode._tui_textual.dialogs.tool_confirm import TOOL_ICONS

        assert "run_command" in TOOL_ICONS
        assert "bash" in TOOL_ICONS
        assert "read_file" in TOOL_ICONS


# ---------------------------------------------------------------------------
# Command palette slash commands
# ---------------------------------------------------------------------------


class TestSlashCommands:
    """Tests for slash command definitions."""

    def test_slash_commands_defined(self) -> None:
        from ollamacode._tui_textual.dialogs.command_palette import SlashCommands

        assert len(SlashCommands.COMMANDS) >= 10
        # Each should be a tuple of (command, description)
        for cmd, desc in SlashCommands.COMMANDS:
            assert cmd.startswith("/")
            assert isinstance(desc, str)
            assert len(desc) > 0


# ---------------------------------------------------------------------------
# Sanitize stream text (backward compat import)
# ---------------------------------------------------------------------------


class TestNewStateFields:
    """Tests for new state fields added during Textual TUI implementation."""

    def test_session_state_permissions_denied(self) -> None:
        from ollamacode._tui_textual.context.state import SessionState

        state = SessionState()
        assert state.permissions_denied == 0
        state.permissions_denied = 5
        assert state.permissions_denied == 5

    def test_session_state_compact_mode(self) -> None:
        from ollamacode._tui_textual.context.state import SessionState

        state = SessionState()
        assert state.compact_mode == "off"
        state.compact_mode = "auto"
        assert state.compact_mode == "auto"

    def test_session_state_variant_name(self) -> None:
        from ollamacode._tui_textual.context.state import SessionState

        state = SessionState()
        assert state.variant_name == ""
        state.variant_name = "fast"
        assert state.variant_name == "fast"

    def test_session_state_checkpoint_count(self) -> None:
        from ollamacode._tui_textual.context.state import SessionState

        state = SessionState()
        assert state.checkpoint_count == 0

    def test_session_state_trace_filter(self) -> None:
        from ollamacode._tui_textual.context.state import SessionState

        state = SessionState()
        assert state.trace_filter == ""

    def test_app_state_managers_default_none(self) -> None:
        from ollamacode._tui_textual.context.state import AppState

        state = AppState()
        assert state.permissions_manager is None
        assert state.permission_state is None
        assert state.mode_manager is None
        assert state.command_manager is None
        assert state.variant_manager is None
        assert state.plugin_manager is None
        assert state.file_watcher_handle is None


# ---------------------------------------------------------------------------
# New dialog tests
# ---------------------------------------------------------------------------


class TestNewDialogs:
    """Tests for the new dialogs added during Textual TUI implementation."""

    def test_checkpoint_list_dialog_import(self) -> None:
        from ollamacode._tui_textual.dialogs.checkpoint_list import (
            CheckpointListDialog,
        )

        assert CheckpointListDialog is not None

    def test_export_import_dialog_import(self) -> None:
        from ollamacode._tui_textual.dialogs.export_import import (
            ExportDialog,
            ImportDialog,
        )

        assert ExportDialog is not None
        assert ImportDialog is not None

    def test_refactor_dialog_import(self) -> None:
        from ollamacode._tui_textual.dialogs.refactor import RefactorDialog

        assert RefactorDialog is not None
        assert len(RefactorDialog.OPERATIONS) >= 4

    def test_refactor_operations_have_ids(self) -> None:
        from ollamacode._tui_textual.dialogs.refactor import RefactorDialog

        for op_id, label, desc in RefactorDialog.OPERATIONS:
            assert isinstance(op_id, str)
            assert len(op_id) > 0
            assert isinstance(label, str)
            assert isinstance(desc, str)


# ---------------------------------------------------------------------------
# App COMMANDS provider registration
# ---------------------------------------------------------------------------


class TestAppCommands:
    """Tests for the command palette provider registration."""

    def test_commands_are_provider_classes(self) -> None:
        from textual.command import Provider

        from ollamacode._tui_textual.app import OllamaCodeApp

        assert len(OllamaCodeApp.COMMANDS) == 4
        for cmd in OllamaCodeApp.COMMANDS:
            assert isinstance(cmd, type)
            assert issubclass(cmd, Provider)

    def test_app_accepts_tool_params(self) -> None:
        from ollamacode._tui_textual.app import OllamaCodeApp

        app = OllamaCodeApp(
            model="test",
            allowed_tools=["read_file"],
            blocked_tools=["bash"],
        )
        assert app.allowed_tools == ["read_file"]
        assert app.blocked_tools == ["bash"]

    def test_app_accepts_config(self) -> None:
        from ollamacode._tui_textual.app import OllamaCodeApp

        app = OllamaCodeApp(model="test", config={"key": "val"})
        assert app._config == {"key": "val"}


# ---------------------------------------------------------------------------
# Footer reactives
# ---------------------------------------------------------------------------


class TestFooterReactives:
    """Tests for the new footer reactive properties."""

    def test_footer_has_variant_name(self) -> None:
        from ollamacode._tui_textual.widgets.footer import SessionFooter

        f = SessionFooter()
        assert hasattr(f, "variant_name")

    def test_footer_has_sandbox_level(self) -> None:
        from ollamacode._tui_textual.widgets.footer import SessionFooter

        f = SessionFooter()
        assert hasattr(f, "sandbox_level")


# ---------------------------------------------------------------------------
# Sidebar reactives
# ---------------------------------------------------------------------------


class TestSidebarReactives:
    """Tests for the new sidebar reactive properties."""

    def test_sidebar_has_permissions_denied(self) -> None:
        from ollamacode._tui_textual.widgets.sidebar import Sidebar

        s = Sidebar()
        assert hasattr(s, "permissions_denied")

    def test_sidebar_has_checkpoint_count(self) -> None:
        from ollamacode._tui_textual.widgets.sidebar import Sidebar

        s = Sidebar()
        assert hasattr(s, "checkpoint_count")

    def test_sidebar_has_plugin_count(self) -> None:
        from ollamacode._tui_textual.widgets.sidebar import Sidebar

        s = Sidebar()
        assert hasattr(s, "plugin_count")

    def test_sidebar_has_agent_mode(self) -> None:
        from ollamacode._tui_textual.widgets.sidebar import Sidebar

        s = Sidebar()
        assert hasattr(s, "agent_mode")


# ---------------------------------------------------------------------------
# Prompt @-ref expansion
# ---------------------------------------------------------------------------


class TestPromptExpansion:
    """Tests for the prompt @-ref expansion method."""

    def test_expand_at_refs_noop_without_at(self) -> None:
        from ollamacode._tui_textual.widgets.prompt import PromptInput

        p = PromptInput()
        assert p.expand_at_refs("hello world") == "hello world"

    def test_expand_at_refs_returns_string(self) -> None:
        from ollamacode._tui_textual.widgets.prompt import PromptInput

        p = PromptInput()
        result = p.expand_at_refs("check @nonexistent_file_xyz")
        assert isinstance(result, str)


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
