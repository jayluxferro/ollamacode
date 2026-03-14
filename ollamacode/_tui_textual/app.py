"""OllamaCode TUI application -- Textual-based OpenCode-style interface."""

from __future__ import annotations

import logging
import os
from typing import Any

from textual.app import App
from textual.binding import Binding

from .context.state import AppState, SessionState
from .context.theme import generate_css, get_theme

logger = logging.getLogger(__name__)


class OllamaCodeApp(App):
    """Main OllamaCode TUI application."""

    CSS_PATH = "styles.tcss"

    TITLE = "OllamaCode"

    BINDINGS = [
        Binding("ctrl+n", "new_session", "New Session", show=True),
        Binding("ctrl+p", "command_palette", "Commands", show=True),
        Binding("ctrl+l", "clear_screen", "Clear", show=False),
        Binding("ctrl+backslash", "toggle_sidebar", "Sidebar", show=True),
        Binding("escape", "cancel_generation", "Cancel", show=False),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    # Application state
    app_state: AppState
    session_state: SessionState

    def __init__(
        self,
        *,
        session: Any = None,
        model: str = "",
        system_extra: str = "",
        workspace_root: str | None = None,
        provider: Any = None,
        provider_name: str = "ollama",
        theme_name: str = "opencode",
        session_id: str | None = None,
        session_title: str | None = None,
        session_history: list[dict[str, Any]] | None = None,
        confirm_tool_calls: bool = False,
        autonomous_mode: bool = False,
        max_tool_rounds: int = 20,
        max_messages: int = 0,
        max_tool_result_chars: int = 0,
        linter_command: str | None = None,
        test_command: str | None = None,
        docs_command: str | None = None,
        profile_command: str | None = None,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.mcp_session = session
        self.model = model
        self.system_extra = system_extra
        self.provider = provider
        self.provider_name = provider_name
        self.confirm_tool_calls = confirm_tool_calls
        self.autonomous_mode = autonomous_mode
        self.max_tool_rounds = max_tool_rounds
        self.max_messages = max_messages
        self.max_tool_result_chars = max_tool_result_chars
        self.linter_command = linter_command
        self.test_command = test_command
        self.docs_command = docs_command
        self.profile_command = profile_command
        self._config = config or {}

        self.app_state = AppState(
            workspace_root=workspace_root or os.getcwd(),
        )
        self.session_state = SessionState(
            session_id=session_id or "",
            title=session_title or "",
            model=model,
            provider_name=provider_name,
            autonomous=autonomous_mode,
        )
        self.session_history: list[dict[str, Any]] = session_history or []
        self._theme_name = theme_name

        # Initialize integration managers
        self._init_managers()

    def _init_managers(self) -> None:
        """Initialize integration managers from config."""
        cfg = self._config
        try:
            from ollamacode.permission_runtime import SessionApprovalStore

            self.app_state.permission_state = SessionApprovalStore()
        except Exception:
            logger.debug("Failed to init SessionApprovalStore", exc_info=True)

        # Permissions
        try:
            from ollamacode.permissions import PermissionManager

            self.app_state.permissions_manager = PermissionManager.from_config(cfg or None)
        except Exception:
            logger.debug("Failed to init PermissionManager", exc_info=True)

        # Agent modes
        try:
            from ollamacode.agent_modes import ModeManager

            self.app_state.mode_manager = ModeManager.from_config(cfg)
        except Exception:
            logger.debug("Failed to init ModeManager", exc_info=True)

        # Custom commands
        try:
            from ollamacode.custom_commands import CommandManager

            cm = CommandManager()
            cm.load_from_config(cfg)
            self.app_state.command_manager = cm
        except Exception:
            logger.debug("Failed to init CommandManager", exc_info=True)

        # Model variants
        try:
            from ollamacode.model_variants import VariantManager

            vm = VariantManager()
            vm.load_from_config(cfg)
            self.app_state.variant_manager = vm
        except Exception:
            logger.debug("Failed to init VariantManager", exc_info=True)

        # Plugin system
        try:
            from ollamacode.plugins import PluginManager

            pm = PluginManager()
            plugins_dir = cfg.get("plugins_dir", "")
            if plugins_dir and os.path.isdir(plugins_dir):
                pm.load_plugins_from_dir(plugins_dir)
            self.app_state.plugin_manager = pm
        except Exception:
            logger.debug("Failed to init PluginManager", exc_info=True)

    def compose(self):
        """Mount the toast layer."""
        from .widgets.toast import ToastContainer

        yield ToastContainer(id="toast-container")

    def on_mount(self) -> None:
        """Set up theme and push initial screen."""
        # Apply theme CSS
        theme = get_theme(self._theme_name)
        self.stylesheet.add_source(generate_css(theme), "theme")

        # Start file watcher
        self._start_file_watcher()

        # Push home or session screen
        from .screens.home import HomeScreen
        from .screens.session import SessionScreen

        if self.session_state.session_id:
            self.push_screen(SessionScreen())
        else:
            self.push_screen(HomeScreen())

    def _start_file_watcher(self) -> None:
        """Start file watcher for workspace directory."""
        try:
            from ollamacode.file_watcher import watch_directory

            patterns = ["*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.go",
                        "*.rs", "*.java", "*.c", "*.cpp", "*.h", "*.md",
                        "*.yaml", "*.yml", "*.json", "*.toml"]

            def on_file_change(event_type: str, file_path: str) -> None:
                try:
                    self.call_from_thread(
                        self._record_file_change, event_type, file_path
                    )
                except Exception:
                    logger.debug("Failed to enqueue file change", exc_info=True)

            handle = watch_directory(
                self.app_state.workspace_root,
                on_file_change,
                patterns=patterns,
            )
            self.app_state.file_watcher_handle = handle
        except Exception:
            logger.debug("Failed to start file watcher", exc_info=True)

    def _record_file_change(self, event_type: str, file_path: str) -> None:
        """Record a file watcher event on the UI thread."""
        rel_path = os.path.relpath(file_path, self.app_state.workspace_root)
        updated = [
            item
            for item in self.app_state.modified_files
            if item.get("path") != rel_path
        ]
        updated.append(
            {
                "path": rel_path,
                "event": event_type,
                "added": 0,
                "removed": 0,
            }
        )
        self.app_state.modified_files = updated[-50:]
        self._refresh_sidebar()

    def load_session_todos(self, session_id: str | None = None) -> None:
        """Load persisted todos for the active session."""
        sid = session_id or self.session_state.session_id
        if not sid:
            self.app_state.todos = []
            self._refresh_sidebar()
            return
        try:
            from ollamacode.sessions import load_session_todos

            self.app_state.todos = load_session_todos(sid) or []
        except Exception:
            logger.debug("Failed to load session todos", exc_info=True)
            self.app_state.todos = []
        self._refresh_sidebar()

    def set_session_todos(self, todos: list[dict[str, Any]]) -> None:
        """Persist todos for the active session and refresh the UI."""
        self.app_state.todos = list(todos)
        session_id = self.session_state.session_id
        if session_id:
            try:
                from ollamacode.sessions import save_session_todos

                save_session_todos(session_id, self.app_state.todos)
            except Exception:
                logger.debug("Failed to save session todos", exc_info=True)
        self._refresh_sidebar()

    def _refresh_sidebar(self) -> None:
        """Synchronize sidebar widgets with app state."""
        try:
            from .widgets.sidebar import Sidebar

            sidebar = self.screen.query_one(Sidebar)
            sidebar.session_title = self.session_state.title or "New Session"
            sidebar.context_limit = self.session_state.context_limit
            sidebar.token_count = self.session_state.token_count
            sidebar.cost = self.session_state.cost
            sidebar.agent_mode = self.session_state.agent_mode
            sidebar.permissions_granted = self.session_state.permissions_granted
            sidebar.permissions_denied = 0
            sidebar.checkpoint_count = self.session_state.checkpoint_count
            sidebar.plugin_count = (
                len(self.app_state.plugin_manager.plugins)
                if self.app_state.plugin_manager is not None
                else 0
            )
            sidebar.mcp_servers = list(self.app_state.mcp_servers)
            sidebar.lsp_servers = list(self.app_state.lsp_servers)
            sidebar.todos = list(self.app_state.todos)
            sidebar.modified_files = list(self.app_state.modified_files)
        except Exception:
            logger.debug("Sidebar refresh skipped", exc_info=True)

    def show_toast(
        self,
        message: str,
        *,
        title: str = "",
        variant: str = "info",
        duration: float = 5.0,
    ) -> None:
        """Show a toast notification."""
        try:
            from .widgets.toast import ToastContainer

            container = self.query_one("#toast-container", ToastContainer)
            container.show(message, title=title, variant=variant, duration=duration)
        except Exception:
            # Fall back to Textual's built-in notify
            self.notify(message, title=title, timeout=int(duration))

    def action_new_session(self) -> None:
        """Start a new session."""
        from .screens.home import HomeScreen

        self.session_state = SessionState(
            model=self.model,
            provider_name=self.provider_name,
        )
        self.session_history = []
        self.app_state.todos = []
        self.push_screen(HomeScreen())

    def action_toggle_sidebar(self) -> None:
        """Toggle sidebar visibility."""
        try:
            from .widgets.sidebar import Sidebar

            sidebar = self.query_one(Sidebar)
            sidebar.toggle_class("-hidden")
        except Exception:
            logger.debug("Sidebar not found or toggle failed", exc_info=True)

    def action_cancel_generation(self) -> None:
        """Cancel any running generation."""
        self.session_state.is_busy = False
        self.session_state.is_streaming = False

    def action_clear_screen(self) -> None:
        """Clear the current screen."""
        from .screens.session import SessionScreen

        screen = self.screen
        if isinstance(screen, SessionScreen):
            screen.clear_messages()

    def on_unmount(self) -> None:
        """Clean up resources on exit."""
        # Stop file watcher
        handle = self.app_state.file_watcher_handle
        if handle is not None:
            try:
                handle.stop()
            except Exception:
                pass

        # Unload plugins
        pm = self.app_state.plugin_manager
        if pm is not None:
            try:
                pm.unload_all()
            except Exception:
                pass
