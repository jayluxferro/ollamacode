"""Session screen — main chat view with messages, sidebar, and streaming."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Static

from ..widgets.footer import SessionFooter
from ..widgets.header import SessionHeader
from ..widgets.messages import AssistantMessage, MessageList, UserMessage
from ..widgets.prompt import PromptInput
from ..widgets.sidebar import Sidebar
from ..widgets.spinner import BrailleSpinner
from ..widgets.tool_display import make_tool_widget

logger = logging.getLogger(__name__)


# ── Custom Messages ──────────────────────────────────────────────────


class ToolStarted(Message):
    """A tool has started executing."""

    def __init__(self, name: str, args: dict[str, Any]) -> None:
        super().__init__()
        self.name = name
        self.args = args


class ToolFinished(Message):
    """A tool has finished executing."""

    def __init__(self, name: str, args: dict[str, Any], summary: str) -> None:
        super().__init__()
        self.name = name
        self.args = args
        self.summary = summary


class StreamChunk(Message):
    """A streaming text chunk from the LLM."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class StreamDone(Message):
    """Streaming is complete."""

    def __init__(self, full_text: str) -> None:
        super().__init__()
        self.full_text = full_text


class GenerationError(Message):
    """An error occurred during generation."""

    def __init__(self, error: str) -> None:
        super().__init__()
        self.error = error


# ── Session Screen ───────────────────────────────────────────────────


class SessionScreen(Screen):
    """Main chat session screen with messages, prompt, and sidebar."""

    BINDINGS = [
        Binding("ctrl+n", "new_session", "New", show=True),
        Binding("ctrl+backslash", "toggle_sidebar", "Sidebar", show=True),
        Binding("escape", "cancel_generation", "Cancel", show=False),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    def __init__(
        self,
        initial_prompt: str = "",
        resume_session_id: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._initial_prompt = initial_prompt
        self._resume_session_id = resume_session_id
        self._current_assistant_msg: AssistantMessage | None = None
        self._generation_cancelled = False
        self._tool_confirm_future: asyncio.Future[str] | None = None

    def compose(self) -> ComposeResult:
        yield SessionHeader(id="session-header")

        with Horizontal(id="session-body"):
            with Vertical(id="session-main"):
                yield MessageList(id="message-list")
                yield BrailleSpinner(id="prompt-spinner")
                yield PromptInput(id="session-prompt")
                yield Static("", id="context-bar")
            yield Sidebar(id="session-sidebar")

        yield SessionFooter(id="session-footer")

    def on_mount(self) -> None:
        """Initialize session state and widgets."""
        app = self.app
        state = app.session_state

        # Configure header
        header = self.query_one(SessionHeader)
        header.title = state.title or "New Session"
        header.model_name = f"{app.provider_name}/{state.model}"

        # Configure footer
        footer = self.query_one(SessionFooter)
        footer.directory = app.app_state.workspace_root
        footer.agent_mode = state.agent_mode

        # Configure sidebar
        sidebar = self.query_one(Sidebar)
        sidebar.session_title = state.title or "New Session"
        sidebar.context_limit = state.context_limit

        # Hide spinner initially
        spinner = self.query_one("#prompt-spinner", BrailleSpinner)
        spinner.display = False

        # Update context bar
        self._update_context_bar()

        # Configure prompt
        prompt = self.query_one("#session-prompt", PromptInput)
        prompt.workspace_root = app.app_state.workspace_root

        # Load existing messages if resuming
        if self._resume_session_id:
            self._load_session_messages()

        # Send initial prompt if provided
        if self._initial_prompt:
            self.call_after_refresh(self._send_initial_prompt)

    def _send_initial_prompt(self) -> None:
        """Send the initial prompt after screen is mounted."""
        if self._initial_prompt:
            self._handle_user_input(self._initial_prompt)
            self._initial_prompt = ""

    def _load_session_messages(self) -> None:
        """Load messages from a resumed session."""
        try:
            from ollamacode.sessions import load_session

            messages = load_session(self._resume_session_id)
            if messages:
                self.app.session_history = messages
                message_list = self.query_one("#message-list", MessageList)
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        message_list.mount(UserMessage(content))
                    elif role == "assistant":
                        am = AssistantMessage()
                        message_list.mount(am)
                        am._accumulated_text = content
                        if am._markdown_widget:
                            am._markdown_widget.update(content)
                message_list.scroll_to_latest()
        except Exception:
            logger.debug("Failed to load session messages", exc_info=True)

    def _update_context_bar(self) -> None:
        """Update the context info bar below the prompt."""
        app = self.app
        state = app.session_state
        try:
            bar = self.query_one("#context-bar", Static)
            model = f"{app.provider_name}/{state.model}"
            agent = state.agent_mode
            bar.update(f"  {model}  \u2502  {agent} agent")
        except Exception:
            pass

    # ── Input Handling ───────────────────────────────────────────────

    def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
        """Handle user prompt submission."""
        text = event.text.strip()
        if not text:
            return

        # Check for slash commands
        if text.startswith("/"):
            self._handle_slash_command(text)
            return

        # Check for shell mode
        if text.startswith("!"):
            self._handle_shell_command(text[1:])
            return

        self._handle_user_input(text)

    def _handle_user_input(self, text: str) -> None:
        """Process a user message and start generation."""
        app = self.app

        # Mount user message
        message_list = self.query_one("#message-list", MessageList)
        message_list.mount(UserMessage(text))
        message_list.scroll_to_latest()

        # Add to history
        app.session_history.append({"role": "user", "content": text})

        # Start generation
        self._generation_cancelled = False
        self._start_generation(text)

    def _handle_slash_command(self, command: str) -> None:
        """Handle slash commands."""
        parts = command.split(None, 1)
        cmd = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        if cmd in ("/quit", "/exit"):
            self.app.exit()
        elif cmd == "/new":
            self.app.action_new_session()
        elif cmd == "/clear":
            self.clear_messages()
        elif cmd == "/model":
            self._show_model_picker()
        elif cmd == "/sessions":
            self._show_session_list()
        elif cmd == "/theme":
            self._show_theme_picker()
        elif cmd == "/auto":
            self.app.session_state.autonomous = not self.app.session_state.autonomous
            mode = "ON" if self.app.session_state.autonomous else "OFF"
            self.app.notify(f"Autonomous mode: {mode}")
        elif cmd == "/help":
            self._show_help()
        elif cmd == "/copy":
            self._copy_last_response()
        elif cmd in ("/fix", "/test", "/docs", "/profile"):
            self._run_dev_command(cmd, rest)
        elif cmd == "/plan":
            if rest:
                self._handle_user_input(f"Create a plan for: {rest}")
            else:
                self.app.notify("Usage: /plan <description>")
        elif cmd == "/continue":
            self._handle_user_input("Continue with the next step of the plan.")
        elif cmd == "/summary":
            self._handle_user_input("Please summarize our conversation so far.")
        else:
            self.app.notify(f"Unknown command: {cmd}", severity="warning")

    def _handle_shell_command(self, command: str) -> None:
        """Run a shell command and display output."""
        import subprocess

        message_list = self.query_one("#message-list", MessageList)
        message_list.mount(UserMessage(f"!{command}"))

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.app.app_state.workspace_root,
            )
            output = result.stdout + result.stderr
            if output:
                from ..widgets.tool_display import BlockToolCall

                widget = BlockToolCall("run_command", {"command": command}, output)
                message_list.mount(widget)
        except subprocess.TimeoutExpired:
            self.app.notify("Command timed out (30s)", severity="error")
        except Exception as e:
            self.app.notify(f"Command failed: {e}", severity="error")

        message_list.scroll_to_latest()

    # ── Generation / Streaming ───────────────────────────────────────

    @work(exclusive=True, group="llm")
    async def _start_generation(self, prompt: str) -> None:
        """Run the agent loop and stream results."""
        app = self.app
        state = app.session_state

        # Show spinner
        try:
            spinner = self.query_one("#prompt-spinner", BrailleSpinner)
            spinner.label = "Thinking..."
            spinner.display = True
        except Exception:
            pass

        state.is_busy = True
        state.is_streaming = True

        # Create assistant message widget
        message_list = self.query_one("#message-list", MessageList)
        self._current_assistant_msg = AssistantMessage()
        self._current_assistant_msg.is_streaming = True
        await message_list.mount(self._current_assistant_msg)

        start_time = time.monotonic()

        try:
            # Import agent
            from ollamacode.agent import run_agent_loop_stream

            # Tool callbacks
            def on_tool_start(name: str, args: dict[str, Any]) -> None:
                self.post_message(ToolStarted(name, args))

            def on_tool_end(name: str, args: dict[str, Any], summary: str) -> None:
                self.post_message(ToolFinished(name, args, summary))

            async def before_tool_call(
                name: str, args: dict[str, Any]
            ) -> str | tuple[str, dict[str, Any]]:
                """Handle tool confirmation."""
                if app.autonomous_mode or app.session_state.autonomous:
                    return "run"
                if not app.confirm_tool_calls:
                    return "run"

                # Show confirmation dialog
                from ..dialogs.tool_confirm import ToolConfirmDialog

                result = await app.push_screen_wait(ToolConfirmDialog(name, args))
                if result == "allow":
                    return "run"
                elif result == "always":
                    return "run"
                else:
                    return "skip"

            # Build messages list for the agent
            messages = list(app.session_history)

            # Stream response
            accumulated = ""
            stream = run_agent_loop_stream(
                session=app.mcp_session,
                model=state.model,
                messages=messages,
                system_extra=app.system_extra,
                max_tool_rounds=app.max_tool_rounds,
                max_messages=app.max_messages,
                max_tool_result_chars=app.max_tool_result_chars,
                on_tool_start=on_tool_start,
                on_tool_end=on_tool_end,
                before_tool_call=before_tool_call,
                provider=app.provider,
            )

            async for chunk in stream:
                if self._generation_cancelled:
                    break

                if isinstance(chunk, str):
                    accumulated += chunk
                    state.token_count += 1  # Approximate
                    if self._current_assistant_msg:
                        await self._current_assistant_msg.append_text(chunk)
                        message_list.scroll_to_latest()

            # Finalize
            if self._current_assistant_msg:
                await self._current_assistant_msg.finalize()

            # Save to history
            if accumulated:
                app.session_history.append(
                    {"role": "assistant", "content": accumulated}
                )

            # Save session
            self._save_session()

            elapsed = time.monotonic() - start_time
            logger.debug("Generation completed in %.1fs", elapsed)

        except Exception as e:
            logger.error("Generation error: %s", e, exc_info=True)
            self.post_message(GenerationError(str(e)))

        finally:
            state.is_busy = False
            state.is_streaming = False
            self._current_assistant_msg = None

            # Hide spinner
            try:
                spinner = self.query_one("#prompt-spinner", BrailleSpinner)
                spinner.display = False
            except Exception:
                pass

            # Update sidebar
            self._update_sidebar_stats()

    # ── Message Handlers ─────────────────────────────────────────────

    def on_tool_started(self, event: ToolStarted) -> None:
        """Handle tool start — update spinner and sidebar."""
        try:
            spinner = self.query_one("#prompt-spinner", BrailleSpinner)
            icon = {
                "run_command": "$",
                "bash": "$",
                "read_file": "\u2192",
                "write_file": "\u2190",
                "edit_file": "\u270e",
            }.get(event.name, "\u2699")
            spinner.label = f"{icon} {event.name}..."
        except Exception:
            pass

        self.app.session_state.tool_calls += 1

    def on_tool_finished(self, event: ToolFinished) -> None:
        """Handle tool end — mount tool widget in assistant message."""
        if self._current_assistant_msg:
            widget = make_tool_widget(event.name, event.args, event.summary)
            self._current_assistant_msg.mount(widget)
            try:
                self.query_one("#message-list", MessageList).scroll_to_latest()
            except Exception:
                pass

        # Update spinner
        try:
            spinner = self.query_one("#prompt-spinner", BrailleSpinner)
            spinner.label = "Thinking..."
        except Exception:
            pass

    def on_generation_error(self, event: GenerationError) -> None:
        """Handle generation error — show notification."""
        self.app.notify(f"Error: {event.error}", severity="error", timeout=10)

    # ── Sidebar Updates ──────────────────────────────────────────────

    def _update_sidebar_stats(self) -> None:
        """Update sidebar with current session stats."""
        try:
            sidebar = self.query_one(Sidebar)
            state = self.app.session_state
            sidebar.token_count = state.token_count
            sidebar.cost = state.cost
        except Exception:
            pass

    # ── Session Persistence ──────────────────────────────────────────

    def _save_session(self) -> None:
        """Save the current session to database."""
        app = self.app
        state = app.session_state
        if not state.session_id:
            return
        try:
            from ollamacode.sessions import save_session

            save_session(
                state.session_id,
                state.title,
                app.session_history,
                app.app_state.workspace_root,
            )
        except Exception:
            logger.debug("Failed to save session", exc_info=True)

    # ── UI Actions ───────────────────────────────────────────────────

    def clear_messages(self) -> None:
        """Clear all messages from the message list."""
        try:
            message_list = self.query_one("#message-list", MessageList)
            message_list.remove_children()
        except Exception:
            pass
        self.app.session_history.clear()

    def _show_model_picker(self) -> None:
        from ..dialogs.model_picker import ModelPickerDialog

        def on_result(model: str) -> None:
            if model:
                self.app.model = model
                self.app.session_state.model = model
                self._update_context_bar()
                self.query_one(
                    SessionHeader
                ).model_name = f"{self.app.provider_name}/{model}"
                self.app.notify(f"Model: {model}")

        self.app.push_screen(ModelPickerDialog(current=self.app.model), on_result)

    def _show_session_list(self) -> None:
        from ..dialogs.session_list import SessionListDialog

        def on_result(session_id: str) -> None:
            if session_id:
                self.app.session_state.session_id = session_id
                self._resume_session_id = session_id
                self.clear_messages()
                self._load_session_messages()

        self.app.push_screen(SessionListDialog(), on_result)

    def _show_theme_picker(self) -> None:
        from ..dialogs.theme_picker import ThemePickerDialog

        def on_result(theme_name: str) -> None:
            if theme_name:
                from ..context.theme import generate_css, get_theme

                theme = get_theme(theme_name)
                self.app.stylesheet.add_source(generate_css(theme), "theme")
                self.app.notify(f"Theme: {theme_name}")

        self.app.push_screen(ThemePickerDialog(), on_result)

    def _show_help(self) -> None:
        """Show help text as a notification."""
        help_text = (
            "Commands: /new /clear /model /sessions /theme /auto /fix /test "
            "/plan /continue /summary /copy /help /quit\n"
            "Keys: Ctrl+N=New Ctrl+P=Palette Ctrl+\\=Sidebar Esc=Cancel"
        )
        self.app.notify(help_text, title="Help", timeout=15)

    def _copy_last_response(self) -> None:
        """Copy the last assistant response to clipboard."""
        for msg in reversed(self.app.session_history):
            if msg.get("role") == "assistant":
                import subprocess

                try:
                    subprocess.run(
                        ["pbcopy"],
                        input=msg["content"].encode(),
                        check=True,
                        timeout=5,
                    )
                    self.app.notify("Copied to clipboard")
                except Exception:
                    try:
                        subprocess.run(
                            ["xclip", "-selection", "clipboard"],
                            input=msg["content"].encode(),
                            check=True,
                            timeout=5,
                        )
                        self.app.notify("Copied to clipboard")
                    except Exception:
                        self.app.notify("Clipboard not available", severity="warning")
                return
        self.app.notify("No response to copy", severity="warning")

    def _run_dev_command(self, cmd: str, args: str) -> None:
        """Run a dev command (/fix, /test, /docs, /profile) and send output to model."""
        app = self.app
        cmd_map = {
            "/fix": app.linter_command,
            "/test": app.test_command,
            "/docs": app.docs_command,
            "/profile": app.profile_command,
        }
        command = cmd_map.get(cmd)
        if not command:
            self.app.notify(f"No command configured for {cmd}", severity="warning")
            return

        import subprocess

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=app.app_state.workspace_root,
            )
            output = result.stdout + result.stderr
            prompt = f"I ran `{command}` and got:\n```\n{output[:8000]}\n```\nPlease analyze and fix any issues."
            self._handle_user_input(prompt)
        except subprocess.TimeoutExpired:
            self.app.notify("Command timed out", severity="error")
        except Exception as e:
            self.app.notify(f"Command failed: {e}", severity="error")

    # ── Actions ──────────────────────────────────────────────────────

    def action_new_session(self) -> None:
        self.app.action_new_session()

    def action_toggle_sidebar(self) -> None:
        self.app.action_toggle_sidebar()

    def action_cancel_generation(self) -> None:
        self._generation_cancelled = True
        self.app.session_state.is_busy = False
        self.app.session_state.is_streaming = False
        try:
            spinner = self.query_one("#prompt-spinner", BrailleSpinner)
            spinner.display = False
        except Exception:
            pass
        self.app.notify("Generation cancelled")
