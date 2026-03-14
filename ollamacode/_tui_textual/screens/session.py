"""Session screen — main chat view with messages, sidebar, and streaming."""

from __future__ import annotations

import asyncio
import logging
import shlex
import time
from typing import Any

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Static

from ...question_runtime import format_question_answers, normalize_question_list
from ...task_runtime import run_task_delegation
from ...permission_runtime import evaluate_permission
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
        self.app.load_session_todos()
        self.app._refresh_sidebar()

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
        elif cmd == "/todo":
            self._handle_todo_command(rest)
        elif cmd == "/export":
            self._show_export_dialog()
        elif cmd == "/import":
            self._show_import_dialog()
        elif cmd == "/workspace":
            self._show_workspace_info()
        elif cmd == "/workspaces":
            self._show_workspaces()
        elif cmd == "/workspace_health":
            self._show_workspace_health(rest)
        elif cmd == "/workspace_add_remote":
            self._add_remote_workspace(rest)
        elif cmd == "/branch":
            self._branch_session()
        elif cmd == "/checkpoints":
            self._show_checkpoints()
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
                normalized_name = name.removeprefix("functions::")
                if name.endswith("question") or name == "question":
                    return await self._handle_question_tool(args)
                if name.endswith("task") or name == "task":
                    return await self._handle_task_tool(args)
                permission = evaluate_permission(
                    app.app_state.permissions_manager,
                    app.app_state.permission_state,
                    app.session_state.session_id,
                    [name, normalized_name],
                )
                if permission.value == "deny":
                    app.session_state.permissions_denied += 1
                    return ("skip", f"Blocked by permission rule for tool: {normalized_name}")
                if permission.value == "allow":
                    app.session_state.permissions_granted += 1
                    return "run"
                if app.autonomous_mode or app.session_state.autonomous:
                    return "run"
                if not app.confirm_tool_calls:
                    return "run"

                # Show confirmation dialog
                from ..dialogs.tool_confirm import ToolConfirmDialog

                result = await app.push_screen_wait(ToolConfirmDialog(name, args))
                if result == "allow":
                    app.session_state.permissions_granted += 1
                    return "run"
                elif result == "always":
                    if app.app_state.permission_state is not None:
                        app.app_state.permission_state.allow(
                            app.session_state.session_id,
                            [name, normalized_name],
                        )
                    app.session_state.permissions_granted += 1
                    return "run"
                else:
                    app.session_state.permissions_denied += 1
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
                confirm_tool_calls=app.mcp_session is not None,
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
            self.app._refresh_sidebar()
        except Exception:
            pass

    def _handle_todo_command(self, rest: str) -> None:
        """Manage the per-session TODO list."""
        todos = list(self.app.app_state.todos)
        try:
            parts = shlex.split(rest)
        except ValueError as exc:
            self.app.notify(f"Invalid /todo arguments: {exc}", severity="warning")
            return

        if not parts or parts[0].lower() == "list":
            self._show_todo_summary(todos)
            return

        action = parts[0].lower()
        if action == "add":
            content = rest.partition(" ")[2].strip()
            if not content:
                self.app.notify("Usage: /todo add <task>", severity="warning")
                return
            todos.append(
                {"content": content, "status": "pending", "priority": "medium"}
            )
            self.app.set_session_todos(todos)
            self.app.notify(f"Added todo #{len(todos)}")
            return

        if action == "clear":
            self.app.set_session_todos([])
            self.app.notify("Cleared todos")
            return

        if len(parts) < 2:
            self.app.notify(
                "Usage: /todo <list|add|start|done|pending|remove|clear> ...",
                severity="warning",
            )
            return

        try:
            index = int(parts[1]) - 1
        except ValueError:
            self.app.notify("Todo index must be a number", severity="warning")
            return

        if index < 0 or index >= len(todos):
            self.app.notify("Todo index out of range", severity="warning")
            return

        if action == "start":
            todos[index]["status"] = "in_progress"
            self.app.set_session_todos(todos)
            self.app.notify(f"Todo #{index + 1} in progress")
        elif action == "done":
            todos[index]["status"] = "completed"
            self.app.set_session_todos(todos)
            self.app.notify(f"Completed todo #{index + 1}")
        elif action == "pending":
            todos[index]["status"] = "pending"
            self.app.set_session_todos(todos)
            self.app.notify(f"Reset todo #{index + 1}")
        elif action == "remove":
            removed = todos.pop(index)
            self.app.set_session_todos(todos)
            self.app.notify(f"Removed todo: {removed.get('content', '')[:40]}")
        else:
            self.app.notify(f"Unknown /todo action: {action}", severity="warning")

    def _show_todo_summary(self, todos: list[dict[str, Any]]) -> None:
        """Show a compact TODO summary notification."""
        if not todos:
            self.app.notify("No todos for this session")
            return
        lines = []
        for index, todo in enumerate(todos[:8], start=1):
            status = str(todo.get("status") or "pending").lower()
            icon = {
                "pending": "\u2610",
                "in_progress": "\u25d0",
                "completed": "\u2611",
                "cancelled": "\u2298",
            }.get(status, "\u2610")
            lines.append(f"{index}. {icon} {todo.get('content', '')}")
        self.app.notify("\n".join(lines), title="Session TODOs", timeout=10)

    async def _handle_question_tool(self, arguments: dict[str, Any]) -> tuple[str, str]:
        """Collect answers for the interactive question tool."""
        questions = normalize_question_list(arguments)
        if not questions:
            return ("skip", "Question tool called without valid questions.")
        from ..dialogs.question import QuestionDialog

        answers: list[str] = []
        for item in questions:
            answer = await self.app.push_screen_wait(
                QuestionDialog(item["question"], item.get("options") or [])
            )
            answers.append(str(answer or ""))
        return ("skip", format_question_answers(questions, answers))

    async def _handle_task_tool(self, arguments: dict[str, Any]) -> tuple[str, str]:
        """Delegate work to a configured subagent."""
        result = await run_task_delegation(
            session=self.app.mcp_session,
            session_id=self.app.session_state.session_id,
            workspace_root=self.app.app_state.workspace_root,
            subagents=self.app._config.get("subagents") or [],
            arguments=arguments,
            default_model=self.app.session_state.model or self.app.model,
            system_prompt=self.app.system_extra,
            max_messages=self.app.max_messages,
            max_tool_rounds=self.app.max_tool_rounds,
            max_tool_result_chars=self.app.max_tool_result_chars,
            provider=self.app.provider,
            before_tool_call=self._task_before_tool_call,
        )
        return ("skip", result)

    async def _task_before_tool_call(
        self, name: str, args: dict[str, Any]
    ) -> str | tuple[str, dict[str, Any]] | tuple[str, str]:
        """Nested tool interception used by delegated subagent tasks."""
        if name.endswith("question") or name == "question":
            return await self._handle_question_tool(args)
        if name.endswith("task") or name == "task":
            return ("skip", "Nested task delegation is disabled for subagents.")
        if self.app.autonomous_mode or self.app.session_state.autonomous:
            return "run"
        if not self.app.confirm_tool_calls:
            return "run"
        from ..dialogs.tool_confirm import ToolConfirmDialog

        result = await self.app.push_screen_wait(ToolConfirmDialog(name, args))
        if result == "allow":
            return "run"
        if result == "always":
            return "run"
        return "skip"

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
                self.app.load_session_todos(session_id)
                self.app._refresh_sidebar()

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

    def _show_export_dialog(self) -> None:
        from ..dialogs.export_import import ExportDialog

        session_id = self.app.session_state.session_id
        if not session_id:
            self.app.notify("No active session to export", severity="warning")
            return
        try:
            from ollamacode.sessions import export_session

            data = export_session(session_id)
        except Exception:
            logger.debug("Failed to export session", exc_info=True)
            data = None
        if not data:
            self.app.notify("Failed to export session", severity="error")
            return
        self.app.push_screen(ExportDialog(data))

    def _show_import_dialog(self) -> None:
        from ..dialogs.export_import import ImportDialog

        def on_result(json_text: str) -> None:
            if not json_text.strip():
                return
            try:
                from ollamacode.sessions import import_session

                session_id = import_session(json_text)
            except Exception as exc:
                self.app.notify(f"Import failed: {exc}", severity="error")
                return
            self.app.session_state.session_id = session_id
            self._resume_session_id = session_id
            self.clear_messages()
            self._load_session_messages()
            self.app.load_session_todos(session_id)
            self.app._refresh_sidebar()
            self.app.notify("Session imported")

        self.app.push_screen(ImportDialog(), on_result)

    def _show_workspace_info(self) -> None:
        try:
            from ollamacode.sessions import list_sessions

            count = len(list_sessions(limit=1000, workspace_root=self.app.app_state.workspace_root))
        except Exception:
            count = 0
        self.app.notify(
            f"Workspace: {self.app.app_state.workspace_root}\nSessions: {count}",
            title="Workspace",
            timeout=10,
        )

    def _show_workspaces(self) -> None:
        try:
            from ollamacode.workspaces import list_workspaces

            rows = list_workspaces()
        except Exception:
            rows = []
        if not rows:
            self.app.notify("No registered workspaces", title="Workspaces", timeout=8)
            return
        text = "\n".join(
            f"{row.get('name', 'Workspace')} [{row.get('type', 'local')}]"
            for row in rows[:10]
        )
        self.app.notify(text, title="Workspaces", timeout=10)

    def _show_workspace_health(self, rest: str) -> None:
        workspace_id = rest.strip()
        if not workspace_id:
            self.app.notify("Usage: /workspace_health <workspace-id>", severity="warning")
            return
        try:
            from ollamacode.workspaces import get_workspace

            row = get_workspace(workspace_id)
        except Exception:
            row = None
        if row is None:
            self.app.notify("Workspace not found", severity="warning")
            return
        status = row.get("last_status") or "unknown"
        error = row.get("last_error") or ""
        body = f"{row.get('name', 'Workspace')}: {status}"
        if error:
            body += f"\n{error}"
        self.app.notify(body, title="Workspace Health", timeout=10)

    def _add_remote_workspace(self, rest: str) -> None:
        parts = rest.split(None, 1)
        if len(parts) < 2:
            self.app.notify(
                "Usage: /workspace_add_remote <name> <base_url>",
                severity="warning",
            )
            return
        name, base_url = parts[0].strip(), parts[1].strip()
        try:
            from ollamacode.workspaces import create_workspace

            row = create_workspace(name=name, kind="remote", base_url=base_url)
        except Exception as exc:
            self.app.notify(f"Failed to create workspace: {exc}", severity="error")
            return
        self.app.notify(
            f"Added remote workspace {row['name']}",
            title="Workspaces",
            timeout=8,
        )

    def _branch_session(self) -> None:
        session_id = self.app.session_state.session_id
        if not session_id:
            self.app.notify("No active session to branch", severity="warning")
            return
        try:
            from ollamacode.sessions import branch_session

            new_id = branch_session(session_id)
        except Exception as exc:
            self.app.notify(f"Branch failed: {exc}", severity="error")
            return
        if not new_id:
            self.app.notify("Branch failed", severity="error")
            return
        self.app.session_state.session_id = new_id
        self._resume_session_id = new_id
        self.clear_messages()
        self._load_session_messages()
        self.app.load_session_todos(new_id)
        self.app._refresh_sidebar()
        self.app.notify("Session branched")

    def _show_checkpoints(self) -> None:
        session_id = self.app.session_state.session_id
        if not session_id:
            self.app.notify("No active session", severity="warning")
            return
        from ..dialogs.checkpoint_list import CheckpointListDialog

        self.app.push_screen(CheckpointListDialog(session_id))

    def _show_help(self) -> None:
        """Show help text as a notification."""
        help_text = (
            "Commands: /new /clear /model /sessions /theme /auto /fix /test "
            "/plan /continue /summary /todo /copy /help /quit\n"
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
