"""Session screen — main chat view with messages, sidebar, and streaming."""

from __future__ import annotations

import asyncio
import logging
import os
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
        footer.variant_name = state.variant_name
        footer.sandbox_level = os.environ.get("OLLAMACODE_SANDBOX_LEVEL", "")

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

    @work(group="load_session")
    async def _load_session_messages(self) -> None:
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
                        await message_list.mount(UserMessage(content))
                    elif role == "assistant":
                        am = AssistantMessage()
                        await message_list.mount(am)
                        am._accumulated_text = content
                        if am._markdown_widget:
                            await am._markdown_widget.update(content)
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

    def _handle_user_input(self, text: str, *, image_paths: list[str] | None = None) -> None:
        """Process a user message and start generation."""
        app = self.app

        # Expand @file references
        try:
            prompt_widget = self.query_one("#session-prompt", PromptInput)
            text = prompt_widget.expand_at_refs(text)
        except Exception:
            logger.debug("@-ref expansion failed", exc_info=True)

        # Mount user message
        message_list = self.query_one("#message-list", MessageList)
        message_list.mount(UserMessage(text))
        message_list.scroll_to_latest()

        # Add to history
        app.session_history.append({"role": "user", "content": text})

        # Start generation
        self._generation_cancelled = False
        self._start_generation(text, image_paths=image_paths)

    def _handle_slash_command(self, command: str) -> None:
        """Handle slash commands."""
        parts = command.split(None, 1)
        cmd = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        # Core
        if cmd in ("/quit", "/exit"):
            self.app.exit()
        elif cmd == "/new":
            self.app.action_new_session()
        elif cmd == "/clear":
            self.clear_messages()
        elif cmd == "/help":
            self._show_help()

        # Model & display
        elif cmd == "/model":
            self._show_model_picker()
        elif cmd == "/theme":
            self._show_theme_picker()
        elif cmd == "/auto":
            self.app.session_state.autonomous = not self.app.session_state.autonomous
            mode = "ON" if self.app.session_state.autonomous else "OFF"
            self.app.notify(f"Autonomous mode: {mode}")
        elif cmd == "/compact":
            self._handle_compact(rest)
        elif cmd == "/trace":
            self.app.session_state.trace_filter = rest
            self.app.notify(f"Trace filter: {rest!r}" if rest else "Trace filter cleared")
        elif cmd == "/reset-state":
            self._reset_state()

        # Sessions
        elif cmd == "/sessions":
            self._show_session_list()
        elif cmd == "/search":
            self._search_sessions(rest)
        elif cmd == "/resume":
            self._resume_session(rest)
        elif cmd == "/session":
            self._show_session_info(rest)
        elif cmd == "/branch":
            self._branch_session()
        elif cmd == "/export":
            self._show_export_dialog()
        elif cmd == "/import":
            self._show_import_dialog()

        # Checkpoints
        elif cmd == "/checkpoints":
            self._show_checkpoints()
        elif cmd == "/rewind":
            self._rewind_checkpoint(rest)

        # Context & memory
        elif cmd == "/kg_add":
            self._kg_add(rest)
        elif cmd == "/kg_query":
            self._kg_query(rest)
        elif cmd == "/rag_index":
            self._rag_index(rest)
        elif cmd == "/rag_query":
            self._rag_query(rest)

        # Dev commands
        elif cmd in ("/fix", "/test", "/docs", "/profile"):
            self._run_dev_command(cmd, rest)

        # Agent
        elif cmd == "/plan":
            if rest:
                self._handle_user_input(f"Create a plan for: {rest}")
            else:
                self.app.notify("Usage: /plan <description>")
        elif cmd == "/continue":
            self._handle_user_input("Continue with the next step of the plan.")
        elif cmd == "/summary":
            self._handle_user_input("Please summarize our conversation so far.")
        elif cmd == "/copy":
            self._copy_last_response()
        elif cmd == "/mode":
            self._switch_mode(rest)
        elif cmd == "/variant":
            self._switch_variant(rest)
        elif cmd == "/commands":
            self._list_commands()

        # Multi-agent
        elif cmd == "/multi":
            self._handle_user_input(f"[multi-agent] {rest}" if rest else "What multi-agent task should I run?")
        elif cmd == "/agents":
            self._handle_user_input(f"[agents] {rest}" if rest else "Usage: /agents <N> <task>")
        elif cmd == "/agents_show":
            self.app.notify("No agent outputs to display", severity="warning")
        elif cmd == "/agents_summary":
            self.app.notify("No agent outputs to summarize", severity="warning")
        elif cmd == "/subagent":
            self._handle_user_input(f"[subagent] {rest}" if rest else "Usage: /subagent <type> <task>")

        # Media
        elif cmd == "/image":
            self._handle_image(rest)
        elif cmd == "/listen":
            self.app.notify("Voice input not available in this environment", severity="warning")
        elif cmd == "/say":
            self.app.notify("TTS not available in this environment", severity="warning")

        # Feedback & tools
        elif cmd == "/rate":
            self._rate_response(rest)
        elif cmd == "/refactor":
            self._show_refactor_dialog()
        elif cmd == "/palette":
            self.app.action_command_palette()

        # Workspace
        elif cmd == "/workspace":
            self._show_workspace_info()
        elif cmd == "/workspaces":
            self._show_workspaces()
        elif cmd == "/workspace_health":
            self._show_workspace_health(rest)
        elif cmd == "/workspace_add_remote":
            self._add_remote_workspace(rest)

        # Todo
        elif cmd == "/todo":
            self._handle_todo_command(rest)

        else:
            # Check custom commands
            cm = self.app.app_state.command_manager
            if cm is not None:
                cmd_name = cmd.lstrip("/")
                result = cm.execute_command(cmd_name, rest)
                if result is not None:
                    self._handle_user_input(result)
                    return
            self.app.notify(f"Unknown command: {cmd}", severity="warning")

    @work(group="shell")
    async def _handle_shell_command(self, command: str) -> None:
        """Run a shell command in a background worker and display output."""
        message_list = self.query_one("#message-list", MessageList)
        await message_list.mount(UserMessage(f"!{command}"))

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.app.app_state.workspace_root,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            except asyncio.TimeoutError:
                proc.kill()
                self.app.notify("Command timed out (30s)", severity="error")
                return
            output = (stdout or b"").decode(errors="replace") + (stderr or b"").decode(errors="replace")
            if output:
                from ..widgets.tool_display import BlockToolCall

                widget = BlockToolCall("run_command", {"command": command}, output)
                await message_list.mount(widget)
        except Exception as e:
            self.app.notify(f"Command failed: {e}", severity="error")

        message_list.scroll_to_latest()

    # ── Generation / Streaming ───────────────────────────────────────

    @work(exclusive=True, group="llm")
    async def _start_generation(
        self, prompt: str, *, image_paths: list[str] | None = None
    ) -> None:
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

        # Emit plugin event
        pm = app.app_state.plugin_manager
        if pm is not None:
            try:
                pm.emit_event("message_sent", prompt=prompt)
            except Exception:
                pass

        # Build system prompt with mode prefix and dynamic memory
        system_extra = app.system_extra or ""
        mm = app.app_state.mode_manager
        if mm is not None:
            try:
                prefix = mm.get_system_prompt_prefix()
                if prefix:
                    system_extra = prefix + "\n\n" + system_extra
            except Exception:
                pass

        try:
            from ollamacode.memory import build_dynamic_memory_context

            mem_block = build_dynamic_memory_context(prompt)
            if mem_block:
                system_extra = (
                    system_extra + "\n\n--- Retrieved memory ---\n\n" + mem_block
                )
        except Exception:
            logger.debug("Dynamic memory context failed", exc_info=True)

        # Collect blocked tools from mode manager
        blocked_tools: list[str] | None = None
        if mm is not None:
            try:
                bt = mm.get_blocked_tools()
                if bt:
                    blocked_tools = bt
            except Exception:
                pass

        # Checkpoint recorder
        checkpoint_recorder = None
        try:
            from ollamacode.checkpoints import CheckpointRecorder

            if state.session_id:
                checkpoint_recorder = CheckpointRecorder(
                    session_id=state.session_id,
                    workspace_root=app.app_state.workspace_root,
                    prompt=prompt,
                    message_index=len(app.session_history) - 1,
                )
        except Exception:
            logger.debug("CheckpointRecorder init failed", exc_info=True)

        try:
            # Import agent — use no-MCP variant when session is None
            from ollamacode.agent import (
                run_agent_loop_no_mcp_stream,
                run_agent_loop_stream,
            )

            has_mcp = app.mcp_session is not None

            # Tool callbacks
            def on_tool_start(name: str, args: dict[str, Any]) -> None:
                self.post_message(ToolStarted(name, args))
                if pm is not None:
                    try:
                        pm.emit_event("tool_start", name=name, args=args)
                    except Exception:
                        pass

            def on_tool_end(name: str, args: dict[str, Any], summary: str) -> None:
                self.post_message(ToolFinished(name, args, summary))
                if pm is not None:
                    try:
                        pm.emit_event("tool_end", name=name, args=args, summary=summary)
                    except Exception:
                        pass

            async def before_tool_call(
                name: str, args: dict[str, Any]
            ) -> str | tuple[str, str] | tuple[str, dict[str, Any]]:
                """Handle tool confirmation."""
                normalized_name = name.removeprefix("functions::")

                # Record pre-change state for checkpointing
                if checkpoint_recorder is not None:
                    if normalized_name in (
                        "write_file",
                        "edit_file",
                        "create_directory",
                        "run_command",
                        "bash",
                    ):
                        path = args.get("path") or args.get("file_path", "")
                        if path:
                            try:
                                checkpoint_recorder.record_pre(path)
                            except Exception:
                                pass

                # Check mode manager tool restrictions
                if mm is not None:
                    try:
                        if not mm.is_tool_allowed(normalized_name):
                            return (
                                "skip",
                                f"Tool '{normalized_name}' blocked in {mm.current.value} mode",
                            )
                    except Exception:
                        pass

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
                    return (
                        "skip",
                        f"Blocked by permission rule for tool: {normalized_name}",
                    )
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

            # Build messages list for the agent — exclude the last user
            # message because run_agent_loop_stream appends `prompt` itself.
            messages = list(app.session_history[:-1]) if app.session_history else []

            # Stream response
            accumulated = ""
            # Merge mode-blocked tools with app-level blocked tools
            all_blocked = list(blocked_tools or [])
            if app.blocked_tools:
                all_blocked.extend(app.blocked_tools)

            if has_mcp:
                stream = run_agent_loop_stream(
                    app.mcp_session,
                    state.model,
                    prompt,
                    system_prompt=system_extra or None,
                    max_tool_rounds=app.max_tool_rounds,
                    max_messages=app.max_messages,
                    max_tool_result_chars=app.max_tool_result_chars,
                    message_history=messages,
                    on_tool_start=on_tool_start,
                    on_tool_end=on_tool_end,
                    before_tool_call=before_tool_call,
                    confirm_tool_calls=app.confirm_tool_calls,
                    allowed_tools=app.allowed_tools,
                    blocked_tools=all_blocked or None,
                    image_paths=image_paths,
                    provider=app.provider,
                )
            else:
                stream = run_agent_loop_no_mcp_stream(
                    state.model,
                    prompt,
                    system_prompt=system_extra or None,
                    message_history=messages,
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

            # Save to history
            if accumulated:
                app.session_history.append(
                    {"role": "assistant", "content": accumulated}
                )

            # Finalize checkpoint
            if checkpoint_recorder is not None:
                try:
                    cp_id = checkpoint_recorder.finalize()
                    if cp_id:
                        state.checkpoint_count += 1
                except Exception:
                    logger.debug("Checkpoint finalize failed", exc_info=True)

            # Save session
            self._save_session()

            # Emit plugin event
            if pm is not None:
                try:
                    pm.emit_event("message_received", content=accumulated)
                except Exception:
                    pass

            elapsed = time.monotonic() - start_time
            logger.debug("Generation completed in %.1fs", elapsed)

        except Exception as e:
            logger.error("Generation error: %s", e, exc_info=True)
            self.post_message(GenerationError(str(e)))
            if pm is not None:
                try:
                    pm.emit_event("error", error=str(e))
                except Exception:
                    pass

        finally:
            state.is_busy = False
            state.is_streaming = False

            # Always finalize the assistant message to exit streaming state
            if self._current_assistant_msg:
                try:
                    await self._current_assistant_msg.finalize()
                except Exception:
                    pass
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

    async def on_tool_finished(self, event: ToolFinished) -> None:
        """Handle tool end — mount tool widget in assistant message."""
        if self._current_assistant_msg:
            widget = make_tool_widget(event.name, event.args, event.summary)
            await self._current_assistant_msg.add_tool_call(widget)
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

    def clear_messages(self, *, persist: bool = True) -> None:
        """Clear all messages from the message list.

        Args:
            persist: If True, save the cleared state to the database.
                     Set to False when switching sessions to avoid
                     overwriting the old session with empty history.
        """
        try:
            message_list = self.query_one("#message-list", MessageList)
            message_list.remove_children()
        except Exception:
            pass
        self.app.session_history.clear()
        if persist:
            self._save_session()

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
                self.clear_messages(persist=False)
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
            self.clear_messages(persist=False)
            self._load_session_messages()
            self.app.load_session_todos(session_id)
            self.app._refresh_sidebar()
            self.app.notify("Session imported")

        self.app.push_screen(ImportDialog(), on_result)

    def _show_workspace_info(self) -> None:
        try:
            from ollamacode.sessions import list_sessions

            count = len(
                list_sessions(
                    limit=1000, workspace_root=self.app.app_state.workspace_root
                )
            )
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
            self.app.notify(
                "Usage: /workspace_health <workspace-id>", severity="warning"
            )
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
        self.clear_messages(persist=False)
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

    # ── New Slash Command Implementations ────────────────────────────

    def _handle_compact(self, rest: str) -> None:
        """Handle /compact command — toggle or run compaction."""
        if rest.strip().lower() in ("on", "off", "auto"):
            self.app.session_state.compact_mode = rest.strip().lower()
            self.app.notify(f"Compact mode: {rest.strip().lower()}")
            return
        if self.app.session_state.is_busy:
            self.app.notify("Cannot compact during generation", severity="warning")
            return
        self._run_compaction()

    @work(group="compact")
    async def _run_compaction(self) -> None:
        """Run message compaction in background worker."""
        try:
            from ollamacode.compaction import compact_messages

            messages = list(self.app.session_history)
            before_count = len(messages)
            result = await compact_messages(
                messages,
                self.app.session_state.model,
                self.app.provider,
            )
            after_count = len(result)
            self.app.session_history = result
            self.app.notify(
                f"Compacted: {before_count} -> {after_count} messages"
            )
        except Exception as e:
            self.app.notify(f"Compaction failed: {e}", severity="error")

    def _reset_state(self) -> None:
        """Clear persistent state."""
        try:
            from ollamacode.state import clear_state

            clear_state()
            self.app.notify("Persistent state cleared")
        except Exception as e:
            self.app.notify(f"Failed to clear state: {e}", severity="error")

    def _search_sessions(self, query: str) -> None:
        """Search sessions by query."""
        if not query.strip():
            self.app.notify("Usage: /search <query>", severity="warning")
            return
        try:
            from ollamacode.sessions import search_sessions

            results = search_sessions(query.strip())
            if not results:
                self.app.notify("No sessions found", severity="warning")
                return
            lines = []
            for s in results[:10]:
                sid = s.get("id", "")[:8]
                title = s.get("title", "Untitled")[:40]
                lines.append(f"  {sid}  {title}")
            self.app.notify(
                "\n".join(lines) + "\n\nUse /resume <id> to load",
                title=f"Search: {query}",
                timeout=15,
            )
        except Exception as e:
            self.app.notify(f"Search failed: {e}", severity="error")

    def _resume_session(self, session_id: str) -> None:
        """Resume a session by ID prefix."""
        sid = session_id.strip()
        if not sid:
            self.app.notify("Usage: /resume <session-id>", severity="warning")
            return
        try:
            from ollamacode.sessions import list_sessions

            sessions = list_sessions(limit=200)
            match = None
            for s in sessions:
                if s.get("id", "").startswith(sid):
                    match = s.get("id")
                    break
            if not match:
                self.app.notify(f"No session matching '{sid}'", severity="warning")
                return
            self.app.session_state.session_id = match
            self._resume_session_id = match
            self.clear_messages(persist=False)
            self._load_session_messages()
            self.app.load_session_todos(match)
            self.app._refresh_sidebar()
            self.app.notify(f"Resumed session {match[:8]}...")
        except Exception as e:
            self.app.notify(f"Resume failed: {e}", severity="error")

    def _show_session_info(self, rest: str) -> None:
        """Show or set session info."""
        state = self.app.session_state
        if rest.strip():
            state.title = rest.strip()
            self.app._refresh_sidebar()
            self.app.notify(f"Session title: {state.title}")
            return
        try:
            from ollamacode.sessions import get_session_info

            info = get_session_info(state.session_id) if state.session_id else None
            if info:
                self.app.notify(
                    f"ID: {info.get('id', 'N/A')}\n"
                    f"Title: {info.get('title', 'Untitled')}\n"
                    f"Messages: {info.get('message_count', 0)}\n"
                    f"Created: {info.get('created_at', 'N/A')}\n"
                    f"Updated: {info.get('updated_at', 'N/A')}",
                    title="Session Info",
                    timeout=10,
                )
            else:
                self.app.notify("No active session", severity="warning")
        except Exception as e:
            self.app.notify(f"Session info failed: {e}", severity="error")

    def _rewind_checkpoint(self, checkpoint_id: str) -> None:
        """Restore a checkpoint."""
        cid = checkpoint_id.strip()
        if not cid:
            self._show_checkpoints()
            return
        try:
            from ollamacode.checkpoints import restore_checkpoint

            restored = restore_checkpoint(cid, self.app.app_state.workspace_root)
            self.app.notify(
                f"Restored {len(restored)} file(s) from checkpoint {cid[:8]}..."
            )
        except Exception as e:
            self.app.notify(f"Rewind failed: {e}", severity="error")

    def _kg_add(self, rest: str) -> None:
        """Add knowledge graph entry: topic|summary."""
        if "|" not in rest:
            self.app.notify("Usage: /kg_add <topic>|<summary>", severity="warning")
            return
        topic, summary = rest.split("|", 1)
        try:
            from ollamacode.state import add_knowledge_node

            add_knowledge_node(topic.strip(), summary.strip())
            self.app.notify(f"Added KG entry: {topic.strip()}")
        except Exception as e:
            self.app.notify(f"KG add failed: {e}", severity="error")

    def _kg_query(self, query: str) -> None:
        """Query knowledge graph."""
        if not query.strip():
            self.app.notify("Usage: /kg_query <query>", severity="warning")
            return
        try:
            from ollamacode.state import query_knowledge_graph

            results = query_knowledge_graph(query.strip())
            if not results:
                self.app.notify("No KG results found")
                return
            lines = []
            for r in results[:5]:
                lines.append(f"- {r.get('topic', '')}: {r.get('summary', '')[:80]}")
            self.app.notify("\n".join(lines), title="KG Results", timeout=10)
        except Exception as e:
            self.app.notify(f"KG query failed: {e}", severity="error")

    def _rag_index(self, path: str) -> None:
        """Build vector index."""
        workspace = path.strip() or self.app.app_state.workspace_root
        self.app.notify(f"Indexing {workspace}...")
        self._run_rag_index(workspace)

    @work(group="rag_index")
    async def _run_rag_index(self, workspace: str) -> None:
        """Run RAG indexing in background worker."""
        try:
            from ollamacode.vector_memory import build_vector_index

            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: build_vector_index(workspace))
            self.app.notify(
                f"Indexed {result.get('indexed_files', 0)} files, "
                f"{result.get('chunk_count', 0)} chunks"
            )
        except Exception as e:
            self.app.notify(f"Indexing failed: {e}", severity="error")

    def _rag_query(self, query: str) -> None:
        """Query vector memory."""
        if not query.strip():
            self.app.notify("Usage: /rag_query <query>", severity="warning")
            return
        try:
            from ollamacode.vector_memory import query_vector_memory

            results = query_vector_memory(
                query.strip(), self.app.app_state.workspace_root
            )
            if not results:
                self.app.notify("No RAG results found")
                return
            lines = []
            for r in results[:5]:
                path = r.get("path", "")
                score = r.get("score", 0.0)
                snippet = r.get("snippet", "")[:60]
                lines.append(f"[{score:.2f}] {path}: {snippet}")
            self.app.notify("\n".join(lines), title="RAG Results", timeout=10)
        except Exception as e:
            self.app.notify(f"RAG query failed: {e}", severity="error")

    def _switch_mode(self, mode: str) -> None:
        """Switch agent mode (build/plan/review)."""
        mm = self.app.app_state.mode_manager
        if mm is None:
            self.app.notify("Mode manager not available", severity="warning")
            return
        if not mode.strip():
            modes = mm.list_modes()
            current = mm.current.value
            lines = []
            for m in modes:
                marker = " *" if m["name"] == current else ""
                lines.append(f"  {m['name']}{marker} — {m.get('description', '')}")
            self.app.notify("\n".join(lines), title="Agent Modes", timeout=10)
            return
        try:
            new_mode = mm.switch(mode.strip())
            self.app.session_state.agent_mode = new_mode.value
            self._update_context_bar()
            self.app._refresh_sidebar()
            try:
                self.query_one(SessionFooter).agent_mode = new_mode.value
            except Exception:
                pass
            self.app.notify(f"Mode: {new_mode.value}")
        except Exception as e:
            self.app.notify(f"Mode switch failed: {e}", severity="error")

    def _switch_variant(self, name: str) -> None:
        """Switch model variant."""
        vm = self.app.app_state.variant_manager
        if vm is None:
            self.app.notify("Variant manager not available", severity="warning")
            return
        if not name.strip():
            variants = vm.list_variants()
            current = vm.current
            current_name = current.name if current else "default"
            lines = []
            for v in variants:
                marker = " *" if v["name"] == current_name else ""
                lines.append(f"  {v['name']}{marker}")
            if not lines:
                lines.append("  No variants configured")
            self.app.notify("\n".join(lines), title="Variants", timeout=10)
            return
        variant = vm.select(name.strip())
        if variant is None:
            self.app.notify(f"Unknown variant: {name}", severity="warning")
            return
        self.app.session_state.variant_name = variant.name
        try:
            self.query_one(SessionFooter).variant_name = variant.name
        except Exception:
            pass
        self.app.notify(f"Variant: {variant.name}")

    def _list_commands(self) -> None:
        """List all available commands."""
        lines = [
            "Core: /new /clear /help /quit",
            "Model: /model /theme /auto /compact /trace",
            "Session: /sessions /search /resume /session /branch /export /import",
            "Checkpoint: /checkpoints /rewind",
            "Memory: /kg_add /kg_query /rag_index /rag_query",
            "Dev: /fix /test /docs /profile",
            "Agent: /plan /continue /summary /copy /mode /variant",
            "Media: /image /listen /say",
            "Tools: /refactor /palette /commands /todo",
        ]
        cm = self.app.app_state.command_manager
        if cm is not None:
            custom = cm.list_commands()
            if custom:
                names = ", ".join(f"/{c['name']}" for c in custom)
                lines.append(f"Custom: {names}")
        self.app.notify("\n".join(lines), title="Commands", timeout=15)

    def _handle_image(self, rest: str) -> None:
        """Attach image to message: /image <path> [message]."""
        if not rest.strip():
            self.app.notify("Usage: /image <path> [message]", severity="warning")
            return
        parts = rest.strip().split(None, 1)
        image_path = parts[0]
        message = parts[1] if len(parts) > 1 else "Please analyze this image."
        if not os.path.isfile(image_path):
            self.app.notify(f"File not found: {image_path}", severity="error")
            return
        self._handle_user_input(message, image_paths=[image_path])

    def _rate_response(self, rating: str) -> None:
        """Record feedback on last response."""
        r = rating.strip().lower()
        if r not in ("good", "bad"):
            self.app.notify("Usage: /rate good|bad", severity="warning")
            return
        logger.info("User rated response: %s (session=%s)", r, self.app.session_state.session_id)
        self.app.notify(f"Feedback recorded: {r}")

    def _show_refactor_dialog(self) -> None:
        """Show refactoring options dialog."""
        from ..dialogs.refactor import RefactorDialog

        def on_result(op: str) -> None:
            if not op:
                return
            self._handle_user_input(
                f"Please perform a '{op}' refactoring operation on the codebase."
            )

        self.app.push_screen(RefactorDialog(), on_result)

    # ── Help ──────────────────────────────────────────────────────────

    def _show_help(self) -> None:
        """Show help text as a notification."""
        help_text = (
            "Core: /new /clear /help /quit\n"
            "Model: /model /theme /auto /compact /mode /variant\n"
            "Session: /sessions /search /resume /branch /export /import\n"
            "Dev: /fix /test /docs /plan /continue /summary /copy\n"
            "Memory: /kg_add /kg_query /rag_index /rag_query\n"
            "Tools: /checkpoints /rewind /refactor /image /todo\n"
            "Ctrl+N=New Ctrl+P=Palette Ctrl+\\=Sidebar Esc=Cancel"
        )
        self.app.notify(help_text, title="Help", timeout=15)

    @work(group="clipboard")
    async def _copy_last_response(self) -> None:
        """Copy the last assistant response to clipboard without blocking UI."""
        content = ""
        for msg in reversed(self.app.session_history):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                break
        if not content:
            self.app.notify("No response to copy", severity="warning")
            return
        for cmd in (["pbcopy"], ["xclip", "-selection", "clipboard"]):
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(proc.communicate(input=content.encode()), timeout=5)
                if proc.returncode == 0:
                    self.app.notify("Copied to clipboard")
                    return
            except Exception:
                continue
        self.app.notify("Clipboard not available", severity="warning")

    @work(group="dev_cmd")
    async def _run_dev_command(self, cmd: str, args: str) -> None:
        """Run a dev command (/fix, /test, /docs, /profile) without blocking the UI."""
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

        self.app.notify(f"Running: {command}...")
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=app.app_state.workspace_root,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            except asyncio.TimeoutError:
                proc.kill()
                self.app.notify("Command timed out (120s)", severity="error")
                return
            output = (stdout or b"").decode(errors="replace") + (stderr or b"").decode(errors="replace")
            prompt = f"I ran `{command}` and got:\n```\n{output[:8000]}\n```\nPlease analyze and fix any issues."
            self._handle_user_input(prompt)
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
