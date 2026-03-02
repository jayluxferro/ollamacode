"""Tool call display widgets — bash, read, write, edit, glob, grep, etc."""

from __future__ import annotations

import logging
from typing import Any

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static

logger = logging.getLogger(__name__)

# Tool type icons matching OpenCode
TOOL_ICONS: dict[str, str] = {
    "run_command": "$",
    "bash": "$",
    "read_file": "\u2192",
    "write_file": "\u2190",
    "edit_file": "\u270e",
    "multi_edit": "\u270e",
    "search_codebase": "\u25c7",
    "grep": "\u2731",
    "glob": "\u2731",
    "list_directory": "\u2261",
    "apply_patch": "\u229e",
    "ask_user": "?",
    "fetch_url": "%",
    "web_search": "\u25c8",
}


class InlineToolCall(Widget):
    """Single-line tool call display for simple tools (read, glob, grep)."""

    def __init__(self, name: str, summary: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tool_name = name
        self.summary = summary

    def render(self) -> str:
        icon = TOOL_ICONS.get(self.tool_name, "\u2699")
        return f"{icon} {self.summary}"


class BlockToolCall(Widget):
    """Multi-line tool call display for complex tools (bash, write, edit)."""

    def __init__(
        self,
        name: str,
        args: dict[str, Any],
        result: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.tool_name = name
        self.tool_args = args
        self.result = result

    def compose(self) -> ComposeResult:
        icon = TOOL_ICONS.get(self.tool_name, "\u2699")
        header = self._make_header(icon)
        yield Static(header, classes="tool-call-header")
        if self.result:
            # Truncate long output
            lines = self.result.split("\n")
            if len(lines) > 15:
                display = (
                    "\n".join(lines[:15]) + f"\n... ({len(lines) - 15} more lines)"
                )
            else:
                display = self.result
            yield Static(display, classes="tool-call-body")

    def _make_header(self, icon: str) -> str:
        if self.tool_name in ("run_command", "bash"):
            cmd = self.tool_args.get("command", "")
            return f"{icon} {cmd}"
        elif self.tool_name in ("write_file",):
            path = self.tool_args.get("path", self.tool_args.get("file_path", ""))
            return f"{icon} {path}"
        elif self.tool_name in ("edit_file", "multi_edit"):
            path = self.tool_args.get("path", self.tool_args.get("file_path", ""))
            return f"{icon} {path}"
        elif self.tool_name in ("read_file",):
            path = self.tool_args.get("path", self.tool_args.get("file_path", ""))
            return f"{icon} {path}"
        else:
            return f"{icon} {self.tool_name}"


def make_tool_widget(name: str, args: dict[str, Any], result: str = "") -> Widget:
    """Factory: create the appropriate tool display widget."""
    # Inline tools: read, glob, grep, list_directory, search_codebase
    inline_tools = {
        "read_file",
        "glob",
        "grep",
        "list_directory",
        "search_codebase",
        "fetch_url",
        "web_search",
    }
    if name in inline_tools:
        # Build summary from args
        if name == "read_file":
            summary = f"Read {args.get('path', args.get('file_path', '?'))}"
        elif name in ("glob", "grep", "search_codebase"):
            pattern = args.get("pattern", args.get("query", "?"))
            summary = f"{name}: {pattern}"
        elif name == "list_directory":
            summary = f"ls {args.get('path', '.')}"
        elif name == "fetch_url":
            summary = f"fetch {args.get('url', '?')}"
        elif name == "web_search":
            summary = f"search: {args.get('query', '?')}"
        else:
            summary = name
        return InlineToolCall(name, summary)
    else:
        return BlockToolCall(name, args, result)
