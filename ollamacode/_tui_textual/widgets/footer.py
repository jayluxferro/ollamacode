"""Session footer bar — directory, MCP status, permissions, agent mode, variant, sandbox."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static
from textual.reactive import reactive


class SessionFooter(Widget):
    """Bottom bar showing directory and status info."""

    directory = reactive(".")
    mcp_count = reactive(0)
    permissions_count = reactive(0)
    agent_mode = reactive("build")
    variant_name = reactive("")
    sandbox_level = reactive("")

    def compose(self) -> ComposeResult:
        yield Static(self.directory, id="footer-directory")
        yield Static("", id="footer-mcp")
        yield Static("", id="footer-permissions")
        yield Static(self.agent_mode, id="footer-agent")
        yield Static("", id="footer-variant")
        yield Static("", id="footer-sandbox")

    def watch_directory(self, value: str) -> None:
        try:
            home = str(Path.home())
            display = value.replace(home, "~")
            self.query_one("#footer-directory", Static).update(f"  {display}")
        except Exception:
            pass

    def watch_mcp_count(self, value: int) -> None:
        try:
            self.query_one("#footer-mcp", Static).update(
                f"MCP \u25cf{value}" if value else ""
            )
        except Exception:
            pass

    def watch_permissions_count(self, value: int) -> None:
        try:
            self.query_one("#footer-permissions", Static).update(
                f"perms: {value}" if value else ""
            )
        except Exception:
            pass

    def watch_agent_mode(self, value: str) -> None:
        try:
            self.query_one("#footer-agent", Static).update(f"{value} agent")
        except Exception:
            pass

    def watch_variant_name(self, value: str) -> None:
        try:
            self.query_one("#footer-variant", Static).update(
                f"variant: {value}" if value else ""
            )
        except Exception:
            pass

    def watch_sandbox_level(self, value: str) -> None:
        try:
            self.query_one("#footer-sandbox", Static).update(
                f"sandbox: {value}" if value else ""
            )
        except Exception:
            pass
