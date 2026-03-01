"""
Custom command templates loaded from config.

Allows users to define reusable slash commands in ollamacode.yaml:

  commands:
    - name: explain
      template: "Explain this code in detail: {input}"
      description: "Explain code"
    - name: refactor
      template: "Refactor the following for clarity and performance: {input}"
      description: "Refactor code"
    - name: test
      template: "Write unit tests for: {input}"
      description: "Generate tests"

Commands are invoked in the TUI as /explain <input>, /refactor <input>, etc.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CustomCommand:
    """A user-defined slash command template."""

    name: str
    template: str
    description: str = ""

    def execute(self, user_input: str = "") -> str:
        """Expand the template with user input.

        The template may contain {input} as a placeholder.
        If no placeholder, the input is appended.
        """
        if "{input}" in self.template:
            return self.template.replace("{input}", user_input)
        # If no placeholder, append input after the template
        result = self.template
        if user_input:
            result = f"{result} {user_input}"
        return result


class CommandManager:
    """Manages custom slash commands from config."""

    def __init__(self) -> None:
        self._commands: dict[str, CustomCommand] = {}

    def load_from_config(self, config: dict[str, Any]) -> int:
        """Load custom commands from config dict.

        Expects config to have a 'commands' key with a list of command dicts.
        Returns the number of commands loaded.
        """
        raw = config.get("commands")
        if not isinstance(raw, list):
            return 0

        loaded = 0
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            name = (entry.get("name") or "").strip().lower()
            template = (entry.get("template") or "").strip()
            if not name or not template:
                logger.warning(
                    "Skipping custom command with missing name or template: %s", entry
                )
                continue
            # Prevent overriding built-in commands
            if name in _RESERVED_NAMES:
                logger.warning(
                    "Custom command %r conflicts with built-in command; skipping", name
                )
                continue
            self._commands[name] = CustomCommand(
                name=name,
                template=template,
                description=(entry.get("description") or "").strip(),
            )
            loaded += 1

        logger.debug("Loaded %d custom commands", loaded)
        return loaded

    def get_command(self, name: str) -> CustomCommand | None:
        """Look up a command by name."""
        return self._commands.get(name.strip().lower())

    def execute_command(self, name: str, user_input: str = "") -> str | None:
        """Execute a command by name with the given input.

        Returns the expanded template string, or None if command not found.
        """
        cmd = self.get_command(name)
        if cmd is None:
            return None
        return cmd.execute(user_input)

    def list_commands(self) -> list[dict[str, str]]:
        """Return a list of command info dicts (name, template, description)."""
        return [
            {
                "name": cmd.name,
                "template": cmd.template,
                "description": cmd.description,
            }
            for cmd in sorted(self._commands.values(), key=lambda c: c.name)
        ]

    def has_command(self, name: str) -> bool:
        """Check if a custom command exists."""
        return name.strip().lower() in self._commands

    @property
    def command_names(self) -> list[str]:
        """Return sorted list of command names for autocomplete."""
        return sorted(self._commands.keys())


# Built-in TUI commands that cannot be overridden
_RESERVED_NAMES = frozenset(
    {
        "clear",
        "new",
        "help",
        "model",
        "fix",
        "test",
        "docs",
        "profile",
        "plan",
        "continue",
        "rate",
        "query_docs",
        "multi",
        "copy",
        "trace",
        "compact",
        "reset-state",
        "summary",
        "auto",
        "agents",
        "listen",
        "say",
        "commands",
        "sessions",
        "search",
        "refactor",
        "palette",
        "resume",
        "session",
        "branch",
        "checkpoints",
        "rewind",
        "quit",
        "exit",
        "mode",
        "variant",
        "export",
        "import",
    }
)
