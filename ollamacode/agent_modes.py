"""
Multi-agent mode switching: BUILD, PLAN, REVIEW modes.

Each mode has its own system prompt prefix and tool permission set.
Switch via /mode command in TUI.

Usage in config (ollamacode.yaml):
  default_mode: build   # build | plan | review
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class AgentMode(enum.Enum):
    BUILD = "build"
    PLAN = "plan"
    REVIEW = "review"


@dataclass(frozen=True)
class ModeConfig:
    """Configuration for a single agent mode."""

    name: str
    system_prompt_prefix: str
    allowed_tools: list[str] | None = None  # None means all tools allowed
    blocked_tools: list[str] = field(default_factory=list)
    description: str = ""


# Default mode configurations
_DEFAULT_MODES: dict[AgentMode, ModeConfig] = {
    AgentMode.BUILD: ModeConfig(
        name="build",
        system_prompt_prefix=(
            "You are in BUILD mode. Focus on implementing code changes, "
            "writing files, running commands, and making concrete progress. "
            "Use tools freely to read, write, and test code."
        ),
        allowed_tools=None,  # all tools
        blocked_tools=[],
        description="Full access to all tools for implementation work",
    ),
    AgentMode.PLAN: ModeConfig(
        name="plan",
        system_prompt_prefix=(
            "You are in PLAN mode. Focus on analysis, architecture, and planning. "
            "Read files and search code to understand the codebase, but do NOT write "
            "files or run destructive commands. Produce a clear, actionable plan."
        ),
        allowed_tools=None,
        blocked_tools=["write_file", "edit_file", "run_command", "create_directory"],
        description="Read-only analysis and planning; no file writes or commands",
    ),
    AgentMode.REVIEW: ModeConfig(
        name="review",
        system_prompt_prefix=(
            "You are in REVIEW mode. Focus on code review: read files, search for "
            "patterns, check for bugs, suggest improvements. Do NOT make changes. "
            "Be thorough and specific in your feedback."
        ),
        allowed_tools=None,
        blocked_tools=["write_file", "edit_file", "run_command", "create_directory"],
        description="Code review mode; read-only with detailed feedback",
    ),
}


class ModeManager:
    """Manages agent mode state and provides mode-specific configuration."""

    def __init__(self, default_mode: str | AgentMode = AgentMode.BUILD) -> None:
        if isinstance(default_mode, str):
            default_mode = _parse_mode(default_mode)
        self._current = default_mode
        self._custom_modes: dict[AgentMode, ModeConfig] = {}

    @property
    def current(self) -> AgentMode:
        return self._current

    @property
    def current_config(self) -> ModeConfig:
        return self._custom_modes.get(self._current) or _DEFAULT_MODES[self._current]

    def switch(self, mode: str | AgentMode) -> AgentMode:
        """Switch to a new mode. Returns the new mode."""
        if isinstance(mode, str):
            mode = _parse_mode(mode)
        self._current = mode
        logger.debug("Agent mode switched to: %s", mode.value)
        return mode

    def get_system_prompt_prefix(self) -> str:
        """Return the system prompt prefix for the current mode."""
        return self.current_config.system_prompt_prefix

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed in the current mode."""
        cfg = self.current_config
        if cfg.allowed_tools is not None and tool_name not in cfg.allowed_tools:
            return False
        if tool_name in cfg.blocked_tools:
            return False
        return True

    def get_blocked_tools(self) -> list[str]:
        """Return the list of blocked tools for the current mode."""
        return list(self.current_config.blocked_tools)

    def set_custom_mode(self, mode: AgentMode, config: ModeConfig) -> None:
        """Override the default configuration for a mode."""
        self._custom_modes[mode] = config

    @staticmethod
    def list_modes() -> list[dict[str, str]]:
        """Return info about all available modes."""
        return [
            {
                "name": mode.value,
                "description": _DEFAULT_MODES[mode].description,
            }
            for mode in AgentMode
        ]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ModeManager:
        """Create a ModeManager from config dict."""
        default = config.get("default_mode", "build")
        return cls(default_mode=default)


def _parse_mode(name: str) -> AgentMode:
    """Parse a mode name string to AgentMode enum."""
    name = name.strip().lower()
    try:
        return AgentMode(name)
    except ValueError:
        logger.warning("Unknown agent mode %r, defaulting to BUILD", name)
        return AgentMode.BUILD
