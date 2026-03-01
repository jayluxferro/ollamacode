"""
Granular per-tool permission system: ALLOW / DENY / ASK per tool name.

Permissions are loaded from the ollamacode.yaml config file under the
``permissions`` key, with optional glob-style patterns.

Config example (ollamacode.yaml):
    permissions:
      read_file: allow
      write_file: ask
      run_command: deny
      "ollamacode-terminal_*": deny
      default: ask

Usage:
    pm = PermissionManager.from_config(config_dict)
    decision = pm.check("write_file")  # -> ToolPermission.ASK
"""

from __future__ import annotations

import fnmatch
import logging
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path.home() / ".ollamacode" / "ollamacode.yaml"


class ToolPermission(str, Enum):
    """Permission level for a tool invocation."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


def _parse_permission(value: Any) -> ToolPermission:
    """Parse a permission value from config; default to ASK on unknown."""
    raw = str(value).strip().lower() if value is not None else ""
    try:
        return ToolPermission(raw)
    except ValueError:
        return ToolPermission.ASK


class PermissionManager:
    """Check per-tool permissions against a rule set.

    Rules are stored as a list of (pattern, ToolPermission) in order.
    The first matching pattern wins.  A ``default`` key sets the fallback.
    """

    def __init__(
        self,
        rules: list[tuple[str, ToolPermission]] | None = None,
        default: ToolPermission = ToolPermission.ASK,
    ):
        self._rules: list[tuple[str, ToolPermission]] = rules or []
        self._default = default

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> PermissionManager:
        """Build from a config dict (top-level ``permissions`` key).

        If *config* is None, attempt to load from the default YAML config path.
        """
        if config is None:
            config = _load_yaml_config()
        perms_raw = config.get("permissions") if isinstance(config, dict) else None
        if not isinstance(perms_raw, dict):
            return cls()

        default = ToolPermission.ASK
        rules: list[tuple[str, ToolPermission]] = []

        for key, value in perms_raw.items():
            key_str = str(key).strip()
            perm = _parse_permission(value)
            if key_str == "default":
                default = perm
            else:
                rules.append((key_str, perm))

        return cls(rules=rules, default=default)

    def check(self, tool_name: str) -> ToolPermission:
        """Return the permission for *tool_name*: first matching rule, or default."""
        for pattern, perm in self._rules:
            if pattern == tool_name or fnmatch.fnmatch(tool_name, pattern):
                return perm
        return self._default

    @property
    def default(self) -> ToolPermission:
        return self._default

    def __repr__(self) -> str:
        return f"PermissionManager(rules={len(self._rules)}, default={self._default.value})"


def _load_yaml_config() -> dict[str, Any]:
    """Best-effort load of the global ollamacode.yaml config."""
    if not _CONFIG_PATH.is_file():
        return {}
    try:
        import yaml

        return yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.debug("Failed to load config from %s: %s", _CONFIG_PATH, exc)
        return {}
