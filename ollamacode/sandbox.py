"""Sandbox enforcement for OllamaCode MCP servers.

Three levels (set via OLLAMACODE_SANDBOX_LEVEL or --sandbox CLI flag):

  readonly   — Filesystem: read-only; Terminal: no commands allowed.
  supervised — Filesystem: read+write within workspace only (default).
               Terminal: OLLAMACODE_ALLOWED_COMMANDS enforced; dangerous
               commands blocked.
  full       — Filesystem and terminal: unrestricted (use with care).

Violations are logged to ~/.ollamacode/sandbox_violations.log.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Directories that should never be accessible outside 'full' mode.
_SYSTEM_DIRS: frozenset[str] = frozenset(
    {
        "/etc",
        "/usr",
        "/sbin",
        "/bin",
        "/lib",
        "/lib64",
        "/proc",
        "/sys",
        "/dev",
        "/boot",
        "/run",
        "/private/etc",
        "/private/var/db",  # macOS equivalents
    }
)

# Sensitive dotfile directory names — block in readonly + supervised.
_SENSITIVE_DOTDIRS: frozenset[str] = frozenset(
    {
        ".ssh",
        ".aws",
        ".gnupg",
        ".gpg",
        ".kube",
        ".docker",
        ".netrc",
        ".bash_history",
        ".zsh_history",
        ".npmrc",
        ".pypirc",
        ".gitconfig",
        ".git-credentials",
    }
)


class SandboxLevel(str, Enum):
    READONLY = "readonly"
    SUPERVISED = "supervised"
    FULL = "full"


def get_sandbox_level() -> SandboxLevel:
    """Return the active sandbox level from OLLAMACODE_SANDBOX_LEVEL (default: supervised)."""
    raw = os.environ.get("OLLAMACODE_SANDBOX_LEVEL", "supervised").strip().lower()
    try:
        return SandboxLevel(raw)
    except ValueError:
        return SandboxLevel.SUPERVISED


# ---------------------------------------------------------------------------
# Violation logging
# ---------------------------------------------------------------------------


def _log_violation(kind: str, detail: str) -> None:
    log_path = Path.home() / ".ollamacode" / "sandbox_violations.log"
    try:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{ts}\t{kind}\t{detail[:400]}\n")
    except OSError:
        pass
    logger.warning("Sandbox violation [%s]: %s", kind, detail[:200])


# ---------------------------------------------------------------------------
# Filesystem enforcement
# ---------------------------------------------------------------------------


def check_fs_path(
    path_str: str, workspace_root: Path, *, allow_write: bool = False
) -> None:
    """Raise :class:`PermissionError` if *path_str* violates sandbox policy.

    Checks (in order):
    1. Null byte injection
    2. Path traversal / symlink escape (resolved path must stay inside workspace in non-full modes)
    3. System directory access (blocked in readonly + supervised)
    4. Sensitive dotfile directory access (blocked in readonly + supervised)
    5. Write operation in readonly mode
    """
    level = get_sandbox_level()

    # 1. Null byte injection
    if "\x00" in path_str:
        _log_violation("null_byte_path", repr(path_str[:100]))
        raise PermissionError("Path contains null byte — access denied")

    # Resolve path
    try:
        raw = (workspace_root / path_str.lstrip("/")).resolve()
    except (OSError, ValueError) as exc:
        _log_violation("resolve_error", f"{path_str}: {exc}")
        raise PermissionError(f"Cannot resolve path: {exc}") from exc

    if level != SandboxLevel.FULL:
        # 2. Path traversal / symlink escape
        try:
            raw.relative_to(workspace_root.resolve())
        except ValueError:
            _log_violation("path_escape", str(raw))
            raise PermissionError(
                f"Path {path_str!r} escapes workspace root {workspace_root}"
            )

        # 3. System directory access
        raw_str = str(raw)
        for sys_dir in _SYSTEM_DIRS:
            if (
                raw_str == sys_dir
                or raw_str.startswith(sys_dir + "/")
                or raw_str.startswith(sys_dir + os.sep)
            ):
                _log_violation("system_dir_access", raw_str)
                raise PermissionError(
                    f"Access to system directory {sys_dir!r} is not allowed (sandbox: {level.value})"
                )

        # 4. Sensitive dotfile directory access
        for part in raw.parts:
            if part in _SENSITIVE_DOTDIRS:
                _log_violation("dotfile_access", raw_str)
                raise PermissionError(
                    f"Access to sensitive path component {part!r} is not allowed (sandbox: {level.value})"
                )

    # 5. Write enforcement in readonly mode
    if allow_write and level == SandboxLevel.READONLY:
        _log_violation("write_in_readonly", str(raw))
        raise PermissionError(
            "Filesystem writes are not allowed in 'readonly' sandbox mode"
        )


# ---------------------------------------------------------------------------
# Terminal enforcement
# ---------------------------------------------------------------------------


def check_terminal_command(command: str) -> None:
    """Raise :class:`PermissionError` if running *command* violates sandbox policy.

    ReadOnly: all terminal commands are blocked.
    Supervised / Full: null-byte check only (allowlist/blocklist handled by terminal_mcp).
    """
    level = get_sandbox_level()

    # Null byte in command
    if "\x00" in command:
        _log_violation("null_byte_cmd", repr(command[:100]))
        raise PermissionError("Command contains null byte — access denied")

    if level == SandboxLevel.READONLY:
        _log_violation("terminal_in_readonly", command[:200])
        raise PermissionError(
            "Terminal commands are not allowed in 'readonly' sandbox mode"
        )
