"""
Built-in terminal MCP server: run_command (capture stdout/stderr).

Uses same workspace root as fs/codebase: OLLAMACODE_FS_ROOT or process cwd.
Optional blocklist: set OLLAMACODE_BLOCK_DANGEROUS_COMMANDS=1 to block dangerous patterns.
Optional allowlist: set OLLAMACODE_ALLOWED_COMMANDS to a comma-separated list (e.g. ruff,pytest,git).
When set, only commands whose first word matches one of the entries are allowed.
"""

import os
import re
import subprocess
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ollamacode-terminal")

# Substrings blocked when OLLAMACODE_BLOCK_DANGEROUS_COMMANDS is set (case-insensitive)
_BLOCKED_SUBSTRINGS = [
    "rm -rf /",
    "rm -rf /*",
    "| bash",
    "| sh ",
    "| sh",  # pipe to sh (with or without trailing space)
    " -o- | sh",
    " -o- | bash",
    ":(){ :|:& };:",  # fork bomb
]

# Patterns that require confirmation when OLLAMACODE_CONFIRM_RISKY=1 (optional confirm-before-run)
_RISKY_PATTERNS = [
    r"rm\s+-rf\s+\S+",  # rm -rf <path>
    r"\|\s*(bash|sh)\b",  # | bash or | sh
    r"curl\s+.*\|\s*(bash|sh)\b",
]


def _root() -> str:
    """Workspace root (same as fs_mcp): OLLAMACODE_FS_ROOT env or current working directory."""
    root = os.environ.get("OLLAMACODE_FS_ROOT")
    return os.path.abspath(root) if root else os.getcwd()


def _allowed_commands() -> set[str]:
    """Return set of allowed command prefixes when OLLAMACODE_ALLOWED_COMMANDS is set (else empty = no allowlist)."""
    raw = os.environ.get("OLLAMACODE_ALLOWED_COMMANDS", "").strip()
    if not raw:
        return set()
    return {s.strip().lower() for s in raw.split(",") if s.strip()}


def _is_blocked(command: str) -> bool:
    """Return True if blocklist is enabled and command matches a blocked pattern."""
    if os.environ.get(
        "OLLAMACODE_BLOCK_DANGEROUS_COMMANDS", ""
    ).strip().lower() not in ("1", "true", "yes"):
        return False
    cmd_lower = command.strip().lower()
    for pat in _BLOCKED_SUBSTRINGS:
        if pat.lower() in cmd_lower:
            return True
    # Block rm -rf with path that is clearly root (e.g. rm -rf / or rm -rf //)
    if re.search(r"rm\s+-rf\s+/+(\s|$)", cmd_lower):
        return True
    return False


def _is_disallowed_by_allowlist(command: str) -> bool:
    """Return True if allowlist is set and command's first word is not in it."""
    allowed = _allowed_commands()
    if not allowed:
        return False
    first = (command.strip().split() or [""])[0].lower()
    return first not in allowed


def _is_risky(command: str) -> bool:
    """Return True if command matches patterns that may require confirmation (OLLAMACODE_CONFIRM_RISKY=1)."""
    cmd = command.strip()
    for pattern in _RISKY_PATTERNS:
        if re.search(pattern, cmd, re.IGNORECASE):
            return True
    return False


def _confirm_risky_enabled() -> bool:
    """Return True if optional confirm-before-run for risky commands is enabled."""
    return os.environ.get("OLLAMACODE_CONFIRM_RISKY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _confirm_risky_confirmed() -> bool:
    """Return True if user has confirmed running the risky command (re-run with OLLAMACODE_CONFIRM_RISKY_CONFIRMED=1)."""
    return os.environ.get("OLLAMACODE_CONFIRM_RISKY_CONFIRMED", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


@mcp.tool()
def run_command(
    command: str,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """
    Run a shell command and return stdout, stderr, and return code.

    command: Shell command to run (e.g. "ls -la" or "git status"). In JSON args use \\n for newlines, not literal line breaks.
    cwd: Working directory (default: workspace root from OLLAMACODE_FS_ROOT or process cwd).
    env: Optional env vars to merge with current env (e.g. {"VAR": "value"}).
    timeout_seconds: Kill command after this many seconds (default 60).

    When OLLAMACODE_BLOCK_DANGEROUS_COMMANDS=1, blocks patterns like rm -rf /, curl|bash, etc.
    When OLLAMACODE_ALLOWED_COMMANDS is set, only commands whose first word is in the list are allowed.
    When OLLAMACODE_CONFIRM_RISKY=1, risky patterns (e.g. rm -rf, curl|bash) require confirmation:
    re-run with OLLAMACODE_CONFIRM_RISKY_CONFIRMED=1 to execute.
    """
    if _is_blocked(command):
        return {
            "stdout": "",
            "stderr": "Command blocked by OLLAMACODE_BLOCK_DANGEROUS_COMMANDS (dangerous pattern).",
            "return_code": -1,
        }
    if _is_disallowed_by_allowlist(command):
        return {
            "stdout": "",
            "stderr": "Command not in OLLAMACODE_ALLOWED_COMMANDS allowlist.",
            "return_code": -1,
        }
    if (
        _confirm_risky_enabled()
        and _is_risky(command)
        and not _confirm_risky_confirmed()
    ):
        return {
            "stdout": "",
            "stderr": "Command matches a risky pattern. To run it, set OLLAMACODE_CONFIRM_RISKY_CONFIRMED=1 and re-run, or run the command yourself in a terminal.",
            "return_code": -2,
        }
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    work_dir = cwd or _root()
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=work_dir,
            env=run_env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return {
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
            "return_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Command timed out after {timeout_seconds}s",
            "return_code": -1,
        }
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "return_code": -1}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
