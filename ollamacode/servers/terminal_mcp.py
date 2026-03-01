"""
Built-in terminal MCP server: run_command (capture stdout/stderr).

Uses same workspace root as fs/codebase: OLLAMACODE_FS_ROOT or process cwd.
Optional blocklist: set OLLAMACODE_BLOCK_DANGEROUS_COMMANDS=1 to block dangerous patterns.
Optional allowlist: set OLLAMACODE_ALLOWED_COMMANDS to a comma-separated list (e.g. ruff,pytest,git).
When set, only commands whose first word matches one of the entries are allowed.
Optional command log: set OLLAMACODE_LOG_COMMANDS=1 to append (cwd, command, return_code) to a log file for debugging.
"""

import logging
import os
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from . import configure_server_logging

configure_server_logging()
logger = logging.getLogger(__name__)

mcp = FastMCP("ollamacode-terminal")

# Substrings blocked when OLLAMACODE_BLOCK_DANGEROUS_COMMANDS is set (case-insensitive).
# Whitespace is normalized before matching so "rm  -rf  /" still matches.
_BLOCKED_SUBSTRINGS = [
    "rm -rf /",
    "rm -rf /*",
    "| bash",
    "| sh ",
    "| sh",
    " -o- | sh",
    " -o- | bash",
    ":(){ :|:& };:",  # fork bomb
]

# Environment variable names that must never be overridden by tool callers.
_DANGEROUS_ENV_VARS = frozenset(
    {
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "DYLD_FRAMEWORK_PATH",
        "BASH_ENV",
        "ENV",
        "BASH_FUNC_%%",
        "CDPATH",
        "GLOBIGNORE",
        "IFS",
        "PROMPT_COMMAND",
        "SHELLOPTS",
        "BASHOPTS",
    }
)

# Patterns that require confirmation when OLLAMACODE_CONFIRM_RISKY=1 (optional confirm-before-run)
_RISKY_PATTERNS = [
    r"rm\s+-rf\s+\S+",  # rm -rf <path>
    r"\|\s*(bash|sh)\b",  # | bash or | sh
    r"curl\s+.*\|\s*(bash|sh)\b",
]


def _root() -> str:
    """Workspace root (same as fs_mcp): OLLAMACODE_FS_ROOT env or current working directory. Path is normalized for the current OS."""
    root = os.environ.get("OLLAMACODE_FS_ROOT")
    path = Path(root).resolve() if root else Path.cwd()
    return str(path)


def _allowed_commands() -> set[str]:
    """Return set of allowed command prefixes when OLLAMACODE_ALLOWED_COMMANDS is set (else empty = no allowlist)."""
    raw = os.environ.get("OLLAMACODE_ALLOWED_COMMANDS", "").strip()
    if not raw:
        return set()
    return {s.strip().lower() for s in raw.split(",") if s.strip()}


def _normalize_whitespace(cmd: str) -> str:
    """Collapse runs of whitespace to single space for reliable pattern matching."""
    return re.sub(r"\s+", " ", cmd.strip())


def _is_blocked(command: str) -> bool:
    """Return True if blocklist is enabled and command matches a blocked pattern.

    Normalises whitespace so ``rm  -rf  /`` still matches ``rm -rf /``.
    """
    if os.environ.get(
        "OLLAMACODE_BLOCK_DANGEROUS_COMMANDS", ""
    ).strip().lower() not in ("1", "true", "yes"):
        return False
    cmd_lower = _normalize_whitespace(command).lower()
    for pat in _BLOCKED_SUBSTRINGS:
        if pat.lower() in cmd_lower:
            return True
    # Block rm -rf with path that is clearly root (e.g. rm -rf / or rm -rf //)
    if re.search(r"rm\s+-rf\s+/+(\s|$)", cmd_lower):
        return True
    return False


def _is_disallowed_by_allowlist(command: str) -> bool:
    """Return True if allowlist is set and command's first word is not in it.

    Also checks every sub-command after shell separators (&&, ||, ;, |) so that
    ``git status && rm -rf /`` is blocked when only ``git`` is allowed.
    """
    allowed = _allowed_commands()
    if not allowed:
        return False
    # Split on shell meta-characters to get all sub-commands.
    sub_cmds = re.split(r"\s*(?:&&|\|\||[;|])\s*", command.strip())
    for sub in sub_cmds:
        sub = sub.strip()
        if not sub:
            continue
        try:
            parts = shlex.split(sub)
            first = parts[0].lower() if parts else ""
        except ValueError:
            # Malformed quoting — reject rather than risk a bypass.
            return True
        if first not in allowed:
            return True
    return False


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


def _log_command(cwd: str, command: str, return_code: int) -> None:
    """If OLLAMACODE_LOG_COMMANDS=1, append one line (timestamp, cwd, command, return_code) to log file."""
    if os.environ.get("OLLAMACODE_LOG_COMMANDS", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return
    log_path = os.environ.get("OLLAMACODE_COMMAND_LOG", "").strip()
    if not log_path:
        log_path = str(Path.home() / ".ollamacode" / "command_history.log")
    try:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = command.replace("\n", " ").replace("\r", "")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{ts}\t{cwd}\t{line}\t{return_code}\n")
    except OSError:
        pass


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
    When OLLAMACODE_LOG_COMMANDS=1, each run is appended to a log file (path in OLLAMACODE_COMMAND_LOG or ~/.ollamacode/command_history.log).
    """
    # Clamp timeout to a safe range regardless of what the model passes.
    timeout_seconds = max(1, min(timeout_seconds, 300))

    # Sandbox level enforcement (readonly blocks all commands; null-byte guard).
    try:
        from ollamacode.sandbox import check_terminal_command

        check_terminal_command(command)
    except PermissionError as exc:
        _log_command(cwd or _root(), command, -3)
        return {"stdout": "", "stderr": str(exc), "return_code": -3}

    if _is_blocked(command):
        out = {
            "stdout": "",
            "stderr": "Command blocked by OLLAMACODE_BLOCK_DANGEROUS_COMMANDS (dangerous pattern).",
            "return_code": -1,
        }
        _log_command(cwd or _root(), command, -1)
        return out
    if _is_disallowed_by_allowlist(command):
        out = {
            "stdout": "",
            "stderr": "Command not in OLLAMACODE_ALLOWED_COMMANDS allowlist.",
            "return_code": -1,
        }
        _log_command(cwd or _root(), command, -1)
        return out
    if (
        _confirm_risky_enabled()
        and _is_risky(command)
        and not _confirm_risky_confirmed()
    ):
        out = {
            "stdout": "",
            "stderr": "Command matches a risky pattern. To run it, set OLLAMACODE_CONFIRM_RISKY_CONFIRMED=1 and re-run, or run the command yourself in a terminal.",
            "return_code": -2,
        }
        _log_command(cwd or _root(), command, -2)
        return out
    run_env = os.environ.copy()
    if env:
        # Block dangerous env var overrides (LD_PRELOAD, BASH_ENV, etc.)
        blocked_vars = [k for k in env if k.upper() in _DANGEROUS_ENV_VARS]
        if blocked_vars:
            _log_command(cwd or _root(), command, -1)
            return {
                "stdout": "",
                "stderr": f"Blocked: cannot override dangerous env vars: {', '.join(blocked_vars)}",
                "return_code": -1,
            }
        # Reject env var names with = or null bytes
        for k, v in env.items():
            if "\0" in k or "\0" in v or "=" in k:
                _log_command(cwd or _root(), command, -1)
                return {
                    "stdout": "",
                    "stderr": f"Invalid env var name or value: {k!r}",
                    "return_code": -1,
                }
        run_env.update(env)

    # Validate cwd is within workspace root
    work_dir = cwd or _root()
    if cwd:
        workspace = Path(_root()).resolve()
        resolved_cwd = (
            (workspace / cwd.lstrip("/")).resolve()
            if not Path(cwd).is_absolute()
            else Path(cwd).resolve()
        )
        if not resolved_cwd.is_relative_to(workspace):
            _log_command(str(workspace), command, -1)
            return {
                "stdout": "",
                "stderr": f"cwd {cwd!r} is outside workspace root",
                "return_code": -1,
            }
        if not resolved_cwd.is_dir():
            return {
                "stdout": "",
                "stderr": f"cwd {cwd!r} is not a directory",
                "return_code": -1,
            }
        work_dir = str(resolved_cwd)
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
        _log_command(work_dir, command, result.returncode)
        return {
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
            "return_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        _log_command(work_dir, command, -1)
        return {
            "stdout": "",
            "stderr": f"Command timed out after {timeout_seconds}s",
            "return_code": -1,
        }
    except Exception as e:
        _log_command(work_dir, command, -1)
        return {"stdout": "", "stderr": str(e), "return_code": -1}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
