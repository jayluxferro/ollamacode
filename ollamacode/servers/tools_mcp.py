"""
Built-in tools MCP server: run_linter, run_tests.

Lets the agent run linters (e.g. ruff, eslint) and test commands (e.g. pytest, npm test)
without extra dependencies; uses subprocess.
Root/cwd: OLLAMACODE_FS_ROOT env var, or current working directory.
"""

import os
import subprocess
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ollamacode-tools")


def _root() -> str:
    root = os.environ.get("OLLAMACODE_FS_ROOT")
    return os.path.abspath(root) if root else os.getcwd()


def _run(
    command: str,
    cwd: str | None = None,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    base = cwd or _root()
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=base,
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


@mcp.tool()
def run_linter(
    command: str,
    cwd: str | None = None,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """
    Run a linter or formatter command and return stdout, stderr, and return code.

    command: Shell command (e.g. "ruff check .", "npx eslint src/", "black --check .").
    cwd: Working directory (default: workspace root from OLLAMACODE_FS_ROOT or cwd).
    timeout_seconds: Kill after this many seconds (default 60).
    """
    return _run(command, cwd=cwd, timeout_seconds=timeout_seconds)


@mcp.tool()
def run_tests(
    command: str,
    cwd: str | None = None,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """
    Run a test command and return stdout, stderr, and return code.

    command: Shell command (e.g. "pytest tests/ -v", "npm test", "cargo test").
    cwd: Working directory (default: workspace root from OLLAMACODE_FS_ROOT or cwd).
    timeout_seconds: Kill after this many seconds (default 120).
    """
    return _run(command, cwd=cwd, timeout_seconds=timeout_seconds)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
