"""
Built-in terminal MCP server: run_command (capture stdout/stderr).

Uses same workspace root as fs/codebase: OLLAMACODE_FS_ROOT or process cwd.
"""

import os
import subprocess
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ollamacode-terminal")


def _root() -> str:
    """Workspace root (same as fs_mcp): OLLAMACODE_FS_ROOT env or current working directory."""
    root = os.environ.get("OLLAMACODE_FS_ROOT")
    return os.path.abspath(root) if root else os.getcwd()


@mcp.tool()
def run_command(
    command: str,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """
    Run a shell command and return stdout, stderr, and return code.

    command: Shell command to run (e.g. "ls -la" or "git status").
    cwd: Working directory (default: workspace root from OLLAMACODE_FS_ROOT or process cwd).
    env: Optional env vars to merge with current env (e.g. {"VAR": "value"}).
    timeout_seconds: Kill command after this many seconds (default 60).
    """
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
