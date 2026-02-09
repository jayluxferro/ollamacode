"""
Minimal terminal MCP server: run_command (capture stdout/stderr).

Optional cwd and env. Run: python examples/terminal_mcp.py  (stdio)
Config: {"type": "stdio", "command": "python", "args": ["examples/terminal_mcp.py"]}
"""

import os
import subprocess
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ollamacode-terminal")


@mcp.tool()
def run_command(
    command: str,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """
    Run a shell command and return stdout, stderr, and return code.

    command: Shell command to run (e.g. "ls -la" or "python -c 'print(1)'").
    cwd: Working directory (default: current process cwd).
    env: Optional env vars to merge with current env (e.g. {"VAR": "value"}).
    timeout_seconds: Kill command after this many seconds (default 60).
    """
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd or os.getcwd(),
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


if __name__ == "__main__":
    mcp.run(transport="stdio")
