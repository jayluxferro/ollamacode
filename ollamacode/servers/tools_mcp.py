"""
Built-in tools MCP server: run_linter, run_tests, run_code_quality, run_coverage, fetch_url, fetch_url_rendered.

Lets the agent run linters (e.g. ruff, eslint) and test commands (e.g. pytest, npm test)
without extra dependencies; uses subprocess.
Root/cwd: OLLAMACODE_FS_ROOT env var, or current working directory.
"""

import os
import subprocess
import urllib.request
from urllib.error import HTTPError
from typing import Any, Literal

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


def _default_code_quality_commands() -> list[str]:
    env = os.environ.get("OLLAMACODE_CODE_QUALITY_COMMANDS", "").strip()
    if env:
        return [c.strip() for c in env.split(",") if c.strip()]
    return [
        "ruff check .",
        "black --check .",
        "isort --check-only .",
        "mypy .",
    ]


@mcp.tool()
def run_code_quality(
    commands: list[str] | None = None,
    cwd: str | None = None,
    timeout_per_command: int = 60,
) -> dict[str, Any]:
    """
    Run a suite of code quality commands (linter, formatter, type checker) and return one aggregated report.

    commands: List of shell commands to run (e.g. ["ruff check .", "black --check .", "mypy ."]).
             If omitted, uses default: ruff check, black --check, isort --check-only, mypy.
             Override via env OLLAMACODE_CODE_QUALITY_COMMANDS (comma-separated).
    cwd: Working directory (default: workspace root).
    timeout_per_command: Max seconds per command (default 60).
    """
    base = cwd or _root()
    cmd_list = commands if commands is not None else _default_code_quality_commands()
    if not cmd_list:
        return {"report": "", "results": [], "all_passed": True}
    results: list[dict[str, Any]] = []
    all_passed = True
    report_parts: list[str] = []
    for cmd in cmd_list:
        name = cmd.split()[0] if cmd.strip() else "?"
        out = _run(cmd, cwd=base, timeout_seconds=timeout_per_command)
        ok = out["return_code"] == 0
        if not ok:
            all_passed = False
        results.append(
            {
                "command": cmd,
                "return_code": out["return_code"],
                "stdout": out["stdout"],
                "stderr": out["stderr"],
            }
        )
        report_parts.append(f"--- {name} (exit {out['return_code']}) ---")
        if out["stdout"]:
            report_parts.append(out["stdout"].strip())
        if out["stderr"]:
            report_parts.append(out["stderr"].strip())
    return {
        "report": "\n\n".join(report_parts),
        "results": results,
        "all_passed": all_passed,
    }


@mcp.tool()
def install_deps(
    requirements_file: str | None = None,
    cwd: str | None = None,
    use_uv: bool = True,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    """
    Install Python dependencies. If requirements_file is omitted, install from pyproject.toml (uv sync or pip install -e .).
    If requirements_file is set (e.g. requirements.txt), run uv pip install -r <file> or pip install -r <file>.

    requirements_file: Path to requirements.txt (or omit to use pyproject.toml in cwd).
    cwd: Working directory (default: workspace root).
    use_uv: Prefer uv over pip when available (default True).
    timeout_seconds: Kill after this many seconds (default 300).
    """
    base = cwd or _root()
    if requirements_file:
        if use_uv:
            return _run(
                f"uv pip install -r {requirements_file!r}",
                cwd=base,
                timeout_seconds=timeout_seconds,
            )
        return _run(
            f"pip install -r {requirements_file!r}",
            cwd=base,
            timeout_seconds=timeout_seconds,
        )
    if use_uv:
        return _run("uv sync", cwd=base, timeout_seconds=timeout_seconds)
    return _run("pip install -e .", cwd=base, timeout_seconds=timeout_seconds)


def _parse_coverage_term_missing(stdout: str) -> tuple[list[str], list[str]]:
    """Parse pytest --cov --cov-report=term-missing style output. Returns (uncovered_files, suggested_tests)."""
    uncovered: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if (
            not line
            or line.startswith("-")
            or "TOTAL" in line
            or "Stmts" in line
            and "Miss" in line
        ):
            continue
        # Table: "path.py    N    M    XX%   ..." where M = missing stmts
        parts = line.split()
        if len(parts) >= 3 and parts[0].endswith(".py"):
            path = parts[0]
            if "site-packages" in path or ".venv" in path:
                continue
            try:
                miss = int(parts[2])
                if miss > 0:
                    uncovered.append(path)
            except (ValueError, IndexError):
                pass
    suggested = []
    for p in uncovered:
        base = p.replace("/", "_").replace(".py", "")
        if not base.startswith("test"):
            suggested.append(
                f"Add tests for {p} (e.g. tests/test_{base}.py or test_{base}.py)"
            )
    return uncovered[:50], suggested[:50]


@mcp.tool()
def run_coverage(
    command: str | None = None,
    cwd: str | None = None,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """
    Run test coverage (e.g. pytest --cov) and return summary plus uncovered files and suggested tests.

    command: Shell command (default: pytest --cov --cov-report=term-missing -q).
    cwd: Working directory (default: workspace root).
    timeout_seconds: Kill after this many seconds (default 120).
    """
    base = cwd or _root()
    cmd = command or "pytest --cov --cov-report=term-missing -q"
    out = _run(cmd, cwd=base, timeout_seconds=timeout_seconds)
    raw = (out["stdout"] or "") + "\n" + (out["stderr"] or "")
    uncovered, suggested = _parse_coverage_term_missing(raw)
    return {
        "return_code": out["return_code"],
        "report": raw.strip(),
        "uncovered_files": uncovered,
        "suggested_tests": suggested,
    }


@mcp.tool()
def fetch_url(
    url: str,
    timeout_seconds: int = 30,
    max_chars: int = 100_000,
) -> dict[str, Any]:
    """
    Fetch a URL with HTTP GET and return the response body as text.

    url: Full URL to fetch (e.g. https://example.com/page).
    timeout_seconds: Request timeout (default 30).
    max_chars: Maximum characters to return from the body (default 100000).
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "OllamaCode/1.0"})
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            if len(body) > max_chars:
                body = body[:max_chars] + "\n... [truncated]"
            return {
                "status_code": getattr(resp, "status", 200),
                "body": body,
                "error": None,
            }
    except HTTPError as e:
        return {
            "status_code": e.code,
            "body": (
                e.read().decode("utf-8", errors="replace")[:max_chars] if e.fp else ""
            ),
            "error": f"HTTP {e.code}: {e.reason}",
        }
    except Exception as e:
        return {"status_code": -1, "body": "", "error": str(e)}


@mcp.tool()
def fetch_url_rendered(
    url: str,
    timeout_seconds: int = 30,
    max_chars: int = 100_000,
    wait_until: Literal["commit", "domcontentloaded", "load", "networkidle"] = "networkidle",
) -> dict[str, Any]:
    """
    Fetch a URL using Playwright (rendered DOM) and return readable text plus HTML.

    url: Full URL to fetch (e.g. https://example.com/page).
    timeout_seconds: Request timeout (default 30).
    max_chars: Maximum characters to return for text/html (default 100000).
    wait_until: Playwright wait condition (default "networkidle").
    """
    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]
    except Exception:
        return {
            "status_code": -1,
            "body": "",
            "html": "",
            "error": "Playwright not available. Install browsers with `playwright install`.",
        }
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until=wait_until, timeout=timeout_seconds * 1000)
            text = page.inner_text("body")
            html = page.content()
            browser.close()
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... [truncated]"
        if len(html) > max_chars:
            html = html[:max_chars] + "\n... [truncated]"
        return {
            "status_code": 200,
            "body": text,
            "html": html,
            "error": None,
        }
    except Exception as e:
        return {"status_code": -1, "body": "", "html": "", "error": str(e)}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
