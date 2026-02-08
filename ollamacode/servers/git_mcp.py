"""
Built-in Git MCP server (read-only): status, diff, log, show, branch.

Enables the agent to understand repo state without modifying it.
Root directory: OLLAMACODE_FS_ROOT env var, or current working directory.
"""

import os
import subprocess
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ollamacode-git")


def _root() -> Path:
    root = os.environ.get("OLLAMACODE_FS_ROOT")
    return Path(root).resolve() if root else Path.cwd().resolve()


def _run_git(args: list[str], cwd: Path | None = None) -> tuple[str, str, int]:
    """Run git with args; return (stdout, stderr, returncode)."""
    cwd = cwd or _root()
    try:
        r = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return (r.stdout or "", r.stderr or "", r.returncode)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return ("", str(e), -1)


@mcp.tool()
def git_status(path: str = ".") -> str:
    """
    Show working tree status (short format). path: subdir relative to repo root (default '.').
    Returns which files are modified, staged, or untracked.
    """
    root = _root()
    target = (root / path.lstrip("/")).resolve() if path != "." else root
    if not target.is_dir():
        return f"Not a directory: {path}"
    out, err, code = _run_git(["status", "--short"], cwd=target)
    if code != 0:
        return err or f"git status failed (code {code})"
    return out.strip() or "Nothing to commit, working tree clean"


@mcp.tool()
def git_diff_unstaged(path: str = ".") -> str:
    """
    Show unstaged changes (working tree vs index). path: subdir or file relative to repo root (default '.' = whole repo).
    """
    root = _root()
    if path != ".":
        target = (root / path.lstrip("/")).resolve()
        if not target.exists():
            return f"Path not found: {path}"
    args = ["diff", "--", path] if path != "." else ["diff"]
    out, err, code = _run_git(args, cwd=root)
    if code != 0:
        return err or f"git diff failed (code {code})"
    return out.strip() or "(no unstaged changes)"


@mcp.tool()
def git_diff_staged(path: str = ".") -> str:
    """
    Show staged changes (index vs HEAD). path: subdir or file relative to repo root (default '.' = whole repo).
    """
    root = _root()
    if path != ".":
        target = (root / path.lstrip("/")).resolve()
        if not target.exists():
            return f"Path not found: {path}"
    args = ["diff", "--staged", "--", path] if path != "." else ["diff", "--staged"]
    out, err, code = _run_git(args, cwd=root)
    if code != 0:
        return err or f"git diff --staged failed (code {code})"
    return out.strip() or "(no staged changes)"


@mcp.tool()
def git_log(
    path: str = ".",
    max_count: int = 20,
    oneline: bool = True,
) -> str:
    """
    Show commit log. path: subdir or file (optional). max_count: max commits (default 20). oneline: one line per commit (default true).
    """
    root = _root()
    args = ["log"]
    if oneline:
        args.append("--oneline")
    args.extend(["-n", str(max(1, min(max_count, 100)))])
    if path and path != ".":
        args.extend(["--", path])
    out, err, code = _run_git(args, cwd=root)
    if code != 0:
        return err or f"git log failed (code {code})"
    return out.strip() or "(no commits)"


@mcp.tool()
def git_show(ref: str = "HEAD", path: str | None = None) -> str:
    """
    Show commit content (message + diff). ref: commit ref (default HEAD). path: optional path to limit diff.
    """
    root = _root()
    args = ["show", "--stat", ref]
    if path:
        args.extend(["--", path])
    out, err, code = _run_git(args, cwd=root)
    if code != 0:
        return err or f"git show failed (code {code})"
    return out.strip() or "(empty)"


@mcp.tool()
def git_branch(local: bool = True, remote: bool = False) -> str:
    """
    List branches. local: include local branches (default true). remote: include remote branches (default false).
    """
    root = _root()
    args = ["branch"]
    if remote:
        args.append("-r")
    if not local and not remote:
        args.append("-a")
    out, err, code = _run_git(args, cwd=root)
    if code != 0:
        return err or f"git branch failed (code {code})"
    return out.strip() or "(no branches)"


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
