"""
Built-in Git MCP server: git_status, git_diff_*, git_log, git_show, git_branch, git_add, git_commit, git_push.

Read-only: status, diff, log, show, branch. Write: git_add (stage), git_commit, git_push.
Root: OLLAMACODE_FS_ROOT env var, or current working directory.
"""

import os
import subprocess
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ollamacode-git")


def _root() -> Path:
    root = os.environ.get("OLLAMACODE_FS_ROOT")
    return Path(root).resolve() if root else Path.cwd().resolve()


def _base(cwd: str | None) -> Path:
    """Resolve cwd to a directory under workspace root. cwd can be relative to root or absolute (must be under root)."""
    root = _root()
    if not cwd or cwd.strip() in (".", ""):
        return root
    cwd = cwd.strip()
    p = Path(cwd)
    if p.is_absolute():
        try:
            resolved = p.resolve()
            if resolved.is_relative_to(root) or resolved == root:
                return resolved
        except (ValueError, TypeError):
            pass
        return root
    return (root / cwd.lstrip("/")).resolve()


def _run_git(args: list[str], cwd: Path | None = None, max_output: int = 32 * 1024) -> str:
    base = cwd or _root()
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=base,
            capture_output=True,
            text=True,
            timeout=30,
        )
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        if result.returncode != 0 and err:
            return f"git {' '.join(args)} failed (exit {result.returncode}):\n{err}"
        if len(out) > max_output:
            out = out[:max_output] + "\n... (truncated)"
        return out or "(no output)"
    except subprocess.TimeoutExpired:
        return "Command timed out after 30s"
    except FileNotFoundError:
        return "git not found or not in PATH"
    except Exception as e:
        return str(e)


@mcp.tool()
def git_status(cwd: str | None = None) -> str:
    """Show working-tree status (short format). cwd: path relative to workspace root; default '.'."""
    base = _base(cwd)
    if not base.is_dir():
        return f"Not a directory: {cwd or '.'}"
    return _run_git(["status", "--short"], cwd=base)


@mcp.tool()
def git_diff_unstaged(cwd: str | None = None, path: str | None = None) -> str:
    """Show unstaged changes (working tree vs index). path: optional file/path to limit diff."""
    base = _base(cwd)
    if not base.is_dir():
        return f"Not a directory: {cwd or '.'}"
    args = ["diff"]
    if path:
        args.append(path)
    return _run_git(args, cwd=base)


@mcp.tool()
def git_diff_staged(cwd: str | None = None, path: str | None = None) -> str:
    """Show staged changes (index vs HEAD). path: optional file/path to limit diff."""
    base = _base(cwd)
    if not base.is_dir():
        return f"Not a directory: {cwd or '.'}"
    args = ["diff", "--staged"]
    if path:
        args.append(path)
    return _run_git(args, cwd=base)


@mcp.tool()
def git_log(
    cwd: str | None = None,
    max_count: int = 20,
    oneline: bool = True,
) -> str:
    """Show commit log. max_count: max commits (default 20). oneline: one line per commit (default True)."""
    base = _base(cwd)
    if not base.is_dir():
        return f"Not a directory: {cwd or '.'}"
    args = ["log", f"-{max_count}"]
    if oneline:
        args.append("--oneline")
    return _run_git(args, cwd=base)


@mcp.tool()
def git_show(
    revision: str = "HEAD",
    cwd: str | None = None,
    path: str | None = None,
) -> str:
    """Show commit or blob. revision: ref or commit (default HEAD). path: optional path for that revision."""
    base = _base(cwd)
    if not base.is_dir():
        return f"Not a directory: {cwd or '.'}"
    args = ["show", revision]
    if path:
        args.append("--")
        args.append(path)
    return _run_git(args, cwd=base)


@mcp.tool()
def git_branch(cwd: str | None = None, all_branches: bool = False) -> str:
    """List branches. all_branches: include remote (default False)."""
    base = _base(cwd)
    if not base.is_dir():
        return f"Not a directory: {cwd or '.'}"
    args = ["branch"]
    if all_branches:
        args.append("-a")
    return _run_git(args, cwd=base)


@mcp.tool()
def git_add(path: str = ".", cwd: str | None = None) -> str:
    """Stage files for commit. path: file or directory to add (default '.' = all). cwd: path relative to workspace root."""
    base = _base(cwd)
    if not base.is_dir():
        return f"Not a directory: {cwd or '.'}"
    return _run_git(["add", path], cwd=base)


@mcp.tool()
def git_commit(message: str, cwd: str | None = None) -> str:
    """Commit staged changes. message: commit message (use -m). cwd: path relative to workspace root."""
    base = _base(cwd)
    if not base.is_dir():
        return f"Not a directory: {cwd or '.'}"
    if not message.strip():
        return "Commit message is required."
    return _run_git(["commit", "-m", message.strip()], cwd=base)


@mcp.tool()
def git_push(
    cwd: str | None = None,
    remote: str | None = None,
    branch: str | None = None,
) -> str:
    """Push current branch to remote. remote: remote name (default: origin). branch: branch to push (default: current)."""
    base = _base(cwd)
    if not base.is_dir():
        return f"Not a directory: {cwd or '.'}"
    args = ["push"]
    if remote:
        args.append(remote)
    if branch:
        args.append(branch)
    return _run_git(args, cwd=base)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
