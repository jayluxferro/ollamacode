"""
Built-in Git MCP server: git_status, git_diff_*, git_log, git_show, git_branch, git_add, git_commit, git_push, git_stash, git_checkout, git_merge, git_branch_delete.

Read-only: status, diff, log, show, branch. Write: add, commit, push, stash, checkout, merge, branch_delete.
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


def _run_git(
    args: list[str], cwd: Path | None = None, max_output: int = 32 * 1024
) -> str:
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
def git_log_graph(cwd: str | None = None, max_count: int = 30) -> str:
    """Show commit log as ASCII graph (branches and merges). max_count: max commits (default 30)."""
    base = _base(cwd)
    if not base.is_dir():
        return f"Not a directory: {cwd or '.'}"
    return _run_git(["log", "--oneline", "--graph", f"-{max_count}"], cwd=base)


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


@mcp.tool()
def git_stash(
    action: str = "list",
    message: str | None = None,
    cwd: str | None = None,
) -> str:
    """Stash changes. action: 'list' (default), 'save' (stash with optional message), 'pop' (apply and drop top stash). message: used when action is 'save'."""
    base = _base(cwd)
    if not base.is_dir():
        return f"Not a directory: {cwd or '.'}"
    action = (action or "list").strip().lower()
    if action == "list":
        return _run_git(["stash", "list"], cwd=base)
    if action == "pop":
        return _run_git(["stash", "pop"], cwd=base)
    if action == "save":
        args = ["stash", "push", "--include-untracked"]
        if message:
            args.extend(["-m", message.strip()])
        return _run_git(args, cwd=base)
    return f"Unknown action: {action}. Use list, save, or pop."


@mcp.tool()
def git_checkout(
    branch: str,
    create: bool = False,
    cwd: str | None = None,
) -> str:
    """Switch to a branch, or create and switch. branch: branch name. create: if True, create new branch (git checkout -b)."""
    base = _base(cwd)
    if not base.is_dir():
        return f"Not a directory: {cwd or '.'}"
    if not (branch or branch.strip()):
        return "Branch name is required."
    branch = branch.strip()
    args = ["checkout", "-b", branch] if create else ["checkout", branch]
    return _run_git(args, cwd=base)


@mcp.tool()
def git_merge(
    branch: str,
    cwd: str | None = None,
) -> str:
    """Merge a branch into the current branch. branch: name of branch to merge."""
    base = _base(cwd)
    if not base.is_dir():
        return f"Not a directory: {cwd or '.'}"
    if not (branch or branch.strip()):
        return "Branch name is required."
    return _run_git(["merge", branch.strip()], cwd=base)


@mcp.tool()
def git_branch_delete(
    branch: str,
    force: bool = False,
    cwd: str | None = None,
) -> str:
    """Delete a branch. branch: branch name to delete. force: if True, use -D (delete even if not merged)."""
    base = _base(cwd)
    if not base.is_dir():
        return f"Not a directory: {cwd or '.'}"
    if not (branch or branch.strip()):
        return "Branch name is required."
    args = ["branch", "-D" if force else "-d", branch.strip()]
    return _run_git(args, cwd=base)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
