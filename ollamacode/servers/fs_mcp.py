"""
Built-in filesystem MCP server: read_file, write_file, list_dir (workspace-scoped).

Root directory: OLLAMACODE_FS_ROOT env var, or current working directory.
Sandbox: OLLAMACODE_SANDBOX_LEVEL controls access (readonly/supervised/full).
"""

import json
import logging
import os
import difflib
import subprocess
import tempfile
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from . import configure_server_logging

configure_server_logging()
logger = logging.getLogger(__name__)

mcp = FastMCP("ollamacode-fs")

# Safety limits
_MAX_PATH_LENGTH = 4096
_MAX_READ_BYTES = 10 * 1024 * 1024  # 10 MB
_MAX_WRITE_BYTES = 10 * 1024 * 1024  # 10 MB


# ---------------------------------------------------------------------------
# Auto-formatting after writes (task 137)
# ---------------------------------------------------------------------------


def _get_formatters() -> dict[str, str]:
    """Load per-extension formatter commands.

    Config: OLLAMACODE_AUTO_FORMAT env var as JSON, e.g.
        {"py": "ruff format {path}", "js": "prettier --write {path}"}
    Or OLLAMACODE_AUTO_FORMAT_<EXT> for single extensions, e.g.
        OLLAMACODE_AUTO_FORMAT_PY="ruff format {path}"
    """
    formatters: dict[str, str] = {}
    raw = os.environ.get("OLLAMACODE_AUTO_FORMAT", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                for ext, cmd in parsed.items():
                    formatters[str(ext).lstrip(".")] = str(cmd)
        except (json.JSONDecodeError, TypeError):
            logger.debug("OLLAMACODE_AUTO_FORMAT is not valid JSON: %s", raw[:100])
    # Also check per-extension env vars
    for key, value in os.environ.items():
        if key.startswith("OLLAMACODE_AUTO_FORMAT_") and len(key) > 23:
            ext = key[23:].lower()
            if ext and value.strip():
                formatters[ext] = value.strip()
    return formatters


def _auto_format(path: Path) -> str | None:
    """Run auto-formatter on *path* if configured. Returns formatter output or None."""
    formatters = _get_formatters()
    if not formatters:
        return None
    ext = path.suffix.lstrip(".")
    if not ext or ext not in formatters:
        return None
    cmd_template = formatters[ext]
    cmd = cmd_template.replace("{path}", str(path))
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=str(path.parent),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.debug(
                "Auto-format failed for %s: %s",
                path,
                (result.stderr or "").strip()[:200],
            )
            return f"(auto-format warning: {(result.stderr or '').strip()[:100]})"
        return "(auto-formatted)"
    except subprocess.TimeoutExpired:
        logger.debug("Auto-format timed out for %s", path)
        return "(auto-format timed out)"
    except Exception as exc:
        logger.debug("Auto-format error for %s: %s", path, exc)
        return None


def _root() -> Path:
    root = os.environ.get("OLLAMACODE_FS_ROOT")
    return Path(root).resolve() if root else Path.cwd().resolve()


def _resolve(path: str, *, allow_write: bool = False) -> Path:
    """Resolve *path* relative to workspace root and enforce sandbox policy.

    Guards against symlink traversal, hardlink escapes, and overly long paths.
    """
    from ollamacode.sandbox import check_fs_path

    if len(path) > _MAX_PATH_LENGTH:
        raise ValueError(f"Path too long ({len(path)} chars, max {_MAX_PATH_LENGTH})")

    workspace = _root()
    check_fs_path(path, workspace, allow_write=allow_write)
    p = workspace / path.lstrip("/")
    resolved = p.resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError(f"Path {path!r} is outside workspace root {workspace}")

    # Reject symlinks that point outside workspace (even if resolved path
    # appears to be inside, the link target itself must be verified).
    if p.is_symlink():
        link_target = p.resolve(strict=False)
        if not link_target.is_relative_to(workspace):
            raise ValueError(
                f"Symlink {path!r} points outside workspace (target: {link_target})"
            )

    # Reject hardlinks to files outside workspace (st_nlink > 1 is suspicious
    # for regular files; we verify the resolved path is truly inside).
    if resolved.is_file():
        try:
            stat = resolved.stat()
            if stat.st_nlink > 1:
                # Hardlinked file — verify no component is a symlink escaping root
                for parent in resolved.parents:
                    if parent == workspace:
                        break
                    if parent.is_symlink():
                        raise ValueError(
                            f"Path {path!r} traverses a symlink outside workspace"
                        )
        except OSError:
            pass

    return resolved


@mcp.tool()
def read_file(path: str) -> str:
    """Read the contents of a file. Path is relative to workspace root (OLLAMACODE_FS_ROOT or cwd)."""
    p = _resolve(path)
    if not p.is_file():
        raise FileNotFoundError(f"Not a file or not found: {path}")
    size = p.stat().st_size
    if size > _MAX_READ_BYTES:
        raise ValueError(
            f"File too large to read ({size} bytes, max {_MAX_READ_BYTES}). "
            "Use a more targeted tool or read a portion."
        )
    return p.read_text(encoding="utf-8", errors="replace")


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write content to a file. Path is relative to workspace root (OLLAMACODE_FS_ROOT or cwd). Creates parent dirs if needed."""
    if len(content.encode("utf-8", errors="replace")) > _MAX_WRITE_BYTES:
        raise ValueError(f"Content too large to write (max {_MAX_WRITE_BYTES} bytes)")
    p = _resolve(path, allow_write=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    if os.environ.get("OLLAMACODE_DRY_RUN_DIFF", "0") == "1":
        old = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
        diff = "\n".join(
            difflib.unified_diff(
                old.splitlines(),
                content.splitlines(),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm="",
            )
        )
        return "Dry run (no write).\n" + (diff or "(no changes)")
    # Atomic write: temp file + os.replace() to prevent partial writes on crash
    fd, tmp_path = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
            tmp_f.write(content)
        os.replace(tmp_path, str(p))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    msg = f"Wrote {len(content)} bytes to {path} (absolute: {p})"
    fmt_note = _auto_format(p)
    if fmt_note:
        msg += f" {fmt_note}"
    return msg


@mcp.tool()
def list_dir(path: str = ".") -> list[str]:
    """List directory entries (files and directories). Path is relative to workspace root. Default '.'."""
    p = _resolve(path)
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    return sorted(e.name for e in p.iterdir())


@mcp.tool()
def edit_file(
    path: str, old_string: str, new_string: str, replace_all: bool = False
) -> str:
    """Surgical edit: replace old_string with new_string in file. Path relative to workspace root. replace_all: replace every occurrence (default: first only)."""
    p = _resolve(path, allow_write=True)
    if not p.is_file():
        raise FileNotFoundError(f"Not a file or not found: {path}")
    text = p.read_text(encoding="utf-8", errors="replace")
    if replace_all:
        if old_string not in text:
            return f"No occurrence of old_string in {path}"
        new_text = text.replace(old_string, new_string)
    else:
        if old_string not in text:
            return f"No occurrence of old_string in {path}"
        new_text = text.replace(old_string, new_string, 1)
    if os.environ.get("OLLAMACODE_DRY_RUN_DIFF", "0") == "1":
        diff = "\n".join(
            difflib.unified_diff(
                text.splitlines(),
                new_text.splitlines(),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                lineterm="",
            )
        )
        return "Dry run (no write).\n" + (diff or "(no changes)")
    # Atomic write: temp file + os.replace() to prevent partial writes on crash
    fd, tmp_path = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
            tmp_f.write(new_text)
        os.replace(tmp_path, str(p))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    msg = (
        f"Edited {path} (1 replacement)"
        if not replace_all
        else f"Edited {path} (all replacements)"
    )
    fmt_note = _auto_format(p)
    if fmt_note:
        msg += f" {fmt_note}"
    return msg


@mcp.tool()
def multi_edit(edits: list[dict]) -> str:
    """Batch edits in one call. edits: list of {path, old_string?, new_string}. If old_string omitted, whole file is replaced with new_string. Paths relative to workspace root. Rolls back all changes if any edit fails."""
    dry_run = os.environ.get("OLLAMACODE_DRY_RUN_DIFF", "0") == "1"
    results: list[str] = []
    # Track originals for rollback: list of (Path, original_content)
    originals: list[tuple[Path, str]] = []
    failed = False
    for i, item in enumerate(edits):
        if not isinstance(item, dict):
            results.append(f"[{i}] skip: not a dict")
            continue
        path_val = item.get("path")
        new_text = item.get("newText") or item.get("new_string")
        old_str = item.get("oldText") or item.get("old_string")
        if path_val is None:
            results.append(f"[{i}] skip: missing path")
            continue
        if new_text is None:
            results.append(f"[{i}] {path_val}: skip: missing newText/new_string")
            continue
        try:
            p = _resolve(str(path_val), allow_write=True)
            if not p.is_file():
                results.append(f"[{i}] {path_val}: file not found")
                failed = True
                break
            text = p.read_text(encoding="utf-8", errors="replace")
            if old_str is None or old_str == "":
                new_val = new_text if isinstance(new_text, str) else str(new_text)
                if dry_run:
                    diff = "\n".join(
                        difflib.unified_diff(
                            text.splitlines(),
                            new_val.splitlines(),
                            fromfile=f"a/{path_val}",
                            tofile=f"b/{path_val}",
                            lineterm="",
                        )
                    )
                    results.append(
                        f"[{i}] {path_val}: dry run\n{diff or '(no changes)'}"
                    )
                else:
                    originals.append((p, text))
                    p.write_text(new_val, encoding="utf-8")
                    results.append(f"[{i}] {path_val}: overwrote")
            else:
                old_s = old_str if isinstance(old_str, str) else str(old_str)
                new_s = new_text if isinstance(new_text, str) else str(new_text)
                if old_s not in text:
                    results.append(f"[{i}] {path_val}: old_string not found")
                    failed = True
                    break
                new_content = text.replace(old_s, new_s, 1)
                if dry_run:
                    diff = "\n".join(
                        difflib.unified_diff(
                            text.splitlines(),
                            new_content.splitlines(),
                            fromfile=f"a/{path_val}",
                            tofile=f"b/{path_val}",
                            lineterm="",
                        )
                    )
                    results.append(
                        f"[{i}] {path_val}: dry run\n{diff or '(no changes)'}"
                    )
                else:
                    originals.append((p, text))
                    p.write_text(new_content, encoding="utf-8")
                    results.append(f"[{i}] {path_val}: replaced")
        except Exception as e:
            results.append(f"[{i}] {path_val}: {e}")
            failed = True
            break
    # Rollback all successful writes if any edit failed
    if failed and originals:
        for orig_path, orig_content in originals:
            try:
                orig_path.write_text(orig_content, encoding="utf-8")
            except Exception as rollback_err:
                logger.warning("Rollback failed for %s: %s", orig_path, rollback_err)
        results.append("[rollback] reverted all previous edits due to failure")
    return "\n".join(results)


# ---------------------------------------------------------------------------
# apply_patch tool: apply unified diff patches to codebase files
# ---------------------------------------------------------------------------


@mcp.tool()
def apply_patch(patch_content: str, strip_level: int = 1) -> str:
    """Apply a unified diff patch to files in the workspace.

    patch_content: The unified diff patch text (as produced by `diff -u` or `git diff`).
    strip_level: Number of leading path components to strip (default 1, like `patch -p1`).
    """
    import subprocess as _sp
    import tempfile as _tf

    workspace = _root()
    if not patch_content or not patch_content.strip():
        return "No patch content provided."

    # Write patch to a temp file to avoid shell escaping issues
    fd, tmp_path = _tf.mkstemp(suffix=".patch", dir=str(workspace))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(patch_content)
        # Try system `patch` first (most reliable for unified diffs)
        try:
            result = _sp.run(
                [
                    "patch",
                    f"-p{strip_level}",
                    "--no-backup-if-mismatch",
                    "-i",
                    tmp_path,
                ],
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = (result.stdout or "").strip()
            if result.returncode != 0:
                err = (result.stderr or "").strip()
                return (
                    f"Patch failed (exit {result.returncode}):\n{output}\n{err}".strip()
                )
            return f"Patch applied successfully.\n{output}".strip()
        except FileNotFoundError:
            # `patch` not available; fall back to Python difflib-based application
            return _apply_patch_python(patch_content, workspace, strip_level)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _apply_patch_python(patch_content: str, workspace: Path, strip_level: int) -> str:
    """Minimal Python-only unified diff applier (fallback when `patch` binary is absent)."""
    lines = patch_content.splitlines(keepends=True)
    results: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- "):
            i += 1
            if i >= len(lines) or not lines[i].startswith("+++ "):
                results.append(f"Malformed patch header near line {i}")
                break
            target_path = lines[i][4:].strip()
            # Strip leading path components
            parts = target_path.split("/")
            if len(parts) > strip_level:
                target_path = "/".join(parts[strip_level:])
            # Remove timestamps after tab
            if "\t" in target_path:
                target_path = target_path.split("\t")[0]
            resolved = _resolve(target_path, allow_write=True)
            i += 1
            # Collect hunks
            hunks: list[str] = []
            while i < len(lines) and (
                lines[i].startswith("@@")
                or lines[i].startswith(" ")
                or lines[i].startswith("+")
                or lines[i].startswith("-")
            ):
                hunks.append(lines[i])
                i += 1
            # Apply hunks using difflib
            # Simple application: collect + and - lines per hunk
            new_content = difflib.restore(hunks, 2)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text("".join(new_content), encoding="utf-8")
            results.append(f"Patched {target_path}")
        else:
            i += 1
    return (
        "\n".join(results)
        if results
        else "No files patched (patch may be empty or malformed)."
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
