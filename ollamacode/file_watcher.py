"""
File watcher for external changes: watchdog (optional) with polling fallback.

Usage:
    def on_change(event_type, file_path):
        print(f"{event_type}: {file_path}")

    handle = watch_directory("/path/to/project", on_change, patterns=["*.py", "*.js"])
    # ... later ...
    handle.stop()
"""

from __future__ import annotations

import fnmatch
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Event types
CREATED = "created"
MODIFIED = "modified"
DELETED = "deleted"

FileChangeCallback = Callable[[str, str], None]  # (event_type, file_path)


class WatcherHandle:
    """Opaque handle to stop a directory watcher."""

    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._observer: Any = None  # watchdog observer if used

    def stop(self) -> None:
        """Stop watching."""
        self._stop_event.set()
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=5)
            except Exception as exc:
                logger.debug("Watchdog observer stop error: %s", exc)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set()


def _matches_patterns(path: str, patterns: list[str]) -> bool:
    """Check if filename matches any of the glob patterns."""
    if not patterns:
        return True
    name = os.path.basename(path)
    return any(fnmatch.fnmatch(name, p) for p in patterns)


# ---------------------------------------------------------------------------
# Watchdog backend (preferred)
# ---------------------------------------------------------------------------


def _try_watchdog(
    path: str,
    callback: FileChangeCallback,
    patterns: list[str],
    handle: WatcherHandle,
) -> bool:
    """Try to set up watchdog-based watching. Returns True on success."""
    try:
        from watchdog.observers import Observer  # type: ignore[import-untyped]
        from watchdog.events import FileSystemEventHandler, FileSystemEvent  # type: ignore[import-untyped]
    except ImportError:
        return False

    class _Handler(FileSystemEventHandler):  # type: ignore[misc]
        def on_created(self, event: FileSystemEvent) -> None:
            if not event.is_directory and _matches_patterns(event.src_path, patterns):
                callback(CREATED, event.src_path)

        def on_modified(self, event: FileSystemEvent) -> None:
            if not event.is_directory and _matches_patterns(event.src_path, patterns):
                callback(MODIFIED, event.src_path)

        def on_deleted(self, event: FileSystemEvent) -> None:
            if not event.is_directory and _matches_patterns(event.src_path, patterns):
                callback(DELETED, event.src_path)

    observer = Observer()
    observer.schedule(_Handler(), path, recursive=True)
    observer.daemon = True
    observer.start()
    handle._observer = observer
    logger.info("File watcher started (watchdog) on %s", path)
    return True


# ---------------------------------------------------------------------------
# Polling fallback
# ---------------------------------------------------------------------------


def _polling_watcher(
    path: str,
    callback: FileChangeCallback,
    patterns: list[str],
    handle: WatcherHandle,
    interval: float = 2.0,
) -> None:
    """Poll-based file watcher (fallback when watchdog is not installed)."""
    logger.info("File watcher started (polling, interval=%.1fs) on %s", interval, path)
    snapshot: dict[str, float] = {}

    def _scan() -> dict[str, float]:
        result: dict[str, float] = {}
        try:
            for root_dir, _dirs, files in os.walk(path):
                for fname in files:
                    fpath = os.path.join(root_dir, fname)
                    if not _matches_patterns(fpath, patterns):
                        continue
                    try:
                        result[fpath] = os.path.getmtime(fpath)
                    except OSError:
                        pass
        except OSError:
            pass
        return result

    snapshot = _scan()

    while not handle._stop_event.wait(timeout=interval):
        new_snapshot = _scan()

        # Detect created and modified
        for fpath, mtime in new_snapshot.items():
            if fpath not in snapshot:
                callback(CREATED, fpath)
            elif mtime != snapshot[fpath]:
                callback(MODIFIED, fpath)

        # Detect deleted
        for fpath in snapshot:
            if fpath not in new_snapshot:
                callback(DELETED, fpath)

        snapshot = new_snapshot


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def watch_directory(
    path: str | Path,
    callback: FileChangeCallback,
    patterns: list[str] | None = None,
    poll_interval: float = 2.0,
) -> WatcherHandle:
    """Start watching a directory for file changes.

    Args:
        path: Directory to watch (recursively).
        callback: Called with (event_type, file_path) on each change.
        patterns: Optional glob patterns to filter files (e.g. ["*.py", "*.js"]).
        poll_interval: Seconds between polls when using polling fallback.

    Returns:
        WatcherHandle with a .stop() method to cease watching.
    """
    path_str = str(Path(path).resolve())
    if not os.path.isdir(path_str):
        raise ValueError(f"Not a directory: {path_str}")

    handle = WatcherHandle()
    pat = patterns or []

    # Try watchdog first; fall back to polling
    if not _try_watchdog(path_str, callback, pat, handle):
        thread = threading.Thread(
            target=_polling_watcher,
            args=(path_str, callback, pat, handle, poll_interval),
            daemon=True,
        )
        thread.start()
        handle._thread = thread

    return handle
