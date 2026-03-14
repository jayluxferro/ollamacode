"""Session-scoped interactive permission helpers."""

from __future__ import annotations

import fnmatch
from collections import defaultdict
from typing import Iterable

from .permissions import PermissionManager, ToolPermission


class SessionApprovalStore:
    """In-memory session approvals for `always allow` decisions."""

    def __init__(self) -> None:
        self._approved: dict[str, list[str]] = defaultdict(list)
        self._granted: dict[str, int] = defaultdict(int)
        self._denied: dict[str, int] = defaultdict(int)

    def is_allowed(self, session_key: str | None, names: Iterable[str]) -> bool:
        if not session_key:
            return False
        patterns = self._approved.get(session_key, [])
        if not patterns:
            return False
        for name in names:
            if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
                return True
        return False

    def allow(self, session_key: str | None, patterns: Iterable[str]) -> None:
        if not session_key:
            return
        existing = self._approved[session_key]
        for pattern in patterns:
            if pattern and pattern not in existing:
                existing.append(pattern)

    def record_grant(self, session_key: str | None) -> None:
        if session_key:
            self._granted[session_key] += 1

    def record_deny(self, session_key: str | None) -> None:
        if session_key:
            self._denied[session_key] += 1

    def granted_count(self, session_key: str | None) -> int:
        return self._granted.get(session_key or "", 0)

    def denied_count(self, session_key: str | None) -> int:
        return self._denied.get(session_key or "", 0)


def evaluate_permission(
    permission_manager: PermissionManager | None,
    approval_store: SessionApprovalStore | None,
    session_key: str | None,
    names: Iterable[str],
) -> ToolPermission:
    """Evaluate config and interactive approvals for a tool invocation."""
    checked = [name for name in names if name]
    if approval_store is not None and approval_store.is_allowed(session_key, checked):
        return ToolPermission.ALLOW
    if permission_manager is None:
        return ToolPermission.ASK

    saw_allow = False
    for name in checked:
        decision = permission_manager.check(name)
        if decision is ToolPermission.DENY:
            return decision
        if decision is ToolPermission.ALLOW:
            saw_allow = True
    return ToolPermission.ALLOW if saw_allow else ToolPermission.ASK
