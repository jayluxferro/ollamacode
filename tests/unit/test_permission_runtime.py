from __future__ import annotations

from ollamacode.permission_runtime import SessionApprovalStore, evaluate_permission
from ollamacode.permissions import PermissionManager, ToolPermission


def test_session_approval_store_allows_matching_patterns():
    store = SessionApprovalStore()
    store.allow("session-1", ["run_command"])
    assert store.is_allowed("session-1", ["run_command"]) is True
    assert store.is_allowed("session-1", ["read_file"]) is False


def test_evaluate_permission_prefers_config_deny():
    manager = PermissionManager.from_config({"permissions": {"run_command": "deny"}})
    store = SessionApprovalStore()
    decision = evaluate_permission(manager, store, "s1", ["run_command"])
    assert decision is ToolPermission.DENY


def test_evaluate_permission_uses_session_approval_before_ask():
    manager = PermissionManager.from_config({"permissions": {"default": "ask"}})
    store = SessionApprovalStore()
    store.allow("s1", ["read_file"])
    decision = evaluate_permission(manager, store, "s1", ["read_file"])
    assert decision is ToolPermission.ALLOW
