"""
Hooks: lightweight pre/post tool call hooks similar to Claude Code.

Config files (JSON):
  ~/.ollamacode/hooks.json
  .ollamacode/hooks.json (workspace)

Schema (minimal):
{
  "hooks": {
    "PreToolUse": [
      {"matcher": "write_file|edit_file", "hooks": [{"type": "command", "command": "python ...", "timeout": 30}]}
    ],
    "PostToolUse": [
      {"matcher": ".*", "hooks": [{"type": "http", "url": "http://localhost:8080/hook"}]}
    ]
  }
}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request as _url

logger = logging.getLogger(__name__)

# Shell metacharacters that could allow command injection when passed to shlex.split.
_DANGEROUS_SHELL_PATTERNS = re.compile(r"[;&|`$]|\$\(|>\s*>|<\s*<")


@dataclass
class HookDecision:
    behavior: str | None = None  # allow | deny | modify
    updated_input: dict[str, Any] | None = None
    message: str | None = None


def _hook_paths(workspace_root: str | Path | None) -> list[Path]:
    paths: list[Path] = []
    paths.append(Path.home() / ".ollamacode" / "hooks.json")
    if workspace_root:
        paths.append(Path(workspace_root) / ".ollamacode" / "hooks.json")
    return paths


def _load_hook_config(workspace_root: str | Path | None) -> dict[str, Any]:
    merged: dict[str, Any] = {"hooks": {}}
    if os.environ.get("OLLAMACODE_DISABLE_HOOKS", "0") == "1":
        return merged
    for p in _hook_paths(workspace_root):
        try:
            if p.is_file():
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    hooks = obj.get("hooks")
                    if isinstance(hooks, dict):
                        for k, v in hooks.items():
                            merged["hooks"].setdefault(k, [])
                            if isinstance(v, list):
                                merged["hooks"][k].extend(v)
        except Exception as exc:
            logger.warning("Failed to load hook config from %s: %s", p, exc)
            continue
    return merged


def _match(matcher: str, tool_name: str) -> bool:
    if not matcher:
        return False
    try:
        return re.search(matcher, tool_name) is not None
    except re.error:
        return matcher == tool_name


def _parse_decision(obj: Any) -> HookDecision | None:
    if not isinstance(obj, dict):
        return None
    # Accept {"decision": {"behavior": "allow|deny|modify", "updatedInput": {...}, "message": "..."}}.
    decision = obj.get("decision") if isinstance(obj.get("decision"), dict) else obj
    if not isinstance(decision, dict):
        return None
    behavior = (decision.get("behavior") or decision.get("decision") or "").strip()
    if behavior:
        behavior = behavior.lower()
    updated = decision.get("updatedInput") or decision.get("updated_input")
    if updated is not None and not isinstance(updated, dict):
        updated = None
    message = decision.get("message") or decision.get("reason")
    return HookDecision(
        behavior=behavior or None, updated_input=updated, message=message
    )


async def _run_command_hook(
    command: str,
    payload: dict[str, Any],
    timeout: float,
    cwd: str | None,
) -> HookDecision | None:
    if not command:
        return None
    if _DANGEROUS_SHELL_PATTERNS.search(command):
        logger.warning(
            "Rejecting hook command with dangerous shell metacharacters: %r", command
        )
        return None
    try:
        proc = await asyncio.create_subprocess_exec(
            *shlex.split(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
        )
    except Exception as exc:
        logger.warning("Failed to start hook command %r: %s", command, exc)
        return None
    try:
        stdout, _stderr = await asyncio.wait_for(
            proc.communicate(json.dumps(payload).encode("utf-8")), timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.warning("Hook command %r timed out after %.1fs", command, timeout)
        try:
            proc.kill()
        except Exception:
            pass
        return None
    if not stdout:
        return None
    try:
        obj = json.loads(stdout.decode("utf-8"))
    except Exception:
        return None
    return _parse_decision(obj)


def _run_http_hook_sync(
    url: str, payload: dict[str, Any], timeout: float
) -> HookDecision | None:
    if not url:
        return None
    try:
        req = _url.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _url.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except Exception as exc:
        logger.warning("HTTP hook request to %r failed: %s", url, exc)
        return None
    if not raw:
        return None
    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    return _parse_decision(obj)


class HookManager:
    def __init__(self, workspace_root: str | Path | None, session_id: str | None):
        self._workspace_root = str(workspace_root) if workspace_root else None
        self._session_id = session_id
        self._config = _load_hook_config(workspace_root)

    def _hook_entries(self, event: str) -> list[dict[str, Any]]:
        hooks = self._config.get("hooks") or {}
        entries = hooks.get(event) if isinstance(hooks, dict) else None
        return entries if isinstance(entries, list) else []

    async def run_pre_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        user_prompt: str | None = None,
    ) -> HookDecision | None:
        payload = {
            "event": "PreToolUse",
            "toolName": tool_name,
            "toolInput": tool_input,
            "workspaceRoot": self._workspace_root or "",
            "sessionId": self._session_id or "",
            "userPrompt": user_prompt or "",
            "timestamp": time.time(),
        }
        for entry in self._hook_entries("PreToolUse"):
            matcher = str(entry.get("matcher") or ".*")
            if not _match(matcher, tool_name):
                continue
            for h in entry.get("hooks") or []:
                if not isinstance(h, dict):
                    continue
                htype = (h.get("type") or "command").lower()
                timeout = float(h.get("timeout") or 30)
                if htype == "command":
                    decision = await _run_command_hook(
                        str(h.get("command") or ""),
                        payload,
                        timeout,
                        self._workspace_root,
                    )
                elif htype == "http":
                    decision = await asyncio.to_thread(
                        _run_http_hook_sync,
                        str(h.get("url") or ""),
                        payload,
                        timeout,
                    )
                else:
                    decision = None
                if decision and decision.behavior in ("deny", "modify", "allow"):
                    return decision
        return None

    async def run_post_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: str,
        is_error: bool,
        user_prompt: str | None = None,
    ) -> None:
        payload = {
            "event": "PostToolUse",
            "toolName": tool_name,
            "toolInput": tool_input,
            "toolOutput": tool_output,
            "isError": is_error,
            "workspaceRoot": self._workspace_root or "",
            "sessionId": self._session_id or "",
            "userPrompt": user_prompt or "",
            "timestamp": time.time(),
        }
        for entry in self._hook_entries("PostToolUse"):
            matcher = str(entry.get("matcher") or ".*")
            if not _match(matcher, tool_name):
                continue
            for h in entry.get("hooks") or []:
                if not isinstance(h, dict):
                    continue
                htype = (h.get("type") or "command").lower()
                timeout = float(h.get("timeout") or 30)
                if htype == "command":
                    await _run_command_hook(
                        str(h.get("command") or ""),
                        payload,
                        timeout,
                        self._workspace_root,
                    )
                elif htype == "http":
                    await asyncio.to_thread(
                        _run_http_hook_sync,
                        str(h.get("url") or ""),
                        payload,
                        timeout,
                    )
