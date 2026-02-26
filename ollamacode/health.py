"""
Health check: verify AI provider (Ollama or remote) and optionally MCP availability.

Used by `ollamacode health` CLI and GET /health when serving.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any


def check_provider(config: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Check the configured AI provider. Falls back to Ollama if no config given."""
    if config:
        try:
            from .providers import get_provider

            p = get_provider(config)
            return p.health_check()
        except Exception as e:
            return False, f"Provider check error: {e}"
    return check_ollama()


def check_ollama() -> tuple[bool, str]:
    """
    Check if Ollama is reachable. Returns (success, message).
    Kept for backward compatibility; prefer check_provider().
    """
    try:
        import ollama

        ollama.list()
        return True, "Ollama is reachable."
    except Exception as e:
        msg = str(e).lower() if e else ""
        if "connection" in msg or "refused" in msg or "connect" in msg:
            return (
                False,
                "Ollama is not running or not reachable. Start it with: ollama serve",
            )
        return False, f"Ollama error: {e}"


def check_toolchain_versions(
    version_checks: list[dict],
    cwd: str | Path | None = None,
    timeout_seconds: int = 10,
) -> list[dict]:
    """
    Run optional toolchain version checks from config.
    version_checks: list of {name, command, expect_contains}.
    Returns list of {name, ok, actual, expect_contains}; ok is True if command output contains expect_contains.
    """
    results = []
    base = Path(cwd).resolve() if cwd else Path.cwd()
    for entry in version_checks:
        name = entry.get("name") or "?"
        command = entry.get("command", "").strip()
        expect = (entry.get("expect_contains") or "").strip()
        if not command:
            continue
        try:
            parts = shlex.split(command)
            if not parts:
                continue
            proc = subprocess.run(
                parts,
                cwd=base,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            actual = (proc.stdout or "").strip() + (proc.stderr or "").strip()
            actual = actual[:500] if len(actual) > 500 else actual
            ok = expect.lower() in actual.lower() if expect else True
            results.append(
                {
                    "name": name,
                    "ok": ok,
                    "actual": actual or "(no output)",
                    "expect_contains": expect,
                }
            )
        except subprocess.TimeoutExpired:
            results.append(
                {
                    "name": name,
                    "ok": False,
                    "actual": "(timeout)",
                    "expect_contains": expect,
                }
            )
        except Exception as e:
            results.append(
                {"name": name, "ok": False, "actual": str(e), "expect_contains": expect}
            )
    return results
