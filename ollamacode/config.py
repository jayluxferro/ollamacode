"""
Load optional config from ollamacode.yaml or .ollamacode/config.yaml.

Schema: model?, mcp_servers?, system_prompt_extra?
mcp_servers: list of { type: stdio|sse|streamable_http, command?, args?, url? }
When no config file and no OLLAMACODE_MCP_ARGS, built-in servers (fs, terminal, codebase) are used.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Default MCP servers when no config and no env: built-in fs, terminal, codebase, git (read-only)
DEFAULT_MCP_SERVERS: list[dict[str, Any]] = [
    {"type": "stdio", "command": sys.executable, "args": ["-m", "ollamacode.servers.fs_mcp"]},
    {"type": "stdio", "command": sys.executable, "args": ["-m", "ollamacode.servers.terminal_mcp"]},
    {"type": "stdio", "command": sys.executable, "args": ["-m", "ollamacode.servers.codebase_mcp"]},
    {"type": "stdio", "command": sys.executable, "args": ["-m", "ollamacode.servers.git_mcp"]},
]

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


def _find_config_file(config_path: str | None, cwd: Path | None = None) -> Path | None:
    """Resolve config path: explicit path, or ollamacode.yaml, or .ollamacode/config.yaml."""
    if config_path:
        p = Path(config_path)
        return p.resolve() if p.exists() else None
    base = (cwd or Path.cwd()).resolve()
    for candidate in (
        base / "ollamacode.yaml",
        base / "ollamacode.yml",
        base / ".ollamacode" / "config.yaml",
        base / ".ollamacode" / "config.yml",
    ):
        if candidate.exists():
            return candidate
    return None


def load_config(
    config_path: str | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    """
    Load config from file if present. Returns empty dict if no file or YAML unavailable.

    Supported keys:
      - model: default Ollama model
      - mcp_servers: list of server configs (see below)
      - system_prompt_extra: appended to system prompt
      - max_messages: cap on message history sent to Ollama (0 = no limit; for long chats)

    Each mcp_servers entry:
      - type: "stdio" | "sse" | "streamable_http"
      - For stdio: command (str), args (list of str, optional)
      - For sse / streamable_http: url (str)
    """
    if yaml is None:
        return {}
    path = _find_config_file(config_path, cwd)
    if not path:
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def merge_config_with_env(
    config: dict[str, Any],
    *,
    model_env: str | None = None,
    mcp_args_env: str | None = None,
    system_extra_env: str | None = None,
) -> dict[str, Any]:
    """
    Merge config with env vars. Env overrides config when set.

    - model: config["model"] or model_env
    - mcp_servers: use config["mcp_servers"] if present; else build single stdio from mcp_args_env (split)
    - system_prompt_extra: config["system_prompt_extra"] or system_extra_env
    """
    out: dict[str, Any] = {}
    out["model"] = model_env or config.get("model")
    out["system_prompt_extra"] = (system_extra_env or "").strip() or (config.get("system_prompt_extra") or "").strip()
    out["max_messages"] = config.get("max_messages", 0)

    if mcp_args_env:
        # Legacy: single stdio server from OLLAMACODE_MCP_ARGS "command arg1 arg2" (overrides config)
        parts = mcp_args_env.split()
        out["mcp_servers"] = [
            {"type": "stdio", "command": parts[0], "args": parts[1:] if len(parts) > 1 else []}
        ]
    elif config.get("mcp_servers") is not None:
        out["mcp_servers"] = config["mcp_servers"]
    else:
        # No config and no env: use built-in fs + terminal + codebase servers
        out["mcp_servers"] = list(DEFAULT_MCP_SERVERS)

    return out
