"""
Load optional config from ollamacode.yaml or .ollamacode/config.yaml.

Schema: model?, mcp_servers?, include_builtin_servers?, system_prompt_extra?, rules_file?, ...
mcp_servers: list of { type: stdio|sse|streamable_http, command?, args?, url? }
include_builtin_servers: when true (default), prepend built-in fs/terminal/codebase/tools/git to mcp_servers so the model can list files, run commands, etc. Set false to use only mcp_servers.
When no config file and no OLLAMACODE_MCP_ARGS, built-in servers only are used.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from collections.abc import Mapping
from typing import Any, cast

# Declarative mapping: env var name -> merge_config_with_env keyword argument.
# Single source of truth for which environment variables override config.
ENV_CONFIG_SCHEMA: list[tuple[str, str]] = [
    ("OLLAMACODE_MODEL", "model_env"),
    ("OLLAMACODE_MCP_ARGS", "mcp_args_env"),
    ("OLLAMACODE_SYSTEM_EXTRA", "system_extra_env"),
    ("OLLAMACODE_PYTHON", "python_executable"),
    ("OLLAMACODE_PROVIDER", "provider_env"),
    ("OLLAMACODE_BASE_URL", "base_url_env"),
    ("OLLAMACODE_API_KEY", "api_key_env"),
]

"""Single source of truth for config defaults. Used by merge_config_with_env and load_config."""
DEFAULT_CONFIG: dict[str, Any] = {
    "model": None,
    "provider": "ollama",
    "base_url": None,
    "api_key": None,
    "sandbox_level": "supervised",
    "system_prompt_extra": "",
    "max_messages": 0,
    "max_tool_rounds": 20,
    "max_tool_result_chars": 0,
    "max_edits_per_request": 0,
    "auto_check_after_edits": False,
    "auto_check_mode": "test",
    "auto_check_fix_on_fail": False,
    "auto_check_fix_max_rounds": 1,
    "auto_summarize_after_turns": 0,
    "timing": False,
    "rules_file": None,
    "linter_command": "ruff check .",
    "test_command": "pytest",
    "docs_command": "mkdocs build",
    "profile_command": "python -m cProfile -s cumtime -m pytest tests/ -q --no-header 2>&1 | head -40",
    "semantic_codebase_hint": True,
    "branch_context": True,
    "branch_context_base": "main",
    "pr_description_file": None,
    "serve": None,
    "use_skills": True,
    "prompt_template": None,
    "use_reasoning": True,
    "use_meta_reflection": True,
    "confirm_tool_calls": False,
    "allowed_tools": None,
    "blocked_tools": None,
    "prompt_snippets": [],
    "code_style": None,
    "safety_output_patterns": [],
    "planner_model": None,
    "executor_model": None,
    "reviewer_model": None,
    "multi_agent_max_iterations": 2,
    "multi_agent_require_review": True,
    "tui_tool_trace_max": 5,
    "tui_tool_log_max": 8,
    "tui_tool_log_chars": 160,
    "tui_refresh_hz": 5,
    "toolchain_version_checks": [],
    "inject_recent_context": True,
    "recent_context_max_files": 10,
    "memory_auto_context": True,
    "memory_kg_max_results": 4,
    "memory_rag_max_results": 4,
    "memory_rag_snippet_chars": 220,
    "rlm_sub_model": None,
    "rlm_max_iterations": 20,
    "rlm_stdout_max_chars": 2000,
    "rlm_prefix_chars": 500,
    "rlm_snippet_timeout_seconds": None,
    "rlm_use_subprocess": False,
    "rlm_subprocess_max_memory_mb": 512,
    "rlm_subprocess_max_cpu_seconds": 60,
    "context_management": {
        "enabled": False,
        "summarize_threshold": 0.75,
        "keep_recent_messages": 10,
        "auto_summarize_after_turns": 0,
    },
    "web_search": {
        "enabled": False,
        "provider": None,
        "endpoint": None,
        "api_key": None,
    },
    "subagents": [],
}

# Default MCP servers when no config and no env: fs, terminal, codebase, tools, git (read-only)
DEFAULT_MCP_SERVERS: list[dict[str, Any]] = [
    {
        "type": "stdio",
        "command": sys.executable,
        "args": ["-m", "ollamacode.servers.fs_mcp"],
    },
    {
        "type": "stdio",
        "command": sys.executable,
        "args": ["-m", "ollamacode.servers.terminal_mcp"],
    },
    {
        "type": "stdio",
        "command": sys.executable,
        "args": ["-m", "ollamacode.servers.codebase_mcp"],
    },
    {
        "type": "stdio",
        "command": sys.executable,
        "args": ["-m", "ollamacode.servers.tools_mcp"],
    },
    {
        "type": "stdio",
        "command": sys.executable,
        "args": ["-m", "ollamacode.servers.git_mcp"],
    },
    {
        "type": "stdio",
        "command": sys.executable,
        "args": ["-m", "ollamacode.servers.skills_mcp"],
    },
    {
        "type": "stdio",
        "command": sys.executable,
        "args": ["-m", "ollamacode.servers.state_mcp"],
    },
    {
        "type": "stdio",
        "command": sys.executable,
        "args": ["-m", "ollamacode.servers.reasoning_mcp"],
    },
    {
        "type": "stdio",
        "command": sys.executable,
        "args": ["-m", "ollamacode.servers.screenshot_mcp"],
    },
]

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


class ConfigValidationError(ValueError):
    """Raised when config structure or types are invalid. Message lists all issues found."""

    def __init__(self, message: str, errors: list[str] | None = None):
        super().__init__(message)
        self.errors = errors or []


def validate_config(config: dict[str, Any]) -> None:
    """
    Validate config structure and types. Raises ConfigValidationError with clear messages.
    Only validates keys that are present; missing optional keys are allowed.
    """
    errors: list[str] = []

    if "mcp_servers" in config:
        mcp = config["mcp_servers"]
        if not isinstance(mcp, list):
            errors.append(f"mcp_servers must be a list, got {type(mcp).__name__}")
        else:
            for i, entry in enumerate(mcp):
                if not isinstance(entry, dict):
                    errors.append(
                        f"mcp_servers[{i}] must be a dict, got {type(entry).__name__}"
                    )
                    continue
                kind = (entry.get("type") or "stdio").lower()
                try:
                    from ollamacode.mcp_client import get_registered_mcp_server_types

                    allowed = get_registered_mcp_server_types()
                except Exception:
                    allowed = ("stdio", "sse", "streamable_http")
                if kind not in allowed:
                    errors.append(
                        f"mcp_servers[{i}].type must be one of {', '.join(allowed)}, got {entry.get('type')!r}"
                    )
                elif kind == "stdio":
                    if "command" not in entry or entry.get("command") is None:
                        errors.append(f"mcp_servers[{i}] (stdio) must have 'command'")
                elif kind in ("sse", "streamable_http"):
                    if not entry.get("url"):
                        errors.append(f"mcp_servers[{i}] ({kind}) must have 'url'")

    for key in ("allowed_tools", "blocked_tools"):
        if key not in config:
            continue
        val = config[key]
        if val is not None and not isinstance(val, (list, str)):
            errors.append(f"{key} must be a list or string, got {type(val).__name__}")

    if "prompt_snippets" in config:
        val = config["prompt_snippets"]
        if val is not None and not isinstance(val, list):
            errors.append(f"prompt_snippets must be a list, got {type(val).__name__}")

    if errors:
        msg = "Config validation failed:\n  " + "\n  ".join(errors)
        raise ConfigValidationError(msg, errors)


def find_config_file(
    config_path: str | None,
    cwd: Path | None = None,
    *,
    lookup_parent_dirs: bool | None = None,
) -> Path | None:
    """
    Resolve config path: explicit path, or ollamacode.yaml, or .ollamacode/config.yaml in cwd.
    If lookup_parent_dirs is True (or set via OLLAMACODE_CONFIG_LOOKUP_PARENT=1), also search
    parent directories (e.g. monorepo root).
    """
    if config_path:
        p = Path(config_path)
        return p.resolve() if p.exists() else None
    base = (cwd or Path.cwd()).resolve()
    candidates = (
        base / "ollamacode.yaml",
        base / "ollamacode.yml",
        base / ".ollamacode" / "config.yaml",
        base / ".ollamacode" / "config.yml",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if lookup_parent_dirs is None:
        import os

        lookup_parent_dirs = os.environ.get(
            "OLLAMACODE_CONFIG_LOOKUP_PARENT", ""
        ).strip().lower() in ("1", "true", "yes")
    if lookup_parent_dirs:
        for parent in base.parents:
            for candidate in (
                parent / "ollamacode.yaml",
                parent / "ollamacode.yml",
                parent / ".ollamacode" / "config.yaml",
                parent / ".ollamacode" / "config.yml",
            ):
                if candidate.exists():
                    return candidate
    return None


def _find_config_file(config_path: str | None, cwd: Path | None = None) -> Path | None:
    return find_config_file(config_path, cwd)


def _user_config_path() -> Path:
    """Path to user config: ~/.ollamacode/config.yaml (or config.yml)."""
    base = Path.home() / ".ollamacode"
    for name in ("config.yaml", "config.yml"):
        p = base / name
        if p.exists():
            return p
    return base / "config.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge override into base. Lists and non-dict values are replaced (not concatenated)."""
    out: dict[str, Any] = dict(base)
    for k, v in override.items():
        if k not in out:
            out[k] = v
        elif isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(
    config_path: str | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    """
    Load config: user config (~/.ollamacode/config.yaml) as base, then project config
    (ollamacode.yaml or .ollamacode/config.yaml in cwd) overrides. Deep-merge so project
    overrides user. Returns empty dict if YAML unavailable.

    Supported keys:
      - model: default Ollama model
      - mcp_servers: list of server configs (see below)
      - system_prompt_extra: appended to system prompt
      - max_messages: cap on message history sent to Ollama (0 = no limit; for long chats)
      - max_tool_rounds: max tool-call rounds per turn (default 20)
      - context_management: { enabled, summarize_threshold, keep_recent_messages }
      - rlm_sub_model, rlm_max_iterations, ...: RLM options (see docs/RLM.md)

    Each mcp_servers entry:
      - type: "stdio" | "sse" | "streamable_http"
      - For stdio: command (str), args (list of str, optional)
      - For sse / streamable_http: url (str)
    """
    if yaml is None:
        return {}
    merged: dict[str, Any] = {}
    # 1) User config as base
    user_path = _user_config_path()
    if user_path.exists():
        try:
            data = yaml.safe_load(user_path.read_text()) or {}
            if isinstance(data, dict):
                merged = data
        except Exception:
            pass
    # 2) Project config overrides (explicit path or cwd lookup)
    path = _find_config_file(config_path, cwd)
    if path and path.exists():
        try:
            data = yaml.safe_load(path.read_text()) or {}
            if isinstance(data, dict):
                merged = _deep_merge(merged, data)
        except Exception:
            pass
    return merged


def get_resolved_config(
    config_path: str | None = None,
    cwd: Path | None = None,
    *,
    model_env: str | None = None,
    mcp_args_env: str | None = None,
    system_extra_env: str | None = None,
    python_executable: str | None = None,
    provider_env: str | None = None,
    base_url_env: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    """
    Load config from file (if present) and merge with env, returning a dict that
    always has all keys with defaults. Use this when you need a full config dict
    and do not want to handle missing-file or missing-key yourself.
    """
    raw = load_config(config_path, cwd) or {}
    return merge_config_with_env(
        raw,
        model_env=model_env,
        mcp_args_env=mcp_args_env,
        system_extra_env=system_extra_env,
        python_executable=python_executable,
        provider_env=provider_env,
        base_url_env=base_url_env,
        api_key_env=api_key_env,
    )


def get_env_config_overrides(
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Build merge_config_with_env kwargs from environment (declarative from ENV_CONFIG_SCHEMA).
    Use when calling get_resolved_config(..., **get_env_config_overrides()) or merge_config_with_env(config, **get_env_config_overrides()).
    """
    env_source: Mapping[str, str] = cast(
        Mapping[str, str], env if env is not None else os.environ
    )
    return {param: env_source.get(ev) or None for ev, param in ENV_CONFIG_SCHEMA}


def _clamp_int(
    value: Any, default: int, min_val: int, max_val: int | None = None
) -> int:
    """Coerce value to int and clamp to [min_val, max_val]; use default if invalid."""
    if value is None:
        return default
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    if max_val is not None:
        return max(min_val, min(n, max_val))
    return max(min_val, n)


def _clamp_float(
    value: Any, default: float, min_val: float, max_val: float | None = None
) -> float:
    """Coerce value to float and clamp to [min_val, max_val]; use default if invalid."""
    if value is None:
        return default
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return max(min_val, min(x, max_val)) if max_val is not None else max(min_val, x)


def merge_config_with_env(
    config: dict[str, Any],
    *,
    model_env: str | None = None,
    mcp_args_env: str | None = None,
    system_extra_env: str | None = None,
    python_executable: str | None = None,
    provider_env: str | None = None,
    base_url_env: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    """
    Merge config with env vars. Env overrides config when set.
    Validates config structure (raises ConfigValidationError on invalid mcp_servers etc.).
    Numeric options are validated and clamped to safe ranges.

    - model: config["model"] or model_env
    - mcp_servers: use config["mcp_servers"] if present; else build single stdio from mcp_args_env (split)
    - system_prompt_extra: config["system_prompt_extra"] or system_extra_env
    """
    validate_config(config)
    out: dict[str, Any] = dict(DEFAULT_CONFIG)
    out["model"] = model_env or config.get("model", out["model"])
    out["provider"] = (provider_env or config.get("provider") or out["provider"]).lower().strip()
    out["base_url"] = base_url_env or config.get("base_url") or out["base_url"]
    out["api_key"] = api_key_env or config.get("api_key") or out["api_key"]
    out["system_prompt_extra"] = (system_extra_env or "").strip() or (
        (config.get("system_prompt_extra") or "").strip() or out["system_prompt_extra"]
    )
    out["max_messages"] = _clamp_int(config.get("max_messages"), out["max_messages"], 0)
    out["max_tool_rounds"] = _clamp_int(
        config.get("max_tool_rounds"), out["max_tool_rounds"], 1, 200
    )
    out["max_tool_result_chars"] = _clamp_int(
        config.get("max_tool_result_chars"), out["max_tool_result_chars"], 0
    )
    out["max_edits_per_request"] = _clamp_int(
        config.get("max_edits_per_request"), out["max_edits_per_request"], 0
    )
    out["auto_summarize_after_turns"] = _clamp_int(
        config.get("auto_summarize_after_turns"), out["auto_summarize_after_turns"], 0
    )
    ctx_mgmt = config.get("context_management") or {}
    if isinstance(ctx_mgmt, dict) and ctx_mgmt.get("enabled"):
        after = ctx_mgmt.get("auto_summarize_after_turns")
        if after is not None:
            out["auto_summarize_after_turns"] = _clamp_int(
                after, out["auto_summarize_after_turns"], 0
            )
        else:
            keep = _clamp_int(ctx_mgmt.get("keep_recent_messages"), 10, 1, 200)
            out["auto_summarize_after_turns"] = max(
                out["auto_summarize_after_turns"], keep * 2
            )
    out["timing"] = config.get("timing", out["timing"])
    out["rules_file"] = config.get("rules_file", out["rules_file"])
    out["linter_command"] = config.get("linter_command", out["linter_command"])
    out["test_command"] = config.get("test_command", out["test_command"])
    out["docs_command"] = config.get("docs_command", out["docs_command"])
    out["profile_command"] = config.get("profile_command", out["profile_command"])
    out["semantic_codebase_hint"] = config.get(
        "semantic_codebase_hint", out["semantic_codebase_hint"]
    )
    out["branch_context"] = config.get("branch_context", out["branch_context"])
    out["branch_context_base"] = config.get(
        "branch_context_base", out["branch_context_base"]
    )
    out["pr_description_file"] = config.get(
        "pr_description_file", out["pr_description_file"]
    )
    out["serve"] = config.get("serve", out["serve"])
    out["use_skills"] = config.get("use_skills", out["use_skills"])
    out["prompt_template"] = config.get("prompt_template", out["prompt_template"])
    out["use_reasoning"] = bool(config.get("use_reasoning", out["use_reasoning"]))
    out["use_meta_reflection"] = bool(
        config.get("use_meta_reflection", out["use_meta_reflection"])
    )
    out["confirm_tool_calls"] = bool(
        config.get("confirm_tool_calls", out["confirm_tool_calls"])
    )
    raw_allowed = config.get("allowed_tools")
    out["allowed_tools"] = (
        [raw_allowed]
        if isinstance(raw_allowed, str)
        else (list(raw_allowed) if raw_allowed else out["allowed_tools"])
    )
    raw_blocked = config.get("blocked_tools")
    out["blocked_tools"] = (
        [raw_blocked]
        if isinstance(raw_blocked, str)
        else (list(raw_blocked) if raw_blocked else out["blocked_tools"])
    )
    raw_snippets = config.get("prompt_snippets")
    out["prompt_snippets"] = (
        [raw_snippets]
        if isinstance(raw_snippets, str)
        else (list(raw_snippets) if raw_snippets else out["prompt_snippets"])
    )
    out["code_style"] = (config.get("code_style") or "").strip() or out["code_style"]
    raw_safety = config.get("safety_output_patterns")
    out["safety_output_patterns"] = (
        [raw_safety]
        if isinstance(raw_safety, str)
        else (list(raw_safety) if raw_safety else out["safety_output_patterns"])
    )
    out["planner_model"] = (config.get("planner_model") or "").strip() or out[
        "planner_model"
    ]
    out["executor_model"] = (config.get("executor_model") or "").strip() or out[
        "executor_model"
    ]
    out["reviewer_model"] = (config.get("reviewer_model") or "").strip() or out[
        "reviewer_model"
    ]
    out["multi_agent_max_iterations"] = _clamp_int(
        config.get("multi_agent_max_iterations"),
        out["multi_agent_max_iterations"],
        1,
        10,
    )
    out["multi_agent_require_review"] = bool(
        config.get("multi_agent_require_review", out["multi_agent_require_review"])
    )
    out["tui_tool_trace_max"] = _clamp_int(
        config.get("tui_tool_trace_max"), out["tui_tool_trace_max"], 1, 50
    )
    out["tui_tool_log_max"] = _clamp_int(
        config.get("tui_tool_log_max"), out["tui_tool_log_max"], 1, 200
    )
    out["tui_tool_log_chars"] = _clamp_int(
        config.get("tui_tool_log_chars"), out["tui_tool_log_chars"], 40, 2000
    )
    out["tui_refresh_hz"] = _clamp_int(
        config.get("tui_refresh_hz"), out["tui_refresh_hz"], 1, 30
    )
    raw_checks = config.get("toolchain_version_checks")
    if isinstance(raw_checks, list):
        out["toolchain_version_checks"] = [
            {
                "name": str(c.get("name", "")),
                "command": str(c.get("command", "")),
                "expect_contains": str(c.get("expect_contains", "")),
            }
            for c in raw_checks
            if isinstance(c, dict) and c.get("command")
        ]
    out["inject_recent_context"] = config.get(
        "inject_recent_context", out["inject_recent_context"]
    )
    out["recent_context_max_files"] = _clamp_int(
        config.get("recent_context_max_files"), out["recent_context_max_files"], 1, 500
    )
    out["memory_auto_context"] = bool(
        config.get("memory_auto_context", out["memory_auto_context"])
    )
    out["memory_kg_max_results"] = _clamp_int(
        config.get("memory_kg_max_results"), out["memory_kg_max_results"], 0, 20
    )
    out["memory_rag_max_results"] = _clamp_int(
        config.get("memory_rag_max_results"), out["memory_rag_max_results"], 0, 20
    )
    out["memory_rag_snippet_chars"] = _clamp_int(
        config.get("memory_rag_snippet_chars"),
        out["memory_rag_snippet_chars"],
        40,
        2000,
    )
    out["rlm_sub_model"] = config.get("rlm_sub_model", out["rlm_sub_model"])
    out["rlm_max_iterations"] = _clamp_int(
        config.get("rlm_max_iterations"), out["rlm_max_iterations"], 1, 200
    )
    out["rlm_stdout_max_chars"] = _clamp_int(
        config.get("rlm_stdout_max_chars"), out["rlm_stdout_max_chars"], 0, 100_000
    )
    out["rlm_prefix_chars"] = _clamp_int(
        config.get("rlm_prefix_chars"), out["rlm_prefix_chars"], 0, 50_000
    )
    raw_snippet = config.get("rlm_snippet_timeout_seconds")
    if raw_snippet is not None:
        out["rlm_snippet_timeout_seconds"] = _clamp_float(
            raw_snippet, 30.0, 0.1, 3600.0
        )
    out["rlm_use_subprocess"] = bool(
        config.get("rlm_use_subprocess", out["rlm_use_subprocess"])
    )
    out["rlm_subprocess_max_memory_mb"] = _clamp_int(
        config.get("rlm_subprocess_max_memory_mb"),
        out["rlm_subprocess_max_memory_mb"],
        64,
        4096,
    )
    out["rlm_subprocess_max_cpu_seconds"] = _clamp_int(
        config.get("rlm_subprocess_max_cpu_seconds"),
        out["rlm_subprocess_max_cpu_seconds"],
        5,
        3600,
    )
    raw_ctx = config.get("context_management")
    if isinstance(raw_ctx, dict):
        out["context_management"] = _deep_merge(
            out.get("context_management") or {}, raw_ctx
        )
    raw_web = config.get("web_search")
    if isinstance(raw_web, dict):
        out["web_search"] = _deep_merge(out.get("web_search") or {}, raw_web)
    raw_sub = config.get("subagents")
    if isinstance(raw_sub, list):
        out["subagents"] = [
            {**({"name": "", "tools": [], "model": None}), **e}
            if isinstance(e, dict)
            else e
            for e in raw_sub
        ]

    # Resolve MCP server configuration
    if mcp_args_env:
        # Legacy: single stdio server from OLLAMACODE_MCP_ARGS "command arg1 arg2" (overrides config)
        parts = mcp_args_env.split()
        out["mcp_servers"] = [
            {
                "type": "stdio",
                "command": parts[0],
                "args": parts[1:] if len(parts) > 1 else [],
            }
        ]
    elif "mcp_servers" in config:
        custom = list(config["mcp_servers"])
        if custom:
            # By default include built-in servers; set include_builtin_servers: false to use only custom
            if config.get("include_builtin_servers", True):
                # Skip a built-in server if config already has an equivalent (same type/command/args)
                def _server_key(entry: dict[str, Any]) -> tuple[Any, ...]:
                    t = entry.get("type")
                    if t == "stdio":
                        return (
                            t,
                            (entry.get("command") or ""),
                            tuple(entry.get("args") or []),
                        )
                    return (t, entry.get("url"))

                custom_keys = {_server_key(c) for c in custom}
                builtin_prepend = [
                    s for s in DEFAULT_MCP_SERVERS if _server_key(s) not in custom_keys
                ]
                if python_executable is not None:
                    builtin_prepend = [
                        {**e, "command": python_executable}
                        if (e.get("type") or "stdio").lower() == "stdio"
                        else e
                        for e in builtin_prepend
                    ]
                out["mcp_servers"] = builtin_prepend + custom
            else:
                out["mcp_servers"] = custom
        else:
            # Empty list means no MCP servers
            out["mcp_servers"] = []
    else:
        # No explicit config and no env: use built‑in defaults
        default_list = list(DEFAULT_MCP_SERVERS)
        if python_executable is not None:
            default_list = [
                {**e, "command": python_executable}
                if (e.get("type") or "stdio").lower() == "stdio"
                else e
                for e in default_list
            ]
        out["mcp_servers"] = default_list

    # Optional: append web_search server when enabled and endpoint set
    web_cfg = out.get("web_search") or {}
    if isinstance(web_cfg, dict) and web_cfg.get("enabled") and web_cfg.get("endpoint"):
        env = dict(os.environ)
        env["OLLAMACODE_WEB_SEARCH_ENDPOINT"] = str(web_cfg.get("endpoint", ""))
        if web_cfg.get("api_key"):
            env["OLLAMACODE_WEB_SEARCH_API_KEY"] = str(web_cfg["api_key"])
        out["mcp_servers"] = list(out["mcp_servers"]) + [
            {
                "type": "stdio",
                "command": python_executable or sys.executable,
                "args": ["-m", "ollamacode.servers.web_search_mcp"],
                "env": env,
            }
        ]

    return out
