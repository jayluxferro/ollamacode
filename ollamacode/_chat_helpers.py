"""Shared helpers for serve.py and protocol_server.py.

Extracted to avoid duplication of system prompt building and memory helpers.
"""

from __future__ import annotations

from typing import Any

from .memory import build_dynamic_memory_context
from .skills import load_skills_text
from .state import (
    format_feedback_context,
    format_knowledge_context,
    format_past_errors_context,
    format_plan_context,
    format_preferences,
    format_recent_context,
    get_state,
)
from .templates import load_prompt_template


def coerce_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    """Coerce a value to int within [min_value, max_value], falling back to default."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(n, max_value))


def resolve_memory_request_settings(
    params: dict[str, Any],
    *,
    default_auto: bool,
    default_kg_max: int,
    default_rag_max: int,
    default_rag_chars: int,
) -> tuple[bool, int, int, int]:
    """Extract memory settings from request params with defaults."""
    auto_raw = params.get("memoryAutoContext", params.get("memory_auto_context"))
    kg_raw = params.get("memoryKgMaxResults", params.get("memory_kg_max_results"))
    rag_raw = params.get("memoryRagMaxResults", params.get("memory_rag_max_results"))
    chars_raw = params.get(
        "memoryRagSnippetChars", params.get("memory_rag_snippet_chars")
    )
    auto = bool(auto_raw) if isinstance(auto_raw, bool) else default_auto
    kg_max = coerce_int(kg_raw, default_kg_max, 0, 20)
    rag_max = coerce_int(rag_raw, default_rag_max, 0, 20)
    rag_chars = coerce_int(chars_raw, default_rag_chars, 40, 2000)
    return auto, kg_max, rag_max, rag_chars


def append_dynamic_memory(
    system: str,
    query: str,
    *,
    memory_auto_context: bool = True,
    memory_kg_max_results: int = 4,
    memory_rag_max_results: int = 4,
    memory_rag_snippet_chars: int = 220,
) -> str:
    """Append per-query retrieved memory context to the system prompt."""
    if not memory_auto_context:
        return system
    block = build_dynamic_memory_context(
        query,
        kg_max_results=memory_kg_max_results,
        rag_max_results=memory_rag_max_results,
        rag_snippet_chars=memory_rag_snippet_chars,
    )
    if not block:
        return system
    return system + "\n\n--- Retrieved memory (query-specific) ---\n\n" + block


_BASE_SYSTEM_PROMPT = (
    "You are a coding assistant with full access to the workspace. You are given a list of available tools with their names "
    "and descriptions\u2014use whichever tools fit the task. When the user asks you to run something, check something, or change "
    "something, use the appropriate tool and report the result. When generating code, include docstrings and brief comments where helpful."
)


def build_system_prompt(
    system_extra: str,
    workspace_root: str | None = None,
    use_skills: bool = True,
    prompt_template: str | None = None,
    inject_recent_context: bool = True,
    recent_context_max_files: int = 10,
    use_reasoning: bool = False,
    prompt_snippets: list[str] | None = None,
    code_style: str | None = None,
) -> str:
    """Build the full system prompt with all context sections."""
    out = (
        (_BASE_SYSTEM_PROMPT + "\n\n" + system_extra)
        if system_extra
        else _BASE_SYSTEM_PROMPT
    )
    if use_skills and workspace_root:
        skills_text = load_skills_text(workspace_root)
        if skills_text:
            out = (
                out
                + "\n\n--- Skills (saved instructions & memory) ---\n\n"
                + skills_text
            )
    if prompt_template and workspace_root:
        template_text = load_prompt_template(prompt_template, workspace_root)
        if template_text:
            out = out + "\n\n--- Prompt template ---\n\n" + template_text
    state = get_state()
    if inject_recent_context:
        block = format_recent_context(state, max_files=recent_context_max_files)
        if block:
            out = out + "\n\n--- Recent context ---\n\n" + block
    prefs_block = format_preferences(state)
    if prefs_block:
        out = out + "\n\n--- User preferences ---\n\n" + prefs_block
    plan_block = format_plan_context(state)
    if plan_block:
        out = out + "\n\n--- Plan (use /continue to work on it) ---\n\n" + plan_block
    feedback_block = format_feedback_context(state)
    if feedback_block:
        out = out + "\n\n--- Recent feedback ---\n\n" + feedback_block
    knowledge_block = format_knowledge_context(state)
    if knowledge_block:
        out = out + "\n\n--- " + knowledge_block
    past_errors_block = format_past_errors_context(state, max_entries=5)
    if past_errors_block:
        out = out + "\n\n--- " + past_errors_block
    if use_reasoning:
        out = (
            out
            + "\n\nWhen answering, you may include a brief reasoning or rationale before your conclusion; for code changes, briefly explain the fix."
            + '\n\nOptionally output <<REASONING>>\n{"steps": ["..."], "conclusion": "..."}\n<<END>> or call record_reasoning(steps, conclusion).'
        )
    for snip in prompt_snippets or []:
        if snip and isinstance(snip, str) and snip.strip():
            out = out + "\n\n" + snip.strip()
    if code_style:
        out = (
            out
            + "\n\n--- Code style (follow when generating code) ---\n\n"
            + code_style.strip()
        )
    return out
