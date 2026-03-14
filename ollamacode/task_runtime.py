"""Helpers for interactive subagent task delegation."""

from __future__ import annotations

from typing import Any

from .agent import run_agent_loop, run_agent_loop_no_mcp


def find_subagent(
    subagents: list[dict[str, Any]] | None,
    subagent_type: str,
) -> dict[str, Any] | None:
    """Return the configured subagent matching *subagent_type*."""
    stype = (subagent_type or "").strip().lower()
    if not stype:
        return None
    for item in subagents or []:
        if not isinstance(item, dict):
            continue
        if (item.get("name") or "").strip().lower() == stype:
            return item
    return None


async def run_task_delegation(
    *,
    session: Any | None,
    session_id: str | None,
    workspace_root: str,
    subagents: list[dict[str, Any]] | None,
    arguments: dict[str, Any],
    default_model: str,
    system_prompt: str,
    max_messages: int,
    max_tool_rounds: int,
    max_tool_result_chars: int,
    provider: Any = None,
    quiet: bool = True,
    timing: bool = False,
    before_tool_call: Any = None,
) -> str:
    """Run a delegated subagent task and return a synthetic tool result."""
    subagent_type = str(arguments.get("subagent_type") or "").strip()
    prompt = str(arguments.get("prompt") or "").strip()
    description = (
        str(arguments.get("description") or "").strip() or subagent_type or "task"
    )
    if not subagent_type or not prompt:
        return (
            "Task tool requires both `subagent_type` and `prompt`. "
            "Use `description` for a short label."
        )
    match = find_subagent(subagents, subagent_type)
    if match is None:
        available = ", ".join(
            sorted(
                (str(item.get("name") or "").strip() for item in subagents or [] if isinstance(item, dict) and item.get("name"))
            )
        )
        if not available:
            return "No subagents are configured. Add `subagents` to the OllamaCode config first."
        return f"Unknown subagent type: {subagent_type}. Available subagents: {available}."

    tools = [str(item) for item in (match.get("tools") or []) if str(item).strip()]
    blocked_tools = ["task", "todoread", "todowrite"]
    sub_model = (match.get("model") or "").strip() or default_model

    child_session_id = str(arguments.get("task_id") or "").strip() or None
    child_history: list[dict[str, Any]] = []
    title = f"{description} (@{subagent_type} subagent)"
    try:
        from .sessions import create_session, get_session_info, load_session, save_session

        if child_session_id:
            loaded = load_session(child_session_id)
            if loaded is not None:
                child_history = loaded
            else:
                child_session_id = None
        if not child_session_id:
            parent_info = get_session_info(session_id) if session_id else None
            child_session_id = create_session(
                title=title,
                workspace_root=workspace_root,
                parent_session_id=session_id,
                owner=parent_info.get("owner") if parent_info else None,
                role="delegate",
            )
    except Exception:
        child_session_id = None

    if session is not None:
        out = await run_agent_loop(
            session,
            sub_model,
            prompt,
            system_prompt=system_prompt,
            message_history=child_history or None,
            max_messages=max_messages,
            max_tool_rounds=max_tool_rounds,
            max_tool_result_chars=max_tool_result_chars,
            allowed_tools=tools if tools else None,
            blocked_tools=blocked_tools,
            confirm_tool_calls=before_tool_call is not None,
            before_tool_call=before_tool_call,
            provider=provider,
            quiet=quiet,
            timing=timing,
        )
    else:
        out = await run_agent_loop_no_mcp(
            sub_model,
            prompt,
            system_prompt=system_prompt,
            message_history=child_history or None,
            provider=provider,
            timing=timing,
        )

    if child_session_id:
        try:
            from .sessions import save_session

            save_session(
                child_session_id,
                title,
                child_history
                + [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": out},
                ],
                workspace_root=workspace_root,
            )
        except Exception:
            pass

    return "\n".join(
        [
            f"task_id: {child_session_id or 'ephemeral'}",
            "",
            "<task_result>",
            out,
            "</task_result>",
        ]
    )
