"""
Multi-agent coordinator: planner → executor → reviewer.

Planner and reviewer run without tools; executor can use tools.
Reviewer returns a structured decision in <<REVIEW_DECISION>> JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .agent import BeforeToolCallCB, run_agent_loop, run_agent_loop_no_mcp

REVIEW_DECISION_START = "<<REVIEW_DECISION>>"
REVIEW_DECISION_END = "<<END>>"


@dataclass
class MultiAgentResult:
    content: str
    plan: str | None = None
    review: dict[str, Any] | None = None
    synthesis: str | None = None


def _parse_review_decision(text: str) -> dict[str, Any] | None:
    start = text.find(REVIEW_DECISION_START)
    if start == -1:
        return None
    after_start = (
        text.index("\n", start)
        if "\n" in text[start:]
        else start + len(REVIEW_DECISION_START)
    )
    end_marker = text.find(REVIEW_DECISION_END, after_start)
    if end_marker == -1:
        return None
    raw = text[after_start:end_marker].strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _planner_system(base_system: str) -> str:
    return (
        base_system
        + "\n\nYou are a planner. Produce a concise step-by-step plan with 3-8 steps. "
        "Do not call tools. Do not modify code. Return the plan only."
    )


def _reviewer_system(base_system: str) -> str:
    return (
        base_system
        + "\n\nYou are a reviewer. Review the executor output for correctness, safety, and completeness. "
        "Do not call tools. Return a structured decision block only:\n"
        "<<REVIEW_DECISION>>\n"
        '{"approved": true|false, "issues": ["..."], "suggestions": ["..."]}\n'
        "<<END>>"
    )


async def run_multi_agent(
    session: Any | None,
    model: str,
    message: str,
    *,
    system_prompt: str,
    max_messages: int,
    max_tool_result_chars: int,
    allowed_tools: list[str] | None,
    blocked_tools: list[str] | None,
    confirm_tool_calls: bool,
    before_tool_call: BeforeToolCallCB | None,
    planner_model: str | None = None,
    executor_model: str | None = None,
    reviewer_model: str | None = None,
    synthesis_model: str | None = None,
    max_iterations: int = 2,
    require_review: bool = True,
    synthesize: bool = True,
) -> MultiAgentResult:
    """Run planner → executor → reviewer loop. Returns final content + plan + review decision."""
    planner_model = planner_model or model
    executor_model = executor_model or model
    reviewer_model = reviewer_model or model
    synthesis_model = synthesis_model or model

    # 1) Plan
    if session is not None:
        plan = await run_agent_loop(
            session,
            planner_model,
            message,
            system_prompt=_planner_system(system_prompt),
            max_messages=max_messages,
            max_tool_result_chars=max_tool_result_chars,
            allowed_tools=[],  # no tools for planner
            blocked_tools=None,
            confirm_tool_calls=False,
            disallow_tools=True,
        )
    else:
        plan = await run_agent_loop_no_mcp(
            planner_model,
            message,
            system_prompt=_planner_system(system_prompt),
        )

    # 2) Execute (may use tools)
    executor_prompt = (
        f"Goal:\n{message}\n\n"
        f"Plan:\n{plan}\n\n"
        "Execute the plan step-by-step. Use tools if needed. Provide final answer and <<EDITS>> for code changes."
    )

    async def _run_executor(extra_feedback: str | None = None) -> str:
        prompt = executor_prompt
        if extra_feedback:
            prompt = prompt + "\n\nReviewer feedback:\n" + extra_feedback
        if session is not None:
            return await run_agent_loop(
                session,
                executor_model,
                prompt,
                system_prompt=system_prompt,
                max_messages=max_messages,
                max_tool_result_chars=max_tool_result_chars,
                allowed_tools=allowed_tools,
                blocked_tools=blocked_tools,
                confirm_tool_calls=confirm_tool_calls,
                before_tool_call=before_tool_call,
            )
        return await run_agent_loop_no_mcp(
            executor_model,
            prompt,
            system_prompt=system_prompt,
        )

    content = await _run_executor()

    # 3) Review (optional loop)
    review_decision: dict[str, Any] | None = None
    if require_review:
        for _ in range(max(1, max_iterations)):
            review_prompt = (
                f"Goal:\n{message}\n\nPlan:\n{plan}\n\nExecutor output:\n{content}\n"
            )
            if session is not None:
                review_text = await run_agent_loop(
                    session,
                    reviewer_model,
                    review_prompt,
                    system_prompt=_reviewer_system(system_prompt),
                    max_messages=max_messages,
                    max_tool_result_chars=max_tool_result_chars,
                    allowed_tools=[],
                    blocked_tools=None,
                    confirm_tool_calls=False,
                    disallow_tools=True,
                )
            else:
                review_text = await run_agent_loop_no_mcp(
                    reviewer_model,
                    review_prompt,
                    system_prompt=_reviewer_system(system_prompt),
                )
            review_decision = _parse_review_decision(review_text)
            if review_decision and review_decision.get("approved") is True:
                break
            feedback = ""
            if review_decision:
                issues = review_decision.get("issues") or []
                suggestions = review_decision.get("suggestions") or []
                if isinstance(issues, list):
                    feedback += "\n".join(f"- {x}" for x in issues if x)
                if isinstance(suggestions, list):
                    feedback += "\n" + "\n".join(f"- {x}" for x in suggestions if x)
            else:
                feedback = "Reviewer did not return a valid decision block. Please revise the output."
            content = await _run_executor(feedback.strip())

    synthesis_text: str | None = None
    if synthesize:
        synth_prompt = (
            "Synthesize the plan and executor output into a single final answer. "
            "Explicitly reconcile conflicts. Respond in Markdown with sections:\n"
            "## Final Answer\n"
            "## Conflicts Resolved (use 'None' if no conflicts)\n\n"
            f"Goal:\n{message}\n\nPlan:\n{plan}\n\nExecutor output:\n{content}\n"
        )
        if review_decision:
            synth_prompt += f"\n\nReview decision:\n{json.dumps(review_decision)}"
        if session is not None:
            synthesis_text = await run_agent_loop(
                session,
                synthesis_model,
                synth_prompt,
                system_prompt=system_prompt,
                max_messages=max_messages,
                max_tool_result_chars=max_tool_result_chars,
                allowed_tools=[],
                blocked_tools=None,
                confirm_tool_calls=False,
                disallow_tools=True,
            )
        else:
            synthesis_text = await run_agent_loop_no_mcp(
                synthesis_model,
                synth_prompt,
                system_prompt=system_prompt,
            )
        synthesis_text = (synthesis_text or "").strip()
        content = synthesis_text or content

    return MultiAgentResult(
        content=content, plan=plan, review=review_decision, synthesis=synthesis_text
    )
