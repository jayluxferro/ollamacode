"""
LLM-based context compaction: summarize older messages to stay within token budgets.

The N most recent messages are kept verbatim; everything before them is condensed
into a single assistant summary message by calling the same provider/model.

Usage:
    compacted = await compact_messages(messages, model, provider, max_tokens=4000)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Default number of recent messages to keep intact (system messages are always kept).
_DEFAULT_KEEP_RECENT = 6

# Prompt sent to the model to produce the summary.
_COMPACTION_PROMPT = (
    "You are a context compaction assistant. Summarize the following conversation "
    "history into a concise but complete summary that preserves all important context, "
    "decisions, file paths, code changes, and action items. Keep technical details accurate. "
    "Respond with ONLY the summary, no preamble."
)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English/code."""
    return max(1, len(text) // 4)


def _split_messages(
    messages: list[dict[str, Any]],
    keep_recent: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split messages into (system, to_compact, to_keep).

    System messages (role=system) at the start are always preserved.
    The last *keep_recent* non-system messages are kept verbatim.
    Everything in between is compacted.
    """
    system: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system" and not rest:
            system.append(msg)
        else:
            rest.append(msg)

    if len(rest) <= keep_recent:
        return system, [], rest

    to_compact = rest[: len(rest) - keep_recent]
    to_keep = rest[len(rest) - keep_recent :]
    return system, to_compact, to_keep


def _format_for_summary(messages: list[dict[str, Any]]) -> str:
    """Render messages as a flat conversation transcript for the summarizer."""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        # Truncate very long individual messages to keep the summary prompt manageable.
        if len(content) > 8000:
            content = content[:8000] + "\n... [truncated]"
        lines.append(f"[{role}]: {content}")
    return "\n\n".join(lines)


async def compact_messages(
    messages: list[dict[str, Any]],
    model: str,
    provider: Any,
    *,
    max_tokens: int = 0,
    keep_recent: int = _DEFAULT_KEEP_RECENT,
) -> list[dict[str, Any]]:
    """Return a compacted message list.

    If *max_tokens* > 0, compaction is triggered only when the estimated total
    tokens exceed the budget.  If *max_tokens* is 0, compaction is always
    performed when there are enough messages to compact.

    *provider* must implement ``chat_async(model, messages, tools=None)``.

    Returns the original list unchanged if compaction is not needed or fails.
    """
    if len(messages) <= keep_recent + 1:
        return messages

    system, to_compact, to_keep = _split_messages(messages, keep_recent)

    if not to_compact:
        return messages

    # Check token budget
    if max_tokens > 0:
        total_tokens = sum(_estimate_tokens(m.get("content", "")) for m in messages)
        if total_tokens <= max_tokens:
            return messages

    transcript = _format_for_summary(to_compact)
    if not transcript.strip():
        return messages

    summary_messages = [
        {"role": "system", "content": _COMPACTION_PROMPT},
        {"role": "user", "content": transcript},
    ]

    try:
        resp = await provider.chat_async(model, summary_messages, tools=None)
        summary_text = (resp.get("message") or {}).get("content", "").strip()
    except Exception as exc:
        logger.warning("Context compaction failed (model call): %s", exc)
        return messages

    if not summary_text:
        logger.warning(
            "Context compaction returned empty summary; keeping original messages"
        )
        return messages

    summary_msg: dict[str, Any] = {
        "role": "assistant",
        "content": ("[Context summary of earlier conversation]\n\n" + summary_text),
    }

    compacted = system + [summary_msg] + to_keep
    old_tokens = sum(_estimate_tokens(m.get("content", "")) for m in messages)
    new_tokens = sum(_estimate_tokens(m.get("content", "")) for m in compacted)
    logger.info(
        "Compacted %d messages -> %d messages (~%d -> ~%d tokens)",
        len(messages),
        len(compacted),
        old_tokens,
        new_tokens,
    )
    return compacted
