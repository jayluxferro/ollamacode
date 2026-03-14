"""Helpers for the interactive `question` tool."""

from __future__ import annotations

from typing import Any


def normalize_question_list(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize tool arguments into a list of question dicts."""
    raw = arguments.get("questions")
    if not isinstance(raw, list):
        single = arguments.get("question")
        if isinstance(single, str) and single.strip():
            raw = [
                {
                    "question": single,
                    "header": arguments.get("header") or "Question",
                    "options": arguments.get("options") or [],
                }
            ]
        else:
            return []

    questions: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        if not question:
            continue
        options_raw = item.get("options")
        options: list[str] = []
        if isinstance(options_raw, list):
            for option in options_raw:
                if isinstance(option, dict):
                    label = str(option.get("label") or "").strip()
                    if label:
                        options.append(label)
                elif isinstance(option, str) and option.strip():
                    options.append(option.strip())
        questions.append(
            {
                "header": str(item.get("header") or "Question").strip() or "Question",
                "question": question,
                "options": options,
            }
        )
    return questions


def format_question_answers(
    questions: list[dict[str, Any]],
    answers: list[str],
) -> str:
    """Format collected answers into a synthetic tool result."""
    formatted: list[str] = []
    for idx, question in enumerate(questions):
        answer = answers[idx].strip() if idx < len(answers) and answers[idx] else ""
        rendered = answer if answer else "Unanswered"
        formatted.append(f'"{question["question"]}"="{rendered}"')
    summary = ", ".join(formatted) if formatted else "No answers were provided."
    return (
        "User has answered your questions: "
        + summary
        + ". You can now continue with the user's answers in mind."
    )
