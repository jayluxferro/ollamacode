from __future__ import annotations

from ollamacode.question_runtime import format_question_answers, normalize_question_list


def test_normalize_question_list_supports_option_objects():
    questions = normalize_question_list(
        {
            "questions": [
                {
                    "header": "Mode",
                    "question": "Which path should I use?",
                    "options": [
                        {"label": "src/app.py", "description": "Edit the app"},
                        {"label": "src/cli.py", "description": "Edit the cli"},
                    ],
                }
            ]
        }
    )
    assert questions == [
        {
            "header": "Mode",
            "question": "Which path should I use?",
            "options": ["src/app.py", "src/cli.py"],
        }
    ]


def test_format_question_answers_marks_unanswered():
    text = format_question_answers(
        [{"question": "Proceed?", "header": "Confirm", "options": ["Yes", "No"]}],
        [""],
    )
    assert "Unanswered" in text
