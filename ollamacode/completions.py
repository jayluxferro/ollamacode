"""
IDE completions: return inline completion for a prefix (Ollama generate).

Used by ollamacode/complete protocol and POST /complete for editor ghost-text.
"""

from __future__ import annotations

import ollama


def get_completion(prefix: str, model: str, max_tokens: int = 60) -> str:
    """
    Get a short completion for the given prefix using Ollama generate.
    Returns the generated text (single suggestion) or empty string on error.
    """
    if not (model or "").strip():
        return ""
    try:
        r = ollama.generate(
            model=model,
            prompt=prefix,
            options={"num_predict": max_tokens},
        )
        text = (r.response or "").strip()
        # Prefer a single line for inline completion
        if "\n" in text:
            text = text.split("\n")[0].strip()
        return text[:500]
    except Exception:
        return ""
