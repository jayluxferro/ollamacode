"""
Health check: verify Ollama and optionally MCP availability.

Used by `ollamacode health` CLI and GET /health when serving.
"""

from __future__ import annotations


def check_ollama() -> tuple[bool, str]:
    """
    Check if Ollama is reachable. Returns (success, message).
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
