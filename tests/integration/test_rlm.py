"""Integration test: CLI RLM mode. Requires Ollama. Skips if Ollama not available."""

import subprocess
import sys
from pathlib import Path

import pytest

from ollamacode.health import check_ollama

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _run_rlm_cli(prompt: str, timeout: int = 120) -> tuple[str, str, int]:
    """Run ollamacode --rlm with the given prompt; return (stdout, stderr, returncode)."""
    cmd = [
        sys.executable,
        "-m",
        "ollamacode",
        "--rlm",
        prompt,
    ]
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout, result.stderr, result.returncode


# Known RLM failure modes when the model doesn't cooperate (empty reply, FINAL_VAR without REPL, Ollama 500).
_RLM_MODEL_FAILURE_MARKERS = (
    "Model returned empty response",
    "FINAL_VAR(",
    "but no REPL block was run",
    "500 Internal Server Error",
)


@pytest.mark.integration
@pytest.mark.skipif(not check_ollama()[0], reason="Ollama not running")
def test_rlm_cli_returns_final_or_answer():
    """RLM mode: run with a tiny prompt; expect exit 0 and sensible output, or exit 1 only for known model-side failures."""
    stdout, stderr, code = _run_rlm_cli("Reply with only the word OK and nothing else.")
    combined = stdout + stderr
    assert "Traceback" not in stdout, f"Unexpected traceback: {stdout!r}"

    if code == 0:
        assert stdout.strip(), (
            f"Expected non-empty stdout. stdout={stdout!r} stderr={stderr!r}"
        )
        assert "OK" in stdout or "FINAL" in stdout or len(stdout.strip()) > 2, (
            f"Expected answer or FINAL in output. stdout={stdout!r}"
        )
    else:
        # Allow exit 1 when failure is due to model/Ollama (empty response, FINAL_VAR, 500)
        assert any(m in combined for m in _RLM_MODEL_FAILURE_MARKERS), (
            f"CLI failed with unexpected error: stderr={stderr!r} stdout={stdout!r}"
        )
