"""
Interactive tutorial: run sample commands in a sandbox to show common workflows.

Run with: ollamacode tutorial
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


def run_tutorial() -> None:
    """Run a short wizard: create temp project, run install/tests/list_dir, print next steps."""
    print(
        "OllamaCode tutorial – we'll run a few commands in a temporary directory.\n",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="ollamacode_tutorial_") as tmp:
        root = Path(tmp)
        # Minimal pyproject + script + test
        (root / "pyproject.toml").write_text(
            '[project]\nname = "tutorial"\nversion = "0.1.0"\nrequires-python = ">=3.11"\ndependencies = []\n\n[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n\n[tool.hatch.build.targets.wheel]\npackages = ["src/tutorial"]\n',
            encoding="utf-8",
        )
        (root / "src").mkdir()
        (root / "src/tutorial").mkdir()
        (root / "src/tutorial/__init__.py").write_text(
            '__version__ = "0.1.0"\n', encoding="utf-8"
        )
        (root / "tests").mkdir()
        (root / "tests/test_tutorial.py").write_text(
            'from tutorial import __version__\n\ndef test_version():\n    assert __version__ == "0.1.0"\n',
            encoding="utf-8",
        )
        env = {**os.environ, "OLLAMACODE_FS_ROOT": str(root)}

        def run(cmd: str, desc: str) -> bool:
            print(f"  → {desc} ... ", end="", flush=True)
            try:
                r = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env=env,
                )
                if r.returncode == 0:
                    print("ok", flush=True)
                    return True
                print(f"exit {r.returncode}", flush=True)
                if r.stderr:
                    for line in (r.stderr or "").strip().splitlines()[:3]:
                        print(f"    {line}", flush=True)
                return False
            except Exception as e:
                print(f"error: {e}", flush=True)
                return False

        print("Step 1: Install dependencies (uv sync)")
        run("uv sync", "uv sync")
        print()
        print("Step 2: Run tests (pytest)")
        run("uv run pytest tests/ -q", "pytest")
        print()
        print("Step 3: List directory (list_dir)")
        run("ls -la", "ls")
        print()

    print("Done. Next steps:")
    print("  • Run the TUI: ollamacode --tui")
    print("  • Use /fix to run the linter and send errors to the model")
    print("  • Use /test to run tests and send failures to the model")
    print("  • Ask: ollamacode 'add a function that returns 2+2'")
    print("  • See docs: https://github.com/your-org/ollamacode#readme", flush=True)
