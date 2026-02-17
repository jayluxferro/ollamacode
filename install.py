#!/usr/bin/env python3
"""
Interactive installer for OllamaCode.

- Ensures ~/.ollamacode directory structure
- Optionally creates .ollamacode/config.yaml in current workspace
- Optionally runs uv sync (if pyproject.toml exists)
- Checks Ollama availability and optionally pulls a model
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _prompt_yes_no(prompt: str, default: bool = False, assume_yes: bool = False) -> bool:
    if assume_yes:
        return True
    suffix = " [Y/n] " if default else " [y/N] "
    ans = input(prompt + suffix).strip().lower()
    if not ans:
        return default
    return ans in ("y", "yes")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _run(cmd: list[str]) -> int:
    try:
        return subprocess.call(cmd)
    except OSError:
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Install/bootstrap OllamaCode.")
    parser.add_argument("--yes", action="store_true", help="Assume yes to prompts.")
    parser.add_argument("--workspace", default=".", help="Workspace directory (default: .)")
    parser.add_argument("--no-ollama", action="store_true", help="Skip Ollama checks.")
    parser.add_argument("--no-uv", action="store_true", help="Skip uv sync even if pyproject.toml exists.")
    parser.add_argument("--model", default="", help="Model to pull with Ollama (e.g. gpt-oss:20b).")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    home = Path(os.path.expanduser("~"))
    global_root = home / ".ollamacode"

    print(f"[install] Workspace: {workspace}")
    _ensure_dir(global_root)
    _ensure_dir(global_root / "skills")
    _ensure_dir(global_root / "templates")
    print(f"[install] Ensured {global_root} (skills, templates)")

    # Create workspace config if missing
    ws_cfg_dir = workspace / ".ollamacode"
    ws_cfg_file = ws_cfg_dir / "config.yaml"
    if not ws_cfg_file.exists():
        if _prompt_yes_no("Create .ollamacode/config.yaml in workspace?", default=True, assume_yes=args.yes):
            _ensure_dir(ws_cfg_dir)
            ws_cfg_file.write_text(
                "model: gpt-oss:20b\n"
                "confirm_tool_calls: false\n"
                "prompt_snippets: []\n",
                encoding="utf-8",
            )
            print(f"[install] Wrote {ws_cfg_file}")

    # Run uv sync if pyproject exists
    if not args.no_uv and (workspace / "pyproject.toml").exists():
        if _prompt_yes_no("Run `uv sync` in workspace?", default=True, assume_yes=args.yes):
            code = _run(["uv", "sync"])
            if code != 0:
                print("[install] uv sync failed; you can run it manually.")

    # Check Ollama and optionally pull model
    if not args.no_ollama:
        try:
            from ollamacode.health import check_ollama
        except Exception:
            check_ollama = None
        if check_ollama is not None:
            ok, msg = check_ollama()
            print(f"[install] Ollama: {msg}")
            if ok and args.model:
                if _prompt_yes_no(f"Pull model {args.model} with Ollama?", default=True, assume_yes=args.yes):
                    _run(["ollama", "pull", args.model])
        else:
            print("[install] Ollama check unavailable (package not on PYTHONPATH).")

    print("[install] Done. Next steps: `ollamacode tutorial` or `ollamacode serve`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
