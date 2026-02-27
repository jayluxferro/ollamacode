# Contributing to OllamaCode

Thanks for your interest in contributing. This document explains how to set up the repo, run tests, open pull requests, and follow the project’s coding style.

## Setting up the repo

1. **Clone and enter the repo**
   ```bash
   git clone https://github.com/jayluxferro/ollamacode.git
   cd ollamacode
   ```

2. **Install dependencies** (Python 3.11+)
   ```bash
   uv sync
   ```
   Or with pip: `pip install -e ".[dev]"`

3. **Optional: pre-push hook** (runs ruff format and fix before `git push`)
   ```bash
   git config core.hooksPath .githooks
   chmod +x .githooks/pre-push
   ```
   See `.githooks/README.md` in the repo root. To skip the hook: `git push --no-verify`.

## Running tests

- **Unit tests (no Ollama):**
  ```bash
  uv run pytest tests/ -v -m "not integration"
  ```
- **With coverage:**
  ```bash
  uv run pytest tests/ -v -m "not integration" --cov=ollamacode --cov-report=term-missing
  ```
- **Integration tests** (require Ollama running and a model pulled):
  ```bash
  uv run pytest tests/ -v -m integration
  ```

## Building the distribution

To build wheel and sdist for manual install:

```bash
uv build
```

Output goes to `dist/` (e.g. `ollamacode-1.0.0-py3-none-any.whl`, `ollamacode-1.0.0.tar.gz`). Install with:

```bash
pip install dist/ollamacode-1.0.0-py3-none-any.whl
# or
pip install dist/ollamacode-1.0.0.tar.gz
```

Optional extras: `pip install dist/ollamacode-1.0.0-py3-none-any.whl[tui]` for the TUI. Requires [build](https://pypi.org/project/build/) if not using uv (`pip install build && python -m build`).

## Linting and formatting

- **Ruff (lint + format):**
  ```bash
  uv run ruff check ollamacode tests
  uv run ruff format ollamacode tests
  ```
  To auto-fix fixable issues: `uv run ruff check --fix ollamacode tests`

- **Type checking:**
  ```bash
  uv run pyright ollamacode
  ```

We use **ruff** for style and lint. Please run `ruff format` and `ruff check --fix` (or use the pre-push hook) before pushing so CI stays green.

## Opening a pull request

1. Create a branch from `main` (or `master`): `git checkout -b your-feature`.
2. Make your changes; add or update tests as needed.
3. Run tests and ruff (see above).
4. Commit with a clear message; push and open a PR against `main`/`master`.
5. Fill in the PR template if present. Describe what changed and why.

PRs are reviewed for correctness, tests, and consistency with the codebase. Once approved, a maintainer will merge.

## Coding style

- **Formatting:** Ruff (line length 128, target Python 3.11). Run `ruff format` before pushing.
- **Lint:** Ruff rules E, F, I, W. Fix reported issues or add `# noqa` with a short comment only when necessary.
- **Imports:** Group stdlib, third-party, then local; sort with ruff.
- **Tests:** Prefer `tests/unit/` for fast tests; use `@pytest.mark.integration` for tests that need Ollama or external services.

## Documentation

- User-facing docs: `README.md` (repo root), [Wiki](WIKI.md).
- API: [API reference](api.md). Build with `uv run mkdocs build` (requires doc deps: `uv sync --group docs` or `uv sync --extra docs`).

If you add a feature, update the README or relevant doc when it’s user-visible.

## Questions

Open a [GitHub Discussion](https://github.com/jayluxferro/ollamacode/discussions) or an [Issue](https://github.com/jayluxferro/ollamacode/issues) for questions or ideas.
