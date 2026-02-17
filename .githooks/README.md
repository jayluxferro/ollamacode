# Git hooks

## Pre-push (ruff format + fix)

The **pre-push** hook runs `ruff format` and `ruff check --fix` on `ollamacode` and `tests` before allowing a push. If ruff changes any files, the push is aborted so you can review and commit the fixes.

### Install

From the repo root:

```bash
git config core.hooksPath .githooks
chmod +x .githooks/pre-push
```

To remove:

```bash
git config --unset core.hooksPath
```

### Manual run

```bash
.githooks/pre-push
```

Or: `uv run ruff format ollamacode tests && uv run ruff check --fix ollamacode tests`
