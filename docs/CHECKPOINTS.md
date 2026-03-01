# Checkpoints

OllamaCode can record per‑turn checkpoints for file edits so you can rewind.

## How it works

- When a turn begins, OllamaCode records a **checkpoint** of any file the tools edit.
- After the turn, it stores before/after content in `~/.ollamacode/checkpoints.db`.

## TUI usage

- `/checkpoints` — list recent checkpoints for the current session
- `/rewind <id|index> [code|conversation|both]`

Examples:

```
/checkpoints
/rewind 1
/rewind 6 both
/rewind 9b7a1e code
```

## Disable

Set `OLLAMACODE_CHECKPOINTS=0`.

## Notes

- Only file edits done via MCP tools are captured.
- Rewind only affects the workspace root (safety).
