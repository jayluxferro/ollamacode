# OllamaCode roadmap

## Current (1.0.0)

- **Core**: Local Ollama + MCP (CLI, TUI, HTTP API). Built-in MCP: fs, terminal, codebase, tools, git; optional semantic. Parallel tool calls, lenient tool JSON, truncate tool results.
- **Chat**: Streaming, apply-edits (`<<EDITS>>`), @-context, `--file` / `--lines`, slash commands (/fix, /test, /summary). Optional auto-summarize when approaching context limit; lazy MCP (connect on first message in interactive).
- **Safety & config**: run_command blocklist + allowlist; optional confirm-before-run for risky patterns. apply-edits dry run, `max_edits_per_request`. Deduplicate built-in servers; optional serve auth.
- **Editor protocol**: Chat-with-selection and apply-edits over **HTTP** (`ollamacode serve`: POST /chat, POST /chat/stream, POST /apply-edits) and **stdio** (`ollamacode protocol`: JSON-RPC `ollamacode/chat`, `ollamacode/chatStream`, `ollamacode/applyEdits`). See [STRUCTURED_PROTOCOL.md](STRUCTURED_PROTOCOL.md).
- **CI & docs**: Integration tests (optional job when Ollama available), coverage baseline 30%, editor snippets in docs.

## Future

Ideas welcome. See [CHANGELOG.md](../CHANGELOG.md) for release history.
