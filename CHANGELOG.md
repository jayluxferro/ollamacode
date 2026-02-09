# Changelog

All notable changes to OllamaCode are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2025-02-08

### Added

- **Core**: Local Ollama + MCP tool calling. Agent loop (`run_agent_loop`, `run_agent_loop_stream`), MCP client (stdio, SSE, streamable HTTP), bridge from MCP tools to Ollama tool format.
- **CLI**: `ollamacode` entrypoint; single query or interactive chat; `--stream`, `--tui`, `--apply-edits`, `--no-mcp`; env and config for model, MCP, system prompt.
- **Config**: Optional `ollamacode.yaml` or `.ollamacode/config.yaml` for `model`, `mcp_servers`, `max_messages`, `max_tool_rounds`, `max_tool_result_chars`, `max_edits_per_request`, `include_builtin_servers`, `serve.api_key`, etc. Config merge with env overrides; deduplicate built-in servers when merging to avoid duplicate tool names.
- **Built-in MCP** (default when no config): fs (read_file, write_file, list_dir), terminal (run_command), codebase (search_codebase, get_relevant_files), tools (run_linter, run_tests), git (git_status, git_diff_*, git_log, git_show, git_branch, git_add, git_commit, git_push). Optional semantic server (index_codebase, semantic_search_codebase) via config.
- **Streaming**: `--stream` and TUI stream tokens; POST `/chat/stream` returns SSE for other editors.
- **Apply-edits**: `<<EDITS>>` JSON in model output; `--apply-edits` to show diff and prompt; `--apply-edits-dry-run`; `max_edits_per_request` to cap edits per run.
- **Context**: @-style file/folder refs, `--file` / `--lines`, `rules_file`, branch/PR context, `/summary` to collapse last N turns.
- **Slash commands**: /help, /clear, /model, /fix, /test, /summary, /quit (CLI and TUI).
- **HTTP API**: `ollamacode serve`; POST /chat and POST /chat/stream; optional auth via `serve.api_key` or `OLLAMACODE_SERVE_API_KEY`.
- **Tool-call robustness**: Lenient parsing of tool-call JSON (extra `}`, `]`, unescaped newlines); friendly error when Ollama returns 500 for malformed tool JSON.
- **Parallel tool calls**: Tools in one turn run in parallel.
- **Truncate tool results**: `max_tool_result_chars` to limit tool output in context.
- **run_command guardrails**: `OLLAMACODE_BLOCK_DANGEROUS_COMMANDS` blocklist; `OLLAMACODE_ALLOWED_COMMANDS` allowlist.
- **Structured logging**: `OLLAMACODE_JSON_LOGS=1` emits JSON-lines for ollama/tools/turn timing.
- **Coverage**: pytest-cov in dev deps; CI runs unit tests with coverage (fail_under=0).
- **Docs**: README, docs/ROADMAP.md, docs/OTHER_EDITORS.md (with Neovim/Zed/Sublime snippets), docs/MCP_SERVERS.md.

[1.0.0]: https://github.com/your-org/ollamacode/releases/tag/v1.0.0
