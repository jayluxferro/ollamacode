# Changelog

All notable changes to OllamaCode are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2025-02-08

### Added

- **Core**: Local Ollama + MCP tool calling. Agent loop (`run_agent_loop`, `run_agent_loop_stream`), MCP client (stdio, SSE, streamable HTTP), bridge from MCP tools to Ollama tool format.
- **CLI**: `ollamacode` entrypoint; single query or interactive chat; `--stream`, `--tui`, `--apply-edits`, `--no-mcp`; env and config for model, MCP, system prompt.
- **Config**: Optional `ollamacode.yaml` or `.ollamacode/config.yaml` for `model`, `mcp_servers`, `max_messages`, `max_tool_rounds`, `max_tool_result_chars`, `max_edits_per_request`, `include_builtin_servers`, `serve.api_key`, etc. Config merge with env overrides; deduplicate built-in servers when merging to avoid duplicate tool names.
- **Built-in MCP** (default when no config): fs (read_file, write_file, list_dir), terminal (run_command), codebase (search_codebase, get_relevant_files), tools (run_linter, run_tests, install_deps), git (git_status, git_diff_*, git_log, git_log_graph, git_show, git_branch, git_add, git_commit, git_push, git_stash, git_checkout, git_merge, git_branch_delete), skills (list_skills, read_skill, write_skill, save_memory), state (get_state, update_state, append_recent_file, clear_state). Optional semantic server (index_codebase, semantic_search_codebase) via config.
- **Streaming**: `--stream` and TUI stream tokens; POST `/chat/stream` returns SSE for other editors.
- **Apply-edits**: `<<EDITS>>` JSON in model output; `--apply-edits` to show diff and prompt; `--apply-edits-dry-run`; `max_edits_per_request` to cap edits per run.
- **Context**: @-style file/folder refs, `--file` / `--lines`, `rules_file`, branch/PR context, `/summary` to collapse last N turns. **Context injection**: when building the system prompt (CLI, TUI, serve, protocol), optionally append "Recent files" from `~/.ollamacode/state.json` and a one-line branch/last-commit summary. Config `inject_recent_context` (default true), `recent_context_max_files` (default 10).
- **Slash commands**: /help, /clear, /model, /fix, /test, /docs, /profile, /reset-state, /summary, /quit (CLI and TUI).
- **HTTP API**: `ollamacode serve`; POST /chat and POST /chat/stream; optional auth via `serve.api_key` or `OLLAMACODE_SERVE_API_KEY`.
- **Tool-call robustness**: Lenient parsing of tool-call JSON (extra `}`, `]`, unescaped newlines); friendly error when Ollama returns 500 for malformed tool JSON.
- **Parallel tool calls**: Tools in one turn run in parallel.
- **Truncate tool results**: `max_tool_result_chars` to limit tool output in context.
- **run_command guardrails**: `OLLAMACODE_BLOCK_DANGEROUS_COMMANDS` blocklist; `OLLAMACODE_ALLOWED_COMMANDS` allowlist. Optional command history log: `OLLAMACODE_LOG_COMMANDS=1` appends (timestamp, cwd, command, return_code) to `OLLAMACODE_COMMAND_LOG` or `~/.ollamacode/command_history.log`.
- **Structured logging**: `OLLAMACODE_JSON_LOGS=1` emits JSON-lines for ollama/tools/turn timing.
- **Coverage**: pytest-cov in dev deps; CI runs unit tests with coverage (fail_under=0).
- **Docs**: README, docs/ROADMAP.md, docs/OTHER_EDITORS.md (with Neovim/Zed/Sublime snippets), docs/MCP_SERVERS.md.
- **Memory & skills**: Skills from `~/.ollamacode/skills` and `.ollamacode/skills` loaded into system prompt; MCP tools to read/write skills and save_memory. Config `use_skills`.
- **Git**: `git_log_graph` tool (ASCII graph of branches/commits).
- **Error formatting**: On tool failure, print a short "What failed" + "Next step" hint for common errors (file not found, permission, timeout, module not found, indentation, TypeError, connection refused, port in use, OOM, JSON decode, EISDIR, etc.).
- **Prompt templates**: Config `prompt_template: <name>` loads Markdown from `~/.ollamacode/templates/<name>.md` or `.ollamacode/templates/<name>.md` and appends to system prompt (CLI, TUI, serve, protocol).
- **Init templates**: `ollamacode init` (list templates) and `ollamacode init --template python-cli|python-lib|web-app|rust-cli|go-mod [--dest DIR]` to scaffold projects.
- **Dependency install**: `install_deps` tool in tools MCP (uv sync or `uv pip install -r requirements.txt`).
- **Docs/profiling**: TUI slash commands `/docs` (run docs build, send output to model) and `/profile` (run profiler, send summary); config `docs_command`, `profile_command`.
- **State persistence**: `~/.ollamacode/state.json` for recent files and preferences. State MCP: `get_state`, `update_state`, `append_recent_file`, `clear_state`. TUI `/reset-state` to clear.
- **Interactive tutorial**: `ollamacode tutorial` runs a short wizard in a temp dir (uv sync, pytest, list dir) and prints next steps.
- **TUI (Windows)**: Optional `prompt_toolkit` in `ollamacode[tui]` for arrow-key line editing when readline is not available.
- **IDE diagnostics & completions**: `ollamacode/diagnostics` (stdio) and POST `/diagnostics` (HTTP) run linter and return LSP-like diagnostics; `ollamacode/complete` and POST `/complete` return inline completion from Ollama generate. See docs/STRUCTURED_PROTOCOL.md.
- **Unified code quality**: `run_code_quality` MCP tool runs a configurable list of commands (default: ruff, black, isort, mypy), aggregates stdout/stderr into one report. Env `OLLAMACODE_CODE_QUALITY_COMMANDS` for custom comma-separated commands.
- **Coverage & test suggestion**: `run_coverage` MCP tool runs pytest --cov (or custom command), parses term-missing output, returns report plus uncovered files and suggested test descriptions.
- **Health check**: `ollamacode health` CLI checks Ollama reachability; GET `/health` when serving returns `{ "ollama": true/false, "message": "..." }`.
- **Toolchain registry**: docs/TOOLCHAIN_REGISTRY.md with curated tools (Python, JS, etc.) and usage with built-in run_linter/run_tests/run_code_quality/run_coverage.
- **VS Code extension**: minimal extension in `editors/vscode` (Chat, Chat with selection, Apply edits) using HTTP API; configurable baseUrl and apiKey; see editors/vscode/README.md and docs/OTHER_EDITORS.md. **Streaming:** commands "Chat (streaming)" and "Chat with selection (streaming)" use POST /chat/stream and show reply token-by-token in the output channel.
- **Neovim plugin**: minimal plugin in `editors/neovim` with `:OllamaCode` and `:OllamaCodeSelection`, config via `require("ollamacode").setup()`; reply in floating window and optional apply edits; see editors/neovim/README.md.

[1.0.0]: https://github.com/your-org/ollamacode/releases/tag/v1.0.0
