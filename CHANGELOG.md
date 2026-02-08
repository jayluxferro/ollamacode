# Changelog

All notable changes to OllamaCode are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added (built-in MCP and full Git opt-in)

- **Built-in MCP by default**: When no config and no `OLLAMACODE_MCP_ARGS`, OllamaCode starts four built-in servers: **ollamacode-fs** (read_file, write_file, list_dir), **ollamacode-terminal** (run_command), **ollamacode-codebase** (search_codebase, get_relevant_files), **ollamacode-git** (read-only: git_status, git_diff_*, git_log, git_show, git_branch). Shipped in `ollamacode.servers`; use config `mcp_servers: []` to disable MCP.
- **Full Git opt-in**: README section “Opt-in: full Git (commit, push)” and example config `examples/ollamacode-full-git.yaml` to add the official MCP Git server (add, commit, push) via config when desired.

### Added (Phase 2)

- **Multiple MCP servers**: Config and CLI support multiple servers (stdio, SSE, Streamable HTTP). Tools are aggregated with server-name prefix; `call_tool` routes to the correct server. Single stdio still supported via `--mcp-args` / `OLLAMACODE_MCP_ARGS`.
- **Config file**: Optional `ollamacode.yaml` or `.ollamacode/config.yaml` (or `--config path`) for `model`, `mcp_servers`, `system_prompt_extra`. Env overrides where applicable; `OLLAMACODE_MCP_ARGS` overrides config when set.
- **MCP transports**: Stdio (default), plus `type: sse` and `type: streamable_http` in config for HTTP/SSE and Streamable HTTP endpoints.
- **Built-in MCP servers (examples)**: `examples/fs_mcp.py` (read_file, write_file, list_dir; root via `OLLAMACODE_FS_ROOT` or cwd); `examples/terminal_mcp.py` (run_command with cwd, env, timeout).
- **Integration test**: `test_cli_with_config_two_mcp_servers` runs CLI with config listing two stdio MCP servers and asserts tool use.
- **Config module**: `ollamacode.config` with `load_config`, `merge_config_with_env`, `_find_config_file`.
- **PyYAML**: Added dependency for config file parsing.

### Added (Phase 3)

- **Apply-edits UX**: Setting `ollamacode.reviewEditsBeforeApply`; when true, extension shows QuickPick “Apply all” / “Reject” / “Show diff first”. Show diff opens side-by-side diff per edited file, then Apply all or Reject.
- **Extension helpers**: `getEditsByFile`, `applyEditsToContent` in `ollamacodeRunner` for grouping edits and computing new content (used by diff view).
- **Extension unit tests**: Vitest tests for `parseEditsFromOutput`, `getEditsByFile`, `applyEditsToContent`; run with `npm run test` in `editor/vscode-extension`.
- **Extension docs**: `editor/vscode-extension/README.md` documents all settings (including `reviewEditsBeforeApply`), edit protocol, how to add MCP (mcpArgs + config file), and dev/test commands; main README links to it.
- **Composer-like view**: “Composer” panel in the VS Code extension sidebar. User describes a multi-file task and clicks Run; the agent returns proposed edits. Proposed changes are shown as a list of files; per file: **Preview** (opens side-by-side diff), **Apply**, **Reject**; plus **Apply all** / **Reject all**. Edits are never applied automatically in Composer.

### Added (Phase 4 and 3.1)

- **Streaming in CLI (4.3)**: `--stream` / `-s` flag; agent `run_agent_loop_stream` and `run_agent_loop_no_mcp_stream` (async generators) yield content fragments; tokens printed to stdout with flush for live display.
- **Streaming in extension (4.4, 3.1)**: Extension runs CLI with `--stream` when `ollamacode.streamResponse` is true (default); `runCli` accepts `stream` and `onStreamChunk`; Chat Participant and sidebar Chat update in real time; sidebar handles `streamStart`, `streamChunk`, `streamEnd`, `streamResult`.
- **@file / path context (4.1)**: Extension setting `ollamacode.injectCurrentFile` (default: true) prepends current editor file path to the prompt; README documents fs_mcp for reading workspace files.
- **Context window (4.5)**: README section on long conversations and context limits; future truncate/summarize noted in roadmap.
- **Optional codebase index (4.2)**: Documented in roadmap as optional MCP tool or external index.

### Added (Phase 4 follow-up: codebase index, context, history, inline chat, HTTP API)

- **Optional codebase index (4.2)**: `examples/codebase_mcp.py` with `search_codebase` (keyword search) and `get_relevant_files` (path match); add to config for @codebase-style context.
- **Context window (4.5)**: `--max-messages` and config `max_messages`; agent truncates message history (keep system + last N) before each Ollama call.
- **Conversation history**: CLI `--history-file` appends each interactive turn (user + assistant) to a file; extension persists last 20 messages in workspace state and restores on panel open.
- **Inline chat**: Extension command “OllamaCode: Chat with Selection” (right-click selected code); opens chat panel with selection as context.
- **Local HTTP API**: `ollamacode serve` (optional dep `ollamacode[server]`); POST /chat with JSON `{"message": "..."}` returns `{"content": "..."}`; for other editors and scripts.

### Added (Phase 5)

- **E2E tests (5.1)**: `tests/e2e/test_cli_e2e.py` – full CLI + demo MCP flow; markers `e2e` and `integration`.
- **Extension E2E**: @vscode/test-electron; `src/test/runTest.ts` + `src/test/suite/` (Mocha); `npm run test:e2e` runs extension in VS Code test host; one E2E test (Open Chat Panel command). E2E optional in CI (slow).
- **5.5 Optional TUI**: Rich-based TUI: `--tui` flag; `ollamacode/tui.py` with `run_tui()`; interactive chat with Live panel and streaming; `pip install ollamacode[tui]` (rich).
- **CI pipeline (5.2)**: GitHub Actions: Ruff (check + format), Pyright, unit tests, extension build and Vitest tests.
- **Packaging (5.3)**: PyPI-ready `pyproject.toml` (classifiers, urls, license, optional-dependencies dev). README: `pip install ollamacode`, `pipx install ollamacode`.
- **Performance / observability (5.4)**: CONTRIBUTING section “Debugging slow turns” (Ollama/MCP latency; optional future timing).
- **Other editors (5.6)**: `.docs/OTHER_EDITORS.md` and README subsection: use CLI from Zed, Sublime, Neovim; future HTTP API noted.
- **Lint and typecheck**: Ruff (line-length 128) and Pyright (basic mode); agent/mcp_client type fixes for Pyright.

## [0.1.0] - 2025-02-08

### Added

- **Core agent**: Single-turn agent loop (Ollama chat with tool calling + MCP client). `run_agent_loop` and `run_agent_loop_no_mcp`.
- **MCP client**: Stdio connection to one MCP server; `list_tools`, `call_tool`; bridge MCP tool schema → Ollama tool format.
- **CLI**: `ollamacode` entrypoint; single query or interactive chat; env `OLLAMACODE_MODEL`, `OLLAMACODE_MCP_ARGS`, `OLLAMACODE_SYSTEM_EXTRA`.
- **VS Code extension**: Sidebar chat panel; `@ollamacode` Chat Participant; apply-edits protocol (`<<OLLAMACODE_EDITS>>` … `<<END>>`); settings `ollamacode.cliPath`, `ollamacode.model`, `ollamacode.mcpArgs`.
- **Tests**: Unit tests for bridge, mcp_client, agent (mocked Ollama/MCP), CLI (arg parse, env, run path). Integration test (CLI + demo MCP; requires Ollama).
- **Docs**: MkDocs API reference (agent, bridge, mcp_client, cli). Contributor guide and roadmap in `.docs/`.
- **Changelog**: This file.

[Unreleased]: https://github.com/your-org/ollamacode/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-org/ollamacode/releases/tag/v0.1.0
