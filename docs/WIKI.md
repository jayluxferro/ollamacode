# OllamaCode Wiki

Community and reference documentation for OllamaCode.

---

## Getting started

- **[README](https://github.com/jayluxferro/ollamacode/blob/main/README.md)** — Installation, quick start, CLI options, and config overview.
- **[Quick start](index.md)** — Minimal setup and first run.

---

## Configuration

- **Config file** — `ollamacode.yaml` or `.ollamacode/config.yaml` in your project; optional user config at `~/.ollamacode/config.yaml` (merged with project). See the [Config file](https://github.com/jayluxferro/ollamacode#config-file) section in the README.
- **Config hierarchy** — User config is the base; project config deep-merges over it.

---

## MCP & tools

- **[MCP servers](MCP_SERVERS.md)** — Built-in and custom MCP servers.
- **Built-in tools** — fs (read/write/edit), terminal (run_command), codebase (search, glob, grep), tools (linter, tests, fetch_url, …), git, skills, state, reasoning (think), screenshot (Playwright). See README [Built-in MCP](https://github.com/jayluxferro/ollamacode#built-in-mcp-default).
- **Screenshot** — `screenshot(url)` in default servers; Chromium auto-installs on first use or run `ollamacode install-browsers` once.

---

## TUI & slash commands

- **Interactive chat** — Run `ollamacode` with no query to start the TUI (`pip install ollamacode[tui]` for Rich).
- **Slash commands** — `/help`, `/clear`, `/new`, `/sessions`, `/search`, `/resume`, `/session`, `/branch`, `/model`, `/fix`, `/test`, `/docs`, `/profile`, `/summary`, `/auto`, `/commands`, `/image`, `/subagents`, `/subagent`, and custom commands from `commands.md`. See `/help` in the TUI.

---

## API & protocol

- **[API reference](api.md)** — Public Python API: agent loop, bridge, MCP client.
- **[Structured protocol](STRUCTURED_PROTOCOL.md)** — stdio and HTTP endpoints for editors (diagnostics, completion).
- **[Other editors](OTHER_EDITORS.md)** — Neovim, Zed, Sublime, VS Code extension.

---

## Advanced

- **[RLM mode](RLM.md)** — Recursive Language Model: context as metadata, REPL with `llm_query()`.
- **[Toolchain registry](TOOLCHAIN_REGISTRY.md)** — Curated linters, test runners, and usage with built-in tools.

---

## Changelog

See [CHANGELOG](https://github.com/jayluxferro/ollamacode/blob/main/CHANGELOG.md) for version history.
