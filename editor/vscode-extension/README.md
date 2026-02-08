# OllamaCode VS Code Extension

Chat with **OllamaCode** (local Ollama + MCP tools) inside VS Code, similar to Cursor or Claude Code.

## What it does

- **Sidebar panel**: **Chat** and **Composer** views in the Activity Bar. In **Chat**, type a message and click Send; the extension runs the `ollamacode` CLI and shows the reply. In **Composer**, describe a multi-file task, click Run, then see proposed changes as a list of files with **Preview** (diff) / **Apply** / **Reject** per file, or **Apply all** / **Reject all**.
- **@ollamacode in Chat**: In VS Code Chat (e.g. Copilot Chat), type `@ollamacode` and your prompt; the same CLI runs and the response appears in the chat (requires VS Code 1.109+).
- **Apply edits**: When the model outputs file edits in the protocol (see below), the extension applies them in the workspace. Optionally **review before apply**: show Apply all / Reject / Show diff first (Chat and @ollamacode). In **Composer**, edits are never applied automatically—you choose per file or all.
- **MCP from settings or config**: Set **`ollamacode.mcpArgs`** (e.g. `python path/to/server.py`); the extension passes it as `OLLAMACODE_MCP_ARGS`. Or put an `ollamacode.yaml` in your workspace root—the CLI is run with workspace cwd, so it will load that config and use your MCP servers.

No cloud: everything runs locally (Ollama + your MCP servers).

## Prerequisites

1. **Ollama** installed and a model pulled (e.g. `ollama pull qwen2.5-coder:32b`).
2. **OllamaCode** CLI on your PATH (or set `ollamacode.cliPath`):
   - From this repo: `cd OllamaCode && uv sync` then set CLI path to `uv run ollamacode`, or
   - Install the package so `ollamacode` is on PATH.

## Setup

1. Open the extension folder in VS Code: `File > Open Folder` → `editor/vscode-extension`.
2. Run `npm install` then `npm run compile`.
3. Press **F5** to launch a new VS Code window with the extension loaded.
4. Use the **OllamaCode** sidebar (Chat or Composer) or type **@ollamacode** in the Chat panel.

**Tests:** `npm run test` runs Vitest unit tests. `npm run test:e2e` runs E2E tests in the VS Code Extension Development Host (downloads VS Code; ensure no other VS Code instance is running when run from CLI).

## Settings

| Setting | Default | Description |
|--------|---------|-------------|
| **`ollamacode.cliPath`** | `ollamacode` | Command to run. Use `uv run ollamacode` if you run from repo with uv. |
| **`ollamacode.model`** | `qwen2.5-coder:32b` | Ollama model name. You can pick from a dropdown in the **Chat** panel (models are loaded from the Ollama API) or run **OllamaCode: Select Ollama Model** from the Command Palette. |
| **`ollamacode.mcpArgs`** | (empty) | Space-separated MCP server args (e.g. `python path/to/server.py`). Passed as `OLLAMACODE_MCP_ARGS` when running the CLI. |
| **`ollamacode.reviewEditsBeforeApply`** | `false` | When **true**, before applying edits the extension shows: **Apply all**, **Reject**, or **Show diff first**. If you choose Show diff, it opens a diff view for each edited file, then asks Apply all or Reject. |
| **`ollamacode.streamResponse`** | `true` | When **true**, stream response tokens to the chat/panel as they arrive (CLI is run with `--stream`). |
| **`ollamacode.injectCurrentFile`** | `true` | When **true**, prepend the current editor file path to the prompt (e.g. "Current file: src/main.py") so the model has file context. |

## How to add MCP servers

1. **Via extension setting**: Set **`ollamacode.mcpArgs`** to the command and args for a single stdio MCP server, e.g. `python examples/fs_mcp.py`. The extension passes this as `OLLAMACODE_MCP_ARGS` to the CLI.
2. **Via config file**: Put an **`ollamacode.yaml`** (or `.ollamacode/config.yaml`) in your **workspace root**. The extension runs the CLI with the workspace folder as cwd, so the CLI will find and load that config. There you can define multiple MCP servers (stdio, SSE, or Streamable HTTP). See the main [OllamaCode README](../README.md) for the config schema.

## Edit protocol (apply edits in the workspace)

When running from the extension, the CLI is told (via `OLLAMACODE_SYSTEM_EXTRA`) to output file edits in this format so the extension can apply them:

```
<<OLLAMACODE_EDITS>>
{"edits":[{"path":"relative/path/to/file","range":{"start":{"line":0,"character":0},"end":{"line":0,"character":0}},"newText":"content"}]}
<<END>>
```

- **path**: Relative to the workspace root.
- **range**: 0-based `(line, character)`. Omit **range** to replace the entire file.
- **newText**: The text to insert or replace in that range.

The extension parses this block from the CLI stdout, optionally shows **Apply all / Reject / Show diff** (if `ollamacode.reviewEditsBeforeApply` is true), then applies the edits with `workspace.applyEdit()` and shows the rest of the response (without the block) in the panel or chat.

## Development and tests

- **Compile**: `npm run compile`
- **Watch**: `npm run watch`
- **Unit tests**: `npm run test` (Vitest; tests `parseEditsFromOutput`, `getEditsByFile`, `applyEditsToContent` in `ollamacodeRunner`)
