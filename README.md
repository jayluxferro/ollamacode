# OllamaCode

A coding assistant powered by **local models** (Ollama) and **MCP** (Model Context Protocol), similar in spirit to Cursor Code and Claude Code—all on your machine.

## Features

- **Local-only**: Ollama for all reasoning and code generation; no cloud API required.
- **MCP tools**: Connect any MCP server (filesystem, browser, custom tools) and use them from the agent.
- **CLI**: One-off queries or interactive chat from the terminal.
- **VS Code**: Sidebar chat panel and `@ollamacode` in VS Code Chat; optional workspace edit application.

## Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.com)** installed and running
- A model pulled in Ollama (default: `qwen2.5-coder:32b`). For tool calling, use a tool-capable model (e.g. `qwen2.5-coder:32b`, `qwen3:32b`, `qwen3-vl:32b`).

## Installation

**From PyPI** (when published):

```bash
pip install ollamacode
# or with uv (isolated env, CLI on PATH):
uv tool install ollamacode
# or with pipx:
pipx install ollamacode
```

**From source** (development):

```bash
cd OllamaCode
uv sync
# or: pip install -e .
```

Ensure the `ollamacode` CLI is on your PATH (e.g. `uv run ollamacode` from source, or `ollamacode` after `uv tool install` / pipx).

## Usage

### CLI

**Single query** (uses built-in MCP by default—see below):

```bash
uv run ollamacode "Explain recursion in one sentence"
```

**Interactive chat** (same; agent has read_file, run_command, search_codebase, etc.):

```bash
uv run ollamacode
```

**Custom MCP server** (overrides default; use config or env):

```bash
uv run ollamacode --mcp-command python --mcp-args examples/demo_server.py "What is 2+3?"
```

Use `OLLAMACODE_MCP_ARGS` to pass MCP server args without flags (e.g. `export OLLAMACODE_MCP_ARGS="python examples/demo_server.py"`). To use **no MCP**, add a config file with `mcp_servers: []`.

**Options:**

| Option | Env / default | Description |
|--------|----------------|-------------|
| `--config`, `-c` | (none) | Path to config file (default: `ollamacode.yaml` or `.ollamacode/config.yaml` in cwd) |
| `--model`, `-m` | `OLLAMACODE_MODEL` / `qwen2.5-coder:32b` | Ollama model name |
| `--stream`, `-s` | (flag) | Stream response tokens to stdout (for extension or live display) |
| `--tui` | (flag) | Interactive terminal UI (Rich). Requires: `pip install ollamacode[tui]` |
| `--max-messages` | config `max_messages` / 0 | Cap message history sent to Ollama (0 = no limit). For long chats. |
| `--history-file` | (path) | Append each interactive turn (user + assistant) to this file. |
| `--port` | 8000 | Port for `serve` command. |
| `--mcp-command` | `OLLAMACODE_MCP_COMMAND` / `python` | Command for legacy single-stdio MCP |
| `--mcp-args` | `OLLAMACODE_MCP_ARGS` (space-separated) | Args for legacy single-stdio MCP (overrides config when set) |

### Config file

Optional YAML config (`ollamacode.yaml` or `.ollamacode/config.yaml` in the current directory, or `--config path`):

```yaml
model: qwen2.5-coder:32b
system_prompt_extra: "Optional extra system instructions."
max_messages: 0   # 0 = no limit; cap message history sent to Ollama (e.g. 50 for long chats)

mcp_servers:
  - type: stdio
    command: python
    args: [examples/demo_server.py]
  - type: stdio
    command: python
    args: [examples/fs_mcp.py]
  # HTTP/SSE (if your MCP server supports it):
  # - type: sse
  #   url: http://localhost:8000/sse
  # - type: streamable_http
  #   url: http://localhost:8000/mcp
```

When multiple servers are configured, tool names are prefixed with the server name (e.g. `OllamaCode Demo_add`, `ollamacode-fs_read_file`) so the agent can call the right server.

### Built-in MCP (default)

When you run OllamaCode **without** a config file and **without** `OLLAMACODE_MCP_ARGS`, it automatically starts five built-in MCP servers so the agent can work efficiently out of the box:

- **ollamacode-fs** – `read_file`, `write_file`, `list_dir` (workspace root: `OLLAMACODE_FS_ROOT` or cwd).
- **ollamacode-terminal** – `run_command` (run shell commands, cwd, env, timeout).
- **ollamacode-codebase** – `search_codebase` (keyword search), `get_relevant_files` (path match).
- **ollamacode-git** – read-only Git: `git_status`, `git_diff_unstaged`, `git_diff_staged`, `git_log`, `git_show`, `git_branch` (understand repo state; no commit/push).
- **ollamacode-tools** – `run_linter` (e.g. ruff, eslint), `run_tests` (e.g. pytest, npm test); returns stdout/stderr/return code.

They are shipped inside the package (`ollamacode.servers`) and run as subprocesses. To disable MCP entirely, use a config file with `mcp_servers: []`. To add or replace servers, use `ollamacode.yaml` or `OLLAMACODE_MCP_ARGS`.

### Opt-in: full Git (commit, push)

The built-in Git MCP is read-only. To let the agent **stage, commit, and push**, add the official [MCP Git server](https://github.com/modelcontextprotocol/servers/tree/main/src/git) via config. Example: keep the built-in fs, terminal, and codebase servers, and use the official Git server (full-featured) instead of the built-in one:

```yaml
# ollamacode.yaml — full Git (add, commit, push) + built-in fs, terminal, codebase
model: qwen2.5-coder:32b

mcp_servers:
  - type: stdio
    command: python
    args: ["-m", "ollamacode.servers.fs_mcp"]
  - type: stdio
    command: python
    args: ["-m", "ollamacode.servers.terminal_mcp"]
  - type: stdio
    command: python
    args: ["-m", "ollamacode.servers.codebase_mcp"]
  - type: stdio
    command: npx
    args: ["-y", "@modelcontextprotocol/server-git"]
```

Requires Node.js for `npx`. The agent will then have `git_add`, `git_commit`, `git_checkout`, etc. Use only in repos you’re comfortable having the agent modify. A copy-paste example config is in [examples/ollamacode-full-git.yaml](examples/ollamacode-full-git.yaml).

### Opt-in: semantic codebase search

The built-in **codebase** server does keyword/substring search. For **meaning-based** (semantic) search, add the optional **ollamacode-semantic** server. It uses Ollama embeddings (e.g. `nomic-embed-text`) and caches them under `.ollamacode/embeddings.json`.

1. Pull an embedding model: `ollama pull nomic-embed-text` (or set `OLLAMACODE_EMBED_MODEL` to another model).
2. Add the semantic server to your config (it is **not** in the default list):

```yaml
mcp_servers:
  # ... your other servers (or built-ins) ...
  - type: stdio
    command: python
    args: ["-m", "ollamacode.servers.semantic_mcp"]
```

3. In chat, ask the agent to **index the codebase** first (e.g. “Run index_codebase for this project”). Then you can ask “Where do we handle auth?” or “Find code that loads config” and the agent can use **semantic_search_codebase** for meaning-based results.

Tools: **index_codebase**(file_pattern) – index workspace (or glob); **semantic_search_codebase**(query, max_results) – search by meaning. Cache is stored in the workspace `.ollamacode/` directory. Example config: [examples/ollamacode-semantic.yaml](examples/ollamacode-semantic.yaml).

### External MCP servers

You can add **any** MCP server via config or env. Put entries in `ollamacode.yaml` under `mcp_servers`, or use `OLLAMACODE_MCP_ARGS` for a single stdio server. Examples:

- **Official MCP servers** (full-featured Git, Fetch, Memory, etc.): [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers). Many are runnable via `npx` or `uvx`. Example (Git with commit/push):
  ```yaml
  mcp_servers:
    - type: stdio
      command: npx
      args: ["-y", "@modelcontextprotocol/server-git"]
  ```
- **MCP registry**: Browse [registry.modelcontextprotocol.io](https://registry.modelcontextprotocol.io) or [mcpservers.org](https://mcpservers.org) for community servers (Slack, Docker, databases, etc.).
- **Custom server**: Point `command` and `args` to your script (e.g. `python my_mcp.py`). OllamaCode starts them as subprocesses and forwards tool calls.

Built-in servers are used only when you have **no** config and **no** `OLLAMACODE_MCP_ARGS`. As soon as you set `mcp_servers` in config or set the env var, that list replaces the default (you can include the built-ins by adding them explicitly if you want).

### Optional MCP servers (examples/)

In `examples/` you can run alternate or extra servers and add them via config:

- **demo_server.py** – `add`, `echo` (for testing).
- **fs_mcp.py**, **terminal_mcp.py**, **codebase_mcp.py** – same tools as the built-in servers; useful if you want to run from source or customize.

Run any of them with stdio and add to config or `OLLAMACODE_MCP_ARGS` (e.g. `python examples/demo_server.py`).

### VS Code extension

1. Open `editor/vscode-extension` in VS Code.
2. Run `npm install` and `npm run compile`, then press **F5** to launch the Extension Development Host.
3. In the new window:
   - Open **OllamaCode** from the Activity Bar: **Chat** (quick Q&A) or **Composer** (multi-file task → list of proposed changes → Preview / Apply / Reject per file or all), or
   - In Chat, type **@ollamacode** and your prompt.

**Extension settings:** `ollamacode.cliPath`, `ollamacode.model`, `ollamacode.mcpArgs`, `ollamacode.reviewEditsBeforeApply`, `ollamacode.streamResponse` (stream tokens as they arrive), `ollamacode.injectCurrentFile` (prepend current file path to the prompt). When the model outputs edits in the `<<OLLAMACODE_EDITS>>` … `<<END>>` protocol, the extension applies them in the workspace (optionally after showing Apply all / Reject / Show diff). **Chat history** is persisted in workspace state (last 20 messages). **Inline chat:** right-click selected code → **OllamaCode: Chat with Selection** to open the chat panel with the selection as context. Full settings, edit protocol, and how to add MCP: **[editor/vscode-extension/README.md](editor/vscode-extension/README.md)**.

### File and path context

- **CLI**: Mention paths in your prompt (e.g. “in `src/main.py`”); if you use the **fs_mcp** server (e.g. from config), the model can call `read_file` / `list_dir` to read workspace files.
- **Extension**: With **`ollamacode.injectCurrentFile`** (default: on), the extension prepends the current editor file path to your prompt (e.g. “Current file: src/main.py”) so the model knows which file you’re in. You can still refer to other paths in the prompt; the fs_mcp server can read them if configured.

### Context window and long conversations

Ollama and the model define the context limit. Use **`--max-messages`** (or config `max_messages`) to cap how many messages are sent to Ollama; the agent keeps the system message (if any) and the most recent turns. For long chats, set e.g. `max_messages: 50` in config or `--max-messages 50`. Tool results and long replies still increase token count; if you hit limits, start a new chat or lower `max_messages`.

## Project layout

```
OllamaCode/
├── ollamacode/           # Core package
│   ├── agent.py          # Agent loop (Ollama + MCP tools)
│   ├── bridge.py         # MCP tool schema → Ollama tool format
│   ├── cli.py            # CLI entrypoint
│   ├── config.py         # Config file (YAML) loading
│   ├── mcp_client.py     # MCP client (stdio + multi-server + HTTP/SSE)
│   ├── serve.py          # Optional HTTP API (ollamacode serve)
│   └── tui.py            # Optional TUI (--tui)
├── examples/
│   ├── demo_server.py    # Minimal MCP server (add, echo)
│   ├── fs_mcp.py         # Filesystem MCP (read_file, write_file, list_dir)
│   ├── terminal_mcp.py   # Terminal MCP (run_command)
│   └── codebase_mcp.py   # Codebase search (search_codebase, get_relevant_files)
├── tests/                # Pytest tests
├── editor/
│   └── vscode-extension/ # VS Code extension
├── pyproject.toml
└── README.md
```

## Architecture

User input is sent to the **agent loop**, which:

1. Connects to configured **MCP server(s)** and gets tools (`list_tools`).
2. Converts MCP tools to Ollama’s tool format and calls **Ollama** `/api/chat` with messages and tools.
3. If the model returns **tool_calls**, runs them via MCP (`call_tool`) and appends results to the conversation, then repeats until the model replies with text only.

So: **OllamaCode = MCP client + Ollama (tool calling) + agent loop**. No cloud; all logic and tools are local or under your control.

### Local HTTP API (other editors)

Run **`ollamacode serve`** (or `ollamacode serve --port 9000`) to start a local API. Requires: `pip install ollamacode[server]`. **POST /chat** with JSON `{"message": "..."}` returns `{"content": "..."}`. Use this from Zed, Sublime, Neovim, or scripts instead of spawning the CLI. See [.docs/OTHER_EDITORS.md](.docs/OTHER_EDITORS.md) for integration ideas.

### Conversation history

- **CLI:** Use **`--history-file PATH`** in interactive mode to append each turn (user + assistant) to a file.
- **Extension:** Chat history (last 20 messages) is stored in workspace state and restored when you reopen the panel.

## References

- [Ollama: Tool calling](https://docs.ollama.com/capabilities/tool-calling)
- [Model Context Protocol](https://modelcontextprotocol.io) · [Python SDK](https://github.com/modelcontextprotocol/python-sdk)
