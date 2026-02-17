# OllamaCode

A coding assistant powered by **local models** (Ollama) and **MCP** (Model Context Protocol)—all on your machine.

## Features

- **Local-only**: Ollama for all reasoning and code generation; no cloud API required.
- **MCP tools**: Connect any MCP server (filesystem, terminal, codebase, custom tools) and use them from the agent. Built-in fs, terminal, codebase, tools, and git when no config is present.
- **CLI**: One-off queries or interactive chat from the terminal (interactive mode uses the built-in TUI).
- **HTTP & stdio API**: `ollamacode serve` for REST (POST /chat, /chat/continue, /chat/stream, /apply-edits, /rag/index, /rag/query); `ollamacode protocol` for JSON-RPC over stdin/stdout so editors can integrate without HTTP.
- **Intelligence on by default**: Reasoning (brief rationale), meta-reflection (second-pass review), branch context (git diff in prompt), and planning/feedback/knowledge are enabled by default; set `use_reasoning: false` or `use_meta_reflection: false` in config to disable.
- **RLM (experimental)**: `--rlm` — recursive language model mode (opt-in): context stays in a REPL, model sees only metadata and uses `llm_query()` on slices. See [docs/RLM.md](docs/RLM.md).

## Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.com)** installed and running
- A model pulled in Ollama (default: `gpt-oss:20b`). For tool calling, use a tool-capable model (e.g. `gpt-oss:20b`, `qwen2.5-coder:32b`, `qwen3:32b`).

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
# optional bootstrap (creates .ollamacode/config.yaml, checks Ollama):
python install.py
# or install CLI on PATH from this repo:
uv tool install .
# or: pip install -e .
```

**Homebrew** (when a formula is available in a tap):

```bash
brew tap your-org/ollamacode   # if using a custom tap
brew install ollamacode
```

If the formula is in homebrew-core, use `brew install ollamacode` only. Check the project releases for the current install method.

Ensure the `ollamacode` CLI is on your PATH (e.g. `uv run ollamacode` from source, or `ollamacode` after `uv tool install` / pipx).

## Usage

### CLI

**Single query** (uses built-in MCP by default—see below):

```bash
uv run ollamacode "Explain recursion in one sentence"
```

**Interactive chat** (opens the TUI; agent has read_file, run_command, search_codebase, etc.). Requires `pip install ollamacode[tui]` for the Rich TUI:

```bash
uv run ollamacode
```

**Custom MCP server** (overrides default; use config or env):

```bash
uv run ollamacode --mcp-command python --mcp-args examples/demo_server.py "What is 2+3?"
```

Use `OLLAMACODE_MCP_ARGS` to pass MCP server args without flags (e.g. `export OLLAMACODE_MCP_ARGS="python examples/demo_server.py"`). To use **no MCP**, add a config file with `mcp_servers: []`.

**Commands** (as first argument): `serve` — start HTTP API; `protocol` — stdio JSON-RPC for editors; `convert-mcp` — convert Cursor/Claude MCP JSON to YAML. RLM mode: use `--rlm` (prompt as query or from stdin).

**RLM mode** (opt-in; context not sent to model; model uses REPL + `llm_query()`):

```bash
uv run ollamacode --rlm "Summarize the key points in the text below..."
echo "Long document..." | uv run ollamacode --rlm
```

**Options:**

| Option | Env / default | Description |
|--------|----------------|-------------|
| `--config`, `-c` | (none) | Path to config file (default: `ollamacode.yaml` or `.ollamacode/config.yaml` in cwd) |
| `--model`, `-m` | `OLLAMACODE_MODEL` / `gpt-oss:20b` | Ollama model name |
| `--stream`, `-s` | (flag) | Stream response tokens to stdout |
| `--file`, `-f` | (path) | Prepend file contents to prompt (chat-with-selection) |
| `--lines` | START-END | With `--file`: include only this line range (1-based inclusive) |
| `--apply-edits` | (flag) | Parse `<<EDITS>>` from model output; show diff and prompt to apply |
| `--apply-edits-dry-run` | (flag) | With `--apply-edits`: show diff only, do not apply |
| `--max-messages` | config `max_messages` / 0 | Cap message history sent to Ollama (0 = no limit). For long chats. |
| `--max-tool-rounds` | config `max_tool_rounds` / 20 | Max tool-call rounds per turn. |
| `--no-mcp` | (flag) | Skip starting MCP servers for this run (faster when you don't need tools). |
| `--timing` | config `timing` / false | Log per-step durations (Ollama call, each tool call, turn total) to stderr. |
| `--history-file` | (path) | Append each interactive turn (user + assistant) to this file. |
| `--quiet`, `-q` | (flag) | Suppress [OllamaCode] progress lines (e.g. for scripts). |
| `--port` | 8000 | Port for `serve` command. |
| `--mcp-command` | `OLLAMACODE_MCP_COMMAND` / `python` | Command for legacy single-stdio MCP |
| `--mcp-args` | `OLLAMACODE_MCP_ARGS` (space-separated) | Args for legacy single-stdio MCP (overrides config when set) |
| `--output`, `-o` | (path) | For **convert-mcp**: output YAML file; stdout if omitted. |
| `--headless` | (flag) | CI mode: exit 0/1/2, implies --quiet. Use with --json for machine-readable output. |
| `--auto` | (flag) | Autonomous mode: no per-tool confirm, more tool rounds (CLI and TUI /auto). |
| `--no-write` | (flag) | Read-only: block write_file and git add/commit/push. |
| `--max-tools` | N | Max tool-call rounds for this run (overrides config). |
| `--run-timeout` | SECONDS | Wall-clock timeout for single-query run; exit 1 on timeout. |

**Stopping:** Press **Ctrl+C** to interrupt the agent (single query or interactive TUI).

### Config file

Optional YAML config (`ollamacode.yaml` or `.ollamacode/config.yaml` in the current directory, or `--config path`). In a monorepo, set **`OLLAMACODE_CONFIG_LOOKUP_PARENT=1`** to also search parent directories for a config file.

```yaml
model: gpt-oss:20b
system_prompt_extra: "Optional extra system instructions."
rules_file: .ollamacode/rules.md   # optional: load and append to system prompt (path relative to cwd)
# OLLAMA.md: optional user (~/.ollamacode/OLLAMA.md) and project (.ollamacode/OLLAMA.md or OLLAMA.md) context appended to system prompt
# Custom slash commands: ~/.ollamacode/commands.md or .ollamacode/commands.md (## /name, description, optional prompt with {{rest}}); /commands in TUI lists them
max_messages: 0   # 0 = no limit; cap message history sent to Ollama (e.g. 50 for long chats)
auto_summarize_after_turns: 0   # when > 0, summarize oldest turns when history exceeds this (e.g. 10)
max_tool_rounds: 20   # max tool-call rounds per turn
max_tool_result_chars: 0   # 0 = no limit; truncate tool results to this many chars to save context
max_edits_per_request: 0   # 0 = no limit; cap <<EDITS>> count per turn (e.g. 10)
timing: false         # set true to log per-step durations to stderr
linter_command: "ruff check ."   # used by /fix slash command (CLI and TUI)
test_command: "pytest"          # used by /test slash command (CLI and TUI)
semantic_codebase_hint: true   # one-time tip to add semantic MCP server (set false to disable)
branch_context: true          # (default) inject git diff vs base into system prompt; set false to disable
branch_context_base: "main"   # base branch for branch_context
pr_description_file: null     # optional path to PR/change description (e.g. .git/PR_DESCRIPTION)
include_builtin_servers: true # when true (default), add fs/terminal/codebase/tools/git so the model can list files, run commands; set false to use only mcp_servers
# use_reasoning: true         # ask model for brief reasoning/rationale (default on)
# use_meta_reflection: true   # second-pass review of assistant reply for consistency/clarity (default on)
# confirm_tool_calls: false   # if true, prompt [y/N/e] before each tool (CLI and TUI); e = edit args in $EDITOR
# prompt_template: refactor   # load .ollamacode/templates/refactor.md and append to system prompt
# prompt_snippets: []         # list of strings appended to system prompt (repo-specific instructions)
# allowed_tools: null         # if set (e.g. [read_file, write_file, run_tests]), only these tools are exposed
# blocked_tools: null         # if set (e.g. [run_command]), exclude these tools
# code_style: ""              # optional code style rules injected into system prompt
# safety_output_patterns: []  # list of substrings that trigger a safety warning in CLI output
# memory_auto_context: true   # auto-inject query-specific memory (knowledge graph + local RAG)
# memory_kg_max_results: 4    # max knowledge-graph matches injected per turn
# memory_rag_max_results: 4   # max local RAG snippets injected per turn
# memory_rag_snippet_chars: 220 # max chars per injected RAG snippet
# planner_model: ""           # optional model override for multi-agent planner
# executor_model: ""          # optional model override for multi-agent executor
# reviewer_model: ""          # optional model override for multi-agent reviewer
# multi_agent_max_iterations: 2   # review loops for multi-agent (default 2)
# multi_agent_require_review: true # if false, skip reviewer stage in multi-agent
# tui_tool_trace_max: 5       # max tool trace lines in TUI
# tui_tool_log_max: 8         # max tool log entries in TUI
# tui_tool_log_chars: 160     # max chars per tool log entry in TUI
# tui_refresh_hz: 5           # TUI refresh rate (Hz)
# toolchain_version_checks:
#   - name: pytest
#     command: pytest --version
#     expect_contains: "7"

mcp_servers:
  - name: demo
    type: stdio
    command: python
    args: [examples/demo_server.py]
  - name: fs
    type: stdio
    command: python
    args: [examples/fs_mcp.py]
  # Optional name: label each server (e.g. name: git). HTTP/SSE:
  # - name: remote
  #   type: sse
  #   url: http://localhost:8000/sse
  # - type: streamable_http
  #   url: http://localhost:8000/mcp
```

You can add an optional **`name`** to each server (e.g. `name: git`, `name: burp`) to label it in the config. When multiple servers are configured, tool names are prefixed with the server name (e.g. `OllamaCode Demo_add`, `ollamacode-fs_read_file`) so the agent can call the right server.

**Custom MCP server types (plugins):** Built-in types are `stdio`, `sse`, and `streamable_http`. To add a new type (e.g. a custom transport), register an entry point in the `ollamacode.mcp_server_types` group. The entry point name is the config `type` value; the callable receives the server entry dict and must return `StdioServerParameters`, `SseServerParameters`, or `StreamableHttpParameters`. See `get_registered_mcp_server_types()` and `MCP_SERVER_TYPES_ENTRY_POINT_GROUP` in the [API docs](docs/api.md).

### Built-in MCP (default)

When you run OllamaCode **without** a config file and **without** `OLLAMACODE_MCP_ARGS`, it automatically starts the built-in MCP servers so the agent can work efficiently out of the box:

- **ollamacode-fs** – `read_file`, `write_file`, `list_dir`, `edit_file` (surgical find/replace), `multi_edit` (batch edits).
- **ollamacode-terminal** – `run_command` (run shell commands, cwd, env, timeout). Set **`OLLAMACODE_BLOCK_DANGEROUS_COMMANDS=1`** to block dangerous patterns (e.g. `rm -rf /`, `curl | bash`). Optional **`OLLAMACODE_CONFIRM_RISKY=1`** to require confirmation for risky patterns (re-run with **`OLLAMACODE_CONFIRM_RISKY_CONFIRMED=1`** to execute).
- **ollamacode-codebase** – `search_codebase` (keyword search), `get_relevant_files` (path match), `glob` (pattern), `grep` (regex with context).
- **ollamacode-tools** – `run_linter`, `run_tests`, `run_code_quality`, `run_coverage`, `install_deps`, `fetch_url` (HTTP GET).
- **ollamacode-git** – Git: `git_status`, `git_diff_*`, `git_log`, `git_add`, `git_commit`, `git_push`.
- **ollamacode-skills** – read/write skills and memory.
- **ollamacode-state** – persistent state (recent files, preferences, plan).
- **ollamacode-reasoning** – `think(reasoning)` for structured reasoning steps.
- **ollamacode-screenshot** – `screenshot(url)` to capture a web page as PNG. Chromium is installed automatically on first use, or run **`ollamacode install-browsers`** once to install upfront.

They are shipped inside the package (`ollamacode.servers`) and run as subprocesses. For **checkout** and other advanced Git features, see [Opt-in: full Git (official MCP server)](#opt-in-full-git-commit-push). To disable MCP entirely, use a config file with `mcp_servers: []`. To add or replace servers, use `ollamacode.yaml` or `OLLAMACODE_MCP_ARGS`.

### Opt-in: full Git (official MCP server)

The built-in Git MCP supports **stage, commit, and push**. For **checkout**, **branch**, and other advanced Git features, add the official [MCP Git server](https://github.com/modelcontextprotocol/servers/tree/main/src/git) via config. Example: keep the built-in fs, terminal, and codebase servers, and use the official Git server (full-featured) instead of the built-in one:

```yaml
# ollamacode.yaml — full Git (add, commit, push) + built-in fs, terminal, codebase
model: gpt-oss:20b

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

### @-style context (file / folder)

In your message you can use **@path** to inject file or folder context so the model gets it without extra tool calls:

- **@path/to/file** – Injects the file contents (relative to workspace root).
- **@path/to/folder/** – Injects a list of file names in that folder (up to 100 entries).

Example: `refactor @src/main.py to use async` — the model receives the file contents plus your instruction. Paths are resolved from the directory from which you run OllamaCode (workspace root).

### Chat with selection (--file / --lines)

Use **`--file PATH`** to prepend a file’s contents to your prompt (e.g. “explain this”). Use **`--lines START-END`** to include only a line range (1-based, inclusive):

```bash
ollamacode --file src/main.py "explain this"
ollamacode --file src/main.py --lines 10-30 "refactor this block"
```

Paths are relative to the current directory (workspace root).

### Structured apply-edits (--apply-edits)

Use **`--apply-edits`** so the model is instructed to output file changes in a parseable format. After the response, OllamaCode looks for a `<<EDITS>>` … `<<END>>` block containing a JSON array of `{ "path": "file", "oldText": "optional", "newText": "new content" }`. It shows a diff and prompts **Apply these edits? [y/N]**; if you answer **y**, the edits are applied to the workspace.

### Write scope

All file writes are scoped to the workspace root (the directory from which you run OllamaCode). The built-in **fs_mcp** server rejects paths outside `OLLAMACODE_FS_ROOT`; **--apply-edits** skips any edit whose path would resolve outside the workspace. This keeps model-driven edits from touching files outside your project.

### Semantic codebase

The built-in **codebase** MCP does keyword search. For semantic “where is X?” search, add a semantic MCP server to your config (see **docs/MCP_SERVERS.md**). When you start interactive mode without a semantic server, OllamaCode can show a one-line tip; set **`semantic_codebase_hint: false`** to disable it.

### Session summary

In interactive mode (TUI), use **`/summary [N]`** to summarize the last N turns (default 5) and replace them with one summary message. This frees context for long chats.

### Branch/PR context

Set **`branch_context: true`** (and optionally **`branch_context_base: "main"`**) to inject the current **git diff vs the base branch** into the system prompt so the model knows what you’re working on. Optionally set **`pr_description_file`** (e.g. `.git/PR_DESCRIPTION`) to also inject a PR or change description from a file.

### File and path context

Mention paths in your prompt (e.g. "in `src/main.py`"); if you use the **fs_mcp** server (e.g. from config), the model can call `read_file` / `list_dir` to read workspace files.

### Context window and long conversations

Ollama and the model define the context limit. Use **`--max-messages`** (or config `max_messages`) to cap how many messages are sent to Ollama; the agent keeps the system message (if any) and the most recent turns. For long chats, set e.g. `max_messages: 50` in config or `--max-messages 50`. Tool results and long replies still increase token count; if you hit limits, start a new chat or lower `max_messages`.

## Project layout

```
OllamaCode/
├── ollamacode/             # Core package
│   ├── agent.py            # Agent loop (Ollama + MCP tools)
│   ├── bridge.py           # MCP tool schema → Ollama tool format
│   ├── cli.py              # CLI entrypoint
│   ├── config.py           # Config file (YAML) loading
│   ├── context.py          # @-refs, file/line context, branch context
│   ├── edits.py            # <<EDITS>> parse and apply
│   ├── mcp_client.py       # MCP client (stdio + multi-server + HTTP/SSE)
│   ├── protocol.py         # Editor protocol (normalize request body)
│   ├── protocol_server.py   # Stdio JSON-RPC (ollamacode protocol)
│   ├── serve.py            # HTTP API (ollamacode serve)
│   ├── servers/            # Built-in MCP (fs, terminal, codebase, tools, git, semantic)
│   └── tui.py              # Interactive TUI (default when no query)
├── docs/                   # STRUCTURED_PROTOCOL.md, OTHER_EDITORS.md, MCP_SERVERS.md, RLM.md, etc.
├── examples/               # Demo and optional MCP servers
├── tests/
├── .githooks/               # Pre-push: ruff format + fix (see .githooks/README.md)
├── pyproject.toml
└── README.md
```

See the [Wiki](docs/WIKI.md) for a full index of documentation.

### Git pre-push hook (optional)

A **pre-push** hook runs `ruff format` and `ruff check --fix` before pushing. Install: `git config core.hooksPath .githooks` and `chmod +x .githooks/pre-push`. See [.githooks/README.md](.githooks/README.md).

## Architecture

User input is sent to the **agent loop**, which:

1. Connects to configured **MCP server(s)** and gets tools (`list_tools`).
2. Converts MCP tools to Ollama’s tool format and calls **Ollama** `/api/chat` with messages and tools.
3. If the model returns **tool_calls**, runs them via MCP (`call_tool`) and appends results to the conversation, then repeats until the model replies with text only.

So: **OllamaCode = MCP client + Ollama (tool calling) + agent loop**. No cloud; all logic and tools are local or under your control.

### HTTP API and editor protocol

Run **`ollamacode serve`** (or `ollamacode serve --port 9000`) to start a local HTTP API. Requires: `pip install ollamacode[server]`. Endpoints: **POST /chat** (JSON `{"message": "...", "file?", "lines?", "selection?", "confirmToolCalls?"}` → `{"content": "...", "edits"?}`), **POST /chat/continue** (tool approvals), **POST /chat/stream** (SSE), **POST /apply-edits** (apply edits server-side), **POST /rag/index** (build local retrieval index), **POST /rag/query** (query local retrieval snippets). Optional auth via config `serve.api_key` or `OLLAMACODE_SERVE_API_KEY`.

### VSCode extension (preview)

The VSCode extension lives in `editors/vscode`. It supports chat, streaming, apply-edits, tool approvals, diagnostics, and inline completions.

Key settings:
- `ollamacode.baseUrl` (default `http://localhost:8000`)
- `ollamacode.confirmToolCalls` (tool approvals)
- `ollamacode.multiAgent` (planner/executor/reviewer)
- `ollamacode.enableDiagnostics`, `ollamacode.enableInlineCompletions`

For editors that prefer a process over HTTP: **`ollamacode protocol`** runs a JSON-RPC server on stdin/stdout (one request per line). Methods: `ollamacode/chat`, `ollamacode/chatStream` (streaming), `ollamacode/applyEdits`. See [docs/STRUCTURED_PROTOCOL.md](docs/STRUCTURED_PROTOCOL.md) and [docs/OTHER_EDITORS.md](docs/OTHER_EDITORS.md) for integration.

### Conversation history

Use **`--history-file PATH`** in interactive mode to append each turn (user + assistant) to a file. Format: one block per turn, `---` then `user: <content>` and `assistant: <content>` (one line each, content may be multi-line in YAML style). Example:

```
---
user: What is 2+3?
assistant: 5
---
user: Explain recursion
assistant: Recursion is...
```

### Adding MCP servers (Cursor / Claude style)

To use the same MCP servers as Cursor or Claude, put them in **`ollamacode.yaml`** under `mcp_servers`. See [docs/MCP_SERVERS.md](docs/MCP_SERVERS.md) for the config format and how to translate Cursor/Claude JSON config into OllamaCode YAML.

**Convert existing config:** Use **`convert-mcp`** to turn Cursor or Claude JSON into OllamaCode YAML:
```bash
ollamacode convert-mcp cursor-mcp.json --output ollamacode.yaml
cat claude_mcp.json | ollamacode convert-mcp -o .ollamacode/config.yaml
```

## Roadmap

Current features and release history: [CHANGELOG.md](CHANGELOG.md).

## References

- [Ollama: Tool calling](https://docs.ollama.com/capabilities/tool-calling)
- [Model Context Protocol](https://modelcontextprotocol.io) · [Python SDK](https://github.com/modelcontextprotocol/python-sdk)
