# OllamaCode

A coding assistant powered by **local models** (Ollama) and **MCP** (Model Context Protocol)—all on your machine.

## Features

- **Local-only**: Ollama for reasoning and code generation; no cloud API.
- **MCP tools**: Connect any MCP server; built-in fs, terminal, codebase, tools, and git when no config is present.
- **CLI**: One-off queries or interactive chat (TUI).
- **HTTP & stdio API**: `ollamacode serve` (REST) and `ollamacode protocol` (JSON-RPC over stdin/stdout) for editor integration.
- **Intelligence on by default**: Reasoning, meta-reflection, branch context (git diff in prompt); disable via config if needed.
- **RLM (experimental)**: `--rlm` for recursive language model mode; see [docs/RLM.md](docs/RLM.md).

## Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.com)** installed and running, with a model pulled (e.g. `gpt-oss:20b`, `qwen2.5-coder:32b` for tool calling).

## Installation

**From source** (development):

```bash
git clone https://github.com/jayluxferro/ollamacode.git
cd ollamacode
uv sync
# Optional: bootstrap config and check Ollama
python install.py
# Install CLI on PATH:
uv tool install .
# or: pip install -e .
```

**From a built distribution**: In the repo, run `uv build`, then:

```bash
pip install path/to/ollamacode-1.0.0-py3-none-any.whl
```

Add `[tui]` to the path (e.g. `...whl[tui]`) for the optional TUI deps (Rich, prompt_toolkit). Or install the sdist: `pip install path/to/ollamacode-1.0.0.tar.gz`.

Ensure `ollamacode` is on your PATH (`uv run ollamacode` from source, or `ollamacode` after `uv tool install`).

## Usage

### CLI

**Single query** (built-in MCP by default):

```bash
uv run ollamacode "Explain recursion in one sentence"
```

**Interactive chat** (TUI; requires `ollamacode[tui]`):

```bash
uv run ollamacode
```

**Custom MCP** (config or env):

```bash
uv run ollamacode --mcp-command python --mcp-args examples/demo_server.py "What is 2+3?"
```

Use `OLLAMACODE_MCP_ARGS` for a single stdio server without flags. To use **no MCP**, set `mcp_servers: []` in config.

**Commands**: `serve` (HTTP API), `protocol` (stdio JSON-RPC), `convert-mcp` (Cursor/Claude JSON → YAML). **RLM**: `--rlm` with a prompt or stdin.

**Options** (main ones):

| Option | Description |
|--------|-------------|
| `--config`, `-c` | Config file (default: `ollamacode.yaml` or `.ollamacode/config.yaml`) |
| `--model`, `-m` | Ollama model (`OLLAMACODE_MODEL` / `gpt-oss:20b`) |
| `--stream`, `-s` | Stream tokens to stdout |
| `--file`, `-f` | Prepend file to prompt; use `--lines START-END` for a range |
| `--apply-edits` | Parse `<<EDITS>>` from output, show diff, prompt to apply |
| `--max-messages` | Cap message history (0 = no limit) |
| `--no-mcp` | Skip MCP (faster when tools not needed) |
| `--port` | Port for `serve` (default 8000) |
| `--auto` | Autonomous: no per-tool confirm, more tool rounds |
| `--no-write` | Block write_file and git add/commit/push |

Full list: `ollamacode --help`. **Stop**: Ctrl+C.

### Config file

YAML: `ollamacode.yaml` or `.ollamacode/config.yaml` in the project (or `--config path`). Set **`OLLAMACODE_CONFIG_LOOKUP_PARENT=1`** to search parent dirs.

```yaml
model: gpt-oss:20b
system_prompt_extra: "Optional extra system instructions."
rules_file: .ollamacode/rules.md
max_messages: 0
max_tool_rounds: 20
include_builtin_servers: true

mcp_servers:
  - name: demo
    type: stdio
    command: python
    args: [examples/demo_server.py]
  # HTTP/SSE: type: sse or streamable_http, url: ...
```

See [docs/MCP_SERVERS.md](docs/MCP_SERVERS.md) for full options (prompt_template, memory_auto_context, multi_agent, etc.). Custom server types: register an entry point in `ollamacode.mcp_server_types`; see [API](docs/api.md).

### Built-in MCP (default)

With **no** config and **no** `OLLAMACODE_MCP_ARGS`, OllamaCode starts built-in servers:

- **ollamacode-fs** – read_file, write_file, list_dir, edit_file, multi_edit
- **ollamacode-terminal** – run_command (set `OLLAMACODE_BLOCK_DANGEROUS_COMMANDS=1` to block dangerous patterns)
- **ollamacode-codebase** – search_codebase, get_relevant_files, glob, grep
- **ollamacode-tools** – run_linter, run_tests, fetch_url, etc.
- **ollamacode-git** – git_status, git_diff_*, git_add, git_commit, git_push
- **ollamacode-skills**, **ollamacode-state**, **ollamacode-reasoning**, **ollamacode-screenshot**

To disable MCP: config with `mcp_servers: []`. For full Git (checkout, branch, etc.), add the [official MCP Git server](https://github.com/modelcontextprotocol/servers/tree/main/src/git) in config; example: [examples/ollamacode-full-git.yaml](examples/ollamacode-full-git.yaml).

### Semantic codebase (opt-in)

Built-in codebase does keyword search. For **semantic** (meaning-based) search, pull `nomic-embed-text` and add the semantic server to config:

```yaml
mcp_servers:
  - type: stdio
    command: python
    args: ["-m", "ollamacode.servers.semantic_mcp"]
```

Index via the agent (“Run index_codebase”); then use **semantic_search_codebase**. See [examples/ollamacode-semantic.yaml](examples/ollamacode-semantic.yaml).

### Context and edits

- **@path**: In your message, `@path/to/file` or `@path/to/folder/` injects file contents or folder listing.
- **--file / --lines**: Prepend a file (or line range) to the prompt.
- **--apply-edits**: Model outputs `<<EDITS>>`…`<<END>>`; OllamaCode shows a diff and prompts to apply. Writes are scoped to the workspace root.
- **branch_context**: Set `branch_context: true` (and `branch_context_base: "main"`) to inject git diff into the system prompt.
- **Long chats**: Use `--max-messages N` or `/summary [N]` in the TUI to free context. **--history-file PATH** appends each turn to a file.

### HTTP API and editors

**`ollamacode serve`** (requires `ollamacode[server]`): POST /chat, /chat/continue, /chat/stream, /apply-edits, /rag/index, /rag/query. Optional auth: `serve.api_key` or `OLLAMACODE_SERVE_API_KEY`.

**`ollamacode protocol`**: JSON-RPC on stdin/stdout for editors (ollamacode/chat, ollamacode/chatStream, ollamacode/applyEdits). See [docs/STRUCTURED_PROTOCOL.md](docs/STRUCTURED_PROTOCOL.md) and [docs/OTHER_EDITORS.md](docs/OTHER_EDITORS.md).

**VSCode**: Extension in `editors/vscode` (chat, streaming, apply-edits, tool approvals, diagnostics). Settings: `ollamacode.baseUrl`, `ollamacode.confirmToolCalls`, `ollamacode.multiAgent`.

### Cursor / Claude MCP config

Put MCP servers in `ollamacode.yaml` under `mcp_servers`. Convert existing JSON:

```bash
ollamacode convert-mcp cursor-mcp.json --output ollamacode.yaml
```

See [docs/MCP_SERVERS.md](docs/MCP_SERVERS.md) for YAML format and URL-based (SSE/HTTP) servers.

## Project layout

```
ollamacode/
├── ollamacode/          # Core: agent, cli, config, mcp_client, serve, protocol_server, tui
├── ollamacode/servers/  # Built-in MCP (fs, terminal, codebase, tools, git, semantic)
├── docs/                # WIKI, MCP_SERVERS, STRUCTURED_PROTOCOL, RLM, etc.
├── examples/
├── tests/
├── .githooks/           # Pre-push: ruff (see .githooks/README.md)
└── pyproject.toml
```

[Wiki](docs/WIKI.md) · [CHANGELOG](CHANGELOG.md)

## References

- [Ollama: Tool calling](https://docs.ollama.com/capabilities/tool-calling)
- [Model Context Protocol](https://modelcontextprotocol.io) · [Python SDK](https://github.com/modelcontextprotocol/python-sdk)
