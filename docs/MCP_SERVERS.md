# Adding MCP Servers to OllamaCode (Cursor / Claude style)

OllamaCode uses the same **Model Context Protocol (MCP)** as Cursor and Claude. You can reuse servers from Cursor or Claude configs by translating them into OllamaCode’s YAML format.

---

## 1. OllamaCode config format

Put MCP servers in **`ollamacode.yaml`** (or `.ollamacode/config.yaml`) under `mcp_servers`:

```yaml
model: gpt-oss:20b

mcp_servers:
  - name: git
    type: stdio
    command: npx
    args: ["-y", "@modelcontextprotocol/server-git"]
  - name: fs
    type: stdio
    command: python
    args: ["-m", "ollamacode.servers.fs_mcp"]
```

- **`name`** (optional): Label for this server (e.g. `git`, `burp`, `ghidra`). Helps you recognise entries in the config; tool names are still prefixed with the server’s own name from MCP `initialize` unless overridden later.
- **`type`**: `stdio` (default), `sse`, or `streamable_http`
- **stdio**: `command` (e.g. `npx`, `python`) and `args` (list of strings)
- **sse / streamable_http**: `url` (required); see [URL-based servers (SSE / HTTP)](#url-based-servers-sse--http) below
- **`env`** (optional, stdio only): extra environment variables for that server (e.g. `MY_KEY: value`)

---

<a id="url-based-servers-sse--http"></a>

## 2. URL-based servers (SSE / HTTP)

Some MCP servers are exposed over **HTTP** (SSE or Streamable HTTP) instead of stdio. You point OllamaCode at a **URL**; no `command` or `args` are used.

### SSE (Server-Sent Events)

Use **`type: sse`** and **`url`** (required). Optional: `headers`, `timeout`, `sse_read_timeout`.

```yaml
mcp_servers:
  - type: sse
    url: http://localhost:8000/sse
  # Optional:
  # headers: { "Authorization": "Bearer TOKEN" }
  # timeout: 5
  # sse_read_timeout: 300
```

### Streamable HTTP

Use **`type: streamable_http`** and **`url`** (required). Optional: `headers`, `timeout`, `sse_read_timeout`, `terminate_on_close`.

```yaml
mcp_servers:
  - type: streamable_http
    url: http://localhost:8000/mcp
  # Optional:
  # headers: { "Authorization": "Bearer TOKEN" }
  # timeout: 30
  # sse_read_timeout: 300
  # terminate_on_close: true
```

### Cursor / Claude config with URLs

If Cursor or Claude config uses a URL (e.g. `url: "http://..."` instead of `command`/`args`), map it like this:

| Cursor / Claude (URL) | OllamaCode (YAML) |
|----------------------|-------------------|
| `"url": "http://localhost:8000/sse"` (SSE) | `type: sse`, `url: http://localhost:8000/sse` |
| `"url": "http://localhost:8000/mcp"` (Streamable HTTP) | `type: streamable_http`, `url: http://localhost:8000/mcp` |
| `headers` (if any) | Same key `headers` on that list item |

Example Cursor/Claude JSON with a URL:

```json
{
  "mcpServers": {
    "my-remote": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

**OllamaCode equivalent**:

```yaml
mcp_servers:
  - type: sse
    url: http://localhost:8000/sse
```

You can mix stdio and URL servers in the same config:

```yaml
mcp_servers:
  - type: stdio
    command: python
    args: ["-m", "ollamacode.servers.fs_mcp"]
  - type: sse
    url: https://your-mcp-host.example.com/sse
```

---

## 3. Cursor MCP config → OllamaCode

Cursor uses a JSON config, often under **Settings → MCP** or in a project `.cursor/mcp.json` (or similar). Example Cursor-style config:

```json
{
  "mcpServers": {
    "git": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-git"]
    },
    "filesystem": {
      "command": "python",
      "args": ["-m", "ollamacode.servers.fs_mcp"]
    }
  }
}
```

**OllamaCode equivalent** (YAML, `ollamacode.yaml`):

```yaml
mcp_servers:
  - type: stdio
    command: npx
    args: ["-y", "@modelcontextprotocol/server-git"]
  - type: stdio
    command: python
    args: ["-m", "ollamacode.servers.fs_mcp"]
```

Rules:

- Cursor’s **`mcpServers`** → OllamaCode’s **`mcp_servers`** (list).
- Each Cursor server object **`command`** + **`args`** → one OllamaCode entry with **`type: stdio`**, **`command`**, **`args`**.
- Cursor server names (e.g. `git`, `filesystem`) are not used in OllamaCode’s list; tool names will be prefixed by the server name returned by MCP `initialize` (e.g. `server-git`, `ollamacode-fs`).
- If Cursor has **`env`** on a server, add the same under **`env`** for that entry in OllamaCode.

---

## 4. Claude / Anthropic MCP config → OllamaCode

Claude Desktop (or Anthropic’s config) often uses JSON like:

```json
{
  "mcp_servers": {
    "git": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-git"]
    }
  }
}
```

**OllamaCode equivalent**:

```yaml
mcp_servers:
  - type: stdio
    command: npx
    args: ["-y", "@modelcontextprotocol/server-git"]
```

Same idea: each server’s **`command`** and **`args`** become one list item with **`type: stdio`**, **`command`**, **`args`**.

---

## 5. Quick reference

| Cursor / Claude field | OllamaCode (YAML) |
|----------------------|-------------------|
| `mcpServers` / `mcp_servers` (object) | `mcp_servers` (list) |
| One server: `command` + `args` | One item: `type: stdio`, `command`, `args` |
| One server: `url` (SSE/HTTP) | One item: `type: sse` or `type: streamable_http`, `url` (required) |
| Server key (Cursor/Claude) | Optional `name` on that list item (e.g. `name: git`) |
| Server `env` (if any, stdio) | Same key `env` on that list item |
| Server `headers` (if any, URL) | Same key `headers` on that list item |

---

## 6. Built-in servers (no config)

If you use **no** config file and **no** `OLLAMACODE_MCP_ARGS`, OllamaCode starts four built-in stdio servers: **fs**, **terminal**, **codebase**, **tools**. To add more (e.g. Git, semantic search), create `ollamacode.yaml` and list **all** servers you want, including the built-ins if you still want them:

```yaml
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
    command: python
    args: ["-m", "ollamacode.servers.tools_mcp"]
  - type: stdio
    command: npx
    args: ["-y", "@modelcontextprotocol/server-git"]
```

---

## 7. Converting from Cursor / Claude JSON (convert-mcp)

OllamaCode includes a **convert-mcp** command that turns Cursor or Claude MCP config (JSON) into OllamaCode YAML.

**From a file:**
```bash
ollamacode convert-mcp cursor-mcp.json --output ollamacode.yaml
ollamacode convert-mcp ~/.config/claude/mcp_config.json -o .ollamacode/config.yaml
```

**From stdin (e.g. paste or pipe):**
```bash
cat cursor-mcp.json | ollamacode convert-mcp -o ollamacode.yaml
ollamacode convert-mcp -o ollamacode.yaml   # then paste JSON and Ctrl+D
```

**Output:** A YAML file (or stdout) with `mcp_servers` as a list. Existing Cursor/Claude keys like `mcpServers` or `mcp_servers` (object with `command`/`args` or `url`) are detected and converted. Optional `env` and `headers` are preserved.

---

## 8. Single server via env (legacy)

For one stdio server without a config file you can set:

```bash
export OLLAMACODE_MCP_ARGS="npx -y @modelcontextprotocol/server-git"
ollamacode "Summarize recent commits"
```

This overrides any config and uses that single server.
