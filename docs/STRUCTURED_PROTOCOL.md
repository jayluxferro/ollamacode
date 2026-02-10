# OllamaCode editor protocol

Optional, structured API for **chat-with-selection** and **apply-edits** so editors (VS Code, Cursor, Neovim, etc.) can integrate without implementing the full CLI.

- **Transport**: HTTP (see `ollamacode serve`) or **JSON-RPC over stdio** (see [Stdio](#5-stdio-json-rpc)).
- **Workspace**: All paths are relative to a workspace root (server cwd or `workspaceRoot` when provided).

---

## 1. Chat (non-streaming)

**Request** — `POST /chat`  
JSON body:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | yes | User message (e.g. "Explain this" or "Fix the bug here"). |
| `model` | string | no | Override default model. |
| `file` | string | no | Path relative to workspace (for chat-with-selection). |
| `lines` | string | no | Line range: `"start-end"` or `"start:end"` (1-based inclusive). Use with `file`. |
| `selection` | object | no | Alternative to `file`+`lines`: `{ "file": string, "startLine": number, "endLine": number }` (1-based inclusive). |

If both `file`/`lines` and `selection` are present, `selection` wins. The server prepends the selected file content (or line range) to the message as context.

**Response** — 200 JSON:

| Field | Type | Description |
|-------|------|-------------|
| `content` | string | Full assistant reply. |
| `edits` | array | Optional. Parsed edits from `<<EDITS>>` in the reply; see [Edit format](#edit-format). |
| `error` | string | Set if something went wrong (e.g. "message required"). |

---

## 2. Chat (streaming)

**Request** — `POST /chat/stream`  
Same JSON body as [Chat (non-streaming)](#1-chat-non-streaming).

**Response** — 200, `Content-Type: text/event-stream`  
Server-Sent Events; each event is `data: <JSON>\n\n` with one of:

- `{ "type": "chunk", "content": "..." }` — text delta.
- `{ "type": "done", "content": "<full text>", "edits": [ ... ] }` — final content and parsed edits.
- `{ "type": "error", "error": "..." }` — error message.

---

## 3. Edit format

Edits are extracted from model output (`<<EDITS>>` … `<<END>>`) and returned as an array of:

| Field | Type | Description |
|-------|------|-------------|
| `path` | string | File path relative to workspace. |
| `newText` | string | New content (required). |
| `oldText` | string | Optional. If present, replace first occurrence of `oldText` with `newText`; if omitted, replace entire file. |

This is the same format the CLI uses. Editors can apply these client-side or call [Apply edits](#4-apply-edits-optional) to have the server apply them.

---

## 4. Apply edits (optional)

**Request** — `POST /apply-edits`  
JSON body:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `edits` | array | yes | Same shape as [Edit format](#3-edit-format). |
| `workspaceRoot` | string | no | Override workspace root (default: server cwd). |

**Response** — 200 JSON:

| Field | Type | Description |
|-------|------|-------------|
| `applied` | number | Number of files successfully written. |
| `error` | string | Set if a fatal error occurred. |

---

## 5. Stdio (JSON-RPC)

Run the protocol over stdin/stdout so editors can spawn the process without HTTP:

```bash
ollamacode protocol
```

(Use `ollamacode --no-mcp protocol` to run without MCP tools.)

**Format**: One JSON-RPC 2.0 request per line on stdin; one JSON-RPC 2.0 response per line on stdout.

**Methods**:

- **`ollamacode/chat`** — Params: same as [§1](#1-chat-non-streaming) (`message`, `file?`, `lines?`, `selection?`, `model?`). Result: `{ "content": string, "edits"?: array, "error"?: string }`.
- **`ollamacode/chatStream`** — Same params as `ollamacode/chat`. **Streaming**: the server sends multiple response lines for the same request id. Each line is a JSON-RPC response with `result.type`:
  - `"chunk"` — `result.content` is a text delta.
  - `"done"` — `result.content` is the full reply, `result.edits?` the parsed edits (stream complete).
  - `"error"` — `result.error` is the error message (stream complete).
- **`ollamacode/applyEdits`** — Params: `edits` (array), `workspaceRoot?`. Result: `{ "applied": number, "error"?: string }`.
- **`ollamacode/diagnostics`** — Params: `workspaceRoot?`, `path?` (file to lint), `linterCommand?` (default `ruff check .`). Result: `{ "diagnostics": array }` where each item is LSP-like: `{ "path", "range": { "start": { "line", "character" }, "end" }, "message", "severity" }`.
- **`ollamacode/complete`** — Params: `prefix` (text to complete), `model?`. Result: `{ "completions": string[] }` (inline completion suggestions).

Example request (single line to stdin):

```json
{"jsonrpc":"2.0","id":1,"method":"ollamacode/chat","params":{"message":"What does this do?","selection":{"file":"src/foo.py","startLine":10,"endLine":15}}}
```

Example response:

```json
{"jsonrpc":"2.0","id":1,"result":{"content":"This code ...","edits":[]}}
```

---

## 6. HTTP: diagnostics and completions

- **POST /diagnostics** — Body: `{ "workspaceRoot"?, "path"?, "linterCommand"? }`. Response: `{ "diagnostics": [ { "path", "range", "message", "severity" }, ... ] }` (LSP-like; for editor squiggles).
- **POST /complete** — Body: `{ "prefix", "model"? }`. Response: `{ "completions": [ string ] }` (inline/ghost-text suggestion).

---

## 7. Summary

- **Chat-with-selection**: send `message` plus `file`+`lines` or `selection`; get `content` and optional `edits`.
- **Streaming**: use `POST /chat/stream` (HTTP only) with the same body; consume SSE events.
- **Apply-edits**: send `POST /apply-edits` (HTTP) or `ollamacode/applyEdits` (stdio) with `edits` (and optional `workspaceRoot`).
- **Diagnostics**: send `POST /diagnostics` (HTTP) or `ollamacode/diagnostics` (stdio) for linter results; **Completions**: `POST /complete` or `ollamacode/complete` for inline completion.

**HTTP**: `ollamacode serve` (see README). Optional auth: set `serve.api_key` in config and send `Authorization: Bearer <key>` or `X-API-Key: <key>`.

**Stdio**: `ollamacode protocol` for JSON-RPC (one request per line). Use `ollamacode/chatStream` for streaming; multiple response lines per request. Use `ollamacode/diagnostics` and `ollamacode/complete` for IDE-style diagnostics and inline completions.
