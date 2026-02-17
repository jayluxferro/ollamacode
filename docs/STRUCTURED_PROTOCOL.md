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

**Request** (optional for tool approval):  
| Field | Type | Description |
|-------|------|-------------|
| `confirmToolCalls` | boolean | If true (and server has MCP session), the server may pause before each tool and return approval data instead of content. |
| `multiAgent` | boolean | If true, run planner → executor → reviewer flow and return `plan` and `review` fields. |
| `plannerModel` | string | Optional. Planner model override for multi-agent. |
| `executorModel` | string | Optional. Executor model override for multi-agent. |
| `reviewerModel` | string | Optional. Reviewer model override for multi-agent. |
| `multiAgentMaxIterations` | number | Optional. Max review iterations (default 2). |
| `multiAgentRequireReview` | boolean | Optional. If false, skip reviewer stage. |

**Response** — 200 JSON (normal):

| Field | Type | Description |
|-------|------|-------------|
| `content` | string | Full assistant reply. |
| `plan` | string | Optional. Plan from the planner stage when `multiAgent` is true. |
| `review` | object | Optional. Reviewer decision `{ "approved": bool, "issues": [], "suggestions": [] }` when `multiAgent` is true. |
| `edits` | array | Optional. Parsed edits from `<<EDITS>>` in the reply; see [Edit format](#edit-format). |
| `tool_errors` | array | Optional. List of `{tool, arguments_summary, error, hint}` for tools that failed. |
| `error` | string | Set if something went wrong (e.g. "message required"). |

**Response** — 200 JSON (tool approval required):  
When `confirmToolCalls` is true and the agent needs to run a tool (including during `multiAgent` execution), the server may respond with:

| Field | Type | Description |
|-------|------|-------------|
| `toolApprovalRequired` | object | `{ "tool": string, "arguments": object }` — the tool name and arguments pending approval. |
| `approvalToken` | string | Opaque token to send in [POST /chat/continue](#1a-chat-continue-tool-approval) to resume. |

The client should prompt the user (run/skip/edit), then call **POST /chat/continue** with the token and decision. The continue response is either the final `content`/`edits` or another `toolApprovalRequired` + `approvalToken`.

---

### 1a. Chat continue (tool approval)

**Request** — `POST /chat/continue`  
JSON body:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `approvalToken` | string | yes | Token from a previous response that had `toolApprovalRequired`. |
| `decision` | string | no | `"run"` (default), `"skip"`, or `"edit"`. |
| `editedArguments` | object | no | When `decision` is `"edit"`, the new tool arguments JSON to use. |

**Response** — 200 JSON: same as [Chat (non-streaming)](#1-chat-non-streaming) (final `content`, `edits`, `tool_errors`, `error`) or again `toolApprovalRequired` + `approvalToken` if another tool needs approval.

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

- **`ollamacode/chat`** — Params: same as [§1](#1-chat-non-streaming) (`message`, `file?`, `lines?`, `selection?`, `model?`) plus optional multi-agent fields (`multiAgent`, `plannerModel`, `executorModel`, `reviewerModel`, `multiAgentMaxIterations`, `multiAgentRequireReview`). When server is started with `confirm_tool_calls` and MCP session, result may be `{ "toolApprovalRequired": { "tool", "arguments" }, "approvalToken": string }` instead of content; send **ollamacode/chatContinue** to resume. Result (normal): `{ "content": string, "edits"?: array, "tool_errors"?: array, "plan"?: string, "review"?: object, "error"?: string }`.
- **`ollamacode/chatContinue`** — Params: `approvalToken` (from a result that had `toolApprovalRequired`), `decision` ("run" | "skip" | "edit"), `editedArguments`? (object, when decision is "edit"). Result: same as **ollamacode/chat** (final content/edits) or again `toolApprovalRequired` + `approvalToken`.
- **`ollamacode/chatStream`** — Same params as `ollamacode/chat`. **Streaming**: the server sends multiple response lines for the same request id. Each line is a JSON-RPC response with `result.type`:
  - `"chunk"` — `result.content` is a text delta.
  - `"done"` — `result.content` is the full reply, `result.edits?` the parsed edits (stream complete).
  - `"error"` — `result.error` is the error message (stream complete).
- **`ollamacode/applyEdits`** — Params: `edits` (array), `workspaceRoot?`. Result: `{ "applied": number, "error"?: string }`.
- **`ollamacode/diagnostics`** — Params: `workspaceRoot?`, `path?` (file to lint), `linterCommand?` (default `ruff check .`). Result: `{ "diagnostics": array }` where each item is LSP-like: `{ "path", "range": { "start": { "line", "character" }, "end" }, "message", "severity" }`.
- **`ollamacode/complete`** — Params: `prefix` (text to complete), `model?`. Result: `{ "completions": string[] }` (inline completion suggestions).
- **`ollamacode/ragIndex`** — Params: `workspaceRoot?`, `maxFiles?`, `maxCharsPerFile?`. Builds local index and returns `{ "index_path", "workspace_root", "indexed_files", "chunk_count" }`.
- **`ollamacode/ragQuery`** — Params: `query`, `maxResults?`. Returns `{ "results": [ { "path", "chunk_index", "score", "snippet" } ] }`.

Example request (single line to stdin):

```json
{"jsonrpc":"2.0","id":1,"method":"ollamacode/chat","params":{"message":"What does this do?","selection":{"file":"src/foo.py","startLine":10,"endLine":15}}}
```

Example response:

```json
{"jsonrpc":"2.0","id":1,"result":{"content":"This code ...","edits":[]}}
```

---

## 6. HTTP: diagnostics, completions, RAG

- **POST /diagnostics** — Body: `{ "workspaceRoot"?, "path"?, "linterCommand"? }`. Response: `{ "diagnostics": [ { "path", "range", "message", "severity" }, ... ] }` (LSP-like; for editor squiggles).
- **POST /complete** — Body: `{ "prefix", "model"? }`. Response: `{ "completions": [ string ] }` (inline/ghost-text suggestion).
- **POST /rag/index** — Body: `{ "workspaceRoot"?, "maxFiles"?, "maxCharsPerFile"? }`. Response: `{ "index_path", "workspace_root", "indexed_files", "chunk_count" }`.
- **POST /rag/query** — Body: `{ "query", "maxResults"? }`. Response: `{ "results": [ { "path", "chunk_index", "score", "snippet" } ] }`.

---

## 7. Summary

- **Chat-with-selection**: send `message` plus `file`+`lines` or `selection`; get `content` and optional `edits`.
- **Streaming**: use `POST /chat/stream` (HTTP only) with the same body; consume SSE events.
- **Apply-edits**: send `POST /apply-edits` (HTTP) or `ollamacode/applyEdits` (stdio) with `edits` (and optional `workspaceRoot`).
- **Diagnostics**: send `POST /diagnostics` (HTTP) or `ollamacode/diagnostics` (stdio) for linter results.
- **Completions**: `POST /complete` or `ollamacode/complete` for inline completion.
- **RAG**: `POST /rag/index` + `POST /rag/query` (HTTP) or `ollamacode/ragIndex` + `ollamacode/ragQuery` (stdio) for local retrieval.

**HTTP**: `ollamacode serve` (see README). Optional auth: set `serve.api_key` in config and send `Authorization: Bearer <key>` or `X-API-Key: <key>`.

**Stdio**: `ollamacode protocol` for JSON-RPC (one request per line). Use `ollamacode/chatStream` for streaming; multiple response lines per request. Use `ollamacode/diagnostics`, `ollamacode/complete`, `ollamacode/ragIndex`, and `ollamacode/ragQuery` for IDE-style diagnostics/completions/retrieval.
