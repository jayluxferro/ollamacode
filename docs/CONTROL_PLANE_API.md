# Control Plane API

OllamaCode now exposes a lightweight local control-plane surface over both HTTP and the stdio protocol.

## Browser App

When `ollamacode serve` is running:

- `GET /app`
- `GET /app.css`
- `GET /app.js`

This serves the built-in browser control console for:

- browsing local and remote workspaces
- creating remote workspace entries
- checking remote workspace health
- browsing sessions
- branching, importing, exporting, and deleting sessions
- restoring recent checkpoints
- chatting through the same `/chat` and `/chat/continue` API

## HTTP Routes

### Core chat

- `POST /chat`
- `POST /chat/continue`
- `POST /chat/stream`

### Sessions

- `GET /sessions`
- `POST /sessions`
- `POST /sessions/import`
- `GET /sessions/{id}`
- `GET /sessions/{id}/messages`
- `DELETE /sessions/{id}`
- `GET /sessions/{id}/export`
- `GET /sessions/{id}/todos`
- `POST /sessions/{id}/branch`
- `POST /sessions/{id}/fork`
- `GET /sessions/{id}/checkpoints`
- `POST /sessions/{id}/rewind`

### Workspace

- `GET /workspace`
- `GET /workspaces`
- `POST /workspaces`
- `GET /workspaces/{id}`
- `PATCH /workspaces/{id}`
- `DELETE /workspaces/{id}`
- `GET /workspaces/{id}/health`
- `GET|POST|DELETE|PATCH|PUT /workspaces/{id}/proxy/{target...}`

### Other existing routes

- `POST /apply-edits`
- `POST /diagnostics`
- `POST /complete`
- `POST /rag/index`
- `POST /rag/query`

## Protocol Methods

### Chat

- `ollamacode/chat`
- `ollamacode/chatContinue`
- `ollamacode/chatStream`

### Sessions

- `ollamacode/sessionList`
- `ollamacode/sessionGet`
- `ollamacode/sessionMessages`
- `ollamacode/sessionCreate`
- `ollamacode/sessionDelete`
- `ollamacode/sessionExport`
- `ollamacode/sessionImport`
- `ollamacode/sessionTodo`
- `ollamacode/sessionBranch`
- `ollamacode/sessionFork`
- `ollamacode/sessionCheckpoints`
- `ollamacode/sessionRestoreCheckpoint`

### Workspace

- `ollamacode/workspaceInfo`
- `ollamacode/workspaceList`
- `ollamacode/workspaceGet`
- `ollamacode/workspaceCreate`
- `ollamacode/workspaceUpdate`
- `ollamacode/workspaceDelete`
- `ollamacode/workspaceHealth`
- `ollamacode/workspaceProxy`

### Other methods

- `ollamacode/applyEdits`
- `ollamacode/diagnostics`
- `ollamacode/complete`
- `ollamacode/ragIndex`
- `ollamacode/ragQuery`

## Remote Workspace Model

Remote workspaces are stored in the local workspace registry and currently support:

- name
- type (`local` or `remote`)
- `base_url`
- optional API key
- optional owner
- optional role
- persisted `last_status`
- persisted `last_error`

The current proxy/orchestration model is intentionally lightweight:

- it is local-first
- it forwards requests to another OllamaCode HTTP server
- it does not implement a separate event bus or distributed scheduler

That keeps it maintainable while still enabling multi-workspace usage and browser-based operations.
