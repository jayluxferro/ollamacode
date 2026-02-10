# OllamaCode VS Code extension

Minimal VS Code extension to use [OllamaCode](https://github.com/your-org/ollamacode) from the editor. It talks to **`ollamacode serve`** over HTTP (chat + apply-edits).

## Prerequisites

1. **Ollama** running (`ollama serve`) with a model (e.g. `ollama pull llama3.2`).
2. **OllamaCode** installed (`pip install ollamacode[server]` or `uv add ollamacode[server]`).
3. Start the server from your project root:
   ```bash
   ollamacode serve --port 8000
   ```
   Leave it running; the extension sends requests to `http://localhost:8000` by default.

## Install the extension

- **From source**: open the `editors/vscode` folder in VS Code, run **Run > Run Extension** (F5), or package a `.vsix` with **vsce package** and install the `.vsix`.
- **From workspace**: copy this folder to `~/.vscode/extensions/ollamacode-0.1.0` (or use **Install from VSIX** after packaging).

## Commands

| Command | Description |
|--------|-------------|
| **OllamaCode: Chat** | Ask a question; optionally uses the current file as context. Reply is shown in the OllamaCode output channel. If the reply contains edits, you are prompted to apply them. |
| **OllamaCode: Chat with selection** | Same as Chat, but includes the current selection as context (file + line range). |
| **OllamaCode: Chat (streaming)** | Same as Chat, but uses POST /chat/stream; reply appears token-by-token in the output channel. |
| **OllamaCode: Chat with selection (streaming)** | Same as Chat with selection, but streaming. |
| **OllamaCode: Apply edits from clipboard/response** | Paste JSON from clipboard: either `[{ "path": "...", "newText": "..." }]` or `{ "edits": [ ... ] }`, then run to apply edits in the workspace. |

## Settings

- **ollamacode.baseUrl** — Base URL of the server (default: `http://localhost:8000`).
- **ollamacode.apiKey** — If you set `serve.api_key` in OllamaCode config, set the same value here; the extension sends it as `Authorization: Bearer <key>` and `X-API-Key: <key>`.

## Protocol

The extension uses the same HTTP API as in the main repo:

- **POST /chat** — body: `{ "message", "file?", "lines?" }`; response: `{ "content", "edits?" }`.
- **POST /chat/stream** — same body; Server-Sent Events: `{ "type": "chunk", "content" }` then `{ "type": "done", "content", "edits?" }`.
- Edits are applied with VS Code’s `WorkspaceEdit` (path relative to workspace root).

See **docs/STRUCTURED_PROTOCOL.md** and **docs/OTHER_EDITORS.md** in the OllamaCode repo for full details.
