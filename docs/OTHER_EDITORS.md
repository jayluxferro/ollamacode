# Using OllamaCode from Other Editors

You can use OllamaCode from Zed, Sublime, Neovim, or any editor that can run a CLI or call an HTTP API.

## Option 1: Spawn the CLI

Run the `ollamacode` CLI from your editor’s command palette or a keybinding:

- **Zed / Sublime / Neovim**: Configure a command or script that runs `ollamacode` (or `ollamacode --tui`) in a terminal or panel. Pass the current file path or selection as the first argument if your workflow supports it (e.g. `ollamacode "Explain this function"` with selection).
- **Scripts**: Call `ollamacode "your prompt"` from a shell script; use `--no-stream` if you need a single blob of output.

Ensure the editor’s working directory is your project root (or where your `ollamacode.yaml` lives) so MCP tools (fs, terminal, codebase) see the right files.

## Option 2: Local HTTP API (chat-with-selection and apply edits)

Run **`ollamacode serve`** (requires `pip install ollamacode[server]`):

```bash
ollamacode serve --port 8000
```

**POST** to `http://localhost:8000/chat` with JSON:

| Field     | Required | Description |
|----------|----------|-------------|
| `message` | Yes     | Your prompt. |
| `file`   | No       | Path (relative to cwd) to prepend to the prompt (chat-with-selection). |
| `lines`  | No       | Line range with `file`, e.g. `"10-30"` (1-based inclusive). |
| `model`  | No       | Override Ollama model. |

**Response**: `{"content": "Assistant reply...", "edits": [...]}` — if the model output contains a `<<EDITS>>` block, `edits` is the parsed list of `{path, oldText?, newText}` for your editor to apply.

**Example (chat with selection):**

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain this", "file": "src/main.py", "lines": "10-25"}'
```

**Editor integration:** From Neovim, Zed, or Sublime, call the API with the current buffer path and optional line range, then display `content` and optionally apply `edits` (write each `path`/`newText` or patch using `oldText`/`newText`). Start the server from the project root so `file` paths and MCP tools use the correct workspace.

## Tips

- Use a config file (`ollamacode.yaml`) in your project so model and MCP servers are consistent across CLI and HTTP.
- For long-running chats, use the CLI with `--tui` or keep a single `ollamacode serve` process and reuse it for multiple requests.
