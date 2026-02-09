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

Optional auth: set **`OLLAMACODE_SERVE_API_KEY`** or in config under **`serve.api_key`**; then send **`Authorization: Bearer <key>`** or **`X-API-Key: <key>`** with each request. If the key is set and the header is missing or wrong, the server returns 401.

**POST** to `http://localhost:8000/chat` with JSON:

| Field     | Required | Description |
|----------|----------|-------------|
| `message` | Yes     | Your prompt. |
| `file`   | No       | Path (relative to cwd) to prepend to the prompt (chat-with-selection). |
| `lines`  | No       | Line range with `file`, e.g. `"10-30"` (1-based inclusive). |
| `model`  | No       | Override Ollama model. |

**Response**: `{"content": "Assistant reply...", "edits": [...]}`

- **content** (string): Full assistant reply.
- **edits** (array, optional): Present when the model output includes a `<<EDITS>>` … `<<END>>` block. Each edit is `{"path": "relative/path", "oldText": "optional match", "newText": "replacement"}`. Apply by writing `newText` to `path`, or by patching: replace `oldText` with `newText` in the file (if `oldText` is given). Paths are relative to the server’s workspace root.

**Example (chat with selection):**

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain this", "file": "src/main.py", "lines": "10-25"}'
```

**Editor integration:** From Neovim, Zed, or Sublime, call the API with the current buffer path and optional line range, then display `content` and optionally apply `edits` (write each `path`/`newText` or patch using `oldText`/`newText`). Start the server from the project root so `file` paths and MCP tools use the correct workspace.

### Streaming: POST /chat/stream

For live token-by-token output, **POST** to `http://localhost:8000/chat/stream` with the same JSON body as `/chat`. The response is **Server-Sent Events** (SSE):

- Each event is a line of the form `data: <json>`, where the payload is `{"type": "chunk", "content": "..."}` for a text fragment, or `{"type": "done", "content": "...", "edits": [...]}` for the final message (same `content`/`edits` as the non-streaming response).
- Use `EventSource` (browser) or an SSE client in your editor to consume the stream and update the UI as chunks arrive.

## Editor snippets

Minimal examples to call the HTTP API and (optionally) apply edits. Start `ollamacode serve` from the project root first.

### Neovim (Lua)

Send selection or buffer to OllamaCode and show the reply in a split. Requires `ollamacode serve` running.

```lua
-- Example: send selection (or buffer) to OllamaCode and show reply in a new buffer
local function ollamacode_chat(prompt)
  local file = vim.api.nvim_buf_get_name(0)
  local start_line, end_line = vim.api.nvim_buf_get_mark(0, "<")[1], vim.api.nvim_buf_get_mark(0, ">")[1]
  local body = vim.json.encode({
    message = prompt,
    file = vim.fn.fnamemodify(file, ":."),
    lines = (start_line ~= end_line) and (start_line .. "-" .. end_line) or nil,
  })
  vim.fn.jobstart({ "curl", "-s", "-X", "POST", "http://127.0.0.1:8000/chat",
    "-H", "Content-Type: application/json", "-d", body },
    { stdout_buffered = true, on_stdout = function(_, data)
      local ok, res = pcall(vim.json.decode, table.concat(data))
      if ok and res.content then
        local buf = vim.api.nvim_create_buf(true, true)
        vim.api.nvim_buf_set_lines(buf, 0, -1, true, vim.split(res.content, "\n"))
        vim.api.nvim_open_win(buf, true, { relative = "editor", width = 80, height = 20, row = 2, col = 2 })
      end
    end})
end
-- Usage: select lines, then :lua ollamacode_chat("Explain this")
```

### Zed

Run a command that calls the API (e.g. from a task or keybinding). Example shell script you can bind to a key or run from the terminal:

```bash
# Save as scripts/ollamacode-ask.sh; chmod +x; run from project root with selection or prompt
FILE="${OLLAMACODE_FILE:-}"
LINES="${OLLAMACODE_LINES:-}"
PROMPT="${1:-Explain this}"
BODY="{\"message\": \"$PROMPT\"}"
[ -n "$FILE" ] && BODY="{\"message\": \"$PROMPT\", \"file\": \"$FILE\", \"lines\": \"$LINES\"}"
curl -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "$BODY" | jq -r '.content'
```

Set `OLLAMACODE_FILE` and `OLLAMACODE_LINES` from your editor (e.g. current file path and selected line range) before running the script.

### Sublime Text

Use a build system or plugin to POST the current buffer (or selection) to OllamaCode. Example build system (save as `ollamacode.sublime-build`; adjust `cmd` for your OS):

```json
{
  "shell_cmd": "curl -s -X POST http://127.0.0.1:8000/chat -H 'Content-Type: application/json' -d \"{\\\"message\\\": \\\"Explain this\\\", \\\"file\\\": \\\"$file\\\"}\" | jq -r .content",
  "working_dir": "$project_path"
}
```

Or use a small Python plugin that reads the selection, POSTs to `http://127.0.0.1:8000/chat`, and opens the response in a new tab (Sublime’s API allows getting selection and file path; wire those into the JSON body).

## Tips

- Use a config file (`ollamacode.yaml`) in your project so model and MCP servers are consistent across CLI and HTTP.
- For long-running chats, use the CLI with `--tui` or keep a single `ollamacode serve` process and reuse it for multiple requests.
