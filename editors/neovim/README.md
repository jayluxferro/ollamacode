# ollamacode.nvim

Minimal Neovim plugin to chat with [OllamaCode](https://github.com/your-org/ollamacode) via its HTTP API. Requires **`ollamacode serve`** running (e.g. `ollamacode serve --port 8000` from your project root).

## Prerequisites

- **Ollama** running with a model (e.g. `ollama pull llama3.2`).
- **OllamaCode** with server: `pip install ollamacode[server]` or `uv add ollamacode[server]`.
- Start the server from your project root: `ollamacode serve --port 8000`.
- **curl** (used to send HTTP requests).

## Install

- **From this repo (e.g. packer):**
  ```lua
  use { path = "/path/to/OllamaCode/editors/neovim", as = "ollamacode.nvim" }
  ```
- **Or copy** the `editors/neovim` folder into your Neovim config under a name like `pack/ollamacode/start/ollamacode.nvim`, or clone the OllamaCode repo and add the `editors/neovim` directory to your `runtimepath`.

No external Lua dependencies; Neovim 0.8+ (for `vim.json` and `vim.fn.jobstart`).

## Config (optional)

In your `init.lua`:

```lua
require("ollamacode").setup({
  base_url = "http://127.0.0.1:8000",
  api_key = "",  -- set if your ollamacode serve uses serve.api_key
})
```

Call `setup` before the plugin registers commands (e.g. before loading the plugin, or ensure your config is loaded first).

## Commands

| Command | Description |
|--------|-------------|
| **`:OllamaCode [prompt]`** | Send a message to OllamaCode with the current file as context. If `prompt` is omitted, you are prompted for it. Reply opens in a floating window; if the reply contains edits, you are asked whether to apply them. |
| **`:OllamaCodeSelection [prompt]`** | Same, but uses the current selection (visual or line range) as context (file + lines). |

Examples:

- `:OllamaCode Explain this function`
- Select lines, then `:OllamaCodeSelection Fix the bug`
- `:OllamaCode` then enter the prompt when asked

## Protocol

The plugin POSTs to `{base_url}/chat` with JSON:

- `message` (required)
- `file` (optional, current buffer path relative to cwd)
- `lines` (optional, e.g. `"10-20"` for selection)

Response: `{ "content": "...", "edits": [ { "path", "newText", "oldText?" } ] }`. Edits are applied in the workspace (paths relative to cwd). See **docs/STRUCTURED_PROTOCOL.md** and **docs/OTHER_EDITORS.md** in the OllamaCode repo.
