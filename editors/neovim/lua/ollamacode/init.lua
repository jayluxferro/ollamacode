--- OllamaCode Neovim plugin: chat with ollamacode serve (HTTP API).
--- Requires: ollamacode serve running (default http://127.0.0.1:8000).
--- Commands: :OllamaCode [prompt], :OllamaCodeSelection [prompt]

local M = {}

M.config = {
  base_url = "http://127.0.0.1:8000",
  api_key = "",
}

function M.setup(opts)
  if opts then
    if opts.base_url then M.config.base_url = opts.base_url end
    if opts.api_key then M.config.api_key = opts.api_key end
  end
end

local function get_relative_path()
  local file = vim.api.nvim_buf_get_name(0)
  if file == "" then return nil, nil end
  local cwd = vim.fn.getcwd()
  if vim.startswith(file, cwd) then
    local rel = file:sub(#cwd + 2)
    if rel == "" then return ".", nil end
    return rel, nil
  end
  return vim.fn.fnamemodify(file, ":."), nil
end

local function get_selection_range()
  local start_line = vim.api.nvim_buf_get_mark(0, "<")[1]
  local end_line = vim.api.nvim_buf_get_mark(0, ">")[1]
  if start_line == 0 or end_line == 0 then return nil end
  if start_line == end_line then return tostring(start_line) end
  return start_line .. "-" .. end_line
end

local function build_body(prompt, with_selection)
  local file, _ = get_relative_path()
  local body = { message = prompt }
  if file and file ~= "" then
    body.file = file
    if with_selection then
      local lines = get_selection_range()
      if lines then body.lines = lines end
    end
  end
  return vim.json.encode(body)
end

local function curl_headers()
  local headers = { "Content-Type: application/json" }
  if M.config.api_key and M.config.api_key ~= "" then
    table.insert(headers, "Authorization: Bearer " .. M.config.api_key)
    table.insert(headers, "X-API-Key: " .. M.config.api_key)
  end
  return headers
end

--- Open a floating buffer and set its lines. Optionally offer to apply edits.
local function show_reply(content, edits)
  local buf = vim.api.nvim_create_buf(true, true)
  vim.api.nvim_buf_set_option(buf, "bufhidden", "wipe")
  local lines = vim.split(content or "", "\n")
  vim.api.nvim_buf_set_lines(buf, 0, -1, true, lines)
  local width = math.min(80, vim.o.columns - 4)
  local height = math.min(20, vim.o.lines - 4)
  local win = vim.api.nvim_open_win(buf, true, {
    relative = "editor",
    width = width,
    height = height,
    row = 2,
    col = 2,
    style = "minimal",
    border = "rounded",
  })
  vim.api.nvim_win_set_option(win, "wrap", true)

  if edits and #edits > 0 then
    vim.defer_fn(function()
      local choice = vim.fn.confirm(
        "OllamaCode returned " .. #edits .. " edit(s). Apply?",
        "&Yes\n&No",
        2
      )
      if choice == 1 then
        M.apply_edits(edits)
      end
    end, 100)
  end
end

--- Send prompt to ollamacode serve and show reply in a floating window.
--- @param prompt string
--- @param with_selection boolean if true, include current selection (file + lines)
function M.chat(prompt, with_selection)
  if not prompt or prompt == "" then return end
  local body = build_body(prompt, with_selection)
  local url = M.config.base_url:gsub("/$", "") .. "/chat"
  local args = { "curl", "-s", "-X", "POST", url }
  for _, h in ipairs(curl_headers()) do
    table.insert(args, "-H")
    table.insert(args, h)
  end
  table.insert(args, "-d")
  table.insert(args, body)

  vim.fn.jobstart(args, {
    stdout_buffered = true,
    on_stdout = function(_, data)
      local raw = table.concat(data or {}, "")
      local ok, res = pcall(vim.json.decode, raw)
      if ok and res then
        if res.error then
          vim.notify("OllamaCode: " .. res.error, vim.log.levels.ERROR)
          return
        end
        if res.content or res.edits then
          show_reply(res.content or "", res.edits or {})
        end
      else
        vim.notify("OllamaCode: request failed. Is `ollamacode serve` running at " .. M.config.base_url .. "?", vim.log.levels.ERROR)
      end
    end,
    on_stderr = function(_, err)
      if err and #err > 0 then
        vim.notify("OllamaCode: " .. table.concat(err, " "), vim.log.levels.ERROR)
      end
    end,
  })
end

--- Apply edits (array of { path, newText, oldText? }) in the workspace.
--- Paths are relative to cwd.
function M.apply_edits(edits)
  local cwd = vim.fn.getcwd()
  for _, e in ipairs(edits) do
    local path = e.path
    local full = cwd .. "/" .. path:gsub("^/", "")
    vim.fn.bufadd(full)
    pcall(vim.fn.bufload, full)
    local bufnr = vim.fn.bufnr(full)
    if bufnr < 0 then
      vim.notify("OllamaCode: could not open " .. path, vim.log.levels.WARN)
    else
      local new_text = e.newText or ""
      if e.oldText and e.oldText ~= "" then
        local content = table.concat(vim.api.nvim_buf_get_lines(bufnr, 0, -1, true), "\n")
        local start = content:find(e.oldText, 1, true)
        if start then
          local end_pos = start + #e.oldText - 1
          local before = content:sub(1, start - 1)
          local after = content:sub(end_pos + 1)
          new_text = before .. new_text .. after
        end
      end
      vim.api.nvim_buf_set_lines(bufnr, 0, -1, true, vim.split(new_text, "\n"))
      vim.api.nvim_buf_call(bufnr, function()
        vim.cmd("write")
      end)
    end
  end
  vim.notify("OllamaCode: applied " .. #edits .. " edit(s).", vim.log.levels.INFO)
end

--- Setup and user commands (call from plugin entry or ftplugin).
function M.setup_commands()
  vim.api.nvim_create_user_command("OllamaCode", function(opts)
    local prompt = opts.args
    if prompt == "" then
      prompt = vim.fn.input("OllamaCode prompt: ")
    end
    M.chat(prompt, false)
  end, { nargs = "?", desc = "Chat with OllamaCode (current file as context)" })

  vim.api.nvim_create_user_command("OllamaCodeSelection", function(opts)
    local prompt = opts.args
    if prompt == "" then
      prompt = vim.fn.input("OllamaCode prompt (selection as context): ")
    end
    M.chat(prompt, true)
  end, { nargs = "?", desc = "Chat with OllamaCode (selection as context)" })
end

return M
