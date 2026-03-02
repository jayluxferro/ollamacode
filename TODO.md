# OllamaCode — TODO & Status Tracker

> Comprehensive tracker for the 148-task hardening sprint + 15 feature gaps.
> Last updated: 2026-03-02

---

## Summary

| Category | Range | Done | Total |
|----------|-------|------|-------|
| Critical Security Fixes | 1–10 | 10 | 10 |
| Critical Crash Fixes | 11–20 | 10 | 10 |
| Race Condition Fixes | 21–30 | 10 | 10 |
| Logic Error Fixes | 31–45 | 15 | 15 |
| Missing Validation | 46–60 | 15 | 15 |
| Error Handling | 61–75 | 15 | 15 |
| Design Improvements | 76–90 | 15 | 15 |
| TUI Improvements | 91–100 | 10 | 10 |
| Test Coverage | 101–120 | 20 | 20 |
| Project Config | 121–128 | 8 | 8 |
| Feature Gaps | 129–148 | 20 | 20 |
| **TOTAL** | | **148** | **148** |

---

## Tasks 1–10: Critical Security Fixes

- [x] **1.** Symlink traversal in `fs_mcp.py` — `os.path.realpath()` before open
- [x] **2.** Shell injection in `terminal_mcp.py` — `shlex.split()` for allowlist
- [x] **3.** Regex DoS in `codebase_mcp.py` — bounded regex with `re.TIMEOUT` / length limits
- [x] **4.** SSRF in `webfetch_mcp.py` — private IP blocklist, scheme validation
- [x] **5.** Path traversal null bytes — null-byte check in `sandbox.py`
- [x] **6.** Sensitive dotfile exposure — `sandbox.py` blocks `.env`, `.ssh`, `.aws`, etc.
- [x] **7.** System directory write protection — `/etc`, `/usr`, `/bin` blocked
- [x] **8.** Secret file permissions — keyfile + DB at `0o600`
- [x] **9.** Sandbox violation logging — audit trail in `~/.ollamacode/sandbox_violations.log`
- [x] **10.** Input sanitization — model names, URLs, body size limits validated

## Tasks 11–20: Critical Crash Fixes

- [x] **11.** `asyncio.run()` in nested loop — guarded with `nest_asyncio` or running-loop check
- [x] **12.** `thread.join()` timeout — bounded join in scheduler/channels
- [x] **13.** Resource leaks — explicit close on MCP sessions, DB connections
- [x] **14.** Assert → RuntimeError — `before_tool_call` assertion replaced
- [x] **15.** Silent `except: pass` → `logger.debug` — all callback handlers
- [x] **16.** Double connection in `load_session()` — single connection fix
- [x] **17.** Import-time readline init — moved to module level once
- [x] **18.** `/quit` deadlock — `None` sentinel instead of string
- [x] **19.** Session DB failure warning — user-visible error instead of silent fail
- [x] **20.** Spinner monotonic clock — `time.monotonic()` at 8fps

## Tasks 21–30: Race Condition Fixes

- [x] **21.** Approval lock in `before_tool_call` — threading lock for concurrent calls
- [x] **22.** TTL cleanup race — atomic expiry check in rate limiter
- [x] **23.** `st_mtime_ns` for state cache — nanosecond precision avoids stale reads
- [x] **24.** Status batch update — grouped state writes to avoid partial updates
- [x] **25.** Atomic file writes — `_atomic_write()` with tempfile + rename
- [x] **26.** Thread-safe rate limiter — lock around `check()` + `record_tokens()`
- [x] **27.** TOCTOU in rate limiter — `check()` no longer writes stale token count
- [x] **28.** Session save concurrency — DB transaction isolation
- [x] **29.** Plugin event delivery — synchronous dispatch prevents reordering
- [x] **30.** Config reload safety — validated before swap

## Tasks 31–45: Logic Error Fixes

- [x] **31.** Duplicate `on_tool_end` calls — removed second call in `run_agent_loop`
- [x] **32.** Duplicate `tool_errors_total` increment — was 2x, fixed to 1x
- [x] **33.** Missing error increment in stream+confirm path — added
- [x] **34.** Silent fallback on MCP connect — now raises/logs
- [x] **35.** Tool ID collision — UUID-based tool IDs
- [x] **36.** Fire-and-forget task exceptions — caught and logged
- [x] **37.** Stale tool cache on reconnect — cache cleared on session change
- [x] **38.** History markdown caching — `_history_md_cache` avoids re-render
- [x] **39.** O(n²) join in TUI streaming — incremental append
- [x] **40.** Chat panel exchange limit — raised 2→5
- [x] **41.** Tool trace limit — raised 5→20 (matches `max_tool_rounds`)
- [x] **42.** Refresh Hz default — raised 5→12
- [x] **43.** `auto_refresh=False` on Live — `_tick` as sole render driver
- [x] **44.** `make_update` intermediate frames — only render on `done=True`
- [x] **45.** Status line rendering — `Text.from_markup` instead of Markdown

## Tasks 46–60: Missing Validation

- [x] **46.** Model name validation — alphanumeric + colon + dash + dot
- [x] **47.** URL validation — scheme + netloc required
- [x] **48.** Body size limits — configurable max request body
- [x] **49.** MCP URL validation — block private IPs for SSE/HTTP
- [x] **50.** Config key validation — reject unknown keys with warning
- [x] **51.** Tool argument validation — JSON schema check before execution
- [x] **52.** Session ID format — UUID validation on load
- [x] **53.** File path validation — reject null bytes, normalize traversal
- [x] **54.** Terminal command validation — reject null bytes in READONLY
- [x] **55.** Timeout clamping — `[1, 300]` seconds for terminal commands
- [x] **56.** Rate limit config validation — positive RPM/TPD values
- [x] **57.** Provider config validation — required fields per provider type
- [x] **58.** Channel token validation — non-empty bot tokens
- [x] **59.** Tunnel type validation — known tunnel types only
- [x] **60.** Scheduler cron validation — 5-field standard cron syntax

## Tasks 61–75: Error Handling

- [x] **61.** Session transaction wrapping — all DB ops in transactions
- [x] **62.** JSON decode logging — log malformed responses instead of crash
- [x] **63.** Hook injection prevention — sanitize shell arguments
- [x] **64.** Health check timeouts — bounded HTTP requests
- [x] **65.** MCP reconnect on failure — retry with backoff
- [x] **66.** Provider timeout handling — configurable request timeouts
- [x] **67.** Graceful shutdown — signal handlers for SIGINT/SIGTERM
- [x] **68.** Disk full handling — check space before large writes
- [x] **69.** Network error messages — user-friendly error text
- [x] **70.** Plugin error isolation — exceptions in plugins don't crash host
- [x] **71.** Channel reconnect — auto-reconnect on disconnect
- [x] **72.** Scheduler error recovery — failed tasks don't block next run
- [x] **73.** Tunnel restart on failure — detect process death, restart
- [x] **74.** Config parse error recovery — fall back to defaults
- [x] **75.** Log rotation — bounded log file sizes

## Tasks 76–90: Design Improvements

- [x] **76.** Atomic file writes — `tempfile` + `os.rename` pattern
- [x] **77.** Structured git returns — JSON output from git MCP
- [x] **78.** Per-session reasoning — isolated reasoning context
- [x] **79.** `_chat_helpers.py` extraction — shared serve/protocol helpers
- [x] **80.** Provider registry — `get_provider()` factory with type dispatch
- [x] **81.** Config layering — env vars > CLI flags > config file > defaults
- [x] **82.** Checkpointing system — pre/post file snapshots with restore
- [x] **83.** State mtime cache — avoid disk reads on every access
- [x] **84.** MCP tool cache — per-session cache avoids network RTT
- [x] **85.** Embedding LRU cache — in-memory cache for vector embeddings
- [x] **86.** FTS5 graceful degradation — falls back to LIKE scan
- [x] **87.** Markdown-aware chunking — preserves heading context in vectors
- [x] **88.** Hybrid search scoring — keyword (BM25) + vector (cosine)
- [x] **89.** Provider-specific effort mapping — low/medium/high per provider
- [x] **90.** Event bus design — typed events with sync dispatch

## Tasks 91–100: TUI Improvements

- [x] **91.** Markup escape — prevent Rich markup injection from user input
- [x] **92.** `shlex` command parsing — proper slash command tokenization
- [x] **93.** Theme system — `OLLAMACODE_THEME` env var, color dict
- [x] **94.** Fuzzy `@mentions` — tab completion for tool/model names
- [x] **95.** Sidebar panel — session info, tokens, tool calls, duration
- [x] **96.** Pygments diff rendering — syntax-highlighted code diffs
- [x] **97.** 50+ slash commands — full command palette
- [x] **98.** Multi-agent commands — `/multi`, `/agents`, `/agents_show`
- [x] **99.** Voice commands — `/listen`, `/say` for audio I/O
- [x] **100.** Profile/analysis commands — `/profile`, `/plan`, `/fix`, `/test`

## Tasks 101–120: Test Coverage (579 tests)

- [x] **101.** `test_security_fixes.py` — 52 tests, symlink/injection/SSRF/traversal
- [x] **102.** `test_agent.py` — 20 tests, agent loop/callbacks/streaming
- [x] **103.** `test_config.py` — 29 tests, loading/validation/merging
- [x] **104.** `test_anthropic_provider.py` — 21 tests, Claude provider
- [x] **105.** `test_openai_compat.py` — 25 tests, OpenAI-compatible
- [x] **106.** `test_fs_mcp.py` — 24 tests, filesystem sandbox
- [x] **107.** `test_secrets.py` — 18 tests, encryption/decryption
- [x] **108.** `test_state.py` — 21 tests, persistent state/mtime cache
- [x] **109.** `test_protocol_server.py` — 9 tests, WebSocket/HTTP
- [x] **110.** `test_git_mcp.py` — 21 tests, git operations
- [x] **111.** `test_hooks.py` — 21 tests, pre/post tool hooks
- [x] **112.** `test_sandbox.py` — 20 tests, path/command checks
- [x] **113.** `test_rate_limit.py` — 13 tests, sliding window/tokens
- [x] **114.** `test_terminal_mcp.py` — 14 tests, allowlist/blocklist
- [x] **115.** `test_mcp_client.py` — 16 tests, MCP connections
- [x] **116.** `test_codebase_mcp.py` — 18 tests, code search
- [x] **117.** `test_semantic_mcp.py` — 18 tests, semantic search
- [x] **118.** `test_ollama_provider.py` — 11 tests, Ollama client
- [x] **119.** `test_apple_fm_provider.py` — 14 tests, Apple FM
- [x] **120.** `test_checkpoints.py` — 8 tests, checkpoint create/restore

## Tasks 121–128: Project Config

- [x] **121.** Strict Pyright — `typeCheckingMode = "strict"`, Python 3.11
- [x] **122.** Expanded Ruff rules — E, F, I, W, C4, B, UP, SIM
- [x] **123.** Pre-commit hooks — ruff format + check
- [x] **124.** Bandit CI — security scanning, excludes tests
- [x] **125.** Coverage enforcement — 70% minimum
- [x] **126.** pytest-timeout — test timeout enforcement
- [x] **127.** Dev dependencies — pytest, pytest-asyncio, pytest-cov, ruff, pyright
- [x] **128.** Entry points — `ollamacode = ollamacode.cli:main`

## Tasks 129–148: Feature Gap Implementations

- [x] **129.** LSP client — `lsp_client.py`, stdio JSON-RPC, definition/references/hover/symbols
- [x] **130.** Context compaction — `compaction.py`, LLM-based message summarization
- [x] **131.** Session branching — `sessions.py`, `branch_session()` + `fork_session()`
- [x] **132.** Permissions system — `permissions.py`, `PermissionManager`, ALLOW/DENY/ASK
- [x] **133.** Doom loop detection — `agent.py`, `_DoomLoopDetector` class
- [x] **134.** Question tool — TUI `/ask` integration, confirmation prompts
- [x] **135.** Patch/diff tool — `edits.py`, structured file edits
- [x] **136.** Web fetch tool — `webfetch_mcp.py`, SSRF-guarded URL fetching
- [x] **137.** Auto-format — integrated with terminal MCP
- [x] **138.** File watcher — `file_watcher.py`, watchdog/polling, daemon thread
- [x] **139.** Plugin framework — `plugins.py`, EventBus, PluginManager
- [x] **140.** Agent modes — `agent_modes.py`, BUILD/PLAN/REVIEW with tool restrictions
- [x] **141.** Custom commands — `custom_commands.py`, config-driven, template expansion
- [x] **142.** Model variants — `model_variants.py`, reasoning effort levels
- [x] **143.** Session sharing — export/import in TUI (`/export`, `/import`)
- [x] **144.** Gemini provider — `gemini_provider.py`, Google Gemini API
- [x] **145.** Bedrock provider — `bedrock_provider.py`, AWS Bedrock
- [x] **146.** Azure provider — `azure_provider.py`, Azure OpenAI
- [x] **147.** Apple FM provider — `apple_fm_provider.py`, on-device models
- [x] **148.** Incremental search — `semantic_mcp.py` + `codebase_mcp.py` integration

---

## Architecture Overview

### Core Modules (22,306 lines)

| Module | Lines | Purpose |
|--------|-------|---------|
| `tui.py` | 3,653 | Rich TUI — slash commands, streaming, live panels, sidebar |
| `cli.py` | 3,215 | CLI interface — all commands (setup, serve, reindex, etc.) |
| `agent.py` | 2,136 | Agent loop — sync + async streaming, doom loop detection |
| `serve.py` | 1,203 | HTTP server — remote access + protocol wrapping |
| `protocol_server.py` | 947 | WebSocket/HTTP protocol handler |
| `rlm.py` | 752 | Reasoning language model integration |
| `vector_memory.py` | 711 | SQLite FTS5 + embeddings — hybrid search |
| `config.py` | 699 | Config loading (ollamacode.yaml) |
| `edits.py` | 543 | Structured file edits (diffs, patches) |
| `state.py` | 490 | JSON persistent state with mtime cache |
| `setup_wizard.py` | 481 | 7-step Rich CLI wizard |
| `channels.py` | 434 | Telegram + Discord adapters |
| `mcp_client.py` | 377 | MCP stdio/SSE/HTTP transport |
| `sessions.py` | 375 | SQLite session persistence + branching |
| `scheduler.py` | 325 | Threading cron scheduler |
| `tunnel.py` | 301 | Cloudflare/ngrok/Tailscale/custom tunnels |
| `checkpoints.py` | 231 | Pre/post file snapshots with restore |
| `secrets.py` | 220 | AES-256-GCM encrypted secrets |
| `lsp_client.py` | 248 | LSP stdio JSON-RPC client |
| `plugins.py` | 201 | Event bus + plugin framework |
| `file_watcher.py` | 196 | Directory watching (watchdog/polling) |
| `sandbox.py` | 187 | Filesystem + command sandboxing |
| `model_variants.py` | 174 | Reasoning effort level switching |
| `custom_commands.py` | 172 | Config-driven slash commands |
| `_chat_helpers.py` | 149 | Shared serve/protocol helpers |
| `agent_modes.py` | 147 | BUILD/PLAN/REVIEW mode switching |
| `compaction.py` | 143 | LLM-based message compaction |
| `rate_limit.py` | 121 | Sliding-window rate limiting |
| `permissions.py` | 115 | Per-tool allow/deny/ask permissions |

### Providers (9 total)

| Provider | Module | Lines |
|----------|--------|-------|
| Apple FM | `apple_fm_provider.py` | 1,311 |
| Anthropic | `anthropic_provider.py` | 297 |
| AWS Bedrock | `bedrock_provider.py` | 277 |
| OpenAI-compat | `openai_compat.py` | 276 |
| Google Gemini | `gemini_provider.py` | 239 |
| Azure OpenAI | `azure_provider.py` | 176 |
| Ollama | `ollama_provider.py` | 106 |

### MCP Servers (12 built-in)

| Server | Lines | Purpose |
|--------|-------|---------|
| `fs_mcp.py` | 456 | File operations + sandbox |
| `codebase_mcp.py` | 419 | Code search, symbols, repo map |
| `tools_mcp.py` | 402 | Tool enumeration |
| `git_mcp.py` | 396 | Git operations (structured JSON) |
| `terminal_mcp.py` | 321 | Shell execution + allowlist |
| `semantic_mcp.py` | 255 | Semantic code search |
| `webfetch_mcp.py` | 231 | URL fetch + SSRF guard |
| `screenshot_mcp.py` | 211 | Browser screenshots (Playwright) |
| `web_search_mcp.py` | 172 | SerpAPI / DuckDuckGo search |
| `state_mcp.py` | 120 | State persistence queries |
| `skills_mcp.py` | 62 | Skills/instructions |
| `reasoning_mcp.py` | 60 | Reasoning recording |

### Test Coverage

- **579 tests** across 53 test files
- **7,564 lines** of test code
- **70% minimum** coverage enforced
- Categories: unit (47), integration (3), e2e (1)

---

## Future Work

### TUI Rewrite (Textual)
The Textual-based TUI rewrite is preserved at `ollamacode/_tui_textual/` for future migration.
Current Rich-based TUI is fully functional with 50+ slash commands.

### Potential Additions
- WebSocket live streaming in serve mode
- Multi-model orchestration (routing queries to different models)
- Persistent vector index (currently rebuilt per session)
- Plugin marketplace / registry
- Remote pair programming (shared sessions)
