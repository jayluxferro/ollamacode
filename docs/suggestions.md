# Intelligence‑Centric Enhancement Roadmap

## Review (vs. current codebase)

The table in §1 was written before many features were implemented. **Corrected status**:

| Area | Actual status |
|------|----------------|
| **Skill System** | **Done.** Skills from `~/.ollamacode/skills` and `.ollamacode/skills` are loaded into the system prompt at startup (CLI, TUI, serve, protocol). MCP tools: `list_skills`, `read_skill`, `write_skill`, `save_memory`. |
| **Prompt Management** | **Done.** Config `prompt_template: <name>`; templates from `~/.ollamacode/templates/` and `.ollamacode/templates/`. |
| **State Persistence** | **Done.** `~/.ollamacode/state.json`; state MCP (`get_state`, `update_state`, `append_recent_file`, `clear_state`); TUI `/reset-state`. |
| **Error Handling** | **Done.** Tool failures show "What failed" + "Next step" for common errors (agent.py). |
| **Context Injection** | **Done.** Auto-inject `recent_files` from state + optional branch/last-commit summary (config `inject_recent_context`, `recent_context_max_files`). |
| **Interactive Guidance** | **Done.** `ollamacode tutorial` wizard. |
| **Tool Coverage** | **Done.** `run_linter`, `run_tests`, `install_deps`, `run_code_quality` (unified suite), `run_coverage` (pytest-cov + suggestions). |
| **IDE Integration** | **Done.** Protocol + HTTP; VS Code extension in `editors/vscode`; docs in OTHER_EDITORS.md, STRUCTURED_PROTOCOL.md. |
| **Documentation Generation** | **Done.** TUI `/docs` (runs docs build); config `docs_command`. |

The recommendations in §2 remain valid; §**Status & Todo** lists what is left to implement.

---

## 1. Code‑Base Review Findings

A quick audit of the current repository revealed the following gaps / improvement areas that directly affect the assistant’s *intelligence* and *context awareness*:

| Area | Current Status | Issues / Opportunities |
|------|----------------|------------------------|
| **Skill System** | Only basic `list_skills`, `read_skill`, `write_skill`, `save_memory` MCP tools exist, but they are never loaded into the system prompt. | The assistant cannot *remember* project‑specific knowledge or custom instructions.
| **Prompt Management** | The system prompt is hard‑coded in `servers/mcp.py`. No template system or per‑project override. | Custom prompts per task (refactor, test, docs) are impossible.
| **State Persistence** | No persisted context – recent files, command history, preferences are all transient. | Re‑entering a session loses context.
| **Error Handling** | Raw stderr from `run_command` is displayed verbatim. No normalization or actionable guidance. | Users see cryptic stack traces.
| **Context Injection** | The assistant only receives the full content of the edited file via `apply_edits`. | No automatic inclusion of the most relevant surrounding code or recent edits.
| **Interactive Guidance** | No wizard or tutorial to onboard new users. | First‑time users struggle to discover capabilities.
| **Tool Coverage** | Generic `run_linter`, `run_tests` are available, but no bundled toolchain or coverage checks. | Users must manually install and configure linters, test runners, etc.
| **IDE Integration** | Protocol exists, but no guidance or example extensions for VS Code/Neovim. | Users cannot easily embed the assistant in their editor workflow.
| **Documentation Generation** | No commands to automatically build Sphinx / MkDocs from the repo. | Documentation stays static.

## 2. Intelligence‑Focused Recommendations

Below is a refined, priority‑ordered list of enhancements that will directly boost the assistant’s *smartness* and *context awareness*. Each item includes a brief rationale and the effort level.

### 2.1 Core Enhancements

| # | Feature | Effort | Impact | Rationale |
|---|---------|--------|--------|------------|
| 1 | **Skill Loading & Persistence** | Medium | ★★★★ | Load all `*.md` skill files into the system prompt at startup. Persist user edits via `write_skill`/`save_memory`. Enables the assistant to remember project rules, coding standards, and user preferences.
| 2 | **Prompt Templates** | Low | ★★★ | Store reusable system/user prompt snippets (e.g. *refactor*, *debug*, *docs*). Allow the user to select a template via a slash command, automatically populating the LLM prompt.
| 3 | **Context Injection** | Medium | ★★★★ | On each assistant invocation, automatically include:
  * the edited file’s content
  * the most recent 10 related files (from `recent_files` state)
  * any matching skills
  * a short summary of the last commit or branch status.  This reduces the need for the user to paste context.
| 4 | **State Persistence** | Medium | ★★★ | Persist `recent_files`, `command_history`, `preferences` in `~/.ollamacode/state.json`. Provide `/clear` command to reset.
| 5 | **Error Normalization** | Low | ★★ | Wrap `run_command` output; detect common errors (syntax errors, missing modules) and produce concise guidance.
| 6 | **Interactive Tutorial** | Medium | ★★★ | Implement a wizard that guides the user through: `git status`, `apply_edits`, `run_linter`, `run_tests`, and commit/push. Can be optional and skip after first run.

### 2.2 Toolchain & Automation

| # | Feature | Effort | Impact | Rationale |
|---|---------|--------|--------|------------|
| 7 | **Dependency Manager** | Low | ★★ | CLI helper that installs missing deps from `requirements.txt` or `pyproject.toml` and warns about outdated/insecure packages.
| 8 | **Unified Code Quality Suite** | Medium | ★★★★ | One command that runs `ruff`, `black`, `isort`, `mypy`, optionally `bandit`/`safety`. Aggregates results and presents a concise report.
| 9 | **Coverage & Test Suggestion** | Medium | ★★★ | Run `pytest --cov`; parse uncovered paths and suggest minimal tests. 
|10 | **Project Scaffold** | Low | ★★ | `ollamacode init <template>` that generates a standard project skeleton (CLI, web, library).  Accelerates onboarding.
|11 | **Performance Profiling** | Low | ★★ | One‑liner to run `cProfile` or `py-spy`, summarise hotspots.
|12 | **Git Graph Viewer** | Low | ★★ | Simple command to display `git log --oneline --graph`, improving quick context.

### 2.3 Integration & Extensibility

| # | Feature | Effort | Impact | Rationale |
|---|---------|--------|--------|------------|
|13 | **IDE / Editor Extension** | Medium | ★★★ | Provide a lightweight VS Code/Neovim extension that calls the HTTP/stdio protocol for chat and edits, exposing quick actions.
|14 | **Custom Toolchain Registry** | Low | ★★ | Curated list of popular tools with MCP config snippets.
|15 | **Burp Integration** | Low | ★ | Wrap existing Burp MCP; no code change needed, but add documentation.

### 2.4 Miscellaneous

| # | Feature | Effort | Impact | Rationale |
|---|---------|--------|--------|------------|
|16 | **Documentation Generation** | Low | ★★ | Commands to build Sphinx/MkDocs from docstrings.
|17 | **Health Check Endpoint** | Low | ★ | Quick CLI/HTTP command to verify Ollama & MCP availability.

## 3. Implementation Roadmap

1. **Immediate (0‑2 weeks)** – Skill loading, prompt templates, state persistence, error normalization.  These are low‑effort yet high‑impact.
2. **Short‑term (2‑4 weeks)** – Context injection, interactive tutorial, dependency manager, git graph viewer.
3. **Mid‑term (4‑8 weeks)** – Unified code quality suite, coverage suggestions, project scaffold, performance profiling.
4. **Long‑term (8‑12 weeks)** – IDE integration, custom toolchain registry, documentation generation.

> **Note:** All changes should preserve backward compatibility and expose new slash commands or MCP endpoints in a minimal, incremental way.

---

## Status & Todo

| # | Item | Status | Notes |
|---|------|--------|--------|
| 1 | Skill loading & persistence | ✅ Done | Loaded into system prompt; write_skill/save_memory. |
| 2 | Prompt templates | ✅ Done | Config + template dirs. |
| 3 | Context injection | ✅ Done | Auto-inject `recent_files` (from state) + optional last-commit/branch summary into prompt. |
| 4 | State persistence | ✅ Done | state.json + state MCP. |
| 5 | Error normalization | ✅ Done | _format_tool_error_hint in agent. |
| 6 | Interactive tutorial | ✅ Done | `ollamacode tutorial`. |
| 7 | Dependency manager | ✅ Done | `install_deps` MCP tool. |
| 8 | Unified code quality suite | ✅ Done | One command: ruff + black + isort + mypy (optional bandit/safety); single report. |
| 9 | Coverage & test suggestion | ✅ Done | Run pytest --cov; parse uncovered paths; suggest minimal tests. |
|10 | Project scaffold | ✅ Done | `ollamacode init` + templates. |
|11 | Performance profiling | ✅ Done | TUI `/profile`; config `profile_command`. |
|12 | Git graph viewer | ✅ Done | `git_log_graph` in git MCP. |
|13 | IDE / editor extension | ✅ Done | VS Code extension in editors/vscode (chat, chat-with-selection, apply-edits). |
|14 | Custom toolchain registry | ✅ Done | Doc or section with curated tools + MCP snippets. |
|15 | Burp integration | ⏭️ Out of scope | Per user. |
|16 | Documentation generation | ✅ Done | `/docs` + `docs_command`. |
|17 | Health check endpoint | ✅ Done | CLI `ollamacode health` and/or GET /health for Ollama + MCP. |

### Concrete tasks (in order)

1. **Context injection (#3)**
   - [x] When building the initial prompt (CLI/TUI/serve/protocol), call state MCP or read `~/.ollamacode/state.json` for `recent_files`.
   - [x] Append a "Recent files" block (e.g. last 10 paths) to the system or first user message, with optional one-line "last commit or branch" summary (reuse/extend existing branch context).
   - [x] Add config flag to enable/disable context injection (e.g. `inject_recent_context: true`) and optionally cap length.

2. **Unified code quality (#8)**
   - [x] Add one MCP tool (e.g. `run_code_quality`) or slash command that runs a configurable list of commands (default: ruff, black, isort, mypy; optional: bandit, safety).
   - [x] Aggregate stdout/stderr from each command into a single report (tool result or TUI panel).
   - [x] Document config for command list and paths (e.g. in config or env).

3. **Coverage & test suggestion (#9)**
   - [x] Add tool or slash command that runs `pytest --cov` (or configurable coverage command).
   - [x] Parse coverage output for uncovered files/lines (or use coverage report format).
   - [x] Return a short list of uncovered paths plus suggested test descriptions or stub test names (e.g. as tool result text).

4. **Health check (#17)**
   - [x] Add CLI subcommand `ollamacode health` that checks Ollama reachability (and optionally model list) and MCP server availability; print success/failure and a short message.
   - [x] Add HTTP endpoint `GET /health` (when serving) that returns JSON e.g. `{ "ollama": true, "mcp": true }` or error details.

5. **Custom toolchain registry (#14)**
   - [x] Add a doc section or new doc (e.g. `docs/TOOLCHAIN_REGISTRY.md`) with a curated list of tools (linters, formatters, test runners, security scanners).
   - [x] For each tool, provide a short description and a copy-paste MCP config snippet (or link to existing MCP server).

6. **Optional: IDE extension (#13)**
   - [x] Create a minimal VS Code or Neovim extension (or both) that calls the existing HTTP/stdio protocol for chat and apply-edits.
   - [x] Or expand `OTHER_EDITORS.md` with step-by-step instructions to build such an extension.

---

**This file is the authoritative source for the next set of intelligence‑boosting features.  Keep it up‑to‑date as you implement the roadmap.**
