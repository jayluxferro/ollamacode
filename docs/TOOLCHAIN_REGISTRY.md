# Custom toolchain registry

Curated tools that work well with OllamaCode. The **built-in tools MCP** already provides:

- `run_linter` – run any linter/formatter command
- `run_tests` – run any test command
- `run_code_quality` – run ruff, black, isort, mypy (or custom list) and get one report
- `run_coverage` – run pytest --cov, get uncovered files and suggested tests
- `install_deps` – uv sync / pip install -r

No extra MCP config is needed for these; they run in the workspace via `OLLAMACODE_FS_ROOT`. Below are recommended commands and optional config.

---

## Python

| Tool | Purpose | Example command | Notes |
|------|---------|-----------------|--------|
| **ruff** | Lint + format check | `ruff check .` | Fast; use `run_linter` or default in `run_code_quality`. |
| **black** | Format check | `black --check .` | Part of default `run_code_quality`. |
| **isort** | Import sort check | `isort --check-only .` | Part of default `run_code_quality`. |
| **mypy** | Type checking | `mypy .` | Part of default `run_code_quality`. |
| **pytest** | Tests | `pytest tests/ -v` | Use `run_tests` or config `test_command`. |
| **pytest-cov** | Coverage | `pytest --cov --cov-report=term-missing -q` | Use `run_coverage` for report + suggested tests. |
| **bandit** | Security lint | `bandit -r . -ll` | Add to `run_code_quality` via custom commands. |
| **safety** | Dependency vulns | `safety check` | Add to `run_code_quality` if desired. |

### Custom code quality commands

Override the default suite with env (comma-separated):

```bash
export OLLAMACODE_CODE_QUALITY_COMMANDS="ruff check .,black --check .,mypy ."
```

Or pass a list to the `run_code_quality` tool when calling from the agent.

### Config snippet (optional)

In `ollamacode.yaml` you can set default linter/test commands so slash commands and diagnostics use them:

```yaml
linter_command: ruff check .
test_command: pytest tests/ -v
```

---

## JavaScript / TypeScript

| Tool | Purpose | Example command |
|------|---------|-----------------|
| **eslint** | Lint | `npx eslint src/` |
| **prettier** | Format check | `npx prettier --check .` |
| **npm test** | Tests | `npm test` or `npx vitest run` |

Use `run_linter` and `run_tests` with these commands; no extra MCP server needed.

---

## Other

| Tool | Purpose | Example command |
|------|---------|-----------------|
| **cargo** (Rust) | Lint/test | `cargo clippy`, `cargo test` |
| **go vet** | Go lint | `go vet ./...` |

Use `run_linter` / `run_tests` with the appropriate command.

---

## Adding external MCP servers

To add a **custom MCP server** (e.g. a third-party toolchain server), add it under `mcp_servers` in `ollamacode.yaml`. See [MCP_SERVERS.md](MCP_SERVERS.md) for format (stdio, SSE, streamable HTTP) and examples.
