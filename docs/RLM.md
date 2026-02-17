# Recursive Language Model (RLM) integration

Experimental support for [Recursive Language Models](https://arxiv.org/abs/2512.24601): the user prompt is kept **out of** the model's context and placed in a REPL; the model sees only metadata and writes code to inspect the prompt and call the LLM recursively on slices.

## Design

1. **Entrypoint**: `ollamacode --rlm "prompt"` or `ollamacode --rlm` (read from stdin). Optional `--file` / `--lines` to load context from a file. RLM is off by default; use `--rlm` to enable.
2. **REPL environment** (Python):
   - `context`: string holding the full user prompt (and optional file content).
   - `llm_query(prompt: str) -> str`: calls Ollama with the given prompt and returns the reply. Implemented by the host process (we intercept this in the REPL and perform the HTTP/sync Ollama call).
3. **RLM loop**:
   - Initialize REPL with `context = <user prompt>`.
   - Build initial message for the **root** model: system prompt (RLM instructions) + metadata only (e.g. "Your context has N characters, chunk lengths: [...], prefix: first 500 chars"). No full prompt in context.
   - Loop:
     - Call Ollama (root model) with current history (metadata + previous code + truncated stdout).
     - Parse response: extract ` ```repl ... ``` ` blocks and `FINAL(...)` / `FINAL_VAR(var_name)`.
     - If FINAL/FINAL_VAR found: return that as the answer.
     - Else: run each repl block in the REPL. When code calls `llm_query(x)`, we invoke Ollama (sub-call), pass result back into the REPL. Collect stdout (truncated) and append to history; repeat.
4. **Parsing**: regex or simple scan for:
   - Code blocks with language `repl` (or unlabeled blocks if we prefer).
   - `FINAL(...)` and `FINAL_VAR(...)`.
5. **Sandbox**: REPL runs in a subprocess or restricted executor; timeout per execution; no network (llm_query is implemented by the host). Optional: disable filesystem or allow only read.

## Task list (implementation order)

- [x] **docs/RLM.md** – This design and task list.
- [x] **ollamacode/rlm.py** – REPL environment with `context` and `llm_query`; host resolves sub-calls via Ollama.
- [x] **Parsing** – `parse_repl_blocks`, `parse_final` for ```repl blocks and FINAL / FINAL_VAR.
- [x] **RLM loop** – `run_rlm_loop`: metadata message → Ollama → run REPL / resolve llm_query → repeat until FINAL or max iterations.
- [x] **CLI** – `--rlm` (query or stdin); `--file` / `--lines`; config for rlm_* options.
- [x] **Sandbox** – Timeout per REPL run; restricted builtins (no `open`/`__import__`); unit tests.

## Configuration

Supported in `ollamacode.yaml` (and merged into options when using RLM):

- **`rlm_sub_model`**: model for `llm_query` sub-calls (default: same as main model).
- **`rlm_max_iterations`**: max RLM loop steps (default: 20).
- **`rlm_stdout_max_chars`**: truncate REPL stdout in history (default: 2000).
- **`rlm_prefix_chars`**: characters of context prefix in metadata (default: 500).
- **`rlm_snippet_timeout_seconds`**: max time per single exec of a REPL snippet (optional). When set, each run of a block is capped so one runaway snippet doesn't consume the whole run.
- **`rlm_use_subprocess`**: when true, run each REPL snippet in a separate process with resource limits (Unix: `RLIMIT_AS`, `RLIMIT_CPU`). Reduces risk of runaway code; when enabled, blocks do not share a namespace (FINAL_VAR only within the same block).
- **`rlm_subprocess_max_memory_mb`**: max memory for subprocess (default 512). Clamped 64–4096.
- **`rlm_subprocess_max_cpu_seconds`**: max CPU seconds for subprocess (default 60). Clamped 5–3600.

**Streaming**: Use `ollamacode --rlm --stream "prompt"` to stream step and REPL output to stderr and the final answer to stdout.

## Future / optional

- **Streaming over HTTP/protocol**: Expose RLM streaming in `ollamacode serve` and stdio protocol.
- **Read-only filesystem**: Option to run REPL in a chroot or restrict file access.

## References

- Paper: [Recursive Language Models (arxiv.org/abs/2512.24601)](https://arxiv.org/abs/2512.24601)
- Code: https://github.com/alexzhang13/rlm
