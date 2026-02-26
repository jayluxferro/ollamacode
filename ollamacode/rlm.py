"""
Recursive Language Model (RLM) integration.

The user prompt is kept out of the model context and placed in a REPL as `context`.
The model sees only metadata and emits code in ```repl blocks that can call
llm_query(prompt) to invoke the LLM on slices of context. FINAL(...) or FINAL_VAR(name)
signals the answer. See docs/RLM.md.
"""

from __future__ import annotations

import io
import logging
import multiprocessing
import os
import re
import sys
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from queue import Empty
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:
    import resource
except ImportError:
    resource = None  # type: ignore[assignment]

# --- Parsing ---

REPL_BLOCK_RE = re.compile(r"```(?:repl)?\s*\n(.*?)```", re.DOTALL)
# Non-greedy (.*?) so first ) ends capture; answer with nested ) may be truncated
FINAL_RE = re.compile(r"FINAL\s*\(\s*(.*?)\s*\)", re.DOTALL)
FINAL_VAR_RE = re.compile(r"FINAL_VAR\s*\(\s*(\w+)\s*\)")


def parse_repl_blocks(text: str) -> list[str]:
    """Extract code from ```repl or ``` blocks. Returns list of code strings."""
    blocks: list[str] = []
    for m in REPL_BLOCK_RE.finditer(text):
        blocks.append(m.group(1).strip())
    return blocks


def parse_final(text: str) -> tuple[str | None, str | None]:
    """
    Find FINAL(...) or FINAL_VAR(var_name) in text.
    Returns (final_content, None) for FINAL or (None, var_name) for FINAL_VAR, else (None, None).
    Prefers FINAL over FINAL_VAR if both appear.
    """
    m = FINAL_RE.search(text)
    if m:
        return (m.group(1).strip(), None)
    m = FINAL_VAR_RE.search(text)
    if m:
        return (None, m.group(1).strip())
    return (None, None)


# --- REPL execution ---

# Safe subset of builtins for REPL (no open, __import__, exec, eval, input, etc.)
_SAFE_BUILTIN_NAMES = (
    "abs",
    "all",
    "any",
    "ascii",
    "bin",
    "bool",
    "chr",
    "dict",
    "dir",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "hash",
    "hex",
    "id",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "oct",
    "ord",
    "pow",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "type",
    "zip",
)
if hasattr(__builtins__, "keys"):
    _SAFE_BUILTINS = {
        k: __builtins__[k] for k in _SAFE_BUILTIN_NAMES if k in __builtins__
    }
else:
    _SAFE_BUILTINS = {
        k: getattr(__builtins__, k)
        for k in _SAFE_BUILTIN_NAMES
        if hasattr(__builtins__, k)
    }
# Add constants
for _k in ("True", "False", "None"):
    if hasattr(__builtins__, _k):
        _SAFE_BUILTINS[_k] = getattr(__builtins__, _k)
    elif isinstance(__builtins__, dict) and _k in __builtins__:
        _SAFE_BUILTINS[_k] = __builtins__[_k]


def _serialize_globals(globals_: dict[str, Any]) -> dict[str, Any]:
    """Return a picklable copy of globals_ with simple types only (for subprocess result)."""
    out: dict[str, Any] = {}
    for k, v in globals_.items():
        if k.startswith("_"):
            continue
        if v is None or isinstance(v, (bool, int, float, str)):
            out[k] = v
        elif isinstance(v, (list, dict)):
            try:
                out[k] = v
            except Exception:
                out[k] = repr(v)
        else:
            out[k] = repr(v)
    return out


def _repl_subprocess_worker(
    code: str,
    context: str,
    request_queue: multiprocessing.Queue,
    response_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    max_memory_mb: int,
    max_cpu_seconds: int,
) -> None:
    """Run in child process: set rlimits, exec code with llm_query via queues."""
    stdout_io = io.StringIO()
    stderr_io = io.StringIO()
    if resource is not None and hasattr(resource, "setrlimit"):
        if max_memory_mb > 0:
            try:
                limit = max_memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            except (ValueError, OSError):
                pass
        if max_cpu_seconds > 0:
            try:
                resource.setrlimit(
                    resource.RLIMIT_CPU, (max_cpu_seconds, max_cpu_seconds)
                )
            except (ValueError, OSError):
                pass
    try:

        def llm_query(prompt: str) -> str:
            request_queue.put(prompt)
            return response_queue.get()

        globals_: dict[str, Any] = {
            "context": context,
            "llm_query": llm_query,
            "__builtins__": _SAFE_BUILTINS,
        }
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = stdout_io
            sys.stderr = stderr_io
            exec(code, globals_)
            result_queue.put(
                (
                    "ok",
                    stdout_io.getvalue(),
                    stderr_io.getvalue(),
                    _serialize_globals(globals_),
                )
            )
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
    except Exception as e:
        result_queue.put(("error", str(e), stdout_io.getvalue(), stderr_io.getvalue()))


class LLMQueryPending(Exception):
    """Raised by llm_query stub when it needs the host to call the LLM."""

    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        super().__init__(prompt)


@dataclass
class ReplResult:
    """Result of running one or more REPL snippets (with llm_query resolved)."""

    stdout: str = ""
    stderr: str = ""
    final_stdout: str = ""  # after all re-runs, for this snippet
    llm_calls: list[str] = field(default_factory=list)  # prompts that were resolved
    error: str | None = None  # if execution failed
    final_globals: dict[str, Any] | None = (
        None  # namespace after last run (for FINAL_VAR)
    )


def _run_repl_once(
    code: str,
    context: str,
    llm_responses: list[str],
    pending_prompt: list[str],
    stdout_io: io.StringIO,
    stderr_io: io.StringIO,
    initial_globals: dict[str, Any] | None = None,
    safe_builtins: bool = True,
) -> dict[str, Any]:
    """Run code in a restricted globals with context and llm_query. Returns the globals dict after execution."""

    def llm_query(prompt: str) -> str:
        if llm_responses:
            return llm_responses.pop(0)
        pending_prompt.append(prompt)
        raise LLMQueryPending(prompt)

    builtins = _SAFE_BUILTINS if safe_builtins else __builtins__
    if initial_globals is not None:
        globals_ = dict(initial_globals)
        globals_["context"] = context
        globals_["llm_query"] = llm_query
    else:
        globals_ = {
            "context": context,
            "llm_query": llm_query,
            "__builtins__": builtins,
        }
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout = stdout_io
        sys.stderr = stderr_io
        exec(code, globals_)
        return globals_
    except LLMQueryPending:
        raise
    except Exception:
        raise
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


def _run_repl_snippet_subprocess(
    code: str,
    context: str,
    llm_resolver: Callable[[str], str],
    timeout_seconds: float,
    max_memory_mb: int,
    max_cpu_seconds: int,
) -> ReplResult:
    """Run snippet in a subprocess with resource limits; llm_query via queues."""
    result = ReplResult()
    ctx = multiprocessing.get_context("spawn")
    request_q: multiprocessing.Queue = ctx.Queue()
    response_q: multiprocessing.Queue = ctx.Queue()
    result_q: multiprocessing.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_repl_subprocess_worker,
        args=(
            code,
            context,
            request_q,
            response_q,
            result_q,
            max_memory_mb,
            max_cpu_seconds,
        ),
    )
    proc.start()

    def serve_requests() -> None:
        while proc.is_alive():
            try:
                prompt = request_q.get(timeout=0.2)
                response_q.put(llm_resolver(prompt))
            except Empty:
                pass

    serve_thread = threading.Thread(target=serve_requests, daemon=True)
    serve_thread.start()
    try:
        try:
            status, a, b, c = result_q.get(timeout=timeout_seconds)
        except Empty:
            proc.terminate()
            proc.join(timeout=2)
            result.error = f"REPL timed out after {timeout_seconds}s"
            return result
        if status == "error":
            result.error = a
            result.final_stdout = b
            result.stderr = c
        else:
            result.final_stdout = a
            result.stderr = b
            result.final_globals = c
    finally:
        proc.join(timeout=2)
        if proc.is_alive():
            proc.kill()
    return result


def run_repl_snippet(
    code: str,
    context: str,
    llm_resolver: Callable[[str], str],
    timeout_seconds: float | None = 30.0,
    snippet_timeout_seconds: float | None = None,
    initial_globals: dict[str, Any] | None = None,
    use_subprocess: bool = False,
    subprocess_max_memory_mb: int = 512,
    subprocess_max_cpu_seconds: int = 60,
) -> ReplResult:
    """
    Run Python snippet with `context` and `llm_query` available.
    When the code calls llm_query(prompt), we resolve it via llm_resolver(prompt) and re-run
    the snippet from the start with that response injected (and so on for multiple calls).
    If initial_globals is set, the snippet runs in that namespace (and updates are returned in final_globals).
    timeout_seconds: max time for the whole run (all re-runs for this snippet).
    snippet_timeout_seconds: max time for each single exec (one _run_repl_once). When set, each
        run is capped so one runaway snippet doesn't consume the whole timeout.
    use_subprocess: if True and initial_globals is None, run in a separate process with resource limits.
    subprocess_max_memory_mb, subprocess_max_cpu_seconds: limits when use_subprocess (Unix only).
    """
    if use_subprocess and initial_globals is None:
        return _run_repl_snippet_subprocess(
            code,
            context,
            llm_resolver,
            timeout_seconds=timeout_seconds or 30.0,
            max_memory_mb=subprocess_max_memory_mb,
            max_cpu_seconds=subprocess_max_cpu_seconds,
        )
    result = ReplResult()
    stdout_io = io.StringIO()
    stderr_io = io.StringIO()
    llm_responses: list[str] = []
    pending_prompt: list[str] = []

    def _run_one() -> dict[str, Any]:
        """Single exec of the snippet; raises LLMQueryPending if llm_query is called."""
        return _run_repl_once(
            code,
            context,
            llm_responses,
            pending_prompt,
            stdout_io,
            stderr_io,
            initial_globals=initial_globals,
            safe_builtins=True,
        )

    def _run() -> None:
        nonlocal result
        while True:
            stdout_io.truncate(0)
            stdout_io.seek(0)
            stderr_io.truncate(0)
            stderr_io.seek(0)
            pending_prompt.clear()
            try:
                if snippet_timeout_seconds and snippet_timeout_seconds > 0:
                    with ThreadPoolExecutor(max_workers=1) as ex:
                        fut = ex.submit(_run_one)
                        try:
                            result.final_globals = fut.result(
                                timeout=snippet_timeout_seconds
                            )
                        except FuturesTimeoutError:
                            result.error = f"REPL snippet timed out after {snippet_timeout_seconds}s"
                            result.final_stdout = stdout_io.getvalue()
                            result.stderr = stderr_io.getvalue()
                            return
                else:
                    result.final_globals = _run_one()
                result.final_stdout = stdout_io.getvalue()
                result.stderr = stderr_io.getvalue()
                break
            except LLMQueryPending as e:
                result.llm_calls.append(e.prompt)
                response = llm_resolver(e.prompt)
                llm_responses.append(response)

    try:
        if timeout_seconds and timeout_seconds > 0:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_run)
                try:
                    fut.result(timeout=timeout_seconds)
                except FuturesTimeoutError:
                    result.error = (
                        result.error or f"REPL timed out after {timeout_seconds}s"
                    )
                    result.final_stdout = stdout_io.getvalue()
                    result.stderr = stderr_io.getvalue()
        else:
            _run()
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        result.stderr = stderr_io.getvalue()
        result.final_stdout = stdout_io.getvalue()
    return result


def run_repl_blocks(
    codes: list[str],
    context: str,
    llm_resolver: Callable[[str], str],
    timeout_seconds: float | None = 30.0,
    snippet_timeout_seconds: float | None = None,
    use_subprocess: bool = False,
    subprocess_max_memory_mb: int = 512,
    subprocess_max_cpu_seconds: int = 60,
) -> tuple[list[str], dict[str, Any] | None, str | None]:
    """
    Run multiple REPL snippets in a single shared namespace (so FINAL_VAR can reference a variable set in any block).
    Returns (list of stdout strings per block, final_globals, error).
    When use_subprocess is True, each snippet runs in a separate process (no shared namespace; FINAL_VAR only within same block).
    """
    shared_globals: dict[str, Any] | None = None
    stdout_parts: list[str] = []
    for code in codes:
        result = run_repl_snippet(
            code,
            context,
            llm_resolver,
            timeout_seconds=timeout_seconds,
            snippet_timeout_seconds=snippet_timeout_seconds,
            initial_globals=None if use_subprocess else shared_globals,
            use_subprocess=use_subprocess,
            subprocess_max_memory_mb=subprocess_max_memory_mb,
            subprocess_max_cpu_seconds=subprocess_max_cpu_seconds,
        )
        if result.error:
            return (stdout_parts, shared_globals, result.error)
        stdout_parts.append(result.final_stdout)
        if result.final_globals is not None:
            shared_globals = result.final_globals
    return (stdout_parts, shared_globals, None)


# --- Metadata and system prompt for RLM ---

RLM_SYSTEM_PROMPT = """You are an assistant that answers the user's question using a REPL.

You do NOT see the user's full message. You have:
- A variable `context`: a long string (in the REPL) containing the user's message and any file content.
- A function `llm_query(prompt)`: call it with a string prompt; the system will run an LLM on that prompt and return the result as a string. Use it to analyze parts of `context` (e.g. by slicing context and asking the LLM to summarize or answer).

You must write Python code in ```repl code blocks. The code can:
- Inspect `context` (e.g. len(context), context[0:500], context.split('\\n')).
- Call result = llm_query("your prompt here") to get LLM answers on snippets.
- Use print() to show intermediate results (they will be shown back to you).

When you have the final answer:
- Either output FINAL(your answer here) in your message, or
- Set a variable and output FINAL_VAR(variable_name).

You will then see the REPL stdout after each run. Keep iterating until you can produce FINAL(...) or FINAL_VAR(...)."""


def build_metadata_message(context: str, prefix_chars: int = 500) -> str:
    """Build the initial user message: metadata about context, no full content."""
    n = len(context)
    lines = context.splitlines()
    chunk_info = (
        f"Lines: {len(lines)}. First line length: {len(lines[0]) if lines else 0}."
    )
    prefix = context[:prefix_chars] if n > prefix_chars else context
    if n > prefix_chars:
        prefix += "\n... (truncated)"
    return (
        f"Your REPL has a variable `context` with the user's message.\n"
        f"Metadata: total {n} characters. {chunk_info}\n"
        f"Prefix (first {min(prefix_chars, n)} chars):\n---\n{prefix}\n---\n"
        f"Write ```repl code to explore context and use llm_query() as needed. When done, output FINAL(...) or FINAL_VAR(var_name)."
    )


def truncate(s: str, max_chars: int, suffix: str = "\n... (truncated)") -> str:
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[: max_chars - len(suffix)] + suffix


# --- RLM loop ---


def _ollama_chat(model: str, messages: list[dict[str, Any]]) -> str:
    """Sync Ollama chat with timeout to avoid hanging; returns assistant message content."""
    timeout_s = float(os.environ.get("OLLAMACODE_RLM_TIMEOUT_SECONDS", "60"))
    result: dict[str, Any] = {"text": "", "error": None}

    def _run() -> None:
        try:
            from .ollama_client import chat_sync

            r = chat_sync(model=model, messages=messages, tools=[])
            msg = r.get("message") if isinstance(r, dict) else getattr(r, "message", None)
            if msg is None:
                result["text"] = ""
                return
            result["text"] = (
                (msg.get("content") or "")
                if isinstance(msg, dict)
                else str(getattr(msg, "content", "") or "")
            )
        except Exception as e:
            result["error"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        return ""
    if result.get("error") is not None:
        return f"[Ollama error: {result['error']}]"
    return str(result.get("text") or "")


def run_rlm_loop(
    context: str,
    *,
    model: str = "llama3.2",
    sub_model: str | None = None,
    max_iterations: int = 20,
    stdout_max_chars: int = 2000,
    prefix_chars: int = 500,
    snippet_timeout_seconds: float | None = None,
    use_subprocess: bool = False,
    subprocess_max_memory_mb: int = 512,
    subprocess_max_cpu_seconds: int = 60,
    quiet: bool = False,
) -> str:
    """
    Run the RLM loop: send metadata-only message, get model response, run REPL
    and resolve llm_query via Ollama, append truncated stdout to history, repeat
    until FINAL(...) or FINAL_VAR(...) or max_iterations.
    """
    sub_model = sub_model or model
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": RLM_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_metadata_message(context, prefix_chars=prefix_chars),
        },
    ]

    def llm_resolver(prompt: str) -> str:
        return _ollama_chat(sub_model, [{"role": "user", "content": prompt}])

    for step in range(max_iterations):
        if not quiet:
            logger.info("RLM step %s/%s", step + 1, max_iterations)
        response_content = _ollama_chat(model, messages)
        if not response_content:
            return "[RLM] Model returned empty response."

        final_content, final_var = parse_final(response_content)
        if final_content is not None:
            return final_content

        blocks = parse_repl_blocks(response_content)
        if not blocks:
            if final_var is not None:
                return f"[RLM] Model requested FINAL_VAR({final_var}) but no REPL block was run to set it."
            messages.append({"role": "assistant", "content": response_content})
            messages.append(
                {
                    "role": "user",
                    "content": "You must write Python code in a ```repl code block. Use context and llm_query() then output FINAL(...) or FINAL_VAR(name).",
                }
            )
            continue

        stdout_parts, final_globals, repl_error = run_repl_blocks(
            blocks,
            context,
            llm_resolver,
            snippet_timeout_seconds=snippet_timeout_seconds,
            use_subprocess=use_subprocess,
            subprocess_max_memory_mb=subprocess_max_memory_mb,
            subprocess_max_cpu_seconds=subprocess_max_cpu_seconds,
        )
        if repl_error:
            repl_output = f"Error: {repl_error}"
        else:
            repl_output = "\n---\n".join(
                truncate(part, stdout_max_chars) or "(no stdout)"
                for part in stdout_parts
            )

        if (
            final_var is not None
            and final_globals is not None
            and final_var in final_globals
        ):
            val = final_globals[final_var]
            return str(val) if val is not None else ""

        if final_var is not None:
            return f"[RLM] Model requested FINAL_VAR({final_var}); variable not found in REPL namespace."

        messages.append({"role": "assistant", "content": response_content})
        messages.append(
            {
                "role": "user",
                "content": f"REPL stdout:\n{repl_output}\n\nContinue with more ```repl code if needed, or output FINAL(your_answer) / FINAL_VAR(var_name).",
            }
        )

    return "[RLM] Max iterations reached without FINAL or FINAL_VAR."


def run_rlm_loop_stream(
    context: str,
    *,
    model: str = "llama3.2",
    sub_model: str | None = None,
    max_iterations: int = 20,
    stdout_max_chars: int = 2000,
    prefix_chars: int = 500,
    snippet_timeout_seconds: float | None = None,
    use_subprocess: bool = False,
    subprocess_max_memory_mb: int = 512,
    subprocess_max_cpu_seconds: int = 60,
    quiet: bool = False,
) -> Iterator[dict[str, Any]]:
    """
    Run the RLM loop and yield progress events for streaming.

    Yields dicts: {"type": "step", "step": n, "max": M},
    {"type": "repl_output", "text": str}, {"type": "done", "content": str},
    {"type": "error", "message": str}.
    """
    sub_model = sub_model or model
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": RLM_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_metadata_message(context, prefix_chars=prefix_chars),
        },
    ]

    def llm_resolver(prompt: str) -> str:
        return _ollama_chat(sub_model, [{"role": "user", "content": prompt}])

    for step in range(max_iterations):
        if not quiet:
            yield {"type": "step", "step": step + 1, "max": max_iterations}
        response_content = _ollama_chat(model, messages)
        if not response_content:
            yield {"type": "error", "message": "[RLM] Model returned empty response."}
            return

        final_content, final_var = parse_final(response_content)
        if final_content is not None:
            yield {"type": "done", "content": final_content}
            return

        blocks = parse_repl_blocks(response_content)
        if not blocks:
            if final_var is not None:
                yield {
                    "type": "error",
                    "message": f"[RLM] Model requested FINAL_VAR({final_var}) but no REPL block was run to set it.",
                }
                return
            messages.append({"role": "assistant", "content": response_content})
            messages.append(
                {
                    "role": "user",
                    "content": "You must write Python code in a ```repl code block. Use context and llm_query() then output FINAL(...) or FINAL_VAR(name).",
                }
            )
            continue

        stdout_parts, final_globals, repl_error = run_repl_blocks(
            blocks,
            context,
            llm_resolver,
            snippet_timeout_seconds=snippet_timeout_seconds,
            use_subprocess=use_subprocess,
            subprocess_max_memory_mb=subprocess_max_memory_mb,
            subprocess_max_cpu_seconds=subprocess_max_cpu_seconds,
        )
        if repl_error:
            repl_output = f"Error: {repl_error}"
        else:
            repl_output = "\n---\n".join(
                truncate(part, stdout_max_chars) or "(no stdout)"
                for part in stdout_parts
            )
        yield {"type": "repl_output", "text": repl_output}

        if (
            final_var is not None
            and final_globals is not None
            and final_var in final_globals
        ):
            val = final_globals[final_var]
            yield {"type": "done", "content": str(val) if val is not None else ""}
            return

        if final_var is not None:
            yield {
                "type": "error",
                "message": f"[RLM] Model requested FINAL_VAR({final_var}); variable not found in REPL namespace.",
            }
            return

        messages.append({"role": "assistant", "content": response_content})
        messages.append(
            {
                "role": "user",
                "content": f"REPL stdout:\n{repl_output}\n\nContinue with more ```repl code if needed, or output FINAL(your_answer) / FINAL_VAR(var_name).",
            }
        )

    yield {
        "type": "error",
        "message": "[RLM] Max iterations reached without FINAL or FINAL_VAR.",
    }
