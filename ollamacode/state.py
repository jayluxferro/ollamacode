"""
Persistent state: recent files, preferences, etc. Stored in ~/.ollamacode/state.json.

Used so the assistant can remember context across sessions. MCP tools (state_mcp) expose get/update/clear.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

_STATE_PATH = Path(os.path.expanduser("~")) / ".ollamacode" / "state.json"
_STATE_LOCK_PATH = Path(os.path.expanduser("~")) / ".ollamacode" / "state.lock"
_STATE_LOCK_TTL_SECONDS = 10.0
_MAX_RECENT_FILES = 50
_MAX_KNOWLEDGE_NODES = 200

# In-memory cache: avoid re-reading state.json from disk on every operation.
# Invalidated on every _save() and validated against file mtime on every _load_raw().
_state_cache: dict | None = None
_state_cache_mtime: float = -1.0


def _load_raw() -> dict:
    global _state_cache, _state_cache_mtime
    if not _STATE_PATH.exists():
        _state_cache = {}
        _state_cache_mtime = -1.0
        return {}
    # If a write lock is present, wait briefly to avoid reading mid-write.
    if _STATE_LOCK_PATH.exists():
        try:
            age = time.time() - _STATE_LOCK_PATH.stat().st_mtime
            if age > _STATE_LOCK_TTL_SECONDS:
                _STATE_LOCK_PATH.unlink(missing_ok=True)
        except OSError:
            pass
    if _STATE_LOCK_PATH.exists():
        deadline = time.monotonic() + 0.5
        while _STATE_LOCK_PATH.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
    try:
        mtime = _STATE_PATH.stat().st_mtime
        if _state_cache is not None and mtime == _state_cache_mtime:
            return _state_cache
        data = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        _state_cache = data
        _state_cache_mtime = mtime
        return data
    except (OSError, json.JSONDecodeError):
        return {}


def _acquire_state_lock(timeout_seconds: float = 2.0) -> None:
    """Acquire a best-effort filesystem lock to serialize state writes."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            fd = os.open(
                str(_STATE_LOCK_PATH),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            try:
                os.write(fd, str(time.time()).encode("utf-8"))
            except OSError:
                pass
            os.close(fd)
            return
        except FileExistsError:
            try:
                age = time.time() - _STATE_LOCK_PATH.stat().st_mtime
                if age > _STATE_LOCK_TTL_SECONDS:
                    _STATE_LOCK_PATH.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("Timed out waiting for state lock")
            time.sleep(0.05)


def _release_state_lock() -> None:
    try:
        _STATE_LOCK_PATH.unlink()
    except OSError:
        pass


def _save(data: dict) -> None:
    global _state_cache, _state_cache_mtime
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _STATE_PATH.with_suffix(".tmp")
    _acquire_state_lock()
    try:
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        try:
            tmp_path.chmod(0o600)
        except OSError:
            pass
        os.replace(tmp_path, _STATE_PATH)
        # Restrict permissions so only the owner can read/write (contains history & prefs).
        try:
            _STATE_PATH.chmod(0o600)
        except OSError:
            pass
    finally:
        _release_state_lock()
    # Update cache immediately so the next _load_raw() doesn't re-read from disk.
    _state_cache = data
    try:
        _state_cache_mtime = _STATE_PATH.stat().st_mtime
    except OSError:
        _state_cache_mtime = -1.0


def get_state() -> dict:
    """Return full state dict (recent_files, preferences, etc.)."""
    return _load_raw()


def update_state(
    recent_files: list[str] | None = None,
    preferences: dict | None = None,
    **kwargs: str | list[str] | dict,
) -> str:
    """
    Update state. recent_files: replace with this list (or append if key is append_recent_files).
    preferences: merge into state["preferences"]. Other kwargs merged at top level.
    Returns a short status message.
    """
    data = _load_raw()
    if recent_files is not None:
        data["recent_files"] = list(recent_files)[-_MAX_RECENT_FILES:]
    if preferences is not None:
        prefs = data.get("preferences") or {}
        prefs.update(preferences)
        data["preferences"] = prefs
    for k, v in kwargs.items():
        if k in ("recent_files", "preferences"):
            continue
        if k == "feedback_append" and isinstance(v, dict):
            feedback = data.get("feedback")
            if not isinstance(feedback, list):
                feedback = []
            feedback.append(v)
            data["feedback"] = feedback[-_MAX_FEEDBACK_ENTRIES:]
            continue
        data[k] = v
    _save(data)
    return f"Updated state ({len(data)} keys) in {_STATE_PATH}"


def append_recent_file(path: str) -> str:
    """Append a file path to recent_files (dedupe, keep last MAX)."""
    data = _load_raw()
    recent = data.get("recent_files") or []
    if path in recent:
        recent.remove(path)
    recent.append(path)
    data["recent_files"] = recent[-_MAX_RECENT_FILES:]
    _save(data)
    return "ok"


def clear_state() -> str:
    """Clear all state. Returns status message."""
    _save({})
    return f"Cleared state at {_STATE_PATH}"


def record_reasoning(steps: list[str], conclusion: str = "") -> str:
    """Store last reasoning (steps + conclusion) in state for explainability. Called by model or parsed from reply."""
    data = _load_raw()
    data["last_reasoning"] = {
        "steps": [str(s) for s in steps],
        "conclusion": str(conclusion).strip(),
    }
    _save(data)
    return "ok"


def format_recent_context(state: dict, max_files: int = 10) -> str:
    """
    Format recent_files from state for injection into the system prompt.
    Returns a short block (e.g. "Recent files: path1, path2, ...") or "" if empty.
    """
    recent = state.get("recent_files") or []
    if not recent or max_files <= 0:
        return ""
    take = recent[-max_files:]
    return "Recent files: " + ", ".join(take)


def format_preferences(state: dict) -> str:
    """
    Format user preferences from state for injection into the system prompt.
    preferences can hold keys like coding_style, language_preference, etc.
    Returns a short block or "" if no preferences.
    """
    prefs = state.get("preferences")
    if not prefs or not isinstance(prefs, dict):
        return ""
    parts = [
        f"{k}: {v}"
        for k, v in sorted(prefs.items())
        if v is not None and str(v).strip()
    ]
    if not parts:
        return ""
    return "User preferences: " + "; ".join(parts)


_MAX_FEEDBACK_ENTRIES = 20


def append_feedback(
    feedback_type: str,
    value: str | int | bool,
    context: str = "",
) -> str:
    """Append one feedback entry to state['feedback'] (keeps last MAX). Used for /rate and edit acceptance."""
    data = _load_raw()
    feedback = data.get("feedback")
    if not isinstance(feedback, list):
        feedback = []
    feedback.append({"type": feedback_type, "value": value, "context": context[:200]})
    data["feedback"] = feedback[-_MAX_FEEDBACK_ENTRIES:]
    _save(data)
    return "ok"


def format_plan_context(state: dict) -> str:
    """Format current_plan and completed_steps for system prompt (Phase 5 long-term planning)."""
    plan = state.get("current_plan")
    steps = state.get("completed_steps")
    if not plan or not str(plan).strip():
        return ""
    plan = str(plan).strip()
    if isinstance(steps, list) and steps:
        lines = [
            "Current plan: " + plan,
            "Completed steps so far:",
            *[f"  - {s}" for s in steps[-15:] if s],
        ]
    else:
        lines = ["Current plan: " + plan]
    return "\n".join(lines)


def format_feedback_context(state: dict, max_entries: int = 5) -> str:
    """Format recent feedback for system prompt (Phase 5 human feedback loop)."""
    feedback = state.get("feedback")
    if not isinstance(feedback, list) or not feedback:
        return ""
    take = feedback[-max_entries:]
    parts = []
    for e in take:
        if not isinstance(e, dict):
            continue
        t = e.get("type", "")
        v = e.get("value")
        c = e.get("context", "")
        if t == "rating" and v in (1, "1", True):
            parts.append("User rated a reply positively")
        elif t == "rating" and v in (-1, "0", False):
            parts.append("User rated a reply negatively")
        elif t == "edit_accepted":
            parts.append("User applied suggested edits")
        elif c:
            parts.append(c[:80])
    if not parts:
        return ""
    return "Recent feedback: " + "; ".join(parts)


def append_past_error(tool: str, error_summary: str, hint: str = "") -> str:
    """Append a tool error to state['past_errors'] for knowledge-based debugging (keeps last 30)."""
    _MAX_PAST_ERRORS = 30
    data = _load_raw()
    past = data.get("past_errors")
    if not isinstance(past, list):
        past = []
    past.append(
        {
            "tool": str(tool)[:80],
            "error_summary": str(error_summary)[:200],
            "hint": str(hint)[:200],
        }
    )
    data["past_errors"] = past[-_MAX_PAST_ERRORS:]
    _save(data)
    return "ok"


def format_past_errors_context(state: dict, max_entries: int = 5) -> str:
    """Format past_errors from state for system prompt (similar past errors for debugging)."""
    past = state.get("past_errors")
    if not isinstance(past, list) or not past:
        return ""
    take = past[-max_entries:]
    parts = []
    for e in take:
        if not isinstance(e, dict):
            continue
        tool = e.get("tool") or ""
        err = e.get("error_summary") or ""
        hint = e.get("hint") or ""
        line = f"- {tool}: {err[:100]}"
        if hint:
            line += f" (hint: {hint[:80]})"
        parts.append(line)
    if not parts:
        return ""
    return "Similar past errors (for context):\n" + "\n".join(parts)


def format_knowledge_context(state: dict, max_entries: int = 15) -> str:
    """Format knowledge_index from state for system prompt (Phase 5 self-organizing knowledge)."""
    idx = state.get("knowledge_index")
    if not isinstance(idx, list) or not idx:
        return ""
    take = idx[-max_entries:]
    parts = []
    for e in take:
        if isinstance(e, dict) and (e.get("topic") or e.get("summary")):
            parts.append(f"- {e.get('topic', '')}: {e.get('summary', '')[:150]}")
        elif isinstance(e, str) and e.strip():
            parts.append("- " + e.strip()[:120])
    graph_block = format_knowledge_graph_context(
        state, max_nodes=max(3, max_entries // 2)
    )
    if not parts and not graph_block:
        return ""
    out = ""
    if parts:
        out = "Knowledge index (use for context):\n" + "\n".join(parts)
    if graph_block:
        out = (out + "\n\n" + graph_block).strip() if out else graph_block
    return out


def _normalize_topic(topic: str) -> str:
    return " ".join(str(topic).strip().split()).lower()


def add_knowledge_node(
    topic: str,
    summary: str = "",
    related: list[str] | None = None,
    source: str = "",
) -> str:
    """Upsert a lightweight knowledge-graph node in state['knowledge_graph']['nodes']."""
    t = _normalize_topic(topic)
    if not t:
        return "topic required"
    data = _load_raw()
    graph = data.get("knowledge_graph")
    if not isinstance(graph, dict):
        graph = {"nodes": []}
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        nodes = []

    rel: list[str] = []
    for r in related or []:
        rs = str(r).strip()
        if rs and rs not in rel:
            rel.append(rs)

    idx = -1
    for i, n in enumerate(nodes):
        if isinstance(n, dict) and _normalize_topic(n.get("topic", "")) == t:
            idx = i
            break

    if idx >= 0:
        node = dict(nodes[idx])
        if summary.strip():
            node["summary"] = summary.strip()[:300]
        if source.strip():
            node["source"] = source.strip()[:200]
        existing_rel = node.get("related")
        if not isinstance(existing_rel, list):
            existing_rel = []
        merged_rel = [str(x).strip() for x in existing_rel if str(x).strip()]
        for r in rel:
            if r not in merged_rel:
                merged_rel.append(r)
        node["related"] = merged_rel[:20]
        node["topic"] = node.get("topic") or topic.strip()[:120]
        nodes[idx] = node
    else:
        nodes.append(
            {
                "topic": topic.strip()[:120],
                "summary": summary.strip()[:300],
                "related": rel[:20],
                "source": source.strip()[:200],
            }
        )
    graph["nodes"] = nodes[-_MAX_KNOWLEDGE_NODES:]
    data["knowledge_graph"] = graph
    _save(data)
    return "ok"


def query_knowledge_graph(query: str, max_results: int = 5) -> list[dict]:
    """Keyword query over lightweight state knowledge graph."""
    q = str(query or "").strip().lower()
    data = _load_raw()
    graph = data.get("knowledge_graph")
    if not isinstance(graph, dict):
        return []
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return []

    scored: list[tuple[int, dict]] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        topic = str(n.get("topic", "")).strip()
        summary = str(n.get("summary", "")).strip()
        related = n.get("related")
        rels = [str(x).strip() for x in related] if isinstance(related, list) else []
        haystack = " ".join([topic, summary, " ".join(rels)]).lower()
        score = 0
        if not q:
            score = 1
        else:
            if q in topic.lower():
                score += 4
            if q in summary.lower():
                score += 2
            score += haystack.count(q)
        if score > 0:
            scored.append((score, n))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = [n for _, n in scored[: max(1, max_results)]]
    return out


def format_knowledge_graph_context(state: dict, max_nodes: int = 8) -> str:
    """Format knowledge_graph nodes for prompt context."""
    graph = state.get("knowledge_graph")
    if not isinstance(graph, dict):
        return ""
    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return ""
    take = nodes[-max_nodes:]
    lines: list[str] = []
    for n in take:
        if not isinstance(n, dict):
            continue
        topic = str(n.get("topic", "")).strip()
        summary = str(n.get("summary", "")).strip()
        related = n.get("related")
        rels = [str(x).strip() for x in related] if isinstance(related, list) else []
        if not topic:
            continue
        line = f"- {topic}: {summary[:140]}" if summary else f"- {topic}"
        if rels:
            line += f" (related: {', '.join(rels[:5])})"
        lines.append(line)
    if not lines:
        return ""
    return "Knowledge graph (state):\n" + "\n".join(lines)
