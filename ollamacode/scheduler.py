"""Background scheduler for OllamaCode periodic and cron-based agent tasks.

Tasks are defined in two places (merged at runtime):
  1. ``ollamacode.yaml`` under ``scheduled_tasks`` key (list of task dicts)
  2. ``HEARTBEAT.md`` in the workspace root (parsed task blocks)

Each task dict supports:

  name:        unique task identifier
  message:     prompt sent to the agent loop (required)
  description: human-readable description (optional)
  interval:    run every N seconds  (mutually exclusive with cron)
  cron:        cron expression "min hour day month weekday"  (5-field standard)
  observability: noop (default) | log | metrics

A threading-based scheduler loop runs in a daemon thread alongside
``ollamacode serve``.  Tasks share no MCP session (they run via
``run_agent_loop_no_mcp``), keeping scheduling simple and safe.

CLI:
    ollamacode cron list            — list all configured tasks
    ollamacode cron run <name>      — run a task immediately
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HEARTBEAT_FILENAME = "HEARTBEAT.md"


# ---------------------------------------------------------------------------
# Minimal 5-field cron parser
# ---------------------------------------------------------------------------

def _field_matches(field: str, value: int) -> bool:
    """Return True if cron *field* matches integer *value*."""
    field = field.strip()
    if field == "*":
        return True
    for part in field.split(","):
        part = part.strip()
        if "/" in part:
            base, step_str = part.split("/", 1)
            try:
                step = int(step_str)
            except ValueError:
                continue
            if base == "*":
                if value % step == 0:
                    return True
            else:
                try:
                    start = int(base)
                except ValueError:
                    continue
                if value >= start and (value - start) % step == 0:
                    return True
        elif "-" in part:
            lo_str, hi_str = part.split("-", 1)
            try:
                if int(lo_str) <= value <= int(hi_str):
                    return True
            except ValueError:
                continue
        else:
            try:
                if int(part) == value:
                    return True
            except ValueError:
                continue
    return False


def cron_matches(expression: str, dt: datetime) -> bool:
    """Return True if *expression* (5-field standard cron) matches *dt*."""
    parts = expression.strip().split()
    if len(parts) != 5:
        return False
    minute, hour, day, month, weekday = parts
    return (
        _field_matches(minute, dt.minute)
        and _field_matches(hour, dt.hour)
        and _field_matches(day, dt.day)
        and _field_matches(month, dt.month)
        and _field_matches(weekday, dt.weekday())  # 0=Monday in Python
    )


# ---------------------------------------------------------------------------
# HEARTBEAT.md parser
# ---------------------------------------------------------------------------

_TASK_BLOCK_RE = re.compile(
    r"^##\s+task:\s*(.+?)$"
    r"(.*?)"
    r"(?=^##\s+task:|\Z)",
    re.MULTILINE | re.DOTALL,
)

_KEY_VALUE_RE = re.compile(r"^(\w[\w_-]*):\s*(.+)$", re.MULTILINE)


def parse_heartbeat_md(workspace_root: str) -> list[dict[str, Any]]:
    """Parse task blocks from ``HEARTBEAT.md`` in *workspace_root*.

    Returns a list of task dicts with keys: name, message, description,
    interval (int|None), cron (str|None), observability.
    """
    path = Path(workspace_root) / _HEARTBEAT_FILENAME
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    tasks: list[dict[str, Any]] = []
    for m in _TASK_BLOCK_RE.finditer(text):
        name = m.group(1).strip()
        block = m.group(2)
        props: dict[str, str] = {}
        for kv in _KEY_VALUE_RE.finditer(block):
            props[kv.group(1).lower()] = kv.group(2).strip()
        message = props.get("message", "").strip()
        if not message:
            continue
        task: dict[str, Any] = {
            "name": name,
            "message": message,
            "description": props.get("description", ""),
            "interval": None,
            "cron": None,
            "observability": props.get("observability", "noop"),
        }
        if "interval" in props:
            try:
                task["interval"] = int(props["interval"])
            except ValueError:
                pass
        if "cron" in props:
            task["cron"] = props["cron"]
        tasks.append(task)
    return tasks


# ---------------------------------------------------------------------------
# Observability hooks
# ---------------------------------------------------------------------------

_METRICS_LOG_PATH = Path.home() / ".ollamacode" / "scheduler_metrics.jsonl"


def _emit_event(task_name: str, mode: str, status: str, output: str = "", error: str = "") -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    event = {
        "ts": ts,
        "task": task_name,
        "status": status,
        "output_chars": len(output),
        "error": error[:200] if error else "",
    }
    if mode == "log":
        logger.info("scheduler: %s", json.dumps(event))
    elif mode == "metrics":
        try:
            _METRICS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_METRICS_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except OSError:
            pass
    # noop: no output


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task_now(task: dict[str, Any], model: str = "gpt-oss:20b", config: dict[str, Any] | None = None) -> str:
    """Run a scheduled task immediately.  Returns the agent's response text."""
    import asyncio

    message = task.get("message", "").strip()
    if not message:
        raise ValueError(f"Task '{task.get('name')}' has no message")

    from .agent import run_agent_loop_no_mcp

    provider = None
    if config:
        try:
            from .providers import get_provider

            pname = config.get("provider", "ollama")
            if pname != "ollama":
                provider = get_provider(config)
        except Exception:
            pass

    use_model = (config or {}).get("model") or model

    async def _run() -> str:
        return await run_agent_loop_no_mcp(use_model, message, provider=provider)

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class Scheduler:
    """Background daemon thread that fires tasks on interval or cron schedule."""

    def __init__(
        self,
        tasks: list[dict[str, Any]],
        model: str = "gpt-oss:20b",
        config: dict[str, Any] | None = None,
        tick_interval: float = 30.0,
    ) -> None:
        self.tasks = tasks
        self.model = model
        self.config = config or {}
        self.tick_interval = tick_interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Per-task: last run timestamp (monotonic)
        self._last_run: dict[str, float] = {}
        # Track which cron minute we last fired (to avoid double-fire within same minute)
        self._last_cron_minute: dict[str, str] = {}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ollamacode-scheduler")
        self._thread.start()
        logger.info("Scheduler started with %d task(s).", len(self.tasks))

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Scheduler stopped.")

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            now_mono = time.monotonic()
            now_dt = datetime.now(timezone.utc)
            minute_key = now_dt.strftime("%Y%m%d%H%M")

            for task in self.tasks:
                name = task.get("name", "?")
                interval = task.get("interval")
                cron_expr = task.get("cron")
                obs = task.get("observability", "noop")

                should_run = False
                if interval and isinstance(interval, int) and interval > 0:
                    last = self._last_run.get(name, 0.0)
                    if now_mono - last >= interval:
                        should_run = True
                elif cron_expr:
                    fired_key = self._last_cron_minute.get(name, "")
                    if fired_key != minute_key and cron_matches(cron_expr, now_dt):
                        should_run = True
                        self._last_cron_minute[name] = minute_key

                if should_run:
                    self._last_run[name] = now_mono
                    self._fire_task(task, obs)

            self._stop_event.wait(timeout=self.tick_interval)

    def _fire_task(self, task: dict[str, Any], obs: str) -> None:
        name = task.get("name", "?")
        logger.info("Scheduler: running task '%s'", name)
        try:
            output = run_task_now(task, model=self.model, config=self.config)
            _emit_event(name, obs, "ok", output=output)
        except Exception as exc:
            logger.warning("Scheduler: task '%s' failed: %s", name, exc)
            _emit_event(name, obs, "error", error=str(exc))


# ---------------------------------------------------------------------------
# Config loading helper
# ---------------------------------------------------------------------------

def load_scheduled_tasks(
    config: dict[str, Any],
    workspace_root: str = ".",
) -> list[dict[str, Any]]:
    """Merge scheduled_tasks from config + HEARTBEAT.md."""
    from_config: list[dict[str, Any]] = config.get("scheduled_tasks") or []
    from_heartbeat = parse_heartbeat_md(workspace_root)
    # Config tasks take precedence; HEARTBEAT tasks fill in names not already defined.
    config_names = {t.get("name") for t in from_config if t.get("name")}
    merged = list(from_config)
    for t in from_heartbeat:
        if t.get("name") not in config_names:
            merged.append(t)
    return merged
