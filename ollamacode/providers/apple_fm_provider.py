"""Apple Foundation Models provider via apple-fm-sdk.

This provider bridges OllamaCode's provider interface to Apple's on-device
Foundation Models SDK for Python.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
import logging
from pathlib import Path
from typing import Any, Generator, Optional, Union, get_args, get_origin

from .base import BaseProvider, ProviderCapabilities
from ..bridge import BUILTIN_SERVER_PREFIXES

_IMPORT_ERROR_MSG = (
    "The 'apple-fm-sdk' package is required for provider 'apple_fm'. "
    'Install it with: uv pip install "apple-fm-sdk @ '
    'git+https://github.com/apple/python-apple-fm-sdk.git"'
)
_TRANSCRIPT_PATH = os.environ.get("OLLAMACODE_APPLE_FM_TRANSCRIPT_PATH", "").strip()
_JSON_SCHEMA_PATH = os.environ.get("OLLAMACODE_APPLE_FM_JSON_SCHEMA_PATH", "").strip()
_JSON_SCHEMA_RAW = os.environ.get("OLLAMACODE_APPLE_FM_JSON_SCHEMA", "").strip()

logger = logging.getLogger(__name__)


def _normalize_tool_name(name: str) -> str:
    if name.startswith("functions::"):
        name = name[len("functions::") :]
    for prefix in BUILTIN_SERVER_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _allowed_tool_names(tools: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        out.add(_normalize_tool_name(name))
    return out


def _tool_name_score(name: str) -> tuple[int, int]:
    """
    Lower score is better.
    Prefer canonical short names over alias/prefixed variants:
    - no 'functions::' prefix
    - no MCP server prefix with '-'
    - shorter overall name
    """
    n = name or ""
    pref_penalty = 1 if n.startswith("functions::") else 0
    prefix_penalty = 1 if "-" in n else 0
    return (pref_penalty + prefix_penalty, len(n))


def _canonicalize_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Deduplicate tools by normalized name and keep the most canonical entry.
    """
    by_name: dict[str, dict[str, Any]] = {}
    by_score: dict[str, tuple[int, int]] = {}
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict):
            continue
        raw_name = str(fn.get("name") or "").strip()
        if not raw_name:
            continue
        norm = _normalize_tool_name(raw_name)
        score = _tool_name_score(raw_name)
        prev = by_score.get(norm)
        if prev is None or score < prev:
            by_name[norm] = t
            by_score[norm] = score
    # Stable ordering for prompt reproducibility.
    return [by_name[k] for k in sorted(by_name.keys())]


def _tool_catalog_text(tools: list[dict[str, Any]]) -> str:
    catalog: list[dict[str, Any]] = []
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict):
            continue
        catalog.append(
            {
                "name": fn.get("name") or "",
                "description": fn.get("description") or "",
                "parameters": fn.get("parameters") or {},
            }
        )
    return json.dumps(catalog, ensure_ascii=False)


def _tool_name_list_text(tools: list[dict[str, Any]]) -> str:
    names: list[str] = []
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or "").strip()
        if name:
            names.append(_normalize_tool_name(name))
    return json.dumps(sorted(set(names)), ensure_ascii=False)


def _compress_text(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if max_chars <= 0 or len(t) <= max_chars:
        return t
    head = int(max_chars * 0.7)
    tail = max_chars - head
    return t[:head] + "\n...\n" + t[-tail:]


def _memory_aware_system_text(system_text: str, messages: list[dict[str, Any]]) -> str:
    base_max = int(os.environ.get("OLLAMACODE_APPLE_FM_MAX_SYSTEM_CHARS", "2500"))
    budget = int(os.environ.get("OLLAMACODE_APPLE_FM_CONTEXT_BUDGET_CHARS", "12000"))
    min_system = int(os.environ.get("OLLAMACODE_APPLE_FM_MIN_SYSTEM_CHARS", "500"))
    if base_max <= 0:
        return ""
    non_system_chars = 0
    for m in messages:
        if str(m.get("role") or "").lower() == "system":
            continue
        content = str(m.get("content") or "")
        non_system_chars += len(content)
        tool_calls = m.get("tool_calls") or []
        if tool_calls:
            try:
                non_system_chars += len(
                    json.dumps(tool_calls, ensure_ascii=False, separators=(",", ":"))
                )
            except Exception:
                pass
    remaining = budget - non_system_chars
    if remaining <= 0:
        return _compress_text(system_text, min_system)
    target = min(base_max, max(min_system, remaining))
    return _compress_text(system_text, target)


def _load_json_schema() -> dict[str, Any] | None:
    if _JSON_SCHEMA_RAW:
        try:
            obj = json.loads(_JSON_SCHEMA_RAW)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    if _JSON_SCHEMA_PATH:
        try:
            raw = Path(_JSON_SCHEMA_PATH).read_text(encoding="utf-8")
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


async def _maybe_save_transcript(
    session: Any,
    *,
    model: str,
    use_tools: bool,
    note: str | None = None,
    duration_s: float | None = None,
    prompt_chars: int | None = None,
    response_chars: int | None = None,
    call_id: str | None = None,
) -> None:
    if not _TRANSCRIPT_PATH:
        return
    try:
        transcript = await session.transcript.to_dict()
        metrics = _summarize_transcript(transcript)
        if duration_s is not None:
            metrics["latency_s"] = round(duration_s, 4)
        if prompt_chars is not None:
            metrics["prompt_chars"] = prompt_chars
        if response_chars is not None:
            metrics["response_chars"] = response_chars
        payload = {
            "provider": "apple_fm",
            "model": model,
            "use_tools": use_tools,
            "note": note or "",
            "transcript": transcript,
            "metrics": metrics,
        }
        if call_id:
            payload["call_id"] = call_id
        payload["ts"] = round(time.time(), 3)
        with open(_TRANSCRIPT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning(
            "Failed to write Apple FM transcript to %s: %s", _TRANSCRIPT_PATH, exc
        )


def _summarize_transcript(transcript: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    entries = (
        transcript.get("transcript", {}).get("entries", [])
        if isinstance(transcript, dict)
        else []
    )
    role_counts: dict[str, int] = {}
    text_chars = 0
    tool_calls = 0
    tool_call_names: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        role = str(entry.get("role") or "")
        if role:
            role_counts[role] = role_counts.get(role, 0) + 1
        contents = entry.get("contents") or []
        if isinstance(contents, list):
            for c in contents:
                if isinstance(c, dict) and c.get("type") == "text":
                    text = str(c.get("text") or "")
                    text_chars += len(text)
        tc = entry.get("toolCalls")
        if isinstance(tc, list):
            tool_calls += len(tc)
            for call in tc:
                if not isinstance(call, dict):
                    continue
                name = (
                    call.get("toolName")
                    or call.get("name")
                    or call.get("function", {}).get("name")
                )
                if isinstance(name, str) and name:
                    tool_call_names[name] = tool_call_names.get(name, 0) + 1
    if role_counts:
        metrics["entry_counts"] = role_counts
    metrics["text_chars"] = text_chars
    metrics["tool_calls"] = tool_calls
    if tool_call_names:
        metrics["tool_call_names"] = tool_call_names
    usage: dict[str, Any] = {}
    if isinstance(transcript, dict):
        for key, value in transcript.items():
            if isinstance(value, (int, float)) and any(
                k in key.lower() for k in ("token", "latency", "duration", "usage")
            ):
                usage[key] = value
    if usage:
        metrics["usage_fields"] = usage
    if text_chars:
        metrics["estimated_tokens"] = max(1, int(text_chars / 4))
    return metrics


def _messages_to_prompt(
    messages: list[dict[str, Any]],
    *,
    max_chars: int = 12000,
    include_tool_calls: bool = True,
) -> str:
    parts: list[str] = []
    for m in messages:
        role = str((m.get("role") or "user")).strip().upper()
        if role == "SYSTEM":
            continue
        content = str(m.get("content") or "")
        parts.append(f"{role}:\n{content[:max_chars]}")

        if include_tool_calls:
            tool_calls = m.get("tool_calls") or []
            if tool_calls:
                parts.append(
                    "TOOL_CALLS:\n"
                    + json.dumps(tool_calls, ensure_ascii=False, separators=(",", ":"))
                )

        if role == "TOOL":
            tool_name = str(m.get("tool_name") or "")
            if tool_name:
                parts.append(f"TOOL_NAME: {tool_name}")

    return "\n\n".join(parts)


def _last_user_prompt(messages: list[dict[str, Any]], default: str = "Hello") -> str:
    for m in reversed(messages):
        if str(m.get("role") or "").lower() == "user":
            content = str(m.get("content") or "").strip()
            if content:
                return content
    return default


def _is_text_transform_request(messages: list[dict[str, Any]]) -> bool:
    text = _last_user_prompt(messages, default="").lower()
    if not text:
        return False
    transform_markers = [
        "rephrase",
        "rewrite",
        "paraphrase",
        "summarize",
        "summary",
        "translate",
        "proofread",
        "fix grammar",
        "fix typos",
        "simplify",
        "shorten",
        "condense",
        "expand",
        "edit for",
    ]
    code_markers = ["code", "python", "function", "script", "program", "regex", "sql"]
    if any(m in text for m in code_markers):
        return False
    return any(m in text for m in transform_markers)


def _normalize_tool_calls(
    raw_tool_calls: list[Any],
    allowed_names: set[str],
    *,
    max_calls: int = 8,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for tc in raw_tool_calls[:max_calls]:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        if not isinstance(fn, dict):
            continue
        name = _normalize_tool_name(str(fn.get("name") or "").strip())
        if not name or name not in allowed_names:
            continue
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if not isinstance(args, dict):
            args = {}
        normalized.append({"function": {"name": name, "arguments": args}})
    return normalized


def _extract_json_candidates(raw: str) -> list[str]:
    candidates: list[str] = [raw]
    if raw.startswith("```") and raw.endswith("```"):
        inner = raw.strip("`").strip()
        if inner.lower().startswith("json"):
            inner = inner[4:].strip()
        candidates.append(inner)
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(raw[start : end + 1])
    return candidates


def _parse_tool_call_response(
    text: str,
    allowed_names: set[str],
) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None

    for c in _extract_json_candidates(raw):
        try:
            obj = json.loads(c)
        except Exception:
            continue
        parsed = _normalize_structured_response(obj, allowed_names)
        if parsed is not None:
            return parsed
    return None


def _normalize_structured_response(
    obj: Any,
    allowed_names: set[str],
) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None

    tool_calls = obj.get("tool_calls")
    if isinstance(tool_calls, list):
        normalized = _normalize_tool_calls(tool_calls, allowed_names)
        if normalized:
            return {
                "message": {
                    "content": str(obj.get("content") or ""),
                    "tool_calls": normalized,
                }
            }
        return None

    content = obj.get("content")
    if isinstance(content, str) and content.strip():
        return {"message": {"content": content.strip()}}
    return None


def _is_context_window_error(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    name = exc.__class__.__name__.lower()
    return "context window" in msg or "exceededcontextwindowsize" in name


def _is_guardrail_error(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    name = exc.__class__.__name__.lower()
    return "guardrail" in msg or "guardrail" in name or "refusal" in name


_MAX_ENUM_SIZE = 64  # Apple FM SDK may reject very large enums


def _tool_router_schema(allowed_names: set[str]) -> dict[str, Any]:
    names = sorted(allowed_names) if allowed_names else [""]
    if len(names) > _MAX_ENUM_SIZE:
        logger.warning(
            "Tool enum has %d names (max %d); truncating to first %d. "
            "Some tools may not be callable via structured output.",
            len(names),
            _MAX_ENUM_SIZE,
            _MAX_ENUM_SIZE,
        )
        names = names[:_MAX_ENUM_SIZE]
    return {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "function": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "enum": names},
                                "arguments": {"type": "object"},
                            },
                            "required": ["name", "arguments"],
                            "additionalProperties": False,
                        }
                    },
                    "required": ["function"],
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }


def _resolve_schema_ref(
    schema: dict[str, Any],
    defs: dict[str, Any],
) -> dict[str, Any]:
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/"):
        path = ref[2:].split("/")
        cur: Any = {"$defs": defs, "definitions": defs}
        for part in path:
            if not isinstance(cur, dict) or part not in cur:
                return schema
            cur = cur[part]
        if isinstance(cur, dict):
            merged = dict(cur)
            merged.update({k: v for k, v in schema.items() if k != "$ref"})
            return merged
    return schema


def _unwrap_optional(type_class: Any) -> Any:
    origin = get_origin(type_class)
    if origin is Union:
        args = [a for a in get_args(type_class) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return type_class


def _infer_type_from_values(values: list[Any]) -> Any:
    if not values:
        return str
    if all(isinstance(v, bool) for v in values):
        return bool
    if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return int
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return float
    return str


def _schema_to_guides(
    schema: dict[str, Any],
    type_class: Any,
    defs: dict[str, Any] | None = None,
) -> list[Any]:
    guides: list[Any] = []
    try:
        import apple_fm_sdk as fm  # type: ignore[import-not-found]
    except Exception:
        return guides

    if defs is None:
        defs = {}

    schema = _resolve_schema_ref(schema, defs)
    variants = None
    if isinstance(schema.get("anyOf"), list):
        variants = schema.get("anyOf")
    elif isinstance(schema.get("oneOf"), list):
        variants = schema.get("oneOf")

    if variants:
        enum_union: list[Any] = []
        for option in variants:
            if not isinstance(option, dict):
                continue
            option = _resolve_schema_ref(option, defs)
            if "enum" in option and isinstance(option.get("enum"), list):
                enum_union.extend(option.get("enum") or [])
            if "const" in option:
                enum_union.append(option.get("const"))
        if enum_union:
            enum_union = list(dict.fromkeys(enum_union))
            if _unwrap_optional(type_class) is str:
                guides.append(fm.GenerationGuide.anyOf([str(v) for v in enum_union]))
            elif len(enum_union) == 1 and isinstance(
                enum_union[0], (str, int, float, bool)
            ):
                guides.append(fm.GenerationGuide.constant(str(enum_union[0])))

    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        if type_class is str:
            guides.append(fm.GenerationGuide.anyOf([str(v) for v in enum]))
        elif len(enum) == 1 and isinstance(enum[0], (str, int, float, bool)):
            guides.append(fm.GenerationGuide.constant(str(enum[0])))

    if "const" in schema:
        const_val = schema.get("const")
        if isinstance(const_val, (str, int, float, bool)):
            guides.append(fm.GenerationGuide.constant(str(const_val)))

    pattern = schema.get("pattern")
    if isinstance(pattern, str) and type_class is str:
        guides.append(fm.GenerationGuide.regex(pattern))

    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    if isinstance(minimum, (int, float)) and isinstance(maximum, (int, float)):
        guides.append(fm.GenerationGuide.range((minimum, maximum)))
    elif isinstance(minimum, (int, float)):
        guides.append(fm.GenerationGuide.minimum(minimum))
    elif isinstance(maximum, (int, float)):
        guides.append(fm.GenerationGuide.maximum(maximum))

    min_items = schema.get("minItems")
    max_items = schema.get("maxItems")
    if (
        isinstance(min_items, int)
        and isinstance(max_items, int)
        and min_items == max_items
    ):
        guides.append(fm.GenerationGuide.count(min_items))
    else:
        if isinstance(min_items, int):
            guides.append(fm.GenerationGuide.min_items(min_items))
        if isinstance(max_items, int):
            guides.append(fm.GenerationGuide.max_items(max_items))

    base_type = _unwrap_optional(type_class)
    if get_origin(base_type) is list:
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            items_schema = _resolve_schema_ref(items_schema, defs)
            item_type = _infer_type_from_values(items_schema.get("enum") or [])
            if "type" in items_schema:
                t = items_schema.get("type")
                if t == "string":
                    item_type = str
                elif t == "integer":
                    item_type = int
                elif t == "number":
                    item_type = float
                elif t == "boolean":
                    item_type = bool
            element_guides = _schema_to_guides(items_schema, item_type, defs)
            for g in element_guides:
                guides.append(fm.GenerationGuide.element(g))
    return guides


def _schema_type_to_python(
    schema: dict[str, Any],
    name_hint: str,
    defs: dict[str, Any],
) -> tuple[Any, list[Any]]:
    schema = _resolve_schema_ref(schema, defs)
    nested: list[Any] = []

    if "anyOf" in schema and isinstance(schema["anyOf"], list):
        for option in schema["anyOf"]:
            if isinstance(option, dict):
                option = _resolve_schema_ref(option, defs)
                if option.get("type") and option.get("type") != "null":
                    return _schema_type_to_python(option, name_hint, defs)

    if "oneOf" in schema and isinstance(schema["oneOf"], list):
        for option in schema["oneOf"]:
            if isinstance(option, dict):
                option = _resolve_schema_ref(option, defs)
                if option.get("type") and option.get("type") != "null":
                    return _schema_type_to_python(option, name_hint, defs)

    raw_type = schema.get("type")
    if isinstance(raw_type, list):
        raw_type = next((t for t in raw_type if t != "null"), raw_type[0])

    if raw_type is None:
        if "const" in schema:
            raw_type = type(schema.get("const")).__name__
        elif isinstance(schema.get("enum"), list):
            inferred = _infer_type_from_values(schema.get("enum") or [])
            if inferred is str:
                raw_type = "string"
            elif inferred is int:
                raw_type = "integer"
            elif inferred is float:
                raw_type = "number"
            elif inferred is bool:
                raw_type = "boolean"

    if raw_type == "string":
        return str, nested
    if raw_type == "integer":
        return int, nested
    if raw_type == "number":
        return float, nested
    if raw_type == "boolean":
        return bool, nested
    if raw_type == "array":
        items = schema.get("items") or {}
        item_type, item_nested = _schema_type_to_python(
            items if isinstance(items, dict) else {}, f"{name_hint}Item", defs
        )
        nested.extend(item_nested)
        try:
            from typing import List as _List
        except Exception:
            _List = list  # type: ignore[assignment]
        return _List[item_type], nested
    if raw_type == "object" or isinstance(schema.get("properties"), dict):
        try:
            import apple_fm_sdk as fm  # type: ignore[import-not-found]
            from apple_fm_sdk.generation_property import Property  # type: ignore[import-not-found]
        except Exception:
            return dict, nested
        properties: list[Any] = []
        required = set(schema.get("required") or [])
        for prop_name, prop_schema in (schema.get("properties") or {}).items():
            if not isinstance(prop_schema, dict):
                continue
            prop_schema = _resolve_schema_ref(prop_schema, defs)
            prop_type, prop_nested = _schema_type_to_python(
                prop_schema, f"{name_hint}_{prop_name}", defs
            )
            nested.extend(prop_nested)
            if prop_name not in required:
                prop_type = Optional[prop_type]
            guides = _schema_to_guides(prop_schema, prop_type, defs)
            prop_desc = (
                str(prop_schema.get("description") or "")
                if isinstance(prop_schema, dict)
                else ""
            )
            properties.append(
                Property(
                    name=prop_name,
                    type_class=prop_type,
                    description=prop_desc or None,
                    guides=guides,
                )
            )
        cls = type(f"{name_hint}Args", (), {"__module__": __name__})
        nested_schema = fm.GenerationSchema(
            type_class=cls,
            description=str(schema.get("description") or "") or None,
            properties=properties,
            dynamic_nested_types=[ns for ns in nested if hasattr(ns, "_ptr")],
        )
        nested.append(nested_schema)
        return cls, nested
    return str, nested


def _json_schema_to_generation_schema(
    schema: dict[str, Any],
    name_hint: str,
) -> Any:
    try:
        import apple_fm_sdk as fm  # type: ignore[import-not-found]
        from apple_fm_sdk.generation_property import Property  # type: ignore[import-not-found]
    except Exception:
        return None
    defs = {}
    if isinstance(schema.get("$defs"), dict):
        defs.update(schema.get("$defs") or {})
    if isinstance(schema.get("definitions"), dict):
        defs.update(schema.get("definitions") or {})

    schema = _resolve_schema_ref(schema, defs)
    if not isinstance(schema, dict):
        return None
    properties: list[Any] = []
    required = set(schema.get("required") or [])
    nested: list[Any] = []
    for prop_name, prop_schema in (schema.get("properties") or {}).items():
        if not isinstance(prop_schema, dict):
            continue
        prop_schema = _resolve_schema_ref(prop_schema, defs)
        prop_type, prop_nested = _schema_type_to_python(
            prop_schema, f"{name_hint}_{prop_name}", defs
        )
        nested.extend(prop_nested)
        if prop_name not in required:
            prop_type = Optional[prop_type]
        guides = _schema_to_guides(prop_schema, prop_type, defs)
        prop_desc = str(prop_schema.get("description") or "")
        properties.append(
            Property(
                name=prop_name,
                type_class=prop_type,
                description=prop_desc or None,
                guides=guides,
            )
        )
    cls = type(f"{name_hint}Args", (), {"__module__": __name__})
    nested_schemas = [ns for ns in nested if hasattr(ns, "_ptr")]
    return fm.GenerationSchema(
        type_class=cls,
        description=str(schema.get("description") or "") or None,
        properties=properties,
        dynamic_nested_types=nested_schemas,
    )


def _build_fm_tools(
    tools: list[dict[str, Any]],
    executor,
) -> list[Any]:
    """Build apple-fm-sdk Tool objects with structured arguments."""
    import apple_fm_sdk as fm  # type: ignore[import-not-found]

    fm_tools: list[Any] = []
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict):
            continue
        tool_name = str(fn.get("name") or "").strip()
        tool_desc = str(fn.get("description") or "").strip()
        if not tool_name:
            continue

        args_schema: Any | None = None
        params = fn.get("parameters") if isinstance(fn, dict) else None
        if isinstance(params, dict):
            args_schema = _json_schema_to_generation_schema(params, tool_name)
        if args_schema is None:

            @fm.generable(f"{tool_name}_args")
            class _Args:  # type: ignore
                json: str = fm.guide(
                    'JSON object with arguments for the tool. Example: {"path":"file.txt"}'
                )

            args_schema = _Args.generation_schema()

        class _Tool(fm.Tool):  # type: ignore
            name = tool_name
            description = tool_desc or f"Tool {tool_name}"

            @property
            def arguments_schema(self) -> fm.GenerationSchema:
                return args_schema

            async def call(self, args: Any) -> str:
                try:
                    parsed = args.value()
                except Exception:
                    parsed = {}
                if not isinstance(parsed, dict):
                    parsed = {}
                try:
                    result = await executor(tool_name, parsed)
                except Exception as e:
                    return f"Tool error: {e}"
                return str(result)

        fm_tools.append(_Tool())
    return fm_tools


class AppleFMProvider(BaseProvider):
    """Provider backed by Apple Foundation Models SDK (apple-fm-sdk)."""

    def __init__(self) -> None:
        super().__init__()
        self._tool_executor = None
        self._session_cache: dict[str, tuple[Any, float]] = {}
        self._session_cache_lock = threading.Lock()

    def _session_key(
        self,
        *,
        model: str,
        instructions: str,
        tools: list[dict[str, Any]] | None,
        native_tools: bool,
    ) -> str:
        override = os.environ.get("OLLAMACODE_APPLE_FM_SESSION_KEY", "").strip()
        if override:
            return override
        tool_names: list[str] = []
        if tools:
            for t in tools:
                fn = t.get("function") if isinstance(t, dict) else None
                if not isinstance(fn, dict):
                    continue
                name = str(fn.get("name") or "").strip()
                if name:
                    tool_names.append(name)
        key_parts = [
            model,
            instructions or "",
            "native_tools" if native_tools else "prompt_tools",
            ",".join(sorted(tool_names)),
        ]
        return "|".join(key_parts)

    def _get_session(
        self,
        *,
        model: str,
        instructions: str,
        tools: list[dict[str, Any]] | None,
        native_tools: bool,
        fm,
    ) -> Any:
        if os.environ.get("OLLAMACODE_APPLE_FM_STATEFUL", "0") != "1":
            return fm.LanguageModelSession(
                instructions=instructions or None, tools=tools
            )
        now = time.time()
        ttl = float(os.environ.get("OLLAMACODE_APPLE_FM_SESSION_TTL_SECONDS", "3600"))
        max_sessions = int(os.environ.get("OLLAMACODE_APPLE_FM_SESSION_MAX", "32"))
        key = self._session_key(
            model=model,
            instructions=instructions,
            tools=tools,
            native_tools=native_tools,
        )
        with self._session_cache_lock:
            cached = self._session_cache.get(key)
            if cached is not None:
                session, _ = cached
                self._session_cache[key] = (session, now)
                if os.environ.get("OLLAMACODE_APPLE_FM_LOG_SESSIONS", "0") == "1":
                    logger.info("Apple FM session hit: %s", key)
                return session
            # Prune expired sessions
            if ttl > 0 and self._session_cache:
                expired = [
                    k for k, (_, ts) in self._session_cache.items() if now - ts > ttl
                ]
                for k in expired:
                    self._session_cache.pop(k, None)
            # Enforce max session count (drop oldest)
            if max_sessions > 0 and len(self._session_cache) >= max_sessions:
                if self._session_cache:
                    oldest_key = min(
                        self._session_cache.items(), key=lambda kv: kv[1][1]
                    )[0]
                    self._session_cache.pop(oldest_key, None)
        session = fm.LanguageModelSession(
            instructions=instructions or None, tools=tools
        )
        with self._session_cache_lock:
            self._session_cache[key] = (session, now)
        if os.environ.get("OLLAMACODE_APPLE_FM_LOG_SESSIONS", "0") == "1":
            logger.info("Apple FM session created: %s", key)
        return session

    async def chat_async(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        # apple-fm-sdk currently uses the system model configuration.
        try:
            import apple_fm_sdk as fm  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(_IMPORT_ERROR_MSG) from e

        system_text = ""
        for m in messages:
            if str(m.get("role") or "").lower() == "system":
                system_text = str(m.get("content") or "")
                break
        text_only = _is_text_transform_request(messages)
        if text_only:
            system_text = (
                system_text
                + "\n\nYou are a writing assistant. Focus on rewriting text only. "
                "Return only the rewritten text. Do not write code."
            ).strip()
        system_text_compact = _memory_aware_system_text(system_text, messages)

        allowed_names = _allowed_tool_names(tools or [])
        canonical_tools = _canonicalize_tools(tools or [])
        max_catalog_tools = int(
            os.environ.get("OLLAMACODE_APPLE_FM_MAX_CATALOG_TOOLS", "48")
        )
        bounded_tools = (
            canonical_tools[: max(1, max_catalog_tools)] if canonical_tools else []
        )
        json_schema = _load_json_schema() if not bounded_tools else None

        if text_only:
            allowed_names = set()
            canonical_tools = []
            bounded_tools = []
            json_schema = None

        # Native tool path (optional): let apple-fm-sdk call tools directly.
        if (
            os.environ.get("OLLAMACODE_APPLE_FM_NATIVE_TOOLS", "0") == "1"
            and self._tool_executor is not None
            and bounded_tools
        ):
            try:
                call_id = uuid.uuid4().hex
                fm_tools = _build_fm_tools(bounded_tools, self._tool_executor)
                session = self._get_session(
                    model=model,
                    instructions=system_text_compact or "",
                    tools=fm_tools,
                    native_tools=True,
                    fm=fm,
                )
                prompt = (
                    "You are a tool-calling coding assistant. Use tools when needed.\n\n"
                    + _messages_to_prompt(messages, include_tool_calls=not text_only)
                )
                t0 = time.perf_counter()
                reply = await session.respond(prompt)
                elapsed = time.perf_counter() - t0
                await _maybe_save_transcript(
                    session,
                    model=model,
                    use_tools=True,
                    note="native_tools",
                    duration_s=elapsed,
                    prompt_chars=len(prompt),
                    response_chars=len(str(reply or "")),
                    call_id=call_id,
                )
                return {"message": {"content": str(reply or "")}}
            except Exception:
                # Fall back to prompt-based tool routing.
                pass

        session = self._get_session(
            model=model,
            instructions=system_text_compact or "",
            tools=None,
            native_tools=False,
            fm=fm,
        )
        prompt_variants: list[tuple[bool, str]] = []
        if bounded_tools and not text_only:
            prompt_variants.append(
                (
                    True,
                    (
                        "You are a tool-calling coding assistant. "
                        "When tools are needed, output tool calls only for listed tools. "
                        "Otherwise return plain assistant text in 'content'.\n\n"
                        "Available tools (JSON):\n"
                        + _tool_catalog_text(bounded_tools)
                        + "\n\n"
                        + _messages_to_prompt(messages)
                    ).strip(),
                )
            )
            prompt_variants.append(
                (
                    True,
                    (
                        "You may call tools only from this list of names:\n"
                        + _tool_name_list_text(canonical_tools)
                        + "\n\n"
                        + _messages_to_prompt(messages, max_chars=6000)
                    ).strip(),
                )
            )
        if text_only:
            prompt_variants.append(
                (
                    False,
                    _messages_to_prompt(
                        messages, max_chars=9000, include_tool_calls=False
                    ),
                )
            )
        else:
            prompt_variants.append(
                (False, _messages_to_prompt(messages, max_chars=5000))
            )

        last_error: Exception | None = None
        for use_tools, full_prompt in prompt_variants:
            call_id = uuid.uuid4().hex
            if json_schema and not use_tools:
                try:
                    t0 = time.perf_counter()
                    structured = await session.respond(
                        full_prompt, json_schema=json_schema
                    )
                    elapsed = time.perf_counter() - t0
                    raw_structured = (
                        structured.value()
                        if hasattr(structured, "value")
                        else structured
                    )
                    if raw_structured is not None:
                        await _maybe_save_transcript(
                            session,
                            model=model,
                            use_tools=False,
                            note="json_schema",
                            duration_s=elapsed,
                            prompt_chars=len(full_prompt),
                            response_chars=len(
                                json.dumps(raw_structured, ensure_ascii=False)
                            ),
                            call_id=call_id,
                        )
                        return {
                            "message": {
                                "content": json.dumps(
                                    raw_structured, ensure_ascii=False
                                )
                            }
                        }
                except Exception as e:
                    if _is_context_window_error(e) or _is_guardrail_error(e):
                        last_error = e
                        continue
                    logger.warning("apple_fm json_schema respond failed: %s", e)
                    raise
            if use_tools:
                try:
                    t0 = time.perf_counter()
                    structured = await session.respond(
                        full_prompt,
                        json_schema=_tool_router_schema(allowed_names),
                    )
                    elapsed = time.perf_counter() - t0
                    raw_structured = (
                        structured.value() if hasattr(structured, "value") else None
                    )
                    parsed_structured = _normalize_structured_response(
                        raw_structured, allowed_names
                    )
                    if parsed_structured is not None:
                        await _maybe_save_transcript(
                            session,
                            model=model,
                            use_tools=True,
                            note="structured",
                            duration_s=elapsed,
                            prompt_chars=len(full_prompt),
                            response_chars=len(
                                json.dumps(parsed_structured, ensure_ascii=False)
                            ),
                            call_id=call_id,
                        )
                        return parsed_structured
                except Exception as e:
                    if _is_context_window_error(e) or _is_guardrail_error(e):
                        last_error = e
                        continue
                    # Log but fall through to plain response parsing for compatibility.
                    logger.warning(
                        "apple_fm structured respond failed (falling back to plain): %s",
                        e,
                    )

            try:
                t0 = time.perf_counter()
                reply = await session.respond(full_prompt)
                elapsed = time.perf_counter() - t0
            except Exception as e:
                if _is_context_window_error(e) or _is_guardrail_error(e):
                    last_error = e
                    continue
                raise

            # Plain respond succeeded — clear any earlier error so we don't
            # fall into the minimal-fallback path if parsed result is None.
            last_error = None
            reply_text = str(reply or "")
            if use_tools:
                parsed = _parse_tool_call_response(reply_text, allowed_names)
                if parsed is not None:
                    await _maybe_save_transcript(
                        session,
                        model=model,
                        use_tools=True,
                        note="parsed",
                        duration_s=elapsed,
                        prompt_chars=len(full_prompt),
                        response_chars=len(reply_text),
                        call_id=call_id,
                    )
                    return parsed
            await _maybe_save_transcript(
                session,
                model=model,
                use_tools=use_tools,
                note="text",
                duration_s=elapsed,
                prompt_chars=len(full_prompt),
                response_chars=len(reply_text),
                call_id=call_id,
            )
            return {"message": {"content": reply_text}}

        if last_error is not None:
            # Final fallback: minimal prompt and no system instructions.
            # Use the full conversation context (truncated) rather than
            # only the last user message, to preserve multi-turn context.
            try:
                minimal_session = fm.LanguageModelSession()
                prompt = _messages_to_prompt(
                    messages, max_chars=4000
                ) or _last_user_prompt(messages)
                call_id = uuid.uuid4().hex
                t0 = time.perf_counter()
                minimal_reply = await minimal_session.respond(prompt)
                elapsed = time.perf_counter() - t0
                await _maybe_save_transcript(
                    minimal_session,
                    model=model,
                    use_tools=False,
                    note="minimal",
                    duration_s=elapsed,
                    prompt_chars=len(prompt),
                    response_chars=len(str(minimal_reply or "")),
                    call_id=call_id,
                )
                return {"message": {"content": str(minimal_reply or "")}}
            except Exception as e:
                if _is_guardrail_error(e):
                    return {
                        "message": {
                            "content": "I can’t answer that request directly. Please rephrase it in a safer or narrower way."
                        }
                    }
                raise last_error
        return {"message": {"content": ""}}

    def chat_stream_sync(
        self,
        model: str,
        messages: list[dict[str, Any]],
    ) -> Generator[tuple[str], None, None]:
        # apple-fm-sdk currently uses the system model configuration.
        try:
            import apple_fm_sdk as fm  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(_IMPORT_ERROR_MSG) from e

        system_text = ""
        for m in messages:
            if str(m.get("role") or "").lower() == "system":
                system_text = str(m.get("content") or "")
                break
        system_text_compact = _memory_aware_system_text(system_text, messages)
        prompt = _messages_to_prompt(messages)

        async def _collect() -> tuple[list[str], Any]:
            session = self._get_session(
                model=model,
                instructions=system_text_compact or "",
                tools=None,
                native_tools=False,
                fm=fm,
            )
            chunks: list[str] = []
            call_id = uuid.uuid4().hex
            t0 = time.perf_counter()
            try:
                async for chunk in session.stream_response(prompt):
                    chunks.append(str(chunk or ""))
            except Exception as exc:
                logger.warning("Apple FM stream interrupted: %s", exc)
                if not chunks:
                    raise
                # Partial response collected; continue with what we have.
            elapsed = time.perf_counter() - t0
            await _maybe_save_transcript(
                session,
                model=model,
                use_tools=False,
                note="stream",
                duration_s=elapsed,
                prompt_chars=len(prompt),
                response_chars=len("".join(chunks)),
                call_id=call_id,
            )
            return chunks, session

        # Avoid asyncio.run() which crashes if an event loop is already running
        # (e.g. when called from the TUI's async context).
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # We're inside an existing event loop — run in a background thread.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _collect())
                chunks, _ = future.result(timeout=300)
        else:
            chunks, _ = asyncio.run(_collect())

        for chunk in chunks:
            if chunk:
                yield (chunk,)

    def set_tool_executor(self, executor) -> None:
        self._tool_executor = executor

    def health_check(self) -> tuple[bool, str]:
        try:
            import apple_fm_sdk as fm  # type: ignore[import-not-found]
        except ImportError:
            return False, _IMPORT_ERROR_MSG

        import concurrent.futures

        def _check() -> tuple[bool, str]:
            model = fm.SystemLanguageModel()
            available, reason = model.is_available()
            if available:
                return True, "Apple Foundation Models is available."
            return False, f"Apple Foundation Models unavailable: {reason}"

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_check)
                return future.result(timeout=10)
        except concurrent.futures.TimeoutError:
            return False, "apple_fm health check timed out after 10s"
        except Exception as e:
            return False, f"apple_fm error: {e}"

    def list_models(self) -> list[str]:
        return ["apple.system"]

    @property
    def name(self) -> str:
        return "apple_fm"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_tools=True,
            supports_streaming=True,
            supports_embeddings=False,
            supports_model_list=False,
        )
