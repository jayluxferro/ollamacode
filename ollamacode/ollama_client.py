"""
Centralised Ollama chat/generate calls with template-error fallback.

When the Chat API raises a template error (e.g. slice index), we fall back to the
generate API (no template) so older or differently templated models still work.
Fallback is only used when no tools are requested (tools=[]).
"""

from __future__ import annotations

from typing import Any, Generator

import ollama


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _messages_to_system_and_prompt(
    messages: list[dict[str, Any]],
) -> tuple[str | None, str]:
    """Convert chat messages to (system, prompt) for the generate API."""
    system: str | None = None
    parts: list[str] = []
    for m in messages:
        role = (_get(m, "role") or "").strip().lower()
        content = (_get(m, "content") or "").strip()
        if role == "system":
            if system is None:
                system = content
            else:
                system = system + "\n\n" + content
        elif role == "user":
            parts.append("User: " + content)
        elif role == "assistant":
            parts.append("Assistant: " + content)
    prompt = "\n\n".join(parts) if parts else " "
    return (system or None, prompt or " ")


def is_ollama_template_error(e: Exception) -> bool:
    """True if e looks like Ollama chat template slice-index error."""
    err_text = (getattr(e, "message", None) or str(e)).lower()
    return bool(
        "template" in err_text
        and (
            "slice index" in err_text
            or "index $prop" in err_text
            or "reflect: slice" in err_text
        )
    )


def wrap_ollama_template_error(e: Exception) -> Exception:
    """Wrap template error in RuntimeError with hint; otherwise return e."""
    if is_ollama_template_error(e):
        hint = (
            "Ollama chat template error: this model's prompt template may be incompatible with the Chat API. "
            "Try --no-mcp (generate API fallback) or a different model."
        )
        err = RuntimeError(f"{e}\n\n{hint}")
        err.__cause__ = e
        return err
    return e


def _response_to_dict(response: Any) -> dict[str, Any]:
    """Normalise Ollama response to dict with 'message' for agent compatibility."""
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if isinstance(response, dict):
        return response
    return {"message": getattr(response, "message", None)}


async def chat_async(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Async Ollama chat (no thread). When tools is empty and chat raises a template error, fall back to generate API.
    Returns chat-shaped dict: {"message": {"content": ..., "tool_calls": ...}}.
    """
    tools = tools if tools is not None else []
    client = ollama.AsyncClient()
    try:
        response = await client.chat(
            model=model, messages=messages, tools=tools, stream=False
        )
        return _response_to_dict(response)
    except Exception as e:
        if not is_ollama_template_error(e):
            raise wrap_ollama_template_error(e)
        if tools:
            raise wrap_ollama_template_error(e)
        system, prompt = _messages_to_system_and_prompt(messages)
        gen_response = await client.generate(
            model=model, prompt=prompt, system=system or ""
        )
        text = (
            gen_response.response
            if hasattr(gen_response, "response")
            else (
                gen_response.get("response") if isinstance(gen_response, dict) else None
            )
        )
        return {"message": {"content": (text or "").strip()}}
    finally:
        if hasattr(client, "close"):
            await client.close()  # type: ignore[attr-defined]


def chat_sync(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> Any:
    """
    Ollama chat (non-stream). When tools is empty and chat raises a template error, fall back to generate API.
    Returns chat-shaped response: {"message": {"content": ...}}.
    With non-empty tools, no fallback (template error is re-raised wrapped).
    """
    tools = tools if tools is not None else []
    try:
        return ollama.chat(model=model, messages=messages, tools=tools, stream=False)
    except Exception as e:
        if not is_ollama_template_error(e):
            raise wrap_ollama_template_error(e)
        if tools:
            raise wrap_ollama_template_error(e)
        system, prompt = _messages_to_system_and_prompt(messages)
        gen = ollama.generate(model=model, prompt=prompt, system=system or "")
        text = (
            gen.get("response")
            if isinstance(gen, dict)
            else getattr(gen, "response", None)
        )
        return {"message": {"content": (text or "").strip()}}


def _generate_stream_fallback(
    model: str, prompt: str, system: str | None
) -> Generator[tuple[str], None, None]:
    """Yield (content,) per chunk from generate API stream."""
    kwargs: dict[str, Any] = {"model": model, "prompt": prompt, "stream": True}
    if system:
        kwargs["system"] = system
    for chunk in ollama.generate(**kwargs):
        content = (
            chunk.get("response")
            if isinstance(chunk, dict)
            else getattr(chunk, "response", "")
        )
        if content:
            yield (content,)


def chat_stream_sync(
    model: str,
    messages: list[dict[str, Any]],
) -> Generator[tuple[str], None, None]:
    """
    Stream Ollama response (no tools). Try chat stream; on template error, fall back to generate stream.
    Yields (content_fragment,) per chunk.
    """
    try:
        stream = ollama.chat(model=model, messages=messages, tools=[], stream=True)
        for chunk in stream:
            msg = (
                chunk.get("message")
                if isinstance(chunk, dict)
                else getattr(chunk, "message", None)
            )
            content = _get(msg, "content", "") or "" if msg else ""
            if content:
                yield (content,)
    except Exception as e:
        if not is_ollama_template_error(e):
            raise wrap_ollama_template_error(e)
        system, prompt = _messages_to_system_and_prompt(messages)
        yield from _generate_stream_fallback(model, prompt, system)
