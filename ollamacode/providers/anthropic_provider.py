"""Anthropic provider: uses the anthropic Python SDK directly.

Requires: pip install anthropic

Handles the message format translation between OllamaCode's
Ollama-shaped messages and Anthropic's native Messages API.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Generator

from .base import BaseProvider, ProviderCapabilities

logger = logging.getLogger(__name__)

_IMPORT_ERROR_MSG = (
    "The 'anthropic' package is required for the Anthropic provider. "
    "Install it with: pip install anthropic"
)

_DEFAULT_MAX_TOKENS = 8192


def _split_system(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract the system message and return (system_text, remaining_messages)."""
    system: str | None = None
    rest: list[dict[str, Any]] = []
    for m in messages:
        role = (m.get("role") or "").strip().lower()
        if role == "system":
            system = str(m.get("content") or "")
        else:
            rest.append(m)
    return system, rest


def _to_anthropic_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert OllamaCode-format messages to Anthropic Messages API format."""
    out: list[dict[str, Any]] = []
    for m in messages:
        role = (m.get("role") or "").strip().lower()
        content = m.get("content") or ""

        # tool result message (from MCP tool execution)
        if role == "tool":
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.get("tool_call_id")
                            or m.get("tool_name")
                            or "unknown",
                            "content": str(content),
                        }
                    ],
                }
            )
            continue

        # assistant message — may contain tool_calls
        if role == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs:
                blocks: list[dict[str, Any]] = []
                if content:
                    blocks.append({"type": "text", "text": str(content)})
                for i, tc in enumerate(tcs):
                    fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                    args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id") or f"toolu_{i:02d}",
                            "name": fn.get("name") or "",
                            "input": args,
                        }
                    )
                out.append({"role": "assistant", "content": blocks})
                continue

        if role in ("user", "assistant"):
            out.append({"role": role, "content": str(content)})

    return out


def _to_anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-style tool dicts to Anthropic tool format."""
    result: list[dict[str, Any]] = []
    for t in tools:
        fn = t.get("function") or {}
        result.append(
            {
                "name": fn.get("name") or "",
                "description": fn.get("description") or "",
                "input_schema": fn.get("parameters")
                or {"type": "object", "properties": {}},
            }
        )
    return result


def _normalize_response(response: Any) -> dict[str, Any]:
    """Convert Anthropic Message to Ollama-shaped dict."""
    content_blocks = getattr(response, "content", []) or []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in content_blocks:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(getattr(block, "text", "") or "")
        elif btype == "tool_use":
            tool_calls.append(
                {
                    "function": {
                        "name": getattr(block, "name", "") or "",
                        "arguments": getattr(block, "input", {}) or {},
                    }
                }
            )

    content = "".join(text_parts)
    out_msg: dict[str, Any] = {"content": content}
    if tool_calls:
        out_msg["tool_calls"] = tool_calls
    return {"message": out_msg}


class AnthropicProvider(BaseProvider):
    """Native Anthropic Messages API provider."""

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self._api_key = api_key
        self._base_url = base_url

    def _sync_client(self) -> Any:
        try:
            import anthropic  # type: ignore[import-not-found]

            kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            return anthropic.Anthropic(**kwargs)
        except ImportError as e:
            raise RuntimeError(_IMPORT_ERROR_MSG) from e

    def _async_client(self) -> Any:
        try:
            import anthropic  # type: ignore[import-not-found]

            kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            return anthropic.AsyncAnthropic(**kwargs)
        except ImportError as e:
            raise RuntimeError(_IMPORT_ERROR_MSG) from e

    async def chat_async(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        client = self._async_client()
        system, anth_messages = _split_system(messages)
        anth_messages = _to_anthropic_messages(anth_messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": anth_messages,
            "max_tokens": _DEFAULT_MAX_TOKENS,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = _to_anthropic_tools(tools)

        try:
            response = await client.messages.create(**kwargs)
            return _normalize_response(response)
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}") from e
        finally:
            if hasattr(client, "close"):
                await client.close()

    def chat_stream_sync(
        self,
        model: str,
        messages: list[dict[str, Any]],
    ) -> Generator[tuple[str], None, None]:
        client = self._sync_client()
        system, anth_messages = _split_system(messages)
        anth_messages = _to_anthropic_messages(anth_messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": anth_messages,
            "max_tokens": _DEFAULT_MAX_TOKENS,
        }
        if system:
            kwargs["system"] = system

        try:
            with client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    if text:
                        yield (text,)
        except Exception as e:
            raise RuntimeError(f"Anthropic stream error: {e}") from e

    def health_check(self) -> tuple[bool, str]:
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError:
            return (
                False,
                "The 'anthropic' package is required. Install: pip install anthropic",
            )
        try:
            client = anthropic.Anthropic(api_key=self._api_key)
            # Minimal call to verify auth — cheapest available model, 1 token
            client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True, "Anthropic API is reachable."
        except Exception as e:
            msg = str(e).lower()
            if (
                "401" in msg
                or "unauthorized" in msg
                or "authentication" in msg
                or "api_key" in msg
            ):
                return False, "Anthropic: invalid or missing API key."
            if "connection" in msg:
                return False, "Anthropic: connection error."
            return False, f"Anthropic error: {e}"

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_tools=True,
            supports_streaming=True,
            supports_embeddings=False,
            supports_model_list=False,
        )
