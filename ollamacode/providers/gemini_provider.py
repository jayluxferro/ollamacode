"""Gemini provider: uses the google-generativeai SDK.

Requires: pip install google-generativeai

Handles message format translation between OllamaCode's
Ollama-shaped messages and Google's Gemini API.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Generator

from .base import BaseProvider, ProviderCapabilities

logger = logging.getLogger(__name__)

_IMPORT_ERROR_MSG = (
    "The 'google-generativeai' package is required for the Gemini provider. "
    "Install it with: pip install google-generativeai"
)


def _to_gemini_contents(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert OllamaCode messages to Gemini contents format.

    Returns (system_instruction, contents).
    Gemini uses 'user' and 'model' roles (not 'assistant').
    """
    system: str | None = None
    contents: list[dict[str, Any]] = []

    for m in messages:
        role = (m.get("role") or "").strip().lower()
        content = str(m.get("content") or "")

        if role == "system":
            system = content
            continue
        if role == "tool":
            # Tool results are sent as user messages with function response
            tool_name = m.get("tool_name") or m.get("tool_call_id") or "unknown"
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "function_response": {
                                "name": tool_name,
                                "response": {"result": content},
                            }
                        }
                    ],
                }
            )
            continue

        gemini_role = "model" if role == "assistant" else "user"

        # Handle tool calls in assistant messages
        tcs = m.get("tool_calls") or []
        if tcs:
            parts: list[dict[str, Any]] = []
            if content:
                parts.append({"text": content})
            for tc in tcs:
                fn = tc.get("function") or {} if isinstance(tc, dict) else {}
                fn_name = fn.get("name", "") if isinstance(fn, dict) else ""
                args = fn.get("arguments", {}) if isinstance(fn, dict) else {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        args = {}
                parts.append({"function_call": {"name": fn_name, "args": args}})
            contents.append({"role": gemini_role, "parts": parts})
            continue

        contents.append({"role": gemini_role, "parts": [{"text": content}]})

    return system, contents


def _to_gemini_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-style tool dicts to Gemini function declarations."""
    declarations: list[dict[str, Any]] = []
    for t in tools:
        fn = t.get("function") or {}
        params = fn.get("parameters") or {"type": "object", "properties": {}}
        declarations.append(
            {
                "name": fn.get("name") or "",
                "description": fn.get("description") or "",
                "parameters": params,
            }
        )
    return declarations


def _normalize_response(response: Any) -> dict[str, Any]:
    """Convert Gemini response to Ollama-shaped dict."""
    try:
        candidate = response.candidates[0]
        parts = candidate.content.parts
    except (AttributeError, IndexError):
        return {"message": {"content": ""}}

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for part in parts:
        if hasattr(part, "text") and part.text:
            text_parts.append(part.text)
        if hasattr(part, "function_call") and part.function_call:
            fc = part.function_call
            tool_calls.append(
                {
                    "function": {
                        "name": fc.name or "",
                        "arguments": dict(fc.args) if fc.args else {},
                    },
                }
            )

    content = "".join(text_parts)
    out_msg: dict[str, Any] = {"content": content}
    if tool_calls:
        out_msg["tool_calls"] = tool_calls
    return {"message": out_msg}


class GeminiProvider(BaseProvider):
    """Google Gemini API provider."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def _get_genai(self) -> Any:
        try:
            import google.generativeai as genai  # type: ignore[import-not-found]

            genai.configure(api_key=self._api_key)
            return genai
        except ImportError as e:
            raise RuntimeError(_IMPORT_ERROR_MSG) from e

    async def chat_async(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        genai = self._get_genai()
        system, contents = _to_gemini_contents(messages)

        kwargs: dict[str, Any] = {}
        if system:
            kwargs["system_instruction"] = system
        if tools:
            kwargs["tools"] = [{"function_declarations": _to_gemini_tools(tools)}]

        try:
            gm = genai.GenerativeModel(model, **kwargs)
            response = await gm.generate_content_async(contents)
            return _normalize_response(response)
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}") from e

    def chat_stream_sync(
        self,
        model: str,
        messages: list[dict[str, Any]],
    ) -> Generator[tuple[str], None, None]:
        genai = self._get_genai()
        system, contents = _to_gemini_contents(messages)

        kwargs: dict[str, Any] = {}
        if system:
            kwargs["system_instruction"] = system

        try:
            gm = genai.GenerativeModel(model, **kwargs)
            response = gm.generate_content(contents, stream=True)
            for chunk in response:
                text = getattr(chunk, "text", "") or ""
                if text:
                    yield (text,)
        except Exception as e:
            raise RuntimeError(f"Gemini stream error: {e}") from e

    def health_check(self) -> tuple[bool, str]:
        try:
            import google.generativeai as genai  # type: ignore[import-not-found]
        except ImportError:
            return (
                False,
                "The 'google-generativeai' package is required. "
                "Install: pip install google-generativeai",
            )
        try:
            genai.configure(api_key=self._api_key)
            models = genai.list_models()
            # Just check we can list models
            _ = list(models)[:1]
            return True, "Gemini API is reachable."
        except Exception as e:
            msg = str(e).lower()
            if "api_key" in msg or "unauthorized" in msg or "403" in msg:
                return False, "Gemini: invalid or missing API key."
            if "connection" in msg:
                return False, "Gemini: connection error."
            return False, f"Gemini error: {e}"

    def list_models(self) -> list[str]:
        try:
            genai = self._get_genai()
            return [
                m.name.replace("models/", "")
                for m in genai.list_models()
                if "generateContent" in (m.supported_generation_methods or [])
            ]
        except Exception:
            return []

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_tools=True,
            supports_streaming=True,
            supports_embeddings=False,
            supports_model_list=True,
        )
