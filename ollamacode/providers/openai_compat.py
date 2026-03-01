"""OpenAI-compatible provider.

Covers: OpenAI, Groq, DeepSeek, Mistral, xAI/Grok, OpenRouter, Venice,
Together AI, Fireworks, Perplexity, Cohere, Cloudflare AI, and any custom base_url.

Requires: pip install openai
"""

from __future__ import annotations

import json
import logging
from typing import Any, Generator

from .base import BaseProvider, ProviderCapabilities

logger = logging.getLogger(__name__)

# Known base URLs for named providers. Override via config base_url or OLLAMACODE_BASE_URL.
PROVIDER_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "mistral": "https://api.mistral.ai/v1",
    "xai": "https://api.x.ai/v1",
    "together": "https://api.together.xyz/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "perplexity": "https://api.perplexity.ai",
    "venice": "https://api.venice.ai/api/v1",
    "cohere": "https://api.cohere.com/compatibility/v1",
    "cloudflare_ai": "https://api.cloudflare.com/client/v4/ai/v1",
}

_IMPORT_ERROR_MSG = (
    "The 'openai' package is required for non-Ollama providers. "
    "Install it with: pip install openai"
)


def _to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Ollama-format messages to OpenAI format.

    Key difference: Ollama tool_calls.function.arguments is a dict;
    OpenAI requires it as a JSON string.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        msg = dict(m)
        tcs = msg.get("tool_calls")
        if tcs:
            new_tcs = []
            for tc in tcs:
                tc = dict(tc)
                fn = dict(tc.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, dict):
                    fn["arguments"] = json.dumps(args)
                tc["function"] = fn
                if "type" not in tc:
                    tc["type"] = "function"
                new_tcs.append(tc)
            msg["tool_calls"] = new_tcs
        out.append(msg)
    return out


def _normalize_response(response: Any) -> dict[str, Any]:
    """Normalize OpenAI-style response to Ollama-shaped dict:
    {"message": {"content": str, "tool_calls": [...]}}
    """
    # openai SDK ChatCompletion object
    if hasattr(response, "choices"):
        choices = response.choices or []
        if not choices:
            return {"message": {"content": ""}}
        choice = choices[0]
        msg = choice.message if hasattr(choice, "message") else {}
        content = getattr(msg, "content", "") or ""
        raw_tcs = getattr(msg, "tool_calls", None) or []
        tool_calls = _parse_tool_calls(raw_tcs)
        out_msg: dict[str, Any] = {"content": content}
        if tool_calls:
            out_msg["tool_calls"] = tool_calls
        return {"message": out_msg}

    # Raw dict response (fallback)
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if not choices:
            return {"message": {"content": ""}}
        msg = (choices[0] or {}).get("message") or {}
        content = msg.get("content") or ""
        raw_tcs = msg.get("tool_calls") or []
        tool_calls = _parse_tool_calls(raw_tcs)
        out_msg = {"content": content}
        if tool_calls:
            out_msg["tool_calls"] = tool_calls
        return {"message": out_msg}

    return {"message": {"content": str(response)}}


def _parse_tool_calls(raw_tcs: list[Any]) -> list[dict[str, Any]]:
    """Convert OpenAI-format tool calls to Ollama format (arguments as dict)."""
    tool_calls: list[dict[str, Any]] = []
    for tc in raw_tcs:
        if tc is None:
            continue
        fn = getattr(tc, "function", None) or (
            tc.get("function") if isinstance(tc, dict) else None
        )
        if fn is None:
            continue
        if isinstance(fn, dict):
            name = fn.get("name") or ""
            args_raw = fn.get("arguments", "{}")
        else:
            name = getattr(fn, "name", None) or ""
            args_raw = getattr(fn, "arguments", "{}")
        if args_raw is None:
            args_raw = "{}"
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "Failed to parse tool arguments for %s: %s",
                name,
                args_raw[:200] if isinstance(args_raw, str) else type(args_raw),
            )
            args = {}
        tool_calls.append({"function": {"name": name or "", "arguments": args}})
    return tool_calls


class OpenAICompatProvider(BaseProvider):
    """Any OpenAI-compatible REST API."""

    def __init__(
        self,
        provider_name: str,
        api_key: str,
        base_url: str | None = None,
    ) -> None:
        self._provider_name = provider_name
        self._api_key = api_key
        self._base_url = base_url or PROVIDER_BASE_URLS.get(
            provider_name, "https://api.openai.com/v1"
        )

    # ------------------------------------------------------------------ clients

    def _async_client(self) -> Any:
        try:
            import openai  # type: ignore[import-not-found]

            return openai.AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        except ImportError as e:
            raise RuntimeError(_IMPORT_ERROR_MSG) from e

    def _sync_client(self) -> Any:
        try:
            import openai  # type: ignore[import-not-found]

            return openai.OpenAI(api_key=self._api_key, base_url=self._base_url)
        except ImportError as e:
            raise RuntimeError(_IMPORT_ERROR_MSG) from e

    # ------------------------------------------------------------------ BaseProvider

    async def chat_async(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        client = self._async_client()
        oai_messages = _to_openai_messages(messages)
        kwargs: dict[str, Any] = {"model": model, "messages": oai_messages}
        if tools:
            kwargs["tools"] = tools
        try:
            response = await client.chat.completions.create(**kwargs)
            return _normalize_response(response)
        except Exception as e:
            raise RuntimeError(f"{self._provider_name} API error: {e}") from e
        finally:
            try:
                if hasattr(client, "close"):
                    await client.close()
            except Exception:
                pass

    def chat_stream_sync(
        self,
        model: str,
        messages: list[dict[str, Any]],
    ) -> Generator[tuple[str], None, None]:
        client = self._sync_client()
        oai_messages = _to_openai_messages(messages)
        try:
            with client.chat.completions.create(
                model=model, messages=oai_messages, stream=True
            ) as stream:
                for chunk in stream:
                    if hasattr(chunk, "choices") and chunk.choices:
                        delta = chunk.choices[0].delta
                        content = getattr(delta, "content", "") or ""
                        if content:
                            yield (content,)
        except Exception as e:
            raise RuntimeError(f"{self._provider_name} stream error: {e}") from e

    def health_check(self) -> tuple[bool, str]:
        try:
            import openai  # type: ignore[import-not-found]
        except ImportError:
            return (
                False,
                "The 'openai' package is required. Install: pip install openai",
            )
        client = None
        try:
            client = openai.OpenAI(api_key=self._api_key, base_url=self._base_url)
            client.models.list()
            return True, f"{self._provider_name} is reachable."
        except Exception as e:
            msg = str(e).lower()
            if (
                "401" in msg
                or "unauthorized" in msg
                or "authentication" in msg
                or "api key" in msg
            ):
                return False, f"{self._provider_name}: invalid or missing API key."
            if "connection" in msg or "refused" in msg:
                return (
                    False,
                    f"{self._provider_name}: connection refused. Check base_url.",
                )
            # Many providers don't support GET /models — treat as reachable
            if "404" in msg or "not found" in msg or "method not allowed" in msg:
                return (
                    True,
                    f"{self._provider_name}: endpoint reachable (models list not supported).",
                )
            return False, f"{self._provider_name} error: {e}"
        finally:
            if client is not None and hasattr(client, "close"):
                try:
                    client.close()
                except Exception:
                    pass

    def list_models(self) -> list[str]:
        try:
            import openai  # type: ignore[import-not-found]

            client = openai.OpenAI(api_key=self._api_key, base_url=self._base_url)
            resp = client.models.list()
            return [m.id for m in resp.data] if hasattr(resp, "data") else []
        except Exception:
            return []

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_tools=True,
            supports_streaming=True,
            supports_embeddings=True,
            supports_model_list=True,
        )

    @property
    def name(self) -> str:
        return self._provider_name
