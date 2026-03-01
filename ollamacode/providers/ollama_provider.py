"""OllamaProvider: wraps the existing ollama_client.py calls."""

from __future__ import annotations

from typing import Any, Generator

from .base import BaseProvider, ProviderCapabilities
from ..ollama_client import (
    chat_async as _chat_async,
    chat_stream_sync as _chat_stream_sync,
    wrap_ollama_template_error as _wrap_template_error,
)


class OllamaProvider(BaseProvider):
    """Local Ollama backend (default)."""

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = base_url

    async def chat_async(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        try:
            if self._base_url:
                import ollama as _ollama_mod  # type: ignore[import-untyped]

                client = _ollama_mod.AsyncClient(host=self._base_url)
                kwargs: dict[str, Any] = {"model": model, "messages": messages}
                if tools:
                    kwargs["tools"] = tools
                resp = await client.chat(**kwargs)
                return resp if isinstance(resp, dict) else dict(resp)
            return await _chat_async(model, messages, tools)
        except Exception as e:
            raise _wrap_template_error(e)

    def chat_stream_sync(
        self,
        model: str,
        messages: list[dict[str, Any]],
    ) -> Generator[tuple[str], None, None]:
        yield from _chat_stream_sync(model, messages)

    def health_check(self) -> tuple[bool, str]:
        try:
            import ollama

            if self._base_url:
                client = ollama.Client(host=self._base_url)
                client.list()
            else:
                ollama.list()
            return True, "Ollama is reachable."
        except Exception as e:
            msg = str(e).lower()
            if "connection" in msg or "refused" in msg or "connect" in msg:
                return (
                    False,
                    "Ollama is not running or not reachable. Start it with: ollama serve",
                )
            return False, f"Ollama error: {e}"

    def list_models(self) -> list[str]:
        try:
            import ollama

            if self._base_url:
                client = ollama.Client(host=self._base_url)
                listed = client.list()
            else:
                listed = ollama.list()
            models_list = (
                getattr(listed, "models", None)
                or (listed.get("models") if isinstance(listed, dict) else None)
                or []
            )
            names: list[str] = []
            for m in models_list:
                n = (
                    getattr(m, "name", None)
                    or (m.get("name") if isinstance(m, dict) else None)
                    or getattr(m, "model", None)
                    or (m.get("model") if isinstance(m, dict) else None)
                )
                if n:
                    names.append(str(n))
            return names
        except Exception:
            return []

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_tools=True,
            supports_streaming=True,
            supports_embeddings=True,
            supports_model_list=True,
        )
