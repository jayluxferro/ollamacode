"""Base provider interface for all AI backends."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Generator


@dataclass(frozen=True)
class ProviderCapabilities:
    supports_tools: bool = True
    supports_streaming: bool = True
    supports_embeddings: bool = False
    supports_model_list: bool = False


class BaseProvider(abc.ABC):
    """Abstract base for all AI providers.

    All providers must implement:
      - chat_async: non-streaming chat with optional tool calling
      - chat_stream_sync: streaming generator (no tool calling; used for final response)
      - health_check: verify connectivity and auth
    """

    @abc.abstractmethod
    async def chat_async(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Send a chat request. Returns Ollama-shaped response:
        {"message": {"content": str, "tool_calls": [{"function": {"name": str, "arguments": dict}}]}}
        """

    @abc.abstractmethod
    def chat_stream_sync(
        self,
        model: str,
        messages: list[dict[str, Any]],
    ) -> Generator[tuple[str], None, None]:
        """Stream a chat response (no tools). Yields (content_fragment,) per chunk."""

    @abc.abstractmethod
    def health_check(self) -> tuple[bool, str]:
        """Returns (ok, message)."""

    def list_models(self) -> list[str]:
        """Return available model names. Returns [] if not supported."""
        return []

    @property
    def name(self) -> str:
        """Short display name for this provider."""
        return self.__class__.__name__

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Feature flags for the provider (tools/streaming/embeddings/model listing)."""
        return ProviderCapabilities()
