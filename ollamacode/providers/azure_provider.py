"""Azure OpenAI provider: uses the openai SDK with Azure-specific configuration.

Requires: pip install openai

Reuses OpenAICompatProvider as the base, adding Azure-specific
authentication (api_version, deployment_name, azure_endpoint).

Config (ollamacode.yaml):
  provider: azure
  base_url: https://my-resource.openai.azure.com
  api_key: ...
  azure_api_version: "2024-10-21"
  model: my-deployment-name  # Azure deployment name
"""

from __future__ import annotations

import logging
from typing import Any, Generator

from .base import BaseProvider, ProviderCapabilities

logger = logging.getLogger(__name__)

_IMPORT_ERROR_MSG = (
    "The 'openai' package is required for the Azure OpenAI provider. "
    "Install it with: pip install openai"
)

_DEFAULT_API_VERSION = "2024-10-21"


class AzureOpenAIProvider(BaseProvider):
    """Azure OpenAI Service provider.

    Uses the openai SDK's AzureOpenAI client which handles
    Azure-specific auth headers and URL routing.
    """

    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        api_version: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._azure_endpoint = azure_endpoint
        self._api_version = api_version or _DEFAULT_API_VERSION

    def _async_client(self) -> Any:
        try:
            import openai  # type: ignore[import-not-found]

            return openai.AsyncAzureOpenAI(
                api_key=self._api_key,
                azure_endpoint=self._azure_endpoint,
                api_version=self._api_version,
            )
        except ImportError as e:
            raise RuntimeError(_IMPORT_ERROR_MSG) from e

    def _sync_client(self) -> Any:
        try:
            import openai  # type: ignore[import-not-found]

            return openai.AzureOpenAI(
                api_key=self._api_key,
                azure_endpoint=self._azure_endpoint,
                api_version=self._api_version,
            )
        except ImportError as e:
            raise RuntimeError(_IMPORT_ERROR_MSG) from e

    async def chat_async(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        from .openai_compat import _to_openai_messages, _normalize_response

        client = self._async_client()
        oai_messages = _to_openai_messages(messages)
        kwargs: dict[str, Any] = {"model": model, "messages": oai_messages}
        if tools:
            kwargs["tools"] = tools
        try:
            response = await client.chat.completions.create(**kwargs)
            return _normalize_response(response)
        except Exception as e:
            raise RuntimeError(f"Azure OpenAI API error: {e}") from e
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
        from .openai_compat import _to_openai_messages

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
            raise RuntimeError(f"Azure OpenAI stream error: {e}") from e

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
            client = openai.AzureOpenAI(
                api_key=self._api_key,
                azure_endpoint=self._azure_endpoint,
                api_version=self._api_version,
            )
            client.models.list()
            return True, "Azure OpenAI is reachable."
        except Exception as e:
            msg = str(e).lower()
            if "401" in msg or "unauthorized" in msg or "api key" in msg:
                return False, "Azure OpenAI: invalid or missing API key."
            if "404" in msg or "not found" in msg:
                return (
                    True,
                    "Azure OpenAI: endpoint reachable (models list not supported).",
                )
            if "connection" in msg:
                return False, "Azure OpenAI: connection error."
            return False, f"Azure OpenAI error: {e}"
        finally:
            if client is not None and hasattr(client, "close"):
                try:
                    client.close()
                except Exception:
                    pass

    def list_models(self) -> list[str]:
        try:
            client = self._sync_client()
            resp = client.models.list()
            return [m.id for m in resp.data] if hasattr(resp, "data") else []
        except Exception:
            return []

    @property
    def name(self) -> str:
        return "azure"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_tools=True,
            supports_streaming=True,
            supports_embeddings=True,
            supports_model_list=True,
        )
