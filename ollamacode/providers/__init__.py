"""Provider registry and factory for OllamaCode AI backends.

Usage in config (ollamacode.yaml):

  # Local Ollama (default)
  provider: ollama
  model: qwen2.5-coder:7b

  # Groq
  provider: groq
  model: llama-3.3-70b-versatile
  api_key: gsk_...   # or set GROQ_API_KEY env var

  # OpenAI
  provider: openai
  model: gpt-4o
  api_key: sk-...    # or set OPENAI_API_KEY env var

  # Anthropic
  provider: anthropic
  model: claude-sonnet-4-6
  api_key: sk-ant-...  # or set ANTHROPIC_API_KEY env var

  # Any OpenAI-compatible API
  provider: custom
  base_url: https://my-llm-host/v1
  model: my-model
  api_key: ...       # or set OLLAMACODE_API_KEY env var

Supported named providers (base_url auto-configured):
  ollama, openai, groq, deepseek, openrouter, mistral, xai,
  together, fireworks, perplexity, venice, cohere, cloudflare_ai, anthropic,
  apple_fm, gemini, bedrock, azure
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .base import BaseProvider
from .ollama_provider import OllamaProvider
from .openai_compat import OpenAICompatProvider, PROVIDER_BASE_URLS
from .anthropic_provider import AnthropicProvider
from .apple_fm_provider import AppleFMProvider

logger = logging.getLogger(__name__)

# Providers that use the OpenAI-compatible REST protocol
_OPENAI_COMPAT_PROVIDERS = set(PROVIDER_BASE_URLS.keys())

# Per-provider environment variables for API keys
_PROVIDER_KEY_ENVS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "xai": "XAI_API_KEY",
    "together": "TOGETHER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
    "venice": "VENICE_API_KEY",
    "cohere": "COHERE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
}


def _resolve_api_key(provider_name: str, config: dict[str, Any]) -> str:
    """Resolve API key: config (with secret: unwrapping) > OLLAMACODE_API_KEY > provider-specific env var."""
    raw = (
        config.get("api_key")
        or os.environ.get("OLLAMACODE_API_KEY")
        or (_provider_env_key(provider_name))
        or ""
    )
    if raw and isinstance(raw, str) and raw.startswith("secret:"):
        try:
            from ..secrets import resolve_secret

            resolved = resolve_secret(raw)
            return resolved or ""
        except Exception as exc:
            logger.warning(
                "Failed to resolve secret '%s': %s. "
                "Falling back to raw value; auth errors may follow.",
                raw,
                exc,
            )
    return raw


def _provider_env_key(provider: str) -> str | None:
    env_var = _PROVIDER_KEY_ENVS.get(provider)
    return os.environ.get(env_var) if env_var else None


def get_provider(config: dict[str, Any]) -> BaseProvider:
    """Build a provider from a resolved config dict.

    Relevant config keys:
      provider  — provider name (default: "ollama")
      base_url  — override base URL (also: OLLAMACODE_BASE_URL env var)
      api_key   — API key (also: OLLAMACODE_API_KEY, or provider-specific env vars)
    """
    provider_name = (config.get("provider") or "ollama").lower().strip()
    base_url: str | None = (
        config.get("base_url") or os.environ.get("OLLAMACODE_BASE_URL") or None
    )

    if provider_name == "ollama":
        return OllamaProvider(base_url=base_url)

    api_key = _resolve_api_key(provider_name, config)

    if provider_name == "anthropic":
        return AnthropicProvider(api_key=api_key, base_url=base_url)
    if provider_name == "apple_fm":
        return AppleFMProvider()
    if provider_name == "gemini":
        from .gemini_provider import GeminiProvider

        return GeminiProvider(api_key=api_key)
    if provider_name == "bedrock":
        from .bedrock_provider import BedrockProvider

        return BedrockProvider(
            region=config.get("aws_region") or os.environ.get("AWS_REGION"),
            profile=config.get("aws_profile") or os.environ.get("AWS_PROFILE"),
        )
    if provider_name == "azure":
        from .azure_provider import AzureOpenAIProvider

        azure_endpoint = base_url or config.get("azure_endpoint") or ""
        api_version = config.get("azure_api_version")
        return AzureOpenAIProvider(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )

    # Validate that the provider is known or has a custom base_url
    _ALL_KNOWN = _OPENAI_COMPAT_PROVIDERS | {
        "ollama",
        "anthropic",
        "apple_fm",
        "gemini",
        "bedrock",
        "azure",
        "custom",
    }
    if provider_name not in _ALL_KNOWN and not base_url:
        logger.warning(
            "Unknown provider '%s'. Known providers: %s. "
            "Treating as OpenAI-compatible; set base_url if needed.",
            provider_name,
            ", ".join(sorted(_ALL_KNOWN)),
        )

    # Everything else speaks OpenAI-compatible REST
    return OpenAICompatProvider(
        provider_name=provider_name,
        api_key=api_key,
        base_url=base_url,
    )


__all__ = [
    "BaseProvider",
    "OllamaProvider",
    "OpenAICompatProvider",
    "AnthropicProvider",
    "AppleFMProvider",
    "PROVIDER_BASE_URLS",
    "get_provider",
    # Lazy-loaded providers (imported on demand in get_provider):
    # "GeminiProvider",
    # "BedrockProvider",
    # "AzureOpenAIProvider",
]
