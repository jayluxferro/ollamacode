"""AWS Bedrock provider: uses boto3 bedrock-runtime.

Requires: pip install boto3

Supports Claude models on AWS Bedrock via the Converse API.
AWS credentials are resolved via standard boto3 chain
(env vars, ~/.aws/credentials, IAM role, etc.).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Generator

from .base import BaseProvider, ProviderCapabilities

logger = logging.getLogger(__name__)

_IMPORT_ERROR_MSG = (
    "The 'boto3' package is required for the Bedrock provider. "
    "Install it with: pip install boto3"
)


def _to_bedrock_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Convert OllamaCode messages to Bedrock Converse format.

    Returns (system_messages, conversation_messages).
    """
    system: list[dict[str, str]] = []
    conversation: list[dict[str, Any]] = []

    for m in messages:
        role = (m.get("role") or "").strip().lower()
        content = str(m.get("content") or "")

        if role == "system":
            system.append({"text": content})
            continue

        if role == "tool":
            tool_use_id = m.get("tool_call_id") or m.get("tool_name") or "unknown"
            conversation.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": tool_use_id,
                                "content": [{"text": content}],
                            }
                        }
                    ],
                }
            )
            continue

        if role == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs:
                blocks: list[dict[str, Any]] = []
                if content:
                    blocks.append({"text": content})
                for tc in tcs:
                    fn = tc.get("function") or {} if isinstance(tc, dict) else {}
                    fn_name = fn.get("name", "") if isinstance(fn, dict) else ""
                    args = fn.get("arguments", {}) if isinstance(fn, dict) else {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            args = {}
                    tc_id = (
                        tc.get("id") if isinstance(tc, dict) else None
                    ) or f"tooluse_{hash(fn_name) & 0xFFFF:04x}"
                    blocks.append(
                        {
                            "toolUse": {
                                "toolUseId": tc_id,
                                "name": fn_name,
                                "input": args,
                            }
                        }
                    )
                conversation.append({"role": "assistant", "content": blocks})
                continue

        bedrock_role = "assistant" if role == "assistant" else "user"
        conversation.append(
            {
                "role": bedrock_role,
                "content": [{"text": content}],
            }
        )

    return system, conversation


def _to_bedrock_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-style tool dicts to Bedrock toolSpec format."""
    specs: list[dict[str, Any]] = []
    for t in tools:
        fn = t.get("function") or {}
        specs.append(
            {
                "toolSpec": {
                    "name": fn.get("name") or "",
                    "description": fn.get("description") or "",
                    "inputSchema": {
                        "json": fn.get("parameters")
                        or {"type": "object", "properties": {}},
                    },
                }
            }
        )
    return specs


def _normalize_response(response: dict[str, Any]) -> dict[str, Any]:
    """Convert Bedrock Converse response to Ollama-shaped dict."""
    output = response.get("output") or {}
    message = output.get("message") or {}
    content_blocks = message.get("content") or []

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in content_blocks:
        if "text" in block:
            text_parts.append(block["text"])
        elif "toolUse" in block:
            tu = block["toolUse"]
            tool_calls.append(
                {
                    "id": tu.get("toolUseId", ""),
                    "function": {
                        "name": tu.get("name", ""),
                        "arguments": tu.get("input", {}),
                    },
                }
            )

    content = "".join(text_parts)
    out_msg: dict[str, Any] = {"content": content}
    if tool_calls:
        out_msg["tool_calls"] = tool_calls
    return {"message": out_msg}


class BedrockProvider(BaseProvider):
    """AWS Bedrock provider using the Converse API."""

    def __init__(
        self,
        region: str | None = None,
        profile: str | None = None,
    ) -> None:
        self._region = region
        self._profile = profile

    def _get_client(self) -> Any:
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(_IMPORT_ERROR_MSG) from e

        kwargs: dict[str, Any] = {"service_name": "bedrock-runtime"}
        if self._region:
            kwargs["region_name"] = self._region
        session_kwargs: dict[str, Any] = {}
        if self._profile:
            session_kwargs["profile_name"] = self._profile
        session = boto3.Session(**session_kwargs)
        return session.client(**kwargs)

    async def chat_async(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        import asyncio

        client = self._get_client()
        system, conversation = _to_bedrock_messages(messages)

        kwargs: dict[str, Any] = {
            "modelId": model,
            "messages": conversation,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["toolConfig"] = {"tools": _to_bedrock_tools(tools)}

        try:
            # boto3 is synchronous; run in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: client.converse(**kwargs)
            )
            return _normalize_response(response)
        except Exception as e:
            raise RuntimeError(f"Bedrock API error: {e}") from e

    def chat_stream_sync(
        self,
        model: str,
        messages: list[dict[str, Any]],
    ) -> Generator[tuple[str], None, None]:
        client = self._get_client()
        system, conversation = _to_bedrock_messages(messages)

        kwargs: dict[str, Any] = {
            "modelId": model,
            "messages": conversation,
        }
        if system:
            kwargs["system"] = system

        try:
            response = client.converse_stream(**kwargs)
            stream = response.get("stream") or []
            for event in stream:
                delta = (event.get("contentBlockDelta") or {}).get("delta") or {}
                text = delta.get("text", "")
                if text:
                    yield (text,)
        except Exception as e:
            raise RuntimeError(f"Bedrock stream error: {e}") from e

    def health_check(self) -> tuple[bool, str]:
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError:
            return (
                False,
                "The 'boto3' package is required. Install: pip install boto3",
            )
        try:
            client = self._get_client()
            # List foundation models as a connectivity check
            client.list_foundation_models = None  # Not on runtime client
            # Use bedrock (not bedrock-runtime) for listing
            kwargs: dict[str, Any] = {"service_name": "bedrock"}
            if self._region:
                kwargs["region_name"] = self._region
            session_kwargs: dict[str, Any] = {}
            if self._profile:
                session_kwargs["profile_name"] = self._profile
            session = boto3.Session(**session_kwargs)
            mgmt_client = session.client(**kwargs)
            mgmt_client.list_foundation_models(maxResults=1)
            return True, "AWS Bedrock is reachable."
        except Exception as e:
            msg = str(e).lower()
            if "credential" in msg or "token" in msg or "403" in msg:
                return False, "Bedrock: invalid or missing AWS credentials."
            if "region" in msg:
                return False, "Bedrock: invalid region or region not configured."
            return False, f"Bedrock error: {e}"

    @property
    def name(self) -> str:
        return "bedrock"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_tools=True,
            supports_streaming=True,
            supports_embeddings=False,
            supports_model_list=False,
        )
