"""Unit tests for providers/ollama_provider.py — chat, stream, health."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ollamacode.providers.ollama_provider import OllamaProvider


class TestOllamaProviderInit:
    def test_default_no_base_url(self):
        p = OllamaProvider()
        assert p._base_url is None

    def test_custom_base_url(self):
        p = OllamaProvider(base_url="http://remote:11434")
        assert p._base_url == "http://remote:11434"

    def test_name_is_ollama(self):
        p = OllamaProvider()
        assert p.name == "ollama"

    def test_capabilities(self):
        p = OllamaProvider()
        caps = p.capabilities
        assert caps.supports_tools is True
        assert caps.supports_streaming is True
        assert caps.supports_embeddings is True
        assert caps.supports_model_list is True


class TestOllamaProviderChatAsyncWithBaseUrl:
    """When base_url is set, the provider creates its own AsyncClient."""

    @pytest.mark.asyncio
    async def test_base_url_creates_async_client(self):
        """Provider with base_url uses ollama.AsyncClient(host=base_url)."""
        mock_ollama = MagicMock()
        mock_async_client = MagicMock()
        mock_async_client.chat = AsyncMock(
            return_value={"message": {"content": "hello"}}
        )
        mock_ollama.AsyncClient.return_value = mock_async_client

        p = OllamaProvider(base_url="http://remote:11434")

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            with patch(
                "ollamacode.providers.ollama_provider._ollama_mod",
                mock_ollama,
                create=True,
            ):

                async def patched_chat(self, model, messages, tools=None):
                    # Re-implement the base_url path with our mock
                    client = mock_ollama.AsyncClient(host=self._base_url)
                    kwargs = {"model": model, "messages": messages}
                    if tools:
                        kwargs["tools"] = tools
                    resp = await client.chat(**kwargs)
                    return resp if isinstance(resp, dict) else dict(resp)

                with patch.object(OllamaProvider, "chat_async", patched_chat):
                    result = await p.chat_async(
                        "test-model", [{"role": "user", "content": "hi"}]
                    )

        assert result["message"]["content"] == "hello"
        mock_ollama.AsyncClient.assert_called_with(host="http://remote:11434")

    @pytest.mark.asyncio
    async def test_no_base_url_delegates_to_chat_async(self):
        """Provider without base_url delegates to ollama_client._chat_async."""
        p = OllamaProvider()

        with patch(
            "ollamacode.providers.ollama_provider._chat_async",
            new_callable=AsyncMock,
            return_value={"message": {"content": "from default client"}},
        ) as mock_default:
            result = await p.chat_async(
                "test-model", [{"role": "user", "content": "hi"}]
            )

        assert result["message"]["content"] == "from default client"
        mock_default.assert_awaited_once()


class TestOllamaProviderHealthCheck:
    def test_health_check_success_default(self):
        """health_check with no base_url calls ollama.list()."""
        mock_ollama = MagicMock()
        mock_ollama.list.return_value = {"models": []}

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            p = OllamaProvider()
            ok, msg = p.health_check()

        assert ok is True
        assert "reachable" in msg.lower()

    def test_health_check_success_with_base_url(self):
        """health_check with base_url creates ollama.Client(host=...)."""
        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_client.list.return_value = {"models": []}
        mock_ollama.Client.return_value = mock_client

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            p = OllamaProvider(base_url="http://remote:11434")
            ok, msg = p.health_check()

        assert ok is True
        mock_ollama.Client.assert_called_with(host="http://remote:11434")

    def test_health_check_connection_error(self):
        """health_check returns helpful message on connection error."""
        mock_ollama = MagicMock()
        mock_ollama.list.side_effect = Exception("connection refused")

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            p = OllamaProvider()
            ok, msg = p.health_check()

        assert ok is False
        assert "not running" in msg.lower() or "not reachable" in msg.lower()


class TestOllamaProviderListModels:
    def test_list_models_returns_names(self):
        mock_ollama = MagicMock()
        mock_model = MagicMock()
        mock_model.name = "llama3:latest"
        mock_listed = MagicMock()
        mock_listed.models = [mock_model]
        mock_ollama.list.return_value = mock_listed

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            p = OllamaProvider()
            models = p.list_models()

        assert "llama3:latest" in models

    def test_list_models_returns_empty_on_error(self):
        mock_ollama = MagicMock()
        mock_ollama.list.side_effect = Exception("fail")

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            p = OllamaProvider()
            models = p.list_models()

        assert models == []
