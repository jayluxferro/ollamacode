"""Unit tests for providers/base.py — interface contract."""

import pytest

from ollamacode.providers.base import BaseProvider, ProviderCapabilities


class TestProviderCapabilities:
    def test_defaults(self):
        caps = ProviderCapabilities()
        assert caps.supports_tools is True
        assert caps.supports_streaming is True
        assert caps.supports_embeddings is False
        assert caps.supports_model_list is False

    def test_custom_values(self):
        caps = ProviderCapabilities(
            supports_tools=False,
            supports_streaming=False,
            supports_embeddings=True,
            supports_model_list=True,
        )
        assert caps.supports_tools is False
        assert caps.supports_embeddings is True

    def test_frozen(self):
        caps = ProviderCapabilities()
        with pytest.raises(AttributeError):
            caps.supports_tools = False  # type: ignore[misc]


class TestBaseProviderAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseProvider()  # type: ignore[abstract]

    def test_subclass_must_implement_chat_async(self):
        class Incomplete(BaseProvider):
            def chat_stream_sync(self, model, messages):
                yield ("",)

            def health_check(self):
                return (True, "ok")

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_subclass_must_implement_chat_stream_sync(self):
        class Incomplete(BaseProvider):
            async def chat_async(self, model, messages, tools=None):
                return {}

            def health_check(self):
                return (True, "ok")

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_subclass_must_implement_health_check(self):
        class Incomplete(BaseProvider):
            async def chat_async(self, model, messages, tools=None):
                return {}

            def chat_stream_sync(self, model, messages):
                yield ("",)

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_complete_subclass_instantiates(self):
        class Complete(BaseProvider):
            async def chat_async(self, model, messages, tools=None):
                return {"message": {"content": "ok"}}

            def chat_stream_sync(self, model, messages):
                yield ("ok",)

            def health_check(self):
                return (True, "ok")

        p = Complete()
        assert isinstance(p, BaseProvider)

    def test_default_list_models_returns_empty(self):
        class Complete(BaseProvider):
            async def chat_async(self, model, messages, tools=None):
                return {}

            def chat_stream_sync(self, model, messages):
                yield ("",)

            def health_check(self):
                return (True, "ok")

        p = Complete()
        assert p.list_models() == []

    def test_default_name_is_class_name(self):
        class MyCustomProvider(BaseProvider):
            async def chat_async(self, model, messages, tools=None):
                return {}

            def chat_stream_sync(self, model, messages):
                yield ("",)

            def health_check(self):
                return (True, "ok")

        p = MyCustomProvider()
        assert p.name == "MyCustomProvider"

    def test_default_capabilities(self):
        class Complete(BaseProvider):
            async def chat_async(self, model, messages, tools=None):
                return {}

            def chat_stream_sync(self, model, messages):
                yield ("",)

            def health_check(self):
                return (True, "ok")

        p = Complete()
        caps = p.capabilities
        assert isinstance(caps, ProviderCapabilities)
        assert caps.supports_tools is True

    def test_set_tool_executor_is_noop(self):
        class Complete(BaseProvider):
            async def chat_async(self, model, messages, tools=None):
                return {}

            def chat_stream_sync(self, model, messages):
                yield ("",)

            def health_check(self):
                return (True, "ok")

        p = Complete()
        # Should not raise
        p.set_tool_executor(lambda: None)
