"""Unit tests for providers/openai_compat.py — chat, stream, health."""

import json
from unittest.mock import MagicMock, patch


from ollamacode.providers.openai_compat import (
    PROVIDER_BASE_URLS,
    OpenAICompatProvider,
    _normalize_response,
    _parse_tool_calls,
    _to_openai_messages,
)


# ---------------------------------------------------------------------------
# _to_openai_messages
# ---------------------------------------------------------------------------


class TestToOpenAIMessages:
    def test_plain_messages_pass_through(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        out = _to_openai_messages(msgs)
        assert out == msgs

    def test_tool_call_arguments_dict_becomes_json_string(self):
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": "a.py"},
                        }
                    }
                ],
            }
        ]
        out = _to_openai_messages(msgs)
        tc = out[0]["tool_calls"][0]
        # arguments should now be a JSON string
        assert isinstance(tc["function"]["arguments"], str)
        parsed = json.loads(tc["function"]["arguments"])
        assert parsed == {"path": "a.py"}

    def test_tool_call_string_arguments_left_as_is(self):
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "test",
                            "arguments": '{"x":1}',
                        }
                    }
                ],
            }
        ]
        out = _to_openai_messages(msgs)
        assert out[0]["tool_calls"][0]["function"]["arguments"] == '{"x":1}'

    def test_type_field_added_to_tool_calls(self):
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "test", "arguments": {}}}],
            }
        ]
        out = _to_openai_messages(msgs)
        assert out[0]["tool_calls"][0]["type"] == "function"

    def test_existing_type_not_overwritten(self):
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "custom", "function": {"name": "test", "arguments": {}}}
                ],
            }
        ]
        out = _to_openai_messages(msgs)
        assert out[0]["tool_calls"][0]["type"] == "custom"


# ---------------------------------------------------------------------------
# _parse_tool_calls
# ---------------------------------------------------------------------------


class TestParseToolCalls:
    def test_parse_from_object_attributes(self):
        tc = MagicMock()
        fn = MagicMock()
        fn.name = "read_file"
        fn.arguments = '{"path":"x.py"}'
        tc.function = fn
        result = _parse_tool_calls([tc])
        assert len(result) == 1
        assert result[0]["function"]["name"] == "read_file"
        assert result[0]["function"]["arguments"] == {"path": "x.py"}

    def test_parse_from_dict(self):
        tc = {"function": {"name": "run", "arguments": '{"cmd":"ls"}'}}
        result = _parse_tool_calls([tc])
        assert result[0]["function"]["arguments"] == {"cmd": "ls"}

    def test_invalid_json_arguments_returns_empty_dict(self):
        tc = {"function": {"name": "test", "arguments": "not-json"}}
        result = _parse_tool_calls([tc])
        assert result[0]["function"]["arguments"] == {}

    def test_none_in_list_skipped(self):
        result = _parse_tool_calls(
            [None, {"function": {"name": "x", "arguments": "{}"}}]
        )
        assert len(result) == 1

    def test_no_function_key_skipped(self):
        result = _parse_tool_calls([{"not_function": {}}])
        assert len(result) == 0

    def test_none_arguments_treated_as_empty(self):
        tc = {"function": {"name": "test", "arguments": None}}
        result = _parse_tool_calls([tc])
        assert result[0]["function"]["arguments"] == {}

    def test_dict_arguments_passed_through(self):
        tc = {"function": {"name": "test", "arguments": {"a": 1}}}
        result = _parse_tool_calls([tc])
        assert result[0]["function"]["arguments"] == {"a": 1}


# ---------------------------------------------------------------------------
# _normalize_response
# ---------------------------------------------------------------------------


class TestNormalizeResponse:
    def test_openai_sdk_object_with_text(self):
        response = MagicMock()
        choice = MagicMock()
        msg = MagicMock()
        msg.content = "Hello world"
        msg.tool_calls = None
        choice.message = msg
        response.choices = [choice]

        result = _normalize_response(response)
        assert result["message"]["content"] == "Hello world"
        assert "tool_calls" not in result["message"]

    def test_openai_sdk_object_with_tool_calls(self):
        response = MagicMock()
        choice = MagicMock()
        msg = MagicMock()
        msg.content = ""
        tc = MagicMock()
        fn = MagicMock()
        fn.name = "read_file"
        fn.arguments = '{"path":"a.py"}'
        tc.function = fn
        msg.tool_calls = [tc]
        choice.message = msg
        response.choices = [choice]

        result = _normalize_response(response)
        assert len(result["message"]["tool_calls"]) == 1
        assert result["message"]["tool_calls"][0]["function"]["name"] == "read_file"

    def test_empty_choices(self):
        response = MagicMock()
        response.choices = []

        result = _normalize_response(response)
        assert result["message"]["content"] == ""

    def test_dict_response_fallback(self):
        response = {
            "choices": [{"message": {"content": "From dict", "tool_calls": []}}]
        }
        result = _normalize_response(response)
        assert result["message"]["content"] == "From dict"

    def test_plain_string_response(self):
        result = _normalize_response("raw text")
        assert result["message"]["content"] == "raw text"


# ---------------------------------------------------------------------------
# OpenAICompatProvider init and properties
# ---------------------------------------------------------------------------


class TestOpenAICompatProviderInit:
    def test_init_stores_fields(self):
        p = OpenAICompatProvider(
            provider_name="groq",
            api_key="gsk_test",
        )
        assert p._provider_name == "groq"
        assert p._api_key == "gsk_test"
        assert p._base_url == PROVIDER_BASE_URLS["groq"]

    def test_custom_base_url_overrides(self):
        p = OpenAICompatProvider(
            provider_name="custom",
            api_key="key",
            base_url="https://my-host/v1",
        )
        assert p._base_url == "https://my-host/v1"

    def test_unknown_provider_defaults_to_openai_url(self):
        p = OpenAICompatProvider(
            provider_name="unknown_provider",
            api_key="key",
        )
        assert p._base_url == "https://api.openai.com/v1"

    def test_name_returns_provider_name(self):
        p = OpenAICompatProvider(provider_name="groq", api_key="key")
        assert p.name == "groq"

    def test_capabilities(self):
        p = OpenAICompatProvider(provider_name="openai", api_key="key")
        caps = p.capabilities
        assert caps.supports_tools is True
        assert caps.supports_streaming is True
        assert caps.supports_embeddings is True
        assert caps.supports_model_list is True


# ---------------------------------------------------------------------------
# Known base URLs
# ---------------------------------------------------------------------------


class TestProviderBaseURLs:
    def test_known_providers_have_urls(self):
        expected = {
            "openai",
            "groq",
            "deepseek",
            "openrouter",
            "mistral",
            "xai",
            "together",
            "fireworks",
            "perplexity",
            "venice",
            "cohere",
            "cloudflare_ai",
        }
        assert expected.issubset(set(PROVIDER_BASE_URLS.keys()))

    def test_all_urls_are_https(self):
        for name, url in PROVIDER_BASE_URLS.items():
            assert url.startswith("https://"), (
                f"{name} URL does not start with https://"
            )


# ---------------------------------------------------------------------------
# Health check (mocked)
# ---------------------------------------------------------------------------


class TestOpenAICompatProviderHealthCheck:
    def test_health_check_missing_openai_package(self):
        """health_check returns (False, ...) when openai is not installed."""
        p = OpenAICompatProvider(provider_name="test", api_key="key")
        with patch.dict("sys.modules", {"openai": None}):
            # When import openai fails inside health_check, it catches ImportError
            with patch(
                "builtins.__import__",
                side_effect=lambda name, *a, **kw: (
                    (_ for _ in ()).throw(ImportError("no openai"))
                    if name == "openai"
                    else __import__(name, *a, **kw)
                ),
            ):
                ok, msg = p.health_check()
        assert ok is False
        assert "openai" in msg.lower()
