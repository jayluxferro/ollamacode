"""Unit tests for providers/anthropic_provider.py — chat, stream, health."""

from unittest.mock import MagicMock, patch


from ollamacode.providers.anthropic_provider import (
    AnthropicProvider,
    _normalize_response,
    _split_system,
    _to_anthropic_messages,
    _to_anthropic_tools,
)


# ---------------------------------------------------------------------------
# _split_system
# ---------------------------------------------------------------------------


class TestSplitSystem:
    def test_extracts_system_message(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        system, rest = _split_system(messages)
        assert system == "You are helpful."
        assert len(rest) == 1
        assert rest[0]["role"] == "user"

    def test_no_system_message(self):
        messages = [{"role": "user", "content": "Hi"}]
        system, rest = _split_system(messages)
        assert system is None
        assert len(rest) == 1

    def test_multiple_system_messages_last_wins(self):
        messages = [
            {"role": "system", "content": "first"},
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "second"},
        ]
        system, rest = _split_system(messages)
        assert system == "second"
        assert len(rest) == 1


# ---------------------------------------------------------------------------
# _to_anthropic_messages
# ---------------------------------------------------------------------------


class TestToAnthropicMessages:
    def test_user_and_assistant_pass_through(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        out = _to_anthropic_messages(messages)
        assert len(out) == 2
        assert out[0] == {"role": "user", "content": "Hello"}
        assert out[1] == {"role": "assistant", "content": "Hi"}

    def test_tool_message_becomes_user_tool_result(self):
        messages = [
            {"role": "tool", "content": "file contents", "tool_call_id": "tc_123"},
        ]
        out = _to_anthropic_messages(messages)
        assert len(out) == 1
        assert out[0]["role"] == "user"
        blocks = out[0]["content"]
        assert blocks[0]["type"] == "tool_result"
        assert blocks[0]["tool_use_id"] == "tc_123"
        assert blocks[0]["content"] == "file contents"

    def test_assistant_with_tool_calls_converted(self):
        messages = [
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": "a.py"},
                        },
                    }
                ],
            }
        ]
        out = _to_anthropic_messages(messages)
        assert len(out) == 1
        assert out[0]["role"] == "assistant"
        blocks = out[0]["content"]
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "Let me check."
        assert blocks[1]["type"] == "tool_use"
        assert blocks[1]["id"] == "tc_1"
        assert blocks[1]["name"] == "read_file"
        assert blocks[1]["input"] == {"path": "a.py"}

    def test_tool_call_with_string_arguments_parsed(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {"name": "run", "arguments": '{"cmd":"ls"}'},
                    }
                ],
            }
        ]
        out = _to_anthropic_messages(messages)
        blocks = out[0]["content"]
        assert blocks[0]["type"] == "tool_use"
        assert blocks[0]["input"] == {"cmd": "ls"}

    def test_tool_call_generates_uuid_id_when_missing(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "test_tool", "arguments": {}}},
                ],
            }
        ]
        out = _to_anthropic_messages(messages)
        block = out[0]["content"][0]
        assert block["type"] == "tool_use"
        assert block["id"].startswith("toolu_")

    def test_system_messages_skipped(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        out = _to_anthropic_messages(messages)
        assert len(out) == 1
        assert out[0]["role"] == "user"


# ---------------------------------------------------------------------------
# _to_anthropic_tools
# ---------------------------------------------------------------------------


class TestToAnthropicTools:
    def test_basic_conversion(self):
        tools = [
            {
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                }
            }
        ]
        out = _to_anthropic_tools(tools)
        assert len(out) == 1
        assert out[0]["name"] == "read_file"
        assert out[0]["description"] == "Read a file"
        assert "properties" in out[0]["input_schema"]

    def test_missing_fields_default_empty(self):
        tools = [{"function": {}}]
        out = _to_anthropic_tools(tools)
        assert out[0]["name"] == ""
        assert out[0]["description"] == ""
        assert out[0]["input_schema"] == {"type": "object", "properties": {}}


# ---------------------------------------------------------------------------
# _normalize_response
# ---------------------------------------------------------------------------


class TestNormalizeResponse:
    def test_text_only(self):
        response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Hello world"
        response.content = [text_block]

        result = _normalize_response(response)
        assert result["message"]["content"] == "Hello world"
        assert "tool_calls" not in result["message"]

    def test_tool_use_block(self):
        response = MagicMock()
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "toolu_abc123"
        tool_block.name = "read_file"
        tool_block.input = {"path": "x.py"}
        response.content = [tool_block]

        result = _normalize_response(response)
        assert result["message"]["content"] == ""
        tcs = result["message"]["tool_calls"]
        assert len(tcs) == 1
        assert tcs[0]["id"] == "toolu_abc123"
        assert tcs[0]["function"]["name"] == "read_file"
        assert tcs[0]["function"]["arguments"] == {"path": "x.py"}

    def test_mixed_text_and_tool(self):
        response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Let me check."
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "toolu_xyz"
        tool_block.name = "list_dir"
        tool_block.input = {}
        response.content = [text_block, tool_block]

        result = _normalize_response(response)
        assert result["message"]["content"] == "Let me check."
        assert len(result["message"]["tool_calls"]) == 1

    def test_tool_use_without_id_generates_uuid(self):
        response = MagicMock()
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = None
        tool_block.name = "test"
        tool_block.input = {}
        response.content = [tool_block]

        result = _normalize_response(response)
        tc_id = result["message"]["tool_calls"][0]["id"]
        assert tc_id.startswith("toolu_")

    def test_empty_content(self):
        response = MagicMock()
        response.content = []

        result = _normalize_response(response)
        assert result["message"]["content"] == ""
        assert "tool_calls" not in result["message"]


# ---------------------------------------------------------------------------
# AnthropicProvider init and properties
# ---------------------------------------------------------------------------


class TestAnthropicProviderInit:
    def test_init_stores_api_key_and_base_url(self):
        p = AnthropicProvider(api_key="sk-test", base_url="https://custom.api/v1")
        assert p._api_key == "sk-test"
        assert p._base_url == "https://custom.api/v1"

    def test_name_is_anthropic(self):
        p = AnthropicProvider(api_key="test")
        assert p.name == "anthropic"

    def test_capabilities(self):
        p = AnthropicProvider(api_key="test")
        caps = p.capabilities
        assert caps.supports_tools is True
        assert caps.supports_streaming is True
        assert caps.supports_embeddings is False
        assert caps.supports_model_list is False


# ---------------------------------------------------------------------------
# AnthropicProvider.health_check — mocked
# ---------------------------------------------------------------------------


class TestAnthropicProviderHealthCheck:
    @patch("ollamacode.providers.anthropic_provider.anthropic", create=True)
    def test_health_check_success(self, mock_anthropic_module):
        """health_check returns (True, ...) when API call succeeds."""
        mock_client = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock()

        # We need to patch the import inside health_check
        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            p = AnthropicProvider(api_key="sk-test")
            ok, msg = p.health_check()
        assert ok is True
        assert "reachable" in msg.lower()

    @patch("ollamacode.providers.anthropic_provider.anthropic", create=True)
    def test_health_check_auth_failure(self, mock_anthropic_module):
        """health_check returns (False, ...) on 401 errors."""
        mock_client = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("401 Unauthorized")

        with patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            p = AnthropicProvider(api_key="bad-key")
            ok, msg = p.health_check()
        assert ok is False
        assert "api key" in msg.lower() or "invalid" in msg.lower()
