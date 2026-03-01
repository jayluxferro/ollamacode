"""Unit tests for Apple FM provider parsing/normalization helpers."""

from ollamacode.providers import get_provider
from ollamacode.providers.apple_fm_provider import (
    AppleFMProvider,
    _allowed_tool_names,
    _canonicalize_tools,
    _compress_text,
    _is_guardrail_error,
    _is_context_window_error,
    _last_user_prompt,
    _messages_to_prompt,
    _normalize_structured_response,
    _normalize_tool_calls,
    _parse_tool_call_response,
    _tool_name_list_text,
    _tool_router_schema,
)


def _tools() -> list[dict]:
    return [
        {"function": {"name": "read_file", "description": "", "parameters": {}}},
        {
            "function": {
                "name": "functions::run_command",
                "description": "",
                "parameters": {},
            }
        },
    ]


def test_allowed_tool_names_normalize_functions_prefix() -> None:
    allowed = _allowed_tool_names(_tools())
    assert "read_file" in allowed
    assert "run_command" in allowed
    assert "functions::run_command" not in allowed


def test_canonicalize_tools_prefers_short_names() -> None:
    tools = [
        {
            "function": {
                "name": "functions::read_file",
                "description": "",
                "parameters": {},
            }
        },
        {
            "function": {
                "name": "ollamacode-fs_read_file",
                "description": "",
                "parameters": {},
            }
        },
        {"function": {"name": "read_file", "description": "", "parameters": {}}},
    ]
    out = _canonicalize_tools(tools)
    assert len(out) == 1
    assert out[0]["function"]["name"] == "read_file"


def test_normalize_tool_calls_filters_unknown_and_bad_args() -> None:
    allowed = {"read_file"}
    raw = [
        {"function": {"name": "read_file", "arguments": {"path": "a.py"}}},
        {"function": {"name": "unknown_tool", "arguments": {"x": 1}}},
        {"function": {"name": "read_file", "arguments": "not-json"}},
    ]
    out = _normalize_tool_calls(raw, allowed)
    assert len(out) == 2
    assert out[0]["function"]["name"] == "read_file"
    assert out[0]["function"]["arguments"] == {"path": "a.py"}
    assert out[1]["function"]["arguments"] == {}


def test_parse_tool_call_response_from_json_block() -> None:
    allowed = {"read_file"}
    text = """```json
{"tool_calls":[{"function":{"name":"read_file","arguments":{"path":"README.md"}}}]}
```"""
    out = _parse_tool_call_response(text, allowed)
    assert out is not None
    msg = out["message"]
    assert msg["tool_calls"][0]["function"]["name"] == "read_file"
    assert msg["tool_calls"][0]["function"]["arguments"]["path"] == "README.md"


def test_parse_tool_call_response_rejects_non_allowed_tool() -> None:
    out = _parse_tool_call_response(
        '{"tool_calls":[{"function":{"name":"delete_all","arguments":{}}}]}',
        {"read_file"},
    )
    assert out is None


def test_normalize_structured_response_content_only() -> None:
    out = _normalize_structured_response({"content": "hello"}, {"read_file"})
    assert out == {"message": {"content": "hello"}}


def test_get_provider_apple_fm() -> None:
    p = get_provider({"provider": "apple_fm"})
    assert isinstance(p, AppleFMProvider)


def test_tool_router_schema_limits_tool_names() -> None:
    schema = _tool_router_schema({"run_command", "read_file"})
    enum_vals = schema["properties"]["tool_calls"]["items"]["properties"]["function"][
        "properties"
    ]["name"]["enum"]
    assert sorted(enum_vals) == ["read_file", "run_command"]


def test_is_context_window_error_detection() -> None:
    assert _is_context_window_error(Exception("Context window size exceeded"))
    assert not _is_context_window_error(Exception("some other error"))


def test_is_guardrail_error_detection() -> None:
    assert _is_guardrail_error(Exception("Guardrail violation occurred"))
    assert not _is_guardrail_error(Exception("some other error"))


def test_tool_name_list_text_is_unique_sorted() -> None:
    txt = _tool_name_list_text(_tools())
    assert txt == '["read_file", "run_command"]'


def test_messages_to_prompt_truncates_content() -> None:
    prompt = _messages_to_prompt(
        [{"role": "user", "content": "x" * 200}],
        max_chars=10,
    )
    assert "x" * 10 in prompt
    assert "x" * 11 not in prompt


def test_compress_text_reduces_length() -> None:
    out = _compress_text("x" * 100, 20)
    assert len(out) <= 25
    assert "..." in out


def test_last_user_prompt_finds_latest_user() -> None:
    prompt = _last_user_prompt(
        [
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "second"},
        ]
    )
    assert prompt == "second"
