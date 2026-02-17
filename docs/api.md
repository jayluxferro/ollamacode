# API Reference

Public Python API for OllamaCode.

## Agent

::: ollamacode.agent
    options:
      show_root_heading: true
      members:
        - run_agent_loop
        - run_agent_loop_no_mcp

## Bridge (MCP → Ollama tool format)

::: ollamacode.bridge
    options:
      show_root_heading: true
      members:
        - mcp_tool_to_ollama
        - mcp_tools_to_ollama

## Ollama client

::: ollamacode.ollama_client
    options:
      show_root_heading: true
      members:
        - chat_async
        - chat_sync
        - chat_stream_sync
        - is_ollama_template_error
        - wrap_ollama_template_error

## MCP client

::: ollamacode.mcp_client
    options:
      show_root_heading: true
      members:
        - connect_mcp_stdio
        - connect_mcp_servers
        - list_tools
        - call_tool
        - get_tool_name
        - get_registered_mcp_server_types
        - MCP_SERVER_TYPES_ENTRY_POINT_GROUP
        - tool_result_to_content
        - McpConnection

## Config

::: ollamacode.config
    options:
      show_root_heading: true
      members:
        - load_config
        - merge_config_with_env
        - get_resolved_config
        - get_env_config_overrides
        - ENV_CONFIG_SCHEMA
        - validate_config
        - ConfigValidationError
        - find_config_file
        - _find_config_file

## CLI

::: ollamacode.cli
    options:
      show_root_heading: true
      members:
        - main
