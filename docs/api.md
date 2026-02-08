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

## MCP client

::: ollamacode.mcp_client
    options:
      show_root_heading: true
      members:
        - connect_mcp_stdio
        - connect_mcp_servers
        - list_tools
        - call_tool
        - tool_result_to_content
        - McpConnection

## Config

::: ollamacode.config
    options:
      show_root_heading: true
      members:
        - load_config
        - merge_config_with_env
        - _find_config_file

## CLI

::: ollamacode.cli
    options:
      show_root_heading: true
      members:
        - main
