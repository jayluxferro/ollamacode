"""
Convert MCP config from Cursor or Claude (JSON) format to OllamaCode YAML format.

Usage:
  ollamacode convert-mcp cursor.json [--output ollamacode.yaml]
  ollamacode convert-mcp claude.json -o .ollamacode/config.yaml
  cat mcp.json | ollamacode convert-mcp -o ollamacode.yaml  # stdin when no input file
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


def load_json(input_path: str | None) -> dict[str, Any]:
    """Load JSON from file or stdin. Returns dict."""
    if input_path:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        text = path.read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()
    return json.loads(text)


def _normalize_server_entry(server_key: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Convert one Cursor/Claude server entry to OllamaCode mcp_servers item. Preserves key as name."""
    if not isinstance(raw, dict):
        return {}
    # URL-based (SSE or Streamable HTTP)
    url = raw.get("url")
    if url:
        # Heuristic: /sse often means SSE; /mcp or streamable_http often means streamable HTTP
        url_lower = (url or "").lower()
        if "/sse" in url_lower or raw.get("transport") == "sse":
            entry = {"type": "sse", "url": url, **(_headers_timeouts(raw))}
        else:
            entry = {"type": "streamable_http", "url": url, **(_headers_timeouts(raw))}
        entry["name"] = server_key
        return entry
    # Stdio: command + args
    command = raw.get("command", "python")
    args = raw.get("args")
    if args is None:
        args = []
    if isinstance(args, str):
        args = args.split()
    entry: dict[str, Any] = {"type": "stdio", "command": command, "args": list(args)}
    entry["name"] = server_key
    if raw.get("env"):
        entry["env"] = dict(raw["env"])
    return entry


def _headers_timeouts(raw: dict[str, Any]) -> dict[str, Any]:
    """Extract optional headers, timeout, sse_read_timeout for URL servers."""
    out: dict[str, Any] = {}
    if raw.get("headers"):
        out["headers"] = dict(raw["headers"])
    if "timeout" in raw:
        out["timeout"] = raw["timeout"]
    if "sse_read_timeout" in raw:
        out["sse_read_timeout"] = raw["sse_read_timeout"]
    return out


def _extract_servers_object(data: dict[str, Any]) -> dict[str, Any] | None:
    """Get mcpServers or mcp_servers object from config."""
    if "mcpServers" in data and isinstance(data["mcpServers"], dict):
        return data["mcpServers"]
    if "mcp_servers" in data and isinstance(data["mcp_servers"], dict):
        return data["mcp_servers"]
    return None


def convert_to_ollamacode_servers(data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Convert Cursor or Claude MCP config (object of named servers) to OllamaCode mcp_servers list.
    Auto-detects format. Returns list of server entries for mcp_servers.
    """
    servers_obj = _extract_servers_object(data)
    if not servers_obj:
        return []
    result: list[dict[str, Any]] = []
    for name, raw in servers_obj.items():
        if not isinstance(raw, dict):
            continue
        entry = _normalize_server_entry(name, raw)
        if entry:
            result.append(entry)
    return result


def emit_yaml(
    mcp_servers: list[dict[str, Any]],
    output_path: str | None,
    top_level_keys: dict[str, Any] | None = None,
) -> None:
    """Write OllamaCode-style YAML to file or stdout."""
    if yaml is None:
        raise RuntimeError(
            "PyYAML required for convert-mcp. Install with: pip install pyyaml"
        )
    out: dict[str, Any] = dict(top_level_keys) if top_level_keys else {}
    out["mcp_servers"] = mcp_servers
    yaml_str = yaml.dump(
        out, default_flow_style=False, allow_unicode=True, sort_keys=False
    )
    if output_path:
        Path(output_path).write_text(yaml_str, encoding="utf-8")
    else:
        sys.stdout.write(yaml_str)


def run_convert(input_path: str | None, output_path: str | None) -> None:
    """
    Load JSON from input_path (or stdin), convert to OllamaCode YAML, write to output_path (or stdout).
    """
    data = load_json(input_path)
    mcp_servers = convert_to_ollamacode_servers(data)
    if not mcp_servers:
        servers_obj = _extract_servers_object(data)
        if servers_obj is None:
            print(
                "No mcpServers or mcp_servers object found in input.",
                file=sys.stderr,
            )
        else:
            print("No valid server entries found.", file=sys.stderr)
        raise SystemExit(1)
    emit_yaml(mcp_servers, output_path)
