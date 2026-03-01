"""
Built-in web search MCP server: web_search(query).

Calls an external search API (e.g. Tavily, Serper) when configured.
Config: web_search.enabled, web_search.endpoint, web_search.api_key; or env:
OLLAMACODE_WEB_SEARCH_ENDPOINT, OLLAMACODE_WEB_SEARCH_API_KEY.
"""

import ipaddress
import json
import logging
import os
import socket
import urllib.request
from urllib.error import HTTPError
from urllib.parse import urlparse
from typing import Any

from mcp.server.fastmcp import FastMCP
from . import configure_server_logging

configure_server_logging()

logger = logging.getLogger(__name__)

mcp = FastMCP("ollamacode-web-search")

_MAX_BODY_CHARS = 8000
_TIMEOUT = 15

_PRIVATE_HOSTNAMES = {"localhost", "localhost.localdomain"}


def _is_private_url(url: str) -> bool:
    """Return True if the URL resolves to a private/loopback address (SSRF guard)."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
    except Exception:
        return True  # Reject unparseable URLs

    if not hostname:
        return True

    if hostname.lower() in _PRIVATE_HOSTNAMES:
        return True

    # Try parsing as an IP literal first
    try:
        addr = ipaddress.ip_address(hostname)
        return (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
        )
    except ValueError:
        pass

    # Resolve hostname and check all resulting IPs
    try:
        for info in socket.getaddrinfo(
            hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM
        ):
            addr_str = info[4][0]
            try:
                addr = ipaddress.ip_address(addr_str)
                if (
                    addr.is_private
                    or addr.is_loopback
                    or addr.is_link_local
                    or addr.is_reserved
                ):
                    return True
            except ValueError:
                continue
    except socket.gaierror:
        pass  # DNS failure will be caught by the actual request

    return False


def _get_endpoint() -> str | None:
    return (os.environ.get("OLLAMACODE_WEB_SEARCH_ENDPOINT") or "").strip() or None


def _get_api_key() -> str | None:
    return (os.environ.get("OLLAMACODE_WEB_SEARCH_API_KEY") or "").strip() or None


@mcp.tool()
def web_search(query: str) -> dict[str, Any]:
    """
    Run a web search and return snippets or a summary. Requires web search to be
    configured (endpoint and optionally API key in config or env).
    """
    endpoint = _get_endpoint()
    if not endpoint:
        return {
            "ok": False,
            "error": "Web search not configured. Set web_search.endpoint in config or OLLAMACODE_WEB_SEARCH_ENDPOINT.",
            "results": [],
        }
    if _is_private_url(endpoint):
        logger.warning(
            "Web search endpoint rejected (private/loopback address): %s", endpoint
        )
        return {
            "ok": False,
            "error": "Web search endpoint resolves to a private or loopback address.",
            "results": [],
        }
    api_key = _get_api_key()
    try:
        url = endpoint
        data = json.dumps({"query": query}).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("User-Agent", "OllamaCode/1.0")
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            if len(body) > _MAX_BODY_CHARS:
                body = body[:_MAX_BODY_CHARS] + "\n... [truncated]"
            # Try to parse as JSON and extract results/text
            try:
                js = json.loads(body)
                if isinstance(js, dict):
                    results = (
                        js.get("results")
                        or js.get("answer")
                        or js.get("text")
                        or js.get("content")
                    )
                    if results is None and "data" in js:
                        results = js["data"]
                    if results is not None:
                        return {
                            "ok": True,
                            "error": None,
                            "results": results
                            if isinstance(results, list)
                            else [results],
                            "raw_preview": body[:500],
                        }
                return {
                    "ok": True,
                    "error": None,
                    "results": [],
                    "raw_preview": body[:500],
                }
            except json.JSONDecodeError:
                return {"ok": True, "error": None, "results": [], "raw_preview": body}
    except HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")[:500] if e.fp else ""
        return {
            "ok": False,
            "error": f"HTTP {e.code}: {e.reason}",
            "results": [],
            "raw_preview": err_body,
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "results": []}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
