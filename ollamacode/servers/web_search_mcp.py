"""
Built-in web search MCP server: web_search(query).

Calls an external search API (e.g. Tavily, Serper) when configured.
Config: web_search.enabled, web_search.endpoint, web_search.api_key; or env:
OLLAMACODE_WEB_SEARCH_ENDPOINT, OLLAMACODE_WEB_SEARCH_API_KEY.
"""

import json
import os
import urllib.request
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ollamacode-web-search")

_MAX_BODY_CHARS = 8000
_TIMEOUT = 15


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
    except urllib.error.HTTPError as e:
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
