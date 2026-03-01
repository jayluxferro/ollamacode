"""
Built-in web fetch MCP server: fetch_url with HTML-to-text extraction and SSRF protection.

Tool: fetch_url(url, extract_text=True) -> {status_code, body, error}
Reuses the _is_private_url SSRF guard from web_search_mcp.
"""

import html.parser
import ipaddress
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

mcp = FastMCP("ollamacode-webfetch")

_MAX_BODY_CHARS = 200_000
_TIMEOUT = 30
_MAX_URL_LEN = 4096

_PRIVATE_HOSTNAMES = {"localhost", "localhost.localdomain"}


def _is_private_url(url: str) -> bool:
    """Return True if the URL resolves to a private/loopback address (SSRF guard)."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
    except Exception:
        return True

    if not hostname:
        return True

    if hostname.lower() in _PRIVATE_HOSTNAMES:
        return True

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
        pass

    return False


class _HTMLTextExtractor(html.parser.HTMLParser):
    """Simple HTML-to-text converter: strips tags, keeps text content."""

    _SKIP_TAGS = frozenset({"script", "style", "noscript", "head", "meta", "link"})

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag.lower() in (
            "br",
            "p",
            "div",
            "li",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "tr",
        ):
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        text = "".join(self._parts)
        # Collapse whitespace
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)


def _extract_text(html_content: str) -> str:
    """Extract readable text from HTML content."""
    parser = _HTMLTextExtractor()
    try:
        parser.feed(html_content)
        return parser.get_text()
    except Exception:
        # Fallback: crude tag stripping
        import re

        text = re.sub(r"<[^>]+>", " ", html_content)
        return re.sub(r"\s+", " ", text).strip()


@mcp.tool()
def fetch_url(
    url: str,
    extract_text: bool = True,
    timeout_seconds: int = _TIMEOUT,
    max_chars: int = _MAX_BODY_CHARS,
) -> dict[str, Any]:
    """
    Fetch a URL and return its content.

    url: Full URL to fetch (must be http or https).
    extract_text: If True, strip HTML tags and return readable text. If False, return raw body.
    timeout_seconds: Request timeout (default 30).
    max_chars: Maximum characters to return (default 200000).
    """
    if not url or not isinstance(url, str):
        return {"status_code": -1, "body": "", "error": "URL is required"}

    if len(url) > _MAX_URL_LEN:
        return {
            "status_code": -1,
            "body": "",
            "error": f"URL too long (max {_MAX_URL_LEN} chars)",
        }

    try:
        parsed = urlparse(url)
    except Exception:
        return {"status_code": -1, "body": "", "error": "Invalid URL"}

    if parsed.scheme not in ("http", "https"):
        return {
            "status_code": -1,
            "body": "",
            "error": f"URL scheme must be http or https, got {parsed.scheme!r}",
        }

    # SSRF protection: block private/loopback addresses
    allow_private = os.environ.get(
        "OLLAMACODE_WEBFETCH_ALLOW_PRIVATE", "0"
    ).strip().lower() in ("1", "true")
    if not allow_private and _is_private_url(url):
        return {
            "status_code": -1,
            "body": "",
            "error": "URL resolves to a private or loopback address (blocked for SSRF protection).",
        }

    timeout_seconds = max(1, min(timeout_seconds, 120))

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "OllamaCode/1.0 (webfetch tool)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            content_type = resp.headers.get("Content-Type", "")

        body = raw
        if extract_text and (
            "html" in content_type.lower() or raw.strip().startswith("<")
        ):
            body = _extract_text(raw)

        if len(body) > max_chars:
            body = body[:max_chars] + "\n... [truncated]"

        return {"status_code": 200, "body": body, "error": None}
    except HTTPError as e:
        err_body = ""
        if e.fp:
            try:
                err_body = e.read().decode("utf-8", errors="replace")[:2000]
            except Exception:
                pass
        return {
            "status_code": e.code,
            "body": err_body,
            "error": f"HTTP {e.code}: {e.reason}",
        }
    except Exception as e:
        return {"status_code": -1, "body": "", "error": str(e)}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
