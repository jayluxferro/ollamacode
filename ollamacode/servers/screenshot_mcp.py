"""
Built-in screenshot MCP server: screenshot(url) using Playwright.

Chromium is installed automatically on first screenshot if missing; or run
`ollamacode install-browsers` once to install upfront.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from . import configure_server_logging

configure_server_logging()

mcp = FastMCP("ollamacode-screenshot")

_PLAYWRIGHT_AVAILABLE: bool | None = None
_CHROMIUM_INSTALL_ATTEMPTED: bool = False


def _playwright_available() -> bool:
    global _PLAYWRIGHT_AVAILABLE
    if _PLAYWRIGHT_AVAILABLE is not None:
        return _PLAYWRIGHT_AVAILABLE
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401

        _PLAYWRIGHT_AVAILABLE = True
    except ImportError:
        _PLAYWRIGHT_AVAILABLE = False
    return _PLAYWRIGHT_AVAILABLE


def _install_chromium() -> tuple[bool, str]:
    """Run playwright install chromium. Returns (success, message)."""
    global _CHROMIUM_INSTALL_ATTEMPTED
    _CHROMIUM_INSTALL_ATTEMPTED = True
    try:
        out = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if out.returncode == 0:
            return True, "Chromium installed successfully."
        return False, (
            out.stderr or out.stdout or f"Exit code {out.returncode}"
        ).strip()[:500]
    except subprocess.TimeoutExpired:
        return False, "Playwright install timed out (5 min)."
    except FileNotFoundError:
        return False, "playwright CLI not found. Is the playwright package installed?"
    except Exception as e:
        return False, str(e)[:500]


def _is_browser_missing_error(exc: BaseException) -> bool:
    """True if the exception indicates Chromium executable is not installed."""
    msg = (getattr(exc, "message", None) or str(exc)).lower()
    return (
        (
            "executable" in msg
            and (
                "doesn't exist" in msg or "does not exist" in msg or "not found" in msg
            )
        )
        or "browser not found" in msg
        or "chromium" in msg
        and "install" in msg
    )


@mcp.tool()
def screenshot(
    url: str,
    timeout_seconds: int = 30,
    viewport_width: int = 1280,
    viewport_height: int = 720,
    full_page: bool = False,
) -> dict[str, Any]:
    """
    Take a screenshot of a web page using a headless browser.

    url: Full URL to open (e.g. https://example.com).
    timeout_seconds: Navigation timeout (default 30).
    viewport_width: Browser viewport width in pixels (default 1280).
    viewport_height: Browser viewport height in pixels (default 720).
    full_page: If true, capture the full scrollable page; otherwise only the viewport.

    Returns path to the saved PNG file and dimensions. Chromium is installed automatically on first use if missing.
    """
    if not url or not url.strip():
        return {"ok": False, "error": "url is required", "path": None}
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    # SSRF prevention: reject URLs targeting private/local addresses
    from urllib.parse import urlparse

    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    _private_hosts = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"}
    if (
        hostname in _private_hosts
        or hostname.startswith("192.168.")
        or hostname.startswith("10.")
        or hostname.startswith("172.")
    ):
        return {
            "ok": False,
            "error": "URLs targeting private/local addresses are not allowed",
            "path": None,
        }
    if parsed.scheme not in ("http", "https"):
        return {
            "ok": False,
            "error": f"Unsupported URL scheme: {parsed.scheme}",
            "path": None,
        }
    # Clamp viewport dimensions
    viewport_width = max(320, min(viewport_width, 4096))
    viewport_height = max(200, min(viewport_height, 4096))
    timeout_seconds = max(1, min(timeout_seconds, 120))
    if not _playwright_available():
        return {
            "ok": False,
            "error": "Playwright not installed. pip install playwright, then run: ollamacode install-browsers",
            "path": None,
        }
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {
            "ok": False,
            "error": "Playwright not installed. pip install playwright, then ollamacode install-browsers",
            "path": None,
        }
    path: str | None = None
    last_error: BaseException | None = None
    for attempt in range(2):
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                path = f.name
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                try:
                    page = browser.new_page(
                        viewport={"width": viewport_width, "height": viewport_height}
                    )
                    page.goto(
                        url,
                        timeout=timeout_seconds * 1000,
                        wait_until="domcontentloaded",
                    )
                    page.screenshot(path=path, full_page=full_page)
                    size = page.viewport_size or {
                        "width": viewport_width,
                        "height": viewport_height,
                    }
                finally:
                    browser.close()
            return {
                "ok": True,
                "error": None,
                "path": path,
                "width": size.get("width", viewport_width),
                "height": size.get("height", viewport_height),
                "full_page": full_page,
            }
        except Exception as e:
            last_error = e
            if path:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    pass
                path = None
            if (
                attempt == 0
                and _is_browser_missing_error(e)
                and not _CHROMIUM_INSTALL_ATTEMPTED
            ):
                ok, msg = _install_chromium()
                if not ok:
                    return {
                        "ok": False,
                        "error": f"Auto-install failed: {msg}",
                        "path": None,
                    }
                continue
            return {"ok": False, "error": str(e), "path": None}
    return {
        "ok": False,
        "error": str(last_error) if last_error else "Screenshot failed",
        "path": None,
    }


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
