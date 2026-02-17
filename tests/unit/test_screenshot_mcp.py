"""Unit tests for screenshot MCP (Playwright)."""

from unittest.mock import patch


from ollamacode.servers import screenshot_mcp


def test_screenshot_empty_url():
    """screenshot returns error when url is empty."""
    out = screenshot_mcp.screenshot("")
    assert out["ok"] is False
    assert "url" in (out.get("error") or "").lower()
    assert out["path"] is None


def test_screenshot_playwright_not_available():
    """screenshot returns clear error when Playwright is not installed."""
    with patch.object(screenshot_mcp, "_playwright_available", return_value=False):
        out = screenshot_mcp.screenshot("https://example.com")
    assert out["ok"] is False
    assert "playwright" in (out.get("error") or "").lower()
    assert out["path"] is None


def test_is_browser_missing_error():
    """_is_browser_missing_error detects executable-doesnt-exist style errors."""
    assert (
        screenshot_mcp._is_browser_missing_error(
            Exception("Executable doesn't exist at /path")
        )
        is True
    )
    assert (
        screenshot_mcp._is_browser_missing_error(Exception("browser not found")) is True
    )
    assert (
        screenshot_mcp._is_browser_missing_error(Exception("chromium install required"))
        is True
    )
    assert (
        screenshot_mcp._is_browser_missing_error(Exception("Something else failed"))
        is False
    )


def test_install_chromium_returns_tuple():
    """_install_chromium returns (success: bool, message: str)."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = type(
            "R", (), {"returncode": 0, "stdout": "", "stderr": ""}
        )()
        ok, msg = screenshot_mcp._install_chromium()
    assert isinstance(ok, bool)
    assert isinstance(msg, str)
