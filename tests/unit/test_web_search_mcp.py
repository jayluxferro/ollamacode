"""Unit tests for web_search_mcp.py — URL validation, SSRF, error handling."""


class TestIsPrivateUrl:
    """Test the SSRF guard _is_private_url()."""

    def test_localhost_rejected(self):
        from ollamacode.servers.web_search_mcp import _is_private_url

        assert _is_private_url("http://localhost:8080/search") is True
        assert _is_private_url("http://localhost.localdomain/search") is True

    def test_loopback_ip_rejected(self):
        from ollamacode.servers.web_search_mcp import _is_private_url

        assert _is_private_url("http://127.0.0.1:8080/search") is True
        assert _is_private_url("http://[::1]:8080/search") is True

    def test_private_ip_rejected(self):
        from ollamacode.servers.web_search_mcp import _is_private_url

        assert _is_private_url("http://192.168.1.1/search") is True
        assert _is_private_url("http://10.0.0.1/search") is True
        assert _is_private_url("http://172.16.0.1/search") is True

    def test_public_ip_allowed(self):
        from ollamacode.servers.web_search_mcp import _is_private_url

        assert _is_private_url("http://8.8.8.8/search") is False

    def test_empty_url_rejected(self):
        from ollamacode.servers.web_search_mcp import _is_private_url

        assert _is_private_url("") is True

    def test_no_hostname_rejected(self):
        from ollamacode.servers.web_search_mcp import _is_private_url

        assert _is_private_url("http:///path") is True


class TestWebSearch:
    """Test the web_search tool function."""

    def test_no_endpoint_configured(self, monkeypatch):
        from ollamacode.servers.web_search_mcp import web_search

        monkeypatch.delenv("OLLAMACODE_WEB_SEARCH_ENDPOINT", raising=False)
        monkeypatch.delenv("OLLAMACODE_WEB_SEARCH_API_KEY", raising=False)
        result = web_search("test query")
        assert result["ok"] is False
        assert "not configured" in result["error"]

    def test_private_endpoint_rejected(self, monkeypatch):
        from ollamacode.servers.web_search_mcp import web_search

        monkeypatch.setenv(
            "OLLAMACODE_WEB_SEARCH_ENDPOINT", "http://localhost:9999/search"
        )
        result = web_search("test query")
        assert result["ok"] is False
        assert (
            "private" in result["error"].lower()
            or "loopback" in result["error"].lower()
        )
