"""
Simple LSP client: spawn a language server via subprocess (stdio JSON-RPC)
and expose basic queries — definitions, references, hover, document symbols.

Usage:
    client = LSPClient("pyright-langserver", ["--stdio"])
    client.initialize(root_uri="file:///path/to/project")
    result = client.get_definition("file:///path/to/file.py", line=10, character=5)
    client.shutdown()
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_JSONRPC_VERSION = "2.0"


class LSPClient:
    """Minimal LSP client communicating over stdio JSON-RPC with a language server."""

    def __init__(
        self, command: str, args: list[str] | None = None, cwd: str | None = None
    ):
        self._command = command
        self._args = args or []
        self._cwd = cwd
        self._process: subprocess.Popen | None = None
        self._request_id = 0
        self._lock = threading.Lock()
        self._initialized = False

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _start_server(self) -> None:
        if self._process is not None:
            return
        try:
            self._process = subprocess.Popen(
                [self._command, *self._args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self._cwd,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"LSP server binary not found: {self._command}") from exc

    def _next_id(self) -> int:
        with self._lock:
            self._request_id += 1
            return self._request_id

    def _send(self, method: str, params: dict[str, Any] | None = None) -> int:
        """Send a JSON-RPC request and return its id."""
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("LSP server not started")
        rid = self._next_id()
        body = json.dumps(
            {
                "jsonrpc": _JSONRPC_VERSION,
                "id": rid,
                "method": method,
                "params": params or {},
            }
        )
        header = f"Content-Length: {len(body)}\r\n\r\n"
        self._process.stdin.write(header.encode("utf-8"))
        self._process.stdin.write(body.encode("utf-8"))
        self._process.stdin.flush()
        return rid

    def _send_notification(
        self, method: str, params: dict[str, Any] | None = None
    ) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("LSP server not started")
        body = json.dumps(
            {
                "jsonrpc": _JSONRPC_VERSION,
                "method": method,
                "params": params or {},
            }
        )
        header = f"Content-Length: {len(body)}\r\n\r\n"
        self._process.stdin.write(header.encode("utf-8"))
        self._process.stdin.write(body.encode("utf-8"))
        self._process.stdin.flush()

    def _recv(self) -> dict[str, Any]:
        """Read one JSON-RPC response from stdout."""
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("LSP server not started")
        # Read headers until blank line
        content_length = 0
        while True:
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError("LSP server closed stdout unexpectedly")
            line_str = line.decode("utf-8", errors="replace").strip()
            if not line_str:
                break
            if line_str.lower().startswith("content-length:"):
                content_length = int(line_str.split(":", 1)[1].strip())
        if content_length <= 0:
            raise RuntimeError("LSP response has no Content-Length")
        data = self._process.stdout.read(content_length)
        return json.loads(data.decode("utf-8", errors="replace"))

    def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send request and wait for the matching response, skipping notifications."""
        rid = self._send(method, params)
        while True:
            resp = self._recv()
            # Skip server-initiated notifications/requests (no id or different id)
            if resp.get("id") == rid:
                if "error" in resp:
                    err = resp["error"]
                    raise RuntimeError(
                        f"LSP error {err.get('code', '?')}: {err.get('message', '')}"
                    )
                return resp.get("result")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(
        self, root_uri: str | None = None, workspace_root: str | None = None
    ) -> dict[str, Any]:
        """Send initialize + initialized. Returns server capabilities."""
        self._start_server()
        if root_uri is None and workspace_root:
            root_uri = Path(workspace_root).resolve().as_uri()
        result = self._request(
            "initialize",
            {
                "processId": None,
                "rootUri": root_uri,
                "capabilities": {
                    "textDocument": {
                        "definition": {"dynamicRegistration": False},
                        "references": {"dynamicRegistration": False},
                        "hover": {"contentFormat": ["plaintext", "markdown"]},
                        "documentSymbol": {"dynamicRegistration": False},
                    },
                },
            },
        )
        self._send_notification("initialized")
        self._initialized = True
        return result or {}

    def shutdown(self) -> None:
        """Gracefully shut down the LSP server."""
        if not self._initialized or self._process is None:
            return
        try:
            self._request("shutdown")
            self._send_notification("exit")
        except Exception:
            logger.debug("LSP shutdown error (ignored)")
        finally:
            self._initialized = False
            if self._process and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
            self._process = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _text_document_position(uri: str, line: int, character: int) -> dict[str, Any]:
        return {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
        }

    @staticmethod
    def _file_uri(path: str) -> str:
        """Convert a filesystem path to a file:// URI."""
        return Path(path).resolve().as_uri()

    # ------------------------------------------------------------------
    # LSP queries
    # ------------------------------------------------------------------

    def get_definition(self, uri: str, line: int, character: int) -> Any:
        """textDocument/definition — returns Location(s) or None."""
        params = self._text_document_position(uri, line, character)
        return self._request("textDocument/definition", params)

    def get_references(
        self, uri: str, line: int, character: int, include_declaration: bool = True
    ) -> Any:
        """textDocument/references — returns list of Locations."""
        params = self._text_document_position(uri, line, character)
        params["context"] = {"includeDeclaration": include_declaration}
        return self._request("textDocument/references", params)

    def get_hover(self, uri: str, line: int, character: int) -> Any:
        """textDocument/hover — returns Hover or None."""
        params = self._text_document_position(uri, line, character)
        return self._request("textDocument/hover", params)

    def get_symbols(self, uri: str) -> Any:
        """textDocument/documentSymbol — returns SymbolInformation[] or DocumentSymbol[]."""
        return self._request(
            "textDocument/documentSymbol",
            {
                "textDocument": {"uri": uri},
            },
        )

    def did_open(self, uri: str, language_id: str, text: str, version: int = 1) -> None:
        """Notify the server that a document was opened (required before most queries)."""
        self._send_notification(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": language_id,
                    "version": version,
                    "text": text,
                },
            },
        )

    def __enter__(self) -> LSPClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.shutdown()
