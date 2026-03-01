"""
Built-in MCP servers: filesystem, terminal, codebase search, git (read-only), tools (linter/tests).
Optional: semantic_mcp (embeddings-based search); add via config.

Used by default when no config or OLLAMACODE_MCP_ARGS is set.
Run with: python -m ollamacode.servers.fs_mcp, etc.
"""

from __future__ import annotations

import logging
import os
import sys


def configure_server_logging() -> None:
    """
    Reduce noisy MCP server logs in stdio mode.

    Default level is ERROR; override with OLLAMACODE_MCP_SERVER_LOG_LEVEL
    (e.g. DEBUG/INFO/WARNING/ERROR/CRITICAL).
    """
    level_name = os.environ.get("OLLAMACODE_MCP_SERVER_LOG_LEVEL", "ERROR").upper()
    level = getattr(logging, level_name, logging.WARNING)
    logging.getLogger().setLevel(level)
    quiet_stdio = os.environ.get(
        "OLLAMACODE_MCP_STDERR_QUIET", "1"
    ).strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if quiet_stdio:
        # For stdio MCP servers, any log noise can corrupt UX.
        logging.disable(logging.CRITICAL)
    elif level >= logging.WARNING:
        logging.disable(logging.INFO)
    targets = (
        "mcp",
        "mcp.server",
        "mcp.server.lowlevel",
        "mcp.server.lowlevel.server",
        "uvicorn",
        "httpx",
        "urllib3",
    )
    for name in targets:
        lg = logging.getLogger(name)
        lg.setLevel(level)
        lg.propagate = False
    if quiet_stdio:
        try:
            sys.stderr = open(os.devnull, "w", encoding="utf-8")
        except OSError:
            pass
