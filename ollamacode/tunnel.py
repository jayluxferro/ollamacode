"""Tunnel integration for OllamaCode serve mode.

Auto-launches a public tunnel alongside ``ollamacode serve`` so the local
HTTP server is accessible from the internet.

Supported tunnels (configured via ``ollamacode.yaml`` under ``tunnel.type``):

  cloudflare  — ``cloudflared tunnel --url http://localhost:<port>``
  ngrok       — ``ngrok http <port>`` (parses JSON API for URL)
  tailscale   — ``tailscale funnel <port>``
  custom      — arbitrary command template with ``{port}`` substitution

CLI flag: ``--no-tunnel`` disables even if configured.

Config example::

    tunnel:
      type: cloudflare   # or ngrok, tailscale, custom
      args: []           # extra args (optional)
      # For custom:
      # command: "my-tunnel http {port}"

Usage::

    from ollamacode.tunnel import start_tunnel, stop_tunnel
    url, process = start_tunnel("cloudflare", port=8000)
    ...
    stop_tunnel(process)
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunnel interface
# ---------------------------------------------------------------------------


class TunnelError(RuntimeError):
    """Raised when a tunnel cannot be started or its public URL cannot be determined."""


def _start_process(args: list[str]) -> "subprocess.Popen[str]":
    """Start a subprocess, returning the Popen handle."""
    logger.debug("Starting tunnel: %s", " ".join(args))
    return subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


# ---------------------------------------------------------------------------
# Cloudflare
# ---------------------------------------------------------------------------


def _start_cloudflare(
    port: int, extra_args: list[str]
) -> tuple[str, "subprocess.Popen[str]"]:
    """Start ``cloudflared tunnel`` and extract the public URL from its output."""
    cmd = ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"] + extra_args
    proc = _start_process(cmd)

    url: str | None = None
    deadline = time.monotonic() + 30.0
    for line in proc.stdout:  # type: ignore[union-attr]
        line = line.strip()
        logger.debug("cloudflared: %s", line)
        m = re.search(r"https?://[a-z0-9\-]+\.trycloudflare\.com", line, re.IGNORECASE)
        if m:
            url = m.group(0)
            break
        if time.monotonic() > deadline:
            break

    if not url:
        proc.terminate()
        raise TunnelError("cloudflared did not print a public URL within 30 s")
    return url, proc


# ---------------------------------------------------------------------------
# ngrok
# ---------------------------------------------------------------------------


def _start_ngrok(
    port: int, extra_args: list[str]
) -> tuple[str, "subprocess.Popen[str]"]:
    """Start ``ngrok http`` and fetch the public URL from its JSON API."""
    cmd = ["ngrok", "http", str(port)] + extra_args
    proc = _start_process(cmd)

    # Give ngrok time to start
    time.sleep(2)
    url: str | None = None
    for attempt in range(10):
        try:
            import urllib.request

            with urllib.request.urlopen(
                "http://localhost:4040/api/tunnels", timeout=3
            ) as resp:
                data = json.loads(resp.read())
            tunnels = data.get("tunnels", [])
            for t in tunnels:
                if t.get("proto") == "https":
                    url = t.get("public_url")
                    break
                if t.get("proto") == "http":
                    url = t.get("public_url")
        except Exception:
            pass
        if url:
            break
        time.sleep(1)

    if not url:
        proc.terminate()
        raise TunnelError(
            "Could not read public URL from ngrok API (http://localhost:4040)"
        )
    return url, proc


# ---------------------------------------------------------------------------
# Tailscale
# ---------------------------------------------------------------------------


def _start_tailscale(
    port: int, extra_args: list[str]
) -> tuple[str, "subprocess.Popen[str]"]:
    """Start ``tailscale funnel`` and derive the public URL from hostname."""
    cmd = ["tailscale", "funnel", str(port)] + extra_args
    proc = _start_process(cmd)

    # Tailscale prints the funnel URL in its output; wait briefly.
    url: str | None = None
    deadline = time.monotonic() + 15.0
    for line in proc.stdout:  # type: ignore[union-attr]
        line = line.strip()
        logger.debug("tailscale funnel: %s", line)
        m = re.search(r"https://[a-z0-9\-\.]+\.ts\.net[\S]*", line, re.IGNORECASE)
        if m:
            url = m.group(0).rstrip("/")
            break
        if time.monotonic() > deadline:
            break

    if not url:
        # Best-effort: derive from tailscale status
        try:
            r = subprocess.run(
                ["tailscale", "status", "--json"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            data = json.loads(r.stdout)
            dns = data.get("Self", {}).get("DNSName", "")
            if dns:
                url = f"https://{dns.rstrip('.')}:{port}"
        except Exception:
            pass

    if not url:
        proc.terminate()
        raise TunnelError("Could not determine public URL from tailscale funnel output")
    return url, proc


# ---------------------------------------------------------------------------
# Custom tunnel
# ---------------------------------------------------------------------------


def _start_custom(
    port: int, command_template: str, extra_args: list[str]
) -> tuple[str, "subprocess.Popen[str]"]:
    """Start an arbitrary tunnel command with ``{port}`` substitution.

    The public URL must be printed to the process's combined stdout/stderr on
    the first line that matches an ``https?://`` URL pattern.
    """
    cmd_str = command_template.format(port=port)
    import shlex

    cmd = shlex.split(cmd_str) + extra_args
    proc = _start_process(cmd)

    url: str | None = None
    deadline = time.monotonic() + 30.0
    for line in proc.stdout:  # type: ignore[union-attr]
        line = line.strip()
        logger.debug("custom tunnel: %s", line)
        m = re.search(r"https?://\S+", line)
        if m:
            url = m.group(0).rstrip("/,;")
            break
        if time.monotonic() > deadline:
            break

    if not url:
        proc.terminate()
        raise TunnelError("Custom tunnel did not print a public URL within 30 s")
    return url, proc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def start_tunnel(
    tunnel_type: str,
    port: int,
    *,
    extra_args: list[str] | None = None,
    command: str | None = None,
) -> tuple[str, "subprocess.Popen[str]"]:
    """Start a tunnel and return ``(public_url, process)``.

    Parameters
    ----------
    tunnel_type:
        One of ``cloudflare``, ``ngrok``, ``tailscale``, ``custom``.
    port:
        Local port to expose.
    extra_args:
        Additional CLI arguments forwarded to the tunnel binary.
    command:
        Required for ``custom`` type: command template with ``{port}`` placeholder.
    """
    extra: list[str] = list(extra_args or [])
    ttype = tunnel_type.lower().strip()
    if ttype == "cloudflare":
        return _start_cloudflare(port, extra)
    elif ttype == "ngrok":
        return _start_ngrok(port, extra)
    elif ttype == "tailscale":
        return _start_tailscale(port, extra)
    elif ttype == "custom":
        if not command:
            raise TunnelError("tunnel.type=custom requires tunnel.command to be set")
        return _start_custom(port, command, extra)
    else:
        raise TunnelError(
            f"Unknown tunnel type: {tunnel_type!r}. Choose cloudflare, ngrok, tailscale, or custom."
        )


def stop_tunnel(process: "subprocess.Popen[str] | None") -> None:
    """Gracefully stop a tunnel process."""
    if process is None:
        return
    try:
        process.terminate()
        process.wait(timeout=5)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def start_tunnel_from_config(
    config: dict[str, Any], port: int
) -> tuple[str | None, "subprocess.Popen[str] | None"]:
    """Read ``tunnel`` section from *config* and start the configured tunnel.

    Returns ``(None, None)`` if no tunnel is configured.
    """
    tunnel_cfg = config.get("tunnel") or {}
    if not isinstance(tunnel_cfg, dict):
        return None, None
    ttype = (tunnel_cfg.get("type") or "").strip()
    if not ttype:
        return None, None
    extra = list(tunnel_cfg.get("args") or [])
    command = tunnel_cfg.get("command")
    try:
        url, proc = start_tunnel(ttype, port, extra_args=extra, command=command)
        return url, proc
    except (TunnelError, FileNotFoundError, OSError) as exc:
        logger.warning("Tunnel start failed (%s): %s", ttype, exc)
        return None, None
    except Exception as exc:
        logger.warning("Tunnel start unexpected error (%s): %s", ttype, exc)
        return None, None
