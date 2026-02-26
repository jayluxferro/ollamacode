"""Interactive setup wizard for OllamaCode first-time configuration.

Run with: ollamacode setup   or   ollamacode init --interactive

Steps:
  1  Workspace: select/confirm working directory
  2  AI Provider: pick provider, enter base URL + API key, live connection test
  3  Channels: optionally configure Discord/Telegram (skip-able)
  4  Tunnel: optionally configure Cloudflare/ngrok/Tailscale (skip-able)
  5  Tool/sandbox mode: Sovereign (full) or Supervised
  6  Personalize: agent name, response style, timezone
  7  Scaffold: write ollamacode.yaml, create .ollamacode/skills/ skeleton

Target: default path < 60 seconds.
"""

from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Minimal Rich helpers (falls back to plain print if rich not installed)
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt

    _console = Console()

    def _print(msg: str, style: str = "") -> None:
        _console.print(msg, style=style)

    def _ask(prompt: str, default: str = "", password: bool = False) -> str:
        if password:
            return getpass.getpass(f"{prompt}: ") or default
        return Prompt.ask(prompt, default=default) if default else Prompt.ask(prompt)

    def _confirm(prompt: str, default: bool = True) -> bool:
        return Confirm.ask(prompt, default=default)

    def _header(title: str) -> None:
        _console.print(Panel(f"[bold]{title}[/bold]", expand=False))

except ImportError:
    def _print(msg: str, style: str = "") -> None:  # type: ignore[misc]
        print(msg)

    def _ask(prompt: str, default: str = "", password: bool = False) -> str:  # type: ignore[misc]
        if password:
            return getpass.getpass(f"{prompt}: ") or default
        val = input(f"{prompt} [{default}]: ").strip() if default else input(f"{prompt}: ").strip()
        return val or default

    def _confirm(prompt: str, default: bool = True) -> bool:  # type: ignore[misc]
        hint = "Y/n" if default else "y/N"
        val = input(f"{prompt} [{hint}]: ").strip().lower()
        if not val:
            return default
        return val in ("y", "yes")

    def _header(title: str) -> None:  # type: ignore[misc]
        sep = "=" * (len(title) + 4)
        print(f"\n{sep}\n  {title}\n{sep}")


# ---------------------------------------------------------------------------
# Provider presets
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "1": ("ollama", "Ollama (local, default)", None),
    "2": ("openai", "OpenAI", "https://api.openai.com/v1"),
    "3": ("anthropic", "Anthropic", None),
    "4": ("groq", "Groq", "https://api.groq.com/openai/v1"),
    "5": ("deepseek", "DeepSeek", "https://api.deepseek.com"),
    "6": ("openrouter", "OpenRouter", "https://openrouter.ai/api/v1"),
    "7": ("custom", "Custom / Self-hosted OpenAI-compatible API", None),
}

_DEFAULT_MODELS = {
    "ollama": "qwen2.5-coder:7b",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "groq": "llama-3.3-70b-versatile",
    "deepseek": "deepseek-chat",
    "openrouter": "mistralai/mistral-7b-instruct",
    "custom": "my-model",
}

# ---------------------------------------------------------------------------
# Wizard steps
# ---------------------------------------------------------------------------


def _step1_workspace() -> str:
    _header("Step 1 — Workspace")
    _print("Where is your project?  (Press Enter to use the current directory)")
    cwd = os.getcwd()
    workspace = _ask("Workspace path", default=cwd)
    workspace = os.path.abspath(workspace)
    if not os.path.isdir(workspace):
        _print(f"  Creating directory: {workspace}")
        os.makedirs(workspace, exist_ok=True)
    _print(f"  Workspace: [bold]{workspace}[/bold]", style="green")
    return workspace


def _step2_provider() -> tuple[str, str, str, str]:
    """Returns (provider, base_url, api_key, model)."""
    _header("Step 2 — AI Provider")
    _print("Choose a provider:")
    for k, (pname, label, _) in _PROVIDERS.items():
        _print(f"  {k}. {label}")
    choice = _ask("Choice", default="1").strip()
    if choice not in _PROVIDERS:
        _print("Invalid choice — defaulting to Ollama.")
        choice = "1"

    provider, label, default_url = _PROVIDERS[choice]

    if provider == "ollama":
        base_url = _ask("Ollama base URL", default="http://localhost:11434")
        api_key = ""
        model = _ask("Model name", default=_DEFAULT_MODELS["ollama"])
    elif provider == "custom":
        base_url = _ask("API base URL (e.g. https://my-host/v1)", default="")
        api_key = _ask("API key", default="", password=True)
        model = _ask("Model name", default="my-model")
    else:
        base_url = default_url or ""
        api_key = _ask(f"{label} API key", default="", password=True)
        model = _ask("Model name", default=_DEFAULT_MODELS.get(provider, ""))

    # Save API key to secrets store if non-empty
    secret_ref = ""
    if api_key:
        try:
            from .secrets import set_secret

            secret_name = f"{provider}_api_key"
            set_secret(secret_name, api_key)
            secret_ref = f"secret:{secret_name}"
            _print(f"  API key saved to secrets store as '{secret_name}'.", style="green")
        except Exception as exc:
            _print(f"  Warning: could not save to secrets store: {exc}")
            secret_ref = api_key

    # Live connectivity test
    _print("  Testing connection…")
    try:
        from .providers import get_provider

        test_config: dict[str, Any] = {
            "provider": provider,
            "base_url": base_url or None,
            "api_key": api_key or None,
        }
        p = get_provider(test_config)
        ok, msg = p.health_check()
        if ok:
            _print(f"  Connection OK: {msg}", style="green")
        else:
            _print(f"  Warning: {msg}", style="yellow")
    except Exception as exc:
        _print(f"  Warning: connectivity check failed: {exc}", style="yellow")

    return provider, base_url, secret_ref, model


def _step3_channels() -> dict[str, Any]:
    _header("Step 3 — Chat Channels (optional)")
    channels: dict[str, Any] = {}
    if not _confirm("Configure Telegram bot?", default=False):
        pass
    else:
        token = _ask("Telegram bot token (from @BotFather)", password=True)
        if token:
            try:
                from .secrets import set_secret

                set_secret("telegram_bot_token", token)
                token_ref = "secret:telegram_bot_token"
            except Exception:
                token_ref = token
            allowed_raw = _ask("Allowed Telegram user IDs (comma-separated, or blank for all)", default="")
            allowed_ids = [int(x.strip()) for x in allowed_raw.split(",") if x.strip().isdigit()]
            channels["telegram"] = {
                "enabled": True,
                "bot_token": token_ref,
                "allowed_user_ids": allowed_ids,
            }
            _print("  Telegram channel configured.", style="green")

    if not _confirm("Configure Discord bot?", default=False):
        pass
    else:
        token = _ask("Discord bot token", password=True)
        if token:
            try:
                from .secrets import set_secret

                set_secret("discord_bot_token", token)
                token_ref = "secret:discord_bot_token"
            except Exception:
                token_ref = token
            guild_raw = _ask("Allowed guild IDs (comma-separated, or blank for all)", default="")
            guild_ids = [int(x.strip()) for x in guild_raw.split(",") if x.strip().isdigit()]
            channels["discord"] = {
                "enabled": True,
                "bot_token": token_ref,
                "allowed_guild_ids": guild_ids,
                "command_prefix": _ask("Command prefix", default="!"),
            }
            _print("  Discord channel configured.", style="green")

    return channels


def _step4_tunnel() -> dict[str, Any]:
    _header("Step 4 — Tunnel (optional)")
    if not _confirm("Configure a public tunnel for serve mode?", default=False):
        return {}
    _print("Tunnel type:")
    _print("  1. Cloudflare (cloudflared)")
    _print("  2. ngrok")
    _print("  3. Tailscale funnel")
    _print("  4. Custom command")
    choice = _ask("Choice", default="1").strip()
    tmap = {"1": "cloudflare", "2": "ngrok", "3": "tailscale", "4": "custom"}
    ttype = tmap.get(choice, "cloudflare")
    tunnel_cfg: dict[str, Any] = {"type": ttype}
    if ttype == "custom":
        cmd = _ask("Command template (use {port} placeholder)", default="my-tunnel http {port}")
        tunnel_cfg["command"] = cmd
    _print(f"  Tunnel: {ttype}", style="green")
    return tunnel_cfg


def _step5_sandbox() -> str:
    _header("Step 5 — Tool & Sandbox Mode")
    _print("  supervised  (default) — agent can read/write workspace, run allowlisted commands")
    _print("  full                  — unrestricted (Sovereign mode)")
    _print("  readonly              — agent can only read files; no commands")
    choice = _ask("Sandbox level", default="supervised").strip().lower()
    if choice not in ("supervised", "full", "readonly"):
        choice = "supervised"
    _print(f"  Sandbox: {choice}", style="green")
    return choice


def _step6_personalize() -> dict[str, Any]:
    _header("Step 6 — Personalize")
    agent_name = _ask("Agent name (shown in TUI header)", default="OllamaCode")
    timezone = _ask("Timezone (for cron tasks, e.g. UTC, America/New_York)", default="UTC")
    style_opts = {"1": "concise", "2": "detailed", "3": "technical"}
    _print("Response style:  1. Concise  2. Detailed  3. Technical")
    style_choice = _ask("Choice", default="1")
    response_style = style_opts.get(style_choice, "concise")
    return {"agent_name": agent_name, "timezone": timezone, "response_style": response_style}


def _step7_scaffold(
    workspace: str,
    provider: str,
    base_url: str,
    api_key_ref: str,
    model: str,
    channels: dict,
    tunnel: dict,
    sandbox: str,
    personalize: dict,
) -> str:
    """Write ollamacode.yaml and create .ollamacode/skills/ skeleton."""
    _header("Step 7 — Writing Configuration")

    # Build YAML content (manual construction to avoid extra dep)
    lines: list[str] = [
        "# ollamacode.yaml — generated by setup wizard",
        f"# Agent: {personalize.get('agent_name', 'OllamaCode')}",
        "",
        f"model: {model}",
        f"provider: {provider}",
    ]
    if base_url:
        lines.append(f"base_url: {base_url}")
    if api_key_ref:
        lines.append(f"api_key: \"{api_key_ref}\"")

    lines += [
        "",
        f"sandbox_level: {sandbox}",
        "",
        "# System prompt extras",
        f"system_prompt_extra: \"You are {personalize.get('agent_name', 'OllamaCode')}, a helpful coding assistant.\"",
        "",
        "# Built-in MCP servers (fs, terminal, codebase, git, tools)",
        "include_builtin_servers: true",
        "",
    ]

    if tunnel:
        lines.append("tunnel:")
        lines.append(f"  type: {tunnel.get('type', 'cloudflare')}")
        if tunnel.get("command"):
            lines.append(f"  command: \"{tunnel['command']}\"")
        lines.append("")

    if channels:
        lines.append("channels:")
        tg = channels.get("telegram")
        if tg:
            lines += [
                "  telegram:",
                f"    enabled: {str(tg.get('enabled', True)).lower()}",
                f"    bot_token: \"{tg.get('bot_token', '')}\"",
            ]
            if tg.get("allowed_user_ids"):
                ids = ", ".join(str(x) for x in tg["allowed_user_ids"])
                lines.append(f"    allowed_user_ids: [{ids}]")
        dc = channels.get("discord")
        if dc:
            lines += [
                "  discord:",
                f"    enabled: {str(dc.get('enabled', True)).lower()}",
                f"    bot_token: \"{dc.get('bot_token', '')}\"",
                f"    command_prefix: \"{dc.get('command_prefix', '!')}\"",
            ]
            if dc.get("allowed_guild_ids"):
                ids = ", ".join(str(x) for x in dc["allowed_guild_ids"])
                lines.append(f"    allowed_guild_ids: [{ids}]")
        lines.append("")

    lines += [
        "# Scheduled tasks (see also: HEARTBEAT.md)",
        "# scheduled_tasks: []",
        "",
        "# Rate limiting for serve mode",
        "serve:",
        "  rate_limit_rpm: 60",
        "",
    ]

    yaml_content = "\n".join(lines) + "\n"
    yaml_path = Path(workspace) / "ollamacode.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    _print(f"  Written: {yaml_path}", style="green")

    # .ollamacode/skills/ skeleton
    skills_dir = Path(workspace) / ".ollamacode" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    example_skill = skills_dir / "example.md"
    if not example_skill.exists():
        example_skill.write_text(
            "# Example skill\n\nDescribe a skill or saved instruction here.\n"
            "OllamaCode will inject this into the system prompt automatically.\n",
            encoding="utf-8",
        )
    _print(f"  Skills directory: {skills_dir}", style="green")

    return str(yaml_path)


# ---------------------------------------------------------------------------
# Main wizard entry point
# ---------------------------------------------------------------------------

def run_setup_wizard(workspace_override: str | None = None) -> None:
    """Run the interactive setup wizard."""
    _print("\n[bold cyan]Welcome to OllamaCode Setup Wizard[/bold cyan]\n", style="")
    _print("This wizard will configure your OllamaCode installation.")
    _print("Press Ctrl+C at any time to abort.\n")

    import time

    start = time.monotonic()

    try:
        workspace = workspace_override or _step1_workspace()
        provider, base_url, api_key_ref, model = _step2_provider()
        channels = _step3_channels()
        tunnel = _step4_tunnel()
        sandbox = _step5_sandbox()
        personalize = _step6_personalize()
        yaml_path = _step7_scaffold(
            workspace, provider, base_url, api_key_ref, model,
            channels, tunnel, sandbox, personalize,
        )
    except KeyboardInterrupt:
        _print("\n\nSetup aborted.", style="yellow")
        return

    elapsed = time.monotonic() - start
    _header("Setup Complete!")
    _print(f"  Provider:    {provider}")
    _print(f"  Model:       {model}")
    _print(f"  Sandbox:     {sandbox}")
    _print(f"  Config:      {yaml_path}")
    if channels:
        _print(f"  Channels:    {', '.join(channels.keys())}")
    if tunnel:
        _print(f"  Tunnel:      {tunnel.get('type', '?')}")
    _print(f"\n  Completed in {elapsed:.1f}s")
    _print("\nStart coding: [bold]ollamacode[/bold]", style="")
    _print("Or serve:     [bold]ollamacode serve[/bold]", style="")
