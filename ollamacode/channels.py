"""Chat channel adapters for OllamaCode.

Routes messages from external chat platforms (Telegram, Discord) through the
agent loop.  Each channel runs in a daemon thread.  Sessions are tracked per
chat user via a lightweight in-memory message history (backed by sessions.py
for persistence).

Supported channels (configured in ``ollamacode.yaml`` under ``channels``):

  telegram:
    enabled: true
    bot_token: "secret:telegram_bot_token"   # or plain value
    allowed_user_ids: [123456789]            # optional allowlist

  discord:
    enabled: true
    bot_token: "secret:discord_bot_token"
    allowed_guild_ids: [123456789]           # optional guild allowlist
    allowed_user_ids: []                     # optional user allowlist
    command_prefix: "!"                      # default "!"

Bot tokens can reference the secrets store: ``"secret:<name>"`` is
transparently decrypted by ``resolve_secret()``.

Usage (from serve.py or standalone)::

    from ollamacode.channels import start_channels, stop_channels
    handles = start_channels(config, model, merged_config)
    ...
    stop_channels(handles)
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base channel interface
# ---------------------------------------------------------------------------


class BaseChannel:
    """Abstract base for a chat channel adapter."""

    name: str = "base"

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def send(self, text: str, channel_id: str | int) -> None:
        """Send a text message to a specific channel/user."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# In-process agent call (no MCP — channel adapters run lightweight)
# ---------------------------------------------------------------------------


def _run_agent(message: str, history: list[dict], model: str, config: dict) -> str:
    """Run the no-MCP agent loop and return the reply."""
    from .agent import run_agent_loop_no_mcp

    provider = None
    try:
        from .providers import get_provider

        pname = config.get("provider", "ollama")
        if pname != "ollama":
            provider = get_provider(config)
    except Exception:
        pass

    async def _go() -> str:
        return await run_agent_loop_no_mcp(
            model,
            message,
            message_history=history,
            provider=provider,
        )

    return asyncio.run(_go())


# ---------------------------------------------------------------------------
# Per-user session history (lightweight in-memory, up to 40 messages)
# ---------------------------------------------------------------------------

_SESSION_MAX = 40
_user_histories: dict[str, list[dict]] = {}


def _get_history(user_key: str) -> list[dict]:
    return _user_histories.setdefault(user_key, [])


def _append_history(user_key: str, role: str, content: str) -> None:
    hist = _get_history(user_key)
    hist.append({"role": role, "content": content})
    if len(hist) > _SESSION_MAX:
        del hist[: len(hist) - _SESSION_MAX]


# ---------------------------------------------------------------------------
# Allowlist helpers
# ---------------------------------------------------------------------------


def _is_allowed_user(user_id: int, allowed: list[int] | None) -> bool:
    if not allowed:
        return True  # No allowlist = open
    return int(user_id) in [int(x) for x in allowed]


def _is_allowed_guild(guild_id: int | None, allowed: list[int] | None) -> bool:
    if not allowed:
        return True
    if guild_id is None:
        return False
    return int(guild_id) in [int(x) for x in allowed]


# ---------------------------------------------------------------------------
# Telegram channel
# ---------------------------------------------------------------------------


class TelegramChannel(BaseChannel):
    """Webhook-less long-polling Telegram bot adapter.

    Uses the pure-Python Telegram Bot API (urllib) — no external library
    required.  Long-polls ``getUpdates`` in a background thread.
    """

    name = "telegram"

    def __init__(
        self,
        bot_token: str,
        allowed_user_ids: list[int] | None,
        model: str,
        config: dict,
    ) -> None:
        self.bot_token = bot_token
        self.allowed_user_ids = allowed_user_ids
        self.model = model
        self.config = config
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._offset = 0
        self._base_url = f"https://api.telegram.org/bot{bot_token}"

    def _api(self, method: str, params: dict | None = None) -> dict:
        import json
        import urllib.parse
        import urllib.request

        url = f"{self._base_url}/{method}"
        if params:
            url += "?" + urllib.parse.urlencode(
                {k: v for k, v in params.items() if v is not None}
            )
        with urllib.request.urlopen(url, timeout=35) as resp:
            return json.loads(resp.read())

    def send(self, text: str, channel_id: str | int) -> None:
        import json
        import urllib.request

        payload = json.dumps({"chat_id": channel_id, "text": text[:4096]}).encode()
        req = urllib.request.Request(
            f"{self._base_url}/sendMessage",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10):
                pass
        except Exception as exc:
            logger.warning("Telegram send failed: %s", exc)

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                data = self._api(
                    "getUpdates", {"offset": self._offset, "timeout": 30, "limit": 10}
                )
                for update in data.get("result", []):
                    self._offset = update["update_id"] + 1
                    self._handle_update(update)
            except Exception as exc:
                if not self._stop.is_set():
                    logger.debug("Telegram poll error: %s", exc)
                    self._stop.wait(timeout=5)

    def _handle_update(self, update: dict) -> None:
        msg = update.get("message") or update.get("channel_post")
        if not msg:
            return
        text = msg.get("text", "").strip()
        if not text:
            return
        from_user = msg.get("from") or {}
        user_id = from_user.get("id")
        chat_id = msg.get("chat", {}).get("id")
        if not chat_id:
            return
        if not _is_allowed_user(user_id, self.allowed_user_ids):
            self.send("Sorry, you are not authorized to use this bot.", chat_id)
            return
        user_key = f"telegram:{user_id}"
        history = _get_history(user_key)
        _append_history(user_key, "user", text)
        try:
            reply = _run_agent(text, history[:-1], self.model, self.config)
        except Exception as exc:
            reply = f"Error: {exc}"
        _append_history(user_key, "assistant", reply)
        # Split long replies
        for i in range(0, len(reply), 4000):
            self.send(reply[i : i + 4000], chat_id)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="ollamacode-telegram"
        )
        self._thread.start()
        logger.info("Telegram channel started.")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Telegram channel stopped.")


# ---------------------------------------------------------------------------
# Discord channel
# ---------------------------------------------------------------------------


class DiscordChannel(BaseChannel):
    """Gateway-based Discord bot adapter using discord.py (optional dependency).

    If ``discord.py`` is not installed, the channel logs a warning and does nothing.
    """

    name = "discord"

    def __init__(
        self,
        bot_token: str,
        allowed_guild_ids: list[int] | None,
        allowed_user_ids: list[int] | None,
        command_prefix: str,
        model: str,
        config: dict,
    ) -> None:
        self.bot_token = bot_token
        self.allowed_guild_ids = allowed_guild_ids
        self.allowed_user_ids = allowed_user_ids
        self.command_prefix = command_prefix
        self.model = model
        self.config = config
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        try:
            import discord  # noqa: F401
        except ImportError:
            logger.warning(
                "discord.py not installed; Discord channel disabled. "
                "Install with: pip install discord.py"
            )
            return
        self._thread = threading.Thread(
            target=self._run_bot, daemon=True, name="ollamacode-discord"
        )
        self._thread.start()
        logger.info("Discord channel started.")

    def _run_bot(self) -> None:
        import discord

        intents = discord.Intents.default()
        intents.message_content = True
        client = discord.Client(intents=intents)

        @client.event
        async def on_ready() -> None:
            logger.info("Discord bot logged in as %s", client.user)

        @client.event
        async def on_message(message: discord.Message) -> None:
            if message.author.bot:
                return
            guild_id = message.guild.id if message.guild else None
            if not _is_allowed_guild(guild_id, self.allowed_guild_ids):
                return
            if not _is_allowed_user(message.author.id, self.allowed_user_ids):
                await message.channel.send(
                    "Sorry, you are not authorized to use this bot."
                )
                return
            content = message.content.strip()
            # React to command prefix or direct mentions
            if self.command_prefix and not content.startswith(self.command_prefix):
                # Also respond to DMs without prefix
                if message.guild is not None:
                    return
            elif self.command_prefix and content.startswith(self.command_prefix):
                content = content[len(self.command_prefix) :].strip()

            if not content:
                return

            user_key = f"discord:{message.author.id}"
            history = _get_history(user_key)
            _append_history(user_key, "user", content)
            try:
                reply = await asyncio.get_event_loop().run_in_executor(
                    None, _run_agent, content, history[:-1], self.model, self.config
                )
            except Exception as exc:
                reply = f"Error: {exc}"
            _append_history(user_key, "assistant", reply)
            # Discord max message length = 2000
            for i in range(0, len(reply), 1900):
                await message.channel.send(reply[i : i + 1900])

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(client.start(self.bot_token))
        except Exception as exc:
            if "Improper token" in str(exc) or "401" in str(exc):
                logger.error("Discord: invalid bot token")
            else:
                logger.debug("Discord bot stopped: %s", exc)
        finally:
            self._loop.close()

    def stop(self) -> None:
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self._loop.shutdown_asyncgens(), self._loop
            )
        logger.info("Discord channel stopped.")

    def send(self, text: str, channel_id: str | int) -> None:
        """Not directly usable from outside the bot's async context."""
        logger.debug("Discord.send() called outside bot context; ignoring.")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def start_channels(
    config: dict[str, Any], model: str, merged_config: dict[str, Any]
) -> list[BaseChannel]:
    """Instantiate and start all configured channels from *config*.

    Returns the list of started channel handles (pass to :func:`stop_channels`).
    """
    channels_cfg = config.get("channels") or {}
    handles: list[BaseChannel] = []

    # Resolve secrets in bot tokens
    from .secrets import resolve_secret

    # --- Telegram ---
    tg_cfg = channels_cfg.get("telegram") or {}
    if tg_cfg.get("enabled"):
        try:
            token = resolve_secret(str(tg_cfg.get("bot_token") or ""))
            if not token:
                logger.warning("Telegram channel: bot_token is empty")
            else:
                allowed = [int(x) for x in (tg_cfg.get("allowed_user_ids") or [])]
                ch = TelegramChannel(token, allowed or None, model, merged_config)
                ch.start()
                handles.append(ch)
        except Exception as exc:
            logger.warning("Telegram channel start failed: %s", exc)

    # --- Discord ---
    dc_cfg = channels_cfg.get("discord") or {}
    if dc_cfg.get("enabled"):
        try:
            token = resolve_secret(str(dc_cfg.get("bot_token") or ""))
            if not token:
                logger.warning("Discord channel: bot_token is empty")
            else:
                allowed_guilds = [
                    int(x) for x in (dc_cfg.get("allowed_guild_ids") or [])
                ]
                allowed_users = [int(x) for x in (dc_cfg.get("allowed_user_ids") or [])]
                prefix = str(dc_cfg.get("command_prefix") or "!")
                ch = DiscordChannel(
                    token,
                    allowed_guilds or None,
                    allowed_users or None,
                    prefix,
                    model,
                    merged_config,
                )
                ch.start()
                handles.append(ch)
        except Exception as exc:
            logger.warning("Discord channel start failed: %s", exc)

    return handles


def stop_channels(handles: list[BaseChannel]) -> None:
    """Stop all channel handles gracefully."""
    for ch in handles:
        try:
            ch.stop()
        except Exception as exc:
            logger.debug("Error stopping channel %s: %s", ch.name, exc)
