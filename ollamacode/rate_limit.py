"""Sliding-window rate limiter for OllamaCode's HTTP serve mode.

Provides per-client IP rate limiting with configurable requests-per-minute
and an optional daily token budget.  Both limits track state in memory; state
is not persisted across restarts.

Usage::

    from ollamacode.rate_limit import RateLimiter

    limiter = RateLimiter(requests_per_minute=60, tokens_per_day=100_000)

    allowed, retry_after = limiter.check("192.168.1.1")
    if not allowed:
        # Return 429 with Retry-After: <retry_after> header
        ...

    # After a successful response, record token usage:
    limiter.record_tokens("192.168.1.1", tokens_used=512)
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque


class RateLimiter:
    """Thread-safe sliding-window rate limiter.

    Parameters
    ----------
    requests_per_minute:
        Maximum number of requests allowed per client per minute (0 = unlimited).
    tokens_per_day:
        Maximum tokens consumed per client per calendar day (0 = unlimited).
    """

    def __init__(
        self,
        requests_per_minute: int = 0,
        tokens_per_day: int = 0,
    ) -> None:
        self.requests_per_minute = requests_per_minute
        self.tokens_per_day = tokens_per_day
        self._lock = threading.Lock()
        # Per-client sliding window of request timestamps (epoch float).
        self._req_windows: dict[str, Deque[float]] = {}
        # Per-client (day_str, tokens_used) token budget tracking.
        self._token_budgets: dict[str, tuple[str, int]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, client_key: str) -> tuple[bool, int]:
        """Check whether *client_key* is within rate limits.

        Returns ``(allowed, retry_after_seconds)``.
        If ``allowed`` is ``False``, ``retry_after_seconds`` is the minimum
        number of seconds the client should wait before retrying.
        """
        now = time.monotonic()
        wall_now = time.time()

        with self._lock:
            # --- requests-per-minute check ---
            if self.requests_per_minute > 0:
                window: Deque[float] = self._req_windows.setdefault(client_key, deque())
                cutoff = now - 60.0
                # Purge timestamps older than the window.
                while window and window[0] < cutoff:
                    window.popleft()
                if len(window) >= self.requests_per_minute:
                    # Oldest request in window; wait until it ages out.
                    oldest = window[0]
                    retry_after = max(1, int(oldest - cutoff) + 1)
                    return False, retry_after
                window.append(now)

            # --- tokens-per-day check ---
            if self.tokens_per_day > 0:
                import datetime

                today = datetime.date.today().isoformat()
                day_str, used = self._token_budgets.get(client_key, (today, 0))
                if day_str != today:
                    # New day — reset budget.
                    used = 0
                if used >= self.tokens_per_day:
                    # Calculate seconds until midnight UTC.
                    import datetime as dt

                    now_dt = dt.datetime.now(dt.timezone.utc)
                    midnight = (now_dt + dt.timedelta(days=1)).replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                    retry_after = max(1, int((midnight - now_dt).total_seconds()))
                    return False, retry_after
                self._token_budgets[client_key] = (today, used)

        return True, 0

    def record_tokens(self, client_key: str, tokens_used: int) -> None:
        """Record *tokens_used* against *client_key*'s daily budget."""
        if self.tokens_per_day <= 0 or tokens_used <= 0:
            return
        import datetime

        today = datetime.date.today().isoformat()
        with self._lock:
            day_str, used = self._token_budgets.get(client_key, (today, 0))
            if day_str != today:
                used = 0
            self._token_budgets[client_key] = (today, used + tokens_used)

    def is_active(self) -> bool:
        """Return True if at least one limit is configured."""
        return self.requests_per_minute > 0 or self.tokens_per_day > 0
