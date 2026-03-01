"""Unit tests for rate_limit.py — sliding window, edge cases."""

import time


from ollamacode.rate_limit import RateLimiter


class TestRateLimiterBasic:
    def test_unlimited_always_allows(self):
        limiter = RateLimiter(requests_per_minute=0, tokens_per_day=0)
        allowed, retry_after = limiter.check("client1")
        assert allowed is True
        assert retry_after == 0

    def test_is_active_when_rpm_set(self):
        limiter = RateLimiter(requests_per_minute=10)
        assert limiter.is_active() is True

    def test_is_active_when_tpd_set(self):
        limiter = RateLimiter(tokens_per_day=1000)
        assert limiter.is_active() is True

    def test_is_not_active_when_both_zero(self):
        limiter = RateLimiter(requests_per_minute=0, tokens_per_day=0)
        assert limiter.is_active() is False


class TestRateLimiterRPM:
    def test_allows_up_to_limit(self):
        limiter = RateLimiter(requests_per_minute=3)
        for _ in range(3):
            allowed, _ = limiter.check("client1")
            assert allowed is True

    def test_blocks_over_limit(self):
        limiter = RateLimiter(requests_per_minute=2)
        limiter.check("client1")
        limiter.check("client1")
        allowed, retry_after = limiter.check("client1")
        assert allowed is False
        assert retry_after >= 1

    def test_different_clients_independent(self):
        limiter = RateLimiter(requests_per_minute=1)
        allowed1, _ = limiter.check("client1")
        allowed2, _ = limiter.check("client2")
        assert allowed1 is True
        assert allowed2 is True
        # client1 is now blocked, but client2 is also blocked
        blocked1, _ = limiter.check("client1")
        assert blocked1 is False
        # client2 should be blocked independently
        blocked2, _ = limiter.check("client2")
        assert blocked2 is False

    def test_sliding_window_expiry(self):
        """After window elapses, requests should be allowed again."""
        limiter = RateLimiter(requests_per_minute=1)
        # First request succeeds
        allowed, _ = limiter.check("client1")
        assert allowed is True
        # Second is blocked
        allowed, _ = limiter.check("client1")
        assert allowed is False

        # Manually age the timestamps by manipulating the deque
        # We can't easily fast-forward monotonic time, so we directly
        # set an old timestamp in the window.
        with limiter._lock:
            window = limiter._req_windows["client1"]
            # Set all timestamps to 120 seconds ago (well outside 60s window)
            old_time = time.monotonic() - 120.0
            window.clear()
            window.append(old_time)

        # Now the next check should purge old and allow
        allowed, _ = limiter.check("client1")
        assert allowed is True


class TestRateLimiterTokens:
    def test_record_tokens_and_check(self):
        limiter = RateLimiter(tokens_per_day=100)
        allowed, _ = limiter.check("client1")
        assert allowed is True
        limiter.record_tokens("client1", 100)
        allowed, retry_after = limiter.check("client1")
        assert allowed is False
        assert retry_after >= 1

    def test_record_tokens_zero_ignored(self):
        limiter = RateLimiter(tokens_per_day=100)
        limiter.record_tokens("client1", 0)
        allowed, _ = limiter.check("client1")
        assert allowed is True

    def test_record_tokens_negative_ignored(self):
        limiter = RateLimiter(tokens_per_day=100)
        limiter.record_tokens("client1", -50)
        allowed, _ = limiter.check("client1")
        assert allowed is True

    def test_record_tokens_noop_when_limit_zero(self):
        limiter = RateLimiter(tokens_per_day=0)
        limiter.record_tokens("client1", 999999)
        allowed, _ = limiter.check("client1")
        assert allowed is True


class TestRateLimiterTOCTOU:
    def test_check_does_not_write_stale_token_count(self):
        """The TOCTOU fix: check() should NOT write back the token usage count.

        Only record_tokens() should update the used count. Calling check()
        after a budget reset should not restore a stale count.
        """
        limiter = RateLimiter(tokens_per_day=100)
        # Simulate usage
        limiter.record_tokens("client1", 50)
        # Check — this reads (today, 50) but should NOT write it back
        allowed, _ = limiter.check("client1")
        assert allowed is True

        # Manually inspect: the budget should still show 50 used, not re-written by check
        with limiter._lock:
            day_str, used = limiter._token_budgets.get("client1", ("", 0))
        assert used == 50

        # Record more tokens
        limiter.record_tokens("client1", 50)
        # Now at exactly the limit — next check should fail
        allowed, _ = limiter.check("client1")
        assert allowed is False
