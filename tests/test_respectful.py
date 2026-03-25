"""Tests for crawler/respectful.py — domain filtering, rate limiting, Playwright detection."""

import asyncio
import time

import pytest

from crawler.respectful import DomainFilter, DomainRateLimiter, PlaywrightDetector, _domain_matches


# ── _domain_matches ───────────────────────────────────────────────────────────

class TestDomainMatches:
    def test_exact_match(self):
        assert _domain_matches("example.com", "example.com")

    def test_wildcard_subdomain(self):
        assert _domain_matches("sub.example.com", "*.example.com")

    def test_wildcard_matches_parent(self):
        # *.example.com should also match example.com itself
        assert _domain_matches("example.com", "*.example.com")

    def test_wildcard_does_not_match_different_domain(self):
        assert not _domain_matches("other.com", "*.example.com")

    def test_no_match(self):
        assert not _domain_matches("other.com", "example.com")


# ── DomainFilter ──────────────────────────────────────────────────────────────

class TestDomainFilter:
    def test_no_lists_allows_all(self):
        f = DomainFilter(allowlist=[], denylist=[])
        assert f.is_allowed("https://anything.com/page")

    def test_denylist_blocks(self):
        f = DomainFilter(allowlist=[], denylist=["ads.example.com"])
        assert not f.is_allowed("https://ads.example.com/banner")

    def test_denylist_does_not_block_other_domains(self):
        f = DomainFilter(allowlist=[], denylist=["ads.example.com"])
        assert f.is_allowed("https://example.com/page")

    def test_allowlist_permits_matching(self):
        f = DomainFilter(allowlist=["example.com"], denylist=[])
        assert f.is_allowed("https://example.com/page")

    def test_allowlist_blocks_non_matching(self):
        f = DomainFilter(allowlist=["example.com"], denylist=[])
        assert not f.is_allowed("https://other.com/page")

    def test_allowlist_wildcard(self):
        f = DomainFilter(allowlist=["*.example.com"], denylist=[])
        assert f.is_allowed("https://sub.example.com/")

    def test_allowlist_wildcard_also_allows_parent(self):
        f = DomainFilter(allowlist=["*.example.com"], denylist=[])
        assert f.is_allowed("https://example.com/")

    def test_deny_overrides_allow(self):
        f = DomainFilter(allowlist=["example.com"], denylist=["example.com"])
        assert not f.is_allowed("https://example.com/page")


# ── DomainRateLimiter ─────────────────────────────────────────────────────────

class TestDomainRateLimiter:
    @pytest.mark.asyncio
    async def test_acquires_without_error(self):
        limiter = DomainRateLimiter(rate=100.0, delay_range=(0.0, 0.0))
        # Should complete quickly
        await asyncio.wait_for(limiter.acquire("https://example.com/"), timeout=2.0)

    @pytest.mark.asyncio
    async def test_rate_enforced(self):
        """Two rapid requests to the same domain should be spaced at least 1/rate apart."""
        rate = 10.0  # 10 req/s → 0.1s min interval
        limiter = DomainRateLimiter(rate=rate, delay_range=(0.0, 0.0))

        t0 = time.monotonic()
        await limiter.acquire("https://example.com/a")
        t1 = time.monotonic()
        await limiter.acquire("https://example.com/b")
        t2 = time.monotonic()

        second_wait = t2 - t1
        # First call may have no wait; second should wait ~1/rate
        assert second_wait >= (1.0 / rate) * 0.8  # 20% tolerance

    @pytest.mark.asyncio
    async def test_different_domains_independent(self):
        """Requests to different domains should not be rate-limited by each other."""
        limiter = DomainRateLimiter(rate=1.0, delay_range=(0.0, 0.0))
        t0 = time.monotonic()
        await limiter.acquire("https://example.com/")
        await limiter.acquire("https://other.com/")
        elapsed = time.monotonic() - t0
        # Should complete in much less than 2 seconds (one per domain at 1 req/s)
        assert elapsed < 1.5


# ── PlaywrightDetector ────────────────────────────────────────────────────────

class TestPlaywrightDetector:
    def test_matches_configured_pattern(self):
        det = PlaywrightDetector(["*.spa-app.com"])
        assert det.needs_js("https://app.spa-app.com/dashboard")

    def test_no_match(self):
        det = PlaywrightDetector(["*.spa-app.com"])
        assert not det.needs_js("https://example.com/page")

    def test_empty_patterns_never_matches(self):
        det = PlaywrightDetector([])
        assert not det.needs_js("https://anything.com/")
