"""
crawler/respectful.py — Rate limiter, rotating user-agents, domain allow/deny,
and depth enforcement.

All limits are read from CrawlConfig and can be adjusted at runtime.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import random
import time
from collections import defaultdict
from urllib.parse import urlparse

from config import CrawlConfig

logger = logging.getLogger(__name__)


# ── Domain matching helper ────────────────────────────────────────────────────

def _domain_matches(domain: str, pattern: str) -> bool:
    """
    Match *domain* against *pattern*, supporting glob wildcards.

    Extends ``fnmatch`` so that ``*.example.com`` also matches ``example.com``
    itself (the parent domain), which is the common expectation for allowlists.
    """
    if fnmatch.fnmatch(domain, pattern):
        return True
    # *.foo.com should also match foo.com
    if pattern.startswith("*."):
        parent_pattern = pattern[2:]  # strip leading "*."
        if domain == parent_pattern or fnmatch.fnmatch(domain, parent_pattern):
            return True
    return False


# ── Rate limiter ──────────────────────────────────────────────────────────────

class DomainRateLimiter:
    """
    Token-bucket rate limiter that enforces a per-domain request rate.

    Usage:
        limiter = DomainRateLimiter(rate=2.0)
        await limiter.acquire("https://example.com/page")
    """

    def __init__(self, rate: float = 2.0, delay_range: tuple[float, float] = (1.0, 3.0)) -> None:
        self.rate = rate  # requests per second
        self.delay_range = delay_range
        self._last_request: dict[str, float] = defaultdict(float)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def _domain(self, url: str) -> str:
        return urlparse(url).netloc

    async def acquire(self, url: str) -> None:
        """
        Wait until the domain rate limit allows another request,
        then add a random jitter delay from delay_range.
        """
        domain = self._domain(url)
        async with self._locks[domain]:
            now = time.monotonic()
            min_interval = 1.0 / max(self.rate, 0.01)
            elapsed = now - self._last_request[domain]
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)

            # Random politeness delay
            jitter = random.uniform(*self.delay_range)
            await asyncio.sleep(jitter)

            self._last_request[domain] = time.monotonic()


# ── User-agent rotation ────────────────────────────────────────────────────────

class UserAgentRotator:
    """Returns a random user-agent from a configured pool."""

    def __init__(self, agents: list[str]) -> None:
        self._agents = agents or [
            "Mozilla/5.0 (compatible; AICrawlerBot/1.0)"
        ]

    def get(self) -> str:
        return random.choice(self._agents)


# ── Domain filter ─────────────────────────────────────────────────────────────

class DomainFilter:
    """
    Enforces allow/deny domain rules.

    - If allowlist is non-empty, only listed domains (or glob patterns) pass.
    - Denylist always wins: a matching deny blocks even allowlisted URLs.
    """

    def __init__(self, allowlist: list[str], denylist: list[str]) -> None:
        self._allow = allowlist
        self._deny = denylist

    def _domain(self, url: str) -> str:
        return urlparse(url).netloc

    def is_allowed(self, url: str) -> bool:
        domain = self._domain(url)

        # Denylist takes priority
        for pattern in self._deny:
            if _domain_matches(domain, pattern):
                logger.debug("Domain %s blocked by denylist pattern: %s", domain, pattern)
                return False

        # If allowlist is set, URL must match at least one pattern
        if self._allow:
            for pattern in self._allow:
                if _domain_matches(domain, pattern):
                    return True
            logger.debug("Domain %s not in allowlist", domain)
            return False

        return True


# ── Playwright detector ────────────────────────────────────────────────────────

class PlaywrightDetector:
    """
    Decides whether a URL should be fetched via Playwright (JS rendering)
    based on configured domain patterns.
    """

    def __init__(self, patterns: list[str]) -> None:
        self._patterns = patterns

    def needs_js(self, url: str) -> bool:
        domain = urlparse(url).netloc
        for pattern in self._patterns:
            if fnmatch.fnmatch(domain, pattern) or fnmatch.fnmatch(url, pattern):
                return True
        return False


# ── Composite respectful-crawl guard ─────────────────────────────────────────

class RespectfulCrawler:
    """
    Combines rate limiting, user-agent rotation, domain filtering,
    depth enforcement, and robots.txt checks in one coordinator.
    """

    def __init__(self, config: CrawlConfig) -> None:
        self.config = config
        self.rate_limiter = DomainRateLimiter(
            rate=config.rate_limit_per_domain,
            delay_range=tuple(config.delay_range),  # type: ignore[arg-type]
        )
        self.ua_rotator = UserAgentRotator(config.user_agents)
        self.domain_filter = DomainFilter(config.domain_allowlist, config.domain_denylist)
        self.playwright_detector = PlaywrightDetector(config.use_playwright_for)

    async def check_and_wait(self, url: str, depth: int) -> tuple[bool, str]:
        """
        Perform all pre-fetch checks and wait for rate limit.

        Returns (allowed: bool, reason: str).
        reason is empty string if allowed.
        """
        # Depth check
        if depth > self.config.max_depth:
            return False, f"Depth {depth} exceeds max {self.config.max_depth}"

        # Domain filter
        if not self.domain_filter.is_allowed(url):
            return False, "Domain blocked by allow/deny list"

        # robots.txt check (imported here to avoid circular)
        from crawler.robots import is_allowed as robots_allowed
        if not await robots_allowed(url, self.ua_rotator.get()):
            return False, "Blocked by robots.txt"

        # Rate limit (awaits delay)
        await self.rate_limiter.acquire(url)
        return True, ""

    def current_user_agent(self) -> str:
        return self.ua_rotator.get()

    def needs_playwright(self, url: str) -> bool:
        return self.playwright_detector.needs_js(url)
