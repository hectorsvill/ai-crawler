"""
crawler/robots.py — robots.txt parser and per-domain rule enforcer.

Fetches and caches robots.txt for each domain. Provides a simple
`is_allowed(url)` check used by the crawl engine before fetching.
"""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import aiohttp

logger = logging.getLogger(__name__)

# How long to cache parsed robots.txt per domain (seconds)
ROBOTS_CACHE_TTL = 3600

# User-agent string used when fetching robots.txt
ROBOTS_FETCH_UA = "AICrawlerBot/1.0 (+https://github.com/ai-crawler)"


class RobotsCache:
    """
    Async, per-domain robots.txt cache.

    Fetches robots.txt once per domain and caches the parsed result.
    Thread-safe for concurrent async use via asyncio.Lock per domain.
    """

    def __init__(self) -> None:
        self._cache: dict[str, tuple[RobotFileParser, float]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, domain: str) -> asyncio.Lock:
        if domain not in self._locks:
            self._locks[domain] = asyncio.Lock()
        return self._locks[domain]

    async def _fetch_robots_txt(self, base_url: str) -> str:
        """Fetch raw robots.txt content from *base_url*/robots.txt."""
        robots_url = urljoin(base_url, "/robots.txt")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    robots_url,
                    headers={"User-Agent": ROBOTS_FETCH_UA},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        return await resp.text(errors="replace")
                    if resp.status == 404:
                        logger.debug("robots.txt not found (404) for %s — treating as allow-all", base_url)
                    else:
                        logger.debug(
                            "robots.txt fetch returned HTTP %d for %s — treating as allow-all",
                            resp.status,
                            base_url,
                        )
                    return ""
        except aiohttp.ClientError as exc:
            logger.debug("Network error fetching robots.txt for %s: %s", base_url, exc)
            return ""
        except Exception as exc:
            logger.warning("Unexpected error fetching robots.txt for %s: %s", base_url, exc)
            return ""

    async def get_parser(self, url: str) -> RobotFileParser:
        """Return a cached (or freshly fetched) RobotFileParser for *url*'s domain."""
        import time

        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"

        async with self._lock_for(domain):
            cached = self._cache.get(domain)
            if cached and (time.time() - cached[1]) < ROBOTS_CACHE_TTL:
                return cached[0]

            content = await self._fetch_robots_txt(domain)
            parser = RobotFileParser()
            parser.parse(content.splitlines())
            self._cache[domain] = (parser, time.time())
            return parser

    async def is_allowed(self, url: str, user_agent: str = "*") -> bool:
        """
        Return True if *user_agent* is permitted to fetch *url* per robots.txt.

        Falls back to True on any error (fail-open strategy so we don't block
        crawls due to network issues).
        """
        try:
            parser = await self.get_parser(url)
            return parser.can_fetch(user_agent, url)
        except Exception as exc:
            logger.warning("robots.txt check failed for %s: %s", url, exc)
            return True

    def clear(self) -> None:
        """Evict all cached entries (useful in tests)."""
        self._cache.clear()
        self._locks.clear()


# Module-level singleton
_robots_cache = RobotsCache()


async def is_allowed(url: str, user_agent: str = "*") -> bool:
    """Convenience wrapper around the module-level RobotsCache."""
    return await _robots_cache.is_allowed(url, user_agent)
