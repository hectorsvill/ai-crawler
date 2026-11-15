"""radar/crawler.py — polite, deterministic topic crawler (no LLM required).

Pipeline per run:
  1. Load seeds (YAML). Each seed is an RSS/Atom feed or an HTML index page.
  2. Fetch each seed (robots.txt checked first) and collect candidate article URLs.
     Generic feeds (``require_match: true``) are pre-filtered by topic keywords on
     the feed title/description so off-topic articles are never fetched.
  3. Fetch candidates breadth-first up to --max-depth / --max-pages, extract
     title, canonical URL, published date, and Markdown text, tag by topic, and
     upsert into the radar DB.
  4. Write every encountered URL to ``run_urls`` (one row per URL per run).
     ``run_url_counts`` is rewritten from those rows. Rows are kept until a
     positive ``--keep-url-log-days`` prune. ``pages`` is unchanged.

Politeness: honest User-Agent, robots.txt (RFC 9309: 5xx/unreachable => disallow),
Crawl-delay honoured, per-domain minimum delay, small global concurrency,
response size cap.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import yaml

from radar.db import (
    DEFAULT_KEEP_URL_LOG_DAYS, NOT_FETCHED_REASONS, URL_STATUSES, RadarDB, utcnow,
)
from radar.extract import (
    content_hash, domain_of, extract_metadata, is_article_link, normalize_url, parse_feed,
)
from radar.topics import tag_text

logger = logging.getLogger("radar")

DEFAULT_UA = os.environ.get(
    "CRAWLER_RADAR_UA",
    "ai-crawler-radar/1.0 (+https://github.com/hectorsvill/ai-crawler)",
)
DEFAULT_SEEDS = Path(__file__).resolve().parent / "seeds.yaml"
MAX_BYTES = 3_000_000


@dataclass
class Seed:
    url: str
    kind: str = "html"              # html | feed
    topics: list[str] = field(default_factory=list)
    max_items: int = 10
    require_match: bool = False     # only keep items whose title/desc match a topic
    name: str | None = None
    feed_only: bool = False         # store feed entries without fetching article pages


@dataclass
class CrawlParams:
    max_pages: int = 200
    keep_url_log_days: int = DEFAULT_KEEP_URL_LOG_DAYS
    max_depth: int = 1
    concurrency: int = 2
    delay: float = 2.0              # min seconds between requests to the same domain
    timeout: float = 20.0
    recent_days: int | None = 7     # drop feed items older than this (None = keep all)
    max_crawl_delay: float = 30.0   # robots Crawl-delay above this => one request per domain
    user_agent: str = DEFAULT_UA
    topics: list[str] | None = None  # only seeds carrying one of these topics


def load_seeds(path: str | Path | None = None, topics: list[str] | None = None) -> list[Seed]:
    p = Path(path or DEFAULT_SEEDS)
    if not p.exists():
        raise FileNotFoundError(f"Seeds file not found: {p}")
    raw = yaml.safe_load(p.read_text()) or {}
    seeds = [Seed(**s) for s in (raw.get("seeds") or [])]
    if topics:
        wanted = set(topics)
        seeds = [s for s in seeds if wanted & set(s.topics) or (s.require_match and not s.topics)]
    return seeds


def robots_lines(text: str) -> list[str]:
    """robots.txt lines for ``RobotFileParser`` with blank/comment-only lines removed.

    RFC 9309 groups rules by ``User-agent`` lines; blank lines are not group
    separators. ``urllib.robotparser`` ends a group at a blank line, so rules after
    one (e.g. militarnyi.com's ``Crawl-delay: 100`` under ``User-agent: *``) were
    silently dropped. Without blank lines the parser starts a new group only at
    the next ``User-agent`` line that follows rules — the RFC behaviour.
    """
    return [ln for ln in text.splitlines() if ln.split("#", 1)[0].strip()]


class RobotsPolicy:
    """Per-origin robots.txt cache evaluated for our real User-Agent."""

    def __init__(self, session: Any, user_agent: str, timeout: float = 10.0) -> None:
        self.session, self.ua, self.timeout = session, user_agent, timeout
        self._cache: dict[str, RobotFileParser | None] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def _load(self, origin: str) -> RobotFileParser | None:
        """Return parser; None means 'disallow everything' (robots unreachable/5xx)."""
        import aiohttp
        rp = RobotFileParser()
        try:
            async with self.session.get(
                origin + "/robots.txt", headers={"User-Agent": self.ua},
                timeout=aiohttp.ClientTimeout(total=self.timeout), allow_redirects=True,
            ) as resp:
                if resp.status >= 500:
                    return None
                if resp.status >= 400:      # RFC 9309: 4xx => no restrictions
                    rp.parse([])
                    return rp
                rp.parse(robots_lines(await resp.text(errors="replace")))
                return rp
        except Exception as exc:  # network error => be conservative
            logger.warning("robots.txt unreachable for %s (%s) — treating as disallow", origin, exc)
            return None

    async def check(self, url: str) -> tuple[bool, float | None]:
        p = urlparse(url)
        origin = f"{p.scheme}://{p.netloc}"
        lock = self._locks.setdefault(origin, asyncio.Lock())
        async with lock:
            if origin not in self._cache:
                self._cache[origin] = await self._load(origin)
        rp = self._cache[origin]
        if rp is None:
            return False, None
        delay = rp.crawl_delay(self.ua)
        return rp.can_fetch(self.ua, url), (float(delay) if delay else None)


class DomainThrottle:
    def __init__(self, delay: float) -> None:
        self.delay = delay
        self._last: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def wait(self, url: str, crawl_delay: float | None = None) -> None:
        d = domain_of(url)
        lock = self._locks.setdefault(d, asyncio.Lock())
        async with lock:
            gap = max(self.delay, crawl_delay or 0) + random.uniform(0, 0.5)
            elapsed = time.monotonic() - self._last.get(d, 0)
            if elapsed < gap:
                await asyncio.sleep(gap - elapsed)
            self._last[d] = time.monotonic()


def plan(seeds: list[Seed], params: CrawlParams, db_path: str) -> dict[str, Any]:
    """Offline description of what a crawl would do (used by --dry-run)."""
    return {
        "dry_run": True,
        "db_path": db_path,
        "user_agent": params.user_agent,
        "limits": {
            "max_pages": params.max_pages, "max_depth": params.max_depth,
            "concurrency": params.concurrency, "per_domain_delay_s": params.delay,
            "recent_days": params.recent_days,
            "keep_url_log_days": params.keep_url_log_days,
        },
        "seeds": [
            {"url": s.url, "kind": s.kind, "topics": s.topics, "max_items": s.max_items,
             "require_match": s.require_match}
            for s in seeds
        ],
        "domains": sorted({domain_of(s.url) for s in seeds}),
        "max_possible_fetches": min(
            params.max_pages, len(seeds) + sum(s.max_items for s in seeds) * max(params.max_depth, 1)
        ),
        # A live run writes one row per URL it encounters. Dry-run does not open the DB.
        "url_log": {
            "table": "run_urls",
            "counts_table": "run_url_counts",
            "key": ["run_id", "url"],
            "statuses": list(URL_STATUSES),
            "not_fetched_reasons": list(NOT_FETCHED_REASONS),
            "keep_url_log_days": params.keep_url_log_days,
            "seed_urls": [s.url for s in seeds],
            "note": (
                "Every encountered URL is one run_urls row, including every "
                "not_fetched reason. run_url_counts mirrors those rows. "
                "keep_url_log_days 0 (the default) keeps rows forever; a positive "
                "value prunes run_urls of older runs at crawl start. "
                "runs and run_url_counts are never pruned. Dry-run does not open the DB."
            ),
        },
    }


class RadarCrawler:
    def __init__(self, db: RadarDB, params: CrawlParams) -> None:
        self.db, self.p = db, params
        self._requested: set[str] = set()  # normalized URLs already handled this run
        self._feed_only_done: set[str] = set()
        self.counters = {"fetched": 0, "stored": 0, "failed": 0, "skipped": 0, "feed_only": 0,
                         "inserted": 0, "updated": 0, "unchanged": 0, "duplicate_content": 0}
        self._http_fail: dict[str, int] = {}
        self._delay_skipped: set[str] = set()
        self._touched: set[str] = set()
        self.errors: list[dict[str, str]] = []
        self.skips: list[dict[str, str]] = []
        self.domains: set[str] = set()
        self._seen: set[str] = set()
        self._budget = params.max_pages

    async def _get(self, session: Any, url: str) -> tuple[int, str, str]:
        import aiohttp
        async with session.get(
            url, headers={"User-Agent": self.p.user_agent,
                          "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.5"},
            timeout=aiohttp.ClientTimeout(total=self.p.timeout), allow_redirects=True,
            max_redirects=5,
        ) as resp:
            # NB: StreamReader.read(n) returns whatever is buffered, so loop to EOF/cap.
            chunks, total = [], 0
            async for chunk in resp.content.iter_chunked(65536):
                chunks.append(chunk)
                total += len(chunk)
                if total >= MAX_BYTES:
                    break
            body = b"".join(chunks)[:MAX_BYTES]
            ctype = resp.headers.get("content-type", "")
            charset = resp.charset or "utf-8"
            return resp.status, ctype, body.decode(charset, errors="replace")

    def _log(self, run_id: int, url: str, *, status: str, seed: Seed, depth: int,
             discovered_from: str | None = None, reason: str | None = None,
             http_status: int | None = None, fetched: bool = False,
             overwrite: bool = True) -> None:
        """Append or upgrade one ``run_urls`` row. Does not touch politeness state."""
        if not url:
            return
        now = utcnow()
        self.db.log_url({
            "run_id": run_id, "url": url, "domain": domain_of(url),
            "source_seed": seed.url, "discovered_from": discovered_from, "depth": depth,
            "status": status, "reason": reason, "http_status": http_status,
            "discovered_at": now, "fetched_at": now if fetched else None,
        }, overwrite=overwrite)

    def _log_stored(self, run_id: int, url: str, action: str, seed: Seed, depth: int,
                    discovered_from: str | None, http_status: int | None) -> None:
        duplicate = action == "duplicate_content"
        self._log(
            run_id, url, status="duplicate" if duplicate else "stored",
            reason="duplicate_content" if duplicate else None, seed=seed, depth=depth,
            discovered_from=discovered_from, http_status=http_status, fetched=True,
        )

    def _mark_seen(self, url: str) -> None:
        if not url:
            return
        self._seen.add(url)
        self._seen.add(normalize_url(url))

    async def _fetch(self, session: Any, robots: RobotsPolicy, throttle: DomainThrottle,
                     url: str, run_id: int, seed: Seed, depth: int,
                     discovered_from: str | None) -> tuple[int, str, str] | None:
        # One outcome per URL per run. A second encounter (the same seed twice,
        # or a seed that is also a discovered link) does not fetch or count again,
        # so the row, run_url_counts, and the run counters stay aligned.
        key = normalize_url(url)
        if key in self._requested:
            return None
        self._requested.add(key)
        if self._budget <= 0:
            self._log(run_id, url, status="not_fetched", reason="budget", seed=seed,
                      depth=depth, discovered_from=discovered_from, overwrite=False)
            return None
        allowed, crawl_delay = await robots.check(url)
        if not allowed:
            self.counters["skipped"] += 1
            self.skips.append({"url": url, "reason": "robots.txt"})
            self._log(run_id, url, status="skipped", reason="robots.txt", seed=seed,
                      depth=depth, discovered_from=discovered_from)
            return None
        if crawl_delay and crawl_delay > self.p.max_crawl_delay and \
                domain_of(url) in self._touched:
            # Honour a very long Crawl-delay by not making a 2nd request this run
            # (instead of blocking a concurrency slot for minutes).
            reason = f"crawl-delay {crawl_delay:g}s"
            self.counters["skipped"] += 1
            self.skips.append({"url": url, "reason": reason})
            self._delay_skipped.add(url)
            self._log(run_id, url, status="skipped", reason=reason, seed=seed,
                      depth=depth, discovered_from=discovered_from)
            return None
        self._touched.add(domain_of(url))
        self._budget -= 1
        await throttle.wait(url, crawl_delay)
        try:
            status, ctype, text = await self._get(session, url)
        except Exception as exc:
            self.counters["failed"] += 1
            msg = f"{type(exc).__name__}: {exc}"[:300]
            self.errors.append({"url": url, "error": msg})
            self._log(run_id, url, status="failed", reason=msg, seed=seed, depth=depth,
                      discovered_from=discovered_from, fetched=True)
            return None
        self.counters["fetched"] += 1
        self.domains.add(domain_of(url))
        if status >= 400:
            self._http_fail[url] = status
            self.counters["failed"] += 1
            self.errors.append({"url": url, "error": f"HTTP {status}"})
            self._log(run_id, url, status="failed", reason=f"HTTP {status}", seed=seed,
                      depth=depth, discovered_from=discovered_from, http_status=status,
                      fetched=True)
            return None
        return status, ctype, text

    def _store(self, url: str, status: int, html: str, seed: Seed, hint: dict, run_id: int
               ) -> tuple[dict, str]:
        meta = extract_metadata(html, url)
        title = meta["title"] or hint.get("title")
        desc = meta["description"] or hint.get("description")
        tags = set(seed.topics) | set(tag_text(title, desc, meta["text_md"]))
        page = {
            "url": url, "canonical_url": meta["canonical_url"], "domain": domain_of(url),
            "title": title, "description": desc,
            "published_at": meta["published_at"] or hint.get("published_at"),
            "fetched_at": utcnow(), "http_status": status,
            "content_hash": content_hash(meta["text_md"] or html),
            "text_md": meta["text_md"], "source_seed": seed.url,
        }
        _, action = self.db.upsert_page(page, sorted(tags), run_id)
        self.counters["stored"] += 1
        self.counters[action] += 1
        return meta, action

    def _store_feed_only(self, url: str, status: int | None, seed: Seed, hint: dict, run_id: int,
                         *, reason: str, depth: int, discovered_from: str | None) -> None:
        """Store a feed entry without its article body: the article refused us (e.g. 403),
        robots Crawl-delay is too long for a 2nd request, or the seed is ``feed_only``.
        Keeps title/date/summary so the item is still queryable."""
        tags = set(seed.topics) | set(tag_text(hint.get("title"), hint.get("description")))
        page = {
            "url": url, "canonical_url": None, "domain": domain_of(url),
            "title": hint.get("title"), "description": hint.get("description"),
            "published_at": hint.get("published_at"), "fetched_at": utcnow(),
            "http_status": status, "content_hash": None, "text_md": None,
            "source_seed": seed.url,
        }
        if normalize_url(url) in self._feed_only_done:
            return
        self._feed_only_done.add(normalize_url(url))
        _, action = self.db.upsert_page(page, sorted(tags), run_id)
        self.counters["feed_only"] += 1
        self.counters["stored"] += 1
        self.counters[action] += 1
        self._log(run_id, url, status="feed_only", reason=reason, seed=seed, depth=depth,
                  discovered_from=discovered_from, http_status=status,
                  fetched=status is not None)

    def _recent(self, iso: str | None) -> bool:
        if self.p.recent_days is None or not iso:
            return True
        try:
            dt = datetime.fromisoformat(iso)
        except ValueError:
            return True
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt >= datetime.now(timezone.utc) - timedelta(days=self.p.recent_days)

    def _log_unfetched(self, run_id: int, items: list[tuple[str, dict]], seed: Seed,
                       discovered_from: str, reason: str, depth: int = 1) -> None:
        for url, _hint in items:
            self._log(run_id, url, status="not_fetched", reason=reason, seed=seed,
                      depth=depth, discovered_from=discovered_from, overwrite=False)

    def _follow_links(self, frontier: list[tuple[str, dict, int, str]], links: list[str],
                      page_url: str, depth: int, seed: Seed, run_id: int) -> None:
        """Enqueue the same slice the crawler always followed, and log the rest.

        Links rejected by ``is_article_link`` (nav, other sites, assets) are not
        candidates and are not written to the log.
        """
        queued = {item[0] for item in frontier}
        more = [link for link in links if is_article_link(link, page_url) and link not in self._seen]
        if depth < self.p.max_depth:
            cap = max(2, seed.max_items // 3)
            frontier.extend((link, {}, depth + 1, page_url) for link in more[:cap])
            for link in more[cap:]:
                if link not in queued:
                    self._log(run_id, link, status="not_fetched", reason="link_cap", seed=seed,
                              depth=depth + 1, discovered_from=page_url, overwrite=False)
            return
        for link in more:
            if link not in queued:
                self._log(run_id, link, status="not_fetched", reason="depth", seed=seed,
                          depth=depth + 1, discovered_from=page_url, overwrite=False)

    async def _crawl_seed(self, session: Any, robots: RobotsPolicy, throttle: DomainThrottle,
                          seed: Seed, run_id: int, sem: asyncio.Semaphore) -> None:
        # Mark the seed before the first await so a link from another seed cannot
        # fetch it a second time once every seed has started.
        self._mark_seen(seed.url)
        async with sem:
            res = await self._fetch(session, robots, throttle, seed.url, run_id, seed,
                                    depth=0, discovered_from=None)
        if not res:
            return
        status, ctype, text = res
        candidates: list[tuple[str, dict]] = []
        if seed.kind == "feed" or ("xml" in ctype and "html" not in ctype):
            self._log(run_id, seed.url, status="fetched", reason="feed", seed=seed, depth=0,
                      http_status=status, fetched=True)
            for it in parse_feed(text, seed.url):
                if not self._recent(it["published_at"]):
                    self._log(run_id, it["url"], status="not_fetched", reason="recent_days",
                              seed=seed, depth=1, discovered_from=seed.url, overwrite=False)
                    continue
                if seed.require_match and not tag_text(it["title"], it["description"], body_min_hits=1):
                    self._log(run_id, it["url"], status="not_fetched", reason="require_match",
                              seed=seed, depth=1, discovered_from=seed.url, overwrite=False)
                    continue
                candidates.append((it["url"], it))
            discovered_from = seed.url
        else:
            page_url = normalize_url(seed.url)
            meta, action = self._store(page_url, status, text, seed, {}, run_id)
            self._log_stored(run_id, page_url, action, seed, depth=0, discovered_from=None,
                             http_status=status)
            candidates = [(link, {}) for link in meta["links"] if is_article_link(link, seed.url)]
            discovered_from = page_url

        chosen = candidates[: seed.max_items]
        self._log_unfetched(run_id, candidates[seed.max_items:], seed, discovered_from, "max_items")

        if seed.feed_only:
            for url, hint in chosen:
                if url in self._seen:
                    continue
                if not hint.get("title"):
                    self._log(run_id, url, status="not_fetched", reason="no_title", seed=seed,
                              depth=1, discovered_from=discovered_from, overwrite=False)
                    continue
                self._seen.add(url)
                self._store_feed_only(url, None, seed, hint, run_id, reason="feed_only",
                                      depth=1, discovered_from=discovered_from)
            return

        frontier: list[tuple[str, dict, int, str]] = [
            (url, hint, 1, discovered_from) for url, hint in chosen
        ]
        while frontier and self._budget > 0:
            url, hint, depth, parent = frontier.pop(0)
            if url in self._seen:
                continue
            self._mark_seen(url)
            async with sem:
                res = await self._fetch(session, robots, throttle, url, run_id, seed,
                                        depth=depth, discovered_from=parent)
            if not res:
                if hint.get("title") and (url in self._http_fail or url in self._delay_skipped):
                    if url in self._http_fail:
                        reason = f"HTTP {self._http_fail[url]}"
                    else:
                        reason = next(
                            (s["reason"] for s in reversed(self.skips) if s["url"] == url),
                            "crawl-delay",
                        )
                    self._store_feed_only(
                        url, self._http_fail.get(url), seed, hint, run_id,
                        reason=reason, depth=depth, discovered_from=parent,
                    )
                continue
            status, ctype, html = res
            if "html" not in ctype and "xml" not in ctype and "text" not in ctype:
                self.counters["skipped"] += 1
                reason = f"content-type {ctype}"
                self.skips.append({"url": url, "reason": reason})
                self._log(run_id, url, status="skipped", reason=reason, seed=seed, depth=depth,
                          discovered_from=parent, http_status=status, fetched=True)
                continue
            meta, action = self._store(url, status, html, seed, hint, run_id)
            self._log_stored(run_id, url, action, seed, depth=depth, discovered_from=parent,
                             http_status=status)
            self._follow_links(frontier, meta["links"], url, depth, seed, run_id)

        for url, _hint, depth, parent in frontier:
            self._log(run_id, url, status="not_fetched", reason="budget", seed=seed,
                      depth=depth, discovered_from=parent, overwrite=False)

    async def run(self, seeds: list[Seed], run_params: dict) -> dict[str, Any]:
        import aiohttp
        pruned = self.db.prune_url_log(self.p.keep_url_log_days)
        run_id = self.db.start_run(run_params)
        started = time.monotonic()
        status = "failed"
        try:
            async with aiohttp.ClientSession() as session:
                robots = RobotsPolicy(session, self.p.user_agent)
                throttle = DomainThrottle(self.p.delay)
                sem = asyncio.Semaphore(max(1, self.p.concurrency))
                await asyncio.gather(*(
                    self._crawl_seed(session, robots, throttle, s, run_id, sem) for s in seeds
                ))
            status = "completed" if self.counters["stored"] else "failed"
            if status == "completed" and self.counters["failed"]:
                status = "partial"
        finally:
            self.db.sync_url_counts(run_id)
            self.db.finish_run(run_id, status, self.counters, self.errors)
        return {
            "run_id": run_id, "status": status, "db_path": self.db.path,
            "elapsed_s": round(time.monotonic() - started, 1),
            "counters": self.counters, "domains": sorted(self.domains),
            "errors": self.errors, "skips": self.skips,
            "urls_logged": self.db.count_urls(run_id),
            "urls_counted": 0,
            "url_log_pruned": {
                k: pruned[k] for k in ("keep_days", "cutoff", "rows", "disabled")
            },
        }
