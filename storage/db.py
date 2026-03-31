"""
storage/db.py — Async SQLite engine, session management, resume support,
and content deduplication helpers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator

from sqlalchemy import select, text as sa_text, update
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from storage.models import (
    Base,
    CrawlLog,
    CrawlSession,
    ExtractedData,
    ExtractionResult,
    PageContent,
    SessionStatus,
    URLItem,
    URLRecord,
    URLStatus,
    VisitedPage,
)

logger = logging.getLogger(__name__)


# ── Engine factory ────────────────────────────────────────────────────────────

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _engine


async def init_db(db_path: str = "crawl_data.db") -> AsyncEngine:
    """Initialize the async SQLite engine and create all tables.

    Returns the engine so callers that need an explicit handle can use it.
    The module-level globals are also set for code that uses get_session()
    without passing an engine.
    """
    global _engine, _session_factory

    url = f"sqlite+aiosqlite:///{db_path}"
    _engine = create_async_engine(url, echo=False)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await migrate_add_columns(_engine)

    logger.info("Database initialized at %s", db_path)
    return _engine


async def migrate_add_columns(engine: AsyncEngine) -> None:
    """
    Idempotent migration: add new columns introduced in Phase 1/2 that may
    not exist in databases created before this enhancement.

    SQLite ALTER TABLE ADD COLUMN is a no-op if the column already exists
    (we catch the OperationalError and continue).  All new columns are
    nullable or have defaults so no data is lost.
    """
    migrations = [
        # URLRecord (urls table)
        "ALTER TABLE urls ADD COLUMN source TEXT NOT NULL DEFAULT 'crawl'",
        "ALTER TABLE urls ADD COLUMN sitemap_priority REAL",
        "ALTER TABLE urls ADD COLUMN sitemap_changefreq TEXT",
        "ALTER TABLE urls ADD COLUMN sitemap_lastmod TEXT",
        "ALTER TABLE urls ADD COLUMN last_crawled_at DATETIME",
        "ALTER TABLE urls ADD COLUMN recrawl_after_days INTEGER",
        "ALTER TABLE urls ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0",
        # VisitedPage (visited_pages table)
        "ALTER TABLE visited_pages ADD COLUMN etag TEXT",
        "ALTER TABLE visited_pages ADD COLUMN last_modified TEXT",
    ]
    async with engine.begin() as conn:
        for stmt in migrations:
            try:
                await conn.execute(sa_text(stmt))
            except Exception:
                pass  # Column already exists — safe to skip


async def close_db() -> None:
    """Dispose the engine connection pool."""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None


@asynccontextmanager
async def get_session(engine: AsyncEngine | None = None) -> AsyncIterator[AsyncSession]:
    """Async context manager providing a database session.

    Accepts an optional *engine* argument for callers that hold an explicit
    engine reference (e.g. tests).  Falls back to the module-level global.
    """
    if engine is not None:
        factory = async_sessionmaker(engine, expire_on_commit=False)
    elif _session_factory is not None:
        factory = _session_factory
    else:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── Session management ────────────────────────────────────────────────────────

async def create_crawl_session(
    goal: str,
    workflow_used: str,
    start_urls: list[str],
) -> str:
    """Create a new CrawlSession and return its ID."""
    session_id = str(uuid.uuid4())
    record = CrawlSession(
        id=session_id,
        goal=goal,
        workflow_used=workflow_used,
        start_urls=start_urls,
        status=SessionStatus.running,
        stats={},
    )
    async with get_session() as db:
        db.add(record)
    logger.info("Created crawl session %s", session_id)
    return session_id


async def update_session_stats(session_id: str, stats: dict) -> None:
    """Persist updated statistics for a crawl session."""
    async with get_session() as db:
        await db.execute(
            update(CrawlSession)
            .where(CrawlSession.id == session_id)
            .values(stats=stats)
        )


async def finish_crawl_session(session_id: str, status: SessionStatus) -> None:
    """Mark a session as completed/failed with a timestamp."""
    async with get_session() as db:
        await db.execute(
            update(CrawlSession)
            .where(CrawlSession.id == session_id)
            .values(status=status, ended_at=datetime.now(timezone.utc))
        )


# ── URL queue helpers ─────────────────────────────────────────────────────────

async def add_to_queue(
    session: AsyncSession,
    urls: list[str],
    *,
    session_id: str | int | None = None,
    priority: float = 0.5,
    depth: int = 0,
    parent_url: str | None = None,
) -> int:
    """Spec-compatible bulk enqueue: add a list of URL strings to the queue.

    Silently skips duplicates.  Returns the number of URLs inserted.
    """
    from utils.url import normalize_url

    inserted = 0
    sid_str = str(session_id) if session_id is not None else None
    for raw_url in urls:
        canonical = normalize_url(raw_url)
        existing = await session.scalar(
            select(URLRecord).where(
                URLRecord.url == canonical,
                URLRecord.session_id == sid_str,
            )
        )
        if existing:
            continue
        record = URLRecord(
            url=canonical,
            priority=priority,
            depth=depth,
            relevance_score=0.0,
            status=URLStatus.pending,
            session_id=str(session_id) if session_id is not None else None,
            parent_url=parent_url,
        )
        session.add(record)
        inserted += 1
    return inserted


async def get_next_url(
    session: AsyncSession,
    session_id: str | int | None = None,
) -> URLRecord | None:
    """Spec-compatible: fetch the highest-priority pending URL and mark it in_progress.

    Returns the ORM URLRecord (with a .url attribute), or None if the queue is empty.
    """
    result = await session.execute(
        select(URLRecord)
        .where(
            URLRecord.session_id == str(session_id) if session_id is not None else True,
            URLRecord.status == URLStatus.pending,
        )
        .order_by(URLRecord.priority.desc())
        .limit(1)
    )
    record = result.scalar_one_or_none()
    if record is None:
        return None
    record.status = URLStatus.in_progress
    return record


async def enqueue_url(item: URLItem) -> bool:
    """
    Add a URL to the queue if not already present.

    The URL is normalized before insertion so that semantically identical
    URLs (different encoding, trailing slashes, etc.) are deduplicated.
    Returns True if inserted, False if already exists.
    """
    from utils.url import normalize_url

    canonical = normalize_url(item.url)
    if canonical != item.url:
        item = item.model_copy(update={"url": canonical})

    async with get_session() as db:
        existing = await db.scalar(
            select(URLRecord).where(
                URLRecord.url == item.url,
                URLRecord.session_id == item.session_id,
            )
        )
        if existing:
            return False
        record = URLRecord(
            url=item.url,
            priority=item.priority,
            depth=item.depth,
            relevance_score=item.relevance_score,
            status=item.status,
            session_id=item.session_id,
            parent_url=item.parent_url,
            source=item.source,
            retry_count=item.retry_count,
        )
        db.add(record)
    return True


async def dequeue_next_url(session_id: str) -> URLItem | None:
    """
    Fetch the highest-priority pending URL and mark it in_progress.
    Returns None if queue is empty.
    """
    async with get_session() as db:
        result = await db.execute(
            select(URLRecord)
            .where(
                URLRecord.session_id == session_id,
                URLRecord.status == URLStatus.pending,
            )
            .order_by(URLRecord.priority.desc())
            .limit(1)
        )
        record = result.scalar_one_or_none()
        if record is None:
            return None
        record.status = URLStatus.in_progress
    return URLItem.model_validate(record)


async def mark_url_done(url: str) -> None:
    async with get_session() as db:
        await db.execute(
            update(URLRecord)
            .where(URLRecord.url == url)
            .values(status=URLStatus.done)
        )


async def mark_url_failed(url: str, error: str) -> None:
    async with get_session() as db:
        await db.execute(
            update(URLRecord)
            .where(URLRecord.url == url)
            .values(status=URLStatus.failed, error_message=error[:1024])
        )


async def get_pending_count(session_id: str) -> int:
    """Return the number of pending URLs for a session."""
    from sqlalchemy import func as sqlfunc
    async with get_session() as db:
        count = await db.scalar(
            select(sqlfunc.count()).where(
                URLRecord.session_id == session_id,
                URLRecord.status == URLStatus.pending,
            )
        )
    return count or 0


async def resume_session(session_id: str) -> list[URLItem]:
    """
    Reload all pending/in_progress URLs for a session (resume support).
    Resets in_progress → pending so they can be retried.
    """
    async with get_session() as db:
        # Reset stale in_progress entries
        await db.execute(
            update(URLRecord)
            .where(
                URLRecord.session_id == session_id,
                URLRecord.status == URLStatus.in_progress,
            )
            .values(status=URLStatus.pending)
        )
        result = await db.execute(
            select(URLRecord).where(
                URLRecord.session_id == session_id,
                URLRecord.status == URLStatus.pending,
            )
        )
        records = result.scalars().all()
    return [URLItem.model_validate(r) for r in records]


# ── Page storage with deduplication ──────────────────────────────────────────

def compute_content_hash(content: str) -> str:
    """SHA-256 hash of page content for deduplication."""
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


async def save_page(page: PageContent) -> int:
    """
    Persist a fetched page.

    Dedup order:
    1. Same URL — update in-place if content changed, return ID if unchanged.
    2. Same content hash at a different URL — return the existing page ID.
    3. Neither — insert a new row.

    Returns the page record ID.
    """
    async with get_session() as db:
        # 1. Look up by URL first (handles re-crawls of the same page)
        existing_by_url = await db.scalar(
            select(VisitedPage).where(VisitedPage.url == page.url)
        )
        if existing_by_url:
            if existing_by_url.content_hash == page.content_hash:
                logger.debug("Content unchanged for %s", page.url)
                return existing_by_url.id
            # Content changed — update in-place
            existing_by_url.content_hash = page.content_hash
            existing_by_url.markdown = page.markdown
            existing_by_url.raw_html = page.raw_html
            existing_by_url.title = page.title
            existing_by_url.fetch_method = page.fetch_method
            existing_by_url.extracted_at = datetime.now(timezone.utc)
            existing_by_url.etag = page.etag
            existing_by_url.last_modified = page.last_modified
            logger.debug("Updated changed content for %s", page.url)
            return existing_by_url.id

        # 2. Same content at a different URL (canonical dedup)
        existing_by_hash = await db.scalar(
            select(VisitedPage).where(VisitedPage.content_hash == page.content_hash)
        )
        if existing_by_hash:
            logger.debug("Skipping duplicate content for %s", page.url)
            return existing_by_hash.id

        # 3. New page
        record = VisitedPage(
            url=page.url,
            content_hash=page.content_hash,
            markdown=page.markdown,
            raw_html=page.raw_html,
            title=page.title,
            fetch_method=page.fetch_method,
            etag=page.etag,
            last_modified=page.last_modified,
        )
        db.add(record)
        await db.flush()
        return record.id


async def get_page_by_url(url: str) -> "VisitedPage | None":
    """Return the most recent VisitedPage for a URL across all sessions."""
    async with get_session() as db:
        return await db.scalar(
            select(VisitedPage).where(VisitedPage.url == url)
        )


async def update_page_headers(
    url: str,
    etag: "str | None",
    last_modified: "str | None",
) -> None:
    """Persist ETag and Last-Modified headers for future conditional requests."""
    async with get_session() as db:
        await db.execute(
            update(VisitedPage)
            .where(VisitedPage.url == url)
            .values(etag=etag, last_modified=last_modified)
        )


async def increment_url_retry(url: str, session_id: str, error: str) -> int:
    """
    Increment retry_count for a URL.  If retry_count reaches max_retries,
    mark the URL as permanently failed and return -1.
    Otherwise reset status to pending and return the new retry count.
    """
    from config import load_config
    max_retries = load_config().crawl.max_retries_per_url

    async with get_session() as db:
        record = await db.scalar(
            select(URLRecord).where(
                URLRecord.url == url,
                URLRecord.session_id == session_id,
            )
        )
        if record is None:
            return -1
        record.retry_count = (record.retry_count or 0) + 1
        if record.retry_count >= max_retries:
            record.status = URLStatus.failed
            record.error_message = error[:1024]
            return -1
        record.status = URLStatus.pending
        record.error_message = error[:1024]
        return record.retry_count


async def save_extraction(
    page_id: int,
    result: ExtractionResult,
    session_id: str | None,
) -> None:
    """Persist an ExtractionResult linked to a page record."""
    record = ExtractedData(
        page_id=page_id,
        data=result.data,
        schema_used=result.schema_used,
        confidence=result.confidence,
        session_id=session_id,
    )
    async with get_session() as db:
        db.add(record)


async def get_all_extractions(session_id: str) -> list[dict]:
    """Return all extracted data for a session as plain dicts."""
    async with get_session() as db:
        result = await db.execute(
            select(ExtractedData).where(ExtractedData.session_id == session_id)
        )
        records = result.scalars().all()
    return [{"data": r.data, "confidence": r.confidence, "schema": r.schema_used} for r in records]


# ── Web dashboard helpers ─────────────────────────────────────────────────────

async def ensure_fts_and_logs(engine: "AsyncEngine") -> None:
    """
    Create the full-text search index table and update triggers (idempotent).

    Called once by the web app lifespan on startup. Also initialises the
    research paper FTS5 table so the web search covers all content types.
    """
    from sqlalchemy import text as sa_text

    async with engine.begin() as conn:
        # search_index: FTS5 table covering page content + extractions
        await conn.execute(sa_text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS search_index
            USING fts5(
                session_id UNINDEXED,
                url        UNINDEXED,
                title,
                content,
                tokenize = 'porter ascii'
            )
        """))

        # Populate if empty
        # (populated lazily via populate_search_index on session completion)

    # Also set up research FTS
    try:
        from storage.research import init_research_fts
        await init_research_fts(engine)
    except Exception:
        pass


async def insert_crawl_log(
    session_id: str,
    message: str,
    component: str = "system",
    level: str = "info",
    extra_data: "dict | None" = None,
) -> None:
    """Append a log entry to the crawl_logs table for the web activity feed."""
    try:
        from storage.models import CrawlLog
        record = CrawlLog(
            session_id=session_id,
            level=level,
            component=component,
            message=message[:2048],
            extra_data=extra_data,
        )
        async with get_session() as db:
            db.add(record)
    except Exception:
        pass  # Log writes must never crash the crawler


async def get_session_logs(
    session_id: str,
    limit: int = 200,
    offset: int = 0,
    component: "str | None" = None,
) -> list[dict]:
    """Fetch crawl log entries for a session, newest first."""
    from storage.models import CrawlLog
    async with get_session() as db:
        q = (
            select(CrawlLog)
            .where(CrawlLog.session_id == session_id)
            .order_by(CrawlLog.timestamp.desc())
            .limit(limit)
            .offset(offset)
        )
        if component:
            q = q.where(CrawlLog.component == component)
        result = await db.execute(q)
        records = result.scalars().all()
    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "level": r.level,
            "component": r.component,
            "message": r.message,
            "extra_data": r.extra_data,
        }
        for r in reversed(records)  # chronological order
    ]


async def get_new_logs_since(session_id: str, last_id: int) -> list[dict]:
    """Return log entries with id > last_id for SSE live streaming."""
    from storage.models import CrawlLog
    async with get_session() as db:
        result = await db.execute(
            select(CrawlLog)
            .where(
                CrawlLog.session_id == session_id,
                CrawlLog.id > last_id,
            )
            .order_by(CrawlLog.id.asc())
            .limit(50)
        )
        records = result.scalars().all()
    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "level": r.level,
            "component": r.component,
            "message": r.message,
        }
        for r in records
    ]


async def get_recent_activity(limit: int = 50) -> list[dict]:
    """Return recent log entries across all sessions for the dashboard feed."""
    from storage.models import CrawlLog
    async with get_session() as db:
        result = await db.execute(
            select(CrawlLog)
            .order_by(CrawlLog.timestamp.desc())
            .limit(limit)
        )
        records = result.scalars().all()
    return [
        {
            "id": r.id,
            "session_id": r.session_id,
            "session_short": r.session_id[:8] if r.session_id else "—",
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "level": r.level,
            "component": r.component,
            "message": r.message,
        }
        for r in records
    ]


async def get_global_stats() -> dict:
    """Aggregate counts for the dashboard stats bar."""
    from sqlalchemy import func as sqlfunc
    async with get_session() as db:
        total_sessions = await db.scalar(select(sqlfunc.count(CrawlSession.id))) or 0
        total_pages = await db.scalar(select(sqlfunc.count(VisitedPage.id))) or 0
        total_extractions = await db.scalar(select(sqlfunc.count(ExtractedData.id))) or 0
        running = await db.scalar(
            select(sqlfunc.count(CrawlSession.id)).where(
                CrawlSession.status == SessionStatus.running
            )
        ) or 0
    return {
        "total_sessions": total_sessions,
        "total_pages": total_pages,
        "total_extractions": total_extractions,
        "running_crawls": running,
    }


async def populate_search_index(session_id: str) -> int:
    """
    Index all visited pages for a session into the FTS5 search_index table.

    Returns the count of rows inserted. Skips already-indexed rows.
    """
    from sqlalchemy import text as sa_text

    async with get_session() as db:
        # Remove old entries for this session first (clean re-index)
        await db.execute(
            sa_text("DELETE FROM search_index WHERE session_id = :sid"),
            {"sid": session_id},
        )

        # Pull pages that have been visited for this session
        result = await db.execute(
            select(VisitedPage)
            .join(URLRecord, URLRecord.url == VisitedPage.url)
            .where(URLRecord.session_id == session_id)
            .where(URLRecord.status == URLStatus.done)
        )
        pages = result.scalars().all()

        count = 0
        for page in pages:
            content = (page.markdown or page.raw_html or "")[:50_000]
            await db.execute(
                sa_text("""
                    INSERT INTO search_index(session_id, url, title, content)
                    VALUES (:sid, :url, :title, :content)
                """),
                {
                    "sid": session_id,
                    "url": page.url,
                    "title": page.title or "",
                    "content": content,
                },
            )
            count += 1

    return count


async def search_fulltext(
    query: str,
    limit: int = 50,
    session_id_filter: "str | None" = None,
) -> list[dict]:
    """
    Full-text search across the search_index FTS5 table.

    Returns a list of hit dicts: {url, title, snippet, session_id, rank}.
    """
    from sqlalchemy import text as sa_text

    # Safe FTS5 query (same logic as storage.research._safe_fts_query)
    import re
    if not re.search(r'\b(AND|OR|NOT|NEAR)\b|[*"]', query):
        tokens = query.split()
        fts_q = " ".join(f'"{t}"' for t in tokens)
    else:
        fts_q = query

    async with get_session() as db:
        try:
            if session_id_filter:
                rows = await db.execute(
                    sa_text("""
                        SELECT session_id, url, title,
                               snippet(search_index, 3, '<mark>', '</mark>', '…', 20) AS snippet,
                               rank
                        FROM search_index
                        WHERE search_index MATCH :q
                          AND session_id = :sid
                        ORDER BY rank
                        LIMIT :lim
                    """),
                    {"q": fts_q, "sid": session_id_filter, "lim": limit},
                )
            else:
                rows = await db.execute(
                    sa_text("""
                        SELECT session_id, url, title,
                               snippet(search_index, 3, '<mark>', '</mark>', '…', 20) AS snippet,
                               rank
                        FROM search_index
                        WHERE search_index MATCH :q
                        ORDER BY rank
                        LIMIT :lim
                    """),
                    {"q": fts_q, "lim": limit},
                )
            return [dict(r._mapping) for r in rows]
        except Exception:
            return []
