"""
storage/db.py — Async SQLite engine, session management, resume support,
and content deduplication helpers.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from storage.models import (
    Base,
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


async def init_db(db_path: str = "crawl_data.db") -> None:
    """Initialize the async SQLite engine and create all tables."""
    global _engine, _session_factory

    url = f"sqlite+aiosqlite:///{db_path}"
    _engine = create_async_engine(url, echo=False)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database initialized at %s", db_path)


async def close_db() -> None:
    """Dispose the engine connection pool."""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Async context manager providing a database session."""
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    async with _session_factory() as session:
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
            select(URLRecord).where(URLRecord.url == item.url)
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
    Persist a fetched page. If the content hash already exists,
    skip insertion and return the existing page ID.
    Returns the page record ID.
    """
    async with get_session() as db:
        existing = await db.scalar(
            select(VisitedPage).where(
                VisitedPage.content_hash == page.content_hash
            )
        )
        if existing:
            logger.debug("Skipping duplicate content for %s", page.url)
            return existing.id

        record = VisitedPage(
            url=page.url,
            content_hash=page.content_hash,
            markdown=page.markdown,
            raw_html=page.raw_html,
            title=page.title,
            fetch_method=page.fetch_method,
        )
        db.add(record)
        await db.flush()
        return record.id


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
