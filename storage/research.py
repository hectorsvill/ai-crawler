"""
storage/research.py — Persistence helpers for the research pipeline.

Owns:
  - FTS5 virtual-table + trigger setup (init_research_fts)
  - save_research_paper  — insert with DOI/arXiv dedup
  - search_research_papers — full-text search via FTS5
  - list_research_papers  — paginated browse
  - get_research_paper    — fetch by ID
  - get_research_stats    — aggregate counts
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncEngine

from storage.db import get_session
from storage.models import ResearchPaper, ResearchPaperData

logger = logging.getLogger(__name__)


# ── FTS5 setup ─────────────────────────────────────────────────────────────────

async def init_research_fts(engine: AsyncEngine) -> None:
    """
    Create the FTS5 content table + update triggers if they don't exist.

    Safe to call multiple times — all statements use IF NOT EXISTS / OR IGNORE.
    Must be called after init_db() has created the research_papers table.
    """
    async with engine.begin() as conn:
        # FTS5 virtual table backed by research_papers
        await conn.execute(text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS research_papers_fts
            USING fts5(
                title,
                abstract,
                authors,
                keywords,
                venue,
                content='research_papers',
                content_rowid='id'
            )
        """))

        # Trigger: after INSERT → populate FTS
        await conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS rp_ai
            AFTER INSERT ON research_papers BEGIN
                INSERT INTO research_papers_fts(
                    rowid, title, abstract, authors, keywords, venue
                ) VALUES (
                    new.id,
                    new.title,
                    new.abstract,
                    COALESCE(new.authors, '[]'),
                    COALESCE(new.keywords, '[]'),
                    new.venue
                );
            END
        """))

        # Trigger: after DELETE → remove from FTS
        await conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS rp_ad
            AFTER DELETE ON research_papers BEGIN
                INSERT INTO research_papers_fts(
                    research_papers_fts, rowid, title, abstract, authors, keywords, venue
                ) VALUES (
                    'delete',
                    old.id,
                    old.title,
                    old.abstract,
                    COALESCE(old.authors, '[]'),
                    COALESCE(old.keywords, '[]'),
                    old.venue
                );
            END
        """))

        # Trigger: after UPDATE → delete old FTS entry, insert new
        await conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS rp_au
            AFTER UPDATE ON research_papers BEGIN
                INSERT INTO research_papers_fts(
                    research_papers_fts, rowid, title, abstract, authors, keywords, venue
                ) VALUES (
                    'delete',
                    old.id,
                    old.title,
                    old.abstract,
                    COALESCE(old.authors, '[]'),
                    COALESCE(old.keywords, '[]'),
                    old.venue
                );
                INSERT INTO research_papers_fts(
                    rowid, title, abstract, authors, keywords, venue
                ) VALUES (
                    new.id,
                    new.title,
                    new.abstract,
                    COALESCE(new.authors, '[]'),
                    COALESCE(new.keywords, '[]'),
                    new.venue
                );
            END
        """))

    logger.debug("Research FTS5 table and triggers ready")


# ── Save ───────────────────────────────────────────────────────────────────────

async def save_research_paper(
    paper_data: ResearchPaperData,
    source_url: str,
    session_id: str | None,
    page_id: int | None = None,
    content_hash: str | None = None,
) -> int:
    """
    Persist a research paper.  Returns the record ID.

    Deduplication priority:
    1. Same DOI across any session → skip, return existing ID
    2. Same arXiv ID across any session → skip, return existing ID
    3. Same content_hash → skip, return existing ID
    4. Otherwise insert new row
    """
    async with get_session() as db:
        # DOI dedup
        if paper_data.doi:
            existing_id = await db.scalar(
                select(ResearchPaper.id).where(ResearchPaper.doi == paper_data.doi)
            )
            if existing_id:
                logger.debug("Skipping duplicate DOI %s", paper_data.doi)
                return existing_id

        # arXiv ID dedup
        if paper_data.arxiv_id:
            existing_id = await db.scalar(
                select(ResearchPaper.id).where(ResearchPaper.arxiv_id == paper_data.arxiv_id)
            )
            if existing_id:
                logger.debug("Skipping duplicate arXiv ID %s", paper_data.arxiv_id)
                return existing_id

        # Content hash dedup
        if content_hash:
            existing_id = await db.scalar(
                select(ResearchPaper.id).where(ResearchPaper.content_hash == content_hash)
            )
            if existing_id:
                logger.debug("Skipping duplicate content hash for %s", source_url)
                return existing_id

        record = ResearchPaper(
            session_id=session_id,
            source_url=source_url,
            page_id=page_id,
            title=paper_data.title,
            authors=paper_data.authors or [],
            abstract=paper_data.abstract,
            year=paper_data.year,
            doi=paper_data.doi,
            arxiv_id=paper_data.arxiv_id,
            venue=paper_data.venue,
            keywords=paper_data.keywords or [],
            pdf_url=paper_data.pdf_url,
            content_hash=content_hash,
            confidence=paper_data.confidence,
        )
        db.add(record)
        await db.flush()
        return record.id


# ── Search ─────────────────────────────────────────────────────────────────────

def _safe_fts_query(query: str) -> str:
    """
    Prepare a user query string for FTS5 MATCH.

    If the query looks like raw FTS5 syntax (contains AND/OR/NOT/NEAR/*/"),
    pass it through unchanged.  Otherwise tokenise on whitespace and
    join with implicit AND so that each token must appear independently,
    avoiding FTS5 misinterpreting hyphens (e.g. "peer-to-peer") as SQL
    column references.
    """
    import re
    # Detect explicit FTS5 operator usage
    if re.search(r'\b(AND|OR|NOT|NEAR)\b|[*"]', query):
        return query
    # Wrap each whitespace-delimited token in double quotes for exact-token match
    tokens = query.split()
    return " ".join(f'"{t}"' for t in tokens)


async def search_research_papers(
    query: str,
    limit: int = 20,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Full-text search across all research papers using SQLite FTS5.

    ``query`` supports FTS5 syntax:
      - Simple:     "jailbreak LLM"
      - Phrase:     '"prompt injection"'
      - Boolean:    "transformer AND attention NOT vision"
      - Prefix:     "neural*"

    Returns list of dicts with all ResearchPaper columns plus a ``rank`` score.
    Lower rank = better match (FTS5 BM25 convention — negate for DESC sort).
    """
    fts_query = _safe_fts_query(query)
    async with get_session() as db:
        if session_id:
            rows = await db.execute(
                text("""
                    SELECT rp.*, fts.rank
                    FROM research_papers rp
                    JOIN research_papers_fts fts ON fts.rowid = rp.id
                    WHERE research_papers_fts MATCH :query
                      AND rp.session_id = :session_id
                    ORDER BY fts.rank
                    LIMIT :limit
                """),
                {"query": fts_query, "session_id": session_id, "limit": limit},
            )
        else:
            rows = await db.execute(
                text("""
                    SELECT rp.*, fts.rank
                    FROM research_papers rp
                    JOIN research_papers_fts fts ON fts.rowid = rp.id
                    WHERE research_papers_fts MATCH :query
                    ORDER BY fts.rank
                    LIMIT :limit
                """),
                {"query": fts_query, "limit": limit},
            )
        return [dict(r._mapping) for r in rows]


# ── List / get ─────────────────────────────────────────────────────────────────

async def list_research_papers(
    session_id: str | None = None,
    year: int | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """
    Paginated list of all research papers, newest first.

    Optionally filtered by session_id and/or year.
    """
    async with get_session() as db:
        q = select(ResearchPaper)
        if session_id:
            q = q.where(ResearchPaper.session_id == session_id)
        if year is not None:
            q = q.where(ResearchPaper.year == year)
        q = q.order_by(ResearchPaper.year.desc(), ResearchPaper.created_at.desc())
        q = q.limit(limit).offset(offset)
        result = await db.execute(q)
        records = result.scalars().all()

    return [_paper_to_dict(r) for r in records]


async def get_research_paper(paper_id: int) -> dict[str, Any] | None:
    """Fetch a single research paper by primary key."""
    async with get_session() as db:
        record = await db.scalar(
            select(ResearchPaper).where(ResearchPaper.id == paper_id)
        )
    return _paper_to_dict(record) if record else None


async def get_research_stats() -> dict[str, Any]:
    """Return aggregate statistics across all research papers."""
    async with get_session() as db:
        total = await db.scalar(select(func.count(ResearchPaper.id))) or 0
        sessions = await db.scalar(
            select(func.count(func.distinct(ResearchPaper.session_id)))
        ) or 0
        min_year = await db.scalar(select(func.min(ResearchPaper.year)))
        max_year = await db.scalar(select(func.max(ResearchPaper.year)))
    return {
        "total_papers": total,
        "sessions_with_papers": sessions,
        "year_range": [min_year, max_year],
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _paper_to_dict(r: ResearchPaper) -> dict[str, Any]:
    return {
        "id": r.id,
        "session_id": r.session_id,
        "source_url": r.source_url,
        "title": r.title,
        "authors": r.authors or [],
        "abstract": r.abstract,
        "year": r.year,
        "doi": r.doi,
        "arxiv_id": r.arxiv_id,
        "venue": r.venue,
        "keywords": r.keywords or [],
        "pdf_url": r.pdf_url,
        "confidence": r.confidence,
        "created_at": str(r.created_at),
    }
