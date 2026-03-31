"""
storage/models.py — SQLAlchemy ORM models and Pydantic data-shapes.

Tables:
  - urls            : crawl queue with priority and status tracking
  - visited_pages   : deduplicated page content (SHA-256 content hash)
  - extracted_data  : structured JSON output from the Extractor agent
  - crawl_sessions  : metadata for each crawl run
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ── SQLAlchemy base ───────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── Enums ─────────────────────────────────────────────────────────────────────

class URLStatus(str, enum.Enum):
    pending = "pending"
    in_progress = "in_progress"
    done = "done"
    failed = "failed"
    skipped = "skipped"


class SessionStatus(str, enum.Enum):
    running = "running"
    completed = "completed"
    failed = "failed"
    paused = "paused"


# ── ORM Models ────────────────────────────────────────────────────────────────

class URLRecord(Base):
    """Crawl queue entry — one row per discovered URL."""

    __tablename__ = "urls"
    __table_args__ = (
        # Per-session uniqueness: same URL can appear in different sessions
        UniqueConstraint("url", "session_id", name="uq_urls_url_session"),
        Index("ix_urls_status_priority", "status", "priority"),
        Index("ix_urls_session", "session_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(String(2048), nullable=False)  # unique per session
    priority: Mapped[float] = mapped_column(Float, default=0.5)
    depth: Mapped[int] = mapped_column(Integer, default=0)
    relevance_score: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(
        Enum(URLStatus), default=URLStatus.pending, nullable=False
    )
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    parent_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Phase 1 additions
    source: Mapped[str] = mapped_column(String(32), default="crawl")
    # "seed" | "sitemap" | "llms_txt" | "crawl"
    sitemap_priority: Mapped[float | None] = mapped_column(Float, nullable=True)
    sitemap_changefreq: Mapped[str | None] = mapped_column(String(16), nullable=True)
    sitemap_lastmod: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_crawled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    recrawl_after_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)


class VisitedPage(Base):
    """Stores fetched page content, deduplicated by content hash."""

    __tablename__ = "visited_pages"
    __table_args__ = (
        Index("ix_visited_pages_content_hash", "content_hash"),
        Index("ix_visited_pages_url", "url"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_html: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    extracted_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    fetch_method: Mapped[str] = mapped_column(String(32), default="aiohttp")
    # Phase 1 additions — HTTP conditional request headers
    etag: Mapped[str | None] = mapped_column(String(256), nullable=True)
    last_modified: Mapped[str | None] = mapped_column(String(64), nullable=True)

    extracted_data: Mapped[list[ExtractedData]] = relationship(
        "ExtractedData", back_populates="page", cascade="all, delete-orphan"
    )


class ExtractedData(Base):
    """Structured JSON output produced by the Extractor agent for a page."""

    __tablename__ = "extracted_data"
    __table_args__ = (Index("ix_extracted_data_session", "session_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    page_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("visited_pages.id", ondelete="CASCADE"), nullable=False
    )
    data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    schema_used: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    page: Mapped[VisitedPage] = relationship("VisitedPage", back_populates="extracted_data")


class CrawlSession(Base):
    """Metadata for a single crawl run."""

    __tablename__ = "crawl_sessions"

    id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    workflow_used: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    stats: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(
        Enum(SessionStatus), default=SessionStatus.running, nullable=False
    )
    start_urls: Mapped[list[str]] = mapped_column(JSON, default=list)


# ── Pydantic data-shapes (passed between components) ─────────────────────────

class URLItem(BaseModel):
    """In-memory representation of a URL queue entry."""

    url: str
    priority: float = 0.5
    depth: int = 0
    relevance_score: float = 0.0
    status: URLStatus = URLStatus.pending
    session_id: str | None = None
    parent_url: str | None = None
    source: str = "crawl"
    sitemap_priority: float | None = None
    retry_count: int = 0

    model_config = {"from_attributes": True}


class PageContent(BaseModel):
    """Fetched page content passed between crawler and agents."""

    url: str
    markdown: str
    raw_html: str | None = None
    title: str | None = None
    content_hash: str
    fetch_method: str = "aiohttp"
    links: list[str] = Field(default_factory=list)
    status_code: int = 200
    error: str | None = None
    # Phase 1 additions — conditional request support
    etag: str | None = None
    last_modified: str | None = None
    changed: bool = True  # False when 304 or hash unchanged


class LinkPriority(BaseModel):
    """A discovered link with its priority assessment from the Navigator."""

    url: str
    priority: float = Field(default=0.5)
    reasoning: str
    estimated_value: float = Field(default=0.5)

    @field_validator("priority", "estimated_value", mode="before")
    @classmethod
    def clamp_01(cls, v: Any) -> float:
        """Clamp to [0, 1] instead of rejecting out-of-range LLM output."""
        return max(0.0, min(1.0, float(v)))


class NavigatorDecision(BaseModel):
    """Decision returned by the Navigator agent for a given page."""

    relevance_score: float = Field(default=0.0)
    links_to_follow: list[LinkPriority] = Field(default_factory=list)
    action: str = "deepen"  # "deepen" | "backtrack" | "complete"
    reasoning: str = ""

    @field_validator("relevance_score", mode="before")
    @classmethod
    def clamp_relevance(cls, v: Any) -> float:
        return max(0.0, min(1.0, float(v)))

    @field_validator("action", mode="before")
    @classmethod
    def validate_action(cls, v: Any) -> str:
        valid = {"deepen", "backtrack", "complete"}
        s = str(v).lower().strip()
        return s if s in valid else "deepen"


class ExtractionResult(BaseModel):
    """Structured data returned by the Extractor agent."""

    data: dict[str, Any]
    schema_used: str
    confidence: float = Field(default=0.0)
    explanation: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: Any) -> float:
        return max(0.0, min(1.0, float(v)))


class SessionStats(BaseModel):
    """Live statistics for a crawl session."""

    pages_crawled: int = 0
    pages_failed: int = 0
    pages_skipped: int = 0
    extractions: int = 0
    queue_size: int = 0
    current_url: str = ""
    elapsed_seconds: float = 0.0


# ── Research paper models ──────────────────────────────────────────────────────

class CrawlLog(Base):
    """Audit trail for crawler decisions, fetches, and errors."""

    __tablename__ = "crawl_logs"
    __table_args__ = (
        Index("ix_crawl_logs_session_id", "session_id"),
        Index("ix_crawl_logs_timestamp", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    level: Mapped[str] = mapped_column(String(16), default="info")
    component: Mapped[str] = mapped_column(String(32), default="")
    message: Mapped[str] = mapped_column(Text, nullable=False)
    extra_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class ResearchPaper(Base):
    """
    Structured bibliographic record extracted from a crawled academic page.

    Persists across crawl sessions — every research crawl feeds the same
    central table so all papers are cross-session searchable.
    An FTS5 virtual table (research_papers_fts) is kept in sync via triggers
    created in storage.research.init_research_fts().
    """

    __tablename__ = "research_papers"
    __table_args__ = (
        Index("ix_research_papers_session", "session_id"),
        Index("ix_research_papers_year", "year"),
        Index("ix_research_papers_content_hash", "content_hash"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    page_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("visited_pages.id", ondelete="SET NULL"), nullable=True
    )

    # Bibliographic fields — all nullable; LLM confidence varies
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    authors: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    doi: Mapped[str | None] = mapped_column(String(256), nullable=True)
    arxiv_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    venue: Mapped[str | None] = mapped_column(String(512), nullable=True)
    keywords: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    pdf_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Provenance
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    schema_used: Mapped[str] = mapped_column(String(64), default="research_paper_v1")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )


class ResearchPaperData(BaseModel):
    """Structured data returned by ResearchExtractorAgent."""

    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    venue: str | None = None
    keywords: list[str] = Field(default_factory=list)
    pdf_url: str | None = None
    confidence: float = Field(default=0.0)
    explanation: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: Any) -> float:
        return max(0.0, min(1.0, float(v)))
