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
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
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
        Index("ix_urls_status_priority", "status", "priority"),
        Index("ix_urls_session", "session_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(String(2048), unique=True, nullable=False)
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

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
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


class LinkPriority(BaseModel):
    """A discovered link with its priority assessment from the Navigator."""

    url: str
    priority: float = Field(ge=0.0, le=1.0)
    reasoning: str
    estimated_value: float = Field(ge=0.0, le=1.0)


class NavigatorDecision(BaseModel):
    """Decision returned by the Navigator agent for a given page."""

    relevance_score: float = Field(ge=0.0, le=1.0)
    links_to_follow: list[LinkPriority] = Field(default_factory=list)
    action: str  # "deepen" | "backtrack" | "complete"
    reasoning: str


class ExtractionResult(BaseModel):
    """Structured data returned by the Extractor agent."""

    data: dict[str, Any]
    schema_used: str
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str


class SessionStats(BaseModel):
    """Live statistics for a crawl session."""

    pages_crawled: int = 0
    pages_failed: int = 0
    pages_skipped: int = 0
    extractions: int = 0
    queue_size: int = 0
    current_url: str = ""
    elapsed_seconds: float = 0.0
