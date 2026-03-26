"""web/routes/sessions.py — Session detail page."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import func as sqlfunc, select

from storage.db import get_session as db_session, get_session_logs
from storage.models import (
    CrawlSession,
    ExtractedData,
    URLRecord,
    URLStatus,
    VisitedPage,
)
from web.app import templates

router = APIRouter(tags=["sessions"])


@router.get("/session/{session_id}")
async def session_detail(session_id: str, request: Request):
    async with db_session() as db:
        s = await db.scalar(select(CrawlSession).where(CrawlSession.id == session_id))
        if not s:
            raise HTTPException(404, "Session not found")

        pages_done = await db.scalar(
            select(sqlfunc.count()).where(
                URLRecord.session_id == session_id,
                URLRecord.status == URLStatus.done,
            )
        ) or 0
        pages_failed = await db.scalar(
            select(sqlfunc.count()).where(
                URLRecord.session_id == session_id,
                URLRecord.status == URLStatus.failed,
            )
        ) or 0
        pages_pending = await db.scalar(
            select(sqlfunc.count()).where(
                URLRecord.session_id == session_id,
                URLRecord.status == URLStatus.pending,
            )
        ) or 0
        extractions_count = await db.scalar(
            select(sqlfunc.count()).where(ExtractedData.session_id == session_id)
        ) or 0
        avg_conf = await db.scalar(
            select(sqlfunc.avg(ExtractedData.confidence)).where(
                ExtractedData.session_id == session_id
            )
        )

        # First 50 pages
        pages_result = await db.execute(
            select(URLRecord, VisitedPage)
            .outerjoin(VisitedPage, VisitedPage.url == URLRecord.url)
            .where(URLRecord.session_id == session_id)
            .order_by(URLRecord.created_at)
            .limit(50)
        )
        pages_rows = pages_result.all()

        pages = []
        for url_rec, vp in pages_rows:
            ec = 0
            mc = None
            if vp:
                ec = await db.scalar(
                    select(sqlfunc.count()).where(
                        ExtractedData.page_id == vp.id,
                        ExtractedData.session_id == session_id,
                    )
                ) or 0
                if ec:
                    mc = await db.scalar(
                        select(sqlfunc.max(ExtractedData.confidence)).where(
                            ExtractedData.page_id == vp.id,
                            ExtractedData.session_id == session_id,
                        )
                    )
            pages.append({
                "url": url_rec.url,
                "depth": url_rec.depth,
                "relevance_score": url_rec.relevance_score,
                "status": url_rec.status,
                "parent_url": url_rec.parent_url,
                "created_at": url_rec.created_at,
                "title": vp.title if vp else None,
                "fetch_method": vp.fetch_method if vp else None,
                "page_id": vp.id if vp else None,
                "extraction_count": ec,
                "max_confidence": round(mc, 3) if mc else None,
            })

        # Extractions
        extr_result = await db.execute(
            select(ExtractedData, VisitedPage)
            .outerjoin(VisitedPage, VisitedPage.id == ExtractedData.page_id)
            .where(ExtractedData.session_id == session_id)
            .order_by(ExtractedData.created_at.desc())
        )
        extractions = [
            {
                "id": ed.id,
                "url": vp.url if vp else None,
                "title": vp.title if vp else None,
                "data": ed.data,
                "schema_used": ed.schema_used,
                "confidence": ed.confidence,
                "created_at": ed.created_at,
            }
            for ed, vp in extr_result.all()
        ]

    # Logs (separate DB session)
    logs = await get_session_logs(session_id, limit=200)

    # Duration calc
    from datetime import datetime
    ended = s.ended_at
    started = s.started_at
    if started:
        end_t = ended or datetime.utcnow()
        secs = int((end_t - started).total_seconds())
        if secs < 60:
            duration = f"{secs}s"
        elif secs < 3600:
            duration = f"{secs // 60}m {secs % 60}s"
        else:
            duration = f"{secs // 3600}h {(secs % 3600) // 60}m"
    else:
        duration = "—"

    return templates.TemplateResponse(
        request,
        "session_detail.html",
        {
            "session": s,
            "pages": pages,
            "extractions": extractions,
            "logs": logs,
            "stats": {
                "pages_done": pages_done,
                "pages_failed": pages_failed,
                "pages_pending": pages_pending,
                "extractions": extractions_count,
                "avg_confidence": round(avg_conf, 3) if avg_conf else 0,
            },
            "duration": duration,
            "is_running": s.status == "running",
        },
    )
