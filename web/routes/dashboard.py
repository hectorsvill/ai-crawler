"""web/routes/dashboard.py — Main dashboard page and live-refresh partials."""

from __future__ import annotations

from fastapi import APIRouter, Request
from sqlalchemy import select

from storage.db import get_global_stats, get_recent_activity, get_session as db_session
from storage.models import CrawlSession
from web.app import templates

router = APIRouter(tags=["dashboard"])


@router.get("/")
async def dashboard(request: Request):
    async with db_session() as db:
        result = await db.execute(
            select(CrawlSession).order_by(CrawlSession.started_at.desc()).limit(20)
        )
        sessions = result.scalars().all()

    stats = await get_global_stats()
    recent_logs = await get_recent_activity(limit=50)

    session_list = []
    async with db_session() as db:
        from sqlalchemy import func as sqlfunc
        from storage.models import ExtractedData, URLRecord, URLStatus
        for s in sessions:
            pages = await db.scalar(
                select(sqlfunc.count()).where(
                    URLRecord.session_id == s.id,
                    URLRecord.status == URLStatus.done,
                )
            ) or 0
            extr = await db.scalar(
                select(sqlfunc.count()).where(ExtractedData.session_id == s.id)
            ) or 0
            session_list.append({
                "id": s.id,
                "goal": s.goal,
                "workflow_used": s.workflow_used,
                "status": s.status,
                "started_at": s.started_at,
                "ended_at": s.ended_at,
                "pages_crawled": pages,
                "extractions": extr,
                "stats": s.stats or {},
            })

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "sessions": session_list,
            "stats": stats,
            "recent_logs": recent_logs,
        },
    )


@router.get("/partials/stats")
async def stats_partial(request: Request):
    stats = await get_global_stats()
    return templates.TemplateResponse(request, "partials/stats_bar.html", {"stats": stats})


@router.get("/partials/activity")
async def activity_partial(request: Request):
    logs = await get_recent_activity(limit=50)
    return templates.TemplateResponse(request, "partials/activity_log.html", {"logs": logs})
