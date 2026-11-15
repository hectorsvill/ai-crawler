"""
web/routes/api.py — JSON API endpoints consumed by HTMX partials and external tools.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import delete, func as sqlfunc, select, update

from storage.db import (
    ensure_fts_and_logs,
    finish_crawl_session,
    get_global_stats,
    get_new_logs_since,
    get_session_logs,
    get_session as db_session,
    insert_crawl_log,
    populate_search_index,
    search_fulltext,
)
from storage.models import (
    CrawlLog,
    CrawlSession,
    ExtractedData,
    SessionStatus,
    URLRecord,
    URLStatus,
    VisitedPage,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["api"])


# ── Helpers ────────────────────────────────────────────────────────────────────

def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _duration_str(started: datetime | None, ended: datetime | None) -> str:
    if not started:
        return "—"
    end = ended or datetime.utcnow()
    secs = int((end - started).total_seconds())
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m {secs % 60}s"
    return f"{secs // 3600}h {(secs % 3600) // 60}m"


async def _session_summary(s: CrawlSession, db) -> dict:
    stats = s.stats or {}
    pages = await db.scalar(
        select(sqlfunc.count()).where(
            URLRecord.session_id == s.id,
            URLRecord.status == URLStatus.done,
        )
    ) or 0
    extractions = await db.scalar(
        select(sqlfunc.count()).where(ExtractedData.session_id == s.id)
    ) or 0
    return {
        "id": s.id,
        "goal": s.goal,
        "workflow_used": s.workflow_used,
        "status": s.status,
        "started_at": _iso(s.started_at),
        "ended_at": _iso(s.ended_at),
        "duration": _duration_str(s.started_at, s.ended_at),
        "pages_crawled": pages,
        "extractions": extractions,
        "tokens_used": stats.get("tokens_used", 0),
        "start_urls": s.start_urls or [],
        "stats": stats,
    }


# ── Global stats ──────────────────────────────────────────────────────────────

@router.get("/stats")
async def api_stats():
    return await get_global_stats()


# ── Sessions list ─────────────────────────────────────────────────────────────

@router.get("/sessions")
async def api_sessions(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    status: str | None = None,
):
    offset = (page - 1) * limit
    async with db_session() as db:
        q = select(CrawlSession).order_by(CrawlSession.started_at.desc())
        if status:
            q = q.where(CrawlSession.status == status)
        total = await db.scalar(
            select(sqlfunc.count(CrawlSession.id))
            .where(CrawlSession.status == status) if status
            else select(sqlfunc.count(CrawlSession.id))
        )
        q = q.offset(offset).limit(limit)
        result = await db.execute(q)
        sessions = result.scalars().all()
        rows = [await _session_summary(s, db) for s in sessions]

    return {"sessions": rows, "total": total or 0, "page": page, "limit": limit}


# ── Session detail ────────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}")
async def api_session(session_id: str):
    async with db_session() as db:
        s = await db.scalar(select(CrawlSession).where(CrawlSession.id == session_id))
        if not s:
            raise HTTPException(404, "Session not found")
        data = await _session_summary(s, db)

        # Additional detail fields
        failed = await db.scalar(
            select(sqlfunc.count()).where(
                URLRecord.session_id == session_id,
                URLRecord.status == URLStatus.failed,
            )
        ) or 0
        avg_conf = await db.scalar(
            select(sqlfunc.avg(ExtractedData.confidence)).where(
                ExtractedData.session_id == session_id
            )
        )
        domains_result = await db.execute(
            select(URLRecord.url).where(URLRecord.session_id == session_id)
        )
        urls = [r[0] for r in domains_result.fetchall()]
        domains = {u.split("/")[2] for u in urls if "://" in u}

        data.update({
            "pages_failed": failed,
            "avg_confidence": round(avg_conf, 3) if avg_conf else 0,
            "unique_domains": len(domains),
        })
    return data


# ── Session pages ─────────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/pages")
async def api_session_pages(
    session_id: str,
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
):
    offset = (page - 1) * limit
    async with db_session() as db:
        total = await db.scalar(
            select(sqlfunc.count()).where(URLRecord.session_id == session_id)
        ) or 0
        result = await db.execute(
            select(URLRecord, VisitedPage)
            .outerjoin(VisitedPage, VisitedPage.url == URLRecord.url)
            .where(URLRecord.session_id == session_id)
            .order_by(URLRecord.created_at)
            .offset(offset)
            .limit(limit)
        )
        rows = result.all()

        pages = []
        for url_rec, vp in rows:
            extraction_count = 0
            max_conf = None
            if vp:
                extraction_count = await db.scalar(
                    select(sqlfunc.count()).where(
                        ExtractedData.page_id == vp.id,
                        ExtractedData.session_id == session_id,
                    )
                ) or 0
                if extraction_count:
                    max_conf = await db.scalar(
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
                "created_at": _iso(url_rec.created_at),
                "title": vp.title if vp else None,
                "fetch_method": vp.fetch_method if vp else None,
                "page_id": vp.id if vp else None,
                "extraction_count": extraction_count,
                "max_confidence": round(max_conf, 3) if max_conf else None,
            })

    return {"pages": pages, "total": total, "page": page, "limit": limit}


# ── Session extractions ───────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/extractions")
async def api_session_extractions(session_id: str):
    async with db_session() as db:
        result = await db.execute(
            select(ExtractedData, VisitedPage)
            .outerjoin(VisitedPage, VisitedPage.id == ExtractedData.page_id)
            .where(ExtractedData.session_id == session_id)
            .order_by(ExtractedData.created_at.desc())
        )
        rows = result.all()

    extractions = [
        {
            "id": ed.id,
            "page_id": ed.page_id,
            "url": vp.url if vp else None,
            "title": vp.title if vp else None,
            "data": ed.data,
            "schema_used": ed.schema_used,
            "confidence": ed.confidence,
            "created_at": _iso(ed.created_at),
        }
        for ed, vp in rows
    ]
    return {"extractions": extractions}


# ── Session logs ──────────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/logs")
async def api_session_logs(
    session_id: str,
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    component: str | None = None,
):
    logs = await get_session_logs(
        session_id, limit=limit, offset=offset, component=component
    )
    return {"logs": logs}


# ── Session tree ──────────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/tree")
async def api_session_tree(session_id: str):
    async with db_session() as db:
        result = await db.execute(
            select(
                URLRecord.url,
                URLRecord.parent_url,
                URLRecord.relevance_score,
                URLRecord.depth,
                URLRecord.status,
            ).where(URLRecord.session_id == session_id)
            .order_by(URLRecord.depth, URLRecord.relevance_score.desc())
        )
        records = result.all()

    node_map: dict[str, dict] = {}
    for r in records:
        node_map[r.url] = {
            "url": r.url,
            "parent": r.parent_url,
            "relevance": r.relevance_score,
            "depth": r.depth,
            "status": r.status,
            "children": [],
        }

    roots: list[dict] = []
    for r in records:
        node = node_map[r.url]
        if r.parent_url and r.parent_url in node_map:
            node_map[r.parent_url]["children"].append(node)
        else:
            roots.append(node)

    return {"tree": roots}


# ── Export ────────────────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/export")
async def api_export(session_id: str, format: str = Query("json")):
    async with db_session() as db:
        result = await db.execute(
            select(ExtractedData, VisitedPage)
            .outerjoin(VisitedPage, VisitedPage.id == ExtractedData.page_id)
            .where(ExtractedData.session_id == session_id)
        )
        rows = result.all()

    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["url", "title", "confidence", "schema_used", "data_json"])
        for ed, vp in rows:
            writer.writerow([
                vp.url if vp else "",
                vp.title if vp else "",
                ed.confidence,
                ed.schema_used or "",
                json.dumps(ed.data),
            ])
        output.seek(0)
        return StreamingResponse(
            iter([output.read()]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="session_{session_id[:8]}.csv"'},
        )

    data = [
        {
            "url": vp.url if vp else None,
            "title": vp.title if vp else None,
            "confidence": ed.confidence,
            "schema_used": ed.schema_used,
            "data": ed.data,
        }
        for ed, vp in rows
    ]
    return JSONResponse(
        content=data,
        headers={"Content-Disposition": f'attachment; filename="session_{session_id[:8]}.json"'},
    )


# ── Delete session ────────────────────────────────────────────────────────────

@router.delete("/sessions/{session_id}")
async def api_delete_session(session_id: str):
    from sqlalchemy import text as sa_text
    async with db_session() as db:
        s = await db.scalar(select(CrawlSession).where(CrawlSession.id == session_id))
        if not s:
            raise HTTPException(404, "Session not found")
        await db.execute(sa_text("DELETE FROM search_index WHERE session_id = :sid"), {"sid": session_id})
        await db.execute(delete(CrawlLog).where(CrawlLog.session_id == session_id))
        await db.execute(delete(ExtractedData).where(ExtractedData.session_id == session_id))
        await db.execute(delete(URLRecord).where(URLRecord.session_id == session_id))
        await db.execute(delete(CrawlSession).where(CrawlSession.id == session_id))
        # Clean up VisitedPage rows that are no longer referenced by any ExtractedData
        await db.execute(
            sa_text(
                "DELETE FROM visited_pages WHERE id NOT IN "
                "(SELECT DISTINCT page_id FROM extracted_data)"
            )
        )
    return {"status": "deleted", "session_id": session_id}


# ── Launch crawl ──────────────────────────────────────────────────────────────

class CrawlRequest(BaseModel):
    goal: str
    start_url: str
    workflow: str = "auto"
    max_pages: int | None = None
    max_depth: int | None = None
    navigator_model: str | None = None
    extractor_model: str | None = None
    provider: str | None = None   # ollama | gemini | anthropic | openai | deepseek
    model: str | None = None      # API providers: all roles; Ollama: extractor model
    no_validate: bool = False


@router.post("/crawl")
async def api_start_crawl(request: Request):
    """Accept both JSON body and HTML form submissions."""
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
    else:
        form = await request.form()
        body = {k: v for k, v in form.items() if v != ""}
        # Convert numeric strings
        for field in ("max_pages", "max_depth"):
            if field in body and body[field]:
                try:
                    body[field] = int(body[field])
                except (ValueError, TypeError):
                    body.pop(field, None)
    if isinstance(body.get("no_validate"), str):
        body["no_validate"] = body["no_validate"].lower() in ("1", "true", "on", "yes")
    req = CrawlRequest(**{k: v for k, v in body.items() if k in CrawlRequest.model_fields})
    if req.workflow not in ("auto", "simple", "langgraph", "crewai", "research"):
        raise HTTPException(400, f"Unknown workflow '{req.workflow}'")
    from llm.providers import PROVIDERS
    config = request.app.state.config
    provider = (req.provider or config.llm.provider).lower()
    if provider not in PROVIDERS:
        raise HTTPException(400, f"Unknown provider '{provider}'. Choose from: {', '.join(PROVIDERS)}")
    if provider != "ollama":
        from llm.providers import api_key_for
        if not api_key_for(provider)[0]:
            raise HTTPException(400, f"Provider '{provider}' has no API key configured on the server")
    if req.model and not req.no_validate:
        from llm.models import check_model, list_models
        listing = await list_models(provider, base_url=_base_url_for(config, provider))
        chk = check_model(listing, req.model)
        if chk["valid"] is False:
            return JSONResponse(status_code=400, content={
                "error": f"Unknown model '{req.model}' for provider '{provider}'",
                "provider": provider, "model": req.model, "suggestions": chk["suggestions"]})
    from storage.db import create_crawl_session, enqueue_url
    from storage.models import URLItem
    from utils.url import normalize_url

    session_id = await create_crawl_session(
        goal=req.goal,
        workflow_used=req.workflow if req.workflow != "auto" else "simple",
        start_urls=[req.start_url],
    )
    await enqueue_url(URLItem(url=normalize_url(req.start_url), session_id=session_id, depth=0))
    await insert_crawl_log(session_id, f"Crawl queued via web UI: {req.goal}", component="web")

    task = asyncio.create_task(
        _run_crawl_background(
            goal=req.goal,
            start_url=req.start_url,
            workflow=req.workflow,
            session_id=session_id,
            config=config,
            max_pages=req.max_pages,
            max_depth=req.max_depth,
            navigator_model=req.navigator_model,
            extractor_model=req.extractor_model,
            running_crawls=request.app.state.running_crawls,
            provider=provider,
            model=req.model,
        )
    )
    request.app.state.running_crawls[session_id] = task
    return {"session_id": session_id, "status": "started", "workflow": req.workflow,
            "provider": provider, "model": req.model,
            "usage_url": f"/api/usage?run_id={session_id}"}


def _base_url_for(config: Any, provider: str) -> str | None:
    if provider == "ollama":
        return config.ollama.base_url
    return config.llm.base_url if config.llm.provider == provider else None


async def _run_crawl_background(
    goal: str,
    start_url: str,
    workflow: str,
    session_id: str,
    config: Any,
    max_pages: int | None,
    max_depth: int | None,
    navigator_model: str | None,
    extractor_model: str | None,
    running_crawls: dict,
    provider: str | None = None,
    model: str | None = None,
) -> None:
    try:
        await insert_crawl_log(session_id, f"Starting: {goal}", component="workflow")

        # Per-run copy of the config: provider + model overrides never leak into app state
        run_config = config.model_copy(deep=True)
        if provider:
            run_config.llm.provider = provider
        if navigator_model:
            run_config.ollama.navigator_model = navigator_model
        if extractor_model:
            run_config.ollama.extractor_model = extractor_model
        await insert_crawl_log(
            session_id, f"LLM provider={run_config.llm.provider} model={model or 'default'}",
            component="web")

        # Select workflow
        selected = workflow
        if workflow == "auto":  # (research is explicit, never routed)
            try:
                from llm.client import make_llm_client
                from workflows.router import select_workflow
                llm = make_llm_client(run_config)
                selected, reasoning = await select_workflow(goal, llm)
                await insert_crawl_log(
                    session_id,
                    f"Router selected '{selected}': {reasoning}",
                    component="router",
                )
            except Exception as exc:
                selected = "simple"
                await insert_crawl_log(
                    session_id, f"Router failed ({exc}), using simple", component="router", level="warning"
                )

        kwargs = dict(
            goal=goal,
            start_urls=[start_url],
            config=run_config,
            session_id=session_id,
            max_pages=max_pages,
            max_depth=max_depth,
            model=model,
        )

        if selected == "research":
            from workflows.research import run_research
            await run_research(**kwargs)
        elif selected == "langgraph":
            try:
                from workflows.langgraph_flow import run_langgraph
                await run_langgraph(**kwargs)
            except ImportError:
                from workflows.simple import run_simple
                await run_simple(**kwargs)
        elif selected == "crewai":
            try:
                from workflows.crewai_flow import run_crewai
                await run_crewai(**kwargs)
            except ImportError:
                from workflows.simple import run_simple
                await run_simple(**kwargs)
        else:
            from workflows.simple import run_simple
            await run_simple(**kwargs)

        await insert_crawl_log(session_id, "Crawl completed", component="workflow")

    except asyncio.CancelledError:
        await finish_crawl_session(session_id, SessionStatus.paused)
        await insert_crawl_log(session_id, "Crawl stopped by user", component="workflow", level="warning")
    except Exception as exc:
        logger.exception("Background crawl %s failed", session_id)
        await finish_crawl_session(session_id, SessionStatus.failed)
        await insert_crawl_log(session_id, f"Crawl failed: {exc}", component="workflow", level="error")
    finally:
        running_crawls.pop(session_id, None)


# ── Stop crawl ────────────────────────────────────────────────────────────────

@router.post("/crawl/{session_id}/stop")
async def api_stop_crawl(session_id: str, request: Request):
    task = request.app.state.running_crawls.get(session_id)
    if task:
        task.cancel()
        return {"status": "stopping", "session_id": session_id}
    raise HTTPException(404, "No running crawl with that session ID")


# ── Analyse goal (workflow router preview) ────────────────────────────────────

@router.post("/crawl/analyze")
async def api_analyze_goal(request: Request):
    body = await request.json()
    goal = body.get("goal", "")
    if not goal.strip():
        return {"workflow": "simple", "reasoning": "No goal provided — defaulting to simple."}
    config = request.app.state.config
    try:
        from llm.client import make_llm_client
        from workflows.router import select_workflow
        llm = make_llm_client(config)
        workflow, reasoning = await select_workflow(goal, llm)
        return {"workflow": workflow, "reasoning": reasoning}
    except Exception as exc:
        return {"workflow": "simple", "reasoning": f"Router unavailable: {exc}"}


# ── LLM providers / models / usage (keys never leave the server) ────────────

@router.get("/models")
async def api_models(request: Request, provider: str | None = Query(None),
                     refresh: bool = Query(False)):
    """No provider: status of every provider. With provider: its live model list."""
    from llm.models import all_providers_status, list_models
    from llm.providers import PROVIDERS
    config = request.app.state.config
    if provider:
        provider = provider.lower()
        if provider not in PROVIDERS:
            raise HTTPException(400, f"Unknown provider '{provider}'")
        r = await list_models(provider, base_url=_base_url_for(config, provider),
                              refresh=refresh, timeout=10)
        ids = {m["id"] for m in r["models"]}
        r["default_model_listed"] = (r["default_model"] in ids) if ids else None
        return r
    providers = await all_providers_status(with_models=False)
    for p in providers:
        if p["provider"] == "ollama":  # "configured" == reachable for the local server
            r = await list_models("ollama", base_url=config.ollama.base_url, timeout=3)
            p["configured"] = r["configured"] and not r["error"]
        p["status"] = "ok" if p["configured"] else (
            "not reachable" if p["provider"] == "ollama" else "no key")
    return {"active_provider": config.llm.provider, "providers": providers}


@router.get("/usage")
async def api_usage(request: Request, provider: str | None = Query(None),
                    model: str | None = Query(None), run_id: str | None = Query(None),
                    since: str | None = Query(None), group_by: str = Query("model"),
                    limit: int = Query(20, ge=0, le=500)):
    """Token usage (llm_usage table): totals, groups and the most recent calls."""
    from llm.usage import UsageStore
    config = request.app.state.config
    store = UsageStore(config.llm.usage_db_path or config.storage.db_path)
    try:
        out = store.summary(provider=provider, model=model, since=since, run_id=run_id,
                            group_by=group_by or None)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    out["recent"] = store.rows(provider=provider, model=model, since=since, run_id=run_id,
                               limit=limit)
    out.pop("db_path", None)
    return out


# ── Search ────────────────────────────────────────────────────────────────────

@router.get("/search")
async def api_search(
    q: str = Query(""),
    scope: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    session_id: str | None = None,
):
    if not q.strip():
        return {"hits": [], "query": q}
    hits = await search_fulltext(q, limit=limit, session_id_filter=session_id)
    return {"hits": hits, "query": q, "count": len(hits)}


# ── Rebuild search index ──────────────────────────────────────────────────────

@router.post("/sessions/{session_id}/reindex")
async def api_reindex(session_id: str):
    count = await populate_search_index(session_id)
    return {"session_id": session_id, "indexed": count}


# ── SSE live progress ─────────────────────────────────────────────────────────

@router.get("/live/{session_id}")
async def api_live(session_id: str, request: Request):
    async def event_stream():
        last_id = 0
        while True:
            if await request.is_disconnected():
                break
            async with db_session() as db:
                s = await db.scalar(
                    select(CrawlSession).where(CrawlSession.id == session_id)
                )
            if not s:
                yield f"event: error\ndata: {json.dumps({'error': 'Session not found'})}\n\n"
                break

            new_logs = await get_new_logs_since(session_id, last_id)
            if new_logs:
                last_id = new_logs[-1]["id"]

            payload = {
                "status": s.status,
                "stats": s.stats or {},
                "logs": new_logs,
            }
            yield f"data: {json.dumps(payload)}\n\n"

            if s.status in (SessionStatus.completed, SessionStatus.failed, SessionStatus.paused):
                yield f"event: done\ndata: {json.dumps({'status': s.status})}\n\n"
                break

            await asyncio.sleep(1.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
