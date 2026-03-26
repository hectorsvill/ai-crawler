"""web/routes/search.py — Full-text search page and HTMX results partial."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from storage.db import search_fulltext
from web.app import templates

router = APIRouter(tags=["search"])


@router.get("/search")
async def search_page(request: Request, q: str = ""):
    hits: list[dict] = []
    if q.strip():
        hits = await search_fulltext(q, limit=50)
    return templates.TemplateResponse(
        request, "search_results.html", {"query": q, "hits": hits}
    )


@router.get("/search/results", response_class=HTMLResponse)
async def search_results_partial(request: Request, q: str = "", scope: str = "all"):
    hits: list[dict] = []
    if q.strip():
        hits = await search_fulltext(q, limit=50)
    return templates.TemplateResponse(
        request, "partials/search_hit.html", {"hits": hits, "query": q}
    )
