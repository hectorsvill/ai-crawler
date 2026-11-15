"""run_simple persists final stats even when the Navigator ends the loop with "complete"."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.models import ExtractionResult, LinkPriority, NavigatorDecision, PageContent  # noqa: E402


async def test_complete_branch_persists_stats(tmp_path, monkeypatch):
    import workflows.simple as ws
    from config import AppConfig
    from storage.db import close_db, get_session, init_db
    from storage.models import CrawlSession

    await init_db(str(tmp_path / "wf.db"))

    async def fake_fetch(url, user_agent, force_playwright=False, timeout=30):
        return PageContent(url=url, markdown=f"# page {url}\n" + "text " * 50,
                           content_hash=f"h-{url}", links=["https://ex.org/b"])

    class Resp:
        def __init__(self, cfg):
            pass

        async def check_and_wait(self, url, depth):
            return True, ""

        def needs_playwright(self, url):
            return False

        def current_user_agent(self):
            return "test-ua"

    class Nav:
        def __init__(self, llm):
            self.n = 0

        async def decide(self, **kw):
            self.n += 1
            if self.n == 1:
                return NavigatorDecision(relevance_score=0.9, action="deepen", links_to_follow=[
                    LinkPriority(url="https://ex.org/b", priority=0.9, reasoning="r")])
            return NavigatorDecision(relevance_score=0.9, action="complete")

    class Ext:
        def __init__(self, llm):
            pass

        async def extract_chunks(self, **kw):
            return ExtractionResult(data={"ok": True}, schema_used="s", confidence=0.9)

    monkeypatch.setattr(ws, "fetch_page", fake_fetch)
    monkeypatch.setattr(ws, "RespectfulCrawler", Resp)
    monkeypatch.setattr(ws, "NavigatorAgent", Nav)
    monkeypatch.setattr(ws, "ExtractorAgent", Ext)
    monkeypatch.setattr(ws, "make_llm_client", lambda cfg, model=None: object())

    try:
        await ws.run_simple("goal", ["https://ex.org/a"], AppConfig(), max_pages=3, max_depth=2)
        async with get_session() as db:
            from sqlalchemy import select
            sess = (await db.scalars(select(CrawlSession))).one()
        assert sess.stats["pages_crawled"] == 2
        assert sess.stats["extractions"] == 2
        assert sess.stats["elapsed_seconds"] > 0
    finally:
        await close_db()
