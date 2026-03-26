"""
workflows/research.py — Research-focused crawl workflow.

Specialised variant of the simple workflow that:
  - Uses ResearchExtractorAgent to extract bibliographic metadata
  - Saves results to the central research_papers table (cross-session searchable)
  - Deduplicates by DOI / arXiv ID across all previous sessions
  - Steers the Navigator toward abstract pages, DOI links, and PDF landing pages

Usage::

    from workflows.research import run_research
    results = await run_research(
        goal="Find papers on LLM prompt injection",
        start_urls=["https://arxiv.org/search/?query=prompt+injection"],
        config=config,
        max_pages=30,
    )
"""

from __future__ import annotations

import logging
from typing import Any

from agents.navigator import NavigatorAgent, summarize_history
from agents.research_extractor import ResearchExtractorAgent
from config import AppConfig
from crawler.engine import fetch_page
from crawler.respectful import RespectfulCrawler
from llm.client import OllamaClient
from storage.db import (
    create_crawl_session,
    dequeue_next_url,
    enqueue_url,
    finish_crawl_session,
    get_pending_count,
    mark_url_done,
    mark_url_failed,
    save_page,
    update_session_stats,
)
from storage.models import SessionStats, SessionStatus, URLItem, URLStatus
from storage.research import (
    get_research_stats,
    init_research_fts,
    list_research_papers,
    save_research_paper,
)
from utils.progress import CrawlProgress, print_summary
from utils.url import normalize_url

logger = logging.getLogger(__name__)

# Goal prefix injected to steer the Navigator toward academic content
_RESEARCH_GOAL_PREFIX = (
    "Prioritize: arXiv abstract pages, DOI landing pages, Semantic Scholar, "
    "PubMed, SSRN, and direct PDF links. "
    "Skip author profile pages, login walls, citation lists, and non-research pages. "
    "Original goal: "
)


async def run_research(
    goal: str,
    start_urls: list[str],
    config: AppConfig,
    session_id: str | None = None,
    *,
    max_pages: int | None = None,
    max_depth: int | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """
    Execute the research crawl workflow.

    Crawls academic sources, extracts bibliographic metadata, and stores
    results in the central research_papers table.

    Args:
        goal:        Natural language research goal.
        start_urls:  Seed URLs (arXiv search, Semantic Scholar, etc.).
        config:      Full application config.
        session_id:  Resume an existing session (None = create new).
        max_pages:   Override config.crawl.max_pages.
        max_depth:   Override config.crawl.max_depth.
        model:       Override extractor model name.

    Returns:
        List of ResearchPaper dicts saved during this session.
    """
    from storage.db import get_engine

    effective_max_pages = max_pages or config.crawl.max_pages
    effective_max_depth = max_depth or config.crawl.max_depth

    # Ensure FTS5 table exists
    engine = get_engine()
    await init_research_fts(engine)

    # Setup LLM
    llm_config = config.ollama
    if model:
        from pydantic import BaseModel as _BM

        class _Patched(_BM):
            base_url: str = llm_config.base_url
            navigator_model: str = llm_config.navigator_model
            extractor_model: str = model
            router_model: str = llm_config.router_model
            timeout: int = llm_config.timeout
            max_retries: int = llm_config.max_retries

        llm_config = _Patched()

    llm = OllamaClient(llm_config)
    navigator = NavigatorAgent(llm)
    extractor = ResearchExtractorAgent(llm)
    respectful = RespectfulCrawler(config.crawl)

    # Navigator goal steers toward academic content
    nav_goal = _RESEARCH_GOAL_PREFIX + goal

    # Create or resume session
    if session_id is None:
        session_id = await create_crawl_session(
            goal=goal,
            workflow_used="research",
            start_urls=start_urls,
        )
        for url in start_urls:
            await enqueue_url(URLItem(url=normalize_url(url), session_id=session_id, depth=0))
    else:
        logger.info("Resuming research session %s", session_id)

    stats = SessionStats()
    visited_urls: list[str] = []
    papers_saved = 0

    with CrawlProgress(session_id, "research") as progress:
        while stats.pages_crawled < effective_max_pages:
            item = await dequeue_next_url(session_id)
            if item is None:
                logger.info("Queue empty — research crawl complete.")
                break

            stats.current_url = item.url
            stats.queue_size = await get_pending_count(session_id)
            progress.update(stats)

            # Rate limit / robots / depth check
            allowed, reason = await respectful.check_and_wait(item.url, item.depth)
            if not allowed:
                logger.info("Skipping %s: %s", item.url, reason)
                await mark_url_failed(item.url, reason)
                stats.pages_skipped += 1
                continue

            # Fetch page
            try:
                use_playwright = respectful.needs_playwright(item.url)
                page = await fetch_page(
                    item.url,
                    user_agent=respectful.current_user_agent(),
                    force_playwright=use_playwright,
                )
            except Exception as exc:
                logger.error("Fetch failed for %s: %s", item.url, exc)
                await mark_url_failed(item.url, str(exc))
                stats.pages_failed += 1
                continue

            # Navigator decision (uses augmented goal to prefer academic links)
            history_summary = summarize_history(visited_urls)
            decision = await navigator.decide(
                url=item.url,
                markdown=page.markdown,
                goal=nav_goal,
                links=page.links,
                history_summary=history_summary,
                content_hash=page.content_hash,
            )

            # Store page and mark done
            page_id = await save_page(page)
            await mark_url_done(item.url)
            visited_urls.append(item.url)
            stats.pages_crawled += 1

            # Queue new links BEFORE extraction (so interrupts don't lose decisions)
            if decision.action == "deepen" and item.depth < effective_max_depth:
                new_depth = item.depth + 1
                for lp in decision.links_to_follow[:10]:
                    await enqueue_url(
                        URLItem(
                            url=normalize_url(lp.url),
                            priority=lp.priority,
                            depth=new_depth,
                            relevance_score=lp.estimated_value,
                            session_id=session_id,
                            parent_url=item.url,
                        )
                    )
            elif decision.action == "complete":
                progress.print("[bold green]Navigator signaled research goal complete![/bold green]")

            # Extract bibliographic metadata (threshold 0.3 — academic pages)
            if decision.relevance_score >= 0.3:
                try:
                    paper_data = await extractor.extract(
                        url=item.url,
                        markdown=page.markdown,
                        goal=goal,
                        content_hash=page.content_hash,
                    )
                    if paper_data.confidence >= 0.3:
                        paper_id = await save_research_paper(
                            paper_data=paper_data,
                            source_url=item.url,
                            session_id=session_id,
                            page_id=page_id,
                            content_hash=page.content_hash,
                        )
                        papers_saved += 1
                        stats.extractions += 1
                        title_short = (paper_data.title or item.url)[:55]
                        progress.print(
                            f"[green]Saved paper #{paper_id}[/green] {title_short} "
                            f"(conf: {paper_data.confidence:.2f})"
                        )
                except Exception as exc:
                    logger.error("Research extraction failed for %s: %s", item.url, exc)

            if decision.action == "complete":
                break

            await update_session_stats(session_id, stats.model_dump())
            progress.update(stats)

    await finish_crawl_session(session_id, SessionStatus.completed)

    # Print research-specific summary
    db_stats = await get_research_stats()
    from rich.console import Console
    from rich.panel import Panel
    Console().print(
        Panel(
            f"[bold]Research session complete[/bold]\n"
            f"Pages crawled:     {stats.pages_crawled}\n"
            f"Papers saved:      {papers_saved}\n"
            f"Total in DB:       {db_stats['total_papers']} papers "
            f"across {db_stats['sessions_with_papers']} session(s)\n"
            f"Year range:        {db_stats['year_range'][0]} – {db_stats['year_range'][1]}",
            title="Research Summary",
            border_style="green",
        )
    )

    return await list_research_papers(session_id=session_id)
