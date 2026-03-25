"""
workflows/simple.py — Single-agent crawl loop for shallow, straightforward tasks.

Flow: fetch → navigate → extract → store → repeat until goal is met,
pages are exhausted, or limits are reached.

Best for: "Get the price of X", "Extract the contact info from this page."
"""

from __future__ import annotations

import logging
from typing import Any

from agents.extractor import ExtractorAgent
from agents.navigator import NavigatorAgent, summarize_history
from config import AppConfig
from crawler.engine import fetch_page
from crawler.respectful import RespectfulCrawler
from llm.client import OllamaClient
from storage.db import (
    create_crawl_session,
    dequeue_next_url,
    enqueue_url,
    finish_crawl_session,
    get_all_extractions,
    get_pending_count,
    mark_url_done,
    mark_url_failed,
    save_extraction,
    save_page,
    update_session_stats,
)
from storage.models import (
    ExtractionResult,
    SessionStats,
    SessionStatus,
    URLItem,
    URLStatus,
)
from utils.progress import CrawlProgress, print_summary
from utils.url import normalize_url

logger = logging.getLogger(__name__)


async def run_simple(
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
    Execute the simple single-agent crawl workflow.

    Args:
        goal:        Natural language crawl goal.
        start_urls:  Seed URLs to begin crawling.
        config:      Full application config.
        session_id:  Resume an existing session (None = create new).
        max_pages:   Override config.crawl.max_pages.
        max_depth:   Override config.crawl.max_depth.
        model:       Override extractor model name.

    Returns:
        List of extracted data dicts from the session.
    """
    effective_max_pages = max_pages or config.crawl.max_pages
    effective_max_depth = max_depth or config.crawl.max_depth

    # Setup LLM client with optional model override
    llm_config = config.ollama
    if model:
        # Monkey-patch for this run only (config is immutable Pydantic model)
        from pydantic import BaseModel

        class _PatchedOllama(BaseModel):
            base_url: str = llm_config.base_url
            navigator_model: str = llm_config.navigator_model
            extractor_model: str = model
            router_model: str = llm_config.router_model
            timeout: int = llm_config.timeout
            max_retries: int = llm_config.max_retries

        llm_config = _PatchedOllama()

    llm = OllamaClient(llm_config)
    navigator = NavigatorAgent(llm)
    extractor = ExtractorAgent(llm)
    respectful = RespectfulCrawler(config.crawl)

    # Create or resume session
    if session_id is None:
        session_id = await create_crawl_session(
            goal=goal,
            workflow_used="simple",
            start_urls=start_urls,
        )
        # Seed the queue — normalize URLs before enqueuing
        for url in start_urls:
            await enqueue_url(URLItem(url=normalize_url(url), session_id=session_id, depth=0))
    else:
        logger.info("Resuming session %s", session_id)

    stats = SessionStats()
    visited_urls: list[str] = []

    with CrawlProgress(session_id, "simple") as progress:
        while stats.pages_crawled < effective_max_pages:
            item = await dequeue_next_url(session_id)
            if item is None:
                logger.info("Queue empty — crawl complete.")
                break

            stats.current_url = item.url
            stats.queue_size = await get_pending_count(session_id)
            progress.update(stats)

            # Pre-fetch checks (rate limit, robots, domain filter, depth)
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

            # Navigator decision
            history_summary = summarize_history(visited_urls)
            decision = await navigator.decide(
                url=item.url,
                markdown=page.markdown,
                goal=goal,
                links=page.links,
                history_summary=history_summary,
                content_hash=page.content_hash,
            )

            # Store page
            page_id = await save_page(page)
            await mark_url_done(item.url)
            visited_urls.append(item.url)
            stats.pages_crawled += 1

            # Extract if relevant enough
            # Use extract_chunks for long pages so data in later chunks isn't lost
            if decision.relevance_score >= 0.3:
                try:
                    extraction = await extractor.extract_chunks(
                        url=item.url,
                        markdown=page.markdown,
                        goal=goal,
                        content_hash=page.content_hash,
                    )
                    await save_extraction(page_id, extraction, session_id)
                    stats.extractions += 1
                    progress.print(
                        f"[green]Extracted[/green] from {item.url[:60]} "
                        f"(confidence: {extraction.confidence:.2f})"
                    )
                except Exception as exc:
                    logger.error("Extraction failed for %s: %s", item.url, exc)

            # Queue new links — normalize URLs to prevent duplicate queue entries
            if decision.action in ("deepen",) and item.depth < effective_max_depth:
                new_depth = item.depth + 1
                for link_priority in decision.links_to_follow[:10]:
                    queued = await enqueue_url(
                        URLItem(
                            url=normalize_url(link_priority.url),
                            priority=link_priority.priority,
                            depth=new_depth,
                            relevance_score=link_priority.estimated_value,
                            session_id=session_id,
                            parent_url=item.url,
                        )
                    )
                    if queued:
                        logger.debug("Queued: %s (priority %.2f)", link_priority.url, link_priority.priority)

            elif decision.action == "complete":
                progress.print("[bold green]Navigator signaled goal complete![/bold green]")
                break

            # Persist stats
            await update_session_stats(session_id, stats.model_dump())
            progress.update(stats)

    # Finalize session
    await finish_crawl_session(session_id, SessionStatus.completed)
    extractions = await get_all_extractions(session_id)
    print_summary(stats, extractions)
    return extractions
