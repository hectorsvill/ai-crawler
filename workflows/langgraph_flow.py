"""
workflows/langgraph_flow.py — LangGraph stateful crawl workflow.

Defines a StateGraph with nodes: navigate → extract → store → decide_next.
Conditional edges route based on relevance scores.
Uses LangGraph's SQLite checkpointer for session resumability.

Graceful degradation: if langgraph is not installed, falls back to simple workflow
with a warning.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from config import AppConfig

logger = logging.getLogger(__name__)

try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.graph import END, StateGraph
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    logger.warning(
        "LangGraph not installed. Install with: pip install langgraph\n"
        "Falling back to simple workflow."
    )

# ── Graph state ────────────────────────────────────────────────────────────────

class CrawlState(TypedDict):
    """Shared state passed between LangGraph nodes."""
    goal: str
    session_id: str
    current_url: str
    current_depth: int
    markdown: str
    content_hash: str
    links: list[str]
    relevance_score: float
    action: str
    pages_crawled: int
    extractions: int
    max_pages: int
    max_depth: int
    visited_urls: list[str]
    queue: list[dict[str, Any]]  # list of {url, priority, depth}
    last_extraction: dict[str, Any]


# ── Node implementations ───────────────────────────────────────────────────────

async def fetch_page_node(state: CrawlState) -> CrawlState:
    """Fetch the current URL and update state with page content."""
    from crawler.engine import fetch_page
    from crawler.respectful import RespectfulCrawler

    # Import config for respectful crawling — stored in state via closure
    url = state["current_url"]
    logger.info("[LangGraph] Fetching: %s", url)

    try:
        page = await fetch_page(url, user_agent="Mozilla/5.0 (compatible; AICrawlerBot/1.0)")
        return {
            **state,
            "markdown": page.markdown,
            "content_hash": page.content_hash,
            "links": page.links,
        }
    except Exception as exc:
        logger.error("[LangGraph] Fetch failed for %s: %s", url, exc)
        return {**state, "markdown": "", "content_hash": "", "links": [], "action": "backtrack"}


async def navigate_node(state: CrawlState, llm: Any, navigator: Any) -> CrawlState:
    """Run the Navigator agent and update state with decision."""
    from agents.navigator import summarize_history

    if not state.get("markdown"):
        return {**state, "relevance_score": 0.0, "action": "backtrack"}

    decision = await navigator.decide(
        url=state["current_url"],
        markdown=state["markdown"],
        goal=state["goal"],
        links=state["links"],
        history_summary=summarize_history(state.get("visited_urls", [])),
        content_hash=state.get("content_hash", ""),
    )

    # Merge new high-priority links into queue
    new_queue = list(state.get("queue", []))
    current_depth = state.get("current_depth", 0)

    if decision.action == "deepen" and current_depth < state["max_depth"]:
        for lp in decision.links_to_follow[:10]:
            if lp.url not in state.get("visited_urls", []):
                new_queue.append({
                    "url": lp.url,
                    "priority": lp.priority,
                    "depth": current_depth + 1,
                })
        new_queue.sort(key=lambda x: x["priority"], reverse=True)

    return {
        **state,
        "relevance_score": decision.relevance_score,
        "action": decision.action,
        "queue": new_queue,
    }


async def extract_node(state: CrawlState, extractor: Any) -> CrawlState:
    """Run the Extractor agent and update state with extraction result."""
    result = await extractor.extract(
        url=state["current_url"],
        markdown=state["markdown"],
        goal=state["goal"],
        content_hash=state.get("content_hash", ""),
    )
    return {
        **state,
        "extractions": state.get("extractions", 0) + 1,
        "last_extraction": result.model_dump(),
    }


async def store_node(state: CrawlState) -> CrawlState:
    """Persist the current page and extraction to the database."""
    from storage.db import mark_url_done, save_extraction, save_page
    from storage.models import ExtractionResult, PageContent

    page = PageContent(
        url=state["current_url"],
        markdown=state.get("markdown", ""),
        content_hash=state.get("content_hash", ""),
        fetch_method="aiohttp",
        links=state.get("links", []),
    )
    page_id = await save_page(page)
    await mark_url_done(state["current_url"])

    if state.get("last_extraction") and state.get("relevance_score", 0) >= 0.2:
        extraction = ExtractionResult.model_validate(state["last_extraction"])
        await save_extraction(page_id, extraction, state["session_id"])

    visited = list(state.get("visited_urls", []))
    visited.append(state["current_url"])

    return {
        **state,
        "pages_crawled": state.get("pages_crawled", 0) + 1,
        "visited_urls": visited,
        "last_extraction": {},
    }


async def decide_next_node(state: CrawlState) -> CrawlState:
    """Pop the next URL from the queue and set it as current_url."""
    queue = list(state.get("queue", []))
    if not queue or state.get("pages_crawled", 0) >= state["max_pages"]:
        return {**state, "current_url": "", "action": "complete"}

    next_item = queue.pop(0)
    return {
        **state,
        "current_url": next_item["url"],
        "current_depth": next_item.get("depth", 0),
        "queue": queue,
        "action": "deepen",
        "markdown": "",
        "content_hash": "",
        "links": [],
    }


# ── Routing conditions ─────────────────────────────────────────────────────────

def should_extract(state: CrawlState) -> str:
    """Route after navigation: extract if relevant, else skip to store."""
    score = state.get("relevance_score", 0.0)
    action = state.get("action", "")

    if action == "complete":
        return "complete"
    if score >= 0.2:
        return "extract"
    return "store"  # skip extraction for low-relevance pages


def should_continue(state: CrawlState) -> str:
    """Route after store: continue or end."""
    if state.get("action") == "complete":
        return "end"
    if not state.get("current_url") or state.get("pages_crawled", 0) >= state["max_pages"]:
        return "end"
    return "continue"


# ── Graph factory ──────────────────────────────────────────────────────────────

def build_graph(llm: Any, navigator: Any, extractor: Any, checkpointer: Any = None) -> Any:
    """
    Construct and compile the LangGraph StateGraph.
    Accepts an optional checkpointer for persistence/resumability.
    Returns the compiled graph (or None if LangGraph is not installed).
    """
    if not HAS_LANGGRAPH:
        return None

    import functools

    graph = StateGraph(CrawlState)

    # Bind agent instances to node functions
    graph.add_node("fetch_page", fetch_page_node)
    graph.add_node(
        "navigate",
        functools.partial(navigate_node, llm=llm, navigator=navigator),
    )
    graph.add_node("extract", functools.partial(extract_node, extractor=extractor))
    graph.add_node("store", store_node)
    graph.add_node("decide_next", decide_next_node)

    # Entry point
    graph.set_entry_point("fetch_page")

    # Edges
    graph.add_edge("fetch_page", "navigate")
    graph.add_conditional_edges(
        "navigate",
        should_extract,
        {"extract": "extract", "store": "store", "complete": END},
    )
    graph.add_edge("extract", "store")
    graph.add_conditional_edges(
        "store",
        should_continue,
        {"continue": "decide_next", "end": END},
    )
    graph.add_edge("decide_next", "fetch_page")

    return graph.compile(checkpointer=checkpointer)


# ── Main entrypoint ────────────────────────────────────────────────────────────

async def run_langgraph(
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
    Execute the LangGraph stateful workflow.
    Falls back to simple workflow if LangGraph is unavailable.
    """
    if not HAS_LANGGRAPH:
        logger.warning("LangGraph unavailable — using simple workflow instead.")
        from workflows.simple import run_simple
        return await run_simple(
            goal, start_urls, config, session_id,
            max_pages=max_pages, max_depth=max_depth, model=model,
        )

    from llm.client import OllamaClient
    from agents.navigator import NavigatorAgent
    from agents.extractor import ExtractorAgent
    from storage.db import create_crawl_session, finish_crawl_session, get_all_extractions
    from storage.models import SessionStatus

    effective_max_pages = max_pages or config.crawl.max_pages
    effective_max_depth = max_depth or config.crawl.max_depth

    llm = OllamaClient(config.ollama)
    navigator = NavigatorAgent(llm)
    extractor = ExtractorAgent(llm)

    if session_id is None:
        session_id = await create_crawl_session(
            goal=goal, workflow_used="langgraph", start_urls=start_urls
        )

    # Seed URLs into DB queue
    from storage.db import enqueue_url
    from storage.models import URLItem
    for url in start_urls:
        await enqueue_url(URLItem(url=url, session_id=session_id, depth=0))

    initial_queue = [{"url": url, "priority": 1.0, "depth": 0} for url in start_urls]
    first_url = initial_queue[0]["url"] if initial_queue else ""

    initial_state: CrawlState = {
        "goal": goal,
        "session_id": session_id,
        "current_url": first_url,
        "current_depth": 0,
        "markdown": "",
        "content_hash": "",
        "links": [],
        "relevance_score": 0.0,
        "action": "deepen",
        "pages_crawled": 0,
        "extractions": 0,
        "max_pages": effective_max_pages,
        "max_depth": effective_max_depth,
        "visited_urls": [],
        "queue": initial_queue[1:],  # rest of seed URLs go to queue
        "last_extraction": {},
    }

    logger.info("[LangGraph] Starting graph execution for session %s", session_id)

    config_dict = {"configurable": {"thread_id": session_id}}

    try:
        async with AsyncSqliteSaver.from_conn_string(config.storage.db_path) as checkpointer:
            compiled = build_graph(llm, navigator, extractor, checkpointer=checkpointer)
            async for _ in compiled.astream(initial_state, config=config_dict):
                pass  # progress is tracked inside nodes
        await finish_crawl_session(session_id, SessionStatus.completed)
    except Exception as exc:
        logger.error("[LangGraph] Graph execution error: %s", exc)
        await finish_crawl_session(session_id, SessionStatus.failed)

    return await get_all_extractions(session_id)
