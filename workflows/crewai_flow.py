"""
workflows/crewai_flow.py — CrewAI multi-agent crawl workflow.

Defines four agents:
  - Navigator:   Decides which URLs to follow and scores relevance.
  - Extractor:   Extracts structured data from each page.
  - Researcher:  Synthesizes findings across pages into a coherent summary.
  - Summarizer:  Produces the final structured output report.

Graceful degradation: if crewai is not installed, falls back to simple workflow.
"""

from __future__ import annotations

import logging
from typing import Any

from config import AppConfig

logger = logging.getLogger(__name__)

try:
    from crewai import Agent, Crew, Process, Task
    from crewai.tools import BaseTool

    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False
    logger.warning(
        "CrewAI not installed. Install with: pip install crewai[tools]\n"
        "Falling back to simple workflow."
    )


# ── Custom CrewAI tools ────────────────────────────────────────────────────────

if HAS_CREWAI:
    class FetchPageTool(BaseTool):
        name: str = "fetch_page"
        description: str = (
            "Fetches a web page and returns its markdown content. "
            "Input: a URL string."
        )

        def _run(self, url: str) -> str:
            """Synchronous wrapper around the async fetch engine."""
            import asyncio

            from crawler.engine import fetch_page

            async def _fetch() -> str:
                page = await fetch_page(url, user_agent="Mozilla/5.0 (compatible; AICrawlerBot/1.0)")
                return f"URL: {url}\n\n{page.markdown[:4000]}"

            try:
                return asyncio.run(_fetch())
            except Exception as exc:
                return f"Error fetching {url}: {exc}"

    class StorageQueryTool(BaseTool):
        name: str = "query_extractions"
        description: str = (
            "Returns previously extracted data from the current crawl session. "
            "Useful for the Researcher to synthesize findings."
        )
        session_id: str = ""

        model_config = {"arbitrary_types_allowed": True}

        def _run(self, _: str = "") -> str:
            import asyncio
            import json
            from storage.db import get_all_extractions

            async def _query() -> list[dict]:
                return await get_all_extractions(self.session_id)

            try:
                results = asyncio.run(_query())
                return json.dumps(results[:20], indent=2)
            except Exception as exc:
                return f"Query error: {exc}"


# ── Crew builder ───────────────────────────────────────────────────────────────

def _build_crew(goal: str, start_urls: list[str], session_id: str, ollama_base_url: str) -> Any:
    """Construct and return a CrewAI Crew for the given goal."""
    from crewai import LLM

    fetch_tool = FetchPageTool()
    query_tool = StorageQueryTool(session_id=session_id)

    # CrewAI v1.x uses LLM object via LiteLLM; Ollama is supported via ollama/ prefix
    llm = LLM(
        model="ollama/qwen2.5:7b",
        base_url=ollama_base_url,
        api_key="ollama",  # required by LiteLLM but unused for local Ollama
    )

    navigator_agent = Agent(
        role="Web Navigator",
        goal=(
            "Determine which URLs from the provided list are most relevant to the research goal. "
            "Fetch pages and assess their relevance. Return a prioritized list of URLs."
        ),
        backstory=(
            "You are an expert web researcher who knows how to quickly assess page relevance "
            "and navigate websites efficiently to find the most valuable content."
        ),
        tools=[fetch_tool],
        llm=llm,
        verbose=False,
        max_iter=10,
    )

    extractor_agent = Agent(
        role="Data Extractor",
        goal=(
            "Extract structured, relevant data from web pages. "
            "Return clean JSON with only the information pertinent to the research goal."
        ),
        backstory=(
            "You are a meticulous data extraction specialist who transforms raw web content "
            "into structured, actionable information."
        ),
        tools=[fetch_tool],
        llm=llm,
        verbose=False,
        max_iter=15,
    )

    researcher_agent = Agent(
        role="Research Synthesizer",
        goal=(
            "Review all extracted data and synthesize it into a coherent research summary. "
            "Identify patterns, key findings, and gaps."
        ),
        backstory=(
            "You are a senior analyst who connects the dots between individual data points "
            "to produce actionable research insights."
        ),
        tools=[query_tool],
        llm=llm,
        verbose=False,
        max_iter=8,
    )

    summarizer_agent = Agent(
        role="Report Summarizer",
        goal=(
            "Produce a final, structured report summarizing all research findings. "
            "Format as clean JSON with a summary section."
        ),
        backstory=(
            "You are an expert report writer who distills complex research into "
            "clear, structured, actionable reports."
        ),
        tools=[],
        llm=llm,
        verbose=False,
        max_iter=5,
    )

    urls_str = "\n".join(f"- {u}" for u in start_urls[:10])

    navigate_task = Task(
        description=(
            f"Research goal: {goal}\n\n"
            f"Starting URLs:\n{urls_str}\n\n"
            "Fetch each starting URL. For each page, assess its relevance to the goal (0-10). "
            "Identify the top 5 most promising links for deeper research. "
            "Return a JSON list of {url, relevance_score, key_findings}."
        ),
        expected_output="JSON list of relevant URLs with relevance scores and initial findings.",
        agent=navigator_agent,
    )

    extract_task = Task(
        description=(
            f"Research goal: {goal}\n\n"
            "Using the Navigator's findings, fetch and extract detailed structured data "
            "from the top relevant pages. "
            "Return a JSON array of extracted records matching the goal's data requirements."
        ),
        expected_output="JSON array of extracted structured data records.",
        agent=extractor_agent,
        context=[navigate_task],
    )

    research_task = Task(
        description=(
            f"Research goal: {goal}\n\n"
            "Review all extracted data (use the query_extractions tool to fetch stored results). "
            "Synthesize the findings, identify the most important patterns, "
            "and note any gaps or contradictions. "
            "Return a JSON synthesis with: summary, key_findings[], gaps[], confidence."
        ),
        expected_output="JSON synthesis of all research findings with key insights.",
        agent=researcher_agent,
        context=[extract_task],
    )

    summarize_task = Task(
        description=(
            f"Research goal: {goal}\n\n"
            "Create a final research report based on all findings. "
            "Structure as JSON: {goal, summary, findings[], recommendations[], sources[]}. "
            "Be concise and actionable."
        ),
        expected_output="Final structured JSON report with summary, findings, and recommendations.",
        agent=summarizer_agent,
        context=[research_task],
    )

    return Crew(
        agents=[navigator_agent, extractor_agent, researcher_agent, summarizer_agent],
        tasks=[navigate_task, extract_task, research_task, summarize_task],
        process=Process.sequential,
        verbose=False,
    )


# ── Main entrypoint ────────────────────────────────────────────────────────────

async def run_crewai(
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
    Execute the CrewAI collaborative workflow.
    Falls back to simple workflow if CrewAI is unavailable.
    """
    if not HAS_CREWAI:
        logger.warning("CrewAI unavailable — using simple workflow instead.")
        from workflows.simple import run_simple
        return await run_simple(
            goal, start_urls, config, session_id,
            max_pages=max_pages, max_depth=max_depth, model=model,
        )

    import asyncio
    import json
    from storage.db import create_crawl_session, finish_crawl_session, get_all_extractions
    from storage.models import SessionStatus

    if session_id is None:
        session_id = await create_crawl_session(
            goal=goal, workflow_used="crewai", start_urls=start_urls
        )

    crew = _build_crew(goal, start_urls, session_id, config.ollama.base_url)

    logger.info("[CrewAI] Starting crew for session %s", session_id)

    try:
        # CrewAI is synchronous — run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, crew.kickoff)

        logger.info("[CrewAI] Crew completed. Raw output: %.200s", str(result))

        # Parse the final output as JSON if possible
        output_text = str(result)
        try:
            parsed = json.loads(output_text)
            if isinstance(parsed, list):
                crew_results = parsed
            else:
                crew_results = [parsed]
        except json.JSONDecodeError:
            crew_results = [{"raw_output": output_text, "goal": goal}]

        # Store crew results
        from storage.db import save_extraction, save_page
        from storage.models import ExtractionResult, PageContent
        for i, item in enumerate(crew_results):
            fake_page = PageContent(
                url=f"crew://session/{session_id}/result/{i}",
                markdown=str(item),
                content_hash=f"crew_{session_id}_{i}",
                fetch_method="crewai",
                links=[],
            )
            page_id = await save_page(fake_page)
            extraction = ExtractionResult(
                data=item if isinstance(item, dict) else {"result": item},
                schema_used="crewai_output",
                confidence=0.8,
                explanation="CrewAI multi-agent result",
            )
            from storage.db import save_extraction
            await save_extraction(page_id, extraction, session_id)

    except Exception as exc:
        logger.error("[CrewAI] Crew execution failed: %s", exc)

    await finish_crawl_session(session_id, SessionStatus.completed)
    return await get_all_extractions(session_id)
