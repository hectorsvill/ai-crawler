"""
main.py — CLI entrypoint for the AI Web Crawler.

Commands:
  crawl            Run a crawl with a natural language goal.
  resume           Resume a paused or interrupted crawl session.
  list-sessions    List recent crawl sessions.
  export           Export extracted data from a session to JSON.
  research         Crawl academic/P2P research sources and store papers.
  research-search  Full-text search across all stored research papers.
  research-list    Browse all stored research papers.
  research-show    Show full details for one paper by ID.

Uses Typer for CLI parsing and Rich for terminal output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Quiet noisy third-party loggers
for _noisy in ("aiohttp", "playwright", "httpx", "httpcore", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

app = typer.Typer(
    name="ai-crawler",
    help="Intelligent AI web crawler powered by local Ollama models.",
    add_completion=False,
)
console = Console()


# ── Crawl command ──────────────────────────────────────────────────────────────

@app.command()
def crawl(
    goal: str = typer.Option(..., "--goal", "-g", help="Natural language crawl goal."),
    start_url: list[str] = typer.Option(
        [], "--start-url", "-u", help="Seed URL(s) to begin crawling. Repeatable."
    ),
    workflow: str = typer.Option(
        "auto",
        "--workflow",
        "-w",
        help="Workflow: auto | simple | langgraph | crewai",
    ),
    resume: bool = typer.Option(False, "--resume", help="Resume the most recent session."),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to a custom config YAML file."
    ),
    max_pages: Optional[int] = typer.Option(None, "--max-pages", help="Override max pages."),
    max_depth: Optional[int] = typer.Option(None, "--max-depth", help="Override max depth."),
    model: Optional[str] = typer.Option(None, "--model", help="Override extractor model name."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Start a new intelligent crawl session."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    asyncio.run(
        _run_crawl(
            goal=goal,
            start_urls=start_url,
            workflow=workflow,
            resume=resume,
            config_path=config_path,
            max_pages=max_pages,
            max_depth=max_depth,
            model=model,
        )
    )


async def _run_crawl(
    goal: str,
    start_urls: list[str],
    workflow: str,
    resume: bool,
    config_path: Path | None,
    max_pages: int | None,
    max_depth: int | None,
    model: str | None,
) -> None:
    """Async implementation of the crawl command."""
    from config import load_config
    from storage.db import init_db
    from workflows.router import select_workflow

    config = load_config(config_path)

    # Fail fast with a helpful message if Ollama is unreachable
    from llm.client import check_ollama_reachable
    if not await check_ollama_reachable(config.ollama.base_url):
        raise typer.Exit(1)

    await init_db(config.storage.db_path)

    console.rule("[bold green]AI Web Crawler[/bold green]")
    console.print(f"  Goal:     [cyan]{goal}[/cyan]")
    console.print(f"  Workflow: [yellow]{workflow}[/yellow]")

    if not start_urls:
        console.print(
            "[red]No start URLs provided. Use --start-url/-u to specify at least one URL.[/red]"
        )
        raise typer.Exit(1)

    for url in start_urls:
        console.print(f"  Seed URL: [dim]{url}[/dim]")
    console.print()

    # Set up LLM client for router
    from llm.client import OllamaClient
    llm = OllamaClient(config.ollama)

    # Select workflow
    override = workflow if workflow != "auto" else None
    selected_workflow, reasoning = await select_workflow(goal, llm, override=override)
    console.print(f"  Selected workflow: [bold]{selected_workflow}[/bold]")
    console.print(f"  Reasoning: [dim]{reasoning}[/dim]\n")

    # Optionally resume
    session_id: str | None = None
    if resume:
        session_id = await _find_latest_session(config.storage.db_path)
        if session_id:
            console.print(f"  Resuming session: [cyan]{session_id}[/cyan]\n")
        else:
            console.print("  [yellow]No session to resume — starting fresh.[/yellow]\n")

    # Dispatch to workflow
    kwargs = dict(
        goal=goal,
        start_urls=start_urls,
        config=config,
        session_id=session_id,
        max_pages=max_pages,
        max_depth=max_depth,
        model=model,
    )

    if selected_workflow == "simple":
        from workflows.simple import run_simple
        await run_simple(**kwargs)

    elif selected_workflow == "langgraph":
        from workflows.langgraph_flow import run_langgraph
        await run_langgraph(**kwargs)

    elif selected_workflow == "crewai":
        from workflows.crewai_flow import run_crewai
        await run_crewai(**kwargs)

    else:
        console.print(f"[red]Unknown workflow: {selected_workflow}[/red]")
        raise typer.Exit(1)


async def _find_latest_session(db_path: str) -> str | None:
    """
    Return the most recent resumable session ID.

    Preference order:
    1. running / paused sessions (interrupted crawls)
    2. completed sessions that still have pending URLs (hit max-pages mid-crawl)
    """
    try:
        from sqlalchemy import func as sqlfunc, select
        from storage.db import get_session
        from storage.models import CrawlSession, SessionStatus, URLRecord, URLStatus

        # First: look for actively running / paused sessions
        async with get_session() as db:
            result = await db.execute(
                select(CrawlSession)
                .where(CrawlSession.status.in_([SessionStatus.running, SessionStatus.paused]))
                .order_by(CrawlSession.started_at.desc())
                .limit(1)
            )
            session = result.scalar_one_or_none()
            if session:
                return session.id

        # Second: look for recently completed sessions with pending URLs
        async with get_session() as db:
            result = await db.execute(
                select(CrawlSession)
                .where(CrawlSession.status == SessionStatus.completed)
                .order_by(CrawlSession.started_at.desc())
                .limit(5)
            )
            candidates = result.scalars().all()

        for candidate in candidates:
            async with get_session() as db:
                count = await db.scalar(
                    select(sqlfunc.count()).where(
                        URLRecord.session_id == candidate.id,
                        URLRecord.status == URLStatus.pending,
                    )
                )
            if count and count > 0:
                return candidate.id

        return None
    except Exception:
        return None


# ── List command ───────────────────────────────────────────────────────────────

@app.command()
def list_sessions(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of sessions to show."),
) -> None:
    """List recent crawl sessions."""
    asyncio.run(_list_sessions(config_path, limit))


async def _list_sessions(config_path: Path | None, limit: int) -> None:
    from config import load_config
    from sqlalchemy import select
    from storage.db import get_session, init_db
    from storage.models import CrawlSession

    config = load_config(config_path)
    await init_db(config.storage.db_path)

    async with get_session() as db:
        result = await db.execute(
            select(CrawlSession).order_by(CrawlSession.started_at.desc()).limit(limit)
        )
        sessions = result.scalars().all()

    if not sessions:
        console.print("[yellow]No sessions found.[/yellow]")
        return

    table = Table(title="Recent Crawl Sessions", show_lines=True)
    table.add_column("ID", style="cyan", max_width=16)
    table.add_column("Goal", max_width=40)
    table.add_column("Workflow", style="yellow")
    table.add_column("Status", style="green")
    table.add_column("Started", style="dim")
    table.add_column("Pages", style="white")

    for s in sessions:
        stats = s.stats or {}
        table.add_row(
            s.id[:16],
            s.goal[:40],
            s.workflow_used,
            s.status,
            s.started_at.strftime("%Y-%m-%d %H:%M") if s.started_at else "?",
            str(stats.get("pages_crawled", 0)),
        )
    console.print(table)


# ── Export command ─────────────────────────────────────────────────────────────

@app.command()
def export(
    session_id: str = typer.Argument(..., help="Session ID to export."),
    output: Path = typer.Option(Path("extracted_data.json"), "--output", "-o"),
    fmt: str = typer.Option("json", "--format", "-f", help="Output format: json | markdown"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Export crawl data from a session.

    --format json      (default) Structured JSON extractions from the LLM agent.
    --format markdown  Clean Markdown of every crawled page — ready for RAG,
                       vector DBs, and LLM pipelines.
    """
    if fmt not in ("json", "markdown"):
        console.print(f"[red]Unknown format '{fmt}'. Use 'json' or 'markdown'.[/red]")
        raise typer.Exit(1)
    asyncio.run(_export(session_id, output, fmt, config_path))


async def _export(session_id: str, output: Path, fmt: str, config_path: Path | None) -> None:
    from config import load_config
    from storage.db import get_all_extractions, get_all_pages_markdown, init_db

    config = load_config(config_path)
    await init_db(config.storage.db_path)

    if fmt == "markdown":
        pages = await get_all_pages_markdown(session_id)
        if not pages:
            console.print("[yellow]No crawled pages with markdown content found for this session.[/yellow]")
            raise typer.Exit(0)

        # Default output name when user didn't override
        if output == Path("extracted_data.json"):
            output = Path("crawled_pages.md")

        lines: list[str] = [
            f"# Crawled Pages — Session {session_id[:8]}\n",
            f"*{len(pages)} pages · exported by ai-crawler*\n",
            "\n---\n",
        ]
        for page in pages:
            title = page["title"] or page["url"]
            lines.append(f"\n## {title}\n")
            lines.append(f"**URL:** {page['url']}  \n")
            if page["fetched_at"]:
                lines.append(f"**Fetched:** {page['fetched_at']}  \n")
            lines.append("\n")
            lines.append(page["markdown"])
            lines.append("\n\n---\n")

        output.write_text("\n".join(lines), encoding="utf-8")
        console.print(f"[green]Exported {len(pages)} pages as Markdown → {output}[/green]")

    else:
        extractions = await get_all_extractions(session_id)
        with open(output, "w") as f:
            json.dump(extractions, f, indent=2, default=str)
        console.print(f"[green]Exported {len(extractions)} records to {output}[/green]")


# ── Resume command ─────────────────────────────────────────────────────────────

@app.command()
def resume(
    session_id: str = typer.Argument(..., help="Session ID to resume."),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    max_pages: Optional[int] = typer.Option(None, "--max-pages"),
    max_depth: Optional[int] = typer.Option(None, "--max-depth"),
) -> None:
    """Resume an interrupted crawl session."""
    asyncio.run(_resume_session(session_id, config_path, max_pages, max_depth))


async def _resume_session(
    session_id: str,
    config_path: Path | None,
    max_pages: int | None,
    max_depth: int | None,
) -> None:
    from config import load_config
    from sqlalchemy import select
    from storage.db import get_session, init_db, resume_session
    from storage.models import CrawlSession

    config = load_config(config_path)
    await init_db(config.storage.db_path)

    async with get_session() as db:
        session = await db.scalar(
            select(CrawlSession).where(CrawlSession.id == session_id)
        )

    if not session:
        console.print(f"[red]Session {session_id} not found.[/red]")
        raise typer.Exit(1)

    console.print(f"Resuming session [cyan]{session_id}[/cyan]: {session.goal}")

    from llm.client import OllamaClient
    from workflows.router import select_workflow

    llm = OllamaClient(config.ollama)
    workflow_mode = session.workflow_used

    kwargs = dict(
        goal=session.goal,
        start_urls=session.start_urls or [],
        config=config,
        session_id=session_id,
        max_pages=max_pages,
        max_depth=max_depth,
        model=None,
    )

    if workflow_mode == "langgraph":
        from workflows.langgraph_flow import run_langgraph
        await run_langgraph(**kwargs)
    elif workflow_mode == "crewai":
        from workflows.crewai_flow import run_crewai
        await run_crewai(**kwargs)
    else:
        from workflows.simple import run_simple
        await run_simple(**kwargs)


# ── Research command ───────────────────────────────────────────────────────────

@app.command()
def research(
    goal: str = typer.Option(..., "--goal", "-g", help="Research goal (natural language)."),
    start_url: list[str] = typer.Option(
        [], "--start-url", "-u", help="Seed URL(s). Repeatable."
    ),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    max_pages: Optional[int] = typer.Option(None, "--max-pages", help="Max pages to crawl."),
    max_depth: Optional[int] = typer.Option(None, "--max-depth", help="Max link depth."),
    model: Optional[str] = typer.Option(None, "--model", help="Override extractor model."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging."),
) -> None:
    """
    Crawl academic/P2P research sources and store papers in the central DB.

    All papers are deduplicated by DOI/arXiv ID across sessions and are
    full-text searchable via `research-search`.

    Example seeds:
      --start-url "https://arxiv.org/search/?query=peer+to+peer+network&searchtype=all"
      --start-url "https://scholar.google.com/scholar?q=distributed+systems"
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    asyncio.run(_run_research(goal, start_url, config_path, max_pages, max_depth, model))


async def _run_research(
    goal: str,
    start_urls: list[str],
    config_path: Path | None,
    max_pages: int | None,
    max_depth: int | None,
    model: str | None,
) -> None:
    from config import load_config
    from llm.client import check_ollama_reachable
    from storage.db import init_db
    from workflows.research import run_research

    config = load_config(config_path)

    if not await check_ollama_reachable(config.ollama.base_url):
        raise typer.Exit(1)

    await init_db(config.storage.db_path)

    if not start_urls:
        console.print("[red]No start URLs provided. Use --start-url/-u[/red]")
        raise typer.Exit(1)

    console.rule("[bold green]AI Research Crawler[/bold green]")
    console.print(f"  Goal:  [cyan]{goal}[/cyan]")
    for url in start_urls:
        console.print(f"  Seed:  [dim]{url}[/dim]")
    console.print()

    await run_research(
        goal=goal,
        start_urls=start_urls,
        config=config,
        max_pages=max_pages,
        max_depth=max_depth,
        model=model,
    )


# ── Research search command ────────────────────────────────────────────────────

@app.command(name="research-search")
def research_search(
    query: str = typer.Argument(..., help="FTS5 search query."),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Filter to one session."),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results."),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """
    Full-text search across ALL stored research papers (cross-session).

    Supports FTS5 query syntax:
      research-search "prompt injection"
      research-search '"adversarial attack" AND transformer'
      research-search 'neural*'
    """
    asyncio.run(_research_search(query, session, limit, config_path))


async def _research_search(
    query: str,
    session_id: str | None,
    limit: int,
    config_path: Path | None,
) -> None:
    from config import load_config
    from storage.db import init_db
    from storage.research import init_research_fts, search_research_papers

    config = load_config(config_path)
    engine_obj = await init_db(config.storage.db_path)
    await init_research_fts(engine_obj)

    results = await search_research_papers(query, limit=limit, session_id=session_id)

    if not results:
        console.print(f"[yellow]No papers found for: {query}[/yellow]")
        return

    _print_papers_table(results, title=f'Research: "{query}" — {len(results)} result(s)')


# ── Research list command ──────────────────────────────────────────────────────

@app.command(name="research-list")
def research_list(
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Filter to one session."),
    year: Optional[int] = typer.Option(None, "--year", "-y", help="Filter by publication year."),
    limit: int = typer.Option(50, "--limit", "-n", help="Max results."),
    offset: int = typer.Option(0, "--offset", help="Pagination offset."),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Browse all stored research papers (newest first)."""
    asyncio.run(_research_list(session, year, limit, offset, config_path))


async def _research_list(
    session_id: str | None,
    year: int | None,
    limit: int,
    offset: int,
    config_path: Path | None,
) -> None:
    from config import load_config
    from storage.db import init_db
    from storage.research import get_research_stats, init_research_fts, list_research_papers

    config = load_config(config_path)
    engine_obj = await init_db(config.storage.db_path)
    await init_research_fts(engine_obj)

    stats = await get_research_stats()
    papers = await list_research_papers(session_id=session_id, year=year, limit=limit, offset=offset)

    if not papers:
        console.print("[yellow]No research papers stored yet. Run `research` first.[/yellow]")
        return

    title = (
        f"Research Papers — {stats['total_papers']} total "
        f"({stats['sessions_with_papers']} session(s))"
    )
    _print_papers_table(papers, title=title)


# ── Research show command ──────────────────────────────────────────────────────

@app.command(name="research-show")
def research_show(
    paper_id: int = typer.Argument(..., help="Paper ID (from research-list)."),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Show full details for one research paper."""
    asyncio.run(_research_show(paper_id, config_path))


async def _research_show(paper_id: int, config_path: Path | None) -> None:
    from rich.panel import Panel

    from config import load_config
    from storage.db import init_db
    from storage.research import get_research_paper, init_research_fts

    config = load_config(config_path)
    engine_obj = await init_db(config.storage.db_path)
    await init_research_fts(engine_obj)

    paper = await get_research_paper(paper_id)
    if not paper:
        console.print(f"[red]No paper found with ID {paper_id}[/red]")
        raise typer.Exit(1)

    authors_str = ", ".join(paper.get("authors") or []) or "—"
    keywords_str = ", ".join(paper.get("keywords") or []) or "—"
    year_str = str(paper.get("year") or "—")
    doi_str = paper.get("doi") or "—"
    arxiv_str = paper.get("arxiv_id") or "—"
    venue_str = paper.get("venue") or "—"
    conf_str = f"{paper.get('confidence', 0):.2f}"
    abstract = paper.get("abstract") or "(no abstract extracted)"

    body = (
        f"[bold]Title:[/bold]    {paper.get('title') or '(unknown)'}\n"
        f"[bold]Authors:[/bold]  {authors_str}\n"
        f"[bold]Year:[/bold]     {year_str}\n"
        f"[bold]Venue:[/bold]    {venue_str}\n"
        f"[bold]DOI:[/bold]      {doi_str}\n"
        f"[bold]arXiv:[/bold]    {arxiv_str}\n"
        f"[bold]Keywords:[/bold] {keywords_str}\n"
        f"[bold]Confidence:[/bold] {conf_str}\n"
        f"[bold]URL:[/bold]      {paper.get('source_url', '')}\n"
        f"[bold]Session:[/bold]  {paper.get('session_id', '')[:16]}\n\n"
        f"[bold]Abstract:[/bold]\n{abstract}"
    )
    console.print(Panel(body, title=f"Paper #{paper_id}", border_style="cyan"))


# ── Shared rendering helper ────────────────────────────────────────────────────

def _print_papers_table(papers: list[dict], title: str = "Research Papers") -> None:
    """Render a list of paper dicts as a Rich table."""
    table = Table(title=title, show_lines=True)
    table.add_column("ID", style="cyan", width=5)
    table.add_column("Year", style="yellow", width=6)
    table.add_column("Title", max_width=55)
    table.add_column("First Author", max_width=22)
    table.add_column("Venue", max_width=25)
    table.add_column("DOI / arXiv", max_width=22)
    table.add_column("Conf", width=5)

    for p in papers:
        authors = p.get("authors") or []
        first_author = authors[0] if authors else "—"
        if len(authors) > 1:
            first_author += " et al."

        doi_arxiv = p.get("doi") or p.get("arxiv_id") or "—"

        table.add_row(
            str(p.get("id", "")),
            str(p.get("year") or "—"),
            (p.get("title") or p.get("source_url", ""))[:55],
            first_author[:22],
            (p.get("venue") or "—")[:25],
            doi_arxiv[:22],
            f"{p.get('confidence', 0):.2f}",
        )

    console.print(table)


# ── Web dashboard command ──────────────────────────────────────────────────────

@app.command()
def web(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to."),
    port: int = typer.Option(8420, "--port", "-p", help="Port to listen on."),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Config YAML path."),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (dev mode)."),
) -> None:
    """Launch the web dashboard at http://localhost:8420"""
    import os
    import uvicorn

    if config_path:
        os.environ["CRAWLER_CONFIG"] = str(config_path)

    console.rule("[bold cyan]AI Crawler Dashboard[/bold cyan]")
    console.print(f"  URL: [cyan]http://{'localhost' if host == '0.0.0.0' else host}:{port}[/cyan]")
    console.print()

    uvicorn.run(
        "web.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


# ── Entrypoint ─────────────────────────────────────────────────────────────────

def main() -> None:
    app()


if __name__ == "__main__":
    main()
