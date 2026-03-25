"""
main.py — CLI entrypoint for the AI Web Crawler.

Commands:
  crawl   Run a crawl with a natural language goal.
  resume  Resume a paused or interrupted crawl session.
  list    List recent crawl sessions.
  export  Export extracted data from a session to JSON.

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
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Export extracted data from a session to a JSON file."""
    asyncio.run(_export(session_id, output, config_path))


async def _export(session_id: str, output: Path, config_path: Path | None) -> None:
    from config import load_config
    from storage.db import get_all_extractions, init_db

    config = load_config(config_path)
    await init_db(config.storage.db_path)

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


# ── Entrypoint ─────────────────────────────────────────────────────────────────

def main() -> None:
    app()


if __name__ == "__main__":
    main()
