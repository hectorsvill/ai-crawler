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
  usage            Token usage / estimated cost per provider, model, run, day.
  models           Live model list + default per provider (cached ~10 min).
  llm-ping         One tiny LLM call to verify a provider + key; records usage.
  doctor           Check environment readiness (deps, Ollama, models, DB).
  radar ...        Deterministic topic crawler with its own DB (no LLM needed).

Agent use: every command is non-interactive; data-returning commands accept
--json (data on stdout, human/progress output on stderr); live crawls accept
--dry-run; exit codes are defined in utils/exit_codes.py. See AGENTS.md.

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

from utils import exit_codes as rc

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
# Human-readable output goes to stderr so stdout stays clean for --json data.
console = Console(stderr=True)

from radar.cli import radar_app  # noqa: E402

app.add_typer(radar_app, name="radar")


def _emit_json(data) -> None:
    """Write machine-readable JSON to stdout."""
    sys.stdout.write(json.dumps(data, indent=2, default=str) + "\n")
    sys.stdout.flush()


def _fail(msg: str, code: int, as_json: bool = False) -> None:
    """Report an error (JSON on stdout when requested) and exit with *code*."""
    if as_json:
        _emit_json({"error": msg, "exit_code": code})
    else:
        console.print(f"[red]{msg}[/red]")
    raise typer.Exit(code)


NO_VALIDATE_HELP = "Skip checking --model against the provider's live model list."


def _validate_model(config, model: str | None, no_validate: bool, as_json: bool) -> dict | None:
    """Check --model against the provider's live list before any live call.

    Unknown model -> exit 2 with suggestions. If the list can't be fetched (no key,
    provider down) nothing is decided here; the readiness check reports that.
    """
    if not model or no_validate:
        return None
    from llm.models import check_model, list_models
    p = config.llm.provider
    base = config.ollama.base_url if p == "ollama" else config.llm.base_url
    listing = asyncio.run(list_models(p, base_url=base))
    chk = check_model(listing, model)
    if chk["valid"] is False:
        msg = f"Unknown model '{model}' for provider '{p}'."
        if chk["suggestions"]:
            msg += " Closest: " + ", ".join(chk["suggestions"])
        msg += f" (see: main.py models -P {p}; or pass --no-validate)"
        if as_json:
            _emit_json({"error": msg, "exit_code": rc.USAGE, "provider": p, "model": model,
                        "suggestions": chk["suggestions"]})
        else:
            console.print(f"[red]{msg}[/red]")
        raise typer.Exit(rc.USAGE)
    return chk


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
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Model override (Ollama: extractor model; API providers: model for all roles).",
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-P", envvar="CRAWLER_LLM__PROVIDER",
        help="LLM provider: ollama | gemini | anthropic | openai | deepseek.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate inputs, config and provider; print the plan; no crawl."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit a JSON result on stdout."),
    no_validate: bool = typer.Option(False, "--no-validate", help=NO_VALIDATE_HELP),
) -> None:
    """Start a new intelligent crawl session (LLM-driven: Ollama or an API provider)."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    _check_provider(provider, as_json)
    if workflow not in ("auto", "simple", "langgraph", "crewai"):
        _fail(f"Unknown workflow: {workflow}", rc.USAGE, as_json)
    if not start_url:
        _fail("No start URLs provided. Use --start-url/-u to specify at least one URL.",
              rc.USAGE, as_json)
    _validate_model(_load_config_with_provider(config_path, provider), model, no_validate, as_json)

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
            dry_run=dry_run,
            as_json=as_json,
            provider=provider,
        )
    )


def _check_provider(provider: str | None, as_json: bool = False) -> None:
    from llm.providers import PROVIDERS
    if provider and provider.lower() not in PROVIDERS:
        _fail(f"Unknown provider '{provider}'. Choose from: {', '.join(PROVIDERS)}",
              rc.USAGE, as_json)


def _load_config_with_provider(config_path, provider: str | None):
    from config import load_config
    config = load_config(config_path)
    if provider:
        config.llm.provider = provider.lower()
    return config


def _usage_summary(config, run_id: str | None) -> dict | None:
    """Token usage recorded for *run_id* (None if nothing recorded / no run)."""
    if not run_id:
        return None
    from llm.usage import UsageStore
    store = UsageStore(config.llm.usage_db_path or config.storage.db_path)
    return store.summary(run_id=run_id, group_by="model")


async def _run_crawl(
    goal: str,
    start_urls: list[str],
    workflow: str,
    resume: bool,
    config_path: Path | None,
    max_pages: int | None,
    max_depth: int | None,
    model: str | None,
    dry_run: bool = False,
    as_json: bool = False,
    provider: str | None = None,
) -> None:
    """Async implementation of the crawl command."""
    import contextlib

    from storage.db import init_db
    from workflows.router import select_workflow

    config = _load_config_with_provider(config_path, provider)

    # Fail fast if the provider is not usable (Ollama down / API key missing)
    from llm.client import check_llm_ready, resolve_models
    reachable, why = await check_llm_ready(config)

    if dry_run:
        plan = {
            "dry_run": True,
            "goal": goal,
            "start_urls": start_urls,
            "workflow": workflow,
            "resume": resume,
            "max_pages": max_pages or config.crawl.max_pages,
            "max_depth": max_depth or config.crawl.max_depth,
            "provider": config.llm.provider,
            "models": resolve_models(config, config.llm.provider, model),
            "provider_ready": reachable,
            "provider_problem": why or None,
            "ollama_base_url": config.ollama.base_url,
            "ollama_reachable": reachable if config.llm.provider == "ollama" else None,
            "db_path": config.storage.db_path,
        }
        if as_json:
            _emit_json(plan)
        else:
            for k, v in plan.items():
                console.print(f"  {k}: {v}")
        raise typer.Exit(rc.OK if reachable else rc.DEPENDENCY)

    if not reachable:
        _fail(why, rc.DEPENDENCY, as_json)

    if as_json:
        # Keep stdout clean for the final JSON result; progress goes to stderr.
        with contextlib.redirect_stdout(sys.stderr):
            await _run_crawl_inner(goal, start_urls, workflow, resume, config,
                                   max_pages, max_depth, model, init_db, select_workflow)
        session_id = await _find_latest_any_session(config.storage.db_path)
        _emit_json({"session_id": session_id, "db_path": config.storage.db_path,
                    "provider": config.llm.provider,
                    "summary": await _session_summary(session_id),
                    "llm_usage": _usage_summary(config, session_id)})
        return
    await _run_crawl_inner(goal, start_urls, workflow, resume, config,
                           max_pages, max_depth, model, init_db, select_workflow)


async def _find_latest_any_session(db_path: str) -> str | None:
    from sqlalchemy import select
    from storage.db import get_session
    from storage.models import CrawlSession
    async with get_session() as db:
        result = await db.execute(
            select(CrawlSession.id).order_by(CrawlSession.started_at.desc()).limit(1)
        )
        return result.scalar_one_or_none()


async def _session_summary(session_id: str | None) -> dict | None:
    if not session_id:
        return None
    from sqlalchemy import select
    from storage.db import get_session
    from storage.models import CrawlSession
    async with get_session() as db:
        s = await db.scalar(select(CrawlSession).where(CrawlSession.id == session_id))
    if not s:
        return None
    return {
        "id": s.id, "goal": s.goal, "workflow": s.workflow_used,
        "status": s.status.value if hasattr(s.status, "value") else s.status,
        "started_at": s.started_at, "ended_at": s.ended_at, "stats": s.stats or {},
        "start_urls": s.start_urls or [],
    }


async def _run_crawl_inner(goal, start_urls, workflow, resume, config,
                           max_pages, max_depth, model, init_db, select_workflow) -> None:
    await init_db(config.storage.db_path)

    console.rule("[bold green]AI Web Crawler[/bold green]")
    console.print(f"  Goal:     [cyan]{goal}[/cyan]")
    console.print(f"  Workflow: [yellow]{workflow}[/yellow]")

    if not start_urls:
        _fail("No start URLs provided. Use --start-url/-u to specify at least one URL.", rc.USAGE)

    for url in start_urls:
        console.print(f"  Seed URL: [dim]{url}[/dim]")
    console.print()

    # Set up LLM client for router
    from llm.client import make_llm_client
    llm = make_llm_client(config, model=model)

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
        _fail(f"Unknown workflow: {selected_workflow}", rc.USAGE)


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
    as_json: bool = typer.Option(False, "--json", help="Emit JSON on stdout."),
) -> None:
    """List recent crawl sessions."""
    asyncio.run(_list_sessions(config_path, limit, as_json))


async def _list_sessions(config_path: Path | None, limit: int, as_json: bool = False) -> None:
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

    if as_json:
        _emit_json([
            {
                "id": s.id, "goal": s.goal, "workflow": s.workflow_used,
                "status": s.status.value if hasattr(s.status, "value") else s.status,
                "started_at": s.started_at, "ended_at": s.ended_at,
                "pages_crawled": (s.stats or {}).get("pages_crawled", 0),
                "start_urls": s.start_urls or [],
            }
            for s in sessions
        ])
        return

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
    output: Path = typer.Option(
        Path("extracted_data.json"), "--output", "-o", help="Output file, or '-' for stdout."
    ),
    fmt: str = typer.Option("json", "--format", "-f", help="Output format: json | markdown"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Export crawl data from a session.

    --format json      (default) Structured JSON extractions from the LLM agent.
    --format markdown  Clean Markdown of every crawled page — ready for RAG,
                       vector DBs, and LLM pipelines.
    """
    if fmt not in ("json", "markdown"):
        _fail(f"Unknown format '{fmt}'. Use 'json' or 'markdown'.", rc.USAGE)
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

        if str(output) == "-":
            sys.stdout.write("\n".join(lines))
        else:
            output.write_text("\n".join(lines), encoding="utf-8")
        console.print(f"[green]Exported {len(pages)} pages as Markdown → {output}[/green]")

    else:
        extractions = await get_all_extractions(session_id)
        if str(output) == "-":
            _emit_json(extractions)
        else:
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
    provider: Optional[str] = typer.Option(
        None, "--provider", "-P", envvar="CRAWLER_LLM__PROVIDER",
        help="LLM provider: ollama | gemini | anthropic | openai | deepseek.",
    ),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model override."),
    no_validate: bool = typer.Option(False, "--no-validate", help=NO_VALIDATE_HELP),
) -> None:
    """Resume an interrupted crawl session."""
    _check_provider(provider)
    _validate_model(_load_config_with_provider(config_path, provider), model, no_validate, False)
    asyncio.run(_resume_session(session_id, config_path, max_pages, max_depth, provider, model))


async def _resume_session(
    session_id: str,
    config_path: Path | None,
    max_pages: int | None,
    max_depth: int | None,
    provider: str | None = None,
    model: str | None = None,
) -> None:
    from sqlalchemy import select
    from storage.db import get_session, init_db, resume_session
    from storage.models import CrawlSession

    config = _load_config_with_provider(config_path, provider)
    await init_db(config.storage.db_path)

    async with get_session() as db:
        session = await db.scalar(
            select(CrawlSession).where(CrawlSession.id == session_id)
        )

    if not session:
        _fail(f"Session {session_id} not found.", rc.NOT_FOUND)

    console.print(f"Resuming session [cyan]{session_id}[/cyan]: {session.goal}")

    from llm.client import check_llm_ready

    ready, why = await check_llm_ready(config)
    if not ready:
        _fail(why, rc.DEPENDENCY)
    workflow_mode = session.workflow_used

    kwargs = dict(
        goal=session.goal,
        start_urls=session.start_urls or [],
        config=config,
        session_id=session_id,
        max_pages=max_pages,
        max_depth=max_depth,
        model=model,
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
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model override."),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-P", envvar="CRAWLER_LLM__PROVIDER",
        help="LLM provider: ollama | gemini | anthropic | openai | deepseek.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate and print the plan; no crawl."),
    no_validate: bool = typer.Option(False, "--no-validate", help=NO_VALIDATE_HELP),
    as_json: bool = typer.Option(False, "--json", help="Errors as JSON on stdout (dry-run is always JSON)."),
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
    if not start_url:
        _fail("No start URLs provided. Use --start-url/-u", rc.USAGE, as_json)
    _check_provider(provider, as_json)
    cfg = _load_config_with_provider(config_path, provider)
    model_check = _validate_model(cfg, model, no_validate, as_json or dry_run)
    if dry_run:
        from llm.client import check_llm_ready, resolve_models
        ready, why = asyncio.run(check_llm_ready(cfg))
        _emit_json({"dry_run": True, "goal": goal, "start_urls": start_url,
                    "provider": cfg.llm.provider,
                    "models": resolve_models(cfg, cfg.llm.provider, model),
                    "model_check": model_check,
                    "provider_ready": ready, "provider_problem": why or None,
                    "max_pages": max_pages or cfg.crawl.max_pages,
                    "max_depth": max_depth or cfg.crawl.max_depth,
                    "db_path": cfg.storage.db_path})
        raise typer.Exit(rc.OK)
    asyncio.run(_run_research(goal, start_url, config_path, max_pages, max_depth, model,
                              provider))


async def _run_research(
    goal: str,
    start_urls: list[str],
    config_path: Path | None,
    max_pages: int | None,
    max_depth: int | None,
    model: str | None,
    provider: str | None = None,
) -> None:
    from llm.client import check_llm_ready
    from storage.db import init_db
    from workflows.research import run_research

    config = _load_config_with_provider(config_path, provider)

    ready, why = await check_llm_ready(config)
    if not ready:
        _fail(why, rc.DEPENDENCY)

    await init_db(config.storage.db_path)

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
    as_json: bool = typer.Option(False, "--json", help="Emit JSON on stdout."),
) -> None:
    """
    Full-text search across ALL stored research papers (cross-session).

    Supports FTS5 query syntax:
      research-search "prompt injection"
      research-search '"adversarial attack" AND transformer'
      research-search 'neural*'
    """
    asyncio.run(_research_search(query, session, limit, config_path, as_json))


async def _research_search(
    query: str,
    session_id: str | None,
    limit: int,
    config_path: Path | None,
    as_json: bool = False,
) -> None:
    from config import load_config
    from storage.db import init_db
    from storage.research import init_research_fts, search_research_papers

    config = load_config(config_path)
    engine_obj = await init_db(config.storage.db_path)
    await init_research_fts(engine_obj)

    results = await search_research_papers(query, limit=limit, session_id=session_id)

    if as_json:
        _emit_json(results)
        return
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
    as_json: bool = typer.Option(False, "--json", help="Emit JSON on stdout."),
) -> None:
    """Browse all stored research papers (newest first)."""
    asyncio.run(_research_list(session, year, limit, offset, config_path, as_json))


async def _research_list(
    session_id: str | None,
    year: int | None,
    limit: int,
    offset: int,
    config_path: Path | None,
    as_json: bool = False,
) -> None:
    from config import load_config
    from storage.db import init_db
    from storage.research import get_research_stats, init_research_fts, list_research_papers

    config = load_config(config_path)
    engine_obj = await init_db(config.storage.db_path)
    await init_research_fts(engine_obj)

    stats = await get_research_stats()
    papers = await list_research_papers(session_id=session_id, year=year, limit=limit, offset=offset)

    if as_json:
        _emit_json({"stats": stats, "papers": papers})
        return
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
    as_json: bool = typer.Option(False, "--json", help="Emit JSON on stdout."),
) -> None:
    """Show full details for one research paper."""
    asyncio.run(_research_show(paper_id, config_path, as_json))


async def _research_show(paper_id: int, config_path: Path | None, as_json: bool = False) -> None:
    from rich.panel import Panel

    from config import load_config
    from storage.db import init_db
    from storage.research import get_research_paper, init_research_fts

    config = load_config(config_path)
    engine_obj = await init_db(config.storage.db_path)
    await init_research_fts(engine_obj)

    paper = await get_research_paper(paper_id)
    if not paper:
        _fail(f"No paper found with ID {paper_id}", rc.NOT_FOUND, as_json)
    if as_json:
        _emit_json(paper)
        return

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


# ── Usage command ──────────────────────────────────────────────────────────────

@app.command()
def usage(
    provider: Optional[str] = typer.Option(None, "--provider", "-P", help="Filter by provider."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Filter by model."),
    since: Optional[str] = typer.Option(None, "--since", help="UTC date/ISO lower bound."),
    until: Optional[str] = typer.Option(None, "--until", help="UTC date (inclusive) / ISO upper bound."),
    run_id: Optional[str] = typer.Option(None, "--run", help="Crawl session id / run label."),
    group_by: str = typer.Option("model", "--group-by", "-g",
                                 help="provider | model | run | day | purpose | none"),
    calls: int = typer.Option(0, "--calls", help="Also list the N most recent calls."),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON on stdout."),
) -> None:
    """Token usage and estimated cost from the llm_usage table."""
    from config import load_config
    from llm.usage import UsageStore

    cfg = load_config(config_path)
    store = UsageStore(cfg.llm.usage_db_path or cfg.storage.db_path)
    try:
        data = store.summary(provider, model, since, until, run_id,
                             None if group_by == "none" else group_by)
    except ValueError as exc:
        _fail(str(exc), rc.USAGE, as_json)
    if calls:
        data["calls"] = store.rows(provider, model, since, until, run_id, limit=calls)
    if as_json:
        _emit_json(data)
        return
    t = data["totals"]
    console.print(f"calls={t['calls']} prompt={t['prompt_tokens']:,} "
                  f"completion={t['completion_tokens']:,} total={t['total_tokens']:,} "
                  f"cost_usd={t['cost_usd']}  db={data['db_path']}")
    table = Table(title=f"LLM usage by {group_by}")
    for col in ("group", "calls", "prompt", "completion", "total", "cost_usd"):
        table.add_column(col)
    for g in data["groups"]:
        label = " / ".join(str(g.get(k)) for k in ("provider", "model", "run_id", "day", "purpose")
                           if k in g)
        table.add_row(label, str(g["calls"]), f"{g['prompt_tokens']:,}",
                      f"{g['completion_tokens']:,}", f"{g['total_tokens']:,}", str(g["cost_usd"]))
    console.print(table)


# ── LLM ping command ───────────────────────────────────────────────────────────

@app.command(name="llm-ping")
def llm_ping(
    provider: Optional[str] = typer.Option(None, "--provider", "-P", envvar="CRAWLER_LLM__PROVIDER",
                                           help="ollama | gemini | anthropic | openai | deepseek"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model (default: provider default)."),
    prompt: str = typer.Option(
        'Return {"ok": true, "word": "<one word describing a web crawler>"}', "--prompt",
        help="Prompt to send (kept tiny by default)."),
    run_label: str = typer.Option("llm-ping", "--run", help="run_id label stored in llm_usage."),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be called; no request."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON on stdout."),
    no_validate: bool = typer.Option(False, "--no-validate", help=NO_VALIDATE_HELP),
) -> None:
    """Make ONE small LLM call to verify a provider/key; records tokens in llm_usage."""
    _check_provider(provider, as_json)
    from llm.client import check_llm_ready, make_llm_client, resolve_models
    from llm.providers import ProviderError
    from llm.usage import set_current_run

    cfg = _load_config_with_provider(config_path, provider)
    name = cfg.llm.provider
    from llm.providers import default_model
    if model:
        target = model
    elif name == "ollama":
        target = resolve_models(cfg, name)["extractor_model"]
    else:
        target = cfg.llm.model or default_model(name)
    model_check = _validate_model(cfg, model, no_validate, as_json)
    ready, why = asyncio.run(check_llm_ready(cfg))
    if dry_run:
        out = {"dry_run": True, "provider": name, "model": target, "ready": ready,
               "model_check": model_check,
               "problem": why or None, "prompt": prompt}
        if as_json:
            _emit_json(out)
        else:
            console.print(out)
        raise typer.Exit(rc.OK if ready else rc.DEPENDENCY)
    if not ready:
        _fail(why, rc.DEPENDENCY, as_json)

    cfg.llm.max_retries = 1
    cfg.ollama.max_retries = 1
    client = make_llm_client(cfg, model=target)

    async def _go() -> str:
        set_current_run(run_label)
        return await client.raw_call(target, [{"role": "user", "content": prompt}],
                                     purpose="llm-ping", json_mode=True)

    try:
        text = asyncio.run(_go())
    except (ProviderError, RuntimeError) as exc:
        _fail(str(exc), rc.ERROR, as_json)
    r = client.last_response
    out = {"provider": name, "model": target, "text": text,
           "usage": {"prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                     "total_tokens": r.total_tokens, "provider_reported": r.has_usage},
           "latency_ms": r.latency_ms,
           "usage_db": client.usage_store.path if client.usage_store else None}
    if as_json:
        _emit_json(out)
    else:
        console.print(out)


# ── Models command ─────────────────────────────────────────────────────────────

@app.command()
def models(
    provider: Optional[str] = typer.Option(None, "--provider", "-P",
                                           help="Only this provider (default: all)."),
    refresh: bool = typer.Option(False, "--refresh", help="Bypass the ~10 min cache."),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON on stdout."),
) -> None:
    """Live model list per provider (status, models, defaults). Keys are never shown."""
    _check_provider(provider, as_json)
    from config import load_config
    from llm.models import list_models
    from llm.providers import PROVIDERS
    cfg = load_config(config_path)
    names = [provider.lower()] if provider else list(PROVIDERS)

    async def _all():
        return await asyncio.gather(*[
            list_models(p, refresh=refresh,
                        base_url=cfg.ollama.base_url if p == "ollama"
                        else (cfg.llm.base_url if cfg.llm.provider == p else None))
            for p in names])

    results = asyncio.run(_all())
    for r in results:
        ids = {m["id"] for m in r["models"]}
        r["default_model_listed"] = (r["default_model"] in ids) if ids else None
        r["active"] = r["provider"] == cfg.llm.provider
    out = {"active_provider": cfg.llm.provider, "providers": results}
    if as_json:
        _emit_json(out)
    else:
        table = Table(title="LLM providers")
        for col in ("provider", "status", "default", "models"):
            table.add_column(col)
        for r in results:
            status = ("no key (" + "/".join(_key_names(r["provider"])) + ")" if not r["configured"]
                      else ("error: " + r["error"][:60]) if r["error"] else "ok")
            table.add_row(r["provider"] + (" *" if r["active"] else ""), status,
                          r["default_model"], str(len(r["models"])))
        console.print(table)
        for r in results:
            if r["models"]:
                console.print(f"[bold]{r['provider']}[/bold]: " + ", ".join(m["id"] for m in r["models"]))
    if provider and results[0]["error"]:
        raise typer.Exit(rc.DEPENDENCY)


def _key_names(provider: str) -> tuple[str, ...]:
    from llm.providers import KEY_ENV
    return KEY_ENV.get(provider, ()) or ("ollama not reachable",)


# ── Doctor command ─────────────────────────────────────────────────────────────

@app.command()
def doctor(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON on stdout."),
) -> None:
    """Check readiness: Python deps, Ollama + configured models, DB paths.

    Exit 0 if the radar crawler is usable; the `llm_crawl_ready` field says
    whether LLM commands (crawl/research) can run. Exit 3 if core deps are missing.
    """
    import importlib.util

    from config import load_config
    from radar.db import DEFAULT_DB_PATH

    cfg = load_config(config_path)
    core = ["aiohttp", "bs4", "trafilatura", "pydantic", "yaml", "sqlalchemy", "aiosqlite",
            "typer", "rich", "ollama", "tiktoken"]
    optional = ["playwright", "crawl4ai", "langgraph", "crewai", "fastapi", "uvicorn"]
    deps = {m: importlib.util.find_spec(m) is not None for m in core}
    opt = {m: importlib.util.find_spec(m) is not None for m in optional}

    async def _ollama() -> tuple[bool, list[str]]:
        import aiohttp
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(f"{cfg.ollama.base_url.rstrip('/')}/api/tags",
                                 timeout=aiohttp.ClientTimeout(total=5)) as r:
                    data = await r.json()
                    return r.status == 200, [m["name"] for m in data.get("models", [])]
        except Exception:
            return False, []

    reachable, models = asyncio.run(_ollama())
    from llm.client import resolve_models
    from llm.providers import key_status
    providers = key_status()
    providers["ollama"].update({"reachable": reachable})
    active = cfg.llm.provider
    wanted = {cfg.ollama.navigator_model, cfg.ollama.extractor_model, cfg.ollama.router_model}
    missing_models = sorted(m for m in wanted if m not in models)
    report = {
        "python": sys.version.split()[0],
        "core_deps": deps,
        "optional_deps": opt,
        "ollama": {"base_url": cfg.ollama.base_url, "reachable": reachable,
                   "installed_models": models, "missing_models": missing_models},
        "llm_provider": active,
        "llm_models": resolve_models(cfg, active),
        "providers": providers,
        "llm_db_path": cfg.storage.db_path,
        "radar_db_path": DEFAULT_DB_PATH,
        "radar_ready": all(deps.values()),
        "llm_crawl_ready": all(deps.values()) and (
            (reachable and not missing_models) if active == "ollama"
            else bool(providers.get(active, {}).get("configured"))
        ),
    }
    if as_json:
        _emit_json(report)
    else:
        for k, v in report.items():
            console.print(f"  [bold]{k}[/bold]: {v}")
    raise typer.Exit(rc.OK if report["radar_ready"] else rc.DEPENDENCY)


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
