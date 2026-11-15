"""radar/cli.py — `main.py radar ...` subcommands. Agent-first: --json everywhere."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Optional

import typer

from utils import exit_codes as rc

radar_app = typer.Typer(
    name="radar",
    help="Deterministic topic crawler (no LLM) with its own SQLite DB (data/crawler.db).",
    no_args_is_help=True,
    add_completion=False,
)

DBOpt = typer.Option(None, "--db", envvar="CRAWLER_RADAR_DB", help="SQLite path (default data/crawler.db).")
JsonOpt = typer.Option(False, "--json", help="Emit machine-readable JSON on stdout.")


def emit(data: Any, as_json: bool, human: str | None = None) -> None:
    if as_json:
        sys.stdout.write(json.dumps(data, indent=2, default=str) + "\n")
    else:
        typer.echo(human if human is not None else json.dumps(data, indent=2, default=str))


def fail(msg: str, code: int, as_json: bool) -> None:
    if as_json:
        sys.stdout.write(json.dumps({"error": msg, "exit_code": code}) + "\n")
    else:
        typer.echo(f"error: {msg}", err=True)
    raise typer.Exit(code)


def _db(path: Optional[str]):
    from radar.db import RadarDB
    return RadarDB(path)


@radar_app.command("crawl")
def crawl_cmd(
    seeds: Optional[Path] = typer.Option(None, "--seeds", help="Seeds YAML (default radar/seeds.yaml)."),
    url: list[str] = typer.Option([], "--url", "-u", help="Ad-hoc seed URL (repeatable); replaces the seeds file."),
    kind: str = typer.Option("html", "--kind", help="Kind for --url seeds: html | feed."),
    topic: list[str] = typer.Option([], "--topic", "-t", help="Only seeds with this topic (repeatable). With --url: tag pages with it."),
    max_pages: int = typer.Option(200, "--max-pages", min=1, help="Hard cap on HTTP fetches (excl. robots.txt)."),
    max_depth: int = typer.Option(1, "--max-depth", min=1, max=3, help="Link depth from seed (1-3)."),
    max_items: int = typer.Option(10, "--max-items", min=1, help="Per-seed cap for --url seeds."),
    concurrency: int = typer.Option(2, "--concurrency", min=1, max=8, help="Global concurrent fetches."),
    delay: float = typer.Option(2.0, "--delay", min=0.5, help="Min seconds between requests per domain."),
    recent_days: int = typer.Option(7, "--recent-days", help="Skip feed items older than N days (0 = keep all)."),
    user_agent: Optional[str] = typer.Option(None, "--user-agent", envvar="CRAWLER_RADAR_UA", help="Override UA."),
    db: Optional[str] = DBOpt,
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the plan; no network, no DB writes."),
    strict: bool = typer.Option(False, "--strict", help="Exit 5 if any page failed."),
    as_json: bool = JsonOpt,
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Crawl seeds politely and upsert pages (tagged by topic) into the radar DB."""
    import logging
    from radar.crawler import DEFAULT_UA, CrawlParams, RadarCrawler, Seed, load_seeds, plan
    from radar.db import DEFAULT_DB_PATH

    logging.getLogger().setLevel(logging.DEBUG if verbose else logging.WARNING)
    if kind not in ("html", "feed"):
        fail("--kind must be html or feed", rc.USAGE, as_json)
    try:
        if url:
            seed_list = [Seed(url=u, kind=kind, topics=list(topic), max_items=max_items) for u in url]
        else:
            seed_list = load_seeds(seeds, topic or None)
    except FileNotFoundError as exc:
        fail(str(exc), rc.DEPENDENCY, as_json)
    except (TypeError, ValueError) as exc:
        fail(f"invalid seeds file: {exc}", rc.USAGE, as_json)
    if not seed_list:
        fail("no seeds selected", rc.USAGE, as_json)

    params = CrawlParams(
        max_pages=max_pages, max_depth=max_depth, concurrency=concurrency, delay=delay,
        recent_days=recent_days or None, user_agent=user_agent or DEFAULT_UA,
        topics=list(topic) or None,
    )
    db_path = db or DEFAULT_DB_PATH
    if dry_run:
        p = plan(seed_list, params, db_path)
        emit(p, as_json, human=(
            f"DRY RUN: {len(p['seeds'])} seeds across {len(p['domains'])} domains, "
            f"<= {p['max_possible_fetches']} fetches, db={db_path}\n"
            f"url log: {p['url_log']['table']} "
            f"(one row per URL encountered; no writes on dry-run)\n"
            + "\n".join(f"  [{s['kind']}] {s['url']}  topics={s['topics']}" for s in p["seeds"])))
        raise typer.Exit(rc.OK)

    with _db(db_path) as store:
        run_params = {"seeds": [s.url for s in seed_list], **params.__dict__}
        result = asyncio.run(RadarCrawler(store, params).run(seed_list, run_params))
    c = result["counters"]
    emit(result, as_json, human=(
        f"run {result['run_id']} {result['status']}: fetched={c['fetched']} stored={c['stored']} "
        f"(new={c['inserted']} updated={c['updated']} unchanged={c['unchanged']}) "
        f"failed={c['failed']} skipped={c['skipped']} domains={len(result['domains'])} "
        f"in {result['elapsed_s']}s → {result['db_path']}"))
    if result["status"] == "failed":
        raise typer.Exit(rc.ERROR)
    if strict and c["failed"]:
        raise typer.Exit(rc.PARTIAL)


@radar_app.command("pages")
def pages_cmd(
    topic: Optional[str] = typer.Option(None, "--topic", "-t"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d"),
    since: Optional[str] = typer.Option(None, "--since", help="published_at >= (YYYY-MM-DD or ISO)."),
    until: Optional[str] = typer.Option(None, "--until", help="published_at <= (YYYY-MM-DD inclusive)."),
    q: Optional[str] = typer.Option(None, "--q", help="Full-text query (FTS5 syntax)."),
    run_id: Optional[int] = typer.Option(None, "--run", help="Pages touched by this run id."),
    limit: int = typer.Option(50, "--limit", "-n", min=1),
    offset: int = typer.Option(0, "--offset", min=0),
    text: bool = typer.Option(False, "--text", help="Include Markdown body."),
    db: Optional[str] = DBOpt,
    as_json: bool = JsonOpt,
) -> None:
    """Query stored pages (newest published first)."""
    import sqlite3
    with _db(db) as store:
        try:
            rows = store.query_pages(topic, domain, since, until, q, run_id, limit, offset, text)
        except sqlite3.OperationalError as exc:
            fail(f"bad query: {exc}", rc.USAGE, as_json)
    emit(rows, as_json, human="\n".join(
        f"{r['id']:>5}  {(r['published_at'] or '-')[:10]}  {r['domain']:<22.22}  "
        f"{(r['title'] or r['url'])[:80]}  [{','.join(r['tags'])}]" for r in rows) or "(no pages)")


@radar_app.command("urls")
def urls_cmd(
    run_id: Optional[int] = typer.Option(None, "--run", help="Only URLs from this run id."),
    status: Optional[str] = typer.Option(
        None, "--status", "-s",
        help="stored | feed_only | failed | skipped | not_fetched | duplicate | fetched.",
    ),
    domain: Optional[str] = typer.Option(None, "--domain", "-d"),
    limit: int = typer.Option(100, "--limit", "-n", min=1),
    offset: int = typer.Option(0, "--offset", min=0),
    counts: bool = typer.Option(
        False, "--counts",
        help="With --run: per-status/reason counts reconciled with the run's counters.",
    ),
    db: Optional[str] = DBOpt,
    as_json: bool = JsonOpt,
) -> None:
    """List every URL a run encountered (not only stored pages)."""
    from radar.db import URL_STATUSES

    if status is not None and status not in URL_STATUSES:
        fail(
            f"unknown status {status!r}. Expected: {', '.join(URL_STATUSES)}",
            rc.USAGE, as_json,
        )
    with _db(db) as store:
        if run_id is not None and store.get_run(run_id) is None:
            fail(f"run {run_id} not found", rc.NOT_FOUND, as_json)
        if counts:
            if run_id is None:
                fail("--counts needs --run N", rc.USAGE, as_json)
            summary = store.url_counts(run_id)
            emit(summary, as_json, human=(
                f"run {run_id}: {summary['total']} urls  "
                + "  ".join(f"{k}={v}" for k, v in summary["by_status"].items() if v) + "\n"
                + "\n".join(f"  {k:<14} run={r['run']} url_log={r['url_log']}"
                             f"{'' if r['match'] else '  MISMATCH'}"
                             for k, r in summary["reconciliation"].items())))
            return
        try:
            rows = store.query_urls(run_id, status, domain, limit, offset)
        except ValueError as exc:
            fail(str(exc), rc.USAGE, as_json)
    emit(rows, as_json, human="\n".join(
        f"{r['run_id']:>4}  {r['status']:<12} d={r['depth']}  {r['domain']:<24.24}  {r['url']}"
        + (f"  ({r['reason']})" if r["reason"] else "")
        for r in rows) or "(no urls)")


@radar_app.command("show")
def show_cmd(
    page_id: Optional[int] = typer.Argument(None, help="Page id."),
    url: Optional[str] = typer.Option(None, "--url"),
    db: Optional[str] = DBOpt,
    as_json: bool = JsonOpt,
) -> None:
    """Show one page (with Markdown text) by id or --url."""
    if page_id is None and not url:
        fail("give a page id or --url", rc.USAGE, as_json)
    with _db(db) as store:
        page = store.get_page(page_id, url)
    if not page:
        fail("page not found", rc.NOT_FOUND, as_json)
    emit(page, as_json, human=(
        f"# {page['title']}\n{page['url']}\npublished: {page['published_at']}  "
        f"tags: {', '.join(page['tags'])}\n\n{page['text_md'] or ''}"))


@radar_app.command("export")
def export_cmd(
    fmt: str = typer.Option("json", "--format", "-f", help="json | jsonl | markdown"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="File path (default stdout)."),
    topic: Optional[str] = typer.Option(None, "--topic", "-t"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d"),
    since: Optional[str] = typer.Option(None, "--since"),
    until: Optional[str] = typer.Option(None, "--until"),
    q: Optional[str] = typer.Option(None, "--q"),
    run_id: Optional[int] = typer.Option(None, "--run"),
    limit: int = typer.Option(1000, "--limit", "-n", min=1),
    text: bool = typer.Option(True, "--text/--no-text", help="Include Markdown body."),
    db: Optional[str] = DBOpt,
) -> None:
    """Export pages as JSON / JSONL / Markdown (to stdout or --output)."""
    if fmt not in ("json", "jsonl", "markdown"):
        fail("--format must be json, jsonl or markdown", rc.USAGE, False)
    with _db(db) as store:
        rows = store.query_pages(topic, domain, since, until, q, run_id, limit, 0, text)
    if fmt == "json":
        out = json.dumps(rows, indent=2, default=str) + "\n"
    elif fmt == "jsonl":
        out = "".join(json.dumps(r, default=str) + "\n" for r in rows)
    else:
        out = "".join(
            f"## {r['title'] or r['url']}\n\n- URL: {r['url']}\n- Published: {r['published_at'] or '-'}\n"
            f"- Tags: {', '.join(r['tags'])}\n\n{r.get('text_md') or ''}\n\n---\n\n" for r in rows)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(out, encoding="utf-8")
        typer.echo(json.dumps({"exported": len(rows), "path": str(output), "format": fmt}), err=True)
    else:
        sys.stdout.write(out)


@radar_app.command("runs")
def runs_cmd(
    limit: int = typer.Option(20, "--limit", "-n", min=1),
    db: Optional[str] = DBOpt,
    as_json: bool = JsonOpt,
) -> None:
    """List crawl runs (newest first)."""
    with _db(db) as store:
        runs = store.list_runs(limit)
    emit(runs, as_json, human="\n".join(
        f"{r['id']:>4}  {r['started_at']}  {r['status']:<9} fetched={r['pages_fetched']} "
        f"stored={r['pages_stored']} failed={r['pages_failed']} skipped={r['pages_skipped']}"
        for r in runs) or "(no runs)")


@radar_app.command("stats")
def stats_cmd(db: Optional[str] = DBOpt, as_json: bool = JsonOpt) -> None:
    """DB totals: pages, domains, runs, tag counts."""
    with _db(db) as store:
        s = store.stats()
    emit(s, as_json, human=(
        f"db={s['db_path']} pages={s['pages']} (dated {s['pages_with_published_at']}) "
        f"domains={s['domains']} runs={s['runs']} url_log={s['url_log']}\n"
        + "\n".join(f"  {t['tag']:<24} {t['pages']}" for t in s["tags"])))


@radar_app.command("topics")
def topics_cmd(as_json: bool = JsonOpt) -> None:
    """List built-in topic tags and their keyword patterns."""
    from radar.topics import TOPIC_PATTERNS
    emit(TOPIC_PATTERNS, as_json, human="\n".join(TOPIC_PATTERNS))
