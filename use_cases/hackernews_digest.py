"""
use_cases/hackernews_digest.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Real-world use case: HackerNews Top Stories Digest

Crawls the HackerNews front page and extracts:
  - Story title
  - Score (points)
  - Number of comments
  - Submitter username
  - Link domain

Run:
    .venv/bin/python use_cases/hackernews_digest.py

Output:
    - Terminal summary table (Rich)
    - hn_digest.json  (all extracted records)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Allow importing project modules from the parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

console = Console()


GOAL = (
    "Extract every story on the HackerNews front page. "
    "For each story capture: title, score (points), comment_count, "
    "submitter username, and the link domain (e.g. github.com). "
    "Return as a list under the key 'stories'."
)

START_URL = "https://news.ycombinator.com/"
DB_PATH = "hn_digest.db"
OUTPUT_FILE = "hn_digest.json"


async def run() -> None:
    from config import load_config
    from llm.client import check_ollama_reachable
    from storage.db import close_db, get_all_extractions, init_db
    from workflows.simple import run_simple

    console.rule("[bold orange1]HackerNews Digest Crawler[/bold orange1]")
    console.print(f"  Goal:   [cyan]{GOAL[:80]}…[/cyan]")
    console.print(f"  URL:    [dim]{START_URL}[/dim]")
    console.print()

    config = load_config()

    if not await check_ollama_reachable(config.ollama.base_url):
        sys.exit(1)

    await init_db(DB_PATH)

    try:
        extractions = await run_simple(
            goal=GOAL,
            start_urls=[START_URL],
            config=config,
            max_pages=1,   # HN front page is one page
        )
    finally:
        await close_db()

    # ── Flatten results ────────────────────────────────────────────────────────
    stories: list[dict] = []
    for record in extractions:
        data = record.get("data", {})
        raw_stories = data.get("stories", [])
        if isinstance(raw_stories, list):
            stories.extend(raw_stories)
        elif isinstance(data, dict) and "title" in data:
            stories.append(data)

    if not stories:
        console.print("[yellow]No stories extracted. Try running with --verbose to debug.[/yellow]")
        return

    # ── Render Rich table ──────────────────────────────────────────────────────
    console.rule("[bold]Top Stories[/bold]")
    table = Table(show_lines=True, title=f"HackerNews Front Page — {len(stories)} stories")
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", max_width=50)
    table.add_column("Score", style="green", justify="right")
    table.add_column("Comments", style="yellow", justify="right")
    table.add_column("Domain", style="cyan")
    table.add_column("By", style="dim")

    for i, story in enumerate(stories[:30], 1):
        table.add_row(
            str(i),
            str(story.get("title", "?"))[:50],
            str(story.get("score", "?")),
            str(story.get("comment_count", story.get("comments", "?"))),
            str(story.get("link_domain", story.get("domain", "?"))),
            str(story.get("submitter", story.get("user", "?"))),
        )

    console.print(table)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w") as f:
        json.dump(stories, f, indent=2, default=str)
    console.print(f"\n[green]Saved {len(stories)} stories → {OUTPUT_FILE}[/green]")

    # Cleanup DB
    try:
        os.remove(DB_PATH)
    except OSError:
        pass


if __name__ == "__main__":
    asyncio.run(run())
