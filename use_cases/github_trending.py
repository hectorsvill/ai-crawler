"""
use_cases/github_trending.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Real-world use case: GitHub Trending Repositories

Crawls GitHub Trending and extracts:
  - Repository name and owner
  - Description
  - Programming language
  - Star count and stars gained today
  - Forks count

Run:
    .venv/bin/python use_cases/github_trending.py
    .venv/bin/python use_cases/github_trending.py --language python
    .venv/bin/python use_cases/github_trending.py --period weekly

Output:
    - Terminal table of trending repos
    - github_trending.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

console = Console()

GOAL = (
    "Extract every repository listed on GitHub Trending. "
    "For each repository capture: "
    "owner (GitHub username), repo_name, description, "
    "language (programming language), stars_total (integer), "
    "stars_today (integer stars gained today), forks (integer). "
    "Return as a list under the key 'repositories'."
)


def _build_url(language: str, period: str) -> str:
    base = "https://github.com/trending"
    parts = []
    if language:
        parts.append(language.lower().replace(" ", "-"))
    url = base + ("/" + parts[0] if parts else "")
    if period != "daily":
        url += f"?since={period}"
    return url


async def run(language: str, period: str) -> None:
    from config import load_config
    from llm.client import check_ollama_reachable
    from storage.db import close_db, get_all_extractions, init_db
    from workflows.simple import run_simple

    url = _build_url(language, period)
    db_path = "github_trending.db"
    output_file = "github_trending.json"

    lang_label = f" [{language}]" if language else ""
    console.rule(f"[bold green]GitHub Trending{lang_label} — {period}[/bold green]")
    console.print(f"  URL: [dim]{url}[/dim]")
    console.print()

    config = load_config()

    if not await check_ollama_reachable(config.ollama.base_url):
        sys.exit(1)

    # GitHub is JS-heavy — force Playwright
    config_dict = config.model_dump()
    config_dict["crawl"]["use_playwright_for"] = ["github.com"]
    from config import AppConfig
    patched_config = AppConfig.model_validate(config_dict)

    await init_db(db_path)

    try:
        extractions = await run_simple(
            goal=GOAL,
            start_urls=[url],
            config=patched_config,
            max_pages=1,   # trending is a single page
        )
    finally:
        await close_db()

    # ── Flatten repositories ───────────────────────────────────────────────────
    repos: list[dict] = []
    for record in extractions:
        data = record.get("data", {})
        raw_repos = data.get("repositories", [])
        if isinstance(raw_repos, list):
            repos.extend(raw_repos)
        elif isinstance(data, dict) and "repo_name" in data:
            repos.append(data)

    if not repos:
        console.print(
            "[yellow]No repositories extracted. GitHub may require JS rendering.\n"
            "Check that Playwright is installed: playwright install chromium[/yellow]"
        )
        return

    # ── Render Rich table ──────────────────────────────────────────────────────
    table = Table(
        show_lines=True,
        title=f"GitHub Trending{lang_label} ({period}) — {len(repos)} repos",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Repository", style="bold cyan", max_width=35)
    table.add_column("Description", max_width=40)
    table.add_column("Language", style="yellow")
    table.add_column("Stars", style="green", justify="right")
    table.add_column("+Today", style="green", justify="right")
    table.add_column("Forks", style="dim", justify="right")

    for i, repo in enumerate(repos[:25], 1):
        owner = repo.get("owner", "?")
        name = repo.get("repo_name", repo.get("name", "?"))
        table.add_row(
            str(i),
            f"{owner}/{name}",
            str(repo.get("description", ""))[:40],
            str(repo.get("language", "—")),
            str(repo.get("stars_total", repo.get("stars", "?"))),
            str(repo.get("stars_today", "?")),
            str(repo.get("forks", "?")),
        )

    console.print(table)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    with open(output_file, "w") as f:
        json.dump(repos, f, indent=2, default=str)
    console.print(f"\n[green]Saved {len(repos)} repositories → {output_file}[/green]")

    # Cleanup DB
    try:
        os.remove(db_path)
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="GitHub Trending Repositories")
    parser.add_argument("--language", default="", help="Filter by language (e.g. python, rust)")
    parser.add_argument(
        "--period",
        default="daily",
        choices=["daily", "weekly", "monthly"],
        help="Trending period (default: daily)",
    )
    args = parser.parse_args()
    asyncio.run(run(args.language, args.period))


if __name__ == "__main__":
    main()
