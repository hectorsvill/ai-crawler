"""
use_cases/wikipedia_research.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Real-world use case: Wikipedia Research Assistant

Builds a structured knowledge base on any topic by crawling Wikipedia
and following relevant links up to a configurable depth.

Run:
    .venv/bin/python use_cases/wikipedia_research.py "machine learning"
    .venv/bin/python use_cases/wikipedia_research.py "renewable energy" --pages 10
    .venv/bin/python use_cases/wikipedia_research.py "quantum computing" --depth 3

Output:
    - Live Rich progress panel
    - Token usage report
    - <topic>_research.json  (all extracted facts)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

console = Console()


def _wiki_url(topic: str) -> str:
    """Convert a topic string to a Wikipedia URL."""
    slug = topic.strip().replace(" ", "_")
    return f"https://en.wikipedia.org/wiki/{slug}"


def _safe_filename(topic: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")


async def run(topic: str, max_pages: int, max_depth: int) -> None:
    from config import load_config
    from llm.client import check_ollama_reachable
    from storage.db import close_db, get_all_extractions, init_db
    from workflows.simple import run_simple

    goal = (
        f"Build a comprehensive knowledge base on '{topic}'. "
        "For each page extract: "
        "page_title, summary (2-3 sentences), "
        "key_concepts (list of important terms), "
        "notable_entities (people, organizations, or places mentioned), "
        "related_topics (list of Wikipedia links that look relevant)."
    )

    start_url = _wiki_url(topic)
    db_path = f"{_safe_filename(topic)}_research.db"
    output_file = f"{_safe_filename(topic)}_research.json"

    console.rule(f"[bold blue]Wikipedia Research: {topic}[/bold blue]")
    console.print(f"  Start URL: [dim]{start_url}[/dim]")
    console.print(f"  Max pages: [cyan]{max_pages}[/cyan]   Max depth: [cyan]{max_depth}[/cyan]")
    console.print()

    config = load_config()

    if not await check_ollama_reachable(config.ollama.base_url):
        sys.exit(1)

    await init_db(db_path)

    try:
        extractions = await run_simple(
            goal=goal,
            start_urls=[start_url],
            config=config,
            max_pages=max_pages,
            max_depth=max_depth,
        )
    finally:
        await close_db()

    # ── Collect all facts ─────────────────────────────────────────────────────
    facts = [r["data"] for r in extractions if r.get("data")]

    if not facts:
        console.print("[yellow]No data extracted.[/yellow]")
        return

    # ── Print summary ─────────────────────────────────────────────────────────
    console.rule("[bold]Knowledge Base Summary[/bold]")
    console.print(f"  Pages processed:  [cyan]{len(facts)}[/cyan]")
    console.print(f"  Avg confidence:   "
                  f"[green]{sum(r.get('confidence', 0) for r in extractions) / max(len(extractions), 1):.2f}[/green]")

    all_concepts: list[str] = []
    all_entities: list[str] = []
    for fact in facts:
        all_concepts.extend(fact.get("key_concepts", []))
        all_entities.extend(fact.get("notable_entities", []))

    unique_concepts = list(dict.fromkeys(all_concepts))[:20]
    unique_entities = list(dict.fromkeys(all_entities))[:20]

    if unique_concepts:
        console.print(f"\n  [bold]Key Concepts:[/bold] {', '.join(unique_concepts)}")
    if unique_entities:
        console.print(f"  [bold]Notable Entities:[/bold] {', '.join(unique_entities)}")

    # Print page summaries
    console.print()
    for i, fact in enumerate(facts[:5], 1):
        title = fact.get("page_title", f"Page {i}")
        summary = fact.get("summary", "")
        console.print(f"  [bold cyan]{i}. {title}[/bold cyan]")
        if summary:
            console.print(f"     {summary[:200]}")

    if len(facts) > 5:
        console.print(f"  [dim]… and {len(facts) - 5} more pages[/dim]")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "topic": topic,
        "pages_crawled": len(facts),
        "knowledge_base": facts,
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    console.print(f"\n[green]Saved knowledge base → {output_file}[/green]")

    # Cleanup DB
    try:
        os.remove(db_path)
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Wikipedia Research Assistant")
    parser.add_argument("topic", help="Topic to research (e.g. 'machine learning')")
    parser.add_argument("--pages", type=int, default=5, help="Max pages to crawl (default: 5)")
    parser.add_argument("--depth", type=int, default=2, help="Max link depth (default: 2)")
    args = parser.parse_args()

    asyncio.run(run(args.topic, args.pages, args.depth))


if __name__ == "__main__":
    main()
