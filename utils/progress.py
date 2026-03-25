"""
utils/progress.py — Rich progress bars and live stats display.

Provides a CrawlProgress context manager that renders a live table
with crawl statistics updated in real time.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from storage.models import SessionStats

console = Console()


def _build_stats_table(stats: SessionStats, session_id: str, workflow: str) -> Panel:
    """Render a Rich table showing current crawl statistics."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold cyan", min_width=20)
    table.add_column("Value", style="white")

    table.add_row("Session ID", session_id[:16] + "...")
    table.add_row("Workflow", workflow)
    table.add_row("Pages Crawled", str(stats.pages_crawled))
    table.add_row("Failed", str(stats.pages_failed))
    table.add_row("Skipped", str(stats.pages_skipped))
    table.add_row("Extractions", str(stats.extractions))
    table.add_row("Queue Size", str(stats.queue_size))
    table.add_row(
        "Current URL",
        Text(stats.current_url[:80] + ("..." if len(stats.current_url) > 80 else ""), style="dim"),
    )
    table.add_row("Elapsed", f"{stats.elapsed_seconds:.1f}s")

    return Panel(table, title="[bold green]AI Crawler[/bold green]", border_style="green")


class CrawlProgress:
    """
    Live stats display for a crawl session.

    Usage:
        progress = CrawlProgress(session_id, workflow)
        async with progress:
            progress.update(stats)
    """

    def __init__(self, session_id: str, workflow: str) -> None:
        self.session_id = session_id
        self.workflow = workflow
        self._stats = SessionStats()
        self._start_time = time.monotonic()
        self._live: Live | None = None

    def start(self) -> None:
        self._live = Live(
            _build_stats_table(self._stats, self.session_id, self.workflow),
            console=console,
            refresh_per_second=2,
        )
        self._live.start()

    def stop(self) -> None:
        if self._live:
            self._live.stop()

    def update(self, stats: SessionStats) -> None:
        """Push updated stats to the live display."""
        stats.elapsed_seconds = time.monotonic() - self._start_time
        self._stats = stats
        if self._live:
            self._live.update(
                _build_stats_table(stats, self.session_id, self.workflow)
            )

    def print(self, message: str, style: str = "") -> None:
        """Print a message above the live display."""
        if self._live:
            self._live.console.print(message, style=style)
        else:
            console.print(message, style=style)

    def __enter__(self) -> "CrawlProgress":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def print_summary(stats: SessionStats, extractions: list[dict]) -> None:
    """Print a final summary table after crawl completes."""
    console.rule("[bold green]Crawl Complete[/bold green]")
    console.print(f"  Pages crawled:  [cyan]{stats.pages_crawled}[/cyan]")
    console.print(f"  Failed:         [red]{stats.pages_failed}[/red]")
    console.print(f"  Skipped:        [yellow]{stats.pages_skipped}[/yellow]")
    console.print(f"  Extractions:    [green]{stats.extractions}[/green]")
    console.print(f"  Total time:     [white]{stats.elapsed_seconds:.1f}s[/white]")

    # Print token usage report
    try:
        from llm.client import token_usage
        total = token_usage.prompt_tokens + token_usage.completion_tokens
        if total > 0:
            console.rule("[bold]Token Usage[/bold]")
            console.print(
                f"  Prompt:     [dim]{token_usage.prompt_tokens:,}[/dim]\n"
                f"  Completion: [dim]{token_usage.completion_tokens:,}[/dim]\n"
                f"  Total:      [bold]{total:,}[/bold]"
            )
            for model, counts in token_usage._per_model.items():
                m_total = counts["prompt"] + counts["completion"]
                console.print(
                    f"  [dim]{model}:[/dim] {m_total:,} tokens "
                    f"(p={counts['prompt']:,} c={counts['completion']:,})"
                )
    except Exception:
        pass

    if extractions:
        console.rule("[bold]Sample Extracted Data[/bold]")
        for i, item in enumerate(extractions[:3]):
            console.print(f"\n[bold]Result {i+1}[/bold] (confidence: {item.get('confidence', '?'):.2f})")
            import json
            try:
                console.print(json.dumps(item.get("data", {}), indent=2)[:500])
            except Exception:
                console.print(str(item)[:500])
