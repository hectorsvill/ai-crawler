"""
agents/navigator.py — Navigator agent.

Receives a page's markdown content, the crawl goal, history summary, and
discovered links. Returns a NavigatorDecision with relevance score, prioritized
links to follow, and a recommended action (deepen / backtrack / complete).

Optimized for token efficiency — uses the faster navigator model.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from storage.models import LinkPriority, NavigatorDecision

if TYPE_CHECKING:
    from llm.client import OllamaClient

logger = logging.getLogger(__name__)

# ── Prompt templates ───────────────────────────────────────────────────────────

NAVIGATOR_SYSTEM_PROMPT = """You are a web crawl navigator. Given a page's content and a research goal, you must:
1. Assess how relevant this page is to the goal (0.0 = irrelevant, 1.0 = highly relevant).
2. From the provided link list, select the most promising URLs to follow next.
3. Decide whether to deepen (follow more links), backtrack (this path is unproductive), or complete (goal achieved).

Be concise. Prioritize breadth of relevant content over depth of irrelevant paths.
Focus only on URLs that are likely to yield goal-relevant information.
Limit links_to_follow to at most 10 entries."""

NAVIGATOR_USER_TEMPLATE = """## Crawl Goal
{goal}

## Current URL
{url}

## Crawl History Summary
{history_summary}

## Page Content (truncated)
{markdown}

## Links Found on This Page
{links_list}

Analyze the page and return your NavigatorDecision JSON."""


class NavigatorAgent:
    """
    LLM-powered agent that decides which links to follow next.

    Designed to minimize token usage by using a small/fast model
    and concise prompts.
    """

    def __init__(self, llm: "OllamaClient") -> None:
        self.llm = llm

    async def decide(
        self,
        url: str,
        markdown: str,
        goal: str,
        links: list[str],
        history_summary: str = "",
        content_hash: str = "",
    ) -> NavigatorDecision:
        """
        Analyze the current page and return a NavigatorDecision.

        Falls back to a safe default decision on LLM errors to prevent
        crawl interruption.
        """
        # Format links for the prompt (limit to 50 to save tokens)
        links_display = links[:50]
        links_list = "\n".join(f"- {link}" for link in links_display)
        if len(links) > 50:
            links_list += f"\n... and {len(links) - 50} more"

        user_content = NAVIGATOR_USER_TEMPLATE.format(
            goal=goal,
            url=url,
            history_summary=history_summary or "First page in session.",
            markdown=markdown[:3000],  # pre-trim before chunking
            links_list=links_list or "(no links found)",
        )

        try:
            decision = await self.llm.structured_call(
                model=self.llm.navigator_model,
                system_prompt=NAVIGATOR_SYSTEM_PROMPT,
                user_content=user_content,
                response_schema=NavigatorDecision,
                cache_key=content_hash if content_hash else None,
                goal=goal,
            )
            # Validate and sanitize link URLs
            decision.links_to_follow = _filter_valid_links(
                decision.links_to_follow, allowed_urls=set(links)
            )
            return decision

        except Exception as exc:
            logger.warning("Navigator agent failed for %s: %s", url, exc)
            # Safe fallback: mark as low relevance, follow no links
            return NavigatorDecision(
                relevance_score=0.1,
                links_to_follow=[],
                action="backtrack",
                reasoning=f"Navigator error: {exc}",
            )


def _filter_valid_links(
    priorities: list[LinkPriority], allowed_urls: set[str]
) -> list[LinkPriority]:
    """
    Remove any links the LLM hallucinated (not present on the page).
    Sort by priority descending.
    """
    if not allowed_urls:
        return priorities

    valid = [lp for lp in priorities if lp.url in allowed_urls]
    valid.sort(key=lambda x: x.priority, reverse=True)
    return valid


def summarize_history(visited_urls: list[str], max_entries: int = 10) -> str:
    """Generate a concise history summary from recently visited URLs."""
    recent = visited_urls[-max_entries:]
    if not recent:
        return "No pages visited yet."
    lines = [f"- {url}" for url in recent]
    return f"Recently visited ({len(visited_urls)} total):\n" + "\n".join(lines)
