"""
workflows/router.py — Goal-based workflow selector.

Uses a single LLM call to analyze the user's natural-language goal and
classify it into one of: simple, langgraph, or crewai.

Logs the reasoning for observability. Respects the --workflow CLI override.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel

from llm.client import OllamaClient

logger = logging.getLogger(__name__)

WorkflowMode = Literal["simple", "langgraph", "crewai"]

# ── Prompt templates ───────────────────────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = """You are a workflow routing assistant for an AI web crawler.
Given a user's research goal, classify it into one of three workflow modes:

- "simple":    Shallow crawl, few pages, direct extraction.
               Examples: "Get the price of X", "Find contact info on this page",
               "Extract the product specs from this URL".

- "langgraph": Deep, stateful, multi-step exploration with branching.
               Examples: "Build a knowledge base on solar energy startups",
               "Map out all documentation pages for this library",
               "Crawl this news site and find all articles about climate change".

- "crewai":    Collaborative multi-role research requiring synthesis.
               Examples: "Generate a list of B2B SaaS leads in fintech",
               "Create a competitive analysis of cloud providers",
               "Research and summarize the top 10 AI startups".

Choose the SIMPLEST mode that can accomplish the goal.
Prefer simple over langgraph, langgraph over crewai."""

ROUTER_USER_TEMPLATE = """User Goal: {goal}

Classify this goal into the appropriate workflow mode and explain your reasoning."""


class RouterDecision(BaseModel):
    workflow: WorkflowMode
    reasoning: str
    estimated_pages: int  # rough estimate
    complexity: Literal["low", "medium", "high"]


async def select_workflow(
    goal: str,
    llm: OllamaClient,
    override: str | None = None,
) -> tuple[WorkflowMode, str]:
    """
    Analyze *goal* and return the recommended (workflow_mode, reasoning).

    If *override* is provided and valid, it takes precedence over the LLM decision.
    """
    if override and override in ("simple", "langgraph", "crewai"):
        logger.info("Workflow override: %s", override)
        return override, f"Manually specified by user: {override}"  # type: ignore[return-value]

    user_content = ROUTER_USER_TEMPLATE.format(goal=goal)

    try:
        decision = await llm.structured_call(
            model=llm.router_model,
            system_prompt=ROUTER_SYSTEM_PROMPT,
            user_content=user_content,
            response_schema=RouterDecision,
        )
        logger.info(
            "Workflow selected: %s (complexity=%s, ~%d pages) — %s",
            decision.workflow,
            decision.complexity,
            decision.estimated_pages,
            decision.reasoning,
        )
        return decision.workflow, decision.reasoning

    except Exception as exc:
        logger.warning("Router LLM failed, defaulting to simple: %s", exc)
        return "simple", f"Router error, defaulted to simple: {exc}"
