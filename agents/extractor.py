"""
agents/extractor.py — Extractor agent.

Receives page markdown and a crawl goal (plus optional schema hint).
Returns an ExtractionResult with structured JSON data, the schema used,
a confidence score, and a plain-language explanation.

Uses the larger extractor model for higher-quality structured output.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from storage.models import ExtractionResult

if TYPE_CHECKING:
    from llm.client import OllamaClient

logger = logging.getLogger(__name__)

# ── Prompt templates ───────────────────────────────────────────────────────────

EXTRACTOR_SYSTEM_PROMPT = """You are a precise data extraction assistant for web crawling.
Given page content and a research goal, extract all relevant structured information.

Rules:
- Extract only information present in the page — do NOT invent data.
- Infer the best JSON schema from the goal and page content if no schema is provided.
- Set confidence between 0.0 (guessing) and 1.0 (clear, explicit data).
- Keep field names snake_case and descriptive.
- If the page is irrelevant to the goal, return {"relevant": false} with low confidence."""

EXTRACTOR_USER_TEMPLATE = """## Research Goal
{goal}

## Schema Hint (optional)
{schema_hint}

## Page URL
{url}

## Page Content
{markdown}

Extract all relevant structured data and return your ExtractionResult JSON."""


class ExtractorAgent:
    """
    LLM-powered agent that extracts structured data from page content.

    Handles long pages by processing the most relevant chunk first,
    then merging results from subsequent chunks if needed.
    """

    def __init__(self, llm: "OllamaClient") -> None:
        self.llm = llm

    async def extract(
        self,
        url: str,
        markdown: str,
        goal: str,
        schema_hint: str = "",
        content_hash: str = "",
    ) -> ExtractionResult:
        """
        Extract structured data from the page.

        For long pages, processes the first content chunk. Falls back to
        a minimal result on LLM errors.
        """
        user_content = EXTRACTOR_USER_TEMPLATE.format(
            goal=goal,
            url=url,
            schema_hint=schema_hint or "Infer the best schema from context.",
            markdown=markdown[:6000],  # pre-trim; client will chunk if still too large
        )

        try:
            result = await self.llm.structured_call(
                model=self.llm.extractor_model,
                system_prompt=EXTRACTOR_SYSTEM_PROMPT,
                user_content=user_content,
                response_schema=ExtractionResult,
                cache_key=content_hash if content_hash else None,
                goal=goal,
            )
            return result

        except Exception as exc:
            logger.warning("Extractor agent failed for %s: %s", url, exc)
            return ExtractionResult(
                data={"error": str(exc), "url": url},
                schema_used="error",
                confidence=0.0,
                explanation=f"Extraction failed: {exc}",
            )

    async def extract_chunks(
        self,
        url: str,
        markdown: str,
        goal: str,
        schema_hint: str = "",
        content_hash: str = "",
    ) -> ExtractionResult:
        """
        Extract from all content chunks and merge results.
        Useful for very long pages where important data may be spread throughout.
        """
        from llm.client import chunk_text

        chunks = chunk_text(markdown)
        if len(chunks) == 1:
            return await self.extract(url, markdown, goal, schema_hint, content_hash)

        logger.info("Extracting from %d chunks for %s", len(chunks), url)
        results: list[ExtractionResult] = []

        for i, chunk in enumerate(chunks):
            chunk_hash = f"{content_hash}_chunk{i}" if content_hash else ""
            try:
                result = await self.extract(
                    url, chunk, goal,
                    schema_hint=schema_hint,
                    content_hash=chunk_hash,
                )
                results.append(result)
            except Exception as exc:
                logger.debug("Chunk %d extraction failed: %s", i, exc)

        if not results:
            return ExtractionResult(
                data={},
                schema_used="none",
                confidence=0.0,
                explanation="All chunks failed to extract.",
            )

        return _merge_results(results)


def _merge_results(results: list[ExtractionResult]) -> ExtractionResult:
    """
    Merge extraction results from multiple chunks.
    Combines data dicts and averages confidence scores.
    """
    merged_data: dict[str, Any] = {}
    total_confidence = 0.0

    for r in results:
        # Deep merge: lists are extended, scalars are overwritten by higher-confidence
        for key, value in r.data.items():
            if key in merged_data:
                if isinstance(merged_data[key], list) and isinstance(value, list):
                    merged_data[key].extend(value)
                elif r.confidence > total_confidence / max(len(results), 1):
                    merged_data[key] = value
            else:
                merged_data[key] = value
        total_confidence += r.confidence

    best = max(results, key=lambda r: r.confidence)
    return ExtractionResult(
        data=merged_data,
        schema_used=best.schema_used,
        confidence=total_confidence / len(results),
        explanation=f"Merged from {len(results)} chunks. Best confidence: {best.confidence:.2f}",
    )
