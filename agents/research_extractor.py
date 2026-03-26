"""
agents/research_extractor.py — Research-specific extractor agent.

Extracts structured bibliographic metadata from academic/research pages:
arXiv abstracts, DOI landing pages, Semantic Scholar, PubMed, SSRN, etc.

Returns a ResearchPaperData Pydantic model instead of the generic
ExtractionResult, with fields: title, authors, abstract, year, doi,
arxiv_id, venue, keywords, pdf_url, confidence.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from storage.models import ResearchPaperData

if TYPE_CHECKING:
    from llm.client import OllamaClient

logger = logging.getLogger(__name__)

# ── Prompt ─────────────────────────────────────────────────────────────────────

RESEARCH_SYSTEM_PROMPT = """You are an academic metadata extraction assistant.
Given the content of a research-related web page, extract bibliographic information.

Rules:
- Extract ONLY information explicitly present on the page — never invent data.
- doi: the exact DOI string (e.g. "10.1145/3586183"). Never guess or fabricate.
- arxiv_id: bare arXiv ID only (e.g. "2305.12345v2"), NOT the full URL.
- authors: always a list of strings, even for a single author.
- year: publication year as integer. Use submission year for preprints if no other year available.
- venue: conference name, journal name, or "arXiv preprint" for preprints.
- keywords: list of topic keywords. Infer from title/abstract if not explicit.
- pdf_url: direct link to the PDF if present on the page, otherwise null.
- If the page lists multiple papers (a search results page), extract the first / most prominent.
- Set confidence 0.8–1.0 for full abstract pages, 0.3–0.5 for listing pages, 0.1 for irrelevant pages.
- If the page is clearly not research-related, return title=null with confidence=0.0."""

RESEARCH_USER_TEMPLATE = """## Research Goal
{goal}

## Page URL
{url}

## Page Content
{markdown}

Extract bibliographic metadata and return your ResearchPaperData JSON."""


class ResearchExtractorAgent:
    """
    LLM-powered agent that extracts academic paper metadata.

    Returns ResearchPaperData with structured bibliographic fields.
    Uses the extractor model (higher quality) for accurate metadata extraction.
    """

    def __init__(self, llm: "OllamaClient | None" = None) -> None:
        if llm is None:
            from config import load_config
            from llm.client import OllamaClient as _OC
            llm = _OC(load_config().ollama)
        self.llm = llm

    async def extract(
        self,
        url: str = "",
        markdown: str = "",
        goal: str = "",
        content_hash: str = "",
    ) -> ResearchPaperData:
        """
        Extract academic metadata from a page.

        For abstract pages, processes the full content (up to 8k chars).
        Falls back to a zero-confidence result on LLM errors.
        """
        # arXiv/DOI pages are concise; use more content than the generic extractor
        user_content = RESEARCH_USER_TEMPLATE.format(
            goal=goal,
            url=url,
            markdown=markdown[:8000],
        )

        try:
            result = await self.llm.structured_call(
                model=self.llm.extractor_model,
                system_prompt=RESEARCH_SYSTEM_PROMPT,
                user_content=user_content,
                response_schema=ResearchPaperData,
                cache_key=content_hash if content_hash else None,
                goal=goal,
            )

            # Post-process: strip arXiv URL prefix if the LLM included it
            if result.arxiv_id:
                result.arxiv_id = _clean_arxiv_id(result.arxiv_id)

            # Normalize DOI: remove URL prefix
            if result.doi:
                result.doi = _clean_doi(result.doi)

            return result

        except Exception as exc:
            logger.warning("ResearchExtractor failed for %s: %s", url, exc)
            return ResearchPaperData(confidence=0.0, explanation=f"Extraction failed: {exc}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clean_arxiv_id(raw: str) -> str:
    """Strip URL prefix from arXiv IDs: 'https://arxiv.org/abs/2305.12345' → '2305.12345'."""
    raw = raw.strip()
    # Remove URL prefixes
    for prefix in ("https://arxiv.org/abs/", "http://arxiv.org/abs/", "arxiv.org/abs/", "arXiv:"):
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix):]
    return raw


def _clean_doi(raw: str) -> str:
    """Normalize DOI: strip URL prefix if present."""
    raw = raw.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi.org/", "DOI:", "doi:"):
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix):]
    return raw
