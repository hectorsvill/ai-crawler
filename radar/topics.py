"""radar/topics.py — keyword-based topic tagging (deterministic, no LLM).

Each topic maps to case-insensitive regex patterns. A page gets a tag when its
title/description/text matches. The seed's own topics are always applied too.
"""

from __future__ import annotations

import re

TOPIC_PATTERNS: dict[str, list[str]] = {
    "frontier-labs": [
        r"\bopenai\b", r"\banthropic\b", r"\bdeepmind\b", r"\bgemini\b", r"\bxai\b",
        r"\bgrok\b", r"\bmeta ai\b", r"\bllama\b", r"\bmistral\b", r"\bclaude\b",
        r"\bgpt-?\d", r"\bopus\b", r"\bfrontier model",
    ],
    "open-weights": [
        r"\bopen[- ]weights?\b", r"\bopen[- ]source model", r"\bhugging ?face\b",
        r"\bdeepseek\b", r"\bqwen\b", r"\bkimi\b", r"\bminimax\b", r"\bgguf\b",
    ],
    "agents-mcp-coding": [
        r"\bai agents?\b", r"\bagentic\b", r"\bmodel context protocol\b", r"\bmcp\b",
        r"\bcoding (agent|assistant|tool)", r"\bcodex\b", r"\bclaude code\b", r"\bcursor\b",
        r"\bcopilot\b", r"\bcli\b",
    ],
    "chips-export-controls": [
        r"\bnvidia\b", r"\bblackwell\b", r"\bh20\b", r"\bh200\b", r"\bb200\b", r"\bb300\b",
        r"\bgb200\b", r"\bexport controls?\b", r"\bbureau of industry and security\b",
        r"\bbis\b", r"\bentity list\b", r"\bsemiconductor", r"\btsmc\b", r"\bgpus?\b",
    ],
    "ai-security": [
        r"\bred[- ]team", r"\bjailbreak", r"\bsandbox escape", r"\bprompt injection",
        r"\bsupply[- ]chain attack", r"\bsecret[- ]steal", r"\bcredential", r"\bmalware\b",
        r"\bzero[- ]day\b", r"\bexploit", r"\bvulnerabilit", r"\bcve-\d{4}-\d+",
        r"\bgithub actions\b", r"\bai safety\b", r"\bmisuse\b", r"\bsafeguards?\b",
    ],
    "war-ukraine-russia": [
        r"\bukrain", r"\bkyiv\b", r"\bkremlin\b", r"\bputin\b", r"\bzelensk",
        r"\brussian (forces|army|navy|strike|drone|missile)", r"\bdonbas", r"\bprisoner swap",
    ],
    "war-iran-gulf": [
        r"\biran", r"\birgc\b", r"\bhormuz\b", r"\bpersian gulf\b", r"\bbahrain\b",
        r"\bhouthi", r"\bred sea\b", r"\bisrael", r"\bcentcom\b", r"\bukmto\b",
    ],
    "taiwan-pla": [
        r"\btaiwan", r"\bpla\b", r"\bpeople'?s liberation army\b", r"\bsouth china sea\b",
        r"\btaiwan strait\b", r"\bchina coast guard\b",
    ],
    "shtf": [
        r"\bnuclear\b", r"\bblackout\b", r"\bgrid (failure|attack)", r"\bmartial law\b",
        r"\bundersea cable", r"\bevacuat", r"\bstate of emergency\b", r"\bmobiliz",
    ],
}

_COMPILED = {t: [re.compile(p, re.I) for p in pats] for t, pats in TOPIC_PATTERNS.items()}


def tag_text(*parts: str | None, min_hits: int = 1, body_min_hits: int = 2) -> list[str]:
    """Return topic tags for the given text parts.

    The first part is treated as the headline (title): one hit is enough.
    Remaining parts (description/body) need ``body_min_hits`` distinct pattern
    hits so a single passing mention doesn't tag the page.
    """
    head = (parts[0] or "") if parts else ""
    body = " ".join(p for p in parts[1:] if p)[:20000]
    tags = []
    for topic, regs in _COMPILED.items():
        head_hits = sum(1 for r in regs if r.search(head))
        body_hits = sum(1 for r in regs if r.search(body))
        if head_hits >= min_hits or body_hits >= body_min_hits:
            tags.append(topic)
    return tags


def known_topics() -> list[str]:
    return list(TOPIC_PATTERNS)
