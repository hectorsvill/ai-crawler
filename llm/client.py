"""
llm/client.py — Ollama client wrapper.

Responsibilities:
  - Model selection and switching (navigator / extractor / router models)
  - JSON-mode structured output with Pydantic schema injection
  - Token counting via tiktoken (cl100k_base approximation)
  - Content chunking to fit context windows
  - Retry with exponential backoff on timeouts
  - LLM response caching (content_hash + goal → response)
  - Fallback to smaller model if primary is overloaded
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, TypeVar

from pydantic import BaseModel

try:
    import tiktoken
    _TOKENIZER = tiktoken.get_encoding("cl100k_base")
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    _TOKENIZER = None

try:
    import ollama as _ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    _ollama = None  # type: ignore

import aiohttp

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# ── Constants ─────────────────────────────────────────────────────────────────

# Approximate context window for Ollama models (tokens)
DEFAULT_CONTEXT_WINDOW = 8192
# Reserve headroom for prompt + output
PROMPT_OVERHEAD_TOKENS = 1024
MAX_CONTENT_TOKENS = DEFAULT_CONTEXT_WINDOW - PROMPT_OVERHEAD_TOKENS

SYSTEM_JSON_INSTRUCTION = (
    "You are a precise JSON extraction assistant. "
    "Always respond with valid JSON only — no markdown fences, no prose. "
    "Match the exact schema provided."
)


# ── Token utilities ────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """Approximate token count using tiktoken (cl100k_base)."""
    if HAS_TIKTOKEN and _TOKENIZER:
        return len(_TOKENIZER.encode(text))
    # Rough fallback: ~4 chars per token
    return len(text) // 4


def chunk_text(text: str, max_tokens: int = MAX_CONTENT_TOKENS) -> list[str]:
    """
    Split *text* into chunks that each fit within *max_tokens*.
    Splits on paragraph boundaries where possible.
    """
    if count_tokens(text) <= max_tokens:
        return [text]

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)
        if current_tokens + para_tokens > max_tokens and current:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0
        # If a single paragraph exceeds limit, split by sentence
        if para_tokens > max_tokens:
            sentences = para.replace(". ", ".\n").split("\n")
            for sent in sentences:
                sent_tokens = count_tokens(sent)
                if current_tokens + sent_tokens > max_tokens and current:
                    chunks.append("\n\n".join(current))
                    current = []
                    current_tokens = 0
                current.append(sent)
                current_tokens += sent_tokens
        else:
            current.append(para)
            current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))
    return chunks


# ── LLM response cache ─────────────────────────────────────────────────────────

class _LLMCache:
    """Simple in-memory cache keyed by (content_hash, goal, model)."""

    def __init__(self, max_size: int = 512):
        self._store: dict[str, str] = {}
        self._max_size = max_size

    def _key(self, content: str, goal: str, model: str) -> str:
        raw = f"{content}|{goal}|{model}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, content: str, goal: str, model: str) -> str | None:
        return self._store.get(self._key(content, goal, model))

    def set(self, content: str, goal: str, model: str, response: str) -> None:
        if len(self._store) >= self._max_size:
            # Evict oldest entry (dict insertion order in Python 3.7+)
            oldest = next(iter(self._store))
            del self._store[oldest]
        self._store[self._key(content, goal, model)] = response


_cache = _LLMCache()


# ── Ollama HTTP client (fallback when ollama library unavailable) ─────────────

async def _call_ollama_http(
    base_url: str,
    model: str,
    messages: list[dict],
    timeout: int,
) -> str:
    """Direct HTTP call to Ollama API (used when ollama library is absent)."""
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {"model": model, "messages": messages, "stream": False}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["message"]["content"]


# ── Main client class ──────────────────────────────────────────────────────────

class OllamaClient:
    """
    Async wrapper around Ollama for structured LLM calls.

    Usage:
        client = OllamaClient(config.ollama)
        result = await client.structured_call(
            model=client.extractor_model,
            system_prompt=EXTRACT_SYSTEM,
            user_content=page_markdown,
            response_schema=ExtractionResult,
        )
    """

    def __init__(self, config: Any) -> None:
        self.base_url: str = config.base_url
        self.navigator_model: str = config.navigator_model
        self.extractor_model: str = config.extractor_model
        self.router_model: str = config.router_model
        self.timeout: int = config.timeout
        self.max_retries: int = config.max_retries

    async def raw_call(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        attempt: int = 0,
    ) -> str:
        """
        Send a chat completion request to Ollama.
        Retries with exponential backoff on failure.
        Falls back to navigator_model if extractor_model fails.
        """
        backoff = 2.0 ** attempt

        try:
            if HAS_OLLAMA:
                client = _ollama.AsyncClient(host=self.base_url)
                response = await asyncio.wait_for(
                    client.chat(model=model, messages=messages),
                    timeout=self.timeout,
                )
                return response["message"]["content"]
            else:
                return await _call_ollama_http(self.base_url, model, messages, self.timeout)

        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning("LLM call failed (attempt %d): %s", attempt + 1, exc)

            if attempt < self.max_retries - 1:
                await asyncio.sleep(backoff)
                return await self.raw_call(model, messages, attempt=attempt + 1)

            # Fallback to smaller model if primary was large
            if model == self.extractor_model and model != self.navigator_model:
                logger.warning("Falling back to navigator model: %s", self.navigator_model)
                return await self.raw_call(
                    self.navigator_model, messages, attempt=0
                )

            raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {exc}") from exc

    async def structured_call(
        self,
        model: str,
        system_prompt: str,
        user_content: str,
        response_schema: type[T],
        *,
        cache_key: str | None = None,
        goal: str = "",
    ) -> T:
        """
        Call the LLM and parse the response as a Pydantic model.

        Injects the JSON schema into the system prompt.
        Uses the LLM cache when *cache_key* is provided.
        """
        schema_str = json.dumps(response_schema.model_json_schema(), indent=2)
        full_system = (
            f"{system_prompt}\n\n"
            f"{SYSTEM_JSON_INSTRUCTION}\n\n"
            f"Response JSON schema:\n```json\n{schema_str}\n```"
        )

        # Check cache
        if cache_key:
            cached = _cache.get(cache_key, goal, model)
            if cached:
                logger.debug("LLM cache hit for %s", cache_key[:16])
                return _parse_llm_response(cached, response_schema)

        # Chunk content if too long
        chunks = chunk_text(user_content)
        if len(chunks) > 1:
            logger.debug("Content chunked into %d pieces; using first chunk", len(chunks))
            user_content = chunks[0] + "\n\n[Content truncated for context window]"

        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_content},
        ]

        raw = await self.raw_call(model, messages)

        if cache_key:
            _cache.set(cache_key, goal, model, raw)

        return _parse_llm_response(raw, response_schema)

    async def simple_call(
        self,
        model: str,
        system_prompt: str,
        user_content: str,
    ) -> str:
        """Plain text LLM call (no schema enforcement)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        return await self.raw_call(model, messages)


# ── Response parsing ───────────────────────────────────────────────────────────

def _parse_llm_response(raw: str, schema: type[T]) -> T:
    """
    Parse a raw LLM string response as the given Pydantic model.
    Strips markdown code fences if present.
    """
    text = raw.strip()

    # Strip ```json ... ``` fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        text = text.strip()

    # Find the first JSON object/array
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start != -1:
            end = text.rfind(end_char)
            if end > start:
                text = text[start:end + 1]
                break

    try:
        data = json.loads(text)
        return schema.model_validate(data)
    except Exception as exc:
        logger.error("Failed to parse LLM response: %s\nRaw: %.200s", exc, raw)
        raise ValueError(f"LLM returned unparseable JSON: {exc}") from exc
