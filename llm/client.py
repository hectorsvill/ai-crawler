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
    Splits on paragraph → sentence → word boundaries in order of preference.
    """
    if count_tokens(text) <= max_tokens:
        return [text]

    chunks: list[str] = []

    def _flush(parts: list[str], sep: str) -> None:
        if parts:
            chunks.append(sep.join(parts))

    def _split_by_words(segment: str) -> None:
        """Last-resort: split a segment into word-level chunks."""
        words = segment.split()
        current_words: list[str] = []
        current_tok = 0
        for word in words:
            wt = count_tokens(word)
            if current_tok + wt > max_tokens and current_words:
                chunks.append(" ".join(current_words))
                current_words = []
                current_tok = 0
            current_words.append(word)
            current_tok += wt
        if current_words:
            chunks.append(" ".join(current_words))

    paragraphs = text.split("\n\n")
    current_paras: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if current_tokens + para_tokens > max_tokens and current_paras:
            _flush(current_paras, "\n\n")
            current_paras = []
            current_tokens = 0

        if para_tokens > max_tokens:
            # Split paragraph by sentences
            sentences = para.replace(". ", ".\n").split("\n")
            curr_sents: list[str] = []
            curr_sent_tok = 0
            for sent in sentences:
                st = count_tokens(sent)
                if curr_sent_tok + st > max_tokens and curr_sents:
                    _flush(curr_sents, " ")
                    curr_sents = []
                    curr_sent_tok = 0
                if st > max_tokens:
                    # Even a single sentence is too long — go word-level
                    if curr_sents:
                        _flush(curr_sents, " ")
                        curr_sents = []
                        curr_sent_tok = 0
                    _split_by_words(sent)
                else:
                    curr_sents.append(sent)
                    curr_sent_tok += st
            if curr_sents:
                _flush(curr_sents, " ")
        else:
            current_paras.append(para)
            current_tokens += para_tokens

    if current_paras:
        _flush(current_paras, "\n\n")

    return chunks if chunks else [text]


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

        # Build prompt string for token accounting
        _prompt_str = " ".join(m.get("content", "") for m in messages)

        try:
            if HAS_OLLAMA:
                client = _ollama.AsyncClient(host=self.base_url)
                response = await asyncio.wait_for(
                    client.chat(model=model, messages=messages),
                    timeout=self.timeout,
                )
                raw = response["message"]["content"]
            else:
                raw = await _call_ollama_http(self.base_url, model, messages, self.timeout)

            token_usage.record(model, _prompt_str, raw)
            return raw

        except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
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

        # Include the schema name in the cache key so navigator and extractor
        # calls for the same page don't collide when using the same model.
        schema_name = response_schema.__name__
        effective_cache_key = f"{cache_key}:{schema_name}" if cache_key else None

        # Check cache
        if effective_cache_key:
            cached = _cache.get(effective_cache_key, goal, model)
            if cached:
                logger.debug("LLM cache hit for %s [%s]", cache_key[:16], schema_name)
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

        if effective_cache_key:
            _cache.set(effective_cache_key, goal, model, raw)

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
    except json.JSONDecodeError as json_exc:
        # Progressive JSON repair — each step tries increasingly aggressive fixes
        import re as _re

        repaired = text

        # Step 1: Remove trailing commas before } or ]
        repaired = _re.sub(r",\s*([}\]])", r"\1", repaired)
        try:
            data = json.loads(repaired)
            logger.debug("JSON recovered after trailing-comma cleanup")
            return schema.model_validate(data)
        except Exception:
            pass

        # Step 2: Fix invalid \escape sequences (e.g. \( \) \_ from markdown)
        # Replace lone backslashes not followed by valid JSON escape chars
        repaired2 = _re.sub(r'\\([^"\\/bfnrtu])', r'\1', repaired)
        try:
            data = json.loads(repaired2)
            logger.debug("JSON recovered after backslash-escape repair")
            return schema.model_validate(data)
        except Exception:
            pass

        # Step 3: Both fixes combined
        repaired3 = _re.sub(r'\\([^"\\/bfnrtu])', r'\1', repaired)
        try:
            data = json.loads(repaired3)
            logger.debug("JSON recovered after combined repair")
            return schema.model_validate(data)
        except Exception:
            pass

        logger.error("Failed to parse LLM response: %s\nRaw: %.200s", json_exc, raw)
        raise ValueError(f"LLM returned unparseable JSON: {json_exc}") from json_exc
    except Exception as exc:
        logger.error("Failed to validate LLM response schema: %s\nRaw: %.200s", exc, raw)
        raise ValueError(f"LLM returned unparseable JSON: {exc}") from exc


# ── Token usage tracker ────────────────────────────────────────────────────────

class TokenUsage:
    """Accumulates token counts across multiple LLM calls."""

    def __init__(self) -> None:
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self._per_model: dict[str, dict[str, int]] = {}

    def record(self, model: str, prompt: str, completion: str) -> None:
        pt = count_tokens(prompt)
        ct = count_tokens(completion)
        self.prompt_tokens += pt
        self.completion_tokens += ct
        bucket = self._per_model.setdefault(model, {"prompt": 0, "completion": 0})
        bucket["prompt"] += pt
        bucket["completion"] += ct
        logger.debug("[tokens] model=%s prompt=%d completion=%d", model, pt, ct)

    def summary(self) -> str:
        lines = [
            f"Token usage — prompt: {self.prompt_tokens:,}  "
            f"completion: {self.completion_tokens:,}  "
            f"total: {self.prompt_tokens + self.completion_tokens:,}"
        ]
        for model, counts in self._per_model.items():
            lines.append(
                f"  {model}: prompt={counts['prompt']:,} completion={counts['completion']:,}"
            )
        return "\n".join(lines)


# Module-level usage tracker (shared across all LLMClient instances)
token_usage = TokenUsage()


# ── Graceful startup check ─────────────────────────────────────────────────────

async def check_ollama_reachable(base_url: str = "http://localhost:11434") -> bool:
    """
    Ping Ollama and return True if it responds.
    Prints a clear actionable error message and returns False otherwise.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base_url.rstrip('/')}/api/tags",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                return resp.status == 200
    except Exception:
        print(
            "\n[ERROR] Cannot reach Ollama at "
            f"{base_url}\n"
            "  → Make sure Ollama is installed and running:\n"
            "      ollama serve\n"
            "  → Then verify with:\n"
            "      curl http://localhost:11434/api/tags\n"
        )
        return False


# ── Friendly high-level client ─────────────────────────────────────────────────

class LLMClient:
    """
    Simplified Ollama client for one-off usage without pre-built configs.

    Accepts either an OllamaConfig object or keyword args:
        LLMClient(Settings().ollama)
        LLMClient(base_url="...", default_model="...")

    Methods:
      - generate(prompt, model)                       → plain text
      - generate_json(prompt, response_model, model)  → Pydantic instance
      - check_health()                                → bool
      - count_tokens(text)                            → int
      - chunk_text(text, max_tokens, overlap)         → list[str]

    Token counts are accumulated in .total_input_tokens / .total_output_tokens
    and in the module-level token_usage singleton.
    """

    DEFAULT_MODEL = "qwen2.5:7b"
    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        config_or_base_url: Any = None,
        default_model: str = DEFAULT_MODEL,
        timeout: int = 120,
        max_retries: int = 3,
    ) -> None:
        # Accept an OllamaConfig object or a plain base_url string
        if config_or_base_url is not None and hasattr(config_or_base_url, "base_url"):
            cfg = config_or_base_url
            self.default_model = getattr(cfg, "extractor_model", default_model)
        else:
            base_url = config_or_base_url or self.DEFAULT_BASE_URL

            class _Cfg:
                pass

            cfg = _Cfg()
            cfg.base_url = base_url  # type: ignore[attr-defined]
            cfg.navigator_model = default_model  # type: ignore[attr-defined]
            cfg.extractor_model = default_model  # type: ignore[attr-defined]
            cfg.router_model = default_model  # type: ignore[attr-defined]
            cfg.timeout = timeout  # type: ignore[attr-defined]
            cfg.max_retries = max_retries  # type: ignore[attr-defined]
            self.default_model = default_model

        self._client = OllamaClient(cfg)
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

    # ── Instance-level helpers ────────────────────────────────────────────────

    def count_tokens(self, text: str) -> int:
        """Approximate token count using tiktoken (cl100k_base)."""
        return count_tokens(text)

    def chunk_text(self, text: str, max_tokens: int = MAX_CONTENT_TOKENS, overlap: int = 200) -> list[str]:
        """Split text into chunks respecting token limits."""
        return chunk_text(text, max_tokens=max_tokens)

    async def check_health(self) -> bool:
        """Return True if Ollama is reachable and responding."""
        return await check_ollama_reachable(self._client.base_url)

    # ── OllamaClient compatibility (used by NavigatorAgent / ExtractorAgent) ──

    @property
    def navigator_model(self) -> str:
        return self._client.navigator_model

    @property
    def extractor_model(self) -> str:
        return self._client.extractor_model

    @property
    def router_model(self) -> str:
        return self._client.router_model

    async def structured_call(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to OllamaClient.structured_call (used by agents)."""
        return await self._client.structured_call(*args, **kwargs)

    # ── Core LLM calls ────────────────────────────────────────────────────────

    async def generate(self, prompt: str, model: str | None = None) -> str:
        """Call the LLM and return the raw text response."""
        model = model or self.default_model
        messages = [{"role": "user", "content": prompt}]
        raw = await self._client.raw_call(model, messages)
        pt = count_tokens(prompt)
        ct = count_tokens(raw)
        self.total_input_tokens += pt
        self.total_output_tokens += ct
        token_usage.record(model, prompt, raw)
        print(f"[tokens] model={model} prompt≈{pt} completion≈{ct}")
        return raw

    async def generate_json(
        self,
        prompt: str,
        response_model: type[T] | None = None,
        model: str | None = None,
        *,
        max_attempts: int = 2,
    ) -> T | dict:
        """
        Call the LLM expecting structured JSON.

        If *response_model* is given, parses and returns a validated Pydantic
        instance.  Otherwise returns a plain dict.
        On first parse failure, retries with a stricter prompt.
        """
        model = model or self.default_model

        if response_model is not None:
            # Use OllamaClient's structured_call which injects the schema
            system_prompt = (
                "You are a precise JSON assistant. "
                "Always respond with a single valid JSON object matching the schema provided. "
                "No prose, no markdown fences."
            )
            result = await self._client.structured_call(
                model=model,
                system_prompt=system_prompt,
                user_content=prompt,
                response_schema=response_model,
            )
            pt = count_tokens(prompt)
            ct = count_tokens(str(result))
            self.total_input_tokens += pt
            self.total_output_tokens += ct
            print(f"[tokens] model={model} prompt≈{pt} completion≈{ct}")
            return result  # type: ignore[return-value]

        # Fallback: return plain dict
        system = (
            "You are a precise JSON assistant. "
            "Always respond with a single valid JSON object — no prose, no markdown fences."
        )
        for attempt in range(max_attempts):
            user_content = prompt if attempt == 0 else (
                f"{prompt}\n\nIMPORTANT: Return ONLY raw JSON. No explanations. "
                "Start your response with { and end with }."
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ]
            raw = await self._client.raw_call(model, messages)
            pt = count_tokens(user_content)
            ct = count_tokens(raw)
            self.total_input_tokens += pt
            self.total_output_tokens += ct
            token_usage.record(model, user_content, raw)
            print(f"[tokens] model={model} prompt≈{pt} completion≈{ct}")
            try:
                text = raw.strip()
                if text.startswith("```"):
                    lines = text.split("\n")
                    text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end > start:
                    return json.loads(text[start : end + 1])
                return json.loads(text)
            except Exception as exc:
                if attempt == max_attempts - 1:
                    raise ValueError(
                        f"LLM returned invalid JSON after {max_attempts} attempts: {exc}\nRaw: {raw[:200]}"
                    ) from exc
                logger.warning("generate_json attempt %d failed, retrying: %s", attempt + 1, exc)
        return {}  # unreachable
