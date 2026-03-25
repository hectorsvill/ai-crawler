"""Tests for llm/client.py — chunking, caching, response parsing."""

import pytest
from pydantic import BaseModel

from llm.client import (
    _LLMCache,
    _parse_llm_response,
    chunk_text,
    count_tokens,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

class _SimpleSchema(BaseModel):
    value: str
    score: float = 0.0


# ── count_tokens ──────────────────────────────────────────────────────────────

class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_short_text(self):
        # Should be a small positive number
        tokens = count_tokens("hello world")
        assert tokens > 0
        assert tokens < 20

    def test_long_text_more_tokens(self):
        short = count_tokens("hi")
        long = count_tokens("hi " * 1000)
        assert long > short


# ── chunk_text ────────────────────────────────────────────────────────────────

class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "Short paragraph."
        chunks = chunk_text(text, max_tokens=1000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_splits(self):
        # Build a text that is guaranteed to exceed 50 tokens
        paragraph = "word " * 100
        text = "\n\n".join([paragraph] * 10)
        chunks = chunk_text(text, max_tokens=50)
        assert len(chunks) > 1

    def test_chunks_cover_all_content(self):
        paragraphs = [f"Paragraph {i}." for i in range(20)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, max_tokens=30)
        combined = " ".join(chunks)
        # Every paragraph should appear somewhere in the combined output
        for p in paragraphs:
            assert p in combined

    def test_each_chunk_within_token_limit(self):
        paragraph = "word " * 50  # ~50 tokens per paragraph
        text = "\n\n".join([paragraph] * 20)
        max_tokens = 100
        chunks = chunk_text(text, max_tokens=max_tokens)
        for chunk in chunks:
            assert count_tokens(chunk) <= max_tokens * 1.1  # 10% tolerance


# ── _LLMCache ─────────────────────────────────────────────────────────────────

class TestLLMCache:
    def test_miss_returns_none(self):
        cache = _LLMCache()
        assert cache.get("content", "goal", "model") is None

    def test_set_then_get(self):
        cache = _LLMCache()
        cache.set("content", "goal", "model", '{"value": "ok"}')
        result = cache.get("content", "goal", "model")
        assert result == '{"value": "ok"}'

    def test_different_keys_are_distinct(self):
        cache = _LLMCache()
        cache.set("c", "g1", "m", "response1")
        cache.set("c", "g2", "m", "response2")
        assert cache.get("c", "g1", "m") == "response1"
        assert cache.get("c", "g2", "m") == "response2"

    def test_eviction_at_max_size(self):
        cache = _LLMCache(max_size=3)
        for i in range(4):
            cache.set(f"content{i}", "goal", "model", f"resp{i}")
        # The oldest (content0) should be evicted
        assert cache.get("content0", "goal", "model") is None
        # The most recent should still be there
        assert cache.get("content3", "goal", "model") == "resp3"


# ── _parse_llm_response ───────────────────────────────────────────────────────

class TestParseLLMResponse:
    def test_plain_json(self):
        raw = '{"value": "hello", "score": 0.9}'
        result = _parse_llm_response(raw, _SimpleSchema)
        assert result.value == "hello"
        assert result.score == pytest.approx(0.9)

    def test_strips_json_fence(self):
        raw = '```json\n{"value": "world"}\n```'
        result = _parse_llm_response(raw, _SimpleSchema)
        assert result.value == "world"

    def test_strips_plain_fence(self):
        raw = '```\n{"value": "test"}\n```'
        result = _parse_llm_response(raw, _SimpleSchema)
        assert result.value == "test"

    def test_extracts_json_from_surrounding_prose(self):
        raw = 'Here is the result:\n{"value": "embedded"}\nThank you.'
        result = _parse_llm_response(raw, _SimpleSchema)
        assert result.value == "embedded"

    def test_raises_on_unparseable(self):
        with pytest.raises(ValueError, match="unparseable"):
            _parse_llm_response("this is not json at all", _SimpleSchema)

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            _parse_llm_response("", _SimpleSchema)
