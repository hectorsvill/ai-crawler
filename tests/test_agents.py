"""Tests for agent helpers — _filter_valid_links and extraction merging."""

import pytest

from agents.extractor import _merge_results
from agents.navigator import _filter_valid_links, summarize_history
from storage.models import ExtractionResult, LinkPriority


# ── _filter_valid_links ───────────────────────────────────────────────────────

def _make_lp(url: str, priority: float = 0.5) -> LinkPriority:
    return LinkPriority(url=url, priority=priority, reasoning="test", estimated_value=priority)


class TestFilterValidLinks:
    def test_removes_hallucinated_urls(self):
        priorities = [_make_lp("https://real.com"), _make_lp("https://hallucinated.com")]
        allowed = {"https://real.com"}
        result = _filter_valid_links(priorities, allowed)
        assert len(result) == 1
        assert result[0].url == "https://real.com"

    def test_sorts_by_priority_descending(self):
        priorities = [_make_lp("https://a.com", 0.3), _make_lp("https://b.com", 0.9)]
        allowed = {"https://a.com", "https://b.com"}
        result = _filter_valid_links(priorities, allowed)
        assert result[0].url == "https://b.com"

    def test_empty_allowed_returns_all(self):
        priorities = [_make_lp("https://a.com"), _make_lp("https://b.com")]
        result = _filter_valid_links(priorities, allowed_urls=set())
        assert len(result) == 2

    def test_empty_priorities(self):
        result = _filter_valid_links([], {"https://example.com"})
        assert result == []


# ── summarize_history ─────────────────────────────────────────────────────────

class TestSummarizeHistory:
    def test_empty_history(self):
        result = summarize_history([])
        assert "No pages" in result

    def test_recent_urls_included(self):
        urls = [f"https://example.com/page{i}" for i in range(20)]
        result = summarize_history(urls, max_entries=5)
        # Only last 5 should appear
        for url in urls[-5:]:
            assert url in result
        assert urls[0] not in result

    def test_total_count_shown(self):
        urls = [f"https://example.com/{i}" for i in range(15)]
        result = summarize_history(urls, max_entries=5)
        assert "15" in result


# ── _merge_results ────────────────────────────────────────────────────────────

def _make_result(data: dict, confidence: float, schema: str = "auto") -> ExtractionResult:
    return ExtractionResult(data=data, schema_used=schema, confidence=confidence, explanation="test")


class TestMergeResults:
    def test_single_result_returned_unchanged(self):
        r = _make_result({"key": "value"}, 0.9)
        merged = _merge_results([r])
        assert merged.data == {"key": "value"}
        assert merged.confidence == pytest.approx(0.9)

    def test_merges_disjoint_keys(self):
        r1 = _make_result({"a": 1}, 0.8)
        r2 = _make_result({"b": 2}, 0.6)
        merged = _merge_results([r1, r2])
        assert "a" in merged.data
        assert "b" in merged.data

    def test_lists_extended(self):
        r1 = _make_result({"items": [1, 2]}, 0.8)
        r2 = _make_result({"items": [3, 4]}, 0.7)
        merged = _merge_results([r1, r2])
        assert set(merged.data["items"]) == {1, 2, 3, 4}

    def test_confidence_averaged(self):
        r1 = _make_result({}, 0.8)
        r2 = _make_result({}, 0.4)
        merged = _merge_results([r1, r2])
        assert merged.confidence == pytest.approx(0.6)

    def test_schema_from_best_confidence(self):
        r1 = _make_result({}, 0.3, schema="schema_low")
        r2 = _make_result({}, 0.9, schema="schema_high")
        merged = _merge_results([r1, r2])
        assert merged.schema_used == "schema_high"
