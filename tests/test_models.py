"""Tests for storage/models.py — Pydantic validators and clamping."""

import pytest
from pydantic import ValidationError

from storage.models import (
    ExtractionResult,
    LinkPriority,
    NavigatorDecision,
    URLItem,
    URLStatus,
)


class TestLinkPriority:
    def test_valid_values(self):
        lp = LinkPriority(url="https://example.com", priority=0.7, reasoning="good", estimated_value=0.8)
        assert lp.priority == pytest.approx(0.7)

    def test_clamps_priority_above_1(self):
        lp = LinkPriority(url="https://example.com", priority=1.5, reasoning="x", estimated_value=0.5)
        assert lp.priority == pytest.approx(1.0)

    def test_clamps_priority_below_0(self):
        lp = LinkPriority(url="https://example.com", priority=-0.5, reasoning="x", estimated_value=0.5)
        assert lp.priority == pytest.approx(0.0)

    def test_clamps_estimated_value_above_1(self):
        lp = LinkPriority(url="https://example.com", priority=0.5, reasoning="x", estimated_value=99.0)
        assert lp.estimated_value == pytest.approx(1.0)


class TestNavigatorDecision:
    def test_valid(self):
        nd = NavigatorDecision(relevance_score=0.8, action="deepen", reasoning="relevant")
        assert nd.relevance_score == pytest.approx(0.8)
        assert nd.action == "deepen"

    def test_clamps_relevance_above_1(self):
        nd = NavigatorDecision(relevance_score=2.0, action="deepen", reasoning="x")
        assert nd.relevance_score == pytest.approx(1.0)

    def test_clamps_relevance_below_0(self):
        nd = NavigatorDecision(relevance_score=-1.0, action="deepen", reasoning="x")
        assert nd.relevance_score == pytest.approx(0.0)

    def test_valid_actions(self):
        for action in ("deepen", "backtrack", "complete"):
            nd = NavigatorDecision(relevance_score=0.5, action=action, reasoning="x")
            assert nd.action == action

    def test_invalid_action_defaults_to_deepen(self):
        nd = NavigatorDecision(relevance_score=0.5, action="invalid_action", reasoning="x")
        assert nd.action == "deepen"

    def test_action_case_insensitive(self):
        nd = NavigatorDecision(relevance_score=0.5, action="COMPLETE", reasoning="x")
        assert nd.action == "complete"

    def test_default_links_empty(self):
        nd = NavigatorDecision(relevance_score=0.5, action="deepen", reasoning="x")
        assert nd.links_to_follow == []


class TestExtractionResult:
    def test_valid(self):
        er = ExtractionResult(data={"key": "val"}, schema_used="auto", confidence=0.9, explanation="ok")
        assert er.confidence == pytest.approx(0.9)

    def test_clamps_confidence_above_1(self):
        er = ExtractionResult(data={}, schema_used="auto", confidence=1.5, explanation="")
        assert er.confidence == pytest.approx(1.0)

    def test_clamps_confidence_below_0(self):
        er = ExtractionResult(data={}, schema_used="auto", confidence=-0.1, explanation="")
        assert er.confidence == pytest.approx(0.0)

    def test_default_explanation(self):
        er = ExtractionResult(data={}, schema_used="auto", confidence=0.5)
        assert er.explanation == ""


class TestURLItem:
    def test_default_status(self):
        item = URLItem(url="https://example.com")
        assert item.status == URLStatus.pending

    def test_from_attributes(self):
        # Ensure from_attributes works (used when loading from ORM)
        item = URLItem.model_validate(
            {"url": "https://example.com", "priority": 0.8, "depth": 2}
        )
        assert item.depth == 2
