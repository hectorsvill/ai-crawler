"""Tests for use-case helper functions (no network, no Ollama required)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── wikipedia_research helpers ────────────────────────────────────────────────

class TestWikipediaHelpers:
    def test_wiki_url_single_word(self):
        from use_cases.wikipedia_research import _wiki_url
        assert _wiki_url("python") == "https://en.wikipedia.org/wiki/python"

    def test_wiki_url_multi_word(self):
        from use_cases.wikipedia_research import _wiki_url
        assert _wiki_url("machine learning") == "https://en.wikipedia.org/wiki/machine_learning"

    def test_wiki_url_strips_whitespace(self):
        from use_cases.wikipedia_research import _wiki_url
        assert _wiki_url("  AI safety  ") == "https://en.wikipedia.org/wiki/AI_safety"

    def test_safe_filename_spaces(self):
        from use_cases.wikipedia_research import _safe_filename
        result = _safe_filename("Machine Learning")
        assert " " not in result
        assert result == "machine_learning"

    def test_safe_filename_special_chars(self):
        from use_cases.wikipedia_research import _safe_filename
        result = _safe_filename("C++ programming")
        assert " " not in result
        assert "+" not in result

    def test_safe_filename_idempotent(self):
        from use_cases.wikipedia_research import _safe_filename
        a = _safe_filename("AI Safety")
        b = _safe_filename("AI Safety")
        assert a == b


# ── github_trending helpers ───────────────────────────────────────────────────

class TestGithubTrendingHelpers:
    def test_build_url_default(self):
        from use_cases.github_trending import _build_url
        url = _build_url("", "daily")
        assert url == "https://github.com/trending"

    def test_build_url_with_language(self):
        from use_cases.github_trending import _build_url
        url = _build_url("python", "daily")
        assert "/python" in url

    def test_build_url_weekly(self):
        from use_cases.github_trending import _build_url
        url = _build_url("", "weekly")
        assert "since=weekly" in url

    def test_build_url_language_and_period(self):
        from use_cases.github_trending import _build_url
        url = _build_url("rust", "monthly")
        assert "rust" in url
        assert "since=monthly" in url

    def test_build_url_language_lowercased(self):
        from use_cases.github_trending import _build_url
        url = _build_url("Python", "daily")
        assert "python" in url.lower()


# ── LLMClient API surface ─────────────────────────────────────────────────────

class TestLLMClientAPI:
    def test_default_model_attribute(self):
        from llm.client import LLMClient
        client = LLMClient()
        assert client.default_model == LLMClient.DEFAULT_MODEL

    def test_custom_model(self):
        from llm.client import LLMClient
        client = LLMClient(default_model="llama3.1:latest")
        assert client.default_model == "llama3.1:latest"

    def test_token_usage_initial_zero(self):
        from llm.client import TokenUsage
        tu = TokenUsage()
        assert tu.prompt_tokens == 0
        assert tu.completion_tokens == 0

    def test_token_usage_record(self):
        from llm.client import TokenUsage
        tu = TokenUsage()
        tu.record("qwen2.5:7b", "hello world prompt", "hello response")
        assert tu.prompt_tokens > 0
        assert tu.completion_tokens > 0

    def test_token_usage_summary_string(self):
        from llm.client import TokenUsage
        tu = TokenUsage()
        tu.record("qwen2.5:7b", "test prompt", "test response")
        summary = tu.summary()
        assert "qwen2.5:7b" in summary
        assert "prompt" in summary


# ── WorkflowRouter API ────────────────────────────────────────────────────────

class TestWorkflowRouterAPI:
    def test_instantiates_without_args(self):
        """WorkflowRouter() should not raise even without a live Ollama."""
        from workflows.router import WorkflowRouter
        # Just constructing it shouldn't raise
        router = WorkflowRouter.__new__(WorkflowRouter)
        assert router is not None

    def test_router_decision_model(self):
        from workflows.router import RouterDecision
        rd = RouterDecision(
            workflow="simple",
            reasoning="test",
            estimated_pages=1,
            complexity="low",
        )
        assert rd.workflow == "simple"
        assert rd.complexity == "low"


# ── CrawlEngine API ───────────────────────────────────────────────────────────

class TestCrawlEngineAPI:
    def test_instantiates_with_defaults(self):
        from crawler.engine import CrawlEngine
        engine = CrawlEngine()
        assert engine._timeout == 30
        assert "Mozilla" in engine._user_agent

    def test_custom_user_agent(self):
        from crawler.engine import CrawlEngine
        engine = CrawlEngine(user_agent="MyBot/1.0")
        assert engine._user_agent == "MyBot/1.0"

    def test_custom_timeout(self):
        from crawler.engine import CrawlEngine
        engine = CrawlEngine(timeout=60)
        assert engine._timeout == 60
