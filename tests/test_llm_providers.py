"""Provider abstraction + token usage tracking — all HTTP mocked, no keys needed."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel
from typer.testing import CliRunner

import llm.providers as P
from llm.client import ChatClient, make_llm_client, resolve_models, token_usage
from llm.usage import UsageStore, estimate_cost, set_current_run

FAKE_KEY = "sk-test-SECRET-123"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for names in P.KEY_ENV.values():
        for n in names:
            monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv("CRAWLER_ENV_FILE", str(tmp_path / "no-env-file"))
    monkeypatch.delenv("CRAWLER_LLM__PROVIDER", raising=False)
    monkeypatch.setattr(P, "_ENV_LOADED", False)
    set_current_run(None)


class _Resp:
    def __init__(self, status, body):
        self.status, self._body = status, body

    async def text(self):
        return self._body if isinstance(self._body, str) else json.dumps(self._body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class FakeSession:
    """Captures the request; returns a canned response."""
    last: dict = {}

    def __init__(self, status=200, body=None):
        self.status, self.body = status, body

    def __call__(self):  # acts as session_factory
        return self

    def post(self, url, json=None, headers=None, timeout=None):
        FakeSession.last = {"url": url, "json": json, "headers": headers or {}}
        return _Resp(self.status, self.body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


MSGS = [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}]

RESPONSES = {
    "ollama": {"message": {"content": "{\"a\":1}"}, "prompt_eval_count": 11, "eval_count": 4},
    "openai": {"choices": [{"message": {"content": "{\"a\":1}"}}],
               "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}},
    "deepseek": {"choices": [{"message": {"content": "{\"a\":1}"}}],
                 "usage": {"prompt_tokens": 13, "completion_tokens": 6, "total_tokens": 19}},
    "anthropic": {"content": [{"type": "text", "text": "{\"a\":1}"}],
                  "usage": {"input_tokens": 14, "output_tokens": 7}},
    "gemini": {"candidates": [{"content": {"parts": [{"text": "{\"a\":1}"}]}}],
               "usageMetadata": {"promptTokenCount": 15, "candidatesTokenCount": 8,
                                 "thoughtsTokenCount": 2, "totalTokenCount": 25}},
}
EXPECTED = {  # (prompt, completion, total)
    "ollama": (11, 4, 15), "openai": (12, 5, 17), "deepseek": (13, 6, 19),
    "anthropic": (14, 7, 21), "gemini": (15, 10, 25),
}
KEYVAR = {"openai": "OPENAI_API_KEY", "deepseek": "DEEPSEEK_API_KEY",
          "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}


@pytest.mark.parametrize("name", list(RESPONSES))
async def test_provider_parses_usage_and_builds_request(name, monkeypatch):
    if name in KEYVAR:
        monkeypatch.setenv(KEYVAR[name], FAKE_KEY)
    prov = P.make_provider(name, session_factory=FakeSession(200, RESPONSES[name]))
    r = await prov.chat("m-1", MSGS, json_mode=True)
    assert r.text == '{"a":1}' and r.provider == name and r.model == "m-1"
    assert (r.prompt_tokens, r.completion_tokens, r.total_tokens) == EXPECTED[name]
    req = FakeSession.last
    assert FAKE_KEY not in req["url"]  # keys never in URLs
    if name == "gemini":
        assert req["url"].endswith("/models/m-1:generateContent")
        assert req["headers"]["x-goog-api-key"] == FAKE_KEY
        assert req["json"]["systemInstruction"]["parts"][0]["text"] == "be terse"
        assert req["json"]["generationConfig"]["responseMimeType"] == "application/json"
    elif name == "anthropic":
        assert req["url"].endswith("/v1/messages")
        assert req["headers"]["x-api-key"] == FAKE_KEY and req["json"]["system"].startswith("be terse")
        assert all(m["role"] != "system" for m in req["json"]["messages"])
    elif name in ("openai", "deepseek"):
        assert req["url"].endswith("/chat/completions")
        assert req["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
        assert req["json"]["response_format"] == {"type": "json_object"}
        assert ("api.deepseek.com" in req["url"]) == (name == "deepseek")
    else:
        assert req["url"].endswith("/api/chat") and req["json"]["format"] == "json"


def test_google_api_key_fallback(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", FAKE_KEY)
    assert P.api_key_for("gemini") == (FAKE_KEY, "GOOGLE_API_KEY")


@pytest.mark.parametrize("name", ["gemini", "anthropic", "openai", "deepseek"])
async def test_missing_key_raises_without_request(name):
    FakeSession.last = {}
    prov = P.make_provider(name, session_factory=FakeSession(200, {}))
    with pytest.raises(P.MissingAPIKey) as ei:
        await prov.chat("m", MSGS)
    assert KEYVAR[name] in str(ei.value)
    assert FakeSession.last == {}


async def test_http_error_is_redacted(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    prov = P.make_provider("openai", session_factory=FakeSession(401, f"bad key {FAKE_KEY}"))
    with pytest.raises(P.ProviderError) as ei:
        await prov.chat("m", MSGS)
    assert FAKE_KEY not in str(ei.value) and "HTTP 401" in str(ei.value) and ei.value.status == 401


def test_env_file_loads_names_only(tmp_path, monkeypatch):
    f = tmp_path / "env"
    f.write_text(f"# comment\nexport GEMINI_API_KEY='{FAKE_KEY}'\nOTHER=1\n")
    loaded = P.load_env_file(f)
    assert loaded == ["GEMINI_API_KEY", "OTHER"]
    assert P.api_key_for("gemini")[1] == "GEMINI_API_KEY"
    status = P.key_status()
    assert status["gemini"]["configured"] and FAKE_KEY not in json.dumps(status)
    monkeypatch.delenv("OTHER", raising=False)


def test_unknown_provider():
    with pytest.raises(ValueError):
        P.make_provider("nope")


# ── usage store / cost ───────────────────────────────────────────────────────

def test_estimate_cost():
    prices = {"gemini-2.5-flash": {"input": 0.30, "output": 2.50}}
    assert estimate_cost(prices, "gemini-2.5-flash", 1_000_000, 100_000) == 0.55
    assert estimate_cost(prices, "gemini-2.5-flash-001", 1_000_000, 0) == 0.3  # prefix
    assert estimate_cost(prices, "other", 10, 10) is None
    assert estimate_cost({}, "x", 1, 1) is None


def test_usage_store_summary(tmp_path):
    s = UsageStore(tmp_path / "u.db")
    s.record(provider="gemini", model="g", prompt_tokens=10, completion_tokens=5, run_id="r1", cost_usd=0.01)
    s.record(provider="gemini", model="g", prompt_tokens=20, completion_tokens=5, run_id="r2")
    s.record(provider="openai", model="o", prompt_tokens=1, completion_tokens=1, run_id="r1", ok=False, error="x")
    tot = s.summary()["totals"]
    assert tot["calls"] == 3 and tot["total_tokens"] == 42 and tot["errors"] == 1
    assert tot["cost_usd"] == 0.01 and tot["unpriced_calls"] == 1
    assert s.summary(provider="gemini", group_by="run")["groups"][0]["run_id"] == "r2"
    assert s.summary(run_id="r1")["totals"]["calls"] == 2
    with pytest.raises(ValueError):
        s.summary(group_by="bogus")


# ── ChatClient integration ───────────────────────────────────────────────────

class _Schema(BaseModel):
    a: int


def _config(tmp_path, provider="gemini", **llm):
    from config import load_config
    cfg = load_config()
    cfg.llm.provider = provider
    cfg.storage.db_path = str(tmp_path / "crawl.db")
    for k, v in llm.items():
        setattr(cfg.llm, k, v)
    return cfg


async def test_client_records_usage_with_run_and_cost(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    cfg = _config(tmp_path, prices={"gemini-2.5-flash": {"input": 1.0, "output": 2.0}})
    client = make_llm_client(cfg, session_factory=FakeSession(200, RESPONSES["gemini"]))
    assert client.provider_name == "gemini" and client.extractor_model == "gemini-2.5-flash"
    set_current_run("sess-1")
    out = await client.structured_call(client.extractor_model, "sys", "page", _Schema)
    assert out.a == 1
    row = UsageStore(cfg.storage.db_path).rows()[0]
    assert row["run_id"] == "sess-1" and row["provider"] == "gemini"
    assert row["purpose"] == "_Schema" and row["estimated"] == 0
    assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (15, 10, 25)
    assert row["cost_usd"] == round(15 / 1e6 * 1.0 + 10 / 1e6 * 2.0, 6)
    assert "gemini:gemini-2.5-flash" in token_usage.as_dict()["per_model"]


async def test_client_estimates_when_provider_omits_usage(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    cfg = _config(tmp_path, provider="openai")
    body = {"choices": [{"message": {"content": "{\"a\": 2}"}}]}
    client = make_llm_client(cfg, session_factory=FakeSession(200, body))
    await client.raw_call("gpt-x", MSGS)
    row = UsageStore(cfg.storage.db_path).rows()[0]
    assert row["estimated"] == 1 and row["prompt_tokens"] > 0


async def test_client_records_failed_calls_and_retries(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    cfg = _config(tmp_path, provider="anthropic", max_retries=2)
    client = make_llm_client(cfg, session_factory=FakeSession(500, "boom"))
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    with pytest.raises(RuntimeError):
        await client.raw_call(client.extractor_model, MSGS)
    rows = UsageStore(cfg.storage.db_path).rows()
    assert len(rows) == 2 and all(r["ok"] == 0 for r in rows)


async def _no_sleep(*_a, **_k):
    return None


async def test_missing_key_not_retried(tmp_path):
    cfg = _config(tmp_path, provider="deepseek")
    client = make_llm_client(cfg, session_factory=FakeSession(200, {}))
    with pytest.raises(P.MissingAPIKey):
        await client.raw_call("deepseek-chat", MSGS)


def test_resolve_models(tmp_path):
    cfg = _config(tmp_path, provider="ollama")
    assert resolve_models(cfg, "ollama")["navigator_model"] == cfg.ollama.navigator_model
    assert resolve_models(cfg, "ollama", "x")["extractor_model"] == "x"
    assert set(resolve_models(cfg, "gemini", "g-1").values()) == {"g-1"}
    cfg.llm.model, cfg.llm.router_model = "base", "router-special"
    m = resolve_models(cfg, "anthropic")
    assert m["navigator_model"] == "base" and m["router_model"] == "router-special"
    assert resolve_models(_config(tmp_path), "deepseek")["extractor_model"] == "deepseek-chat"


def test_backcompat_ollama_client_alias():
    from llm.client import OllamaClient
    from config import load_config
    c = OllamaClient(load_config().ollama)
    assert c.provider_name == "ollama" and c.usage_store is None


# ── CLI ──────────────────────────────────────────────────────────────────────

runner = CliRunner()


def _app():
    from main import app
    return app


def test_cli_usage_json(tmp_path, monkeypatch):
    db = tmp_path / "c.db"
    monkeypatch.setenv("CRAWLER_STORAGE__DB_PATH", str(db))
    UsageStore(db).record(provider="gemini", model="g", prompt_tokens=3, completion_tokens=2, run_id="r")
    r = runner.invoke(_app(), ["usage", "--json", "--provider", "gemini", "--group-by", "run", "--calls", "5"])
    assert r.exit_code == 0, r.output
    d = json.loads(r.stdout)
    assert d["totals"]["total_tokens"] == 5 and d["groups"][0]["run_id"] == "r" and len(d["calls"]) == 1
    r = runner.invoke(_app(), ["usage", "--json", "--group-by", "bogus"])
    assert r.exit_code == 2


def test_cli_llm_ping_missing_key(tmp_path, monkeypatch):
    monkeypatch.setenv("CRAWLER_STORAGE__DB_PATH", str(tmp_path / "c.db"))
    r = runner.invoke(_app(), ["llm-ping", "--provider", "gemini", "--json"])
    assert r.exit_code == 3
    assert "GEMINI_API_KEY" in json.loads(r.stdout)["error"]
    r = runner.invoke(_app(), ["llm-ping", "--provider", "gemini", "--dry-run", "--json"])
    assert r.exit_code == 3 and json.loads(r.stdout)["ready"] is False


def test_cli_llm_ping_records_tokens(tmp_path, monkeypatch):
    db = tmp_path / "c.db"
    monkeypatch.setenv("CRAWLER_STORAGE__DB_PATH", str(db))
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setattr(P.aiohttp, "ClientSession", FakeSession(200, RESPONSES["gemini"]))
    r = runner.invoke(_app(), ["llm-ping", "--provider", "gemini", "--model", "gemini-2.5-flash-lite", "--json"])
    assert r.exit_code == 0, r.output
    d = json.loads(r.stdout)
    assert d["usage"] == {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25,
                          "provider_reported": True}
    assert FAKE_KEY not in r.output
    row = UsageStore(db).rows()[0]
    assert row["run_id"] == "llm-ping" and row["model"] == "gemini-2.5-flash-lite"


def test_cli_unknown_provider():
    r = runner.invoke(_app(), ["crawl", "-g", "x", "-u", "https://a.com", "--provider", "nope", "--json"])
    assert r.exit_code == 2


def test_cli_crawl_dry_run_with_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("CRAWLER_STORAGE__DB_PATH", str(tmp_path / "c.db"))
    r = runner.invoke(_app(), ["crawl", "-g", "x", "-u", "https://a.com", "--provider", "gemini",
                               "--dry-run", "--json"])
    assert r.exit_code == 3  # no key
    plan = json.loads(r.stdout)
    assert plan["provider"] == "gemini" and plan["provider_ready"] is False
    assert plan["models"]["extractor_model"] == "gemini-2.5-flash"
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    r = runner.invoke(_app(), ["crawl", "-g", "x", "-u", "https://a.com", "-P", "gemini",
                               "-m", "gemini-2.5-flash-lite", "--dry-run", "--json"])
    assert r.exit_code == 0 and FAKE_KEY not in r.output


def test_doctor_lists_providers(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
    r = runner.invoke(_app(), ["doctor", "--json"])
    d = json.loads(r.stdout)
    assert d["providers"]["deepseek"]["configured"] is True
    assert d["providers"]["gemini"]["configured"] is False
    assert d["providers"]["gemini"]["expected_env"] == ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
    assert FAKE_KEY not in r.output
