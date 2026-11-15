"""Live model listing (mocked HTTP), model validation, the `models` CLI and web endpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import llm.models as M  # noqa: E402
import llm.providers as P  # noqa: E402

FAKE_KEY = "sk-test-SECRET-123"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for names in P.KEY_ENV.values():
        for n in names:
            monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv("CRAWLER_ENV_FILE", str(tmp_path / "no-env-file"))
    monkeypatch.delenv("CRAWLER_LLM__PROVIDER", raising=False)
    monkeypatch.delenv("CRAWLER_LLM__MODEL", raising=False)
    monkeypatch.setattr(P, "_ENV_LOADED", False)
    M.clear_cache()
    yield
    M.clear_cache()


class _Resp:
    def __init__(self, status, body):
        self.status, self._body = status, body

    async def json(self, content_type=None):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class FakeGetSession:
    """session_factory: maps URL substrings to (status, body); records every GET."""

    def __init__(self, routes, raise_exc=None):
        self.routes, self.raise_exc, self.calls = routes, raise_exc, []

    def __call__(self):
        return self

    def get(self, url, headers=None, timeout=None):
        self.calls.append({"url": url, "headers": headers or {}})
        if self.raise_exc:
            raise self.raise_exc
        for frag, (status, body) in self.routes.items():
            if frag in url:
                return _Resp(status, body)
        return _Resp(404, {"error": "no route"})

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


GEMINI_P1 = {"models": [
    {"name": "models/gemini-3.5-flash-lite", "displayName": "Gemini 3.5 Flash-Lite",
     "supportedGenerationMethods": ["generateContent", "countTokens"]},
    {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
], "nextPageToken": "tok2"}
GEMINI_P2 = {"models": [
    {"name": "models/gemini-3.5-flash", "displayName": "Gemini 3.5 Flash",
     "supportedGenerationMethods": ["generateContent"]},
]}


class TestListModels:
    async def test_gemini_paginates_and_filters_generate_content(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
        s = FakeGetSession({"pageToken=tok2": (200, GEMINI_P2), "/models?": (200, GEMINI_P1)})
        r = await M.list_models("gemini", session_factory=s)
        assert r["configured"] and r["error"] is None
        assert [m["id"] for m in r["models"]] == ["gemini-3.5-flash", "gemini-3.5-flash-lite"]
        assert r["default_model"] == "gemini-3.5-flash-lite"
        assert r["role_defaults"] == {"extractor_model": "gemini-3.5-flash"}
        assert len(s.calls) == 2
        assert all(FAKE_KEY not in c["url"] for c in s.calls)          # key only in header
        assert s.calls[0]["headers"]["x-goog-api-key"] == FAKE_KEY
        assert FAKE_KEY not in json.dumps(r)

    async def test_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
        s = FakeGetSession({"/v1/models": (200, {"data": [{"id": "gpt-4.1-mini"}, {"id": "gpt-4.1"}]})})
        r = await M.list_models("openai", session_factory=s)
        assert [m["id"] for m in r["models"]] == ["gpt-4.1", "gpt-4.1-mini"]
        assert s.calls[0]["url"] == "https://api.openai.com/v1/models"
        assert s.calls[0]["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"

    async def test_anthropic_pages(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
        s = FakeGetSession({
            "after_id=m1": (200, {"data": [{"id": "claude-haiku-4-5", "display_name": "Haiku"}],
                                  "has_more": False}),
            "/v1/models": (200, {"data": [{"id": "claude-sonnet-4-5", "display_name": "Sonnet"}],
                                 "has_more": True, "last_id": "m1"}),
        })
        r = await M.list_models("anthropic", session_factory=s)
        assert [m["id"] for m in r["models"]] == ["claude-haiku-4-5", "claude-sonnet-4-5"]
        h = s.calls[0]["headers"]
        assert h["x-api-key"] == FAKE_KEY and h["anthropic-version"]

    async def test_deepseek(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
        s = FakeGetSession({"/models": (200, {"data": [{"id": "deepseek-chat"}, {"id": "deepseek-reasoner"}]})})
        r = await M.list_models("deepseek", session_factory=s)
        assert [m["id"] for m in r["models"]] == ["deepseek-chat", "deepseek-reasoner"]
        assert s.calls[0]["url"].endswith("/models")

    async def test_ollama_tags(self):
        s = FakeGetSession({"/api/tags": (200, {"models": [{"name": "qwen2.5:7b"}]})})
        r = await M.list_models("ollama", session_factory=s)
        assert r["configured"] and [m["id"] for m in r["models"]] == ["qwen2.5:7b"]

    async def test_ollama_unreachable_is_not_configured(self):
        s = FakeGetSession({}, raise_exc=ConnectionRefusedError("refused"))
        r = await M.list_models("ollama", session_factory=s)
        assert r["configured"] is False and "refused" in r["error"]

    async def test_missing_key_is_not_an_error(self):
        s = FakeGetSession({})
        r = await M.list_models("openai", session_factory=s)
        assert r["configured"] is False and r["error"] is None and r["models"] == []
        assert s.calls == []

    async def test_http_error_redacts_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
        s = FakeGetSession({"/models": (401, {"error": {"message": f"bad key {FAKE_KEY}"}})})
        r = await M.list_models("openai", session_factory=s)
        assert r["models"] == [] and "HTTP 401" in r["error"] and FAKE_KEY not in r["error"]

    async def test_cache_and_refresh(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
        s = FakeGetSession({"/models": (200, {"data": [{"id": "deepseek-chat"}]})})
        a = await M.list_models("deepseek", session_factory=s)
        b = await M.list_models("deepseek", session_factory=s)
        assert len(s.calls) == 1 and a["cached"] is False and b["cached"] is True
        await M.list_models("deepseek", session_factory=s, refresh=True)
        assert len(s.calls) == 2

    async def test_errors_are_not_cached(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
        bad = FakeGetSession({"/models": (500, {"error": "boom"})})
        await M.list_models("openai", session_factory=bad)
        await M.list_models("openai", session_factory=bad)
        assert len(bad.calls) == 2

    async def test_unknown_provider(self):
        with pytest.raises(ValueError):
            await M.list_models("nope")


def _listing(provider, ids, **kw):
    return {"provider": provider, "configured": True, "models": [{"id": i, "name": i} for i in ids],
            "error": None, **kw}


class TestCheckModel:
    def test_valid(self):
        assert M.check_model(_listing("gemini", ["gemini-3.5-flash"]), "gemini-3.5-flash")["valid"]

    def test_models_prefix_accepted(self):
        assert M.check_model(_listing("gemini", ["gemini-3.5-flash"]), "models/gemini-3.5-flash")["valid"]

    def test_unknown_suggests_closest(self):
        r = M.check_model(_listing("gemini", ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemma-4-31b-it"]),
                          "gemini-3.5-flsh")
        assert r["valid"] is False and r["suggestions"][0] == "gemini-3.5-flash"

    def test_ollama_latest_tag(self):
        assert M.check_model(_listing("ollama", ["llama3:latest"]), "llama3")["valid"]

    def test_unlistable_is_undecided(self):
        r = M.check_model({"provider": "openai", "configured": False, "models": [], "error": None},
                          "gpt-x")
        assert r["valid"] is None


# ── CLI ──────────────────────────────────────────────────────────────────────

GEMINI_IDS = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]


@pytest.fixture
def fake_list(monkeypatch):
    calls = []

    async def _fake(provider, **kw):
        calls.append(provider)
        key, env = P.api_key_for(provider) if provider != "ollama" else ("x", None)
        if not key:
            return {"provider": provider, "configured": False, "env_var": None, "models": [],
                    "default_model": P.default_model(provider), "role_defaults": {},
                    "error": None, "cached": False}
        ids = GEMINI_IDS if provider == "gemini" else [P.default_model(provider)]
        return {**_listing(provider, ids), "env_var": env, "default_model": P.default_model(provider),
                "role_defaults": P.DEFAULT_ROLE_MODELS.get(provider, {}), "cached": False}

    monkeypatch.setattr(M, "list_models", _fake)
    return calls


def _run(args):
    from typer.testing import CliRunner
    from main import app
    return CliRunner().invoke(app, args)


class TestModelsCLI:
    def test_models_json_single_provider(self, monkeypatch, fake_list):
        monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
        r = _run(["models", "-P", "gemini", "--json"])
        assert r.exit_code == 0, r.output
        data = json.loads(r.stdout)
        g = data["providers"][0]
        assert g["provider"] == "gemini" and g["configured"] and g["default_model_listed"] is True
        assert [m["id"] for m in g["models"]] == GEMINI_IDS
        assert FAKE_KEY not in r.output

    def test_models_json_all_providers_reports_missing_keys(self, fake_list):
        r = _run(["models", "--json"])
        assert r.exit_code == 0, r.output
        st = {p["provider"]: p["configured"] for p in json.loads(r.stdout)["providers"]}
        assert set(st) == set(P.PROVIDERS) and st["openai"] is False

    def test_models_unknown_provider_exit_2(self):
        assert _run(["models", "-P", "nope", "--json"]).exit_code == 2

    def test_unknown_model_exits_2_with_suggestions(self, monkeypatch, fake_list):
        monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
        r = _run(["llm-ping", "-P", "gemini", "-m", "gemini-3.5-flsh", "--json"])
        assert r.exit_code == 2, r.output
        err = json.loads(r.stdout)
        assert err["exit_code"] == 2 and "gemini-3.5-flash" in err["suggestions"]
        assert FAKE_KEY not in r.output

    def test_valid_model_dry_run(self, monkeypatch, fake_list):
        monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
        r = _run(["llm-ping", "-P", "gemini", "-m", "gemini-3.5-flash", "--dry-run", "--json"])
        assert r.exit_code == 0, r.output
        assert json.loads(r.stdout)["model_check"]["valid"] is True

    def test_no_validate_skips_listing(self, monkeypatch, fake_list):
        monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
        r = _run(["llm-ping", "-P", "gemini", "-m", "made-up-model", "--no-validate",
                  "--dry-run", "--json"])
        assert r.exit_code == 0, r.output
        assert fake_list == [] and json.loads(r.stdout)["model"] == "made-up-model"

    @pytest.mark.parametrize("cmd", [
        ["crawl", "-g", "x", "-u", "https://example.com", "--dry-run", "--json"],
        ["research", "-g", "x", "-u", "https://example.com", "--dry-run"],
        ["resume", "some-session"],
    ])
    def test_every_llm_command_validates(self, monkeypatch, fake_list, cmd):
        monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
        r = _run(cmd + ["-P", "gemini", "-m", "gemini-9-nope"])
        assert r.exit_code == 2, r.output

    def test_missing_key_does_not_block_validation_path(self, fake_list):
        # No key: list can't be fetched -> not a usage error; readiness check decides (exit 3)
        r = _run(["llm-ping", "-P", "openai", "-m", "gpt-x", "--json"])
        assert r.exit_code == 3, r.output


# ── Web endpoints ────────────────────────────────────────────────────────────

@pytest.fixture
def client(monkeypatch, tmp_path, fake_list):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"storage:\n  db_path: {tmp_path / 'web.db'}\nllm:\n  provider: gemini\n")
    monkeypatch.setenv("CRAWLER_CONFIG", str(cfg))
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    from fastapi.testclient import TestClient
    from web.app import app
    with TestClient(app) as c:
        yield c


class TestWebAPI:
    def test_providers_status_no_keys_leak(self, client):
        r = client.get("/api/models")
        assert r.status_code == 200
        d = r.json()
        st = {p["provider"]: p for p in d["providers"]}
        assert d["active_provider"] == "gemini"
        assert st["gemini"]["configured"] and st["gemini"]["status"] == "ok"
        assert st["openai"]["configured"] is False and st["openai"]["status"] == "no key"
        assert FAKE_KEY not in r.text

    def test_models_for_provider(self, client):
        r = client.get("/api/models", params={"provider": "gemini"})
        d = r.json()
        assert [m["id"] for m in d["models"]] == GEMINI_IDS
        assert d["default_model"] == "gemini-3.5-flash-lite" and FAKE_KEY not in r.text

    def test_models_unknown_provider_400(self, client):
        assert client.get("/api/models", params={"provider": "nope"}).status_code == 400

    def test_usage_endpoint(self, client, tmp_path):
        from llm.usage import UsageStore
        store = UsageStore(str(tmp_path / "web.db"))
        store.record(run_id="run-1", provider="gemini", model="gemini-3.5-flash-lite",
                     purpose="t", prompt_tokens=10, completion_tokens=5, total_tokens=15,
                     latency_ms=100)
        r = client.get("/api/usage", params={"run_id": "run-1"})
        assert r.status_code == 200
        d = r.json()
        assert d["totals"]["total_tokens"] == 15 and d["recent"][0]["model"] == "gemini-3.5-flash-lite"
        assert "db_path" not in d

    def test_crawl_rejects_unknown_model(self, client):
        r = client.post("/api/crawl", json={"goal": "g", "start_url": "https://example.com",
                                            "provider": "gemini", "model": "gemini-3.5-flsh"})
        assert r.status_code == 400
        assert "gemini-3.5-flash" in r.json()["suggestions"]

    def test_crawl_rejects_provider_without_key(self, client):
        r = client.post("/api/crawl", json={"goal": "g", "start_url": "https://example.com",
                                            "provider": "openai"})
        assert r.status_code == 400 and "no API key" in r.json()["detail"]

    def test_crawl_passes_provider_and_model(self, client, monkeypatch):
        import web.routes.api as api
        seen = {}

        async def _fake_bg(**kw):
            seen.update(kw)
        monkeypatch.setattr(api, "_run_crawl_background", _fake_bg)
        r = client.post("/api/crawl", data={"goal": "g", "start_url": "https://example.com",
                                            "workflow": "research", "provider": "gemini",
                                            "model": "gemini-3.5-flash"})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["provider"] == "gemini" and d["model"] == "gemini-3.5-flash"
        assert d["usage_url"].endswith(d["session_id"])
        assert seen["provider"] == "gemini" and seen["model"] == "gemini-3.5-flash"
        assert seen["workflow"] == "research"
