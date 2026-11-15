"""Agent-readiness tests: --json output, --dry-run, exit codes (no network/Ollama)."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from main import app
from utils import exit_codes as rc

runner = CliRunner()


def _json(result):
    return json.loads(result.stdout)


def test_radar_help_lists_commands():
    r = runner.invoke(app, ["radar", "--help"])
    assert r.exit_code == 0
    for cmd in ("crawl", "pages", "show", "export", "runs", "stats", "topics"):
        assert cmd in r.stdout


def test_radar_crawl_dry_run_json(tmp_path):
    db = tmp_path / "r.db"
    r = runner.invoke(app, ["radar", "crawl", "--dry-run", "--json", "--db", str(db),
                            "--topic", "ai-security", "--max-pages", "50"])
    assert r.exit_code == rc.OK, r.output
    plan = _json(r)
    assert plan["dry_run"] is True and plan["limits"]["max_pages"] == 50
    assert plan["seeds"] and not db.exists()  # dry run never creates the DB


def test_radar_crawl_missing_seeds_file(tmp_path):
    r = runner.invoke(app, ["radar", "crawl", "--dry-run", "--json",
                            "--seeds", str(tmp_path / "nope.yaml")])
    assert r.exit_code == rc.DEPENDENCY
    assert _json(r)["exit_code"] == rc.DEPENDENCY


def test_radar_crawl_bad_kind():
    r = runner.invoke(app, ["radar", "crawl", "--dry-run", "--json", "--url", "https://a.com",
                            "--kind", "pdf"])
    assert r.exit_code == rc.USAGE


def test_radar_query_commands_on_empty_db(tmp_path):
    db = str(tmp_path / "r.db")
    for args in (["pages"], ["runs"], ["stats"], ["topics"]):
        r = runner.invoke(app, ["radar", *args, "--json"] + (["--db", db] if args != ["topics"] else []))
        assert r.exit_code == 0, (args, r.output)
        _json(r)  # valid JSON
    r = runner.invoke(app, ["radar", "show", "999", "--db", db, "--json"])
    assert r.exit_code == rc.NOT_FOUND


def test_radar_pages_and_export_with_data(tmp_path):
    from radar.db import RadarDB
    db = str(tmp_path / "r.db")
    with RadarDB(db) as store:
        store.upsert_page({"url": "https://a.com/x", "domain": "a.com", "title": "Anvil",
                           "content_hash": "h", "text_md": "anthropic red team",
                           "published_at": "2026-11-13T00:00:00+00:00"}, ["ai-security"], None)
    r = runner.invoke(app, ["radar", "pages", "--db", db, "--json", "--topic", "ai-security"])
    assert r.exit_code == 0 and _json(r)[0]["title"] == "Anvil"
    r = runner.invoke(app, ["radar", "export", "--db", db, "--format", "jsonl"])
    assert r.exit_code == 0 and json.loads(r.stdout.splitlines()[0])["url"] == "https://a.com/x"
    out = tmp_path / "e.json"
    r = runner.invoke(app, ["radar", "export", "--db", db, "-o", str(out)])
    assert r.exit_code == 0 and json.loads(out.read_text())[0]["text_md"] == "anthropic red team"
    r = runner.invoke(app, ["radar", "pages", "--db", db, "--q", "\"unclosed", "--json"])
    assert r.exit_code == rc.USAGE


def test_crawl_requires_start_url_exit_usage():
    r = runner.invoke(app, ["crawl", "--goal", "x", "--json"])
    assert r.exit_code == rc.USAGE
    assert _json(r)["exit_code"] == rc.USAGE


def test_crawl_bad_workflow_exit_usage():
    r = runner.invoke(app, ["crawl", "--goal", "x", "-u", "https://a.com", "-w", "nope"])
    assert r.exit_code == rc.USAGE


def test_crawl_dry_run_json(monkeypatch, tmp_path):
    async def fake_reachable(url):
        return False
    monkeypatch.setattr("llm.client.check_ollama_reachable", fake_reachable)
    monkeypatch.setenv("CRAWLER_STORAGE__DB_PATH", str(tmp_path / "c.db"))
    r = runner.invoke(app, ["crawl", "--goal", "x", "-u", "https://a.com", "--dry-run", "--json"])
    assert r.exit_code == rc.DEPENDENCY  # plan printed, but Ollama is down
    plan = _json(r)
    assert plan["dry_run"] and plan["ollama_reachable"] is False


def test_list_sessions_json(monkeypatch, tmp_path):
    monkeypatch.setenv("CRAWLER_STORAGE__DB_PATH", str(tmp_path / "c.db"))
    r = runner.invoke(app, ["list-sessions", "--json"])
    assert r.exit_code == 0 and _json(r) == []


def test_doctor_json():
    r = runner.invoke(app, ["doctor", "--json"])
    assert r.exit_code in (rc.OK, rc.DEPENDENCY)
    rep = _json(r)
    assert {"core_deps", "ollama", "radar_ready", "llm_crawl_ready"} <= rep.keys()
