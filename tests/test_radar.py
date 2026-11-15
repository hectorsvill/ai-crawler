"""Tests for the radar package: DB, extraction, feed parsing, tagging, crawler (offline)."""

from __future__ import annotations

import asyncio

import pytest

from radar.db import RadarDB
from radar.extract import (
    content_hash, domain_of, extract_metadata, is_article_link, normalize_url, parse_feed, to_iso,
)
from radar.topics import tag_text


ARTICLE_HTML = """
<html><head>
<title>Fallback title</title>
<meta property="og:title" content="Nvidia Blackwell export controls tighten">
<meta property="article:published_time" content="2026-11-13T09:30:00-05:00">
<meta name="description" content="BIS updates rules on GPUs.">
<link rel="canonical" href="https://example.com/news/nvidia-blackwell-export?utm_source=x">
</head><body><article>
<h1>Nvidia Blackwell export controls tighten</h1>
<p>The Bureau of Industry and Security published new export controls on Nvidia GPUs.
This paragraph exists so that the extractor has enough text to work with, repeated a bit.
The Bureau of Industry and Security published new export controls on Nvidia GPUs.</p>
<a href="/news/2026/11/another-story-about-chips">Another</a>
<a href="/tag/chips">Tag</a>
<a href="https://other.com/x">Other</a>
</article></body></html>
"""

RSS = """<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
<item><title>Iran says it struck radar in Bahrain</title>
<link>https://news.example.org/iran-bahrain-radar?utm_medium=rss</link>
<pubDate>Fri, 13 Nov 2026 10:00:00 GMT</pubDate>
<description>&lt;p&gt;IRGC claim&lt;/p&gt;</description></item>
<item><title>Football results</title><link>https://news.example.org/football</link>
<pubDate>Fri, 13 Nov 2026 11:00:00 GMT</pubDate></item>
</channel></rss>"""

ATOM = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>x</title>
<entry><title>OpenAI ships a thing</title>
<link rel="alternate" href="https://atom.example.com/2026/11/12/openai-ships-a-thing/"/>
<published>2026-11-12T08:00:00Z</published><summary>sum</summary></entry>
</feed>"""


# ── extract ──────────────────────────────────────────────────────────────────

class TestExtract:
    def test_normalize_strips_tracking_and_fragment(self):
        assert normalize_url("HTTPS://Ex.com/a?utm_source=x&id=3#frag") == "https://ex.com/a?id=3"
        assert normalize_url("https://bbc.co.uk/n/a?at_medium=RSS&at_campaign=rss") == "https://bbc.co.uk/n/a"

    def test_domain_of_strips_www(self):
        assert domain_of("https://www.bbc.com/news") == "bbc.com"

    def test_to_iso_formats(self):
        assert to_iso("Fri, 13 Nov 2026 10:00:00 GMT") == "2026-11-13T10:00:00+00:00"
        assert to_iso("2026-11-13T09:30:00-05:00") == "2026-11-13T14:30:00+00:00"
        assert to_iso("2026-11-12") == "2026-11-12T00:00:00"
        assert to_iso("not a date") is None
        assert to_iso(None) is None

    def test_metadata(self):
        m = extract_metadata(ARTICLE_HTML, "https://example.com/news/nvidia-blackwell-export")
        assert m["title"] == "Nvidia Blackwell export controls tighten"
        assert m["canonical_url"] == "https://example.com/news/nvidia-blackwell-export"
        assert m["published_at"] == "2026-11-13T14:30:00+00:00"
        assert m["description"] == "BIS updates rules on GPUs."
        assert "Bureau of Industry" in m["text_md"]
        assert "https://example.com/news/2026/11/another-story-about-chips" in m["links"]

    def test_jsonld_date(self):
        html = ('<html><head><script type="application/ld+json">'
                '{"@graph":[{"@type":"NewsArticle","datePublished":"2026-11-14T01:00:00Z"}]}'
                '</script></head><body><p>x</p></body></html>')
        assert extract_metadata(html, "https://a.com/x")["published_at"] == "2026-11-14T01:00:00+00:00"

    def test_article_link_heuristic(self):
        seed = "https://example.com/news"
        assert is_article_link("https://example.com/news/2026/11/another-story", seed)
        assert is_article_link("https://example.com/news/some-long-slug-here", seed)
        assert not is_article_link("https://example.com/tag/chips", seed)
        assert not is_article_link("https://other.com/news/some-long-slug-here", seed)
        assert not is_article_link("https://example.com/news", seed)
        assert not is_article_link("https://example.com/img/a-b-c.png", seed)

    def test_content_hash_whitespace_insensitive(self):
        assert content_hash("a  b\n c") == content_hash("a b c")


class TestFeeds:
    def test_rss(self):
        items = parse_feed(RSS, "https://news.example.org/rss")
        assert items[0]["url"] == "https://news.example.org/iran-bahrain-radar"
        assert items[0]["published_at"] == "2026-11-13T10:00:00+00:00"
        assert items[0]["description"] == "IRGC claim"
        assert len(items) == 2

    def test_atom(self):
        items = parse_feed(ATOM, "https://atom.example.com/feed")
        assert items == [{
            "url": "https://atom.example.com/2026/11/12/openai-ships-a-thing/",
            "title": "OpenAI ships a thing",
            "published_at": "2026-11-12T08:00:00+00:00",
            "description": "sum",
        }]

    def test_bad_xml(self):
        assert parse_feed("<not xml", "https://x.com") == []


class TestTopics:
    def test_headline_single_hit(self):
        assert "war-iran-gulf" in tag_text("Iran says it struck radar in Bahrain")

    def test_off_topic(self):
        assert tag_text("Football results", "Arsenal won 2-0") == []

    def test_body_needs_two_hits(self):
        assert "taiwan-pla" not in tag_text("Weather", "Taiwan mentioned once.")
        assert "taiwan-pla" in tag_text("Weather", "Taiwan Strait PLA drills.")


# ── DB ───────────────────────────────────────────────────────────────────────

def _page(url="https://a.com/x", h="h1", text="hello", **kw):
    d = {"url": url, "domain": domain_of(url), "title": "T", "content_hash": h, "text_md": text,
         "published_at": "2026-11-13T10:00:00+00:00", "http_status": 200}
    d.update(kw)
    return d


class TestRadarDB:
    def test_upsert_idempotent(self, tmp_path):
        with RadarDB(tmp_path / "r.db") as db:
            run = db.start_run({"x": 1})
            pid, a1 = db.upsert_page(_page(), ["ai-security"], run)
            pid2, a2 = db.upsert_page(_page(), ["ai-security"], run)
            assert (a1, a2) == ("inserted", "unchanged") and pid == pid2
            _, a3 = db.upsert_page(_page(h="h2", text="changed"), ["frontier-labs"], run)
            assert a3 == "updated"
            p = db.get_page(pid)
            assert p["times_seen"] == 3 and p["text_md"] == "changed"
            assert p["tags"] == ["ai-security", "frontier-labs"]
            assert db.stats()["pages"] == 1

    def test_duplicate_content_other_url(self, tmp_path):
        with RadarDB(tmp_path / "r.db") as db:
            db.upsert_page(_page(), [], None)
            _, action = db.upsert_page(_page(url="https://a.com/y"), [], None)
            assert action == "duplicate_content"
            assert db.get_page(url="https://a.com/y")["text_md"] is None

    def test_query_filters(self, tmp_path):
        with RadarDB(tmp_path / "r.db") as db:
            db.upsert_page(_page("https://a.com/1", "1", "nvidia blackwell gpus",
                                 published_at="2026-11-12T10:00:00+00:00"), ["chips-export-controls"], None)
            db.upsert_page(_page("https://www.b.com/2", "2", "iran bahrain radar",
                                 published_at="2026-11-14T10:00:00+00:00"), ["war-iran-gulf"], None)
            db.upsert_page(_page("https://a.com/3", "3", "old", published_at="2026-10-01"), [], None)
            assert [p["url"] for p in db.query_pages(topic="war-iran-gulf")] == ["https://www.b.com/2"]
            assert len(db.query_pages(since="2026-11-12", until="2026-11-14")) == 2
            assert len(db.query_pages(until="2026-11-13")) == 2
            assert [p["url"] for p in db.query_pages(q="blackwell")] == ["https://a.com/1"]
            assert len(db.query_pages(domain="a.com")) == 2
            assert "text_md" in db.query_pages(limit=1, include_text=True)[0]

    def test_runs(self, tmp_path):
        with RadarDB(tmp_path / "r.db") as db:
            rid = db.start_run({"seeds": ["s"]})
            db.finish_run(rid, "completed", {"fetched": 3, "stored": 2}, [{"url": "u", "error": "e"}])
            r = db.list_runs()[0]
            assert r["status"] == "completed" and r["pages_stored"] == 2
            assert r["params"] == {"seeds": ["s"]} and r["errors"][0]["url"] == "u"

    def test_log_url_upgrade_keeps_fetch_facts(self, tmp_path):
        with RadarDB(tmp_path / "r.db") as db:
            run = db.start_run({})
            db.log_url({
                "run_id": run, "url": "https://a.com/x", "domain": "a.com", "depth": 1,
                "status": "failed", "reason": "HTTP 403", "http_status": 403,
                "discovered_at": "2026-11-15T00:00:00+00:00",
                "fetched_at": "2026-11-15T00:00:01+00:00",
                "discovered_from": "https://a.com/rss", "source_seed": "https://a.com/rss",
            })
            db.log_url({
                "run_id": run, "url": "https://a.com/x", "domain": "a.com", "depth": 1,
                "status": "feed_only", "reason": "HTTP 403",
                "discovered_at": "2026-11-15T00:00:09+00:00",
            })
            # A later "never fetched" note must not clobber the outcome.
            db.log_url({
                "run_id": run, "url": "https://a.com/x", "domain": "a.com",
                "status": "not_fetched", "reason": "budget",
            }, overwrite=False)
            row = db.query_urls(run_id=run)[0]
            assert row["status"] == "feed_only" and row["reason"] == "HTTP 403"
            assert row["http_status"] == 403
            assert row["fetched_at"] == "2026-11-15T00:00:01+00:00"
            assert row["discovered_at"] == "2026-11-15T00:00:00+00:00"
            assert row["discovered_from"] == "https://a.com/rss"
            assert db.count_urls(run) == 1
            assert db.stats()["url_log"] == 1
            with pytest.raises(ValueError):
                db.log_url({"run_id": run, "url": "https://a.com/y", "domain": "a.com",
                            "status": "nope"})

    def test_query_urls_filters_are_stable(self, tmp_path):
        with RadarDB(tmp_path / "r.db") as db:
            first = db.start_run({})
            second = db.start_run({})
            db.log_url({"run_id": first, "url": "https://a.com/1", "domain": "a.com",
                        "status": "stored", "depth": 0})
            db.log_url({"run_id": first, "url": "https://news.a.com/2", "domain": "news.a.com",
                        "status": "skipped", "reason": "robots.txt", "depth": 1})
            db.log_url({"run_id": second, "url": "https://b.com/3", "domain": "b.com",
                        "status": "failed", "reason": "HTTP 500", "depth": 0})
            assert [r["url"] for r in db.query_urls(run_id=first, status="skipped")] == [
                "https://news.a.com/2"]
            assert [r["url"] for r in db.query_urls(domain="a.com")] == [
                "https://a.com/1", "https://news.a.com/2"]
            assert db.query_urls(run_id=second, limit=1, offset=1) == []
            assert db.get_run(first)["id"] == first
            assert db.get_run(999) is None

    def test_migrate_existing_db_keeps_pages_and_fts(self, tmp_path):
        import sqlite3
        from radar.db import FTS_SCHEMA, SCHEMA, SCHEMA_VERSION, utcnow
        path = tmp_path / "old.db"
        conn = sqlite3.connect(path)
        conn.executescript(SCHEMA)
        conn.executescript(FTS_SCHEMA)
        conn.execute(
            """INSERT INTO pages (url, domain, title, description, fetched_at, content_hash, text_md)
               VALUES ('https://a.com/old', 'a.com', 'Old title', 'd', ?, 'h',
                       'export controls on nvidia gpus')""",
            (utcnow(),),
        )
        conn.commit()
        conn.close()

        page_cols = [
            "id", "url", "canonical_url", "domain", "title", "description", "published_at",
            "fetched_at", "http_status", "content_hash", "text_md", "source_seed",
            "first_run_id", "last_run_id", "times_seen",
        ]
        with RadarDB(path) as db:
            assert db.conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
            assert [r[1] for r in db.conn.execute("PRAGMA table_info(pages)")] == page_cols
            assert db.has_fts
            assert db.query_pages(q="nvidia")[0]["url"] == "https://a.com/old"
            assert db.count_urls() == 0
            run = db.start_run({})
            db.log_url({"run_id": run, "url": "https://a.com/skip", "domain": "a.com",
                        "status": "skipped", "reason": "robots.txt"})
            db.upsert_page(_page("https://a.com/new", "n", "fresh iran bahrain text"), [], run)
            assert db.query_pages(q="bahrain")[0]["url"] == "https://a.com/new"
        # Second open is a no-op migration.
        with RadarDB(path) as db:
            assert db.count_urls() == 1
            assert db.query_pages(q="nvidia")[0]["title"] == "Old title"

    def test_migrate_adds_missing_run_url_columns(self, tmp_path):
        import sqlite3
        from radar.db import SCHEMA
        path = tmp_path / "partial.db"
        conn = sqlite3.connect(path)
        conn.executescript(SCHEMA)
        conn.execute(
            """CREATE TABLE run_urls (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   run_id INTEGER NOT NULL,
                   url TEXT NOT NULL,
                   domain TEXT NOT NULL,
                   status TEXT NOT NULL,
                   UNIQUE (run_id, url)
               )"""
        )
        conn.execute(
            "INSERT INTO runs (started_at, params) VALUES ('2026-11-01T00:00:00+00:00', '{}')"
        )
        conn.execute(
            "INSERT INTO run_urls (run_id, url, domain, status) "
            "VALUES (1, 'https://a.com/old', 'a.com', 'failed')"
        )
        conn.commit()
        conn.close()
        with RadarDB(path) as db:
            row = db.query_urls(run_id=1)[0]
            assert row["status"] == "failed" and row["reason"] is None and row["depth"] == 0
            db.log_url({
                "run_id": 1, "url": "https://a.com/old", "domain": "a.com",
                "status": "feed_only", "reason": "HTTP 500", "http_status": 500,
                "fetched_at": "2026-11-01T00:00:02+00:00",
            })
            upgraded = db.query_urls(run_id=1)[0]
            assert upgraded["status"] == "feed_only"
            assert upgraded["http_status"] == 500
            assert upgraded["fetched_at"] == "2026-11-01T00:00:02+00:00"
            assert db.count_urls(1) == 1


# ── Crawler (network mocked) ─────────────────────────────────────────────────

class _FakeResp:
    def __init__(self, status, body, ctype):
        self.status, self._body, self.headers = status, body.encode(), {"content-type": ctype}
        self.charset = "utf-8"
        self.content = self

    async def read(self, n=-1):
        return self._body

    async def iter_chunked(self, n):
        # yield in small pieces to prove the reader loops to EOF
        for i in range(0, len(self._body), 7):
            yield self._body[i:i + 7]

    async def text(self, errors="strict"):
        return self._body.decode()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeSession:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, **kw):
        self.calls.append(url)
        status, body, ctype = self.routes.get(url, (404, "", "text/html"))
        return _FakeResp(status, body, ctype)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _run_crawl(tmp_path, monkeypatch, routes, seeds, **params):
    import aiohttp
    from radar.crawler import CrawlParams, RadarCrawler

    session = _FakeSession(routes)
    monkeypatch.setattr(aiohttp, "ClientSession", lambda *a, **k: session)
    crawl_kwargs = {"delay": 0.5, "recent_days": None}
    crawl_kwargs.update(params)
    p = CrawlParams(**crawl_kwargs)
    monkeypatch.setattr("radar.crawler.DomainThrottle.wait", lambda self, url, cd=None: asyncio.sleep(0))
    db = RadarDB(tmp_path / "c.db")
    result = asyncio.run(RadarCrawler(db, p).run(seeds, {"test": True}))
    return result, db, session


class TestRadarCrawler:
    def test_feed_crawl_with_robots_and_filter(self, tmp_path, monkeypatch):
        from radar.crawler import Seed
        routes = {
            "https://news.example.org/robots.txt": (200, "User-agent: *\nDisallow: /private\n", "text/plain"),
            "https://news.example.org/rss": (200, RSS, "application/rss+xml"),
            "https://news.example.org/iran-bahrain-radar": (200, ARTICLE_HTML.replace(
                "Nvidia Blackwell export controls tighten", "Iran says it struck radar in Bahrain"), "text/html"),
        }
        seeds = [Seed(url="https://news.example.org/rss", kind="feed", require_match=True)]
        result, db, session = _run_crawl(tmp_path, monkeypatch, routes, seeds)
        assert result["status"] == "completed"
        assert result["counters"]["stored"] == 1
        # football item filtered out before fetching
        assert "https://news.example.org/football" not in session.calls
        page = db.get_page(url="https://news.example.org/iran-bahrain-radar")
        assert "war-iran-gulf" in page["tags"]

    def test_feed_only_fallback_when_article_403(self, tmp_path, monkeypatch):
        from radar.crawler import Seed
        routes = {
            "https://news.example.org/robots.txt": (404, "", "text/plain"),
            "https://news.example.org/rss": (200, RSS, "application/rss+xml"),
            "https://news.example.org/iran-bahrain-radar": (403, "denied", "text/html"),
        }
        seeds = [Seed(url="https://news.example.org/rss", kind="feed", require_match=True)]
        result, db, _ = _run_crawl(tmp_path, monkeypatch, routes, seeds)
        assert result["counters"]["feed_only"] == 1 and result["counters"]["failed"] == 1
        page = db.get_page(url="https://news.example.org/iran-bahrain-radar")
        assert page["http_status"] == 403 and page["text_md"] is None
        assert page["title"] == "Iran says it struck radar in Bahrain"
        assert page["published_at"] == "2026-11-13T10:00:00+00:00"

    def test_robots_disallow_and_5xx(self, tmp_path, monkeypatch):
        from radar.crawler import Seed
        routes = {
            "https://blocked.com/robots.txt": (200, "User-agent: *\nDisallow: /\n", "text/plain"),
            "https://down.com/robots.txt": (503, "", "text/plain"),
        }
        seeds = [Seed(url="https://blocked.com/news"), Seed(url="https://down.com/news")]
        result, db, session = _run_crawl(tmp_path, monkeypatch, routes, seeds)
        assert result["status"] == "failed"
        assert result["counters"]["skipped"] == 2
        assert "https://blocked.com/news" not in session.calls
        assert "https://down.com/news" not in session.calls

    def test_html_seed_follows_article_links_and_respects_budget(self, tmp_path, monkeypatch):
        from radar.crawler import Seed
        index = ('<html><head><title>News</title></head><body>'
                 + "".join(f'<a href="/news/2026/11/story-number-{i}">s{i}</a>' for i in range(10))
                 + '<a href="/tag/x">tag</a></body></html>')
        routes = {"https://ex.com/robots.txt": (404, "", "text/plain"),
                  "https://ex.com/news": (200, index, "text/html")}
        for i in range(10):
            routes[f"https://ex.com/news/2026/11/story-number-{i}"] = (
                200, ARTICLE_HTML.replace("Nvidia", f"Nvidia {i}"), "text/html")
        seeds = [Seed(url="https://ex.com/news", topics=["chips-export-controls"], max_items=10)]
        result, db, session = _run_crawl(tmp_path, monkeypatch, routes, seeds, max_pages=4)
        assert result["counters"]["fetched"] == 4  # seed + 3 articles
        assert "https://ex.com/tag/x" not in session.calls
        assert db.stats()["pages"] == 4

    def test_plan_is_offline(self):
        from radar.crawler import CrawlParams, Seed, plan
        p = plan([Seed(url="https://a.com/feed", kind="feed", max_items=5)], CrawlParams(), "x.db")
        assert p["dry_run"] and p["domains"] == ["a.com"] and p["max_possible_fetches"] == 6
        assert p["url_log"]["table"] == "run_urls"
        assert p["url_log"]["seed_urls"] == ["https://a.com/feed"]
        assert "not_fetched" in p["url_log"]["statuses"]


def test_default_seeds_file_loads():
    from radar.crawler import load_seeds
    seeds = load_seeds()
    assert len(seeds) >= 20
    assert all(s.kind in ("html", "feed") for s in seeds)
    war = load_seeds(topics=["war-ukraine-russia"])
    assert any("kyivindependent" in s.url for s in war)


class TestWarDrones:
    @pytest.mark.parametrize("title", [
        "Russia launches 400 Shahed drones at Kyiv overnight",
        "Ukraine sea drones hit Russian tanker",
        "Army buys interceptor drones to counter FPV threat",
        "Pentagon tests drone swarm autonomy on the battlefield",
        "New C-UAS system fielded",
        "Loitering munitions in high demand",
        "Geran-2 attacks surge",
        "Navy orders more USVs",
        "Suspected Drones Over RAF Lakenheath Just Triggered Security Response: U.S. Air Force",
    ])
    def test_headlines_tagged(self, title):
        assert "war-drones" in tag_text(title)

    @pytest.mark.parametrize("title", [
        "Amazon expands drone delivery to Texas",
        "Best camera drones for 2026",
    ])
    def test_civilian_drones_not_tagged(self, title):
        assert "war-drones" not in tag_text(title)


class TestFeedOnlyAndCrawlDelay:
    def test_feed_only_seed_never_fetches_articles(self, tmp_path, monkeypatch):
        from radar.crawler import Seed
        routes = {
            "https://news.example.org/robots.txt": (200, "User-agent: *\nCrawl-delay: 100\n", "text/plain"),
            "https://news.example.org/rss": (200, RSS, "application/rss+xml"),
        }
        seeds = [Seed(url="https://news.example.org/rss", kind="feed", require_match=True, feed_only=True)]
        result, db, session = _run_crawl(tmp_path, monkeypatch, routes, seeds)
        assert session.calls.count("https://news.example.org/iran-bahrain-radar") == 0
        assert result["counters"]["feed_only"] == 1 and result["counters"]["stored"] == 1
        assert result["status"] == "completed"
        assert db.get_page(url="https://news.example.org/iran-bahrain-radar")["text_md"] is None

    def test_long_crawl_delay_limits_to_one_request(self, tmp_path, monkeypatch):
        from radar.crawler import Seed
        routes = {
            "https://news.example.org/robots.txt": (200, "User-agent: *\nCrawl-delay: 100\n", "text/plain"),
            "https://news.example.org/rss": (200, RSS, "application/rss+xml"),
            "https://news.example.org/iran-bahrain-radar": (200, ARTICLE_HTML, "text/html"),
        }
        seeds = [Seed(url="https://news.example.org/rss", kind="feed", require_match=True)]
        result, db, session = _run_crawl(tmp_path, monkeypatch, routes, seeds)
        assert "https://news.example.org/iran-bahrain-radar" not in session.calls
        assert any("crawl-delay" in s["reason"] for s in result["skips"])
        page = db.get_page(url="https://news.example.org/iran-bahrain-radar")
        assert page["title"] == "Iran says it struck radar in Bahrain" and page["text_md"] is None


def _row(db, url, run_id=None):
    rows = [r for r in db.query_urls(run_id=run_id, limit=500) if r["url"] == url]
    assert len(rows) == 1, (url, rows)
    return rows[0]


class _RaisingSession(_FakeSession):
    def get(self, url, **kw):
        self.calls.append(url)
        if url.endswith("/robots.txt"):
            status, body, ctype = self.routes.get(url, (404, "", "text/plain"))
            return _FakeResp(status, body, ctype)
        raise RuntimeError("connection reset")


_INDEX = (
    "<html><head><title>News</title></head><body>"
    '<a href="/news/2026/11/first-story-slug">one</a>'
    '<a href="/news/2026/11/second-story-slug">two</a>'
    '<a href="/tag/x">tag</a>'
    "</body></html>"
)

_STORY = (
    "<html><head><title>Iran says it struck radar in Bahrain</title></head><body>"
    "<p>Iran says it struck radar in Bahrain. This paragraph is long enough to extract.</p>"
    '<a href="/news/2026/11/deeper-story-slug">next</a>'
    "</body></html>"
)

_TWO_MATCH_RSS = """<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
<item><title>Iran says it struck radar in Bahrain</title>
<link>https://news.example.org/iran-one</link>
<pubDate>Fri, 13 Nov 2026 10:00:00 GMT</pubDate></item>
<item><title>Iran missile strike hits Bahrain</title>
<link>https://news.example.org/iran-two</link>
<pubDate>Fri, 13 Nov 2026 11:00:00 GMT</pubDate></item>
<item><title>Football results</title><link>https://news.example.org/football</link>
<pubDate>Fri, 13 Nov 2026 12:00:00 GMT</pubDate></item>
<item><title>Old Iran Bahrain note</title><link>https://news.example.org/old-iran</link>
<pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate></item>
</channel></rss>"""

_DELAY_RSS = """<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
<item><title></title><link>https://news.example.org/untitled-story</link>
<pubDate>Fri, 13 Nov 2026 12:00:00 GMT</pubDate></item>
<item><title>Iran says it struck radar in Bahrain</title>
<link>https://news.example.org/titled-story</link>
<pubDate>Fri, 13 Nov 2026 10:00:00 GMT</pubDate></item>
</channel></rss>"""


class TestUrlLog:
    """Every encountered URL is a run_urls row. One test per status, no network."""

    def test_stored_and_duplicate(self, tmp_path, monkeypatch):
        from radar.crawler import Seed
        same = (
            "<html><head><title>Same story</title></head><body><p>Identical body.</p>"
            '<a href="/news/2026/11/deeper-story-slug">next</a></body></html>'
        )
        routes = {
            "https://ex.com/robots.txt": (404, "", "text/plain"),
            "https://ex.com/news": (200, _INDEX, "text/html"),
            "https://ex.com/news/2026/11/first-story-slug": (200, same, "text/html"),
            "https://ex.com/news/2026/11/second-story-slug": (200, same, "text/html"),
        }
        seeds = [Seed(url="https://ex.com/news", max_items=5)]
        result, db, _session = _run_crawl(tmp_path, monkeypatch, routes, seeds, max_pages=10)
        assert _row(db, "https://ex.com/news")["status"] == "stored"
        assert _row(db, "https://ex.com/news")["depth"] == 0
        assert _row(db, "https://ex.com/news")["discovered_from"] is None
        first = _row(db, "https://ex.com/news/2026/11/first-story-slug")
        second = _row(db, "https://ex.com/news/2026/11/second-story-slug")
        assert first["status"] == "stored" and first["http_status"] == 200 and first["fetched_at"]
        assert first["discovered_from"] == "https://ex.com/news" and first["depth"] == 1
        assert second["status"] == "duplicate" and second["reason"] == "duplicate_content"
        assert db.get_page(url=second["url"])["text_md"] is None
        assert db.get_page(url=first["url"])["text_md"]
        # The tag link is not a candidate, so it is neither fetched nor logged.
        assert "https://ex.com/tag/x" not in _session.calls
        assert not any(r["url"].endswith("/tag/x") for r in db.query_urls(limit=50))
        assert not any(r["url"].endswith("/robots.txt") for r in db.query_urls(limit=50))
        assert result["urls_logged"] == db.count_urls(result["run_id"])
        # Depth limit: the story links onward and that URL is not requested.
        deeper = _row(db, "https://ex.com/news/2026/11/deeper-story-slug")
        assert deeper["status"] == "not_fetched" and deeper["reason"] == "depth"
        assert deeper["depth"] == 2 and deeper["fetched_at"] is None
        assert deeper["url"] not in _session.calls
        assert db.get_page(url=deeper["url"]) is None
        assert db.query_pages(q="Identical")  # FTS still indexes stored bodies

    def test_failed_http_and_transport(self, tmp_path, monkeypatch):
        from radar.crawler import Seed
        routes = {
            "https://ex.com/robots.txt": (404, "", "text/plain"),
            "https://ex.com/news": (200, _INDEX.replace("second-story-slug", "missing-story-slug"), "text/html"),
            "https://ex.com/news/2026/11/first-story-slug": (200, _STORY, "text/html"),
            "https://ex.com/news/2026/11/missing-story-slug": (500, "down", "text/html"),
        }
        result, db, _session = _run_crawl(
            tmp_path, monkeypatch, routes, [Seed(url="https://ex.com/news")], max_pages=10,
        )
        failed = _row(db, "https://ex.com/news/2026/11/missing-story-slug")
        assert failed["status"] == "failed" and failed["reason"] == "HTTP 500"
        assert failed["http_status"] == 500 and failed["fetched_at"]
        assert db.get_page(url=failed["url"]) is None
        assert any(e["url"] == failed["url"] and e["error"] == "HTTP 500" for e in result["errors"])
        assert result["status"] == "partial"
        db.close()

        import aiohttp
        from radar.crawler import CrawlParams, RadarCrawler
        session = _RaisingSession({})
        monkeypatch.setattr(aiohttp, "ClientSession", lambda *a, **k: session)
        monkeypatch.setattr("radar.crawler.DomainThrottle.wait", lambda self, url, cd=None: asyncio.sleep(0))
        db2 = RadarDB(tmp_path / "boom.db")
        boom = asyncio.run(RadarCrawler(db2, CrawlParams(delay=0.5, recent_days=None)).run(
            [Seed(url="https://down.example/news")], {"test": True}))
        row = _row(db2, "https://down.example/news")
        assert row["status"] == "failed" and "connection reset" in row["reason"]
        assert row["http_status"] is None and row["fetched_at"]
        assert db2.get_page(url=row["url"]) is None
        assert boom["errors"] and boom["errors"][0]["url"] == row["url"]
        assert all(not r["url"].endswith("/robots.txt") for r in db2.query_urls(limit=20))

    def test_skipped_robots_content_type_and_crawl_delay(self, tmp_path, monkeypatch):
        from radar.crawler import Seed
        routes = {
            "https://blocked.com/robots.txt": (200, "User-agent: *\nDisallow: /\n", "text/plain"),
            "https://down.com/robots.txt": (503, "", "text/plain"),
        }
        result, db, session = _run_crawl(
            tmp_path, monkeypatch, routes,
            [Seed(url="https://blocked.com/news"), Seed(url="https://down.com/news")],
        )
        for url in ("https://blocked.com/news", "https://down.com/news"):
            row = _row(db, url)
            assert row["status"] == "skipped" and row["reason"] == "robots.txt"
            assert row["fetched_at"] is None and row["http_status"] is None
            assert db.get_page(url=url) is None
            assert url not in session.calls
        assert result["counters"]["skipped"] == 2
        assert {s["url"] for s in result["skips"]} == {
            "https://blocked.com/news", "https://down.com/news"}
        db.close()

        pdf_index = (
            "<html><head><title>News</title></head><body>"
            '<a href="/news/2026/11/real-story-here">a</a>'
            '<a href="/news/2026/11/file-download">b</a>'
            "</body></html>"
        )
        routes = {
            "https://ex.com/robots.txt": (404, "", "text/plain"),
            "https://ex.com/news": (200, pdf_index, "text/html"),
            "https://ex.com/news/2026/11/real-story-here": (200, _STORY, "text/html"),
            "https://ex.com/news/2026/11/file-download": (200, "%PDF-1.4", "application/pdf"),
        }
        result, db, session = _run_crawl(
            tmp_path, monkeypatch, routes, [Seed(url="https://ex.com/news")], max_pages=10,
        )
        pdf = _row(db, "https://ex.com/news/2026/11/file-download")
        assert pdf["status"] == "skipped" and pdf["reason"] == "content-type application/pdf"
        assert pdf["http_status"] == 200 and pdf["fetched_at"]
        assert db.get_page(url=pdf["url"]) is None
        assert _row(db, "https://ex.com/news/2026/11/real-story-here")["status"] == "stored"
        db.close()

        routes = {
            "https://news.example.org/robots.txt": (200, "User-agent: *\nCrawl-delay: 100\n", "text/plain"),
            "https://news.example.org/rss": (200, _DELAY_RSS, "application/rss+xml"),
            "https://news.example.org/untitled-story": (200, _STORY, "text/html"),
            "https://news.example.org/titled-story": (200, _STORY, "text/html"),
        }
        result, db, session = _run_crawl(
            tmp_path, monkeypatch, routes,
            [Seed(url="https://news.example.org/rss", kind="feed")],
        )
        assert "https://news.example.org/untitled-story" not in session.calls
        assert "https://news.example.org/titled-story" not in session.calls
        untitled = _row(db, "https://news.example.org/untitled-story")
        titled = _row(db, "https://news.example.org/titled-story")
        assert untitled["status"] == "skipped" and untitled["reason"] == "crawl-delay 100s"
        assert untitled["fetched_at"] is None
        assert db.get_page(url=untitled["url"]) is None
        # A titled entry is still saved from the feed, so the final status is feed_only.
        assert titled["status"] == "feed_only" and titled["reason"] == "crawl-delay 100s"
        assert any(s["reason"] == "crawl-delay 100s" for s in result["skips"])
        assert _row(db, "https://news.example.org/rss")["status"] == "fetched"

    def test_feed_only_http_error_and_seed_flag(self, tmp_path, monkeypatch):
        from radar.crawler import Seed
        routes = {
            "https://news.example.org/robots.txt": (404, "", "text/plain"),
            "https://news.example.org/rss": (200, RSS, "application/rss+xml"),
            "https://news.example.org/iran-bahrain-radar": (403, "denied", "text/html"),
        }
        result, db, _session = _run_crawl(
            tmp_path, monkeypatch, routes,
            [Seed(url="https://news.example.org/rss", kind="feed", require_match=True)],
        )
        row = _row(db, "https://news.example.org/iran-bahrain-radar")
        assert row["status"] == "feed_only" and row["reason"] == "HTTP 403"
        assert row["http_status"] == 403 and row["fetched_at"]
        assert row["source_seed"] == "https://news.example.org/rss"
        assert row["discovered_from"] == "https://news.example.org/rss"
        page = db.get_page(url=row["url"])
        assert page["text_md"] is None and page["title"] == "Iran says it struck radar in Bahrain"
        football = _row(db, "https://news.example.org/football")
        assert football["status"] == "not_fetched" and football["reason"] == "require_match"
        assert football["url"] not in _session.calls
        feed = _row(db, "https://news.example.org/rss")
        assert feed["status"] == "fetched" and feed["reason"] == "feed" and feed["http_status"] == 200
        assert db.get_page(url=feed["url"]) is None
        assert any(e["error"] == "HTTP 403" for e in result["errors"])
        assert result["counters"]["failed"] == 1 and result["counters"]["feed_only"] == 1
        db.close()

        routes = {
            "https://news.example.org/robots.txt": (200, "User-agent: *\nAllow: /\n", "text/plain"),
            "https://news.example.org/rss": (200, RSS, "application/rss+xml"),
        }
        result, db, session = _run_crawl(
            tmp_path, monkeypatch, routes,
            [Seed(url="https://news.example.org/rss", kind="feed", require_match=True, feed_only=True)],
        )
        row = _row(db, "https://news.example.org/iran-bahrain-radar", result["run_id"])
        assert row["status"] == "feed_only" and row["reason"] == "feed_only"
        assert row["fetched_at"] is None and row["http_status"] is None
        assert session.calls.count(row["url"]) == 0
        assert result["counters"]["feed_only"] == 1

    def test_not_fetched_budget_max_items_and_link_cap(self, tmp_path, monkeypatch):
        from radar.crawler import Seed
        routes = {"https://ex.com/robots.txt": (404, "", "text/plain"),
                  "https://ex.com/news": (200, _INDEX, "text/html")}
        for slug in ("first-story-slug", "second-story-slug"):
            routes[f"https://ex.com/news/2026/11/{slug}"] = (200, _STORY, "text/html")
        result, db, session = _run_crawl(
            tmp_path, monkeypatch, routes, [Seed(url="https://ex.com/news", max_items=2)],
            max_pages=2,
        )
        assert result["counters"]["fetched"] == 2  # index + one story
        pending = [r for r in db.query_urls(status="not_fetched", limit=20)
                   if r["reason"] == "budget"]
        assert len(pending) == 1
        assert pending[0]["url"] not in session.calls
        assert pending[0]["fetched_at"] is None
        assert db.get_page(url=pending[0]["url"]) is None
        db.close()

        routes = {
            "https://news.example.org/robots.txt": (404, "", "text/plain"),
            "https://news.example.org/rss": (200, _TWO_MATCH_RSS, "application/rss+xml"),
            "https://news.example.org/iran-one": (
                200, _STORY.replace("Bahrain", "Bahrain today"), "text/html"),
            "https://news.example.org/iran-two": (200, _STORY, "text/html"),
        }
        result, db, session = _run_crawl(
            tmp_path, monkeypatch, routes,
            [Seed(url="https://news.example.org/rss", kind="feed", require_match=True, max_items=1)],
            recent_days=7, max_pages=10,
        )
        assert _row(db, "https://news.example.org/iran-one")["status"] == "stored"
        capped = _row(db, "https://news.example.org/iran-two")
        assert capped["status"] == "not_fetched" and capped["reason"] == "max_items"
        assert capped["url"] not in session.calls
        assert _row(db, "https://news.example.org/football")["reason"] == "require_match"
        old = _row(db, "https://news.example.org/old-iran")
        assert old["status"] == "not_fetched" and old["reason"] == "recent_days"
        assert old["url"] not in session.calls
        assert db.get_page(url=old["url"]) is None
        db.close()

        hub = (
            "<html><head><title>Hub story title</title></head><body><p>Hub body.</p>"
            + "".join(f'<a href="/news/2026/11/child-story-{n}">c</a>' for n in ("aaa", "bbb", "ccc", "ddd"))
            + "</body></html>"
        )
        index = (
            '<html><head><title>News</title></head><body>'
            '<a href="/news/2026/11/hub-story-here">hub</a></body></html>'
        )
        routes = {
            "https://ex.com/robots.txt": (404, "", "text/plain"),
            "https://ex.com/news": (200, index, "text/html"),
            "https://ex.com/news/2026/11/hub-story-here": (200, hub, "text/html"),
        }
        for n in ("aaa", "bbb", "ccc", "ddd"):
            routes[f"https://ex.com/news/2026/11/child-story-{n}"] = (
                200, "<html><head><title>Child</title></head><body><p>Child body.</p></body></html>",
                "text/html",
            )
        _result, db, session = _run_crawl(
            tmp_path, monkeypatch, routes,
            [Seed(url="https://ex.com/news", max_items=3)],
            max_pages=20, max_depth=2,
        )
        capped = [
            r for r in db.query_urls(run_id=_result["run_id"], status="not_fetched", limit=20)
            if r["reason"] == "link_cap"
        ]
        assert {r["url"] for r in capped} == {
            "https://ex.com/news/2026/11/child-story-ccc",
            "https://ex.com/news/2026/11/child-story-ddd",
        }
        for row in capped:
            assert row["url"] not in session.calls
            assert row["discovered_from"] == "https://ex.com/news/2026/11/hub-story-here"
            assert row["depth"] == 2
            assert db.get_page(url=row["url"]) is None
        assert "https://ex.com/news/2026/11/child-story-aaa" in session.calls
        assert "https://ex.com/news/2026/11/child-story-bbb" in session.calls

    def test_second_run_appends_rows_for_the_same_url(self, tmp_path, monkeypatch):
        from radar.crawler import Seed
        routes = {
            "https://ex.com/robots.txt": (404, "", "text/plain"),
            "https://ex.com/news": (200, "<html><head><title>News</title></head><body>Hi</body></html>",
                                    "text/html"),
        }
        seeds = [Seed(url="https://ex.com/news")]
        first, db, _session = _run_crawl(tmp_path, monkeypatch, routes, seeds, max_pages=2)
        db.close()
        second, db, _session = _run_crawl(tmp_path, monkeypatch, routes, seeds, max_pages=2)
        rows = db.query_urls(limit=20)
        assert {r["run_id"] for r in rows} == {first["run_id"], second["run_id"]}
        assert all(r["status"] == "stored" and r["url"] == "https://ex.com/news" for r in rows)
        assert db.get_page(url="https://ex.com/news")["times_seen"] == 2
        assert db.query_pages(q="News")


# Real-world shape (militarnyi.com, 2026-11-15): Crawl-delay and a Disallow come after a
# blank line inside the "User-agent: *" group. urllib.robotparser used to drop both.
_BLANK_LINE_ROBOTS = (
    "User-agent: *\nDisallow: /wp-admin/\nAllow: /*.css$\n\nCrawl-delay: 100\n\n"
    "#WP Import Export Rule\nDisallow: /wp-content/uploads/private/\n\n"
    "Sitemap: https://news.example.org/sitemap.xml\n"
)


class TestRobotsBlankLines:
    UA = "ai-crawler-radar/1.0 (+https://github.com/hectorsvill/ai-crawler)"

    def _rp(self, text):
        from urllib.robotparser import RobotFileParser
        from radar.crawler import robots_lines
        rp = RobotFileParser()
        rp.parse(robots_lines(text))
        return rp

    def test_rules_after_blank_line_stay_in_group(self):
        rp = self._rp(_BLANK_LINE_ROBOTS)
        assert rp.crawl_delay(self.UA) == 100
        assert not rp.can_fetch(self.UA, "https://news.example.org/wp-content/uploads/private/a")
        assert not rp.can_fetch(self.UA, "https://news.example.org/wp-admin/x")
        assert rp.can_fetch(self.UA, "https://news.example.org/news/story")

    def test_groups_still_split_on_user_agent(self):
        rp = self._rp("User-agent: badbot\nDisallow: /\n\nUser-agent: *\nDisallow: /private\n")
        assert rp.can_fetch(self.UA, "https://x.org/public")
        assert not rp.can_fetch(self.UA, "https://x.org/private/a")
        assert not rp.can_fetch("badbot", "https://x.org/public")

    def test_crawler_honours_crawl_delay_after_blank_line(self, tmp_path, monkeypatch):
        from radar.crawler import Seed
        routes = {
            "https://news.example.org/robots.txt": (200, _BLANK_LINE_ROBOTS, "text/plain"),
            "https://news.example.org/rss": (200, RSS, "application/rss+xml"),
            "https://news.example.org/iran-bahrain-radar": (200, ARTICLE_HTML, "text/html"),
        }
        seeds = [Seed(url="https://news.example.org/rss", kind="feed", require_match=True)]
        result, db, session = _run_crawl(tmp_path, monkeypatch, routes, seeds)
        assert "https://news.example.org/iran-bahrain-radar" not in session.calls
        row = _row(db, "https://news.example.org/iran-bahrain-radar", result["run_id"])
        assert row["status"] == "feed_only" and row["reason"].startswith("crawl-delay")
        counts = db.url_counts(result["run_id"])
        assert counts["ok"], counts["reconciliation"]
        assert counts["reconciliation"]["pages_skipped"]["url_log"] == 1


class TestUrlCounts:
    def test_counts_reconcile_with_run_counters(self, tmp_path, monkeypatch):
        from radar.crawler import Seed
        routes = {
            "https://news.example.org/robots.txt": (404, "", "text/plain"),
            "https://news.example.org/rss": (200, _TWO_MATCH_RSS, "application/rss+xml"),
            "https://news.example.org/iran-one": (200, ARTICLE_HTML, "text/html"),
            "https://news.example.org/iran-two": (403, "no", "text/html"),
        }
        seeds = [Seed(url="https://news.example.org/rss", kind="feed", require_match=True)]
        result, db, session = _run_crawl(tmp_path, monkeypatch, routes, seeds, recent_days=30)
        c = db.url_counts(result["run_id"])
        assert c["ok"], c["reconciliation"]
        assert c["by_status"]["stored"] == 1 and c["by_status"]["feed_only"] == 1
        assert c["by_status"]["fetched"] == 1
        reasons = {r["reason"] for r in c["by_reason"] if r["status"] == "not_fetched"}
        assert {"recent_days", "require_match"} <= reasons
        assert c["reconciliation"]["pages_failed"] == {"run": 1, "url_log": 1, "match": True}
        assert c["total"] == sum(c["by_status"].values())
        assert {"status": "feed_only", "reason": "HTTP 403", "count": 1} in c["by_reason"]
