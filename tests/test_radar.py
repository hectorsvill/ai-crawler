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
    p = CrawlParams(delay=0.5, recent_days=None, **params)
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
