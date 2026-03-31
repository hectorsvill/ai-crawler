"""Tests for the --format markdown export feature.

Covers:
  - get_all_pages_markdown() DB layer
  - Markdown file output structure and content
  - Session isolation (only returns pages for the requested session)
  - Empty/no-content edge cases
  - Default filename logic
  - JSON export regression (existing behaviour unchanged)
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── helpers ───────────────────────────────────────────────────────────────────

async def _seed_db(db_path: str, session_id: str, pages: list[dict]) -> None:
    """Insert URLRecord + VisitedPage rows so get_all_pages_markdown can find them."""
    from storage.db import init_db, get_session as db_session
    from storage.models import URLRecord, VisitedPage

    await init_db(db_path)

    async with db_session() as db:
        for page in pages:
            url_record = URLRecord(
                url=page["url"],
                session_id=session_id,
                depth=page.get("depth", 0),
                status=page.get("status", "visited"),
            )
            db.add(url_record)

            visited = VisitedPage(
                url=page["url"],
                content_hash=f"hash_{page['url']}",
                markdown=page.get("markdown"),
                title=page.get("title"),
                fetch_method="aiohttp",
            )
            db.add(visited)

        await db.commit()


# ── get_all_pages_markdown ────────────────────────────────────────────────────

class TestGetAllPagesMarkdown:

    async def test_returns_pages_with_markdown(self, tmp_path):
        from storage.db import get_all_pages_markdown, init_db
        db_path = str(tmp_path / "test.db")
        session = "session-abc-001"

        await _seed_db(db_path, session, [
            {"url": "https://example.com/a", "markdown": "# Page A\nContent here.", "title": "Page A"},
            {"url": "https://example.com/b", "markdown": "# Page B\nMore content.", "title": "Page B"},
        ])

        results = await get_all_pages_markdown(session)
        assert len(results) == 2
        urls = {r["url"] for r in results}
        assert "https://example.com/a" in urls
        assert "https://example.com/b" in urls

    async def test_skips_pages_with_no_markdown(self, tmp_path):
        from storage.db import get_all_pages_markdown
        db_path = str(tmp_path / "test.db")
        session = "session-abc-002"

        await _seed_db(db_path, session, [
            {"url": "https://example.com/good", "markdown": "# Good\nHas content.", "title": "Good"},
            {"url": "https://example.com/empty", "markdown": None, "title": "Empty"},
            {"url": "https://example.com/whitespace", "markdown": "   \n  ", "title": "Whitespace"},
        ])

        results = await get_all_pages_markdown(session)
        assert len(results) == 1
        assert results[0]["url"] == "https://example.com/good"

    async def test_session_isolation(self, tmp_path):
        """Pages from a different session must not appear."""
        from storage.db import get_all_pages_markdown
        db_path = str(tmp_path / "test.db")

        await _seed_db(db_path, "session-A", [
            {"url": "https://example.com/page-a", "markdown": "# A", "title": "A"},
        ])
        await _seed_db(db_path, "session-B", [
            {"url": "https://example.com/page-b", "markdown": "# B", "title": "B"},
        ])

        results_a = await get_all_pages_markdown("session-A")
        results_b = await get_all_pages_markdown("session-B")

        assert len(results_a) == 1
        assert results_a[0]["url"] == "https://example.com/page-a"
        assert len(results_b) == 1
        assert results_b[0]["url"] == "https://example.com/page-b"

    async def test_returns_empty_list_for_unknown_session(self, tmp_path):
        from storage.db import get_all_pages_markdown, init_db
        db_path = str(tmp_path / "test.db")
        await init_db(db_path)

        results = await get_all_pages_markdown("no-such-session")
        assert results == []

    async def test_result_contains_expected_keys(self, tmp_path):
        from storage.db import get_all_pages_markdown
        db_path = str(tmp_path / "test.db")
        session = "session-abc-003"

        await _seed_db(db_path, session, [
            {"url": "https://example.com/x", "markdown": "# X", "title": "X Page"},
        ])

        results = await get_all_pages_markdown(session)
        assert len(results) == 1
        record = results[0]
        assert "url" in record
        assert "title" in record
        assert "markdown" in record
        assert "fetched_at" in record

    async def test_markdown_content_preserved(self, tmp_path):
        from storage.db import get_all_pages_markdown
        db_path = str(tmp_path / "test.db")
        session = "session-abc-004"
        original_md = "# Heading\n\nParagraph with **bold** and [link](https://x.com).\n\n- item 1\n- item 2"

        await _seed_db(db_path, session, [
            {"url": "https://example.com/rich", "markdown": original_md, "title": "Rich"},
        ])

        results = await get_all_pages_markdown(session)
        assert results[0]["markdown"] == original_md

    async def test_title_returned_correctly(self, tmp_path):
        from storage.db import get_all_pages_markdown
        db_path = str(tmp_path / "test.db")
        session = "session-abc-005"

        await _seed_db(db_path, session, [
            {"url": "https://example.com/t", "markdown": "# T", "title": "My Page Title"},
        ])

        results = await get_all_pages_markdown(session)
        assert results[0]["title"] == "My Page Title"

    async def test_missing_title_returns_empty_string(self, tmp_path):
        from storage.db import get_all_pages_markdown
        db_path = str(tmp_path / "test.db")
        session = "session-abc-006"

        await _seed_db(db_path, session, [
            {"url": "https://example.com/notitle", "markdown": "# Content", "title": None},
        ])

        results = await get_all_pages_markdown(session)
        assert results[0]["title"] == ""


# ── Markdown file output format ───────────────────────────────────────────────

class TestMarkdownOutputFormat:

    async def test_export_creates_markdown_file(self, tmp_path, monkeypatch):
        from main import _export
        db_path = str(tmp_path / "test.db")
        session = "session-fmt-001"
        out = tmp_path / "out.md"

        await _seed_db(db_path, session, [
            {"url": "https://example.com/p1", "markdown": "# Hello\nWorld.", "title": "Hello"},
        ])

        monkeypatch.setenv("CRAWLER_STORAGE__DB_PATH", db_path)
        await _export(session, out, "markdown", None)

        assert out.exists()

    async def test_markdown_contains_session_id(self, tmp_path, monkeypatch):
        from main import _export
        db_path = str(tmp_path / "test.db")
        session = "abcd1234-5678-0000-0000-000000000000"
        out = tmp_path / "out.md"

        await _seed_db(db_path, session, [
            {"url": "https://example.com/p", "markdown": "# Test\nContent.", "title": "Test"},
        ])

        monkeypatch.setenv("CRAWLER_STORAGE__DB_PATH", db_path)
        await _export(session, out, "markdown", None)

        content = out.read_text()
        assert "abcd1234" in content  # first 8 chars of session ID

    async def test_markdown_contains_page_count(self, tmp_path, monkeypatch):
        from main import _export
        db_path = str(tmp_path / "test.db")
        session = "session-fmt-002"
        out = tmp_path / "out.md"

        await _seed_db(db_path, session, [
            {"url": "https://example.com/a", "markdown": "# A", "title": "A"},
            {"url": "https://example.com/b", "markdown": "# B", "title": "B"},
            {"url": "https://example.com/c", "markdown": "# C", "title": "C"},
        ])

        monkeypatch.setenv("CRAWLER_STORAGE__DB_PATH", db_path)
        await _export(session, out, "markdown", None)

        content = out.read_text()
        assert "3 pages" in content

    async def test_each_page_has_h2_section(self, tmp_path, monkeypatch):
        from main import _export
        db_path = str(tmp_path / "test.db")
        session = "session-fmt-003"
        out = tmp_path / "out.md"

        await _seed_db(db_path, session, [
            {"url": "https://example.com/alpha", "markdown": "# Alpha\nText.", "title": "Alpha Page"},
            {"url": "https://example.com/beta",  "markdown": "# Beta\nText.",  "title": "Beta Page"},
        ])

        monkeypatch.setenv("CRAWLER_STORAGE__DB_PATH", db_path)
        await _export(session, out, "markdown", None)

        content = out.read_text()
        assert "## Alpha Page" in content
        assert "## Beta Page" in content

    async def test_each_page_has_url_metadata(self, tmp_path, monkeypatch):
        from main import _export
        db_path = str(tmp_path / "test.db")
        session = "session-fmt-004"
        out = tmp_path / "out.md"

        await _seed_db(db_path, session, [
            {"url": "https://example.com/meta", "markdown": "# Meta\nContent.", "title": "Meta"},
        ])

        monkeypatch.setenv("CRAWLER_STORAGE__DB_PATH", db_path)
        await _export(session, out, "markdown", None)

        content = out.read_text()
        assert "https://example.com/meta" in content

    async def test_page_markdown_body_included(self, tmp_path, monkeypatch):
        from main import _export
        db_path = str(tmp_path / "test.db")
        session = "session-fmt-005"
        out = tmp_path / "out.md"
        body = "Some unique RAG-ready content 12345"

        await _seed_db(db_path, session, [
            {"url": "https://example.com/body", "markdown": body, "title": "Body"},
        ])

        monkeypatch.setenv("CRAWLER_STORAGE__DB_PATH", db_path)
        await _export(session, out, "markdown", None)

        content = out.read_text()
        assert body in content

    async def test_pages_separated_by_horizontal_rule(self, tmp_path, monkeypatch):
        from main import _export
        db_path = str(tmp_path / "test.db")
        session = "session-fmt-006"
        out = tmp_path / "out.md"

        await _seed_db(db_path, session, [
            {"url": "https://example.com/x", "markdown": "# X", "title": "X"},
            {"url": "https://example.com/y", "markdown": "# Y", "title": "Y"},
        ])

        monkeypatch.setenv("CRAWLER_STORAGE__DB_PATH", db_path)
        await _export(session, out, "markdown", None)

        content = out.read_text()
        assert content.count("---") >= 2  # intro divider + at least one page divider

    async def test_title_falls_back_to_url_when_missing(self, tmp_path, monkeypatch):
        from main import _export
        db_path = str(tmp_path / "test.db")
        session = "session-fmt-007"
        out = tmp_path / "out.md"

        await _seed_db(db_path, session, [
            {"url": "https://example.com/notitle", "markdown": "# Content", "title": None},
        ])

        monkeypatch.setenv("CRAWLER_STORAGE__DB_PATH", db_path)
        await _export(session, out, "markdown", None)

        content = out.read_text()
        # URL used as the section heading when title is absent
        assert "## https://example.com/notitle" in content

    async def test_default_output_filename_is_crawled_pages_md(self, tmp_path, monkeypatch):
        """When the caller passes the default output path, it should be renamed."""
        import os
        from main import _export
        db_path = str(tmp_path / "test.db")
        session = "session-fmt-008"

        await _seed_db(db_path, session, [
            {"url": "https://example.com/d", "markdown": "# D", "title": "D"},
        ])

        monkeypatch.setenv("CRAWLER_STORAGE__DB_PATH", db_path)
        monkeypatch.chdir(tmp_path)  # so crawled_pages.md lands in tmp_path

        default_output = Path("extracted_data.json")
        await _export(session, default_output, "markdown", None)

        assert (tmp_path / "crawled_pages.md").exists()


# ── JSON export regression ────────────────────────────────────────────────────

class TestJsonExportRegression:

    async def test_json_export_still_works(self, tmp_path, monkeypatch):
        """Existing JSON export must be unaffected by the new --format flag."""
        from main import _export
        from storage.db import init_db, get_session as db_session
        from storage.models import ExtractedData, VisitedPage, URLRecord

        db_path = str(tmp_path / "test.db")
        session = "session-json-001"
        out = tmp_path / "results.json"

        await init_db(db_path)
        async with db_session() as db:
            url_record = URLRecord(url="https://x.com", session_id=session, depth=0, status="visited")
            db.add(url_record)
            visited = VisitedPage(url="https://x.com", content_hash="abc123", markdown="# X", fetch_method="aiohttp")
            db.add(visited)
            await db.flush()
            extraction = ExtractedData(
                page_id=visited.id,
                data={"name": "Widget", "price": 9.99},
                schema_used="product",
                confidence=0.9,
                session_id=session,
            )
            db.add(extraction)
            await db.commit()

        monkeypatch.setenv("CRAWLER_STORAGE__DB_PATH", db_path)
        await _export(session, out, "json", None)

        assert out.exists()
        records = json.loads(out.read_text())
        assert len(records) == 1
        assert records[0]["data"]["name"] == "Widget"
        assert records[0]["confidence"] == pytest.approx(0.9)

    async def test_json_is_default_format(self, tmp_path, monkeypatch):
        """Passing fmt='json' explicitly behaves identically to the default."""
        from main import _export
        from storage.db import init_db

        db_path = str(tmp_path / "test.db")
        await init_db(db_path)
        out = tmp_path / "out.json"
        session = "session-json-002"

        monkeypatch.setenv("CRAWLER_STORAGE__DB_PATH", db_path)
        # No extractions — should write an empty array, not raise
        await _export(session, out, "json", None)

        assert out.exists()
        assert json.loads(out.read_text()) == []
