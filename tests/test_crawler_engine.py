"""Tests for crawler/engine.py — HTML parsing, link extraction, markdown conversion."""

import pytest
from crawler.engine import extract_links, extract_title, html_to_markdown


BASE_URL = "https://example.com"

SIMPLE_HTML = """
<html>
<head><title>Test Page</title></head>
<body>
  <p>Hello world. This is a test page.</p>
  <a href="/about">About</a>
  <a href="https://external.com/page">External</a>
  <a href="mailto:admin@example.com">Email</a>
  <a href="#section">Fragment</a>
  <a href="javascript:void(0)">JS link</a>
  <a href="/about">About (duplicate)</a>
</body>
</html>
"""


class TestExtractLinks:
    def test_extracts_absolute_links(self):
        links = extract_links(SIMPLE_HTML, BASE_URL)
        assert "https://example.com/about" in links

    def test_extracts_external_links(self):
        links = extract_links(SIMPLE_HTML, BASE_URL)
        assert "https://external.com/page" in links

    def test_skips_mailto(self):
        links = extract_links(SIMPLE_HTML, BASE_URL)
        assert not any("mailto:" in link for link in links)

    def test_skips_fragment_only(self):
        links = extract_links(SIMPLE_HTML, BASE_URL)
        # Fragment-only hrefs should be dropped
        assert not any(link.endswith("#section") and link.count("/") < 3 for link in links)

    def test_skips_javascript(self):
        links = extract_links(SIMPLE_HTML, BASE_URL)
        assert not any("javascript:" in link for link in links)

    def test_deduplicates(self):
        links = extract_links(SIMPLE_HTML, BASE_URL)
        assert links.count("https://example.com/about") == 1

    def test_empty_html(self):
        assert extract_links("", BASE_URL) == []

    def test_relative_links_resolved(self):
        html = '<html><body><a href="../parent">Up</a></body></html>'
        links = extract_links(html, "https://example.com/sub/page")
        assert "https://example.com/parent" in links


class TestExtractTitle:
    def test_finds_title(self):
        assert extract_title(SIMPLE_HTML) == "Test Page"

    def test_no_title_returns_none(self):
        html = "<html><body><p>No title here</p></body></html>"
        assert extract_title(html) is None

    def test_empty_html(self):
        assert extract_title("") is None


class TestHtmlToMarkdown:
    def test_returns_non_empty_string(self):
        result = html_to_markdown(SIMPLE_HTML, BASE_URL)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_page_text(self):
        result = html_to_markdown(SIMPLE_HTML, BASE_URL)
        assert "Hello world" in result

    def test_strips_script_tags(self):
        html = "<html><head><script>alert('xss')</script></head><body>Content</body></html>"
        result = html_to_markdown(html, BASE_URL)
        assert "alert" not in result
        assert "Content" in result

    def test_strips_style_tags(self):
        html = "<html><head><style>body{color:red}</style></head><body>Visible</body></html>"
        result = html_to_markdown(html, BASE_URL)
        assert "color:red" not in result

    def test_collapses_excessive_newlines(self):
        html = "<html><body><p>A</p>\n\n\n\n<p>B</p></body></html>"
        result = html_to_markdown(html, BASE_URL)
        # Should not have more than 2 consecutive newlines
        assert "\n\n\n" not in result

    def test_includes_title_as_heading(self):
        result = html_to_markdown(SIMPLE_HTML, BASE_URL)
        # Title should appear prominently (BeautifulSoup path prepends # title)
        assert "Test Page" in result
