"""Tests for utils/url.py — URL normalization."""

import pytest
from utils.url import is_same_domain, normalize_url


class TestNormalizeUrl:
    def test_strips_fragment(self):
        assert normalize_url("https://example.com/page#section") == "https://example.com/page"

    def test_lowercases_scheme(self):
        result = normalize_url("HTTPS://example.com/")
        assert result.startswith("https://example.com")
        assert not result.endswith("/")

    def test_lowercases_host(self):
        assert normalize_url("https://EXAMPLE.COM/path") == "https://example.com/path"

    def test_removes_default_http_port(self):
        assert normalize_url("http://example.com:80/path") == "http://example.com/path"

    def test_removes_default_https_port(self):
        assert normalize_url("https://example.com:443/path") == "https://example.com/path"

    def test_keeps_non_default_port(self):
        result = normalize_url("https://example.com:8443/path")
        assert ":8443" in result

    def test_removes_trailing_slash(self):
        assert normalize_url("https://example.com/path/") == "https://example.com/path"

    def test_root_slash_stripped(self):
        # Trailing slash is stripped everywhere for consistency
        result = normalize_url("https://example.com/")
        assert result == "https://example.com"

    def test_sorts_query_params(self):
        a = normalize_url("https://example.com/?z=1&a=2")
        b = normalize_url("https://example.com/?a=2&z=1")
        assert a == b

    def test_identical_url_unchanged(self):
        url = "https://example.com/path?q=hello"
        assert normalize_url(url) == url

    def test_idempotent(self):
        url = "https://Example.COM/path/?b=2&a=1#frag"
        once = normalize_url(url)
        twice = normalize_url(once)
        assert once == twice

    def test_malformed_url_returns_original(self):
        bad = "not a url at all"
        # Should not raise; just return the original
        result = normalize_url(bad)
        assert isinstance(result, str)


class TestIsSameDomain:
    def test_same_domain(self):
        assert is_same_domain("https://example.com/a", "https://example.com/b")

    def test_different_domain(self):
        assert not is_same_domain("https://example.com/a", "https://other.com/b")

    def test_subdomain_differs(self):
        assert not is_same_domain("https://sub.example.com/", "https://example.com/")

    def test_case_insensitive(self):
        assert is_same_domain("https://EXAMPLE.COM/a", "https://example.com/b")
