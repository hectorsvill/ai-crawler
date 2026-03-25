"""Tests for storage helpers — content hash and DB operations."""

import pytest
import pytest_asyncio

from storage.db import compute_content_hash


class TestComputeContentHash:
    def test_deterministic(self):
        h1 = compute_content_hash("hello world")
        h2 = compute_content_hash("hello world")
        assert h1 == h2

    def test_different_content_different_hash(self):
        assert compute_content_hash("abc") != compute_content_hash("xyz")

    def test_empty_string(self):
        h = compute_content_hash("")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest

    def test_returns_hex_string(self):
        h = compute_content_hash("test content")
        assert all(c in "0123456789abcdef" for c in h)

    def test_unicode_content(self):
        # Should not raise on non-ASCII content
        h = compute_content_hash("日本語テスト")
        assert len(h) == 64
