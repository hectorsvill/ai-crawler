"""
utils/url.py — URL normalization utilities.

Provides a canonical form for URLs before queuing/deduplication so that
semantically identical URLs (e.g., different encoding of the same path,
duplicate query params, fragments) are treated as the same resource.
"""

from __future__ import annotations

from urllib.parse import (
    ParseResult,
    parse_qsl,
    urlencode,
    urldefrag,
    urlparse,
    urlunparse,
)


def normalize_url(url: str) -> str:
    """
    Return a canonical form of *url* suitable for deduplication.

    Normalizations applied (in order):
    1. Strip the fragment (``#section``) — fragments are client-side only.
    2. Lowercase scheme and host.
    3. Remove default ports (80 for http, 443 for https).
    4. Percent-decode unreserved characters, then re-encode consistently.
    5. Sort query parameters so ``?b=2&a=1`` and ``?a=1&b=2`` compare equal.
    6. Remove trailing slash from paths longer than one char
       (``/path/`` → ``/path``), but preserve bare root ``/``.

    Returns the original URL if parsing fails.
    """
    try:
        # 1. Strip fragment
        url_no_frag, _ = urldefrag(url)

        parsed: ParseResult = urlparse(url_no_frag)

        # 2. Lowercase scheme and host
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()

        # 3. Remove default ports
        if ":" in netloc:
            host, port_str = netloc.rsplit(":", 1)
            if port_str.isdigit():
                port = int(port_str)
                if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
                    netloc = host

        # 4 & 5. Sort query params (also normalises percent-encoding via parse_qsl)
        query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))

        # 6. Normalize trailing slashes — strip from all paths for consistency
        path = parsed.path.rstrip("/") or ""

        normalized = urlunparse((scheme, netloc, path, parsed.params, query, ""))
        return normalized

    except Exception:  # pragma: no cover — malformed URLs pass through unchanged
        return url


def is_same_domain(url: str, reference: str) -> bool:
    """Return True if *url* shares the same registered domain as *reference*."""
    try:
        return urlparse(url).netloc.lower() == urlparse(reference).netloc.lower()
    except Exception:
        return False
