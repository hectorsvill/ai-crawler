"""
crawler/engine.py — Async crawl engine.

Two fetch modes:
  - Lightweight: asyncio + aiohttp for static HTML pages.
  - JS-heavy:    Playwright async API for SPAs and dynamic content.

Auto-detection: if static fetch yields very little content (< 500 chars of
markdown), the engine retries with Playwright.

Optional crawl4ai integration for enhanced markdown extraction — imported
conditionally; the engine works without it.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

try:
    from crawl4ai import AsyncWebCrawler
    HAS_CRAWL4AI = True
except ImportError:
    HAS_CRAWL4AI = False

try:
    from playwright.async_api import async_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

from storage.db import compute_content_hash
from storage.models import PageContent

logger = logging.getLogger(__name__)

# Minimum meaningful markdown length before triggering Playwright retry
STATIC_CONTENT_THRESHOLD = 500

# Timeout for individual page fetches (seconds)
FETCH_TIMEOUT = 30


# ── HTML → Markdown conversion ─────────────────────────────────────────────────

def html_to_markdown(html: str, url: str = "") -> str:
    """
    Convert raw HTML to clean markdown text.

    Preference order:
    1. trafilatura (best quality)
    2. BeautifulSoup text extraction (fallback)
    """
    if HAS_TRAFILATURA:
        result = trafilatura.extract(
            html,
            include_links=True,
            include_tables=True,
            output_format="markdown",
            url=url,
        )
        if result and len(result.strip()) > 100:
            return result

    # BeautifulSoup fallback
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # Extract title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)

    if title:
        return f"# {title}\n\n{text}"
    return text


def extract_links(html: str, base_url: str) -> list[str]:
    """Extract all absolute href links from an HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    base_domain = urlparse(base_url).netloc

    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        # Only http/https
        if parsed.scheme not in ("http", "https"):
            continue
        links.append(absolute)

    return list(dict.fromkeys(links))  # deduplicate, preserve order


def extract_title(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find("title")
    return tag.get_text(strip=True) if tag else None


# ── Lightweight aiohttp fetcher ────────────────────────────────────────────────

async def fetch_static(
    url: str,
    user_agent: str,
    timeout: int = FETCH_TIMEOUT,
) -> tuple[str, int]:
    """
    Fetch a URL with aiohttp.
    Returns (html_content, status_code).
    Raises on non-2xx responses.
    """
    headers = {"User-Agent": user_agent}
    async with aiohttp.ClientSession() as session:
        async with session.get(
            url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
            allow_redirects=True,
            max_redirects=5,
        ) as resp:
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "html" not in content_type and "text" not in content_type:
                raise ValueError(f"Non-HTML content type: {content_type}")
            html = await resp.text(errors="replace")
            return html, resp.status


# ── Playwright JS fetcher ─────────────────────────────────────────────────────

async def fetch_with_playwright(
    url: str,
    user_agent: str,
    timeout: int = FETCH_TIMEOUT,
) -> str:
    """
    Render a page with Playwright (Chromium) and return the final HTML.
    Waits for network idle to ensure JS has run.
    """
    if not HAS_PLAYWRIGHT:
        raise RuntimeError(
            "Playwright is not installed. Run: pip install playwright && playwright install chromium"
        )

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=user_agent)
        page = await context.new_page()
        try:
            await page.goto(url, timeout=timeout * 1000, wait_until="networkidle")
            html = await page.content()
        finally:
            await browser.close()
    return html


# ── crawl4ai integration ───────────────────────────────────────────────────────

async def fetch_with_crawl4ai(url: str) -> str | None:
    """Use crawl4ai for enhanced markdown extraction. Returns None on failure."""
    if not HAS_CRAWL4AI:
        return None
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            if result and result.markdown:
                return result.markdown
    except Exception as exc:
        logger.debug("crawl4ai failed for %s: %s", url, exc)
    return None


# ── Main engine function ───────────────────────────────────────────────────────

async def fetch_page(
    url: str,
    user_agent: str,
    force_playwright: bool = False,
    timeout: int = FETCH_TIMEOUT,
) -> PageContent:
    """
    Fetch a page and return a PageContent object.

    Auto-detection logic:
    1. Try crawl4ai first (if available and not force_playwright)
    2. Try static aiohttp fetch
    3. If static yields < STATIC_CONTENT_THRESHOLD chars → retry with Playwright
    4. If force_playwright=True → skip straight to Playwright
    """
    raw_html: str | None = None
    markdown: str = ""
    fetch_method = "aiohttp"

    # crawl4ai path (fast markdown extraction)
    if not force_playwright and HAS_CRAWL4AI:
        try:
            crawl4ai_md = await fetch_with_crawl4ai(url)
            if crawl4ai_md and len(crawl4ai_md) >= STATIC_CONTENT_THRESHOLD:
                return PageContent(
                    url=url,
                    markdown=crawl4ai_md,
                    raw_html=None,
                    title=None,
                    content_hash=compute_content_hash(crawl4ai_md),
                    fetch_method="crawl4ai",
                    links=[],
                )
        except Exception as exc:
            logger.debug("crawl4ai path failed, falling back: %s", exc)

    if force_playwright:
        raw_html = await fetch_with_playwright(url, user_agent, timeout)
        fetch_method = "playwright"
        markdown = html_to_markdown(raw_html, url)
    else:
        try:
            raw_html, _ = await fetch_static(url, user_agent, timeout)
            markdown = html_to_markdown(raw_html, url)

            # Auto-detect JS-heavy page
            if len(markdown.strip()) < STATIC_CONTENT_THRESHOLD and HAS_PLAYWRIGHT:
                logger.info("Static content too thin for %s — retrying with Playwright", url)
                raw_html = await fetch_with_playwright(url, user_agent, timeout)
                fetch_method = "playwright"
                markdown = html_to_markdown(raw_html, url)
        except Exception as exc:
            logger.warning("Static fetch failed for %s: %s", url, exc)
            if HAS_PLAYWRIGHT:
                logger.info("Falling back to Playwright for %s", url)
                raw_html = await fetch_with_playwright(url, user_agent, timeout)
                fetch_method = "playwright"
                markdown = html_to_markdown(raw_html, url)
            else:
                raise

    links = extract_links(raw_html or "", url) if raw_html else []
    title = extract_title(raw_html) if raw_html else None
    content_hash = compute_content_hash(markdown or raw_html or url)

    return PageContent(
        url=url,
        markdown=markdown,
        raw_html=raw_html,
        title=title,
        content_hash=content_hash,
        fetch_method=fetch_method,
        links=links,
    )
