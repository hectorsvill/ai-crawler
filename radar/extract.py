"""radar/extract.py — metadata/text extraction and RSS/Atom parsing (offline-testable)."""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:  # pragma: no cover
    HAS_TRAFILATURA = False

_TRACKING = re.compile(r"^(utm_|fbclid|gclid|mc_|ref$|ref_src$|cmpid$)", re.I)

SKIP_PATH = re.compile(
    r"/(tag|tags|category|categories|author|authors|topic|topics|search|login|signin|"
    r"signup|register|account|subscribe|newsletter|privacy|terms|cookie|careers|jobs|"
    r"about|contact|help|feed|rss|wp-json|cdn-cgi|static|assets|page/\d+)(/|$)", re.I)
SKIP_EXT = re.compile(r"\.(png|jpe?g|gif|svg|webp|ico|css|js|pdf|zip|gz|mp4|mp3|xml|json)$", re.I)


def normalize_url(url: str) -> str:
    """Strip fragments and tracking params; lowercase scheme/host."""
    p = urlparse(url.strip())
    query = "&".join(
        kv for kv in p.query.split("&") if kv and not _TRACKING.match(kv.split("=", 1)[0])
    )
    return urlunparse((p.scheme.lower(), p.netloc.lower(), p.path or "/", "", query, ""))


def domain_of(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def content_hash(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text or "").strip().encode()).hexdigest()


def to_iso(value: str | None) -> str | None:
    """Parse many date formats into ISO-8601 (UTC if tz-aware). None if unparseable."""
    if not value:
        return None
    v = value.strip()
    try:
        dt = parsedate_to_datetime(v)  # RFC 2822 (RSS pubDate)
    except (TypeError, ValueError, IndexError):
        dt = None
    if dt is None:
        try:
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            m = re.match(r"(\d{4}-\d{2}-\d{2})", v)
            if not m:
                return None
            try:
                dt = datetime.fromisoformat(m.group(1))
            except ValueError:
                return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
        return dt.replace(microsecond=0).isoformat()
    return dt.replace(microsecond=0).isoformat()


def _jsonld_dates(soup: BeautifulSoup) -> str | None:
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except (ValueError, TypeError):
            continue
        stack = [data]
        while stack:
            item = stack.pop()
            if isinstance(item, list):
                stack.extend(item)
            elif isinstance(item, dict):
                if item.get("datePublished"):
                    return str(item["datePublished"])
                stack.extend(v for v in item.values() if isinstance(v, (dict, list)))
    return None


def extract_metadata(html: str, url: str) -> dict:
    """Return title, canonical_url, description, published_at, text_md, links."""
    soup = BeautifulSoup(html, "html.parser")

    def meta(*names: str) -> str | None:
        for n in names:
            tag = soup.find("meta", attrs={"property": n}) or soup.find("meta", attrs={"name": n})
            if tag and tag.get("content"):
                return tag["content"].strip()
        return None

    title = meta("og:title", "twitter:title")
    if not title and soup.title:
        title = soup.title.get_text(strip=True)
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else None

    canonical = None
    link = soup.find("link", rel=lambda v: v and "canonical" in (v if isinstance(v, list) else [v]))
    if link and link.get("href"):
        canonical = normalize_url(urljoin(url, link["href"]))

    published = (
        meta("article:published_time", "og:article:published_time", "datePublished",
             "pubdate", "publish-date", "date", "dc.date", "DC.date.issued", "sailthru.date",
             "parsely-pub-date")
        or _jsonld_dates(soup)
    )
    if not published:
        t = soup.find("time", attrs={"datetime": True})
        if t:
            published = t["datetime"]

    text = ""
    if HAS_TRAFILATURA:
        text = trafilatura.extract(html, url=url, include_links=False, include_tables=True,
                                   output_format="markdown") or ""
        if not published:
            try:
                md = trafilatura.extract_metadata(html, default_url=url)
                published = getattr(md, "date", None) if md else None
            except Exception:
                published = None
    if not text:
        for t in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            t.decompose()
        text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n", strip=True))

    links = []
    for a in BeautifulSoup(html, "html.parser").find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        absu = urljoin(url, href)
        if urlparse(absu).scheme in ("http", "https"):
            links.append(normalize_url(absu))

    return {
        "title": title,
        "canonical_url": canonical,
        "description": meta("og:description", "description", "twitter:description"),
        "published_at": to_iso(published),
        "text_md": text,
        "links": list(dict.fromkeys(links)),
    }


def is_article_link(link: str, seed_url: str) -> bool:
    """Heuristic: same site as the seed, looks like an article (not nav/asset)."""
    lp, sp = urlparse(link), urlparse(seed_url)
    if domain_of(link) != domain_of(seed_url):
        return False
    path = lp.path.rstrip("/")
    if not path or path == sp.path.rstrip("/"):
        return False
    if SKIP_EXT.search(path) or SKIP_PATH.search(path + "/"):
        return False
    segs = [s for s in path.split("/") if s]
    last = segs[-1] if segs else ""
    # article-like: a dated path, a hyphenated slug, or a numeric id
    return bool(
        re.search(r"/20\d{2}/", path + "/")
        or last.count("-") >= 2
        or re.fullmatch(r"\d{4,}", last)
        or (len(segs) >= 2 and len(last) > 12)
    )


def parse_feed(xml_text: str, base_url: str) -> list[dict]:
    """Parse RSS 2.0 / Atom into [{url, title, published_at, description}]."""
    try:
        root = ET.fromstring(xml_text.encode() if isinstance(xml_text, str) else xml_text)
    except ET.ParseError:
        return []

    def local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    items = []
    for el in root.iter():
        name = local(el.tag)
        if name not in ("item", "entry"):
            continue
        fields = {local(c.tag): c for c in el}
        link = None
        if "link" in fields:
            lk = fields["link"]
            link = (lk.text or "").strip() or lk.get("href")
            if not link:  # Atom may have several <link>; prefer rel=alternate
                for c in el:
                    if local(c.tag) == "link" and c.get("rel", "alternate") == "alternate":
                        link = c.get("href")
                        break
        if not link and "guid" in fields and (fields["guid"].text or "").startswith("http"):
            link = fields["guid"].text.strip()
        if not link:
            continue
        # NB: Element truthiness depends on child count, so test membership explicitly.
        date_el = next((fields[k] for k in ("pubDate", "published", "updated", "date")
                        if k in fields), None)
        desc_el = next((fields[k] for k in ("description", "summary") if k in fields), None)
        desc = None
        if desc_el is not None and desc_el.text:
            desc = BeautifulSoup(desc_el.text, "html.parser").get_text(" ", strip=True)[:500]
        items.append({
            "url": normalize_url(urljoin(base_url, link)),
            "title": (fields["title"].text or "").strip() if "title" in fields else None,
            "published_at": to_iso(date_el.text if date_el is not None else None),
            "description": desc,
        })
    return items
