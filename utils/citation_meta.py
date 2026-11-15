"""
utils/citation_meta.py — Read Highwire/Google Scholar ``citation_*`` meta tags.

arXiv, most journals and many repositories publish authoritative bibliographic
metadata as ``<meta name="citation_author" content="...">`` etc. Markdown converters
often drop author blocks (arXiv's author links vanish), so the research workflow uses
these tags to correct/fill what the LLM extracted.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any


class _MetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, list[str]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "meta":
            return
        a = {k.lower(): (v or "") for k, v in attrs}
        name = (a.get("name") or a.get("property") or "").lower()
        if name.startswith("citation_") and a.get("content", "").strip():
            self.meta.setdefault(name, []).append(a["content"].strip())


def parse_citation_meta(html: str | None) -> dict[str, Any]:
    """Return {title, authors, doi, arxiv_id, pdf_url, year, abstract} (only keys found)."""
    if not html or "citation_" not in html:
        return {}
    p = _MetaParser()
    try:
        p.feed(html)
    except Exception:  # noqa: BLE001 — malformed HTML: use what was parsed
        pass
    m = p.meta
    out: dict[str, Any] = {}
    if m.get("citation_title"):
        out["title"] = m["citation_title"][0]
    if m.get("citation_author"):
        out["authors"] = list(dict.fromkeys(m["citation_author"]))
    if m.get("citation_doi"):
        out["doi"] = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", m["citation_doi"][0], flags=re.I)
    if m.get("citation_arxiv_id"):
        out["arxiv_id"] = m["citation_arxiv_id"][0]
    if m.get("citation_pdf_url"):
        out["pdf_url"] = m["citation_pdf_url"][0]
    if m.get("citation_abstract"):
        out["abstract"] = m["citation_abstract"][0]
    for key in ("citation_publication_date", "citation_date", "citation_online_date"):
        if m.get(key):
            y = re.search(r"\b(19|20)\d{2}\b", m[key][0])
            if y:
                out["year"] = int(y.group(0))
                break
    return out


def merge_into_paper(paper: Any, meta: dict[str, Any]) -> Any:
    """Apply *meta* to a ResearchPaperData: authors from meta win; other fields fill gaps."""
    if not meta:
        return paper
    updates: dict[str, Any] = {}
    if meta.get("authors"):
        updates["authors"] = meta["authors"]
    for k in ("title", "doi", "arxiv_id", "pdf_url", "abstract", "year"):
        if meta.get(k) and not getattr(paper, k, None):
            updates[k] = meta[k]
    return paper.model_copy(update=updates) if updates else paper
