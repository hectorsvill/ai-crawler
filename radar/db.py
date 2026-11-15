"""radar/db.py — stdlib sqlite3 store for radar crawls.

Default path: data/crawler.db (override with --db or CRAWLER_RADAR_DB).

Tables
  runs       one row per crawl run (params, counters, status)
  pages      one row per URL (idempotent upsert on url); content_hash for dedupe
  tags       topic vocabulary
  page_tags  many-to-many pages<->tags
  pages_fts  FTS5 index over title + text (if the SQLite build supports FTS5)
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_DB_PATH = os.environ.get(
    "CRAWLER_RADAR_DB", str(Path(__file__).resolve().parent.parent / "data" / "crawler.db")
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    ended_at      TEXT,
    status        TEXT NOT NULL DEFAULT 'running',   -- running|completed|failed|partial
    params        TEXT NOT NULL DEFAULT '{}',        -- JSON
    pages_fetched INTEGER NOT NULL DEFAULT 0,
    pages_stored  INTEGER NOT NULL DEFAULT 0,
    pages_failed  INTEGER NOT NULL DEFAULT 0,
    pages_skipped INTEGER NOT NULL DEFAULT 0,
    errors        TEXT NOT NULL DEFAULT '[]'         -- JSON list of {url, error}
);

CREATE TABLE IF NOT EXISTS pages (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    url            TEXT NOT NULL UNIQUE,
    canonical_url  TEXT,
    domain         TEXT NOT NULL,
    title          TEXT,
    description    TEXT,
    published_at   TEXT,              -- ISO-8601 if found on the page/feed
    fetched_at     TEXT NOT NULL,
    http_status    INTEGER,
    content_hash   TEXT,
    text_md        TEXT,
    source_seed    TEXT,
    first_run_id   INTEGER REFERENCES runs(id),
    last_run_id    INTEGER REFERENCES runs(id),
    times_seen     INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS ix_pages_domain    ON pages(domain);
CREATE INDEX IF NOT EXISTS ix_pages_published ON pages(published_at);
CREATE INDEX IF NOT EXISTS ix_pages_hash      ON pages(content_hash);

CREATE TABLE IF NOT EXISTS tags (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS page_tags (
    page_id INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    tag_id  INTEGER NOT NULL REFERENCES tags(id)  ON DELETE CASCADE,
    PRIMARY KEY (page_id, tag_id)
);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
    title, text_md, content='pages', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS pages_ai AFTER INSERT ON pages BEGIN
    INSERT INTO pages_fts(rowid, title, text_md) VALUES (new.id, new.title, new.text_md);
END;
CREATE TRIGGER IF NOT EXISTS pages_ad AFTER DELETE ON pages BEGIN
    INSERT INTO pages_fts(pages_fts, rowid, title, text_md)
    VALUES ('delete', old.id, old.title, old.text_md);
END;
CREATE TRIGGER IF NOT EXISTS pages_au AFTER UPDATE OF title, text_md ON pages BEGIN
    INSERT INTO pages_fts(pages_fts, rowid, title, text_md)
    VALUES ('delete', old.id, old.title, old.text_md);
    INSERT INTO pages_fts(rowid, title, text_md) VALUES (new.id, new.title, new.text_md);
END;
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class RadarDB:
    """Thin wrapper around a sqlite3 connection. Safe to construct repeatedly."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = str(path or DEFAULT_DB_PATH)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)
        self.has_fts = True
        try:
            self.conn.executescript(FTS_SCHEMA)
        except sqlite3.OperationalError:
            self.has_fts = False
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "RadarDB":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── runs ────────────────────────────────────────────────────────────────
    def start_run(self, params: dict[str, Any]) -> int:
        cur = self.conn.execute(
            "INSERT INTO runs (started_at, params) VALUES (?, ?)",
            (utcnow(), json.dumps(params, default=str)),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, status: str, counters: dict[str, int],
                   errors: list[dict[str, str]]) -> None:
        self.conn.execute(
            """UPDATE runs SET ended_at=?, status=?, pages_fetched=?, pages_stored=?,
                   pages_failed=?, pages_skipped=?, errors=? WHERE id=?""",
            (utcnow(), status, counters.get("fetched", 0), counters.get("stored", 0),
             counters.get("failed", 0), counters.get("skipped", 0),
             json.dumps(errors[:500]), run_id),
        )
        self.conn.commit()

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["params"] = json.loads(d["params"] or "{}")
            d["errors"] = json.loads(d["errors"] or "[]")
            out.append(d)
        return out

    # ── pages ───────────────────────────────────────────────────────────────
    def upsert_page(self, page: dict[str, Any], tags: Iterable[str], run_id: int | None) -> tuple[int, str]:
        """Insert or update a page by URL. Returns (page_id, action).

        action is one of: inserted | updated | unchanged | duplicate_content
        ``duplicate_content`` means a *different* URL already stores the same
        content hash; the row is still recorded (so the URL is known) but
        without duplicating the text.
        """
        url = page["url"]
        existing = self.conn.execute(
            "SELECT id, content_hash FROM pages WHERE url = ?", (url,)
        ).fetchone()
        h = page.get("content_hash")
        if existing is None:
            dup = None
            if h:
                dup = self.conn.execute(
                    "SELECT id FROM pages WHERE content_hash = ? AND url != ? LIMIT 1", (h, url)
                ).fetchone()
            text = None if dup else page.get("text_md")
            cur = self.conn.execute(
                """INSERT INTO pages (url, canonical_url, domain, title, description, published_at,
                       fetched_at, http_status, content_hash, text_md, source_seed,
                       first_run_id, last_run_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (url, page.get("canonical_url"), page["domain"], page.get("title"),
                 page.get("description"), page.get("published_at"),
                 page.get("fetched_at") or utcnow(), page.get("http_status"), h, text,
                 page.get("source_seed"), run_id, run_id),
            )
            page_id = int(cur.lastrowid)
            action = "duplicate_content" if dup else "inserted"
        else:
            page_id = int(existing["id"])
            action = "unchanged" if (h and existing["content_hash"] == h) else "updated"
            self.conn.execute(
                """UPDATE pages SET canonical_url=COALESCE(?, canonical_url),
                       title=COALESCE(?, title), description=COALESCE(?, description),
                       published_at=COALESCE(?, published_at), fetched_at=?, http_status=?,
                       content_hash=COALESCE(?, content_hash),
                       text_md=CASE WHEN ? IS NOT NULL THEN ? ELSE text_md END,
                       last_run_id=?, times_seen=times_seen+1
                   WHERE id=?""",
                (page.get("canonical_url"), page.get("title"), page.get("description"),
                 page.get("published_at"), page.get("fetched_at") or utcnow(),
                 page.get("http_status"), h, page.get("text_md"), page.get("text_md"),
                 run_id, page_id),
            )
        for t in set(tags):
            self.conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (t,))
            self.conn.execute(
                "INSERT OR IGNORE INTO page_tags (page_id, tag_id) "
                "SELECT ?, id FROM tags WHERE name = ?", (page_id, t))
        self.conn.commit()
        return page_id, action

    def query_pages(
        self,
        topic: str | None = None,
        domain: str | None = None,
        since: str | None = None,
        until: str | None = None,
        q: str | None = None,
        run_id: int | None = None,
        limit: int = 50,
        offset: int = 0,
        include_text: bool = False,
    ) -> list[dict[str, Any]]:
        where, args = [], []
        joins = ""
        if topic:
            joins += " JOIN page_tags pt ON pt.page_id = p.id JOIN tags t ON t.id = pt.tag_id"
            where.append("t.name = ?")
            args.append(topic)
        if domain:
            where.append("(p.domain = ? OR p.domain LIKE ?)")
            args += [domain, f"%.{domain}"]
        if since:
            where.append("p.published_at >= ?")
            args.append(since)
        if until:
            # inclusive day bound when a bare date is given
            where.append("p.published_at < ?")
            args.append(until + "T23:59:59.999" if len(until) == 10 else until)
        if run_id is not None:
            where.append("(p.last_run_id = ? OR p.first_run_id = ?)")
            args += [run_id, run_id]
        if q:
            if self.has_fts:
                joins += " JOIN pages_fts f ON f.rowid = p.id"
                where.append("pages_fts MATCH ?")
            else:
                where.append("(p.title LIKE ? OR p.text_md LIKE ?)")
                args.append(f"%{q}%")
                q = f"%{q}%"
            args.append(q)
        cols = ("p.id, p.url, p.canonical_url, p.domain, p.title, p.description, "
                "p.published_at, p.fetched_at, p.http_status, p.content_hash, p.source_seed, "
                "p.first_run_id, p.last_run_id, p.times_seen")
        if include_text:
            cols += ", p.text_md"
        sql = f"SELECT DISTINCT {cols} FROM pages p{joins}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY COALESCE(p.published_at, p.fetched_at) DESC, p.id DESC LIMIT ? OFFSET ?"
        args += [limit, offset]
        rows = [dict(r) for r in self.conn.execute(sql, args).fetchall()]
        for r in rows:
            r["tags"] = self.tags_for(r["id"])
        return rows

    def get_page(self, page_id: int | None = None, url: str | None = None) -> dict[str, Any] | None:
        if page_id is not None:
            row = self.conn.execute("SELECT * FROM pages WHERE id = ?", (page_id,)).fetchone()
        else:
            row = self.conn.execute("SELECT * FROM pages WHERE url = ?", (url,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["tags"] = self.tags_for(d["id"])
        return d

    def tags_for(self, page_id: int) -> list[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT t.name FROM tags t JOIN page_tags pt ON pt.tag_id = t.id "
            "WHERE pt.page_id = ? ORDER BY t.name", (page_id,)).fetchall()]

    def tag_counts(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT t.name AS tag, COUNT(pt.page_id) AS pages FROM tags t "
            "LEFT JOIN page_tags pt ON pt.tag_id = t.id GROUP BY t.id ORDER BY pages DESC, t.name"
        ).fetchall()]

    def stats(self) -> dict[str, Any]:
        c = self.conn
        return {
            "db_path": self.path,
            "fts5": self.has_fts,
            "pages": c.execute("SELECT COUNT(*) FROM pages").fetchone()[0],
            "pages_with_published_at": c.execute(
                "SELECT COUNT(*) FROM pages WHERE published_at IS NOT NULL").fetchone()[0],
            "domains": c.execute("SELECT COUNT(DISTINCT domain) FROM pages").fetchone()[0],
            "runs": c.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
            "tags": self.tag_counts(),
            "by_domain": [dict(r) for r in c.execute(
                "SELECT domain, COUNT(*) AS pages FROM pages GROUP BY domain ORDER BY pages DESC"
            ).fetchall()],
        }
