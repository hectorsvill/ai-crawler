"""radar/db.py — stdlib sqlite3 store for radar crawls.

Default path: data/crawler.db (override with --db or CRAWLER_RADAR_DB).

Tables
  runs       one row per crawl run (params, counters, status)
  pages      one row per stored URL (idempotent upsert on url); content_hash for dedupe
  run_urls         one row per URL encountered in a run (every status and reason)
  run_url_counts   convenience mirror of those rows, grouped by status and reason
  tags       topic vocabulary
  page_tags  many-to-many pages<->tags
  pages_fts  FTS5 index over title + text (if the SQLite build supports FTS5)

Schema version (PRAGMA user_version):
  1  original tables (runs, pages, tags, page_tags, pages_fts)
  2  adds run_urls. Opening any older file applies this additively.
  3  adds run_url_counts. New runs rewrite it from run_urls so the two match.
     Older files may still hold count-only totals with no row; those rows and
     totals are left in place. Retention is not a schema setting: rows are kept
     until a positive --keep-url-log-days prune.
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

# Bump when _migrate applies a new additive change. Existing files upgrade on open.
SCHEMA_VERSION = 3

# Every not_fetched reason is a run_urls row. run_url_counts mirrors those rows.
NOT_FETCHED_REASONS = (
    "budget", "depth", "link_cap", "max_items", "no_title", "recent_days", "require_match",
)

# 0 keeps run_urls forever. A positive value is an explicit prune window.
DEFAULT_KEEP_URL_LOG_DAYS = 0

# run_urls.status. ``fetched`` is a seed document (RSS/Atom) that was retrieved and
# parsed but is not itself a pages row. A fetch that later saved feed metadata is
# ``feed_only`` (reason/http_status still describe the failure or crawl-delay).
URL_STATUSES = (
    "stored",       # pages row written with a body (inserted / updated / unchanged)
    "feed_only",    # pages row written from feed metadata, no body
    "failed",       # request failed and no pages row was written
    "skipped",      # robots.txt, long Crawl-delay with nothing to store, non-HTML type
    "not_fetched",  # discovered (or filtered) and never requested
    "duplicate",    # pages row written but text omitted (same content_hash, other URL)
    "fetched",      # seed feed/index document consumed, not stored as a page
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

# Applied by _migrate (not only on first create) so databases built before this
# table existed pick it up. CREATE TABLE IF NOT EXISTS is the idempotent path;
# missing columns on an older run_urls are added below.
RUN_URLS_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_urls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    url             TEXT NOT NULL,
    domain          TEXT NOT NULL,
    source_seed     TEXT,                 -- seed URL this candidate came from
    discovered_from TEXT,                 -- parent URL (NULL for the seed itself)
    depth           INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL,        -- see URL_STATUSES
    reason          TEXT,                 -- robots.txt, budget, HTTP 403, duplicate_content, ...
    http_status     INTEGER,
    discovered_at   TEXT NOT NULL,
    fetched_at      TEXT,                 -- set when a request was actually sent
    UNIQUE (run_id, url)
);
CREATE INDEX IF NOT EXISTS ix_run_urls_run_status ON run_urls(run_id, status);
CREATE INDEX IF NOT EXISTS ix_run_urls_domain    ON run_urls(domain);

CREATE TABLE IF NOT EXISTS run_url_counts (
    run_id  INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    status  TEXT NOT NULL,
    reason  TEXT NOT NULL,
    n       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, status, reason)
);
"""

# Columns added after the table first shipped. ALTER is a no-op when present.
_RUN_URL_ADD_COLUMNS: tuple[tuple[str, str], ...] = (
    ("source_seed", "TEXT"),
    ("discovered_from", "TEXT"),
    ("depth", "INTEGER NOT NULL DEFAULT 0"),
    ("reason", "TEXT"),
    ("http_status", "INTEGER"),
    ("discovered_at", "TEXT"),
    ("fetched_at", "TEXT"),
)

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
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Additive, idempotent upgrades. Safe on a brand-new file and on a DB
        created before ``run_urls`` existed. Does not rewrite ``pages`` or FTS."""
        self.conn.executescript(RUN_URLS_SCHEMA)
        have = {row[1] for row in self.conn.execute("PRAGMA table_info(run_urls)")}
        for name, decl in _RUN_URL_ADD_COLUMNS:
            if name not in have:
                self.conn.execute(f"ALTER TABLE run_urls ADD COLUMN {name} {decl}")
        current = int(self.conn.execute("PRAGMA user_version").fetchone()[0])
        if current < SCHEMA_VERSION:
            self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

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

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["params"] = json.loads(d["params"] or "{}")
        d["errors"] = json.loads(d["errors"] or "[]")
        return d

    # ── per-run URL log ─────────────────────────────────────────────────────
    def log_url(self, row: dict[str, Any], *, overwrite: bool = True) -> None:
        """Record one encountered URL for a run.

        ``overwrite=False`` inserts only when ``(run_id, url)`` is new, so a
        later "not fetched" note cannot clobber a fetch outcome. ``overwrite=True``
        replaces status/reason (used to upgrade a failure into ``feed_only`` or a
        discovered URL into ``stored``) and keeps the original ``discovered_at``.
        """
        status = row["status"]
        if status not in URL_STATUSES:
            raise ValueError(f"unknown url status: {status}")
        cols = (
            "run_id", "url", "domain", "source_seed", "discovered_from", "depth",
            "status", "reason", "http_status", "discovered_at", "fetched_at",
        )
        values = (
            row["run_id"], row["url"], row["domain"], row.get("source_seed"),
            row.get("discovered_from"), int(row.get("depth") or 0), status,
            row.get("reason"), row.get("http_status"),
            row.get("discovered_at") or utcnow(), row.get("fetched_at"),
        )
        placeholders = ",".join("?" * len(cols))
        if not overwrite:
            self.conn.execute(
                f"INSERT OR IGNORE INTO run_urls ({', '.join(cols)}) VALUES ({placeholders})",
                values,
            )
        else:
            self.conn.execute(
                f"""INSERT INTO run_urls ({', '.join(cols)}) VALUES ({placeholders})
                    ON CONFLICT(run_id, url) DO UPDATE SET
                        domain=excluded.domain,
                        source_seed=COALESCE(excluded.source_seed, run_urls.source_seed),
                        discovered_from=COALESCE(run_urls.discovered_from, excluded.discovered_from),
                        depth=excluded.depth,
                        status=excluded.status,
                        reason=excluded.reason,
                        http_status=COALESCE(excluded.http_status, run_urls.http_status),
                        fetched_at=COALESCE(run_urls.fetched_at, excluded.fetched_at)""",
                values,
            )
        self.conn.commit()

    def sync_url_counts(self, run_id: int) -> None:
        """Replace this run's ``run_url_counts`` with the exact ``run_urls`` totals.

        One group per ``(status, reason)``. A null reason is stored as ``''`` because
        the counts table requires a reason. Other runs are not touched.
        """
        self.conn.execute("DELETE FROM run_url_counts WHERE run_id = ?", (run_id,))
        self.conn.execute(
            """INSERT INTO run_url_counts (run_id, status, reason, n)
               SELECT run_id, status, COALESCE(reason, ''), COUNT(*)
               FROM run_urls WHERE run_id = ?
               GROUP BY run_id, status, COALESCE(reason, '')""",
            (run_id,),
        )
        self.conn.commit()

    def add_url_counts(self, run_id: int, counts: dict[tuple[str, str], int]) -> None:
        """Add aggregate (status, reason) -> n counts for a run (accumulates).

        Used for historical count-only totals. A finished crawl calls
        ``sync_url_counts`` instead, so its counts match its rows.
        """
        for (status, reason), n in counts.items():
            if status not in URL_STATUSES:
                raise ValueError(f"unknown url status: {status}")
            if not n:
                continue
            self.conn.execute(
                """INSERT INTO run_url_counts (run_id, status, reason, n) VALUES (?,?,?,?)
                   ON CONFLICT(run_id, status, reason) DO UPDATE SET n = n + excluded.n""",
                (run_id, status, reason, int(n)),
            )
        self.conn.commit()

    def prune_url_log(self, keep_days: int, *, dry_run: bool = False) -> dict[str, Any]:
        """Delete run_urls rows of runs started more than ``keep_days`` ago.

        ``runs`` counters and ``run_url_counts`` are never touched. ``keep_days`` <= 0
        is a no-op. Returns {keep_days, cutoff, runs, rows, dry_run}.
        """
        from datetime import timedelta
        if keep_days <= 0:
            return {"keep_days": keep_days, "cutoff": None, "runs": [], "rows": 0,
                    "dry_run": dry_run, "disabled": True}
        cutoff = (datetime.now(timezone.utc) - timedelta(days=keep_days)).replace(
            microsecond=0).isoformat()
        sel = ("SELECT u.run_id, COUNT(*) FROM run_urls u JOIN runs r ON r.id = u.run_id "
               "WHERE r.started_at < ? GROUP BY u.run_id ORDER BY u.run_id")
        per_run = [(int(a), int(b)) for a, b in self.conn.execute(sel, (cutoff,))]
        rows = sum(n for _, n in per_run)
        if per_run and not dry_run:
            self.conn.execute(
                "DELETE FROM run_urls WHERE run_id IN "
                "(SELECT id FROM runs WHERE started_at < ?)", (cutoff,))
            self.conn.commit()
        return {"keep_days": keep_days, "cutoff": cutoff, "runs": [r for r, _ in per_run],
                "rows": rows, "dry_run": dry_run, "disabled": False}

    def query_urls(
        self,
        run_id: int | None = None,
        status: str | None = None,
        domain: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """URLs encountered in a run, oldest discovery first (stable)."""
        if status is not None and status not in URL_STATUSES:
            raise ValueError(f"unknown url status: {status}")
        where, args = [], []
        if run_id is not None:
            where.append("run_id = ?")
            args.append(run_id)
        if status is not None:
            where.append("status = ?")
            args.append(status)
        if domain:
            where.append("(domain = ? OR domain LIKE ?)")
            args += [domain, f"%.{domain}"]
        sql = (
            "SELECT id, run_id, url, domain, source_seed, discovered_from, depth, "
            "status, reason, http_status, discovered_at, fetched_at FROM run_urls"
        )
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY run_id ASC, depth ASC, id ASC LIMIT ? OFFSET ?"
        args += [limit, offset]
        return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def count_urls(self, run_id: int | None = None) -> int:
        if run_id is None:
            return int(self.conn.execute("SELECT COUNT(*) FROM run_urls").fetchone()[0])
        return int(self.conn.execute(
            "SELECT COUNT(*) FROM run_urls WHERE run_id = ?", (run_id,)
        ).fetchone()[0])

    def url_counts(self, run_id: int) -> dict[str, Any]:
        """Per-status / per-reason counts for one run, reconciled with ``runs`` counters.

        Mapping (a URL has one row; its final status wins):
          pages_fetched = rows with an HTTP response (http_status set)
          pages_stored  = stored + duplicate + feed_only
          pages_failed  = failed + feed_only whose reason is "HTTP ..." (403 -> feed metadata)
          pages_skipped = skipped + feed_only whose reason is "crawl-delay ..."
        A URL is stored once per run. ``run_url_counts`` for a crawl matches those
        rows (``aggregated`` is 0). Older count-only totals are added only when
        they exceed the rows, so they are not counted twice. ``ok`` is false when
        the run counters and the rows disagree.
        """
        c = self.conn

        def q(sql: str) -> int:
            return int(c.execute(sql, (run_id,)).fetchone()[0])

        row_groups = [dict(r) for r in c.execute(
            "SELECT status, reason, COUNT(*) AS count FROM run_urls WHERE run_id = ? "
            "GROUP BY status, reason ORDER BY status, count DESC", (run_id,)).fetchall()]
        by_status: dict[str, int] = {}
        by_reason: list[dict[str, Any]] = []
        row_n: dict[tuple[str, str], int] = {}
        for group in row_groups:
            status, reason, count = group["status"], group["reason"], int(group["count"])
            by_status[status] = by_status.get(status, 0) + count
            by_reason.append({"status": status, "reason": reason, "count": count})
            key = (status, reason or "")
            row_n[key] = row_n.get(key, 0) + count
        # Legacy v3 totals had no row. Add only the surplus so a mirror of the
        # rows (n == row count, null reason stored as '') is not double-counted.
        aggregated = 0
        for group in c.execute(
            "SELECT status, reason, n AS count FROM run_url_counts WHERE run_id = ? "
            "ORDER BY status, n DESC", (run_id,)
        ):
            status, reason, count = group["status"], group["reason"] or "", int(group["count"])
            surplus = count - row_n.get((status, reason), 0)
            if surplus <= 0:
                continue
            aggregated += surplus
            by_status[status] = by_status.get(status, 0) + surplus
            by_reason.append({
                "status": status, "reason": reason or None, "count": surplus, "aggregated": True,
            })
        derived = {
            "pages_fetched": q("SELECT COUNT(*) FROM run_urls WHERE run_id = ? "
                               "AND http_status IS NOT NULL"),
            "pages_stored": sum(by_status.get(k, 0) for k in ("stored", "duplicate", "feed_only")),
            "pages_failed": by_status.get("failed", 0) + q(
                "SELECT COUNT(*) FROM run_urls WHERE run_id = ? AND status = 'feed_only' "
                "AND reason LIKE 'HTTP %'"),
            "pages_skipped": by_status.get("skipped", 0) + q(
                "SELECT COUNT(*) FROM run_urls WHERE run_id = ? AND status = 'feed_only' "
                "AND reason LIKE 'crawl-delay%'"),
        }
        run = c.execute("SELECT pages_fetched, pages_stored, pages_failed, pages_skipped "
                        "FROM runs WHERE id = ?", (run_id,)).fetchone()
        counters = dict(run) if run else {}
        recon = {k: {"run": counters.get(k), "url_log": v, "match": counters.get(k) == v}
                 for k, v in derived.items()}
        rows_total = sum(int(group["count"]) for group in row_groups)
        return {"run_id": run_id, "total": rows_total + aggregated, "rows": rows_total,
                "aggregated": aggregated,
                "by_status": {k: by_status.get(k, 0) for k in URL_STATUSES},
                "by_reason": by_reason, "reconciliation": recon,
                "ok": all(r["match"] for r in recon.values())}

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
            "url_log": c.execute("SELECT COUNT(*) FROM run_urls").fetchone()[0],
            "url_log_aggregated": c.execute(
                "SELECT COALESCE(SUM(n), 0) FROM run_url_counts").fetchone()[0],
            "tags": self.tag_counts(),
            "by_domain": [dict(r) for r in c.execute(
                "SELECT domain, COUNT(*) AS pages FROM pages GROUP BY domain ORDER BY pages DESC"
            ).fetchall()],
        }
