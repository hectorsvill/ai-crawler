# AGENTS.md — ai-crawler for AI agents

Exact, copy-pasteable commands. Everything is non-interactive. Run from the repo root.

## Setup (once)

```bash
cd /home/sentinel/Github/ai-crawler
# If `python3 -m venv` fails with "ensurepip is not available" (no python3-venv, no sudo):
python3 -m venv --without-pip .venv && curl -sSL https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
.venv/bin/pip install -r requirements.txt      # full stack (LLM workflows, dashboard)
.venv/bin/pip install pytest pytest-asyncio    # tests
.venv/bin/python main.py doctor --json         # readiness report
```

`doctor` exits 0 when the radar crawler is usable; check `llm_crawl_ready` before using
`crawl` / `research` (they need Ollama plus the models in `default_config.yaml`).

## Contract

- **stdout = data, stderr = humans.** With `--json`, stdout holds exactly one JSON document
  (or JSONL for `radar export --format jsonl`). Progress, logs and tables go to stderr.
- **Errors with `--json`** print `{"error": "...", "exit_code": N}` on stdout.
- **Exit codes** (`utils/exit_codes.py`):

| Code | Meaning |
|------|---------|
| 0 | success (a crawl may still list per-page failures in its JSON) |
| 1 | runtime failure (e.g. a crawl stored nothing) |
| 2 | usage error (bad flag/value, invalid FTS query) |
| 3 | missing dependency / unreachable service (Ollama, seeds file, package) |
| 4 | not found (session, page, paper) |
| 5 | `radar crawl --strict` only: finished but some pages failed |

- **`--dry-run`** on every live crawl command (`radar crawl`, `crawl`, `research`): validates
  inputs and prints the plan, no page fetches, no DB writes.
- **Config** via flags, then env: `CRAWLER_RADAR_DB`, `CRAWLER_RADAR_UA`, and
  `CRAWLER_<SECTION>__<KEY>` for the LLM crawler (e.g. `CRAWLER_OLLAMA__BASE_URL`,
  `CRAWLER_STORAGE__DB_PATH`).
- **Politeness** (radar): honest UA `ai-crawler-radar/1.0 (+https://github.com/hectorsvill/ai-crawler)`,
  robots.txt checked for that UA (RFC 9309: 4xx = allowed, 5xx/unreachable = disallowed),
  `Crawl-delay` honoured, >= 2 s between requests per domain (+ jitter), concurrency 2,
  3 MB response cap, hard `--max-pages` budget.

## Radar crawler (no LLM) — `main.py radar ...`

Own SQLite DB at `data/crawler.db` (gitignored). Seeds: `radar/seeds.yaml`.

```bash
# What would run (offline)
.venv/bin/python main.py radar crawl --dry-run --json

# Full radar crawl: all seeds, depth 1, <= 200 fetches, feed items from the last 7 days
.venv/bin/python main.py radar crawl --max-pages 200 --max-depth 1 --json > /tmp/run.json

# Only some topics (repeatable)
.venv/bin/python main.py radar crawl --topic ai-security --topic chips-export-controls --json

# Ad-hoc seeds instead of the seeds file
.venv/bin/python main.py radar crawl --url https://www.securityweek.com/feed/ --kind feed --topic ai-security --max-items 10 --json

# Query
.venv/bin/python main.py radar pages --since 2026-11-12 --until 2026-11-14 --json
.venv/bin/python main.py radar pages --topic war-iran-gulf --limit 20 --json
.venv/bin/python main.py radar pages --q '"export controls" AND nvidia' --json   # FTS5 syntax
.venv/bin/python main.py radar pages --domain bbc.com --run 3 --json
.venv/bin/python main.py radar show 42 --json          # or: radar show --url https://...
.venv/bin/python main.py radar runs --json
.venv/bin/python main.py radar stats --json
.venv/bin/python main.py radar topics --json

# Export (stdout by default; -o writes a file and prints a JSON receipt on stderr)
.venv/bin/python main.py radar export --format jsonl --since 2026-11-12 > pages.jsonl
.venv/bin/python main.py radar export --format json --topic ai-security -o exports/ai-security.json
.venv/bin/python main.py radar export --format markdown --no-text --limit 50
```

`radar crawl --json` result: `{run_id, status: completed|partial|failed, db_path, elapsed_s,
counters: {fetched, stored, failed, skipped, feed_only, inserted, updated, unchanged, duplicate_content},
domains, errors: [{url, error}], skips: [{url, reason}]}`.

Page objects: `{id, url, canonical_url, domain, title, description, published_at, fetched_at,
http_status, content_hash, source_seed, first_run_id, last_run_id, times_seen, tags[, text_md]}`.
Re-crawling is idempotent: rows upsert on `url`; identical content under another URL is
recorded without duplicating text (`duplicate_content`). If an article URL refuses us
(e.g. HTTP 403) but its feed entry was readable, the feed's title/date/summary is stored with
`http_status` = the error code and `text_md` = null (`feed_only`).

Topics: `frontier-labs, open-weights, agents-mcp-coding, chips-export-controls, ai-security,
war-ukraine-russia, war-iran-gulf, taiwan-pla, shtf` (keyword rules in `radar/topics.py`).
Tags are keyword matches, not judgments; verify before reporting anything.

### DB schema (`radar/db.py`)

```
runs(id, started_at, ended_at, status, params JSON, pages_fetched, pages_stored,
     pages_failed, pages_skipped, errors JSON)
pages(id, url UNIQUE, canonical_url, domain, title, description, published_at, fetched_at,
      http_status, content_hash, text_md, source_seed, first_run_id, last_run_id, times_seen)
tags(id, name UNIQUE)          page_tags(page_id, tag_id)  -- PK(page_id, tag_id)
pages_fts  -- FTS5 over (title, text_md), kept in sync by triggers
```

Direct SQL is fine for reads: `sqlite3 data/crawler.db 'SELECT title, url FROM pages ORDER BY published_at DESC LIMIT 10'`.

## LLM crawler (needs Ollama)

```bash
.venv/bin/python main.py crawl --goal "..." -u https://example.com --workflow simple --dry-run --json
.venv/bin/python main.py crawl --goal "..." -u https://example.com --workflow simple --max-pages 20 --json
.venv/bin/python main.py list-sessions --json
.venv/bin/python main.py export <session-id> --format json -o -          # '-' = stdout
.venv/bin/python main.py export <session-id> --format markdown -o pages.md
.venv/bin/python main.py resume <session-id>
.venv/bin/python main.py research --goal "..." -u "https://arxiv.org/list/cs.CR/new" --dry-run
.venv/bin/python main.py research-search "prompt injection" --json
.venv/bin/python main.py research-list --json
.venv/bin/python main.py research-show 1 --json
```

LLM data lives in `crawl_data.db` (separate from the radar DB).
The `web` dashboard (`main.py web`) is for humans; agents should use the CLI.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q    # offline: no network, no Ollama
```
