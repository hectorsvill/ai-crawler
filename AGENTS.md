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
`crawl` / `research` (needs the active provider ready: Ollama reachable with its models, or
the provider's API key set). `providers` lists which keys are configured (names only).

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
| 3 | missing dependency / unreachable service (Ollama, API key, seeds file, package) |
| 4 | not found (session, page, paper) |
| 5 | `radar crawl --strict` only: finished but some pages failed |

- **`--dry-run`** on every live crawl/LLM command (`radar crawl`, `crawl`, `research`, `llm-ping`): validates
  inputs and prints the plan, no page fetches, no DB writes.
- **Config** via flags, then env: `CRAWLER_RADAR_DB`, `CRAWLER_RADAR_UA`, and
  `CRAWLER_<SECTION>__<KEY>` for the LLM crawler (e.g. `CRAWLER_OLLAMA__BASE_URL`,
  `CRAWLER_STORAGE__DB_PATH`).
- **Politeness** (radar): honest UA `ai-crawler-radar/1.0 (+https://github.com/hectorsvill/ai-crawler)`,
  robots.txt checked for that UA (RFC 9309: 4xx = allowed, 5xx/unreachable = disallowed),
  `Crawl-delay` honoured (above 30 s: only one request to that domain per run; seeds may set
`feed_only: true` to store feed entries without fetching articles), >= 2 s between requests per domain (+ jitter), concurrency 2,
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
war-ukraine-russia, war-iran-gulf, taiwan-pla, war-drones, shtf` (keyword rules in `radar/topics.py`).
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

## LLM crawler (Ollama or an API provider)

Providers: `ollama` (default, local) · `gemini` · `anthropic` · `openai` · `deepseek`.
Pick with `--provider/-P` (or `CRAWLER_LLM__PROVIDER`, or `llm.provider` in YAML) and
`--model/-m` (or `CRAWLER_LLM__MODEL`). Defaults when no model is given:
`gemini-3.5-flash-lite` (extractor role: `gemini-3.5-flash`), `claude-haiku-4-5`, `gpt-4.1-mini`, `deepseek-chat`; Ollama keeps the
`ollama:` section models (`--model` then overrides only the extractor).

**API keys: env vars only** — never in YAML, flags, or logs:

| Provider | Env var(s) checked in order |
|----------|-----------------------------|
| gemini | `GEMINI_API_KEY`, `GOOGLE_API_KEY` |
| anthropic | `ANTHROPIC_API_KEY` |
| openai | `OPENAI_API_KEY` |
| deepseek | `DEEPSEEK_API_KEY` |

Optional key file, loaded automatically without overriding real env vars:
`~/.config/ai-crawler/env` (override path with `CRAWLER_ENV_FILE`):

```bash
mkdir -p ~/.config/ai-crawler && touch ~/.config/ai-crawler/env && chmod 600 ~/.config/ai-crawler/env
vim ~/.config/ai-crawler/env        # add a line:  GEMINI_API_KEY=...
.venv/bin/python main.py doctor --json | jq .providers   # shows configured: true/false (names only)
```

```bash
# Verify a provider with ONE tiny call (records tokens under run_id "llm-ping")
.venv/bin/python main.py llm-ping --provider gemini --model gemini-3.5-flash-lite --dry-run --json
.venv/bin/python main.py llm-ping --provider gemini --model gemini-3.5-flash-lite --json

# Crawl with a provider
.venv/bin/python main.py crawl --goal "..." -u https://example.com -w simple -P gemini -m gemini-3.5-flash-lite --max-pages 5 --dry-run --json
.venv/bin/python main.py crawl --goal "..." -u https://example.com -w simple -P gemini --max-pages 5 --json
.venv/bin/python main.py research --goal "..." -u "https://arxiv.org/list/cs.CR/new" -P anthropic --dry-run
.venv/bin/python main.py resume <session-id> -P openai

# Sessions / data
.venv/bin/python main.py list-sessions --json
.venv/bin/python main.py export <session-id> --format json -o -          # '-' = stdout
.venv/bin/python main.py export <session-id> --format markdown -o pages.md
.venv/bin/python main.py research-search "prompt injection" --json
.venv/bin/python main.py research-list --json
.venv/bin/python main.py research-show 1 --json
```

`crawl --json` result includes `provider` and `llm_usage` (token totals for that session).
A missing key or unreachable Ollama exits **3** before any crawling.

### Choosing a model (live lists)

```bash
.venv/bin/python main.py models --json                 # every provider: configured, default, models
.venv/bin/python main.py models -P gemini --json       # one provider (live API list, cached ~10 min)
.venv/bin/python main.py models -P gemini --refresh --json
```

Each provider entry: `provider, configured, env_var, models[{id,name}], default_model,
role_defaults, default_model_listed, active, error, cached`. A missing key is
`configured: false` (exit 0), never an error; keys are never printed. Lists come from the
provider APIs: Gemini `ListModels` (only `generateContent` models), OpenAI `/v1/models`,
Anthropic `/v1/models`, DeepSeek `/models`, Ollama `/api/tags`.

`-P/--provider` and `-m/--model` work the same on every LLM command (`crawl`, `research`,
`resume`, `llm-ping`). Before any live call, `-m` is checked against the live list; an
unknown name exits **2** with the closest matches, e.g.:

```json
{"error": "Unknown model 'gemini-3.5-flsh' for provider 'gemini'. Closest: gemini-3.5-flash, ...",
 "exit_code": 2, "provider": "gemini", "model": "gemini-3.5-flsh", "suggestions": ["gemini-3.5-flash"]}
```

Skip the check with `--no-validate` (e.g. a model the list doesn't show yet). If the list
can't be fetched (no key, provider down) nothing is decided there; the readiness check
then exits 3. Note: a model can be listed and still be unavailable to a given key
(e.g. `gemini-2.5-flash-lite` returns 404 for new Gemini users) — `llm-ping -m` confirms.

### Token usage

Every LLM call is recorded in table `llm_usage` (same SQLite file as sessions,
`storage.db_path` = `crawl_data.db`, or `llm.usage_db_path`). Token counts come from each
provider's usage fields (Ollama `prompt_eval_count/eval_count`, OpenAI/DeepSeek `usage`,
Anthropic `usage.input/output_tokens`, Gemini `usageMetadata`, thinking tokens counted as
output). If a provider omits usage, a tiktoken estimate is stored with `estimated = 1`.

```
llm_usage(id, ts UTC, run_id, provider, model, purpose, prompt_tokens, completion_tokens,
          total_tokens, estimated, cost_usd, latency_ms, ok, error)
```

`cost_usd` is filled only for models in the `llm.prices` table (USD per 1M tokens,
`{model: {input: x, output: y}}`, prefix match); otherwise null.

```bash
.venv/bin/python main.py usage --json                                  # totals + per model
.venv/bin/python main.py usage --provider gemini --since 2026-11-01 --group-by day --json
.venv/bin/python main.py usage --run <session-id> --calls 20 --json    # + last 20 calls
```
`--group-by`: `provider | model | run | day | purpose | none`.

The `crewai` workflow still talks to Ollama directly (CrewAI's own LLM wrapper).
LLM data lives in `crawl_data.db` (separate from the radar DB).
The `web` dashboard (`main.py web --port 8420`) is for humans; agents should use the CLI.
It has a provider dropdown (only configured providers are selectable; others show "no key"),
a model dropdown filled from the live list, a Research workflow option, and a token-usage
panel. JSON endpoints (keys never sent to the browser):
`GET /api/models` (provider status), `GET /api/models?provider=gemini` (live list),
`GET /api/usage?run_id=&provider=&model=&since=&group_by=model`,
`POST /api/crawl {goal, start_url, workflow, provider, model, no_validate}` (400 + `suggestions`
for an unknown model; response has `usage_url` for per-run tokens).

## Tests

```bash
.venv/bin/python -m pytest tests/ -q    # offline: no network, no Ollama
```
