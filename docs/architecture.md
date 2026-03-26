# Architecture

This document explains how the ai-crawler works internally — the data pipeline, component responsibilities, and how everything connects.

---

## High-level pipeline

```
User (CLI / Python API)
         │
         ▼
      main.py  ─── parses CLI args, health-checks Ollama
         │
         ▼
  WorkflowRouter  ─── one LLM call to classify goal → simple | langgraph | crewai
         │
         ▼
  ┌──────┴───────────────────────┐
  │                              │
SimpleWorkflow          LangGraphFlow / CrewAIFlow
  │                              │
  └──────────────┬───────────────┘
                 │  (all workflows share the same components below)
                 ▼
        ┌─── RespectfulCrawler ──────────────────────────────────────┐
        │   robots.txt check → rate limiter → domain filter → depth  │
        └───────────────────────────────────────────────────────────┘
                 │
                 ▼
           CrawlEngine.fetch()
           ┌────────────────────────────────────┐
           │  1. crawl4ai (if installed)         │
           │  2. aiohttp  (static HTML)          │
           │  3. Playwright (JS-heavy / thin)    │
           └────────────────────────────────────┘
                 │  PageContent (markdown, links, hash)
                 ▼
        NavigatorAgent.decide()
        ── LLM call (navigator_model) ──
        ── returns NavigatorDecision ────
              relevance_score: 0.0–1.0
              links_to_follow: list[LinkPriority]
              action: deepen | backtrack | complete
                 │
                 ├─── if relevance >= 0.2 ──►  ExtractorAgent.extract()
                 │                             ── LLM call (extractor_model) ──
                 │                             ── returns ExtractionResult ─────
                 │                                   data: dict
                 │                                   confidence: 0.0–1.0
                 │
                 ▼
           storage/db.py  (SQLite via SQLAlchemy async)
           ┌────────────────────────────────────┐
           │  urls             (queue)           │
           │  visited_pages    (content + hash)  │
           │  extracted_data   (JSON results)    │
           │  crawl_sessions   (metadata)        │
           └────────────────────────────────────┘
```

---

## Module responsibilities

### `main.py`
Entry point. Parses CLI args with Typer, loads config, checks Ollama health, creates/resumes a session, and dispatches to the selected workflow. Catches `KeyboardInterrupt` and marks the session as interrupted so it can be resumed.

### `config.py`
Pydantic models for the full configuration tree. `Settings()` calls `load_config()`, which deep-merges `default_config.yaml` with an optional user YAML, then applies `CRAWLER_*` environment variable overrides.

Sub-models: `OllamaConfig`, `CrawlConfig`, `StorageConfig`, `WorkflowConfig`.

### `llm/client.py`
Two classes:

- **`OllamaClient`** — low-level async wrapper. `raw_call()` does the HTTP round-trip with exponential-backoff retries and model fallback. `structured_call()` injects the Pydantic schema into the system prompt, calls `raw_call()`, strips markdown fences, and validates the JSON response against the schema.
- **`LLMClient`** — high-level façade used in scripts and tests. Accepts an `OllamaConfig` object directly, exposes `generate()`, `generate_json(response_model=...)`, `check_health()`, `count_tokens()`, `chunk_text()`.

Module-level `token_usage` singleton accumulates prompt/completion tokens across all calls for the final summary.

### `crawler/engine.py`
Three fetch strategies, tried in order:

| Strategy | When used | Notes |
|----------|-----------|-------|
| `fetch_with_crawl4ai` | Always first, if installed | Returns markdown directly; skipped if result < 500 chars |
| `fetch_static` (aiohttp) | Default | Raises on non-HTML MIME types |
| `fetch_with_playwright` | Auto-upgrade when static yields < 500 chars, or forced | Waits for `networkidle`, then extracts HTML |

`html_to_markdown()` tries trafilatura first (with h1 re-injection), falls back to BeautifulSoup. `extract_links()` resolves relative → absolute, strips fragments, deduplicates, and filters non-http(s) schemes.

### `crawler/robots.py`
`RobotsCache` fetches and caches `robots.txt` per domain with a 1-hour TTL. Fails open — if `robots.txt` cannot be fetched, all URLs are allowed.

### `crawler/respectful.py`
Four guards composed into `RespectfulCrawler.check_and_wait()`:

1. **Depth limit** — rejects URLs deeper than `max_depth`
2. **Domain filter** — checks allowlist/denylist with glob patterns; `*.example.com` also matches `example.com`
3. **robots.txt** — via `RobotsCache`
4. **Rate limiter** — token-bucket per domain at `rate_limit_per_domain` req/s plus random jitter from `delay_range`

### `agents/navigator.py`
`NavigatorAgent.decide()` builds a prompt from the current page markdown, link list, goal, and crawl history, then calls `OllamaClient.structured_call()` with `NavigatorDecision` as the response schema.

Returns:
- `relevance_score` — how useful this page is (0–1)
- `links_to_follow` — up to 10 links ranked by estimated value
- `action` — `deepen` (keep following links), `backtrack` (abandon this branch), or `complete` (goal satisfied)

### `agents/extractor.py`
`ExtractorAgent.extract()` asks the LLM to infer an appropriate schema from the goal and page content and return structured JSON. `extract_chunks()` splits long pages, processes each chunk, and merges results (list fields concatenated, confidence averaged).

### `storage/models.py`
All inter-component data passes through Pydantic models — never raw dicts:

| Model | Used for |
|-------|---------|
| `PageContent` | Fetched page from crawler to agents |
| `NavigatorDecision` | Navigator output to workflow |
| `ExtractionResult` | Extractor output to storage |
| `URLItem` | URL queue entry (in-memory) |
| `SessionStats` | Live stats for progress display |

SQLAlchemy ORM models (`URLRecord`, `VisitedPage`, `ExtractedData`, `CrawlSession`) map to SQLite tables.

### `storage/db.py`
All database operations. Key behaviors:
- `enqueue_url()` normalizes URLs before insertion and silently skips duplicates
- `dequeue_next_url()` returns the highest-priority pending URL and marks it `in_progress` atomically
- `save_page()` deduplicates by SHA-256 content hash — the same content is never stored twice even if fetched from different URLs
- `resume_session()` resets stale `in_progress` entries to `pending` so they are retried

### `workflows/`

| File | Class/Function | Notes |
|------|---------------|-------|
| `router.py` | `select_workflow()` / `WorkflowRouter` | One LLM call; has rule-based fast-path for obvious simple goals |
| `simple.py` | `run_simple()` / `SimpleWorkflow` | Single async loop, no persistence between restarts |
| `langgraph_flow.py` | `run_langgraph()` | `StateGraph` with SQLite checkpointing; falls back to simple if LangGraph not installed |
| `crewai_flow.py` | `run_crewai()` | 4-agent crew; falls back to simple if CrewAI not installed |

### `utils/url.py`
`normalize_url()` produces a canonical URL: lowercase scheme/host, strip fragment, remove default ports, sort query params, strip trailing slash. Used before every `enqueue_url()` call to prevent duplicate crawling of the same page reached via different link text.

### `utils/progress.py`
`CrawlProgress` is a context manager that renders a live Rich panel during the crawl. `print_summary()` prints final stats and up to 3 sample extractions.

---

## Data flow in detail

### URL lifecycle

```
discovered link
      │
  normalize_url()          ← utils/url.py
      │
  enqueue_url()            ← checks for duplicates via unique constraint on urls.url
      │
  dequeue_next_url()       ← highest priority pending, marks in_progress
      │
  check_and_wait()         ← depth, domain, robots, rate limit
      │
  CrawlEngine.fetch()      ← returns PageContent
      │
  NavigatorAgent.decide()  ← returns NavigatorDecision
      │
  save_page()              ← dedup by content_hash; returns page_id
      │
  mark_url_done()
      │
  ExtractorAgent.extract() ← if relevance >= 0.2
      │
  save_extraction()        ← linked to page_id + session_id
```

### LLM call flow

```
structured_call(model, system_prompt, user_content, response_schema)
      │
  inject schema JSON into system prompt
      │
  chunk_text() if content > MAX_CONTENT_TOKENS (7168)
      │
  check _LLMCache — return cached result if hit
      │
  raw_call() with exponential backoff
      │
  strip markdown fences from response
      │
  extract JSON object/array
      │
  try json.loads() → schema.model_validate()
      │── success → cache result, return
      │── JSONDecodeError → strip trailing commas, retry
      │── all retries fail → fall back to navigator_model if extractor used
```

---

## Concurrency model

Everything is **async** end-to-end. The crawl loop is a single `async` coroutine — pages are fetched one at a time (no parallel fetches by default) to respect rate limits. The rate limiter uses `asyncio.sleep()` rather than `time.sleep()`, so the event loop stays responsive.

Playwright runs inside `asyncio.wait_for()` with an absolute deadline to prevent browser hangs from blocking the loop indefinitely.

---

## Session resumability

Each crawl session is stored in the `crawl_sessions` table. The URL queue is persisted row-by-row in `urls`. On `--resume`:

1. The most recent session with `status = running | paused` and pending URLs is selected
2. `resume_session()` resets any stale `in_progress` rows to `pending`
3. The workflow re-enters its loop and picks up from the remaining queue

The LangGraph workflow adds SQLite-backed checkpointing on top of this, storing the full graph state so execution resumes from the last completed node, not just from the next URL.
