# AI Web Crawler — Usage Guide

> **See also:**
> - [`architecture.md`](architecture.md) — how the pipeline works internally
> - [`configuration.md`](configuration.md) — every config field explained
> - [`development.md`](development.md) — adding workflows, agents, testing

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Core Concepts](#2-core-concepts)
3. [CLI Reference](#3-cli-reference)
4. [Workflow Modes](#4-workflow-modes)
5. [Configuration](#5-configuration)
6. [Agents](#6-agents)
7. [Python API](#7-python-api)
8. [Real-World Use Cases](#8-real-world-use-cases)
9. [Output & Export](#9-output--export)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Quick Start

```bash
# 1. Activate the virtual environment
source .venv/bin/activate

# 2. Make sure Ollama is running with at least one model
ollama serve &
ollama pull qwen2.5:7b

# 3. Run your first crawl
python main.py crawl \
  --goal "Extract the product name, price, and description" \
  --start-url "https://example.com/product" \
  --workflow simple \
  --max-pages 1
```

Results are printed to the terminal and stored in `crawl_data.db`.

---

## 2. Core Concepts

### Goal
A natural-language description of **what you want to extract**.
The goal drives every decision the crawler makes — which links to follow,
what to extract, and when to stop.

Good goals are specific:
- ✓ `"Extract company name, funding amount, and investor names from TechCrunch articles about Series B rounds"`
- ✗ `"Get info from the web"`

### Workflow
Three execution modes exist (see §4). The router picks one automatically,
or you force it with `--workflow`.

### Session
Every crawl run creates a **session** with a UUID. Sessions are persisted in
SQLite so you can resume interrupted crawls or export their data later.

### Agents
Two LLM-powered agents run inside every workflow:

| Agent | Model (default) | Role |
|-------|----------------|------|
| Navigator | `qwen2.5:7b` | Scores page relevance, selects which links to follow |
| Extractor | `qwen2.5:7b` | Pulls structured JSON from page markdown |

---

## 3. CLI Reference

### `crawl` — Start or resume a crawl

```
python main.py crawl [OPTIONS]
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--goal` | `-g` | str | required | Natural language crawl goal |
| `--start-url` | `-u` | str (repeatable) | required | Seed URL(s) |
| `--workflow` | `-w` | str | `auto` | `auto \| simple \| langgraph \| crewai` |
| `--resume` | | flag | false | Resume the most recent session with pending URLs |
| `--max-pages` | | int | config | Override maximum pages per session |
| `--max-depth` | | int | config | Override maximum link depth |
| `--model` | | str | config | Override extractor model (e.g. `llama3.1:latest`) |
| `--config` | `-c` | path | — | Path to a custom YAML config file |
| `--verbose` | `-v` | flag | false | Enable DEBUG logging |

**Examples:**

```bash
# Minimal
python main.py crawl -g "Get the homepage title" -u "https://example.com"

# Multiple seed URLs
python main.py crawl \
  -g "Compare pricing plans" \
  -u "https://product-a.com/pricing" \
  -u "https://product-b.com/pricing" \
  --workflow simple

# Override model and limits
python main.py crawl \
  -g "Deep research on AI safety" \
  -u "https://en.wikipedia.org/wiki/AI_safety" \
  --workflow langgraph \
  --max-pages 50 \
  --max-depth 3 \
  --model llama3.1:latest

# Resume last session
python main.py crawl -g "..." -u "..." --resume
```

---

### `list-sessions` — Show recent crawl sessions

```
python main.py list-sessions [--limit N]
```

Displays a Rich table with session IDs, goals, workflow, status, and page counts.

```bash
python main.py list-sessions --limit 5
```

---

### `resume` — Resume a specific session by ID

```
python main.py resume <SESSION_ID> [OPTIONS]
```

```bash
python main.py resume 41d0a180-9c1f-4673-b9bd-85cbd4856b5e \
  --max-pages 20
```

---

### `export` — Export extracted data to JSON

```
python main.py export <SESSION_ID> --output <FILE>
```

```bash
python main.py export 41d0a180-9c1f-4673-b9bd-85cbd4856b5e \
  --output results.json
```

Output format:
```json
[
  {
    "data": { "company": "TechSun Inc", "funding": "$50M" },
    "confidence": 0.95,
    "schema": "auto"
  }
]
```

---

## 4. Workflow Modes

### `simple` — Direct extraction (default for shallow tasks)

Best for: single pages, price checks, contact lookups, scraping a known URL.

```
fetch → navigate → extract → store → (repeat until max_pages or queue empty)
```

- Single async loop, no state persistence between restarts
- Fastest startup, lowest overhead
- Use `--max-pages 1` for one-shot extraction

### `langgraph` — Stateful exploration

Best for: knowledge base building, documentation crawling, multi-hop research.

```
StateGraph: fetch → navigate → [extract?] → store → decide_next → fetch → …
```

- SQLite checkpointing — restarts continue from the last checkpoint
- Conditional edges skip extraction for low-relevance pages
- Navigator's priority queue drives which links are followed next

### `crewai` — Multi-role synthesis

Best for: competitive analysis, lead generation, cross-domain research.

```
Navigator Agent → Extractor Agent → Researcher Agent → Summarizer Agent
```

- 4-agent crew collaborates to produce a synthesized report
- Falls back silently to `simple` if crewai is not installed

### `auto` — Let the router decide

The router makes one LLM call to classify the goal:

| Goal characteristics | Selected workflow |
|----------------------|------------------|
| Single URL, one fact to extract | `simple` |
| Multi-page research, "knowledge base", "all articles" | `langgraph` |
| "leads", "outreach", "competitive analysis", synthesis | `crewai` |

---

## 5. Configuration

### Config file hierarchy (later wins)

```
default_config.yaml  →  your_config.yaml  →  CRAWLER_* env vars
```

### Key settings

```yaml
ollama:
  base_url: "http://localhost:11434"
  navigator_model: "qwen2.5:7b"   # smaller/faster = cheaper navigation
  extractor_model: "qwen2.5:7b"   # larger = better structured extraction
  router_model: "qwen2.5:7b"
  timeout: 120                    # seconds per LLM call
  max_retries: 3

crawl:
  max_depth: 5
  max_pages: 500
  rate_limit_per_domain: 2        # requests per second
  delay_range: [1.0, 3.0]         # random jitter between requests (seconds)
  domain_allowlist: []            # restrict to these domains (glob patterns)
  domain_denylist: []             # always block these domains
  use_playwright_for: []          # force JS rendering for these domains

storage:
  db_path: "crawl_data.db"
```

### Environment variable overrides

```bash
export CRAWLER_OLLAMA__BASE_URL="http://remote-server:11434"
export CRAWLER_CRAWL__MAX_PAGES=1000
export CRAWLER_CRAWL__RATE_LIMIT_PER_DOMAIN=1
export CRAWLER_STORAGE__DB_PATH="/data/my_crawl.db"
```

### Custom config file

```bash
cp default_config.yaml my_config.yaml
# edit my_config.yaml …
python main.py crawl --config my_config.yaml --goal "..." --start-url "..."
```

---

## 6. Agents

### NavigatorAgent

Decides which links to follow and how relevant the current page is.

```python
from agents.navigator import NavigatorAgent

nav = NavigatorAgent()          # uses default config/model
decision = await nav.decide(
    url="https://example.com/blog",
    markdown=page_content,
    goal="Find solar energy startup funding rounds",
    links=["https://example.com/post/1", "https://example.com/about"],
    history_summary="Visited homepage and about page.",
)

print(decision.relevance_score)    # float 0–1
print(decision.action)             # "deepen" | "backtrack" | "complete"
for link in decision.links_to_follow:
    print(link.url, link.priority)
```

### ExtractorAgent

Extracts structured JSON from page content.

```python
from agents.extractor import ExtractorAgent

ext = ExtractorAgent()          # uses default config/model
result = await ext.extract(
    url="https://example.com/startup",
    markdown=page_content,
    goal="Extract company name, funding amount, and investors",
)

print(result.data)          # dict with extracted fields
print(result.confidence)    # float 0–1
print(result.schema_used)   # inferred schema name
```

For **long pages** (> context window), use `extract_chunks` to process all
content and merge:

```python
result = await ext.extract_chunks(url, markdown, goal)
```

---

## 7. Python API

### LLMClient — Simple interface

```python
import asyncio
from config import Settings
from llm.client import LLMClient
from pydantic import BaseModel

class Company(BaseModel):
    name: str
    founded: int
    is_public: bool

async def main():
    # Pass an OllamaConfig object (recommended) or keyword args
    client = LLMClient(Settings().ollama)

    # Check Ollama is up
    assert await client.check_health()

    # Plain text
    text = await client.generate("Summarize AI safety in 2 sentences.")
    print(text)

    # Structured JSON — returns a validated Pydantic instance
    company = await client.generate_json(
        "Company: OpenAI. Return name, founding year, and whether it's public.",
        response_model=Company,
    )
    print(company.name, company.founded)   # OpenAI 2015

    # Token utilities
    n = client.count_tokens("some text")
    chunks = client.chunk_text(long_text, max_tokens=500)

asyncio.run(main())
```

### CrawlEngine — Fetch pages programmatically

```python
from config import Settings
from crawler.engine import CrawlEngine

# Pass a CrawlConfig object (uses rotating UAs, respects config)
async with CrawlEngine(Settings().crawl) as engine:
    page = await engine.fetch("https://example.com")
    print(page.markdown)       # extracted text
    print(page.links)          # list of absolute URLs
    print(page.content_hash)   # SHA-256 of content
    print(page.status_code)    # HTTP status (200 for crawl4ai/playwright paths)

    allowed = await engine.is_allowed("https://example.com/path")
    print(allowed)             # True / False (robots.txt)
```

### WorkflowRouter — Classify a goal

```python
from workflows.router import WorkflowRouter

async def main():
    router = WorkflowRouter()
    decision = await router.select("Build a knowledge base on AI safety")
    print(decision.workflow)     # "langgraph"
    print(decision.reasoning)
    print(decision.complexity)   # "low" | "medium" | "high"

asyncio.run(main())
```

### Database helpers

```python
from storage.db import init_db, get_all_extractions

async def main():
    await init_db("crawl_data.db")
    records = await get_all_extractions("your-session-id")
    for r in records:
        print(r["data"], r["confidence"])

asyncio.run(main())
```

---

## 8. Real-World Use Cases

See [use_cases/](../use_cases/) for runnable example scripts.

| Use Case | Script | Workflow |
|----------|--------|----------|
| Wikipedia research assistant | `use_cases/wikipedia_research.py` | langgraph |
| GitHub trending repos | `use_cases/github_trending.py` | simple |
| HackerNews top stories | `use_cases/hackernews_digest.py` | simple |

---

## 9. Output & Export

### Terminal output

After every crawl the terminal shows:
- Live progress panel (pages, queue size, elapsed time)
- Token usage summary (per model)
- Up to 3 sample extracted records

### JSON export

```bash
python main.py export <SESSION_ID> --output results.json
```

### Programmatic access

```python
from storage.db import init_db, get_all_extractions
import json

async def export(session_id, path):
    await init_db("crawl_data.db")
    data = await get_all_extractions(session_id)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
```

---

## 10. Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `Cannot reach Ollama` | Ollama not started | `ollama serve` |
| `Queue empty` on resume | URL already `done` in DB; no pending URLs | Start a new session or remove `crawl_data.db` |
| Extraction confidence 0.0 | LLM returned malformed JSON | Check `--verbose` logs; try a larger model |
| Playwright not found | Browser not installed | `playwright install chromium` |
| Very slow crawl | LLM timeout per page | Lower `max_pages` or use a smaller model |
| LangGraph / CrewAI missing | Optional packages not installed | `pip install langgraph crewai[tools]` |
| Duplicate URLs in queue | URL normalization edge case | All URLs are canonicalized; check `utils/url.py` |
