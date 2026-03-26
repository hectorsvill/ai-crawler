# Configuration Reference

All settings live in `default_config.yaml`. You can override any value by:
1. Providing a custom YAML file via `--config my_config.yaml`
2. Setting `CRAWLER_*` environment variables

The merge order is: **defaults → custom YAML → env vars** (later wins).

---

## Full annotated config

```yaml
# ── Ollama / LLM settings ──────────────────────────────────────────────────

ollama:
  # URL of the running Ollama instance
  base_url: "http://localhost:11434"

  # Model used by the Navigator agent — decides which links to follow.
  # Use a fast, small model here (decisions are many, stakes are low).
  navigator_model: "qwen2.5:7b"

  # Model used by the Extractor agent — pulls structured JSON from pages.
  # Use the most capable model you have; quality matters here.
  extractor_model: "qwen2.5:7b"

  # Model used by the WorkflowRouter — classifies the goal once per run.
  router_model: "qwen2.5:7b"

  # Seconds to wait for a single LLM response before raising a timeout.
  # Larger models on slower hardware may need 180–300.
  timeout: 120

  # Number of times to retry a failed LLM call before giving up (or
  # falling back to the navigator model for extractor calls).
  max_retries: 3


# ── Crawl behaviour ─────────────────────────────────────────────────────────

crawl:
  # Maximum link-following depth from the seed URL.
  # depth 0 = seed URL only, depth 1 = direct links from seed, etc.
  max_depth: 5

  # Hard cap on pages visited per session (across all depths).
  max_pages: 500

  # Maximum requests per second per domain (token-bucket rate limiter).
  # Use 1.0 for polite crawling of smaller sites, 2–5 for large platforms.
  rate_limit_per_domain: 2.0

  # Random delay added between requests to the SAME domain [min, max] seconds.
  # Adds human-like variability on top of the rate limit.
  delay_range: [1.0, 3.0]

  # Rotating User-Agent strings sent in HTTP headers.
  # The engine picks one at random for each CrawlEngine instance.
  user_agents:
    - "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ..."
    - "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1) ..."
    # ... add more for better rotation

  # If non-empty, ONLY these domains are crawled.
  # Supports glob patterns. Leave empty [] to allow all domains.
  # *.example.com also matches example.com itself.
  domain_allowlist: []
    # - "*.ycombinator.com"
    # - "news.ycombinator.com"

  # URLs from these domains are skipped entirely.
  # Takes precedence over domain_allowlist if both match.
  domain_denylist: []
    # - "ads.example.com"
    # - "*.doubleclick.net"

  # Domains that require JavaScript rendering (Playwright).
  # Without this list the engine auto-upgrades when static content < 500 chars,
  # but you can force Playwright for specific domains to avoid the extra attempt.
  use_playwright_for: []
    # - "*.react-app.io"
    # - "app.example.com"


# ── Storage ─────────────────────────────────────────────────────────────────

storage:
  # Path to the SQLite database file.
  # Use an absolute path to keep it outside the project directory.
  db_path: "crawl_data.db"


# ── Workflow ─────────────────────────────────────────────────────────────────

workflow:
  # Default workflow when --workflow is not given on the CLI.
  # "auto" makes one LLM call to select simple | langgraph | crewai.
  # Set to "simple" to always use simple mode and skip the routing call.
  default: "auto"
```

---

## Environment variable overrides

Pattern: `CRAWLER_<SECTION>__<KEY>=value`

The section and key names match the YAML keys, uppercased, with `__` (double underscore) as the separator.

```bash
# Ollama
export CRAWLER_OLLAMA__BASE_URL="http://gpu-server:11434"
export CRAWLER_OLLAMA__EXTRACTOR_MODEL="mistral:7b"
export CRAWLER_OLLAMA__TIMEOUT=180

# Crawl limits
export CRAWLER_CRAWL__MAX_PAGES=1000
export CRAWLER_CRAWL__MAX_DEPTH=3
export CRAWLER_CRAWL__RATE_LIMIT_PER_DOMAIN=1

# Storage
export CRAWLER_STORAGE__DB_PATH="/data/crawl.db"

# Workflow
export CRAWLER_WORKFLOW__DEFAULT="simple"
```

---

## Model recommendations

| Model | Best used as | Notes |
|-------|-------------|-------|
| `qwen2.5:7b` | extractor, router | Strong JSON instruction-following; default choice |
| `qwen2.5:1.5b` | navigator | ~3× faster than 7b; navigation quality is acceptable |
| `qwen2.5:14b` | extractor | Higher quality for complex schemas; ~2× slower |
| `mistral:7b` | extractor | Good alternative if qwen isn't available |
| `llama3.2:3b` | navigator | Fast, compact; good enough for link scoring |
| `phi4-mini` | navigator | Microsoft model, very fast on CPU |
| `deepseek-r1:8b` | extractor | Better reasoning for ambiguous schemas |

Use a **small model** for the navigator (many calls, low stakes) and a **large model** for the extractor (fewer calls, output quality matters).

---

## Per-run CLI overrides

These don't require changing any config file:

```bash
# Override extractor model for this run only
python main.py crawl --goal "..." --start-url "..." --model mistral:7b

# Override page and depth limits
python main.py crawl --goal "..." --start-url "..." --max-pages 10 --max-depth 2

# Use a specific workflow (skip router call)
python main.py crawl --goal "..." --start-url "..." --workflow simple

# Load a custom config file
python main.py crawl --goal "..." --start-url "..." --config production.yaml
```

---

## Example configs by use case

### Fast, polite single-page extraction

```yaml
crawl:
  max_depth: 0
  max_pages: 1
  rate_limit_per_domain: 1.0
  delay_range: [2.0, 4.0]
ollama:
  extractor_model: "qwen2.5:7b"
  timeout: 60
```

### Deep research crawl on a single site

```yaml
crawl:
  max_depth: 5
  max_pages: 500
  rate_limit_per_domain: 1.0
  delay_range: [2.0, 5.0]
  domain_allowlist:
    - "*.docs.python.org"
ollama:
  navigator_model: "qwen2.5:1.5b"
  extractor_model: "qwen2.5:7b"
  timeout: 120
workflow:
  default: "langgraph"
```

### Aggressive multi-domain research (own infrastructure)

```yaml
crawl:
  max_depth: 8
  max_pages: 2000
  rate_limit_per_domain: 5.0
  delay_range: [0.2, 0.5]
ollama:
  base_url: "http://gpu-server:11434"
  navigator_model: "llama3.2:3b"
  extractor_model: "qwen2.5:14b"
  timeout: 180
  max_retries: 5
storage:
  db_path: "/data/research.db"
```
