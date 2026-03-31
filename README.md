# AI Web Crawler

A production-ready, modular intelligent web crawler powered by local LLMs via [Ollama](https://ollama.ai).

Built for AMD GPU (ROCm) workstations running Ollama at `http://localhost:11434`.
Supports three workflow modes: **simple**, **LangGraph** (stateful), and **CrewAI** (multi-agent).

---

## Prerequisites

- **Python 3.10+** (3.12 recommended; `crewai` requires Python ≤3.13)
- **Ollama** installed and running: [https://ollama.ai/download](https://ollama.ai/download)
- **ROCm** (for AMD GPU acceleration) — Ollama handles this automatically on Linux

---

## 1. Clone & Install

```bash
git clone https://github.com/hectorsvill/ai-crawler.git
cd ai-crawler

# Create a virtual environment
# Use python3.12 if available; python3 works on 3.13/3.14 (crewai will be skipped)
python3.12 -m venv .venv   # preferred
# python3 -m venv .venv    # fallback if python3.12 is not in PATH

source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Note: on Python 3.14, crewai will fail to install (it requires ≤3.13).
# All other features work. crewai mode degrades silently to simple mode.

# Install Playwright browser (for JS-heavy sites)
playwright install chromium
```

### Post-clone verification

After installing, confirm everything is wired up correctly:

```bash
# Syntax-check all source files (no Ollama needed)
python -c "import ast, pathlib; [ast.parse(f.read_text()) for f in pathlib.Path('.').rglob('*.py') if '.venv' not in str(f)]"

# Run the full test suite (no Ollama or network needed)
pip install pytest pytest-asyncio   # one-time
pytest tests/ -v
# Expected: 143 passed
```

---

## 2. Pull Ollama Models

```bash
# Start Ollama (if not already running)
ollama serve &

# Required: extraction model (~4GB)
ollama pull qwen2.5:7b

# Required: fast navigation model (~1GB)
ollama pull qwen2.5:1.5b

# Verify models are available
ollama list
```

**Alternative models** (update `default_config.yaml` to use them):

```bash
ollama pull phi4-mini      # fast, efficient (~2.5GB)
ollama pull llama3.2:3b    # good balance (~2GB)
ollama pull mistral:7b     # high quality extraction (~4.4GB)
```

---

## 3. Quick Test

```bash
# Verify everything works — crawls Hacker News and prints top stories
python main.py crawl \
  --goal "Get the top 5 stories with titles and scores" \
  --start-url "https://news.ycombinator.com" \
  --workflow simple \
  --max-pages 3
```

Expected output: a summary table with extracted HN stories and token usage stats.

---

## 4. Usage Examples

### Crypto — Latest Binance Coin Listings

Binance is a JS-heavy site, so Playwright is required. Use a capable model (gemma3 or better)
and set the navigator model via env var since the default small models struggle with structured output.

```bash
CRAWLER_OLLAMA__NAVIGATOR_MODEL=gemma3:latest \
python main.py crawl \
  --goal "Find the latest Binance coin listings and new token launches — extract coin name, symbol, listing date, and description" \
  --start-url "https://www.coingecko.com/en/exchanges/binance" \
  --start-url "https://cryptorank.io/exchanges/binance/new-listings" \
  --workflow simple \
  --max-pages 8 \
  --max-depth 2 \
  --model gemma3:latest
```

> **Tip:** For JS-heavy sites (Binance, CoinGecko), Playwright must be installed:
> ```bash
> playwright install chromium
> ```

---

### Research / Knowledge Base Building

```bash
python main.py crawl \
  --goal "Build a comprehensive knowledge base on solar energy startups in Europe" \
  --start-url "https://solarenergyeurope.org" \
  --start-url "https://www.pv-magazine.com" \
  --workflow langgraph \
  --max-pages 100 \
  --max-depth 3
```

### E-Commerce Price Monitoring

```bash
python main.py crawl \
  --goal "Extract product names, prices, and availability for all laptops" \
  --start-url "https://example-shop.com/laptops" \
  --workflow simple \
  --max-pages 20
```

### Competitive Intelligence

```bash
python main.py crawl \
  --goal "Analyze the features, pricing, and positioning of cloud database providers" \
  --start-url "https://aws.amazon.com/rds/" \
  --start-url "https://cloud.google.com/sql" \
  --start-url "https://azure.microsoft.com/products/azure-sql-database" \
  --workflow crewai \
  --max-pages 50
```

### Lead Generation

```bash
python main.py crawl \
  --goal "Generate a list of B2B SaaS companies in fintech with contact information" \
  --start-url "https://www.crunchbase.com/hub/fintech-startups" \
  --workflow crewai \
  --max-pages 200
```

### Content Aggregation

```bash
python main.py crawl \
  --goal "Collect all blog posts about machine learning published in 2024" \
  --start-url "https://example-blog.com/ml" \
  --workflow langgraph \
  --max-pages 500 \
  --max-depth 5
```

### RAG Dataset Building

```bash
python main.py crawl \
  --goal "Crawl the Python documentation and extract all function signatures and descriptions" \
  --start-url "https://docs.python.org/3/library/" \
  --workflow langgraph \
  --max-pages 300 \
  --max-depth 4
```

---

## 5. Resuming a Crawl

If a crawl is interrupted, resume it using:

```bash
# List recent sessions to find the session ID
python main.py list-sessions

# Resume by session ID
python main.py resume <session-id>

# Or use --resume flag to auto-resume the latest session
python main.py crawl --goal "..." --start-url "..." --resume
```

---

## 6. Exporting Results

```bash
# Export structured JSON extractions (default)
python main.py export <session-id> --output results.json

# Export every crawled page as clean Markdown — ready for RAG, vector DBs, and LLM pipelines
python main.py export <session-id> --format markdown --output pages.md
```

**JSON output** — one record per page, LLM-extracted structured data:
```json
[{"data": {...}, "confidence": 0.85, "schema": "product_listing"}, ...]
```

**Markdown output** — one section per crawled page, clean and LLM-ready:
```markdown
# Crawled Pages — Session f5dbfea0
*8 pages · exported by ai-crawler*

## pokemon · GitHub Topics · GitHub
**URL:** https://github.com/topics/pokemon
**Fetched:** 2026-03-31T04:41:36

# Search code, repositories, users, issues, pull requests...
Here are 5,866 public repositories matching this topic...
...

---
```

---

## 7. Changing Models

Override models per-run via CLI:

```bash
python main.py crawl \
  --goal "..." \
  --start-url "..." \
  --model mistral:7b  # override extractor model
```

Or edit `default_config.yaml`:

```yaml
ollama:
  navigator_model: "qwen2.5:1.5b"   # fast decisions
  extractor_model: "qwen2.5:7b"     # quality extraction
  router_model: "qwen2.5:7b"        # workflow selection
```

---

## 8. Custom Configuration

Copy and modify the default config:

```bash
cp default_config.yaml my_config.yaml
```

Then pass it to any command:

```bash
python main.py crawl --goal "..." --start-url "..." --config my_config.yaml
```

Key settings:

```yaml
crawl:
  max_depth: 5              # how deep to follow links
  max_pages: 500            # session page limit
  rate_limit_per_domain: 2  # requests/sec per domain
  delay_range: [1.0, 3.0]   # random delay between requests (seconds)
  domain_allowlist:         # leave empty for no restriction
    - "*.example.com"
  domain_denylist:          # always block these
    - "ads.example.com"
  use_playwright_for:       # force JS rendering for these domains
    - "*.react-app.com"
    - "*.nextjs-site.io"
```

---

## 9. Architecture Overview

```
Goal → Router → Workflow
                  ├── Simple:     fetch → navigate → extract → store (loop)
                  ├── LangGraph:  StateGraph with conditional edges + SQLite checkpointing
                  └── CrewAI:     Navigator + Extractor + Researcher + Summarizer agents
```

Each workflow uses:
- **Navigator agent** (`qwen2.5:1.5b`) — scores relevance, selects links
- **Extractor agent** (`qwen2.5:7b`) — extracts structured JSON from pages
- **Respectful crawling** — robots.txt, rate limits, domain filters, depth limits

---

## 10. Environment Variable Overrides

Override any config setting without modifying YAML:

```bash
export CRAWLER_OLLAMA__BASE_URL="http://remote-server:11434"
export CRAWLER_CRAWL__MAX_PAGES=1000
export CRAWLER_STORAGE__DB_PATH="/data/crawl.db"
```

---

## 11. Running the Test Suite

Tests require no Ollama instance or network connection — they cover all pure-Python logic.

```bash
# Install test dependencies (one-time)
pip install pytest pytest-asyncio

# Run all tests
pytest tests/ -v

# Run a specific module
pytest tests/test_url_utils.py -v
pytest tests/test_models.py -v
```

Test coverage includes:
| Module | What's tested |
|--------|---------------|
| `utils/url.py` | URL normalization, fragment stripping, port removal, query sorting |
| `storage/models.py` | Score clamping, action validation, Pydantic validators |
| `llm/client.py` | Token counting, chunking, LLM cache, JSON fence stripping |
| `crawler/engine.py` | Link extraction, title parsing, HTML→Markdown conversion |
| `crawler/respectful.py` | Domain filter (allowlist/denylist/wildcard), rate limiter, Playwright detector |
| `agents/` | Link filtering, history summarization, extraction merging |
| `storage/db.py` | Content hash determinism and correctness |

---

## 12. Recent Improvements

| Area | Change |
|------|--------|
| **URL normalization** | All URLs are canonicalized before queuing (lowercase, strip fragments/default ports/trailing slashes, sort query params) preventing duplicate crawls of semantically identical URLs |
| **Score clamping** | LLM-returned scores outside [0, 1] are clamped rather than rejected, preventing crawl interruption from out-of-range model output |
| **Action validation** | `NavigatorDecision.action` values not in `{deepen, backtrack, complete}` safely default to `deepen` |
| **Multi-chunk extraction** | Long pages now use `extract_chunks` so data spread across multiple context windows is captured and merged |
| **Domain filter** | `*.example.com` patterns now correctly match `example.com` itself (parent domain) |
| **Playwright timeout** | Added outer `asyncio.wait_for` around the full browser lifecycle to prevent hang if the browser process stalls |
| **MIME logging** | Skipped non-HTML responses are now logged at INFO level with the actual content-type |
| **robots.txt logging** | Distinguishes 404 (no robots.txt) from network errors vs unexpected HTTP status codes |

---

## 13. Troubleshooting

**Ollama not responding:**
```bash
ollama serve  # start the Ollama server
curl http://localhost:11434/api/tags  # verify it's running
```

**Playwright browser not found:**
```bash
playwright install chromium
playwright install-deps  # install system dependencies
```

**LangGraph / CrewAI not installed:**
The crawler degrades gracefully to `simple` mode. Install with:
```bash
pip install langgraph crewai[tools]
```

**`crewai` fails to install (Python 3.14+):**
`crewai>=0.80` requires Python ≤3.13. Either install Python 3.12 (`brew install python@3.12`)
and recreate the venv, or use the crawler without crewai — it falls back to `simple` mode automatically.
```bash
# Recreate venv with Python 3.12
brew install python@3.12
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**AMD GPU not utilized:**
```bash
# Check ROCm is detected by Ollama
ollama run qwen2.5:1.5b "hello"
# Watch GPU usage
watch -n1 rocm-smi
```
