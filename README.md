# AI Web Crawler

A production-ready, modular intelligent web crawler powered by local LLMs via [Ollama](https://ollama.ai).

Built for AMD GPU (ROCm) workstations running Ollama at `http://localhost:11434`.
Supports three workflow modes: **simple**, **LangGraph** (stateful), and **CrewAI** (multi-agent).

---

## Prerequisites

- **Python 3.12+**
- **Ollama** installed and running: [https://ollama.ai/download](https://ollama.ai/download)
- **ROCm** (for AMD GPU acceleration) — Ollama handles this automatically on Linux

---

## 1. Pull Ollama Models

```bash
# Primary extraction model (best quality, ~4GB)
ollama pull qwen2.5:7b

# Fast navigation model (~1GB)
ollama pull qwen2.5:1.5b

# Alternative: Microsoft Phi-4 Mini (fast, efficient)
ollama pull phi4-mini

# Alternative: Llama 3.2 3B (good balance)
ollama pull llama3.2:3b

# Alternative: Mistral 7B (high quality extraction)
ollama pull mistral:7b

# Verify Ollama is running
ollama list
```

---

## 2. Install Dependencies

```bash
cd ai-crawler

# Create a virtual environment (recommended)
python3.12 -m venv .venv
source .venv/bin/activate

# Install Python packages
pip install -r requirements.txt

# Install Playwright browser (for JS-heavy sites)
playwright install chromium
```

---

## 3. Usage Examples

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

## 4. Resuming a Crawl

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

## 5. Exporting Results

```bash
# Export all extracted data to JSON
python main.py export <session-id> --output results.json

# The JSON file contains an array of extraction records:
# [{"data": {...}, "confidence": 0.85, "schema": "product_listing"}, ...]
```

---

## 6. Changing Models

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

## 7. Custom Configuration

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

## 8. Architecture Overview

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

## 9. Environment Variable Overrides

Override any config setting without modifying YAML:

```bash
export CRAWLER_OLLAMA__BASE_URL="http://remote-server:11434"
export CRAWLER_CRAWL__MAX_PAGES=1000
export CRAWLER_STORAGE__DB_PATH="/data/crawl.db"
```

---

## 10. Running the Test Suite

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

## 11. Recent Improvements

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

## 12. Troubleshooting

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

**AMD GPU not utilized:**
```bash
# Check ROCm is detected by Ollama
ollama run qwen2.5:1.5b "hello"
# Watch GPU usage
watch -n1 rocm-smi
```
