# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Always use the project virtual environment:
```bash
source .venv/bin/activate
# or prefix commands directly:
.venv/bin/python main.py ...
```

Python version: 3.12. The venv is at `ai-crawler/.venv/`.

## Common Commands

```bash
# Run a crawl
.venv/bin/python main.py crawl --goal "..." --start-url "https://example.com" --workflow simple

# List sessions
.venv/bin/python main.py list-sessions

# Resume latest session
.venv/bin/python main.py crawl --goal "..." --start-url "..." --resume

# Resume by session ID
.venv/bin/python main.py resume <session-id>

# Export results
.venv/bin/python main.py export <session-id> --output results.json

# Syntax-check all Python files
.venv/bin/python -c "import ast, pathlib; [ast.parse(f.read_text()) for f in pathlib.Path('.').rglob('*.py') if '.venv' not in str(f)]"

# Run the test suite (no Ollama/network needed)
.venv/bin/python -m pytest tests/ -v

# Run a single test file
.venv/bin/python -m pytest tests/test_url_utils.py -v
```

## Git Workflow

This project uses git for version control. Always track changes:

```bash
# Check current status
git status

# Stage and commit source changes (never commit .venv, *.db, .env)
git add <files>
git commit -m "Short description of change"

# View history
git log --oneline -10
```

**Rules:**
- Commit after every meaningful change (bug fix, new feature, new tests).
- Never commit `.venv/`, `*.db`, `*.log`, or `.env` files — all excluded by `.gitignore`.
- Write commit messages in the imperative mood: "Fix domain filter edge case", not "Fixed…".

## Architecture

The system is a layered pipeline: **CLI → Router → Workflow → Agents → Crawler → Storage**.

### Data flow

1. `main.py` parses CLI args and calls `workflows/router.py` which makes one LLM call to classify the goal into `simple | langgraph | crewai`.
2. The selected workflow drives the crawl loop. All three workflows share the same crawler, agents, and storage — they differ only in orchestration.
3. The **Navigator agent** (`agents/navigator.py`) scores page relevance and returns prioritized links. The **Extractor agent** (`agents/extractor.py`) produces structured JSON from page markdown. Both call `llm/client.py`.
4. `crawler/engine.py` fetches pages: aiohttp for static, Playwright for JS-heavy. It auto-upgrades to Playwright if static content is below 500 chars. `crawl4ai` is used first if available.
5. Before every fetch, `crawler/respectful.py` enforces robots.txt, rate limits (token bucket per domain), domain allow/deny (glob patterns), and depth limits.
6. Results persist to SQLite via `storage/db.py`. URL queue, visited pages (SHA-256 dedup), extracted data, and session metadata are all in the same DB file (`crawl_data.db` by default).

### Workflow modes

| Mode | When | Key file |
|------|------|----------|
| `simple` | Few pages, direct extraction | `workflows/simple.py` — single async loop |
| `langgraph` | Deep stateful exploration | `workflows/langgraph_flow.py` — `StateGraph` with SQLite checkpointing for resume |
| `crewai` | Multi-role synthesis | `workflows/crewai_flow.py` — 4-agent crew (Navigator, Extractor, Researcher, Summarizer) |

Both `langgraph` and `crewai` workflows **fall back silently to `simple`** if their packages are not installed.

### Key Pydantic models (in `storage/models.py`)

All inter-component data uses these — never raw dicts:
- `PageContent` — fetched page passed from crawler to agents
- `NavigatorDecision` — returned by Navigator agent (relevance score, links, action)
- `ExtractionResult` — returned by Extractor agent (data dict, schema, confidence)
- `URLItem` — queue entry
- `SessionStats` — live stats for progress display

### LLM client (`llm/client.py`)

- `OllamaClient.structured_call()` injects the Pydantic model's JSON schema into the system prompt and validates the response.
- Responses are cached in-memory keyed by `(content_hash, goal, model)` — avoid passing empty `content_hash` or caching won't work.
- Content is chunked via `chunk_text()` before sending; `MAX_CONTENT_TOKENS = 7168` (8192 context minus 1024 overhead).
- Retries use exponential backoff; if the extractor model fails all retries it falls back to the navigator model.

### Configuration

Config merges in this order (later wins): `default_config.yaml` → user YAML (`--config`) → `CRAWLER_*` env vars (e.g. `CRAWLER_OLLAMA__BASE_URL`).

### Important version notes

- **LangGraph 1.x**: checkpointer is passed to `graph.compile(checkpointer=...)`, not via a `with_checkpointer()` method.
- **CrewAI 1.x**: `llm=` on `Agent` must be a `crewai.LLM` object, not a plain dict.
- **tiktoken** uses `cl100k_base` as an approximation for all Ollama models.
