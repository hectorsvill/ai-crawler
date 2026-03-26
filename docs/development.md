# Development Guide

How to extend, test, and maintain the ai-crawler.

---

## Project layout

```
ai-crawler/
├── main.py                 CLI entry point (Typer)
├── config.py               Pydantic settings + YAML loader
├── default_config.yaml     Shipped defaults
├── requirements.txt
│
├── llm/
│   └── client.py           OllamaClient + LLMClient
│
├── crawler/
│   ├── engine.py           CrawlEngine — aiohttp + Playwright + crawl4ai
│   ├── robots.py           RobotsCache — robots.txt per-domain TTL cache
│   └── respectful.py       Rate limiter, UA rotation, domain filter
│
├── agents/
│   ├── navigator.py        NavigatorAgent — link scoring + action
│   └── extractor.py        ExtractorAgent — structured JSON extraction
│
├── workflows/
│   ├── router.py           WorkflowRouter — goal → mode classification
│   ├── simple.py           Simple single-loop workflow
│   ├── langgraph_flow.py   LangGraph stateful workflow
│   └── crewai_flow.py      CrewAI multi-agent workflow
│
├── storage/
│   ├── models.py           SQLAlchemy ORM + Pydantic data models
│   └── db.py               Async database helpers
│
├── utils/
│   ├── url.py              URL normalization
│   └── progress.py         Rich live display + summary
│
├── tests/                  pytest test suite (no Ollama required)
├── docs/                   This documentation
└── use_cases/              Runnable example scripts
```

---

## Running tests

The test suite is entirely offline — no Ollama instance, no network access needed.

```bash
# Run all tests
.venv/bin/python -m pytest tests/ -v

# Run a specific file
.venv/bin/python -m pytest tests/test_llm_client.py -v

# Run a specific test
.venv/bin/python -m pytest tests/test_models.py::TestNavigatorDecision -v

# With coverage (install pytest-cov first)
.venv/bin/python -m pytest tests/ --cov=. --cov-report=term-missing
```

### What the tests cover

| Test file | What it tests |
|-----------|--------------|
| `test_llm_client.py` | Token counting, chunking, JSON fence stripping, LLM cache |
| `test_models.py` | Pydantic validators: score clamping, action validation |
| `test_crawler_engine.py` | Link extraction, title parsing, HTML→Markdown |
| `test_respectful.py` | Domain filter (allowlist, denylist, wildcard patterns) |
| `test_agents.py` | Link filtering, history summarization, extraction merging |
| `test_storage.py` | Content hash determinism |
| `test_url_utils.py` | URL normalization edge cases |
| `test_use_cases.py` | Public API surface (LLMClient, CrawlEngine, WorkflowRouter) |

### Writing a new test

Tests live in `tests/`. Use `pytest` conventions; async tests need `@pytest.mark.asyncio`.

```python
# tests/test_my_feature.py
import pytest
from my_module import my_function

class TestMyFeature:
    def test_basic_case(self):
        result = my_function("input")
        assert result == "expected"

    @pytest.mark.asyncio
    async def test_async_case(self):
        result = await async_function()
        assert result is not None
```

---

## Adding a new workflow

1. Create `workflows/my_workflow.py` with an import guard for any optional dependency:

```python
"""workflows/my_workflow.py — Description."""
from __future__ import annotations

try:
    import some_optional_package
    HAS_OPTIONAL = True
except ImportError:
    HAS_OPTIONAL = False

from config import AppConfig

async def run_my_workflow(
    goal: str,
    start_urls: list[str],
    config: AppConfig,
    session_id: str | None = None,
) -> list[dict]:
    if not HAS_OPTIONAL:
        from workflows.simple import run_simple
        return await run_simple(goal, start_urls, config, session_id)

    # ... your implementation
```

2. Register the new mode in `workflows/router.py` — add it to `WorkflowMode` and the router prompt.

3. Add dispatch in `main.py`'s `crawl` command:
```python
elif workflow_mode == "my_workflow":
    from workflows.my_workflow import run_my_workflow
    extractions = asyncio.run(run_my_workflow(goal, start_urls, cfg, session_id))
```

4. Add the new mode to the `--workflow` option choices in `main.py`.

---

## Adding a new agent

Agents are simple async classes that call `OllamaClient.structured_call()`. To add one:

1. Define a Pydantic response model in `storage/models.py` (or locally):

```python
class MyAgentResult(BaseModel):
    field_one: str
    score: float = 0.0
```

2. Create `agents/my_agent.py`:

```python
"""agents/my_agent.py — Description."""
from __future__ import annotations
from storage.models import MyAgentResult

MY_SYSTEM_PROMPT = """You are a ... Respond ONLY with valid JSON matching this schema: {schema}"""

MY_USER_TEMPLATE = """## Goal\n{goal}\n\n## Content\n{content}"""

class MyAgent:
    def __init__(self, llm_client=None):
        if llm_client is None:
            from llm.client import OllamaClient
            from config import load_config
            cfg = load_config()
            llm_client = OllamaClient(cfg.ollama)
        self.llm = llm_client

    async def run(self, content: str, goal: str) -> MyAgentResult:
        from llm.client import chunk_text
        chunks = chunk_text(content)
        user_content = MY_USER_TEMPLATE.format(goal=goal, content=chunks[0])
        return await self.llm.structured_call(
            model=self.llm.extractor_model,
            system_prompt=MY_SYSTEM_PROMPT,
            user_content=user_content,
            response_schema=MyAgentResult,
        )
```

3. Write tests in `tests/test_my_agent.py` that mock the LLM call.

---

## Modifying LLM prompts

All prompts are module-level string constants at the top of each agent file, making them easy to find and tune:

- Navigator prompt: `agents/navigator.py` → `NAVIGATOR_SYSTEM_PROMPT`, `NAVIGATOR_USER_TEMPLATE`
- Extractor prompt: `agents/extractor.py` → `EXTRACTOR_SYSTEM_PROMPT`, `EXTRACTOR_USER_TEMPLATE`
- Router prompt: `workflows/router.py` → `ROUTER_SYSTEM_PROMPT`, `ROUTER_USER_TEMPLATE`

Prompt engineering tips:
- End system prompts with `Respond ONLY with valid JSON matching this schema:` to reduce prose leakage
- Use few-shot examples inside the prompt for unusual schemas
- If a model consistently returns low-confidence results, try a `schema_hint` in the extractor (the `--model` flag or `schema_hint` parameter in `ExtractorAgent.extract()`)

---

## Extending the database schema

1. Add columns to the relevant ORM model in `storage/models.py`
2. Update `db.py` functions that read/write that model
3. Delete `crawl_data.db` locally (or write an Alembic migration for production)
4. Update affected tests

The project uses SQLAlchemy 2.0 async ORM. Schema changes do not need Alembic for development — `init_db()` calls `Base.metadata.create_all()` at startup, which adds new tables but does not alter existing ones.

---

## Adding a new CLI command

Commands are defined with Typer in `main.py`. Add a new function decorated with `@app.command()`:

```python
@app.command()
def my_command(
    session_id: str = typer.Argument(..., help="Session UUID"),
    output: str = typer.Option("out.json", "--output", "-o"),
):
    """One-line description shown in --help."""
    import asyncio
    from config import load_config
    from storage.db import init_db, get_all_extractions

    cfg = load_config()
    asyncio.run(init_db(cfg.storage.db_path))
    data = asyncio.run(get_all_extractions(session_id))
    # ... do something with data
```

---

## Code style conventions

- **Type hints everywhere** — all function signatures have return types
- **Async throughout** — no `time.sleep()`, no blocking I/O in the crawl path
- **Pydantic models for inter-component data** — never pass raw dicts across module boundaries
- **Module-level docstrings** — every `.py` file starts with `"""module description."""`
- **Prompt constants at module level** — all LLM prompts are named constants, not inline strings
- **Optional imports with try/except** — `playwright`, `crawl4ai`, `langgraph`, `crewai` all have graceful fallbacks
- **Fail open on external services** — robots.txt errors → allow, LLM parse errors → safe fallback decision

---

## Debugging tips

### Enable verbose logging

```bash
python main.py crawl --goal "..." --start-url "..." --verbose
```

This sets the root logger to `DEBUG`, printing every LLM call, token count, URL queue operation, and rate-limit wait.

### Inspect the database

```bash
# Open the SQLite database
sqlite3 crawl_data.db

-- List sessions
SELECT id, goal, status, started_at FROM crawl_sessions ORDER BY started_at DESC;

-- Count pages per session
SELECT session_id, COUNT(*) FROM urls WHERE status = 'done' GROUP BY session_id;

-- View extracted data
SELECT data, confidence FROM extracted_data LIMIT 5;

-- Check queue
SELECT url, priority, status, depth FROM urls WHERE status = 'pending' LIMIT 20;
```

### Test the LLM client in isolation

```python
import asyncio
from config import load_config
from llm.client import LLMClient

async def main():
    cfg = load_config()
    client = LLMClient(cfg.ollama)
    print(await client.check_health())
    print(await client.generate("Say hello."))

asyncio.run(main())
```

### Force a single-page extraction

```bash
python main.py crawl \
  --goal "Extract all text from this page" \
  --start-url "https://example.com" \
  --workflow simple \
  --max-pages 1 \
  --max-depth 0 \
  --verbose
```
