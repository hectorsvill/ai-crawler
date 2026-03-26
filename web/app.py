"""
web/app.py — FastAPI application for the AI Crawler Dashboard.

Startup: initialises the shared SQLite database (same file as the CLI),
creates the FTS5 virtual table, and mounts static files + Jinja2 templates.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import load_config
from storage.db import ensure_fts_and_logs, init_db

_BASE = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_env = os.environ.get("CRAWLER_CONFIG")
    config = load_config(Path(config_env) if config_env else None)
    app.state.config = config
    app.state.db_engine = await init_db(config.storage.db_path)
    await ensure_fts_and_logs(app.state.db_engine)
    app.state.running_crawls: dict[str, object] = {}  # session_id -> asyncio.Task
    yield
    if app.state.db_engine:
        await app.state.db_engine.dispose()


app = FastAPI(title="AI Crawler Dashboard", lifespan=lifespan)

_static = _BASE / "static"
_static.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static)), name="static")

import json as _json

templates = Jinja2Templates(directory=str(_BASE / "templates"))
templates.env.filters["tojson"] = lambda v: _json.dumps(v, ensure_ascii=False)

# Register routers (imported after app is defined to avoid circular imports)
from web.routes import api, crawl, dashboard, search, sessions  # noqa: E402

app.include_router(dashboard.router)
app.include_router(sessions.router)
app.include_router(search.router)
app.include_router(crawl.router)
app.include_router(api.router, prefix="/api")
