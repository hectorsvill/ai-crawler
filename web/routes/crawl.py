"""web/routes/crawl.py — New crawl form and goal-analysis partial."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from web.app import templates

router = APIRouter(tags=["crawl"])


@router.get("/crawl/new")
async def new_crawl_page(request: Request):
    config = request.app.state.config
    models: list[str] = []
    try:
        import aiohttp
        async with aiohttp.ClientSession() as sess:
            async with sess.get(
                f"{config.ollama.base_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=3),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [m["name"] for m in data.get("models", [])]
    except Exception:
        pass

    if not models:
        models = list({config.ollama.navigator_model, config.ollama.extractor_model})

    return templates.TemplateResponse(
        request,
        "new_crawl.html",
        {
            "config": config,
            "models": models,
            "default_nav_model": config.ollama.navigator_model,
            "default_ext_model": config.ollama.extractor_model,
        },
    )


@router.post("/crawl/analyze", response_class=HTMLResponse)
async def analyze_goal(request: Request):
    form = await request.form()
    goal = str(form.get("goal", "")).strip()
    if not goal:
        return HTMLResponse('<p class="text-gray-500 text-sm">Enter a goal to analyse.</p>')
    config = request.app.state.config
    try:
        from llm.client import OllamaClient
        from workflows.router import select_workflow
        llm = OllamaClient(config.ollama)
        workflow, reasoning = await select_workflow(goal, llm)
    except Exception as exc:
        workflow, reasoning = "simple", f"Router unavailable ({exc}) — defaulting to simple."

    badge_color = {
        "simple":   "text-cyan-400 border-cyan-700 bg-cyan-900/30",
        "langgraph": "text-purple-400 border-purple-700 bg-purple-900/30",
        "crewai":   "text-amber-400 border-amber-700 bg-amber-900/30",
    }.get(workflow, "text-gray-400 border-gray-700 bg-gray-900/30")

    return HTMLResponse(f"""
<div class="p-3 rounded border border-gray-700 bg-[#12121a] text-sm space-y-1">
  <div class="flex items-center gap-2">
    <span class="text-gray-400">Recommended workflow:</span>
    <span class="px-2 py-0.5 rounded border text-xs font-mono {badge_color}">{workflow}</span>
  </div>
  <p class="text-gray-400 text-xs leading-relaxed">{reasoning}</p>
</div>
""")
