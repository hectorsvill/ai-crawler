"""
llm/models.py — Live model listing per provider (cached ~10 min) + model validation.

    await list_models("gemini")  ->
      {"provider", "configured", "env_var", "models": [{"id", "name"}], "default_model",
       "role_defaults", "error", "cached", "fetched_at"}

A missing API key returns ``configured: false`` (never raises). Network/API errors
return ``error`` (key-redacted) with an empty model list. Keys are only ever sent in
request headers and are never included in the result.
"""

from __future__ import annotations

import difflib
import time
from datetime import datetime, timezone
from typing import Any

import aiohttp

from llm.providers import (
    DEFAULT_BASE_URLS, DEFAULT_ROLE_MODELS, PROVIDERS, _redact, api_key_for, default_model,
)

CACHE_TTL = 600.0
_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}


def clear_cache() -> None:
    _CACHE.clear()


async def _get_json(session: Any, url: str, headers: dict[str, str], timeout: float) -> dict:
    async with session.get(url, headers=headers,
                           timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
        body = await resp.json(content_type=None)
        if resp.status >= 400:
            msg = body.get("error", body) if isinstance(body, dict) else body
            raise RuntimeError(f"HTTP {resp.status}: {str(msg)[:200]}")
        return body


async def _fetch(provider: str, base: str, key: str | None, session: Any,
                 timeout: float) -> list[dict[str, str]]:
    if provider == "gemini":
        out, token = [], ""
        for _ in range(20):
            url = f"{base}/models?pageSize=1000" + (f"&pageToken={token}" if token else "")
            d = await _get_json(session, url, {"x-goog-api-key": key or ""}, timeout)
            for m in d.get("models", []):
                if "generateContent" in m.get("supportedGenerationMethods", []):
                    mid = m["name"].removeprefix("models/")
                    out.append({"id": mid, "name": m.get("displayName", mid)})
            token = d.get("nextPageToken")
            if not token:
                break
        return out
    if provider == "anthropic":
        out, after = [], None
        headers = {"x-api-key": key or "", "anthropic-version": "2023-06-01"}
        for _ in range(20):
            url = f"{base}/models?limit=1000" + (f"&after_id={after}" if after else "")
            d = await _get_json(session, url, headers, timeout)
            out += [{"id": m["id"], "name": m.get("display_name", m["id"])} for m in d.get("data", [])]
            if not d.get("has_more"):
                break
            after = d.get("last_id")
        return out
    if provider in ("openai", "deepseek"):
        d = await _get_json(session, f"{base}/models", {"Authorization": f"Bearer {key}"}, timeout)
        return [{"id": m["id"], "name": m["id"]} for m in d.get("data", [])]
    if provider == "ollama":
        d = await _get_json(session, f"{base}/api/tags", {}, timeout)
        return [{"id": m["name"], "name": m["name"]} for m in d.get("models", [])]
    raise ValueError(f"Unknown provider '{provider}'")


async def list_models(provider: str, *, base_url: str | None = None, refresh: bool = False,
                      session_factory: Any = None, timeout: float = 15.0) -> dict[str, Any]:
    provider = provider.lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {', '.join(PROVIDERS)}")
    base = (base_url or DEFAULT_BASE_URLS[provider]).rstrip("/")
    result: dict[str, Any] = {
        "provider": provider, "configured": True, "env_var": None, "models": [],
        "default_model": default_model(provider),
        "role_defaults": DEFAULT_ROLE_MODELS.get(provider, {}),
        "error": None, "cached": False, "fetched_at": None,
    }
    key = None
    if provider != "ollama":
        key, env_name = api_key_for(provider)
        result["env_var"] = env_name
        if not key:
            result["configured"] = False
            return result
    ck = (provider, base)
    hit = _CACHE.get(ck)
    if hit and not refresh and time.monotonic() - hit[0] < CACHE_TTL:
        return {**hit[1], "cached": True}
    factory = session_factory or aiohttp.ClientSession
    try:
        async with factory() as session:
            models = await _fetch(provider, base, key, session, timeout)
    except Exception as exc:  # noqa: BLE001
        result["error"] = _redact(f"{type(exc).__name__}: {exc}", key)
        if provider == "ollama":
            result["configured"] = False  # unreachable
        return result
    result["models"] = sorted(models, key=lambda m: m["id"])
    result["fetched_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    _CACHE[ck] = (time.monotonic(), result)
    return dict(result)


def _ids(listing: dict[str, Any]) -> list[str]:
    return [m["id"] for m in listing.get("models", [])]


def check_model(listing: dict[str, Any], model: str) -> dict[str, Any]:
    """Validate *model* against a listing.

    Returns {"valid": True|False|None, "suggestions": [...]} where ``None`` means the
    list could not be fetched (not configured / API error) so nothing was checked.
    """
    ids = _ids(listing)
    if not ids:
        return {"valid": None, "suggestions": [], "reason": listing.get("error")
                or ("provider not configured" if not listing.get("configured") else "empty list")}
    candidates = {model, model.removeprefix("models/")}
    if listing["provider"] == "ollama" and ":" not in model:
        candidates.add(model + ":latest")
    if candidates & set(ids):
        return {"valid": True, "suggestions": []}
    sugg = difflib.get_close_matches(model, ids, n=5, cutoff=0.5)
    if not sugg:
        stem = model.split("-")[0].split(":")[0]
        sugg = [i for i in ids if i.startswith(stem)][:5]
    return {"valid": False, "suggestions": sugg}


async def all_providers_status(refresh: bool = False, with_models: bool = True,
                               session_factory: Any = None) -> list[dict[str, Any]]:
    out = []
    for p in PROVIDERS:
        if with_models:
            out.append(await list_models(p, refresh=refresh, session_factory=session_factory))
        else:
            key, env_name = (None, None) if p == "ollama" else api_key_for(p)
            out.append({"provider": p, "configured": True if p == "ollama" else bool(key),
                        "env_var": env_name, "default_model": default_model(p),
                        "role_defaults": DEFAULT_ROLE_MODELS.get(p, {})})
    return out
