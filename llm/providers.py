"""
llm/providers.py — Provider abstraction for chat LLMs.

Supported providers (plain HTTP via aiohttp; no extra SDKs):
  ollama     local Ollama  /api/chat                 (no key)
  gemini     Google Gemini generateContent           GEMINI_API_KEY or GOOGLE_API_KEY
  anthropic  Anthropic Claude /v1/messages           ANTHROPIC_API_KEY
  openai     OpenAI ChatGPT /v1/chat/completions     OPENAI_API_KEY
  deepseek   DeepSeek (OpenAI-compatible)            DEEPSEEK_API_KEY

Each provider returns an ``LLMResponse`` carrying the text plus the token
counts reported by the API (``estimated=False``). If the API omits usage the
caller falls back to a tiktoken estimate and flags it ``estimated=True``.

API keys are read from the environment only (optionally pre-loaded from an
env file, see ``load_env_file``). They are sent in headers — never in URLs —
and are redacted from every error message.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import aiohttp

PROVIDERS = ("ollama", "gemini", "anthropic", "openai", "deepseek")

# Env var names, in lookup order.
KEY_ENV: dict[str, tuple[str, ...]] = {
    "ollama": (),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
}

# Sensible cheap defaults; override with --model / CRAWLER_LLM__MODEL.
DEFAULT_MODELS: dict[str, str] = {
    "ollama": "qwen2.5:7b",
    "gemini": "gemini-3.5-flash-lite",
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-4.1-mini",
    "deepseek": "deepseek-chat",
}

# Per-role overrides of DEFAULT_MODELS (roles: navigator_model, extractor_model,
# router_model). Extraction benefits from a stronger model; navigation/routing stay cheap.
# Gemini names verified against ListModels + a live call (2026-11-15).
DEFAULT_ROLE_MODELS: dict[str, dict[str, str]] = {
    "gemini": {"extractor_model": "gemini-3.5-flash"},
}


def default_model(provider: str, role: str | None = None) -> str:
    """Default model for *provider* (optionally for a specific role)."""
    if role:
        m = DEFAULT_ROLE_MODELS.get(provider, {}).get(role)
        if m:
            return m
    return DEFAULT_MODELS[provider]


DEFAULT_BASE_URLS: dict[str, str] = {
    "ollama": "http://localhost:11434",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "anthropic": "https://api.anthropic.com/v1",
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
}

DEFAULT_ENV_FILE = Path.home() / ".config" / "ai-crawler" / "env"


class ProviderError(RuntimeError):
    """Raised for provider/API failures. Message is always key-redacted."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class MissingAPIKey(ProviderError):
    pass


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: int = 0
    raw_usage: dict[str, Any] = field(default_factory=dict)

    @property
    def has_usage(self) -> bool:
        return self.prompt_tokens is not None and self.completion_tokens is not None


# ── Env / keys ────────────────────────────────────────────────────────────────

_ENV_LOADED = False


def load_env_file(path: str | Path | None = None) -> list[str]:
    """Load KEY=VALUE lines into os.environ without overriding existing vars.

    Path: ``CRAWLER_ENV_FILE`` or ~/.config/ai-crawler/env. Returns the names
    loaded (never values). Silently does nothing if the file is absent.
    """
    global _ENV_LOADED
    p = Path(path or os.environ.get("CRAWLER_ENV_FILE") or DEFAULT_ENV_FILE)
    loaded: list[str] = []
    if p.is_file():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[7:]
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
                loaded.append(k)
    _ENV_LOADED = True
    return loaded


def api_key_for(provider: str) -> tuple[str | None, str | None]:
    """Return (key, env_var_name) for *provider*; (None, None) if unset."""
    if not _ENV_LOADED:
        load_env_file()
    for name in KEY_ENV.get(provider, ()):
        val = os.environ.get(name)
        if val:
            return val, name
    return None, None


def key_status() -> dict[str, dict[str, Any]]:
    """Which providers have keys configured (names only, never values)."""
    out = {}
    for p in PROVIDERS:
        if p == "ollama":
            out[p] = {"needs_key": False}
            continue
        _, name = api_key_for(p)
        out[p] = {"needs_key": True, "configured": bool(name), "env_var": name,
                  "expected_env": list(KEY_ENV[p])}
    return out


def _redact(text: str, *secrets: str | None) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "***")
    return text


def _split_system(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
    rest = [m for m in messages if m.get("role") != "system"]
    return system, rest


# ── Base ──────────────────────────────────────────────────────────────────────

SessionFactory = Callable[[], Any]


class Provider:
    name = "base"

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 timeout: float = 120, session_factory: SessionFactory | None = None) -> None:
        self.base_url = (base_url or DEFAULT_BASE_URLS[self.name]).rstrip("/")
        self._api_key = api_key
        self.timeout = timeout
        self._session_factory = session_factory or aiohttp.ClientSession

    # subclasses implement
    def _request(self, model: str, messages: list[dict[str, str]], json_mode: bool,
                 max_tokens: int | None) -> tuple[str, dict[str, str], dict[str, Any]]:
        raise NotImplementedError

    def _parse(self, data: dict[str, Any], model: str) -> LLMResponse:
        raise NotImplementedError

    def _key(self) -> str | None:
        if not KEY_ENV.get(self.name):
            return None
        if self._api_key:
            return self._api_key
        key, _ = api_key_for(self.name)
        if not key:
            raise MissingAPIKey(
                f"{self.name}: no API key. Set one of {', '.join(KEY_ENV[self.name])} "
                f"in the environment or in {DEFAULT_ENV_FILE} (chmod 600)."
            )
        return key

    async def chat(self, model: str, messages: list[dict[str, str]], *, json_mode: bool = False,
                   max_tokens: int | None = None) -> LLMResponse:
        key = self._key()
        url, headers, payload = self._request(model, messages, json_mode, max_tokens)
        t0 = time.monotonic()
        try:
            async with self._session_factory() as session:
                async with session.post(url, json=payload, headers=headers,
                                        timeout=aiohttp.ClientTimeout(total=self.timeout)) as resp:
                    body = await resp.text()
                    status = resp.status
        except aiohttp.ClientError as exc:
            raise ProviderError(_redact(f"{self.name}: network error: {exc}", key)) from None
        if status >= 400:
            raise ProviderError(_redact(f"{self.name}: HTTP {status}: {body[:300]}", key), status)
        import json as _json
        try:
            data = _json.loads(body)
        except ValueError:
            raise ProviderError(f"{self.name}: non-JSON response: {body[:200]}") from None
        out = self._parse(data, model)
        out.latency_ms = int((time.monotonic() - t0) * 1000)
        return out


# ── Ollama ────────────────────────────────────────────────────────────────────

class OllamaProvider(Provider):
    name = "ollama"

    def _request(self, model, messages, json_mode, max_tokens):
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if json_mode:
            payload["format"] = "json"
        if max_tokens:
            payload["options"] = {"num_predict": max_tokens}
        return f"{self.base_url}/api/chat", {}, payload

    def _parse(self, data, model):
        pt, ct = data.get("prompt_eval_count"), data.get("eval_count")
        return LLMResponse(
            text=(data.get("message") or {}).get("content", ""), provider=self.name, model=model,
            prompt_tokens=pt, completion_tokens=ct,
            total_tokens=(pt + ct) if pt is not None and ct is not None else None,
            raw_usage={"prompt_eval_count": pt, "eval_count": ct},
        )


# ── OpenAI / DeepSeek (chat completions) ─────────────────────────────────────

class OpenAIProvider(Provider):
    name = "openai"

    def _request(self, model, messages, json_mode, max_tokens):
        payload: dict[str, Any] = {"model": model, "messages": messages}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        headers = {"Authorization": f"Bearer {self._key()}"}
        return f"{self.base_url}/chat/completions", headers, payload

    def _parse(self, data, model):
        usage = data.get("usage") or {}
        choices = data.get("choices") or [{}]
        return LLMResponse(
            text=((choices[0].get("message") or {}).get("content") or ""),
            provider=self.name, model=model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"), raw_usage=usage,
        )


class DeepSeekProvider(OpenAIProvider):
    name = "deepseek"


# ── Anthropic ────────────────────────────────────────────────────────────────

class AnthropicProvider(Provider):
    name = "anthropic"
    API_VERSION = "2023-06-01"

    def _request(self, model, messages, json_mode, max_tokens):
        system, rest = _split_system(messages)
        if json_mode:
            system = (system + "\n\nRespond with a single JSON object only.").strip()
        payload: dict[str, Any] = {"model": model, "max_tokens": max_tokens or 2048,
                                   "messages": rest}
        if system:
            payload["system"] = system
        headers = {"x-api-key": self._key() or "", "anthropic-version": self.API_VERSION}
        return f"{self.base_url}/messages", headers, payload

    def _parse(self, data, model):
        usage = data.get("usage") or {}
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        pt, ct = usage.get("input_tokens"), usage.get("output_tokens")
        return LLMResponse(
            text=text, provider=self.name, model=model,
            prompt_tokens=pt, completion_tokens=ct,
            total_tokens=(pt + ct) if pt is not None and ct is not None else None,
            raw_usage=usage,
        )


# ── Gemini ────────────────────────────────────────────────────────────────────

class GeminiProvider(Provider):
    name = "gemini"

    def _request(self, model, messages, json_mode, max_tokens):
        system, rest = _split_system(messages)
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user",
             "parts": [{"text": m["content"]}]}
            for m in rest
        ]
        payload: dict[str, Any] = {"contents": contents}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        gen: dict[str, Any] = {}
        if json_mode:
            gen["responseMimeType"] = "application/json"
        if max_tokens:
            gen["maxOutputTokens"] = max_tokens
        if gen:
            payload["generationConfig"] = gen
        # Key in a header, never in the URL (keeps it out of logs/tracebacks).
        headers = {"x-goog-api-key": self._key() or ""}
        return f"{self.base_url}/models/{model}:generateContent", headers, payload

    def _parse(self, data, model):
        usage = data.get("usageMetadata") or {}
        cands = data.get("candidates") or [{}]
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        pt = usage.get("promptTokenCount")
        ct = usage.get("candidatesTokenCount")
        if ct is not None and usage.get("thoughtsTokenCount"):
            ct += usage["thoughtsTokenCount"]  # thinking tokens are billed as output
        return LLMResponse(
            text=text, provider=self.name, model=model,
            prompt_tokens=pt, completion_tokens=ct, total_tokens=usage.get("totalTokenCount"),
            raw_usage=usage,
        )


_CLASSES: dict[str, type[Provider]] = {
    "ollama": OllamaProvider, "gemini": GeminiProvider, "anthropic": AnthropicProvider,
    "openai": OpenAIProvider, "deepseek": DeepSeekProvider,
}


def make_provider(name: str, base_url: str | None = None, timeout: float = 120,
                  session_factory: SessionFactory | None = None) -> Provider:
    name = (name or "ollama").lower()
    if name not in _CLASSES:
        raise ValueError(f"Unknown provider '{name}'. Choose from: {', '.join(PROVIDERS)}")
    return _CLASSES[name](base_url=base_url, timeout=timeout, session_factory=session_factory)
