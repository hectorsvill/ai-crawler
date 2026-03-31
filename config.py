"""
config.py — Pydantic Settings model and YAML configuration loader.

Merges default_config.yaml with an optional user-provided config file,
then allows environment variable overrides via pydantic-settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Sub-models ────────────────────────────────────────────────────────────────

class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    navigator_model: str = "qwen2.5:1.5b"
    extractor_model: str = "qwen2.5:7b"
    router_model: str = "qwen2.5:7b"
    timeout: int = 120
    max_retries: int = 3


class CrawlConfig(BaseModel):
    max_depth: int = 5
    max_pages: int = 500
    rate_limit_per_domain: float = 2.0
    delay_range: list[float] = Field(default=[1.0, 3.0])
    user_agents: list[str] = Field(default=[
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ])
    domain_allowlist: list[str] = Field(default=[])
    domain_denylist: list[str] = Field(default=[])
    use_playwright_for: list[str] = Field(default=[])

    # ── Phase 1: Sitemap + intelligent queuing ────────────────────────────────
    sitemap_enabled: bool = True
    sitemap_max_urls: int = 10_000
    sitemap_priority_boost: float = 0.1
    # Depth-decay multiplier: effective_priority *= (1 - decay)^depth
    priority_depth_decay: float = 0.15
    priority_url_heuristics: bool = True

    # ── Phase 2: Adaptive rate limiting ──────────────────────────────────────
    backoff_base: float = 2.0        # seconds; doubles per retry
    backoff_max: float = 120.0       # cap on backoff wait
    rate_recovery_factor: float = 1.2  # rate multiplier on sustained success
    rate_success_window: int = 5       # successes before rate recovery
    max_retries_per_url: int = 3

    # ── Phase 4: Incremental crawling ─────────────────────────────────────────
    incremental_crawl: bool = False
    incremental_older_than_days: int = 7
    conditional_requests: bool = True  # send ETag/If-Modified-Since


class StorageConfig(BaseModel):
    db_path: str = "crawl_data.db"


class WorkflowConfig(BaseModel):
    default: str = "auto"


# ── Root config ───────────────────────────────────────────────────────────────

class AppConfig(BaseModel):
    """Complete application configuration assembled from YAML + overrides."""

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    crawl: CrawlConfig = Field(default_factory=CrawlConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)


# ── Loader ────────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(user_config_path: str | Path | None = None) -> AppConfig:
    """
    Load configuration by merging:
    1. default_config.yaml (shipped with the project)
    2. User-supplied config YAML (optional)
    3. Environment variable overrides (prefixed CRAWLER_)

    Returns a fully-validated AppConfig instance.
    """
    # Load defaults
    with open(_DEFAULT_CONFIG_PATH) as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    # Merge user config on top
    if user_config_path:
        path = Path(user_config_path)
        if path.exists():
            with open(path) as f:
                user_data: dict[str, Any] = yaml.safe_load(f) or {}
            data = _deep_merge(data, user_data)
        else:
            import warnings
            warnings.warn(f"Config file not found: {path}. Using defaults.", stacklevel=2)

    # Apply env overrides (CRAWLER_OLLAMA__BASE_URL etc. using __ as separator)
    _apply_env_overrides(data)

    return AppConfig.model_validate(data)


# Alias so both `AppConfig` and `Settings` work as imports.
# Settings() with no args loads from YAML; Settings(path) loads a custom config.
def Settings(user_config_path=None) -> AppConfig:  # type: ignore[misc]
    """Convenience wrapper: Settings() == load_config()."""
    return load_config(user_config_path)


def _apply_env_overrides(data: dict[str, Any]) -> None:
    """
    Apply environment variable overrides using the pattern:
    CRAWLER_<SECTION>__<KEY>=value
    e.g. CRAWLER_OLLAMA__BASE_URL=http://remote:11434
    """
    prefix = "CRAWLER_"
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        parts = env_key[len(prefix):].lower().split("__")
        target = data
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        # Attempt numeric coercion
        final_key = parts[-1]
        try:
            target[final_key] = int(env_val)
        except ValueError:
            try:
                target[final_key] = float(env_val)
            except ValueError:
                target[final_key] = env_val
