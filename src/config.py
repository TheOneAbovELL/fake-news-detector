"""
config.py
---------
Central, validated runtime configuration read from environment variables.

Every setting has a safe default so the app runs locally with no configuration;
only the optional Groq LLM layer needs a credential (GROQ_API_KEY).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(ValueError):
    """Raised when an environment variable holds an invalid value."""


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from None
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from None
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{name} must be a boolean (true/false), got {raw!r}")


@dataclass(frozen=True)
class Settings:
    # Model artefacts
    models_dir: Path = PROJECT_ROOT / "models"

    # Groq LLM layer (optional)
    groq_api_key: str = field(default="", repr=False)
    groq_model: str = "llama-3.3-70b-versatile"
    groq_endpoint: str = "https://api.groq.com/openai/v1/chat/completions"
    groq_timeout: float = 20.0
    groq_max_retries: int = 1

    # Scraper
    scrape_timeout: float = 10.0
    scrape_max_bytes: int = 3_000_000
    # Only for local development/testing: lets the scraper fetch private hosts.
    allow_private_urls: bool = False

    # HTTP API
    max_text_chars: int = 5000
    rate_limit_per_minute: int = 30  # 0 disables
    trust_proxy: bool = False

    @property
    def llm_enabled(self) -> bool:
        return bool(self.groq_api_key)

    @classmethod
    def from_env(cls) -> "Settings":
        models_dir = os.environ.get("MODELS_DIR", "").strip()
        return cls(
            models_dir=Path(models_dir) if models_dir else PROJECT_ROOT / "models",
            groq_api_key=os.environ.get("GROQ_API_KEY", "").strip(),
            groq_model=os.environ.get("GROQ_MODEL", "").strip() or cls.groq_model,
            groq_timeout=_env_float("GROQ_TIMEOUT", cls.groq_timeout, minimum=1.0),
            groq_max_retries=_env_int("GROQ_MAX_RETRIES", cls.groq_max_retries),
            scrape_timeout=_env_float("SCRAPE_TIMEOUT", cls.scrape_timeout, minimum=1.0),
            scrape_max_bytes=_env_int("SCRAPE_MAX_BYTES", cls.scrape_max_bytes, minimum=10_000),
            allow_private_urls=_env_bool("ALLOW_PRIVATE_URLS", cls.allow_private_urls),
            max_text_chars=_env_int("MAX_TEXT_CHARS", cls.max_text_chars, minimum=100),
            rate_limit_per_minute=_env_int("RATE_LIMIT_PER_MINUTE", cls.rate_limit_per_minute),
            trust_proxy=_env_bool("TRUST_PROXY", cls.trust_proxy),
        )
