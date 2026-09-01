from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Env-driven config. Fields below are grouped by which day of the build
    actually consumes them — several are defined now but unused until later,
    so the shape doesn't have to change as each feature lands.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    environment: str = "development"
    log_level: str = "INFO"

    # --- Provider credentials (Day 2) ---
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    provider_request_timeout_seconds: float = 30.0

    # --- Retry (Day 5) ---
    retry_max_attempts: int = 3
    retry_base_delay_seconds: float = 0.5
    retry_max_delay_seconds: float = 8.0

    # --- Circuit breaker (Day 5) ---
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_cooldown_seconds: int = 30

    # --- Rate limiting (Day 7) ---
    rate_limit_requests_per_minute: int = 60
    rate_limit_tokens_per_minute: int = 100_000

    # --- Persistence (Day 3 / Day 7) ---
    database_url: str = "postgresql+asyncpg://gateway:gateway@localhost:5432/gateway"
    redis_url: str = "redis://localhost:6379/0"

    # --- Admin auth (post-Day-10) ---
    # Guards POST /v1/keys, GET /v1/keys, and key revocation. Empty means
    # "not configured" — admin endpoints fail CLOSED (503) rather than
    # silently allowing open access, since an unset secret in a real
    # deployment is a much more dangerous default than a loud failure.
    admin_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
