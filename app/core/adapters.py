from __future__ import annotations

from functools import lru_cache

from app.adapters.openai import OpenAIAdapter
from app.core.config import get_settings


@lru_cache
def get_openai_adapter() -> OpenAIAdapter:
    """One OpenAIAdapter (and its underlying httpx.AsyncClient connection
    pool) shared across all requests, rather than a fresh one per request.
    Tests override this dependency directly with a respx-wired adapter —
    no need to touch app startup/shutdown to swap it out.
    """
    return OpenAIAdapter(get_settings())
