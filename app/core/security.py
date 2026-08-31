from __future__ import annotations

import hashlib
import secrets

_KEY_PREFIX = "sk-gw-"


def generate_api_key() -> str:
    """Returns a new raw API key, e.g. "sk-gw-<43 url-safe chars>".

    Only ever returned to the caller once, at creation time. The gateway
    never stores or logs the raw key — only its hash (see hash_api_key).
    """
    return _KEY_PREFIX + secrets.token_urlsafe(32)


def hash_api_key(raw_key: str) -> str:
    """SHA-256 hex digest, used both to store and to look up keys.

    Deliberately not a password hash (no bcrypt/argon2/salt). Those exist
    to slow down brute-forcing a low-entropy human-chosen password; this
    input is already a 256-bit random secret, so there's nothing a slow
    hash would protect against — it would only add latency to every
    request, since this hash runs on the hot path of every auth check.
    """
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def key_prefix(raw_key: str) -> str:
    """First 12 chars, safe to store and display in plaintext (e.g. a
    dashboard list view) without exposing the secret itself.
    """
    return raw_key[:12]
