"""
API key authentication for Web ↔ FastAPI communication.

Usage:
    @app.post("/web/chat")
    def web_chat(body: ..., _: str = Depends(verify_api_key)):
        ...
"""

from __future__ import annotations

import hashlib

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from core.config import AI_API_KEY

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10] if value else "empty"


def verify_api_key(api_key: str | None = Security(_api_key_header)) -> str:
    """Verify X-API-Key header matches the configured secret."""
    normalized_api_key = api_key.strip() if api_key else ""
    if not normalized_api_key or normalized_api_key != AI_API_KEY:
        print(
            "[auth] invalid API key",
            {
                "received_len": len(normalized_api_key),
                "received_fp": _fingerprint(normalized_api_key),
                "configured_len": len(AI_API_KEY),
                "configured_fp": _fingerprint(AI_API_KEY),
                "configured_is_default": AI_API_KEY == "dev-secret-key-change-in-production",
            },
        )
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return normalized_api_key
