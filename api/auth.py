"""
API key authentication for Web ↔ FastAPI communication.

Usage:
    @app.post("/web/chat")
    def web_chat(body: ..., _: str = Depends(verify_api_key)):
        ...
"""

from __future__ import annotations

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from core.config import AI_API_KEY

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str | None = Security(_api_key_header)) -> str:
    """Verify X-API-Key header matches the configured secret."""
    if not api_key or api_key != AI_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key
