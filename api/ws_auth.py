"""
WebSocket Token Authentication.

Token format: base64url(payload) + "." + base64url(HMAC-SHA256(payload, SECRET))
Payload JSON: {"user_id": "...", "session_id": "...", "exp": <unix_timestamp>}

Used by /ws/call/{session_id} to verify client identity.
Counterpart issuer lives in web: src/app/api/voice/token/route.ts
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time

_raw_secret = os.environ.get("WS_TOKEN_SECRET", "").strip()
if not _raw_secret:
    raise RuntimeError(
        "WS_TOKEN_SECRET is not set. "
        "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\" "
        "and add it to .env"
    )

SECRET = _raw_secret.encode()
TTL = int(os.environ.get("WS_TOKEN_TTL_SECONDS", "300"))


def _b64url_decode(s: str) -> bytes:
    """Decode base64url with padding recovery."""
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


def verify_ws_token(token: str, expected_session_id: str) -> tuple[bool, str]:
    """
    Verify a WS auth token against the shared secret.

    Returns (valid, reason). `reason` is empty when valid.
    """
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return False, "malformed_token"

        payload_b64, sig_b64 = parts
        payload_bytes = _b64url_decode(payload_b64)
        sig = _b64url_decode(sig_b64)

        expected_sig = hmac.new(SECRET, payload_bytes, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected_sig):
            return False, "bad_signature"

        payload = json.loads(payload_bytes)

        if payload.get("exp", 0) < int(time.time()):
            return False, "expired"

        if payload.get("session_id") != expected_session_id:
            return False, "session_mismatch"

        return True, ""

    except Exception as e:
        return False, f"parse_error:{type(e).__name__}"
