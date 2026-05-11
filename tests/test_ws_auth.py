"""
Tests for WebSocket Token Authentication.
"""

import base64
import hmac
import hashlib
import json
import time
from unittest.mock import patch

import os
os.environ["WS_TOKEN_SECRET"] = "dummy-secret-for-tests"

from api.ws_auth import verify_ws_token


def _generate_test_token(secret: str, session_id: str, expired: bool = False) -> str:
    """Helper to generate a valid or expired JWT-like token."""
    exp = int(time.time()) - 1000 if expired else int(time.time()) + 3600
    payload = {
        "session_id": session_id,
        "exp": exp
    }
    payload_json = json.dumps(payload, separators=(',', ':')).encode('utf-8')
    payload_b64 = base64.urlsafe_b64encode(payload_json).rstrip(b'=').decode('utf-8')
    
    secret_bytes = secret.encode("utf-8")
    signature = hmac.new(secret_bytes, payload_json, hashlib.sha256).digest()
    signature_b64 = base64.urlsafe_b64encode(signature).rstrip(b'=').decode('utf-8')
    
    return f"{payload_b64}.{signature_b64}"


def test_verify_ws_token_success():
    """Valid token should return True."""
    secret = "test-secret"
    session_id = "sess-123"
    token = _generate_test_token(secret, session_id)
    
    with patch("api.ws_auth.SECRET", secret.encode()):
        assert verify_ws_token(token, session_id)[0] is True


def test_verify_ws_token_invalid_session():
    """Token with mismatched session_id should return False."""
    secret = "test-secret"
    token = _generate_test_token(secret, "wrong-session")
    
    with patch("api.ws_auth.SECRET", secret.encode()):
        assert verify_ws_token(token, "sess-123")[0] is False


def test_verify_ws_token_expired():
    """Expired token should return False."""
    secret = "test-secret"
    session_id = "sess-123"
    token = _generate_test_token(secret, session_id, expired=True)
    
    with patch("api.ws_auth.SECRET", secret.encode()):
        assert verify_ws_token(token, session_id)[0] is False


def test_verify_ws_token_invalid_signature():
    """Token with manipulated signature should return False."""
    secret = "test-secret"
    session_id = "sess-123"
    token = _generate_test_token(secret, session_id)
    
    # Tamper with the signature
    parts = token.split(".")
    tampered_token = f"{parts[0]}.wrong_signature"
    
    with patch("api.ws_auth.SECRET", secret.encode()):
        assert verify_ws_token(tampered_token, session_id)[0] is False


def test_verify_ws_token_no_secret():
    """If signature is empty or doesn't match the bad key."""
    session_id = "sess-123"
    token = _generate_test_token("some-secret", session_id)
    
    with patch("api.ws_auth.SECRET", b""):
        assert verify_ws_token(token, session_id)[0] is False
