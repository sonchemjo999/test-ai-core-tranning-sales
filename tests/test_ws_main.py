"""
Integration Tests for WebSocket main loop.
"""

from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

import os
os.environ["WS_TOKEN_SECRET"] = "dummy-secret-for-tests"
os.environ["CARTESIA_API_KEY"] = "dummy-key"

from api.main import app
import base64
import json
import hmac
import hashlib
import time

def _generate_test_token(secret: str, session_id: str) -> str:
    payload = {"session_id": session_id, "exp": int(time.time()) + 3600}
    payload_json = json.dumps(payload, separators=(',', ':')).encode('utf-8')
    payload_b64 = base64.urlsafe_b64encode(payload_json).rstrip(b'=').decode('utf-8')
    secret_bytes = secret.encode("utf-8")
    signature = hmac.new(secret_bytes, payload_json, hashlib.sha256).digest()
    signature_b64 = base64.urlsafe_b64encode(signature).rstrip(b'=').decode('utf-8')
    return f"{payload_b64}.{signature_b64}"

client = TestClient(app)

def test_websocket_connection_unauthorized():
    """Missing token should be rejected."""
    import pytest
    from starlette.websockets import WebSocketDisconnect
    
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws/call/sess-123?token=invalid"):
            pass
    assert exc.value.code in (1000, 1008)


def test_websocket_chat_flow():
    """Mock the background worker and LLM to test WS input/output flow."""
    token = _generate_test_token("dummy-secret-for-tests", "sess-123")
    
    # Patch internal AI generators instead of inner function
    async def fake_tts_generator(*args, **kwargs):
        yield "fake_base64_audio"
        
    async def fake_llm_generator(*args, **kwargs):
        yield "Hello"

    with patch("tools.tts_cartesia.generate_tts_cartesia_stream", side_effect=fake_tts_generator), \
         patch("llm.llm_client.generate_customer_turn_voice_stream", side_effect=fake_llm_generator), \
         patch("core.config.CARTESIA_API_KEY", "dummy-key"):
        
        with client.websocket_connect(f"/ws/call/sess-123?token={token}") as ws:
            # Send initial params
            ws.send_json({
                "type": "init",
                "customer_persona": "test",
                "scenario_title": "test",
                "scenario_description": "test",
                "ai_tone": "neutral",
                "follow_up_depth": "light",
                "customer_gender": "female"
            })
            
            # Send user text
            ws.send_json({
                "type": "user_text",
                "text": "Hello"
            })
            
            # Receive transcript
            resp1 = ws.receive_json()
            assert resp1["type"] == "transcript"
            assert resp1["text"] == "Hello"
            
            # Receive audio response (could be audio_chunk or audio_url depending on fallback)
            resp2 = ws.receive_json()
            assert resp2["type"] in ("audio_chunk", "audio_url")
            
            if resp2["type"] == "audio_chunk":
                assert "audio" in resp2
                # Receive audio_chunk_end
                resp3 = ws.receive_json()
                assert resp3["type"] == "audio_chunk_end"
            else:
                assert "url" in resp2
                
            # Close
            ws.close()
