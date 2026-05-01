"""
Internal voice WebSocket — audio-native direct voice for web gateway relay.

This endpoint is for INTERNAL use only (called by the web voice gateway,
NOT by mobile clients directly). It uses an INTERNAL_TOKEN for authentication.

Protocol:
  - Binary audio frames (PCM16 16kHz mono) forwarded to Gemini Live API
  - Audio responses streamed back to the web gateway
  - Web gateway handles auth, session ownership, quota, and DB persistence

Note: Mobile clients connect to the web voice gateway, not here.
"""

from __future__ import annotations

import json
import asyncio
import logging
import base64
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from core.config import GEMINI_API_KEY, AI_API_KEY

logger = logging.getLogger(__name__)


def verify_internal_token(token: str | None) -> bool:
    """Verify the internal token from the web voice gateway."""
    if not token:
        return False
    return token == AI_API_KEY


async def handle_direct_voice_websocket(
    websocket: WebSocket,
    session_id: str,
    token: str | None = None,
) -> None:
    """Handle direct voice WebSocket connection from web voice gateway.

    Args:
        websocket: The WebSocket connection from the web gateway
        session_id: Session identifier (for logging)
        token: Internal API key to verify web gateway authenticity
    """
    if not verify_internal_token(token):
        await websocket.close(code=4001, reason="Invalid internal token")
        logger.warning(f"[direct-voice] Unauthorized access attempt for session {session_id}")
        return

    if not GEMINI_API_KEY:
        await websocket.close(code=4002, reason="Gemini API key not configured")
        logger.error("[direct-voice] GEMINI_API_KEY not configured")
        return

    await websocket.accept()
    logger.info(f"[direct-voice] Connection accepted for session {session_id}")

    gemini_ws = None
    try:
        import websockets

        url = (
            "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha."
            f"GenerativeService.BidiGenerateContent?key={GEMINI_API_KEY}"
        )
        gemini_ws = await websockets.connect(url)

        system_instruction = (
            "Bạn là một khách hàng tiềm năng đang trong cuộc gọi luyện sales B2B. "
            "Hãy phản hồi tự nhiên bằng giọng nói, ngắn gọn và có cảm xúc."
        )

        setup_msg = {
            "setup": {
                "model": "models/gemini-2.0-flash-exp",
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {"voiceName": "Aoede"}
                        }
                    },
                },
                "systemInstruction": {"parts": [{"text": system_instruction}]},
            }
        }
        await gemini_ws.send(json.dumps(setup_msg))

        raw_setup = await asyncio.wait_for(gemini_ws.recv(), timeout=10.0)
        setup_response = json.loads(raw_setup)
        if "setupComplete" in setup_response:
            logger.info(f"[direct-voice] Gemini Live session {session_id} set up")
        else:
            logger.warning(f"[direct-voice] Unexpected setup response: {setup_response}")

        conversation: list[dict[str, str]] = []

        async def forward_audio_to_gemini():
            """Forward binary audio frames from gateway to Gemini."""
            try:
                while True:
                    msg = await websocket.receive()
                    if "bytes" in msg:
                        b64_audio = base64.b64encode(msg["bytes"]).decode("utf-8")
                        turn_msg = {
                            "clientContent": {
                                "turns": [{
                                    "role": "user",
                                    "parts": [{"inlineData": {
                                        "mimeType": "audio/pcm;rate=16000",
                                        "data": b64_audio
                                    }}]
                                }],
                                "turnComplete": True
                            }
                        }
                        await gemini_ws.send(json.dumps(turn_msg))
                    elif "text" in msg:
                        data = json.loads(msg["text"])
                        if data.get("type") == "stop":
                            break
                        elif data.get("type") == "start":
                            persona = data.get("persona", "")
                            if persona:
                                logger.info(f"[direct-voice] Session {session_id} started with persona")
            except WebSocketDisconnect:
                logger.info(f"[direct-voice] Gateway disconnected from session {session_id}")
            except Exception as e:
                logger.error(f"[direct-voice] Error forwarding to Gemini: {e}")

        async def forward_gemini_to_gateway():
            """Forward audio + transcript responses from Gemini to the web gateway."""
            try:
                async for raw_msg in gemini_ws:
                    data = json.loads(raw_msg)
                    server_content = data.get("serverContent")
                    if not server_content:
                        continue

                    model_turn = server_content.get("modelTurn")
                    if model_turn:
                        for part in model_turn.get("parts", []):
                            inline_data = part.get("inlineData")
                            if inline_data and inline_data.get("data"):
                                await websocket.send_json({
                                    "type": "audio",
                                    "audio_b64": inline_data["data"]
                                })

                            text_part = part.get("text")
                            if text_part:
                                conversation.append({"role": "assistant", "content": text_part})
                                await websocket.send_json({
                                    "type": "transcript",
                                    "role": "assistant",
                                    "text": text_part,
                                    "is_partial": False
                                })

                    if server_content.get("turnComplete"):
                        await websocket.send_json({"type": "status", "status": "turn_complete"})
            except websockets.exceptions.ConnectionClosed:
                logger.info("[direct-voice] Gemini connection closed")
            except Exception as e:
                logger.error(f"[direct-voice] Error forwarding to gateway: {e}")

        await asyncio.gather(
            forward_audio_to_gemini(),
            forward_gemini_to_gateway(),
        )

    except asyncio.TimeoutError:
        logger.error(f"[direct-voice] Setup timeout for session {session_id}")
        await websocket.close(code=4003, reason="Setup timeout")
    except WebSocketDisconnect:
        logger.info(f"[direct-voice] Session {session_id} disconnected")
    except Exception as e:
        logger.error(f"[direct-voice] Error in session {session_id}: {e}")
        try:
            await websocket.close(code=4004, reason=f"Internal error: {str(e)[:50]}")
        except Exception:
            pass
    finally:
        if gemini_ws:
            await gemini_ws.close()
        logger.info(f"[direct-voice] Session {session_id} cleaned up")
