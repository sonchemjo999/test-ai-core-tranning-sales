"""
Gemini Multimodal Live API WebSocket Client.
Handles bi-directional audio streaming between FastAPI and Gemini 2.0 Flash.
"""

from __future__ import annotations

import json
import base64
import logging
import websockets
from fastapi import WebSocket, WebSocketDisconnect

from core.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

class GeminiLiveClient:
    def __init__(
        self, 
        client_ws: WebSocket, 
        session_id: str, 
        persona: str, 
        company_context: str | None
    ):
        self.client_ws = client_ws
        self.session_id = session_id
        self.persona = persona
        self.company_context = company_context
        
        self.gemini_ws: websockets.WebSocketClientProtocol | None = None
        # Use Gemini 2.0 Flash for realtime multimodal
        self.model = "models/gemini-2.0-flash-exp"
        self.host = "generativelanguage.googleapis.com"
        self.url = f"wss://{self.host}/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={GEMINI_API_KEY}"

    async def connect(self):
        """Connect to Gemini Live API and send initial setup."""
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing.")
            
        logger.info(f"Connecting to Gemini Live API for session {self.session_id}...")
        self.gemini_ws = await websockets.connect(self.url)
        
        # Build system instructions
        system_instruction = (
            f"You are a potential buyer in a sales call. Keep your answers conversational and concise.\n"
            f"Your persona: {self.persona}\n"
        )
        if self.company_context:
            system_instruction += f"\nCompany Context:\n{self.company_context}"
            
        # Send Setup Message
        setup_msg = {
            "setup": {
                "model": self.model,
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {
                                "voiceName": "Aoede" # Voice options: Aoede, Charon, Fenrir, Kore, Puck
                            }
                        }
                    }
                },
                "systemInstruction": {
                    "parts": [{"text": system_instruction}]
                }
            }
        }
        await self._send_event(setup_msg)
        
        # Wait for Setup Complete response
        raw_response = await self.gemini_ws.recv()
        response = json.loads(raw_response)
        if "setupComplete" in response:
            logger.info(f"Session {self.session_id} successfully set up with Gemini.")
        else:
            logger.warning(f"Unexpected initial response from Gemini: {response}")

    async def _send_event(self, event: dict):
        if self.gemini_ws:
            await self.gemini_ws.send(json.dumps(event))

    async def forward_client_to_gemini(self):
        """Read audio/JSON from client (Browser) and forward to Gemini."""
        try:
            while True:
                message = await self.client_ws.receive()
                
                if not self.gemini_ws:
                    continue
                    
                if "bytes" in message:
                    # Client sends binary audio (PCM 16-bit 16kHz or 24kHz)
                    b64_audio = base64.b64encode(message["bytes"]).decode("utf-8")
                    client_content = {
                        "clientContent": {
                            "turns": [
                                {
                                    "role": "user",
                                    "parts": [{"inlineData": {"mimeType": "audio/pcm;rate=16000", "data": b64_audio}}]
                                }
                            ],
                            "turnComplete": True
                        }
                    }
                    await self._send_event(client_content)
                elif "text" in message:
                    # Client sends JSON control messages (e.g. stop, disconnect)
                    data = json.loads(message["text"])
                    if data.get("type") == "stop":
                        # Cannot explicitly stop generating in Gemini easily, but we can handle logic here
                        pass
        except WebSocketDisconnect:
            logger.info(f"Client disconnected from session {self.session_id}.")
        except Exception as e:
            logger.error(f"Error forwarding to Gemini: {e}")

    async def forward_gemini_to_client(self):
        """Read results from Gemini and forward to browser."""
        try:
            if not self.gemini_ws:
                return
                
            async for message in self.gemini_ws:
                data = json.loads(message)
                
                # Check for server content
                server_content = data.get("serverContent")
                if not server_content:
                    continue
                    
                model_turn = server_content.get("modelTurn")
                if model_turn:
                    parts = model_turn.get("parts", [])
                    for part in parts:
                        # Forward Audio part
                        inline_data = part.get("inlineData")
                        if inline_data and inline_data.get("data"):
                            await self.client_ws.send_json({
                                "type": "audio",
                                "audio_b64": inline_data["data"]
                            })
                        
                        # Forward Text part (Transcript)
                        text_part = part.get("text")
                        if text_part:
                            await self.client_ws.send_json({
                                "type": "transcript",
                                "role": "assistant",
                                "text": text_part
                            })
                            
                # Check if turn is complete
                if server_content.get("turnComplete"):
                    await self.client_ws.send_json({
                        "type": "status",
                        "status": "turn_complete"
                    })
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("Gemini connection closed.")
        except Exception as e:
            logger.error(f"Error forwarding to Client: {e}")

    async def close(self):
        if self.gemini_ws:
            await self.gemini_ws.close()
