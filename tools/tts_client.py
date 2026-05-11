"""
FPT.AI Text-To-Speech (TTS) Client.
"""

from __future__ import annotations

import httpx
import logging
from core.config import TTS_API_KEY, TTS_VOICE, TTS_SPEED

logger = logging.getLogger(__name__)

async def generate_tts_fpt(text: str, voice: str | None = None) -> str | None:
    """
    Sends text to FPT.AI TTS v5 API and returns the audio URL.
    """
    text = text.strip()
    if not TTS_API_KEY or not text:
        return None
        
    url = 'https://api.fpt.ai/hmi/tts/v5'
    headers = {
        'api-key': TTS_API_KEY,
        'speed': str(TTS_SPEED),
        'voice': voice or TTS_VOICE
    }
    
    try:
        async with httpx.AsyncClient() as client:
            # The API requires data encoded as utf-8 string payload
            response = await client.post(
                url, 
                data=text.encode('utf-8'), 
                headers=headers, 
                timeout=15.0
            )
            response.raise_for_status()
            data = response.json()
            
            # FPT.AI v5 API generally returns the async download URL in the "async" field
            audio_url = data.get("async")
            if audio_url and isinstance(audio_url, str) and audio_url.startswith("http"):
                return audio_url
            
            logger.warning(f"Unexpected FPT TTS response: {data}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to generate TTS from FPT.AI: {e}")
        return None

import asyncio

async def wait_for_fpt_audio(url: str, max_retries: int = 20, delay: float = 0.15) -> bool:
    """
    Poll the FPT audio URL until it returns 200 OK or times out.
    """
    async with httpx.AsyncClient() as client:
        for attempt in range(max_retries):
            try:
                # Need to use GET because some CDNs don't support HEAD properly or FPT requires GET
                resp = await client.head(url, timeout=2.0)
                if resp.status_code == 200:
                    return True
            except httpx.RequestError:
                pass
            await asyncio.sleep(delay)
    return False
