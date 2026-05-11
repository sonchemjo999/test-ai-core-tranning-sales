"""Cartesia TTS — sonic-3.5, Vietnamese female voice via SSE streaming.

Returns base64 PCM 16kHz s16le audio, same format as ElevenLabs PCM path
so it's directly compatible with the Simli lip-sync pipeline.

Provides two modes:
  - generate_tts_cartesia_base64()  — full assembly (legacy)
  - generate_tts_cartesia_stream()  — yields individual chunks for low TTFA
"""

import base64
import json
from typing import AsyncGenerator

import httpx

from core.config import CARTESIA_API_KEY, CARTESIA_FEMALE_VOICE_ID

_CARTESIA_SSE_URL = "https://api.cartesia.ai/tts/sse"
_CARTESIA_API_VERSION = "2025-04-16"


def _build_payload(text: str) -> dict:
    return {
        "model_id": "sonic-3.5",
        "transcript": text,
        "voice": {"mode": "id", "id": CARTESIA_FEMALE_VOICE_ID},
        "output_format": {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": 16000,
        },
        "language": "vi",
    }


def _build_headers() -> dict:
    return {
        "Cartesia-Version": _CARTESIA_API_VERSION,
        "X-API-Key": CARTESIA_API_KEY,
        "Content-Type": "application/json",
    }


# ── Streaming mode (low TTFA) ────────────────────────────────────────────────

async def generate_tts_cartesia_stream(text: str) -> AsyncGenerator[str, None]:
    """Yield individual base64 PCM chunks as they arrive from Cartesia SSE.

    Each yielded value is a raw base64 string (NOT a data URI).
    The caller is responsible for wrapping/sending as needed.

    Yields nothing if the request fails.
    """
    if not CARTESIA_API_KEY:
        print("[Cartesia] CARTESIA_API_KEY not configured")
        return

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            async with client.stream(
                "POST", _CARTESIA_SSE_URL,
                headers=_build_headers(), json=_build_payload(text),
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    print(f"[Cartesia] HTTP {response.status_code}: {body[:200]}")
                    return

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue

                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        break

                    try:
                        chunk = json.loads(raw)
                        audio_b64 = chunk.get("data", "")
                        if audio_b64:
                            yield audio_b64
                    except (json.JSONDecodeError, Exception):
                        continue

    except Exception as e:
        print(f"[Cartesia] Streaming error: {e}")


# ── Full-assembly mode (legacy / fallback) ────────────────────────────────────

async def generate_tts_cartesia_base64(text: str) -> tuple[str | None, str]:
    """Generate TTS via Cartesia sonic-3.5 SSE streaming — full assembly.

    Returns:
        (data_uri, format_tag):
            - On success: ("data:audio/pcm;base64,<b64>", "pcm16")
            - On failure: (None, "mp3")
    """
    if not CARTESIA_API_KEY:
        print("[Cartesia] CARTESIA_API_KEY not configured")
        return None, "mp3"

    try:
        pcm_chunks: list[bytes] = []

        async for b64_chunk in generate_tts_cartesia_stream(text):
            pcm_chunks.append(base64.b64decode(b64_chunk))

        if not pcm_chunks:
            print("[Cartesia] No audio data received")
            return None, "mp3"

        full_pcm = b"".join(pcm_chunks)
        b64_audio = base64.b64encode(full_pcm).decode("utf-8")
        print(f"[Cartesia] Generated {len(full_pcm):,} bytes PCM for {len(text)} chars")

        return f"data:audio/pcm;base64,{b64_audio}", "pcm16"

    except Exception as e:
        print(f"[Cartesia] Error generating TTS: {e}")
        return None, "mp3"
