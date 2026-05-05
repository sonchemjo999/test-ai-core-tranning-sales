import base64
import httpx
from core.config import ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, ELEVENLABS_MODEL_ID

async def generate_tts_elevenlabs_base64(text: str, output_pcm: bool = True, voice_id: str | None = None) -> tuple[str | None, str]:
    """Generates TTS using ElevenLabs and returns (Base64 data URI, format).
    
    When output_pcm=True, returns PCM16 16kHz mono (for Simli lip-sync).
    When output_pcm=False, returns MP3 (legacy fallback).
    """
    if not ELEVENLABS_API_KEY:
        return None, "mp3"

    fmt = "pcm_16000" if output_pcm else "mp3_44100_128"
    mime = "audio/pcm" if output_pcm else "audio/mpeg"
    accept = "application/octet-stream" if output_pcm else "audio/mpeg"
    
    vid = voice_id if voice_id else ELEVENLABS_VOICE_ID
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{vid}?output_format={fmt}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "accept": accept,
        "content-type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {
            "stability": 0.35,          # Lowered for more emotional range
            "similarity_boost": 0.70,   # Lowered slightly to allow natural prosody
            "style": 0.25,              # Exaggerate the expression
            "use_speaker_boost": True,
        },
    }

    try:
        # Use httpx AsyncClient to fetch audio bytes
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            # Read binary audio content
            audio_bytes = response.content
            if not audio_bytes:
                return None, "mp3"
                
            # Convert to base64
            b64_audio = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Format as Data URI
            tag = "pcm16" if output_pcm else "mp3"
            return f"data:{mime};base64,{b64_audio}", tag
            
    except Exception as e:
        print(f"[ElevenLabs] Error generating TTS: {e}")
        return None, "mp3"
