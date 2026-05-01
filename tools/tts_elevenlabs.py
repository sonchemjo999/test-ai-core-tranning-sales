import base64
import httpx
from core.config import ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, ELEVENLABS_MODEL_ID

async def generate_tts_elevenlabs_base64(text: str) -> str | None:
    """Generates TTS using ElevenLabs and returns a Base64 data URI."""
    if not ELEVENLABS_API_KEY:
        return None

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "accept": "audio/mpeg",
        "content-type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
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
                return None
                
            # Convert to base64
            b64_audio = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Format as Data URI
            return f"data:audio/mp3;base64,{b64_audio}"
            
    except Exception as e:
        print(f"[ElevenLabs] Error generating TTS: {e}")
        return None
