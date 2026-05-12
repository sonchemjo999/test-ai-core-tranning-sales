import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load root .env
_root = Path(__file__).resolve().parent.parent.parent  # core-ai/core/config.py → core-ai → A20-App-154/
load_dotenv(_root / ".env", override=False)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OPEN_ROUTER_API = os.getenv("OPEN_ROUTER_API", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-20250514")
# LangGraph / Sale Train Agent — OpenAI JSON-mode calls (override in .env)
SALES_LLM_MODEL = os.getenv("SALES_LLM_MODEL", "google/gemini-2.5-flash-lite")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
# Shared secret for Web ↔ FastAPI authentication
AI_API_KEY = os.getenv("AI_API_KEY", "dev-secret-key-change-in-production")

# FPT.AI Text-to-Speech
TTS_API_KEY = os.getenv("TTS_API_KEY", "").strip()
TTS_VOICE = os.getenv("TTS_VOICE", "minhquang")
TTS_MALE_VOICE = os.getenv("TTS_MALE_VOICE", TTS_VOICE).strip()
TTS_FEMALE_VOICE = os.getenv("TTS_FEMALE_VOICE", "banmai").strip()
TTS_SPEED = os.getenv("TTS_SPEED", "1")

# Cartesia Text-to-Speech (male / female voice — sonic-3.5, Vietnamese)
CARTESIA_API_KEY         = os.getenv("CARTESIA_API_KEY", "").strip()
CARTESIA_MALE_VOICE_ID   = os.getenv("CARTESIA_MALE_VOICE_ID", "CAAD5E2A-0BB2-4CB6-89C0-5D09C7B3D94B")
CARTESIA_FEMALE_VOICE_ID = os.getenv("CARTESIA_FEMALE_VOICE_ID", "935a9060-373c-49e4-b078-f4ea6326987a")

# Simli Avatar
SIMLI_API_KEY = os.getenv("SIMLI_API_KEY", "").strip()
SIMLI_ENABLED = os.getenv("SIMLI_ENABLED", "true").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
SIMLI_MALE_FACE_ID = os.getenv("SIMLI_MALE_FACE_ID", "").strip()
SIMLI_FEMALE_FACE_ID = os.getenv("SIMLI_FEMALE_FACE_ID", "").strip()


def _csv_env(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default).strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _json_env(name: str) -> list[dict]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


# WebRTC ICE servers for phone avatar video. TURN is required for many hosted
# environments because the aiortc peer may not be reachable via direct UDP.
WEBRTC_STUN_URLS = _csv_env("WEBRTC_STUN_URLS", "stun:stun.l.google.com:19302")
WEBRTC_TURN_URLS = _csv_env("WEBRTC_TURN_URLS")
WEBRTC_TURN_USERNAME = os.getenv("WEBRTC_TURN_USERNAME", "").strip()
WEBRTC_TURN_CREDENTIAL = os.getenv("WEBRTC_TURN_CREDENTIAL", "").strip()
WEBRTC_ICE_SERVERS_JSON = _json_env("WEBRTC_ICE_SERVERS_JSON")
