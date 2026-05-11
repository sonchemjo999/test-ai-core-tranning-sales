import os
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
TTS_SPEED = os.getenv("TTS_SPEED", "1")

# ElevenLabs Text-to-Speech
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5")

# Simli Avatar (AI-powered video avatar)
SIMLI_API_KEY = os.getenv("SIMLI_API_KEY", "").strip()
SIMLI_ENABLED = os.getenv("SIMLI_ENABLED", "false").lower() in ("true", "1", "yes")
SIMLI_MALE_FACE_ID = os.getenv("SIMLI_MALE_FACE_ID", "dd10cb5a-d31d-4f12-b69f-6db3383c006e").strip()
SIMLI_FEMALE_FACE_ID = os.getenv("SIMLI_FEMALE_FACE_ID", "cace3ef7-a4c4-425d-a8cf-a5358eb0c427").strip()
