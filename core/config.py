import os
from pathlib import Path
from dotenv import load_dotenv

# Load root .env
_root = Path(__file__).resolve().parent.parent.parent  # core-ai/core/config.py → core-ai → A20-App-154/
load_dotenv(_root / ".env", override=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/").strip()
OPEN_ROUTER_API = os.getenv("OPEN_ROUTER_API", "").strip()
OPEN_ROUTER_BASE_URL = os.getenv("OPEN_ROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip()
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
