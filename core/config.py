import os
from pathlib import Path
from dotenv import load_dotenv

# Load repo .env for local development only. Railway variables take precedence.
_root = Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env", override=False)


def env_str(name: str, default: str = "") -> str:
    value = os.getenv(name, default).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    return value


ANTHROPIC_API_KEY = env_str("ANTHROPIC_API_KEY")
OPENAI_API_KEY = env_str("OPENAI_API_KEY")
OPENAI_BASE_URL = env_str("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = env_str("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_API_KEY = env_str("GEMINI_API_KEY")
GEMINI_BASE_URL = env_str("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
GEMINI_MODEL = env_str("GEMINI_MODEL", "gemini-2.5-flash-lite")
GROK_API_KEY = env_str("GROK_API_KEY", env_str("XAI_API_KEY"))
GROK_BASE_URL = env_str("GROK_BASE_URL", env_str("XAI_BASE_URL", "https://api.x.ai/v1"))
GROK_MODEL = env_str("GROK_MODEL", "grok-4")
OPEN_ROUTER_API = env_str("OPEN_ROUTER_API", env_str("OPENROUTER_API_KEY"))
OPEN_ROUTER_BASE_URL = env_str("OPEN_ROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = env_str("OPENROUTER_MODEL", "google/gemini-2.5-flash-lite")
GROQ_API_KEY = env_str("GROQ_API_KEY")
GROQ_BASE_URL = env_str("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = env_str("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_MODEL = env_str("DEFAULT_MODEL", "claude-sonnet-4-20250514")

# LLM_PROVIDER: auto, openrouter, gpt, gemini, grok. groq is also accepted.
LLM_PROVIDER = env_str("LLM_PROVIDER", "auto")
SALES_LLM_MODEL = env_str("SALES_LLM_MODEL", OPENROUTER_MODEL)
VOICE_LLM_PROVIDER = env_str("VOICE_LLM_PROVIDER", LLM_PROVIDER)
VOICE_LLM_MODEL = env_str("VOICE_LLM_MODEL")
LOG_LEVEL = env_str("LOG_LEVEL", "INFO")

# Shared secret for Web -> FastAPI authentication.
AI_API_KEY = env_str("AI_API_KEY", "dev-secret-key-change-in-production")

# FPT.AI Text-to-Speech.
TTS_API_KEY = env_str("TTS_API_KEY")
TTS_VOICE = env_str("TTS_VOICE", "minhquang")
TTS_SPEED = env_str("TTS_SPEED", "1")

# ElevenLabs Text-to-Speech.
ELEVENLABS_API_KEY = env_str("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = env_str("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
ELEVENLABS_MODEL_ID = env_str("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5")
