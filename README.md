# Sale Train Agent

B2B Sales Training Platform — AI-powered role-play simulator and coach built with FastAPI + LangGraph + LLM (OpenRouter / Gemini / Anthropic).

## Project Structure

```
.
├── agent.py              # CLI agent loop (Anthropic Claude)
├── api/
│   ├── main.py           # FastAPI app (REST + WebSocket endpoints)
│   └── auth.py           # API key authentication
├── cli/
│   └── cli_chat.py       # CLI chat interface
├── core/
│   ├── config.py         # Environment configuration
│   ├── schemas.py        # Pydantic request/response models
│   └── state.py          # LangGraph session state
├── graph/
│   ├── graph.py          # LangGraph sales simulation workflow
│   └── nodes/
│       ├── customer.py   # Customer simulation node
│       └── evaluator.py  # Evaluation node
├── llm/
│   ├── llm_client.py     # OpenAI JSON-mode helpers (customer + evaluator)
│   ├── gemini_live_client.py
│   └── prompts.py       # System prompts (personas, scenarios, rubrics)
└── tools/
    ├── calculate.py
    ├── fetch_url.py
    ├── search_web.py
    ├── tts_client.py     # FPT.AI TTS
    └── tts_elevenlabs.py # ElevenLabs TTS
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 3. Run the API server

```bash
uvicorn api.main:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`

### 4. Run the CLI agent

```bash
python agent.py
```

## Key Features

- **Customer Simulation**: AI personas (busy skeptic, friendly indecisive, detail-oriented) respond in real-time
- **Evaluation Rubric**: 5-criteria scoring (CLI) and 6-criteria scoring (Web) with anti-hallucination layers
- **Scenario System**: 3 built-in scenarios (Cold Call, Price Objection, Closing) + auto-generate from documents
- **Voice Support**: WebSocket endpoint with streaming TTS (FPT.AI / ElevenLabs)
- **Difficulty Ladder**: Persona progression from friendly → detail-oriented → busy skeptic

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/init-session` | Create training session |
| POST | `/chat` | Send sales rep message |
| GET | `/feedback/{session_id}` | Get evaluation results |
| POST | `/session/retry/{session_id}` | Reset session (same scenario) |
| POST | `/session/next-level/{session_id}` | Level up persona |
| POST | `/web/chat` | Stateless chat (Next.js) |
| POST | `/web/evaluate` | Evaluate session (Next.js) |
| POST | `/web/generate-scenario` | Auto-generate from document |
| WS | `/ws/call/{session_id}` | Real-time voice call |

## License

MIT
