"""
Sale Train Agent — FastAPI wrapper around the LangGraph sales simulator (MVP).
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

_frontend_url = os.getenv("FRONTEND_URL", "")
_allowed_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
if _frontend_url:
    _allowed_origins.append(_frontend_url)
_railway_public_url = os.getenv("RAILWAY_PUBLIC_URL")
if _railway_public_url:
    _allowed_origins.append(_railway_public_url)

from graph.graph import build_sales_graph
from core.schemas import (
    ChatRequest,
    ChatResponse,
    EvaluationPayload,
    FeedbackResponse,
    InitSessionRequest,
    InitSessionResponse,
    NextLevelResponse,
    RetryResponse,
    WebChatRequest,
    WebChatResponse,
    WebEvaluateRequest,
    WebEvaluateResponse,
    WebGenerateScenarioRequest,
    WebGenerateScenarioResponse,
)
from core.state import SalesSessionState, initial_sales_state
from api.auth import verify_api_key
from core.config import AI_API_KEY

app = FastAPI(
    title="Sale Train Agent API",
    description="B2B sales training — AI customer simulator and coach (LangGraph + FastAPI MVP)",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSION_STORE: dict[str, SalesSessionState] = {}

# Persona difficulty ladder — ordered from easiest to hardest (MVP: 3 personas).
PERSONA_DIFFICULTY_LADDER: list[str] = [
    "friendly_indecisive",
    "detail_oriented",
    "busy_skeptic",
]

_sales_graph = None


def get_sales_graph():
    global _sales_graph
    if _sales_graph is None:
        _sales_graph = build_sales_graph()
    return _sales_graph


def _normalize_slug(value: str) -> str:
    s = value.strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s or "generic"


# ================================================================
# Original CLI/Mobile endpoints (unchanged)
# ================================================================

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "Sale Train Agent API",
        "health": "/health",
    }


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10] if value else "empty"


@app.get("/debug/env")
def debug_env() -> dict[str, object]:
    raw_ai_api_key = os.getenv("AI_API_KEY", "")
    return {
        "has_raw_ai_api_key": bool(raw_ai_api_key),
        "raw_ai_api_key_len": len(raw_ai_api_key),
        "raw_ai_api_key_fp": _fingerprint(raw_ai_api_key.strip().strip("'\"")),
        "configured_ai_api_key_len": len(AI_API_KEY),
        "configured_ai_api_key_fp": _fingerprint(AI_API_KEY),
        "configured_is_default": AI_API_KEY == "dev-secret-key-change-in-production",
        "frontend_url": os.getenv("FRONTEND_URL", ""),
        "allowed_origins": _allowed_origins,
    }


@app.post("/init-session", response_model=InitSessionResponse)
def init_session(body: InitSessionRequest) -> InitSessionResponse:
    session_id = str(uuid.uuid4())
    scenario = _normalize_slug(body.scenario)
    persona = _normalize_slug(body.persona)
    state = initial_sales_state(
        session_id=session_id,
        scenario=scenario,
        persona=persona,
        max_turns=body.max_turns,
    )
    SESSION_STORE[session_id] = state
    return InitSessionResponse(
        session_id=session_id,
        scenario=scenario,
        persona=persona,
        max_turns=body.max_turns,
        current_status=state["current_status"],
    )


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    sid = body.session_id
    if sid not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session not found")

    prev = SESSION_STORE[sid]
    status = prev.get("current_status", "chatting")
    if status == "completed":
        raise HTTPException(
            status_code=409,
            detail="Session already completed. Start a new session or fetch /feedback.",
        )

    payload: SalesSessionState = {
        **prev,
        "last_user_message": body.message.strip(),
    }

    graph = get_sales_graph()
    merged = graph.invoke(payload)
    SESSION_STORE[sid] = merged

    ended = merged.get("current_status") == "completed"
    trace = list(merged.get("graph_trace") or [])
    return ChatResponse(
        session_id=sid,
        customer_reply=str(merged.get("customer_reply") or ""),
        current_status=merged.get("current_status", "chatting"),
        turn_count=int(merged.get("turn_count") or 0),
        graph_trace=trace,
        session_ended=ended,
        end_reason=merged.get("end_reason"),
    )


@app.get("/feedback/{session_id}", response_model=FeedbackResponse)
def feedback(session_id: str) -> FeedbackResponse:
    if session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session not found")

    st = SESSION_STORE[session_id]
    if st.get("current_status") != "completed":
        raise HTTPException(
            status_code=404,
            detail="Feedback not available until the session is completed.",
        )

    ev = st.get("evaluation_results")
    if not ev:
        raise HTTPException(status_code=500, detail="Session completed but evaluation is missing.")

    return FeedbackResponse(
        session_id=session_id,
        current_status="completed",
        evaluation_results=EvaluationPayload.model_validate(ev),
    )


@app.post("/session/retry/{session_id}", response_model=RetryResponse)
def retry_session(session_id: str) -> RetryResponse:
    """Reset chat history, keep same scenario & persona. Allows re-attempt."""
    if session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session not found")

    old = SESSION_STORE[session_id]
    scenario = str(old.get("scenario", ""))
    persona = str(old.get("persona", ""))
    max_turns = int(old.get("max_turns") or 12)

    new_state = initial_sales_state(
        session_id=session_id,
        scenario=scenario,
        persona=persona,
        max_turns=max_turns,
    )
    SESSION_STORE[session_id] = new_state
    return RetryResponse(
        session_id=session_id,
        scenario=scenario,
        persona=persona,
        max_turns=max_turns,
        current_status="chatting",
        message=f"Session reset. Same scenario '{scenario}' and persona '{persona}'. Good luck!",
    )


@app.post("/session/next-level/{session_id}", response_model=NextLevelResponse)
def next_level_session(session_id: str) -> NextLevelResponse:
    """Upgrade persona to the next difficulty level, reset chat history."""
    if session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Session not found")

    old = SESSION_STORE[session_id]
    scenario = str(old.get("scenario", ""))
    old_persona = str(old.get("persona", ""))
    max_turns = int(old.get("max_turns") or 12)

    # Find next persona in the ladder
    try:
        current_idx = PERSONA_DIFFICULTY_LADDER.index(old_persona)
    except ValueError:
        current_idx = -1  # unknown persona → start from first

    if current_idx >= len(PERSONA_DIFFICULTY_LADDER) - 1:
        raise HTTPException(
            status_code=400,
            detail=f"Persona '{old_persona}' is already the hardest level. No next level available.",
        )

    new_persona = PERSONA_DIFFICULTY_LADDER[current_idx + 1]
    new_state = initial_sales_state(
        session_id=session_id,
        scenario=scenario,
        persona=new_persona,
        max_turns=max_turns,
    )
    SESSION_STORE[session_id] = new_state
    return NextLevelResponse(
        session_id=session_id,
        scenario=scenario,
        old_persona=old_persona,
        new_persona=new_persona,
        max_turns=max_turns,
        current_status="chatting",
        message=f"Level up! Scenario '{scenario}' — new persona: '{new_persona}'. Challenge accepted!",
    )


# ================================================================
# Web App endpoints — Stateless, authenticated via API key
# ================================================================

@app.post("/web/chat", response_model=WebChatResponse)
async def web_chat(body: WebChatRequest, _: str = Depends(verify_api_key)):
    """Stateless chat endpoint for Next.js Web App.

    Receives all context from web (no SESSION_STORE).
    Returns structured JSON with session_should_end flag.
    """
    from llm.llm_client import generate_customer_turn_web
    from fastapi.concurrency import run_in_threadpool
    from tools.tts_client import generate_tts_fpt
    from core.config import TTS_API_KEY

    # Build conversation history as list of dicts
    history = [{"role": m.role, "content": m.content} for m in body.conversation_history]

    result = await run_in_threadpool(
        generate_customer_turn_web,
        customer_persona=body.customer_persona,
        scenario_title=body.scenario_title,
        scenario_description=body.scenario_description,
        company_context=body.company_context,
        document_contents=body.document_contents,
        conversation_history=history,
        last_user_message=body.message.strip(),
        current_turn=body.current_turn,
        max_turns=body.max_turns,
        ai_tone=body.ai_tone,
        follow_up_depth=body.follow_up_depth,
        time_remaining_seconds=body.time_remaining_seconds,
    )

    # Force end if max turns reached
    should_end = result.get("session_should_end", False)
    end_reason = result.get("end_reason")
    if body.current_turn >= body.max_turns:
        should_end = True
        end_reason = end_reason or "max_turns_reached"

    customer_reply_text = result.get("customer_message", "")
    audio_url = None
    if TTS_API_KEY and customer_reply_text:
        try:
            audio_url = await generate_tts_fpt(customer_reply_text)
        except Exception as e:
            print(f"TTS Error: {str(e)}")

    return WebChatResponse(
        customer_reply=customer_reply_text,
        session_should_end=should_end,
        end_reason=end_reason,
        turn_count=body.current_turn,
        audio_url=audio_url,
    )


@app.post("/web/evaluate", response_model=WebEvaluateResponse)
def web_evaluate(body: WebEvaluateRequest, _: str = Depends(verify_api_key)):
    """Evaluate session for Web App — 6 criteria + 3-layer anti-hallucination.

    Receives messages from Supabase DB, returns structured evaluation.
    """
    from llm.llm_client import generate_evaluation_web

    messages = [{"role": m.role, "content": m.content} for m in body.messages]

    result = generate_evaluation_web(
        messages=messages,
        scenario_title=body.scenario_title,
        scenario_description=body.scenario_description,
        customer_persona=body.customer_persona,
        document_contents=body.document_contents,
    )

    return WebEvaluateResponse(
        overall_score=result["overall_score"],
        rubric_breakdown=result["rubric_breakdown"],
        improvements=result.get("improvements", []),
        top_3_tips=result.get("top_3_tips", []),
    )


@app.post("/web/generate-scenario", response_model=WebGenerateScenarioResponse)
def web_generate_scenario(body: WebGenerateScenarioRequest, _: str = Depends(verify_api_key)):
    """Auto-generate scenario from document contents."""
    from llm.llm_client import generate_scenario_from_doc

    result = generate_scenario_from_doc(document_contents=body.document_contents)

    return WebGenerateScenarioResponse(
        title=result["title"],
        description=result["description"],
        company_context=result["company_context"],
        customer_persona=result["customer_persona"],
    )

@app.websocket("/ws/call/{session_id}")
async def call_room_websocket(
    websocket: WebSocket,
    session_id: str,
    persona: str = "A potential buyer",
    company_context: str | None = None,
):
    """Real-time Call Room Endpoint via OpenRouter + FPT TTS."""
    await websocket.accept()
    from fastapi.concurrency import run_in_threadpool
    from llm.llm_client import generate_customer_turn_web
    from tools.tts_client import generate_tts_fpt
    from core.config import TTS_API_KEY
    import json
    import time

    def elapsed_ms(start: float) -> int:
        return int((time.perf_counter() - start) * 1000)

    history = []

    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received WS data: {data}")
            try:
                msg = json.loads(data)
                if msg.get("type") != "user_text":
                    continue

                turn_start = time.perf_counter()
                user_text = msg["text"]
                time_remaining = msg.get("time_remaining_seconds")
                client_sent_at_ms = msg.get("client_sent_at_ms")
                client_stt_final_at_ms = msg.get("client_stt_final_at_ms")
                metrics = {
                    "session_id": session_id,
                    "text_length": len(user_text),
                    "client_sent_at_ms": client_sent_at_ms,
                    "client_stt_final_at_ms": client_stt_final_at_ms,
                }
                print(f"[voice-latency] ws_user_text_received {metrics}")
                print(f"User text received: {user_text}, Time remaining: {time_remaining}")
                history.append({"role": "user", "content": user_text})

                llm_start = time.perf_counter()
                print("Generating streaming LLM response...")
                
                from llm.llm_client import generate_customer_turn_voice_stream
                from tools.tts_client import wait_for_fpt_audio
                
                stream = generate_customer_turn_voice_stream(
                    customer_persona=persona,
                    scenario_title="Sales Call",
                    scenario_description="",
                    company_context=company_context,
                    conversation_history=history,
                    last_user_message=user_text,
                    current_turn=(len(history) // 2) + 1,
                    max_turns=12,
                    time_remaining_seconds=time_remaining,
                )
                
                full_bot_text = ""
                
                import asyncio
                from tools.tts_elevenlabs import generate_tts_elevenlabs_base64
                from core.config import ELEVENLABS_API_KEY
                
                async def handle_tts_bg(sentence_text: str, ws: WebSocket, tts_start_ms: float):
                    if ELEVENLABS_API_KEY:
                        # Dùng ElevenLabs Base64
                        b64_url = await generate_tts_elevenlabs_base64(sentence_text)
                        if b64_url:
                            total_tts_ms = int((time.perf_counter() - tts_start_ms) * 1000)
                            print(f"[voice-latency] ElevenLabs TTS Ready | Total: {total_tts_ms}ms")
                            await ws.send_json({
                                "type": "audio_url",
                                "url": b64_url,
                                "metrics": metrics,
                            })
                    elif TTS_API_KEY:
                        # Fallback FPT
                        # print(f"Generating FPT TTS for: {sentence_text}")
                        audio_url = await generate_tts_fpt(sentence_text)
                        if audio_url:
                            # print(f"Polling FPT URL: {audio_url}")
                            poll_start = time.perf_counter()
                            is_ready = await wait_for_fpt_audio(audio_url)
                            poll_elapsed = int((time.perf_counter() - poll_start) * 1000)
                            total_tts_ms = int((time.perf_counter() - tts_start_ms) * 1000)
                            
                            if is_ready:
                                print(f"[voice-latency] TTS Ready | Total: {total_tts_ms}ms | Polling: {poll_elapsed}ms | URL: {audio_url}")
                                await ws.send_json({
                                    "type": "audio_url",
                                    "url": audio_url,
                                    "metrics": metrics,
                                })
                            else:
                                print(f"[voice-latency] TTS Timeout | Polling: {poll_elapsed}ms | URL: {audio_url}")

                first_sentence = True
                async for sentence in stream:
                    if first_sentence:
                        ttfs = elapsed_ms(llm_start)
                        print(f"[voice-latency] LLM TTFS (Time To First Sentence): {ttfs}ms")
                        first_sentence = False
                        
                    full_bot_text += sentence + " "
                    # Gửi text từng câu
                    await websocket.send_json({
                        "type": "transcript",
                        "role": "assistant",
                        "text": sentence,
                        "is_partial": True,
                        "metrics": metrics,
                    })
                    
                    # Chạy TTS ngầm để không block quá trình sinh câu tiếp theo của LLM
                    asyncio.create_task(handle_tts_bg(sentence, websocket, time.perf_counter()))

                metrics["llm_ms"] = elapsed_ms(llm_start)
                metrics["backend_total_ms"] = elapsed_ms(turn_start)
                print(f"[voice-latency] backend_done {metrics}")
                
                history.append({"role": "assistant", "content": full_bot_text.strip()})
                
                # Gửi text final
                await websocket.send_json({
                    "type": "transcript",
                    "role": "assistant",
                    "text": full_bot_text.strip(),
                    "is_partial": False,
                    "metrics": metrics,
                })
                
                await websocket.send_json({
                    "type": "latency_metrics",
                    "metrics": metrics,
                })

            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")


