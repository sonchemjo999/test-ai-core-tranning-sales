"""
Sale Train Agent — FastAPI wrapper around the LangGraph sales simulator (MVP).
"""

from __future__ import annotations

import os
import re
import uuid

from fastapi import Body, FastAPI, HTTPException, Depends, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from graph.graph import build_sales_graph
from core.schemas import (
    ChatRequest,
    ChatResponse,
    DeepReadRequest,
    DeepReadResponse,
    EvaluationPayload,
    FeedbackResponse,
    InitSessionRequest,
    InitSessionResponse,
    NextLevelResponse,
    ParseDocumentResponse,
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

app = FastAPI(
    title="Sale Train Agent API",
    description="B2B sales training — AI customer simulator and coach (LangGraph + FastAPI MVP)",
    version="0.2.0",
)

# CORS — allow Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
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


@app.post("/init-session", response_model=InitSessionResponse)
def init_session(body: InitSessionRequest) -> InitSessionResponse:
    """[CLIENT: CLI / Mobile only]

    Create a new stateful session (stored in-memory SESSION_STORE).
    Web app does NOT use this — web manages sessions via Supabase DB
    and calls /web/chat (stateless). Reserved for CLI and future mobile client.

    DO NOT DELETE — this is the contract for mobile client.
    """
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
    """[CLIENT: CLI / Mobile only]

    Send a message within a stateful session (uses LangGraph + SESSION_STORE).
    Web app does NOT use this — web calls /web/chat which is stateless and
    receives all context from the client. Reserved for CLI and future mobile client.

    DO NOT DELETE — this is the contract for mobile client.
    """
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
    """[CLIENT: CLI / Mobile only]

    Retrieve evaluation results for a completed session from SESSION_STORE.
    Web app does NOT use this — web calls /web/evaluate with messages from
    Supabase DB. Reserved for CLI and future mobile client.

    DO NOT DELETE — this is the contract for mobile client.
    """
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
    """[CLIENT: CLI / Mobile only]

    Reset chat history, keep same scenario & persona. Allows re-attempt.
    Web app does NOT use this — web creates a new Supabase session instead.
    Reserved for CLI and future mobile client.

    DO NOT DELETE — this is the contract for mobile client.
    """
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
    """[CLIENT: CLI / Mobile only]

    Upgrade persona to the next difficulty level, reset chat history.
    Web app does NOT use this — web handles difficulty selection via UI
    and creates a new Supabase session. Reserved for CLI and future mobile client.

    DO NOT DELETE — this is the contract for mobile client.
    """
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


@app.post("/api/v1/training/sessions/{session_id}/avatar-token")
def create_avatar_token(
    session_id: str,
    body: dict = Body(default_factory=dict),
    _: str = Depends(verify_api_key),
) -> dict[str, str | int]:
    """Issue a WebRTC avatar signaling token for phone clients."""
    from api.ws_auth import issue_ws_token, TTL

    gender = str(body.get("gender") or "male")
    user_id = str(body.get("user_id") or "phone")
    token, exp = issue_ws_token(session_id=session_id, user_id=user_id, ttl_seconds=TTL)
    ws_base = (
        os.environ.get("CORE_AI_PUBLIC_WS_URL")
        or os.environ.get("PUBLIC_WS_URL")
        or "ws://localhost:8001"
    ).rstrip("/")
    query = f"token={token}&gender={gender}"
    return {
        "avatarSignalingWsUrl": f"{ws_base}/ws/signal/{session_id}?{query}",
        "avatarSessionToken": token,
        "expiresAt": exp,
    }

@app.post("/web/parse-document", response_model=ParseDocumentResponse)
async def web_parse_document(
    file: UploadFile,
    _: str = Depends(verify_api_key),
):
    """Parse an uploaded document (PDF, DOCX, XLSX, CSV, TXT) into clean Markdown.

    Phase 1 of Hybrid NotebookLM Architecture — Ingestion Pipeline.
    Returns structured Markdown text preserving tables and headings.
    """
    from tools.document_processor import process_document, DocumentProcessingError

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Limit file size to 20MB
    if len(file_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 20MB.")

    try:
        result = process_document(
            file_bytes=file_bytes,
            filename=file.filename,
            content_type=file.content_type,
        )
    except DocumentProcessingError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return ParseDocumentResponse(**result)


@app.post("/web/deep-read", response_model=DeepReadResponse)
def web_deep_read(body: DeepReadRequest, _: str = Depends(verify_api_key)):
    """Phase 2 — NotebookLM Deep Read: Extract CheatSheet + Scenarios from document.

    Receives raw_markdown from Phase 1, sends to high-context LLM (Gemini 2.5)
    for structured knowledge extraction. Returns CheatSheet + training Scenarios.
    """
    from llm.llm_client import generate_deep_read

    if len(body.raw_markdown) > 500_000:
        raise HTTPException(
            status_code=413,
            detail="Document too large for Deep Read. Maximum ~500K characters.",
        )

    try:
        result = generate_deep_read(raw_markdown=body.raw_markdown)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"[deep-read] LLM extraction failed: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"AI extraction failed: {str(e)}",
        )

    return DeepReadResponse(**result)


@app.websocket("/ws/signal/{session_id}")
async def avatar_signal_websocket(
    websocket: WebSocket,
    session_id: str,
    token: str | None = None,
    gender: str = "male",
):
    """WebRTC signaling endpoint for phone avatar video."""
    from api.ws_auth import verify_ws_token

    if not token:
        await websocket.close(code=1008, reason="Missing auth token")
        return

    valid, reason = verify_ws_token(token, session_id)
    if not valid:
        await websocket.close(code=1008, reason=f"Auth failed: {reason}")
        return

    await websocket.accept()
    try:
        from api.webrtc_agent import AvatarWebRTCAgent

        agent = AvatarWebRTCAgent(session_id=session_id, gender=gender)
        await agent.run_signaling(websocket)
    except WebSocketDisconnect:
        print(f"Avatar signaling disconnected: {session_id}")
    except Exception as e:
        print(f"Avatar signaling error for {session_id}: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
        try:
            await websocket.close(code=1011, reason="Avatar signaling error")
        except Exception:
            pass


@app.websocket("/ws/call/{session_id}")
async def call_room_websocket(
    websocket: WebSocket,
    session_id: str,
    persona: str = "A potential buyer",
    company_context: str | None = None,
    cheat_sheet: str | None = None,
    token: str | None = None,
    gender: str = "male",
):
    """Real-time Call Room Endpoint via OpenRouter + FPT TTS.

    Phase 3: Accepts optional `cheat_sheet` query param (compact JSON from Phase 2).
    If provided, it replaces company_context for more accurate AI customer behavior.
    """
    # ── Token auth ──────────────────────────────────────────────
    from api.ws_auth import verify_ws_token

    if not token:
        await websocket.close(code=1008, reason="Missing auth token")
        return

    valid, reason = verify_ws_token(token, session_id)
    if not valid:
        await websocket.close(code=1008, reason=f"Auth failed: {reason}")
        return
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
    _turn_counter = 0  # TIP-003 INSTRUMENTATION - REMOVE AFTER

    try:
        while True:
            data = await websocket.receive_text()
            safe_data = data.encode('ascii', 'backslashreplace').decode('ascii')
            print(f"Received WS data: {safe_data}")
            try:
                msg = json.loads(data)
                if msg.get("type") != "user_text":
                    continue

                turn_start = time.perf_counter()
                _turn_counter += 1  # TIP-003 INSTRUMENTATION - REMOVE AFTER
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
                safe_text = user_text.encode('ascii', 'backslashreplace').decode('ascii')
                print(f"User text received: {safe_text}, Time remaining: {time_remaining}")
                # TIP-003 INSTRUMENTATION - REMOVE AFTER: log turn number
                print(f"[LATENCY][TIP-003] ===== TURN {_turn_counter} START =====")
                if "conversation_history" in msg:
                    history = msg["conversation_history"]
                history.append({"role": "user", "content": user_text})

                # TIP-003 INSTRUMENTATION - REMOVE AFTER: measure import + client creation overhead
                pre_llm_start = time.perf_counter()
                llm_start = time.perf_counter()
                print(f"[LATENCY][TIP-003] turn={_turn_counter} | pre_llm_overhead (history+imports): {(llm_start - turn_start)*1000:.0f}ms")
                print("Generating streaming LLM response...")
                
                from llm.llm_client import generate_customer_turn_voice_stream
                from tools.tts_client import wait_for_fpt_audio
                
                # TIP-003 INSTRUMENTATION - REMOVE AFTER: measure stream creation (includes async client init)
                stream_create_start = time.perf_counter()
                stream = generate_customer_turn_voice_stream(
                    customer_persona=persona,
                    scenario_title="Sales Call",
                    scenario_description="",
                    company_context=company_context,
                    cheat_sheet=cheat_sheet,
                    conversation_history=history,
                    last_user_message=user_text,
                    current_turn=(len(history) // 2) + 1,
                    max_turns=12,
                    time_remaining_seconds=time_remaining,
                )
                print(f"[LATENCY][TIP-003] turn={_turn_counter} | stream_create: {(time.perf_counter() - stream_create_start)*1000:.0f}ms")

                import asyncio
                from tools.tts_cartesia import generate_tts_cartesia_base64, generate_tts_cartesia_stream

                # Check Cartesia availability once per turn
                from core.config import CARTESIA_API_KEY as _ckey, CARTESIA_FEMALE_VOICE_ID, CARTESIA_MALE_VOICE_ID
                _use_cartesia = bool(_ckey)

                full_bot_text = ""
                
                async def generate_tts_task(sentence_text: str, tts_start_ms: float):
                    """Cartesia / FPT TTS — returns full audio as single blob."""
                    if _ckey:
                        voice_id = CARTESIA_FEMALE_VOICE_ID if gender == "female" else CARTESIA_MALE_VOICE_ID
                        b64_url, audio_fmt = await generate_tts_cartesia_base64(sentence_text, voice_id)
                        return b64_url, tts_start_ms, audio_fmt
                    elif TTS_API_KEY:
                        from tools.tts_client import generate_tts_fpt, wait_for_fpt_audio
                        audio_url = await generate_tts_fpt(sentence_text)
                        if audio_url:
                            is_ready = await wait_for_fpt_audio(audio_url)
                            if is_ready:
                                return audio_url, tts_start_ms, "mp3"
                    return None, tts_start_ms, "mp3"

                tts_queue = asyncio.Queue()

                async def tts_sender_worker(ws: WebSocket):
                    """Process TTS queue — handles both streaming chunks and full blobs."""
                    while True:
                        item = await tts_queue.get()
                        if item is None:
                            break

                        kind, payload = item

                        if kind == "stream":
                            # Cartesia streaming: yield chunks as they arrive
                            sentence_text, tts_start_ms, _voice_id = payload
                            chunk_idx = 0
                            total_bytes = 0
                            async for b64_chunk in generate_tts_cartesia_stream(sentence_text, _voice_id):
                                pcm_len = len(b64_chunk) * 3 // 4  # approximate decoded size
                                total_bytes += pcm_len
                                if chunk_idx == 0:
                                    ttfc_ms = int((time.perf_counter() - tts_start_ms) * 1000)
                                    print(f"[Cartesia] TTFC: {ttfc_ms}ms | chunk 0: ~{pcm_len} bytes")
                                await ws.send_json({
                                    "type": "audio_chunk",
                                    "data": b64_chunk,
                                    "format": "pcm16",
                                    "chunk_index": chunk_idx,
                                    "metrics": metrics,
                                })
                                chunk_idx += 1

                            # Signal end of this sentence's audio
                            total_ms = int((time.perf_counter() - tts_start_ms) * 1000)
                            print(f"[Cartesia] Stream done | {chunk_idx} chunks | ~{total_bytes:,} bytes | {total_ms}ms")
                            await ws.send_json({
                                "type": "audio_chunk_end",
                                "format": "pcm16",
                                "total_chunks": chunk_idx,
                                "metrics": metrics,
                            })
                        
                        else:
                            # Cartesia/FPT: await full task result
                            task = payload
                            audio_url, tts_start_ms, audio_fmt = await task
                            if audio_url:
                                total_tts_ms = int((time.perf_counter() - tts_start_ms) * 1000)
                                print(f"[voice-latency] TTS Ready | Total: {total_tts_ms}ms | format: {audio_fmt}")
                                await ws.send_json({
                                    "type": "audio_url",
                                    "url": audio_url,
                                    "format": audio_fmt,
                                    "metrics": metrics,
                                })
                        
                        tts_queue.task_done()

                sender_task = asyncio.create_task(tts_sender_worker(websocket))

                first_sentence = True
                first_tts_dispatched = False  # TIP-003 INSTRUMENTATION - REMOVE AFTER
                async for sentence in stream:
                    if first_sentence:
                        ttfs = elapsed_ms(llm_start)
                        print(f"[voice-latency] LLM TTFS (Time To First Sentence): {ttfs}ms")
                        # TIP-003 INSTRUMENTATION - REMOVE AFTER
                        print(f"[LATENCY][TIP-003] turn={_turn_counter} | TTFS: {ttfs}ms | total_from_ws_recv: {elapsed_ms(turn_start)}ms")
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
                    # TIP-003 INSTRUMENTATION - REMOVE AFTER: log first TTS dispatch
                    if not first_tts_dispatched:
                        print(f"[LATENCY][TIP-003] turn={_turn_counter} | first_tts_dispatch: {elapsed_ms(turn_start)}ms from turn_start")
                        first_tts_dispatched = True
                    
                    if _use_cartesia:
                        # Cartesia streaming: put generator metadata into queue
                        _voice_id = CARTESIA_FEMALE_VOICE_ID if gender == "female" else CARTESIA_MALE_VOICE_ID
                        await tts_queue.put(("stream", (sentence, time.perf_counter(), _voice_id)))
                    else:
                        # FPT TTS: create task and put into queue
                        task = asyncio.create_task(generate_tts_task(sentence, time.perf_counter()))
                        await tts_queue.put(("task", task))

                # Chờ xử lý xong tất cả các câu trong queue
                await tts_queue.put(None)
                await sender_task

                metrics["llm_ms"] = elapsed_ms(llm_start)
                metrics["backend_total_ms"] = elapsed_ms(turn_start)
                metrics["turn_number"] = _turn_counter  # TIP-003 INSTRUMENTATION - REMOVE AFTER
                print(f"[voice-latency] backend_done {metrics}")
                # TIP-003 INSTRUMENTATION - REMOVE AFTER: summary line
                print(f"[LATENCY][TIP-003] turn={_turn_counter} | SUMMARY: llm_total={metrics['llm_ms']}ms | backend_total={metrics['backend_total_ms']}ms")
                
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


