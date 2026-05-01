# Sale Train Agent — Core Agent Workflow & Architecture

## 1. System Overview

Sale Train Agent là nền tảng đào tạo Sales B2B dựa trên AI, cho phép nhân viên Sales luyện tập đóng vai (role-play) với các buyer personas mô phỏng bởi LLM trong thời gian thực. Hệ thống được xây dựng với **FastAPI** + **LangGraph** + **LLM (OpenRouter / Gemini / Anthropic / Groq)**.

### 1.1 Technology Stack

```
Frontend (Next.js Web App)
         ↕  HTTP/REST + WebSocket
─────────────────────────────────────────────
FastAPI Server (api/main.py)
    ├── State Management: LangGraph StateGraph
    ├── LLM Layer: llm/llm_client.py
    │       ├── OpenAI / OpenRouter / Gemini (JSON-mode)
    │       └── Groq (async streaming)
    ├── TTS Layer: tools/tts_*.py
    │       ├── FPT.AI TTS v5
    │       └── ElevenLabs (Base64)
    └── Auth: API key verification (api/auth.py)
─────────────────────────────────────────────
External APIs:
    ├── OpenRouter / Gemini / Groq (LLM)
    ├── FPT.AI / ElevenLabs (TTS)
    └── Supabase DB (Web App — lưu transcripts)
```

---

## 2. Core Agent — LangGraph Workflow

Đây là trái tim của hệ thống, được sử dụng bởi CLI/Mobile endpoints (`/init-session`, `/chat`, `/feedback`).

### 2.1 State Graph Architecture

```mermaid
graph TD
    START([START]) --> CUSTOMER_NODE[customer_persona_node]
    
    CUSTOMER_NODE --> ROUTE{route_after_customer}
    
    ROUTE -- "session_should_end = True" --> EVAL_NODE[evaluation_node]
    ROUTE -- "session_should_end = False" --> END_NODE([END])
    
    EVAL_NODE --> END_NODE

    subgraph "State Fields"
        S1[session_id]
        S2[scenario / persona]
        S3[conversation_history]
        S4[turn_count / max_turns]
        S5[customer_reply]
        S6[session_should_end / end_reason]
        S7[evaluation_results]
        S8[graph_trace]
    end
```

### 2.2 Node: `customer_persona_node`

**Input:** `SalesSessionState` với `last_user_message`

**Process:**
1. Lấy `last_user_message` từ state
2. Gọi `generate_customer_turn(state)` → LLM JSON-mode
3. Đóng gói lịch sử hội thoại (user + assistant)
4. Kiểm tra turn cap (`new_turn >= max_turns`)
5. Trả về partial state update

**Output keys:**
```python
{
    "conversation_history": [user_turn, asst_turn],  # Annotated list — append-only
    "turn_count": new_turn,
    "customer_reply": str,
    "session_should_end": bool,
    "end_reason": str | None,
    "current_status": "chatting",
    "graph_trace": ["customer_persona"],
}
```

### 2.3 Node: `evaluation_node`

**Trigger:** Khi `session_should_end = True`

**Process:**
1. Gọi `generate_evaluation(state)` → LLM JSON-mode
2. Đánh giá toàn bộ `conversation_history` theo 5 tiêu chí rubric
3. Trả về structured feedback

**Output keys:**
```python
{
    "current_status": "completed",
    "evaluation_results": EvaluationResults,  # 5 scores + strengths + mistakes + suggestion
    "graph_trace": ["evaluation"],
}
```

### 2.4 Conditional Routing Logic

```mermaid
flowchart LR
    A[chat /chat endpoint] --> B[Build payload with last_user_message]
    B --> C[graph.invoke payload]
    C --> D[customer_persona_node runs]
    D --> E{session_should_end?}
    E -->|False| F[Return ChatResponse\ncustomer_reply + turn_count]
    E -->|True| G[evaluation_node runs]
    G --> H[evaluation_results populated]
    H --> I[Return ChatResponse\nsession_ended=True]
    
    style F fill:#bbf4d4
    style I fill:#ffd6a5
```

### 2.5 End Conditions (session_should_end)

| Trigger | `end_reason` |
|---------|-------------|
| LLM xác định deal đã đóng thành công | `"deal_closed"` |
| Khách hàng ngắt máy / từ chối | `"customer_hung_up"` |
| User nói "stop", "end session" | `"user_ended"` |
| `turn_count >= max_turns` | `"max_turns_reached"` |
| Vi phạm nghiêm trọng (nói sai thông tin) | `"violation"` |

---

## 3. CLI / Mobile Flow

```mermaid
sequenceDiagram
    participant CLI as CLI Agent (agent.py)
    participant API as FastAPI /api
    participant Graph as LangGraph
    participant LLM as LLM Client
    participant Ext as OpenRouter / Gemini

    CLI->>API: POST /init-session\n{s: scenario, p: persona, t: max_turns}
    API->>API: Generate session_id (UUID)
    API->>API: Create SalesSessionState\nStore in SESSION_STORE
    API-->>CLI: InitSessionResponse {session_id, status: chatting}

    loop Per user message
        CLI->>API: POST /chat\n{session_id, message}
        API->>Graph: graph.invoke({last_user_message, ...state})
        Graph->>LLM: generate_customer_turn(state)
        LLM->>Ext: POST /chat/completions (JSON mode)
        Ext-->>LLM: {customer_message, session_should_end, end_reason}
        LLM-->>Graph: parsed JSON
        Graph->>Graph: Check route_after_customer\nIf session_should_end → call evaluation_node
        Graph-->>API: merged SalesSessionState
        API-->>CLI: ChatResponse {customer_reply, turn_count, session_ended}
    end

    CLI->>API: GET /feedback/{session_id}
    API->>API: Check status == completed
    API-->>CLI: EvaluationPayload (5 rubric scores + feedback)
```

---

## 4. Web App Flow (Next.js — Stateless)

Web App sử dụng 3 endpoint riêng, không dùng LangGraph state graph. Mỗi request chứa toàn bộ context cần thiết.

### 4.1 Chat Flow

```mermaid
sequenceDiagram
    participant Web as Next.js Web App
    participant API as FastAPI /web/chat
    participant LLM as LLM Client
    participant TTS as TTS (FPT.AI / ElevenLabs)
    participant Ext as OpenRouter / Gemini

    Web->>API: POST /web/chat\n{message, conversation_history, scenario,\npersona, company_context, document, \nai_tone, follow_up_depth, time_remaining}

    API->>LLM: generate_customer_turn_web(...)
    LLM->>Ext: POST /chat/completions (JSON mode)
    Ext-->>LLM: {customer_message, session_should_end}
    LLM-->>API: parsed result

    alt TTS enabled + text non-empty
        API->>TTS: generate_tts_fpt(customer_reply_text)
        TTS-->>API: audio_url (FPT) or b64_data_uri (ElevenLabs)
    end

    alt turn >= max_turns
        API->>API: Override should_end = True
        API->>API: end_reason = "max_turns_reached"
    end

    API-->>Web: WebChatResponse\n{customer_reply, session_should_end, \naudio_url, turn_count}
```

### 4.2 Evaluation Flow (3-Layer Anti-Hallucination)

```mermaid
flowchart TD
    A[POST /web/evaluate] --> B[Build transcript\nwith Sales: / Khách hàng: prefixes]
    B --> C[Layer 1: LLM Evaluation]
    C --> D{Validate improvements\nagainst actual sales messages?}
    D -->|All valid| E[Return WebEvaluateResponse]
    D -->|Some discarded| E
    D -->|All discarded| F[Layer 3: Auto-retry with\nwarning prompt]
    F --> G{Validate retry improvements?}
    G -->|Still invalid| H[Return empty improvements\nwith warning logged]
    G -->|Valid found| E
    E --> I[overall_score computed\nfrom 6 rubric scores]
    I --> J[top_3_tips padded\nto exactly 3 items]

    style C fill:#cce5ff
    style F fill:#fff3cd
    style H fill:#f8d7da
    style E fill:#d4edda
```

### 4.3 Scenario Generation Flow

```mermaid
sequenceDiagram
    participant Web as Next.js (Admin)
    participant API as FastAPI /web/generate-scenario
    participant LLM as LLM Client
    participant Doc as Uploaded Document

    Web->>Doc: Upload & extract text
    Web->>API: POST /web/generate-scenario\n{document_contents}
    API->>LLM: generate_scenario_from_doc(...)
    LLM->>Ext: POST /chat/completions\n(GENERATE_SCENARIO_SYSTEM_PROMPT)
    Ext-->>LLM: {title, description, company_context, customer_persona}
    LLM-->>API: parsed fields
    API-->>Web: WebGenerateScenarioResponse\n→ Pre-fill Scenario form fields
```

---

## 5. Voice Call Flow (WebSocket — Real-time)

```mermaid
sequenceDiagram
    participant Browser as Browser (STT + TTS)
    participant API as FastAPI WS /ws/call/{session_id}
    participant LLM as LLM Client (Streaming)
    participant TTS as ElevenLabs / FPT.AI
    participant Ext as Groq / OpenRouter

    Browser->>API: WS Connect
    API-->>Browser: WebSocket accepted

    loop Per user utterance
        Browser->>API: {type: "user_text", text: "...", time_remaining_seconds}
        API->>LLM: generate_customer_turn_voice_stream(...)
        LLM->>Ext: POST /chat/completions (stream=True, model=llama/gemini)
        Ext-->>LLM: stream tokens (sentences)

        loop Per sentence generated
            LLM-->>API: sentence fragment
            API->>TTS: handle_tts_bg(sentence) [async, non-blocking]
            TTS-->>API: audio_url (FPT polling) or b64_data_uri (ElevenLabs)
            API-->>Browser: {type: "transcript", text: sentence, is_partial: true}
            alt audio ready
                API-->>Browser: {type: "audio_url", url: ...}
            end
        end

        API-->>Browser: {type: "transcript", is_partial: false, text: full}
        API-->>Browser: {type: "latency_metrics", metrics: {...}}
    end
```

### 5.1 Latency Metrics Tracked

| Metric | Description |
|--------|-------------|
| `text_length` | Số ký tự user message |
| `client_sent_at_ms` | Thời điểm client gửi |
| `client_stt_final_at_ms` | Thời điểm STT hoàn tất |
| `llm_ms` | Tổng thời gian LLM generate |
| `backend_total_ms` | Từ receive → gửi xong |
| TTS total + polling | Thời gian TTS + FPT polling |

---

## 6. Persona Difficulty Ladder

```mermaid
flowchart LR
    L1[friendly_indecisive] --> L2[detail_oriented] --> L3[busy_skeptic]

    L1:::easy
    L2:::medium
    L3:::hard

    classDef easy fill:#d4edda,stroke:#28a745
    classDef medium fill:#fff3cd,stroke:#ffc107
    classDef hard fill:#f8d7da,stroke:#dc3545
```

| Level | Persona | Đặc điểm | Scoring bias |
|-------|---------|-----------|-------------|
| 1 | `friendly_indecisive` | Ấm áp, quan tâm, nhưng trì hoãn mãi | Cần tạo urgency |
| 2 | `detail_oriented` | Phân tích, hỏi chi tiết, so sánh đối thủ | Cần deep product knowledge |
| 3 | `busy_skeptic` | Bận rộn, thiếu kiên nhẫn, yêu cầu hook nhanh | Cần conciseness + direct value |

---

## 7. LLM Model Routing

```mermaid
flowchart TD
    A[Request] --> B{Has OPEN_ROUTER_API?}

    B -->|Yes| C[OpenAI client → openrouter.ai/v1]
    B -->|No| D{Has GEMINI_API_KEY?}

    D -->|Yes| E[OpenAI client → generativelanguage.googleapis.com]
    D -->|No| F{Has GROQ_API_KEY?}

    F -->|Yes| G[AsyncOpenAI → api.groq.com/v1]
    F -->|No| H[OpenAI client → api.openai.com/v1]

    C --> I[Model: google/gemini-2.5-flash-lite\n(default, gpt* auto-fallback)]
    E --> J[Model: gemini-2.5-flash-lite]
    G --> K[Model: llama-3.3-70b-versatile (voice)]
    H --> L[Model: OPENAI_API_KEY default]

    style C fill:#e8f4fd
    style E fill:#fff8e1
    style G fill:#e8f5e9
    style H fill:#fce4ec
```

---

## 8. Evaluation Rubrics

### 8.1 CLI Rubric (5 criteria)

| Criterion | Mô tả |
|-----------|-------|
| `understanding_needs` | Discovery, listening, acknowledging pain |
| `response_structure` | Clarity, concision, professionalism |
| `objection_handling` | Empathy, reframe, non-defensive |
| `persuasiveness` | Value tied to buyer's problems |
| `next_steps` | Close / micro-commitment |

> **overall_score** = arithmetic mean của 5 scores (computed, not LLM-supplied)

### 8.2 Web Rubric (6 criteria)

| Criterion | Mô tả |
|-----------|-------|
| `process_adherence` | Đủ bước chào hỏi, nêu lý do gọi |
| `talk_to_listen` | Sale nói quá nhiều so với lắng nghe |
| `discovery_depth` | Dùng câu hỏi mở để hiểu vấn đề |
| `confidence` | Không dùng từ đệm thừa |
| `objection_handling` | Đồng cảm + nêu lợi ích |
| `next_step` | Thiết lập lịch hẹn / hành động cụ thể |

> **overall_score** = mean(scores) × 10 → scale 0–100

---

## 9. API Endpoint Summary

```mermaid
flowchart TD
    subgraph "CLI / Mobile Endpoints (LangGraph + SESSION_STORE)"
        E1[GET /health]
        E2[POST /init-session]
        E3[POST /chat]
        E4[GET /feedback/{session_id}]
        E5[POST /session/retry/{session_id}]
        E6[POST /session/next-level/{session_id}]
    end

    subgraph "Web App Endpoints (Stateless + API Key)"
        W1[POST /web/chat]
        W2[POST /web/evaluate]
        W3[POST /web/generate-scenario]
    end

    subgraph "Real-time Voice"
        V1[WS /ws/call/{session_id}]
    end

    style E2 fill:#bbf4d4
    style E3 fill:#ffd6a5
    style E4 fill:#d4c5f9
    style W1 fill:#bbf4d4
    style W2 fill:#d4c5f9
    style W3 fill:#e8f4fd
    style V1 fill:#fff3cd
```

---

## 10. Session Lifecycle

```mermaid
stateDiagram-v2
    [*] --> init_session: POST /init-session
    init_session --> chatting: SESSION_STORE[session_id]
    
    chatting --> chatting: POST /chat\n(no session_should_end)
    
    chatting --> evaluating: POST /chat\n(session_should_end=True)\    evaluating --> evaluating: evaluation_node runs
    
    evaluating --> completed: evaluation_results populated
    completed --> [*]
    
    chatting --> chatting: POST /session/retry/{id}\n(reset, same scenario)
    chatting --> chatting: POST /session/next-level/{id}\n(upgrade persona)
    completed --> chatting: POST /session/retry/{id}\n(new attempt)
```
