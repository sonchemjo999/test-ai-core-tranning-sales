"""
OpenAI JSON-mode helpers for customer simulation and evaluation.
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator

from openai import OpenAI, AsyncOpenAI

from core.config import OPENAI_API_KEY, OPENAI_BASE_URL, SALES_LLM_MODEL, GEMINI_API_KEY, GEMINI_BASE_URL, OPEN_ROUTER_API, OPEN_ROUTER_BASE_URL, GROQ_API_KEY, GROQ_BASE_URL
from core.schemas import EvaluationPayload
from core.state import EvaluationResults, SalesSessionState
from llm.prompts import (
    CUSTOMER_SYSTEM_TEMPLATE,
    CUSTOMER_SYSTEM_TEMPLATE_WEB,
    EVALUATOR_SYSTEM_PROMPT,
    EVALUATOR_SYSTEM_PROMPT_WEB,
    GENERATE_SCENARIO_SYSTEM_PROMPT,
    persona_instruction,
    scenario_brief,
)


def _client() -> OpenAI:
    if OPEN_ROUTER_API:
        return OpenAI(
            base_url=OPEN_ROUTER_BASE_URL,
            api_key=OPEN_ROUTER_API,
            default_headers={
                "HTTP-Referer": "http://localhost:3000",
                "X-OpenRouter-Title": "A20-App",
            }
        )
    if GROQ_API_KEY:
        return OpenAI(
            base_url=GROQ_BASE_URL,
            api_key=GROQ_API_KEY,
        )
    if GEMINI_API_KEY:
        return OpenAI(
            api_key=GEMINI_API_KEY,
            base_url=GEMINI_BASE_URL,
        )
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "Cần cấu hình OPENAI_API_KEY, GEMINI_API_KEY, OPEN_ROUTER_API hoặc GROQ_API_KEY trong file .env."
        )
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


def _async_client() -> AsyncOpenAI:
    if GROQ_API_KEY:
        return AsyncOpenAI(
            base_url=GROQ_BASE_URL,
            api_key=GROQ_API_KEY,
        )
    if OPEN_ROUTER_API:
        return AsyncOpenAI(
            base_url=OPEN_ROUTER_BASE_URL,
            api_key=OPEN_ROUTER_API,
            default_headers={
                "HTTP-Referer": "http://localhost:3000",
                "X-OpenRouter-Title": "A20-App",
            }
        )
    if GEMINI_API_KEY:
        return AsyncOpenAI(
            api_key=GEMINI_API_KEY,
            base_url=GEMINI_BASE_URL,
        )
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "Cần cấu hình OPENAI_API_KEY, GEMINI_API_KEY, OPEN_ROUTER_API hoặc GROQ_API_KEY trong file .env."
        )
    return AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


def _chat_json(
    *,
    system: str,
    user: str,
    temperature: float = 0.6,
) -> dict[str, Any]:
    client = _client()
    
    # Auto fallback to supported model if the selected one isn't compatible
    if OPEN_ROUTER_API and model_name.startswith("gpt") and not OPENAI_API_KEY:
        model_name = "google/gemini-2.5-flash-lite"
    elif GEMINI_API_KEY and model_name.startswith("gpt") and not OPENAI_API_KEY:
        model_name = "gemini-2.5-flash-lite"
    elif GROQ_API_KEY and model_name.startswith("gpt") and not OPENAI_API_KEY:
        model_name = "llama-3.3-70b-versatile"

    completion = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    raw = completion.choices[0].message.content or "{}"
    return json.loads(raw)


def generate_customer_turn(state: SalesSessionState) -> dict[str, Any]:
    """Returns parsed JSON: customer_message, session_should_end, end_reason."""
    scenario_id = state.get("scenario", "negotiation")
    persona_id = state.get("persona", "skeptical")
    system = CUSTOMER_SYSTEM_TEMPLATE.format(
        scenario_id=scenario_id,
        scenario_brief=scenario_brief(scenario_id),
        persona_id=persona_id,
        persona_brief=persona_instruction(persona_id),
    )
    history = state.get("conversation_history", [])
    last_user = state.get("last_user_message", "").strip()
    transcript_lines: list[str] = []
    for m in history:
        role = "Sales rep" if m["role"] == "user" else "Buyer"
        transcript_lines.append(f"{role}: {m['content']}")
    transcript = "\n".join(transcript_lines) if transcript_lines else "(no prior turns)"
    user_payload = (
        f"Transcript so far:\n{transcript}\n\n"
        f"Sales rep just said:\n{last_user}\n\n"
        "Respond as the buyer JSON only."
    )
    data = _chat_json(system=system, user=user_payload, temperature=0.7)
    return {
        "customer_message": str(data.get("customer_message", "")).strip(),
        "session_should_end": bool(data.get("session_should_end", False)),
        "end_reason": data.get("end_reason"),
    }


RUBRIC_KEYS = (
    "understanding_needs",
    "response_structure",
    "objection_handling",
    "persuasiveness",
    "next_steps",
)


def _normalize_eval_raw(data: dict[str, Any]) -> dict[str, Any]:
    """Merge nested legacy 'scores' + legacy main_errors into the strict rubric shape."""
    out = dict(data)
    nested = out.pop("scores", None)
    if isinstance(nested, dict):
        for k in RUBRIC_KEYS:
            if k in nested and k not in out:
                out[k] = nested[k]
    if "key_mistakes" not in out or not out.get("key_mistakes"):
        legacy = out.get("main_errors")
        if isinstance(legacy, list):
            out["key_mistakes"] = legacy
    out.pop("main_errors", None)
    out.pop("overall_score", None)
    return out


def _clamp_int_score(v: Any, default: int = 5) -> int:
    try:
        iv = int(v)
    except (TypeError, ValueError):
        iv = default
    return max(1, min(10, iv))


def generate_evaluation(state: SalesSessionState) -> EvaluationResults:
    history = state.get("conversation_history", [])
    lines: list[str] = []
    for m in history:
        label = "Sales rep" if m["role"] == "user" else "Buyer"
        lines.append(f"{label}: {m['content']}")
    transcript = "\n".join(lines) if lines else "(empty)"
    user = (
        f"scenario_id: {state.get('scenario')}\n"
        f"persona_id: {state.get('persona')}\n"
        f"end_reason flag from sim: {state.get('end_reason')!r}\n\n"
        f"TRANSCRIPT:\n{transcript}"
    )
    raw = _chat_json(system=EVALUATOR_SYSTEM_PROMPT, user=user, temperature=0.3)
    merged = _normalize_eval_raw(raw)

    payload = EvaluationPayload(
        understanding_needs=_clamp_int_score(merged.get("understanding_needs")),
        response_structure=_clamp_int_score(merged.get("response_structure")),
        objection_handling=_clamp_int_score(merged.get("objection_handling")),
        persuasiveness=_clamp_int_score(merged.get("persuasiveness")),
        next_steps=_clamp_int_score(merged.get("next_steps")),
        strengths=list(merged.get("strengths") or [])[:12],
        key_mistakes=list(merged.get("key_mistakes") or [])[:12],
        suggested_better_answer=(
            str(merged.get("suggested_better_answer") or "").strip()
            or "N/A — provide a clearer senior-rep script after a longer exchange."
        ),
    )
    # model_dump includes computed overall_score
    return payload.model_dump()


# ================================================================
# Web App Functions — Used by /web/chat and /web/evaluate endpoints
# ================================================================

from llm.prompts import (
    CUSTOMER_SYSTEM_TEMPLATE_WEB,
    EVALUATOR_SYSTEM_PROMPT_WEB,
)

WEB_RUBRIC_KEYS = (
    "process_adherence",
    "talk_to_listen",
    "discovery_depth",
    "confidence",
    "objection_handling",
    "next_step",
)


def generate_customer_turn_web(
    *,
    customer_persona: str,
    scenario_title: str,
    scenario_description: str,
    company_context: str | None,
    document_contents: str | None = None,
    conversation_history: list[dict[str, str]],
    last_user_message: str,
    current_turn: int,
    max_turns: int,
    ai_tone: str = "neutral",
    follow_up_depth: str = "moderate",
    time_remaining_seconds: int | None = None,
) -> dict[str, Any]:
    """Customer turn for Web App — receives context directly, no slug lookup."""
    tone_map = {
        "friendly": "Phản hồi cởi mở, thân thiện, dễ tính, hay dùng từ ngữ tích cực.",
        "harsh": "Lạnh lùng, khắt khe, hay bắt bẻ câu chữ, tỏ thái độ hoài nghi cao.",
        "neutral": "Giữ thái độ khách quan, lịch sự nhưng giữ khoảng cách."
    }
    depth_map = {
        "light": "Đặt câu hỏi đơn giản, dễ dàng chấp nhận lời giải thích của Sales.",
        "deep": "Liên tục truy vấn sâu, lật lại vấn đề, không dễ dàng bị thuyết phục, hỏi vặn lại câu trả lời trước đó.",
        "moderate": "Hỏi lại 1-2 câu nếu chưa rõ, phản ứng ở mức bình thường."
    }

    time_instruction = ""
    if time_remaining_seconds is not None and time_remaining_seconds <= 60:
        time_instruction = f"\n\n[HỆ THỐNG: CHỈ CÒN {time_remaining_seconds} GIÂY LÀ KẾT THÚC CUỘC GỌI. Đã đến lúc phải chốt lại. Hãy chủ động nhắc đến việc sắp hết thời gian, hẹn lịch tiếp theo và chào tạm biệt ngay lập tức trong câu trả lời này!]"

    system = CUSTOMER_SYSTEM_TEMPLATE_WEB.format(
        scenario_title=scenario_title,
        scenario_description=scenario_description,
        customer_persona=customer_persona,
        ai_tone_instruction=f"Thái độ của bạn: {tone_map.get(ai_tone, tone_map['neutral'])}",
        follow_up_depth_instruction=f"Mức độ truy vấn: {depth_map.get(follow_up_depth, depth_map['moderate'])}",
        time_instruction=time_instruction,
    )

    # Inject company context + guardrails (ported from web route.ts)
    if company_context:
        system += (
            "\n\n== DƯỚI ĐÂY LÀ THÔNG TIN VỀ SẢN PHẨM/DỊCH VỤ CỦA CÔNG TY KINH DOANH ==\n"
            f"{company_context}\n\n"
            "LƯU Ý QUAN TRỌNG (GUARDRAILS):\n"
            "Bạn CHỈ sử dụng thông tin này làm nền tảng kiến thức để thử thách Sales. "
            "Tuyệt đối KHÔNG tự động liệt kê rập khuôn tính năng sản phẩm ra trừ khi Sales hỏi hoặc làm rõ. "
            "Hãy giấu kín thông tin này và chỉ nhả ra từng chút một qua các câu phản biện."
        )

    # Inject document contents (product specs, sales playbook)
    if document_contents:
        system += (
            "\n\n== TÀI LIỆU SẢN PHẨM / QUY TRÌNH BÁN HÀNG ==\n"
            f"{document_contents}\n\n"
            "GUARDRAILS TÀI LIỆU:\n"
            "- Bạn đã đọc tài liệu này và BIẾT rõ sản phẩm/dịch vụ.\n"
            "- Dùng kiến thức này để HỎI XÓAY và THỎ THÁCH Sales: hỏi về thông số, so sánh đối thủ, hỏi giá, hỏi SLA...\n"
            "- TUYỆT ĐỐI KHÔNG tự nói ra thông tin sản phẩm. Chỉ phản ứng khi Sales đề cập.\n"
            "- Nếu Sales nói SAI thông tin so với tài liệu, hãy phản bác lại một cách tự nhiên."
        )

    # Auto-end prompt at last turn
    if current_turn >= max_turns:
        system += (
            f"\n\n== HỆ THỐNG: ĐÂY LÀ LƯỢT CUỐI CÙNG (lượt {current_turn}/{max_turns}) ==\n"
            "Cuộc gọi sắp kết thúc. Hãy đưa ra phản hồi cuối cùng một cách TỰ NHIÊN "
            "(chốt deal, từ chối dứt khoát, hoặc hẹn lịch cụ thể) để kết thúc cuộc trò chuyện. "
            "KHÔNG kéo dài thêm."
        )

    # Build transcript
    transcript_lines: list[str] = []
    for m in conversation_history:
        role = "Sales rep" if m["role"] == "user" else "Buyer"
        transcript_lines.append(f"{role}: {m['content']}")
    transcript = "\n".join(transcript_lines) if transcript_lines else "(no prior turns)"

    user_payload = (
        f"Transcript so far:\n{transcript}\n\n"
        f"Sales rep just said:\n{last_user_message}\n\n"
        "Respond as the buyer JSON only."
    )

    data = _chat_json(system=system, user=user_payload, temperature=0.7)
    return {
        "customer_message": str(data.get("customer_message", "")).strip(),
        "session_should_end": bool(data.get("session_should_end", False)),
        "end_reason": data.get("end_reason"),
    }


def _validate_improvements(
    improvements: list[dict[str, Any]],
    sales_messages: list[str],
) -> list[dict[str, str]]:
    """Layer 2: Verify that improvement user_sentence exists in actual sales messages."""
    validated: list[dict[str, str]] = []
    for imp in improvements:
        candidate = str(imp.get("user_sentence", "")).strip()
        # Strip "Sales: " prefix if LLM added it
        import re
        normalized = re.sub(r"^[Ss]ales:\s*", "", candidate).strip()
        if not normalized:
            continue
        # Check: candidate must be a substring of at least one sales message
        is_found = any(
            normalized.lower() in msg.lower() or msg.lower() in normalized.lower()
            for msg in sales_messages
        )
        if is_found:
            validated.append({
                "user_sentence": normalized,
                "ai_suggestion": str(imp.get("ai_suggestion", "")),
                "playbook_source": imp.get("playbook_source"),
            })
    return validated[:3]


def generate_evaluation_web(
    *,
    messages: list[dict[str, str]],
    scenario_title: str,
    scenario_description: str,
    customer_persona: str,
    document_contents: str | None = None,
) -> dict[str, Any]:
    """Evaluate for Web App — 6 criteria + improvements + top_3_tips.

    Includes 3-layer anti-hallucination:
    - Layer 1: LLM generates structured JSON
    - Layer 2: Validate improvements against actual sales messages
    - Layer 3: Auto-retry if all improvements discarded
    """
    # Build transcript with clear role prefixes
    sales_messages: list[str] = []
    transcript_lines: list[str] = []
    for m in messages:
        if m["role"] == "user":
            transcript_lines.append(f"Sales: {m['content']}")
            sales_messages.append(m["content"].strip())
        else:
            transcript_lines.append(f"Khách hàng: {m['content']}")
    transcript = "\n".join(transcript_lines) if transcript_lines else "(empty)"

    user_payload = (
        f"Kịch bản: {scenario_title}\n"
        f"Mô tả: {scenario_description}\n"
        f"Tính cách khách: {customer_persona}\n\n"
        f"--- BEGIN TRANSCRIPT ---\n{transcript}\n--- END TRANSCRIPT ---"
    )

    # Build evaluator system prompt, optionally with document context
    eval_system = EVALUATOR_SYSTEM_PROMPT_WEB
    if document_contents:
        eval_system += (
            "\n\n== TÀI LIỆU THAM CHIẾU (QUY TRÌNH BÁN HÀNG / THÔNG TIN SẢN PHẨM) ==\n"
            f"{document_contents}\n\n"
            "HƯỚNG DẪN SỬ DỤNG TÀI LIỆU KHI CHẤM ĐIỂM:\n"
            "- Chấm `process_adherence` dựa trên QUY TRÌNH THỰC TẾ trong tài liệu (nếu có), không phải quy trình chung chung.\n"
            "- Nếu Sales nói SAI thông tin sản phẩm so với tài liệu, TRỪ NẶNG điểm `confidence` và ghi rõ lỗi.\n"
            "- Ở phần `improvements`, ƯU TIÊN dùng câu mẫu từ tài liệu làm `ai_suggestion` nếu phù hợp.\n"
            "- Ở phần `top_3_tips`, liên hệ với các bước cụ thể trong tài liệu thay vì khuyên chung."
        )

    # === LAYER 1: Call LLM ===
    raw = _chat_json(
        system=eval_system,
        user=user_payload,
        temperature=0.3,
    )

    # === LAYER 2: Validate improvements ===
    raw_improvements = raw.get("improvements", [])
    validated_improvements = _validate_improvements(raw_improvements, sales_messages)

    discarded_count = len(raw_improvements) - len(validated_improvements)
    if discarded_count > 0:
        print(f"[evaluate_web] ⚠️ Discarded {discarded_count}/{len(raw_improvements)} hallucinations")

    # === LAYER 3: Auto-retry if ALL discarded ===
    if raw_improvements and not validated_improvements:
        print("[evaluate_web] 🔄 All improvements discarded → Auto-retry...")
        retry_system = (
            EVALUATOR_SYSTEM_PROMPT_WEB
            + "\n\n== ⚠️ CẢNH BÁO NGHIÊM TRỌNG TỪ HỆ THỐNG ==\n"
            "LẦN TRƯỚC BẠN ĐÃ TRẢ VỀ CÂU BỊA (HALLUCINATION) — TẤT CẢ ĐỀU BỊ LOẠI.\n"
            "Hệ thống đã verify từng câu bạn trả về và KHÔNG TÌM THẤY câu nào trong transcript thực tế.\n"
            "QUY TẮC TUYỆT ĐỐI LẦN NÀY:\n"
            "- Bạn CHỈ được trích xuất NGUYÊN VĂN từ phần \"Sales: ...\" trong transcript.\n"
            "- KHÔNG được tự tạo câu, KHÔNG paraphrase, KHÔNG suy luận.\n"
            "- Nếu không có câu nào cần cải thiện → trả đúng: improvements: []"
        )
        retry_raw = _chat_json(system=retry_system, user=user_payload, temperature=0.3)
        validated_improvements = _validate_improvements(
            retry_raw.get("improvements", []), sales_messages
        )

    # Normalize rubric scores
    raw_rubric = raw.get("rubric_breakdown", {})
    rubric: dict[str, dict[str, Any]] = {}
    for key in WEB_RUBRIC_KEYS:
        entry = raw_rubric.get(key, {})
        if isinstance(entry, dict):
            rubric[key] = {
                "score": _clamp_int_score(entry.get("score", 5), default=5),
                "reason": str(entry.get("reason", "Không có nhận xét")),
            }
        else:
            rubric[key] = {"score": _clamp_int_score(entry, default=5), "reason": "Không có nhận xét"}

    # Compute overall score (0-100 scale)
    scores = [v["score"] for v in rubric.values()]
    overall_score = round(sum(scores) / len(scores) * 10, 1) if scores else 0.0

    # Ensure top_3_tips has exactly 3 items
    tips = list(raw.get("top_3_tips", []))[:3]
    while len(tips) < 3:
        tips.append("Luyện tập thêm để cải thiện kỹ năng bán hàng.")

    return {
        "overall_score": overall_score,
        "rubric_breakdown": rubric,
        "improvements": validated_improvements,
        "top_3_tips": tips,
    }



def generate_scenario_from_doc(*, document_contents: str) -> dict[str, Any]:
    """Auto-generate scenario details from an uploaded document."""
    user_payload = (
        "Đây là nội dung tài liệu của tôi:\n"
        "--- BEGIN DOCUMENT ---\n"
        f"{document_contents}\n"
        "--- END DOCUMENT ---\n\n"
        "Hãy tạo kịch bản JSON theo hướng dẫn."
    )
    
    data = _chat_json(
        system=GENERATE_SCENARIO_SYSTEM_PROMPT,
        user=user_payload,
        temperature=0.7,
    )
    
    return {
        "title": str(data.get("title", "Kịch bản tự động")),
        "description": str(data.get("description", "Không có mô tả")),
        "company_context": str(data.get("company_context", "Không có tóm tắt")),
        "customer_persona": str(data.get("customer_persona", "friendly_indecisive")),
    }


async def generate_customer_turn_voice_stream(
    *,
    customer_persona: str,
    scenario_title: str,
    scenario_description: str,
    company_context: str | None,
    conversation_history: list[dict[str, str]],
    last_user_message: str,
    current_turn: int,
    max_turns: int,
    time_remaining_seconds: int | None = None,
) -> AsyncGenerator[str, None]:
    """Voice-optimized streaming customer turn. Yields sentences as they are generated."""
    
    # Prompt is concise, not JSON mode, optimized for voice latency.
    time_instruction = ""
    if time_remaining_seconds is not None and time_remaining_seconds <= 60:
        time_instruction = f" CHỈ CÒN {time_remaining_seconds} GIÂY, hãy chủ động nhắc đến việc sắp hết thời gian và hẹn lịch tiếp theo để chốt lại cuộc gọi."

    system = (
        f"Bạn là khách hàng trong cuộc gọi luyện sales B2B. Kịch bản: {scenario_title}.\n"
        f"Tính cách: {customer_persona}.\n"
        f"{time_instruction}\n\n"
        "QUY TẮC PHẢN HỒI QUA GIỌNG NÓI:\n"
        "- Phản hồi cực kỳ tự nhiên, có cảm xúc và điệu bộ như người thật đang nghe điện thoại.\n"
        "- Độ dài vừa phải (khoảng 2-4 câu), đủ để phản biện logic hoặc thể hiện thái độ, nhưng không thao thao bất tuyệt.\n"
        "- Hãy dùng các từ ngữ đời thường, thỉnh thoảng có thể ngập ngừng (ờ, ừm) hoặc cắt ngang nếu bực mình.\n"
        "- Tuyệt đối KHÔNG gạch đầu dòng, KHÔNG dùng markdown, KHÔNG dùng văn viết.\n"
        "- KHÔNG dùng JSON, chỉ trả về chữ.\n"
    )
    
    if company_context:
        system += f"\nSản phẩm Sales đang bán:\n{company_context[:300]}\n(Dùng thông tin này để bắt bẻ, không tự khai ra).\n"

    if current_turn >= max_turns:
        system += "\nLƯỢT CUỐI: Đưa ra quyết định (chốt deal / từ chối) một cách dứt khoát.\n"

    transcript_lines: list[str] = []
    # Take only the last 4 turns to reduce prompt context and speed up TTFT (Time To First Token)
    recent_history = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
    for m in recent_history:
        role = "Sales rep" if m["role"] == "user" else "Buyer"
        transcript_lines.append(f"{role}: {m['content']}")
    transcript = "\n".join(transcript_lines) if transcript_lines else "(no prior turns)"

    user_payload = (
        f"Lịch sử:\n{transcript}\n\n"
        f"Sales vừa nói:\n{last_user_message}\n\n"
        "Hãy phản hồi ngắn gọn:"
    )

    client = _async_client()
    
    # Auto fallback to Gemini model if overriding OpenAI's default for speed
    import os
    if GROQ_API_KEY:
        model_name = os.getenv("VOICE_LLM_MODEL", "llama-3.3-70b-versatile")
    else:
        model_name = os.getenv("VOICE_LLM_MODEL", "google/gemini-2.5-flash-lite") if OPEN_ROUTER_API else SALES_LLM_MODEL
    
    response_stream = await client.chat.completions.create(
        model=model_name,
        temperature=0.6,
        max_tokens=80, # Prevent long rambling
        stream=True,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_payload},
        ],
    )

    buffer = ""
    sentence_end_chars = {'.', '!', '?', '\n'}
    
    async for chunk in response_stream:
        content = chunk.choices[0].delta.content
        if content:
            buffer += content
            
            while True:
                end_idx = -1
                for i, char in enumerate(buffer):
                    if char in sentence_end_chars:
                        end_idx = i
                        break
                
                if end_idx != -1:
                    sentence = buffer[:end_idx+1].strip()
                    if sentence:
                        yield sentence
                    buffer = buffer[end_idx+1:].lstrip()
                else:
                    break

    if buffer.strip():
        yield buffer.strip()

