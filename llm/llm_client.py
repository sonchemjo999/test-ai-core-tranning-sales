"""
OpenAI JSON-mode helpers for customer simulation and evaluation.
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator

from openai import OpenAI, AsyncOpenAI

from core.config import (
    GEMINI_API_KEY,
    GEMINI_BASE_URL,
    GEMINI_MODEL,
    GROK_API_KEY,
    GROK_BASE_URL,
    GROK_MODEL,
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OPEN_ROUTER_API,
    OPEN_ROUTER_BASE_URL,
    OPENROUTER_MODEL,
    SALES_LLM_MODEL,
    VOICE_LLM_MODEL,
    VOICE_LLM_PROVIDER,
)
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


PROVIDER_ALIASES = {
    "": "auto",
    "auto": "auto",
    "openrouter": "openrouter",
    "open-router": "openrouter",
    "open_router": "openrouter",
    "router": "openrouter",
    "gpt": "gpt",
    "openai": "gpt",
    "gemini": "gemini",
    "google": "gemini",
    "grok": "grok",
    "xai": "grok",
    "x-ai": "grok",
    "groq": "groq",
}


def _normalize_provider(provider: str | None) -> str:
    key = (provider or "auto").strip().lower()
    if key not in PROVIDER_ALIASES:
        raise ValueError("Provider khong hop le. Dung: auto, openrouter, gpt, gemini, grok.")
    return PROVIDER_ALIASES[key]


def _provider_config(provider: str) -> tuple[str, str, str, dict[str, str] | None]:
    if provider == "openrouter":
        return (
            OPEN_ROUTER_API,
            OPEN_ROUTER_BASE_URL,
            SALES_LLM_MODEL or OPENROUTER_MODEL,
            {
                "HTTP-Referer": "http://localhost:3000",
                "X-OpenRouter-Title": "A20-App",
            },
        )
    if provider == "gpt":
        return OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, None
    if provider == "gemini":
        return GEMINI_API_KEY, GEMINI_BASE_URL, GEMINI_MODEL, None
    if provider == "grok":
        return GROK_API_KEY, GROK_BASE_URL, GROK_MODEL, None
    if provider == "groq":
        return GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL, None
    raise RuntimeError(f"Provider khong ho tro: {provider}")


def _resolve_provider(provider: str | None) -> str:
    selected = _normalize_provider(provider)
    if selected != "auto":
        return selected
    if OPEN_ROUTER_API:
        return "openrouter"
    if OPENAI_API_KEY:
        return "gpt"
    if GEMINI_API_KEY:
        return "gemini"
    if GROK_API_KEY:
        return "grok"
    if GROQ_API_KEY:
        return "groq"
    raise RuntimeError("Can cau hinh it nhat mot key: OPEN_ROUTER_API, OPENAI_API_KEY, GEMINI_API_KEY, GROK_API_KEY.")


def _resolve_model_and_provider(
    *,
    model_name: str | None,
    provider: str | None,
    default_model: str,
    default_provider: str,
) -> tuple[str, str]:
    selected_provider = _resolve_provider(provider or default_provider)
    api_key, _base_url, provider_model, _headers = _provider_config(selected_provider)
    if not api_key:
        raise RuntimeError(f"Chua cau hinh API key cho provider {selected_provider}.")
    selected_model = (model_name or default_model or provider_model).strip()
    return selected_model, selected_provider


def _selected_client(provider: str) -> OpenAI:
    api_key, base_url, _model, headers = _provider_config(provider)
    if not api_key:
        raise RuntimeError(f"Chua cau hinh API key cho provider {provider}.")
    return OpenAI(api_key=api_key, base_url=base_url, default_headers=headers)


def _selected_async_client(provider: str) -> AsyncOpenAI:
    api_key, base_url, _model, headers = _provider_config(provider)
    if not api_key:
        raise RuntimeError(f"Chua cau hinh API key cho provider {provider}.")
    return AsyncOpenAI(api_key=api_key, base_url=base_url, default_headers=headers)


def _chat_json(
    *,
    system: str,
    user: str,
    temperature: float = 0.6,
    model_name: str | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    model_name, provider = _resolve_model_and_provider(
        model_name=model_name,
        provider=provider,
        default_model="",
        default_provider=LLM_PROVIDER,
    )
    client = _selected_client(provider)
    _api_key, base_url, _provider_model, _headers = _provider_config(provider)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        completion = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=messages,
        )
    except Exception as exc:
        is_auth_error = any(
            marker in str(exc) or marker in type(exc).__name__
            for marker in ["401", "403", "AuthenticationError", "PermissionDenied"]
        )
        # Retry without response_format if JSON mode is unsupported.
        if "response_format" in str(exc) or "json_object" in str(exc):
            completion = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                messages=messages,
            )
        # Auth errors — try other available providers in priority order.
        elif is_auth_error:
            tried = {provider}
            priority = ["gpt", "gemini", "grok", "groq", "openrouter"]
            for next_provider in priority:
                if next_provider in tried:
                    continue
                api_key, _, next_model, _ = _provider_config(next_provider)
                if api_key:
                    try:
                        next_client = _selected_client(next_provider)
                        completion = next_client.chat.completions.create(
                            model=model_name,
                            temperature=temperature,
                            response_format={"type": "json_object"},
                            messages=messages,
                        )
                        print(f"[llm] fallback: {provider} → {next_provider} for model={model_name}")
                        break
                    except Exception:
                        tried.add(next_provider)
                        continue
            else:
                raise RuntimeError(
                    f"LLM provider error ({provider}, model={model_name}, base_url={base_url}): {exc}"
                ) from exc
        else:
            print(
                "[llm] chat completion failed",
                {
                    "provider": provider,
                    "model": model_name,
                    "base_url": base_url,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                },
            )
            raise RuntimeError(
                f"LLM provider error ({provider}, model={model_name}, base_url={base_url}): {exc}"
            ) from exc
    raw = completion.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        print(
            "[llm] invalid json response",
            {
                "provider": provider,
                "model": model_name,
                "base_url": base_url,
                "raw": raw[:1000],
            },
        )
        raise RuntimeError(f"LLM returned invalid JSON ({provider}, model={model_name}).") from exc


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
            or "N/A â€” provide a clearer senior-rep script after a longer exchange."
        ),
    )
    # model_dump includes computed overall_score
    return payload.model_dump()


# ================================================================
# Web App Functions â€” Used by /web/chat and /web/evaluate endpoints
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
    llm_provider: str | None = None,
    llm_model: str | None = None,
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
    """Customer turn for Web App â€” receives context directly, no slug lookup."""
    tone_map = {
        "friendly": "Pháº£n há»“i cá»Ÿi má»Ÿ, thÃ¢n thiá»‡n, dá»… tÃ­nh, hay dÃ¹ng tá»« ngá»¯ tÃ­ch cá»±c.",
        "harsh": "Láº¡nh lÃ¹ng, kháº¯t khe, hay báº¯t báº» cÃ¢u chá»¯, tá» thÃ¡i Ä‘á»™ hoÃ i nghi cao.",
        "neutral": "Giá»¯ thÃ¡i Ä‘á»™ khÃ¡ch quan, lá»‹ch sá»± nhÆ°ng giá»¯ khoáº£ng cÃ¡ch."
    }
    depth_map = {
        "light": "Äáº·t cÃ¢u há»i Ä‘Æ¡n giáº£n, dá»… dÃ ng cháº¥p nháº­n lá»i giáº£i thÃ­ch cá»§a Sales.",
        "deep": "LiÃªn tá»¥c truy váº¥n sÃ¢u, láº­t láº¡i váº¥n Ä‘á», khÃ´ng dá»… dÃ ng bá»‹ thuyáº¿t phá»¥c, há»i váº·n láº¡i cÃ¢u tráº£ lá»i trÆ°á»›c Ä‘Ã³.",
        "moderate": "Há»i láº¡i 1-2 cÃ¢u náº¿u chÆ°a rÃµ, pháº£n á»©ng á»Ÿ má»©c bÃ¬nh thÆ°á»ng."
    }

    time_instruction = ""
    if time_remaining_seconds is not None and time_remaining_seconds <= 60:
        time_instruction = f"\n\n[Há»† THá»NG: CHá»ˆ CÃ’N {time_remaining_seconds} GIÃ‚Y LÃ€ Káº¾T THÃšC CUá»˜C Gá»ŒI. ÄÃ£ Ä‘áº¿n lÃºc pháº£i chá»‘t láº¡i. HÃ£y chá»§ Ä‘á»™ng nháº¯c Ä‘áº¿n viá»‡c sáº¯p háº¿t thá»i gian, háº¹n lá»‹ch tiáº¿p theo vÃ  chÃ o táº¡m biá»‡t ngay láº­p tá»©c trong cÃ¢u tráº£ lá»i nÃ y!]"

    system = CUSTOMER_SYSTEM_TEMPLATE_WEB.format(
        scenario_title=scenario_title,
        scenario_description=scenario_description,
        customer_persona=customer_persona,
        ai_tone_instruction=f"ThÃ¡i Ä‘á»™ cá»§a báº¡n: {tone_map.get(ai_tone, tone_map['neutral'])}",
        follow_up_depth_instruction=f"Má»©c Ä‘á»™ truy váº¥n: {depth_map.get(follow_up_depth, depth_map['moderate'])}",
        time_instruction=time_instruction,
    )

    # Inject company context + guardrails (ported from web route.ts)
    if company_context:
        system += (
            "\n\n== DÆ¯á»šI ÄÃ‚Y LÃ€ THÃ”NG TIN Vá»€ Sáº¢N PHáº¨M/Dá»ŠCH Vá»¤ Cá»¦A CÃ”NG TY KINH DOANH ==\n"
            f"{company_context}\n\n"
            "LÆ¯U Ã QUAN TRá»ŒNG (GUARDRAILS):\n"
            "Báº¡n CHá»ˆ sá»­ dá»¥ng thÃ´ng tin nÃ y lÃ m ná»n táº£ng kiáº¿n thá»©c Ä‘á»ƒ thá»­ thÃ¡ch Sales. "
            "Tuyá»‡t Ä‘á»‘i KHÃ”NG tá»± Ä‘á»™ng liá»‡t kÃª ráº­p khuÃ´n tÃ­nh nÄƒng sáº£n pháº©m ra trá»« khi Sales há»i hoáº·c lÃ m rÃµ. "
            "HÃ£y giáº¥u kÃ­n thÃ´ng tin nÃ y vÃ  chá»‰ nháº£ ra tá»«ng chÃºt má»™t qua cÃ¡c cÃ¢u pháº£n biá»‡n."
        )

    # Inject document contents (product specs, sales playbook)
    if document_contents:
        system += (
            "\n\n== TÃ€I LIá»†U Sáº¢N PHáº¨M / QUY TRÃŒNH BÃN HÃ€NG ==\n"
            f"{document_contents}\n\n"
            "GUARDRAILS TÃ€I LIá»†U:\n"
            "- Báº¡n Ä‘Ã£ Ä‘á»c tÃ i liá»‡u nÃ y vÃ  BIáº¾T rÃµ sáº£n pháº©m/dá»‹ch vá»¥.\n"
            "- DÃ¹ng kiáº¿n thá»©c nÃ y Ä‘á»ƒ Há»ŽI XÃ“AY vÃ  THá»Ž THÃCH Sales: há»i vá» thÃ´ng sá»‘, so sÃ¡nh Ä‘á»‘i thá»§, há»i giÃ¡, há»i SLA...\n"
            "- TUYá»†T Äá»I KHÃ”NG tá»± nÃ³i ra thÃ´ng tin sáº£n pháº©m. Chá»‰ pháº£n á»©ng khi Sales Ä‘á» cáº­p.\n"
            "- Náº¿u Sales nÃ³i SAI thÃ´ng tin so vá»›i tÃ i liá»‡u, hÃ£y pháº£n bÃ¡c láº¡i má»™t cÃ¡ch tá»± nhiÃªn."
        )

    # Auto-end prompt at last turn
    if current_turn >= max_turns:
        system += (
            f"\n\n== Há»† THá»NG: ÄÃ‚Y LÃ€ LÆ¯á»¢T CUá»I CÃ™NG (lÆ°á»£t {current_turn}/{max_turns}) ==\n"
            "Cuá»™c gá»i sáº¯p káº¿t thÃºc. HÃ£y Ä‘Æ°a ra pháº£n há»“i cuá»‘i cÃ¹ng má»™t cÃ¡ch Tá»° NHIÃŠN "
            "(chá»‘t deal, tá»« chá»‘i dá»©t khoÃ¡t, hoáº·c háº¹n lá»‹ch cá»¥ thá»ƒ) Ä‘á»ƒ káº¿t thÃºc cuá»™c trÃ² chuyá»‡n. "
            "KHÃ”NG kÃ©o dÃ i thÃªm."
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

    data = _chat_json(
        system=system,
        user=user_payload,
        temperature=0.7,
        model_name=llm_model,
        provider=llm_provider,
    )
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
    llm_provider: str | None = None,
    llm_model: str | None = None,
    messages: list[dict[str, str]],
    scenario_title: str,
    scenario_description: str,
    customer_persona: str,
    document_contents: str | None = None,
) -> dict[str, Any]:
    """Evaluate for Web App â€” 6 criteria + improvements + top_3_tips.

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
            transcript_lines.append(f"KhÃ¡ch hÃ ng: {m['content']}")
    transcript = "\n".join(transcript_lines) if transcript_lines else "(empty)"

    user_payload = (
        f"Ká»‹ch báº£n: {scenario_title}\n"
        f"MÃ´ táº£: {scenario_description}\n"
        f"TÃ­nh cÃ¡ch khÃ¡ch: {customer_persona}\n\n"
        f"--- BEGIN TRANSCRIPT ---\n{transcript}\n--- END TRANSCRIPT ---"
    )

    # Build evaluator system prompt, optionally with document context
    eval_system = EVALUATOR_SYSTEM_PROMPT_WEB
    if document_contents:
        eval_system += (
            "\n\n== TÃ€I LIá»†U THAM CHIáº¾U (QUY TRÃŒNH BÃN HÃ€NG / THÃ”NG TIN Sáº¢N PHáº¨M) ==\n"
            f"{document_contents}\n\n"
            "HÆ¯á»šNG DáºªN Sá»¬ Dá»¤NG TÃ€I LIá»†U KHI CHáº¤M ÄIá»‚M:\n"
            "- Cháº¥m `process_adherence` dá»±a trÃªn QUY TRÃŒNH THá»°C Táº¾ trong tÃ i liá»‡u (náº¿u cÃ³), khÃ´ng pháº£i quy trÃ¬nh chung chung.\n"
            "- Náº¿u Sales nÃ³i SAI thÃ´ng tin sáº£n pháº©m so vá»›i tÃ i liá»‡u, TRá»ª Náº¶NG Ä‘iá»ƒm `confidence` vÃ  ghi rÃµ lá»—i.\n"
            "- á»ž pháº§n `improvements`, Æ¯U TIÃŠN dÃ¹ng cÃ¢u máº«u tá»« tÃ i liá»‡u lÃ m `ai_suggestion` náº¿u phÃ¹ há»£p.\n"
            "- á»ž pháº§n `top_3_tips`, liÃªn há»‡ vá»›i cÃ¡c bÆ°á»›c cá»¥ thá»ƒ trong tÃ i liá»‡u thay vÃ¬ khuyÃªn chung."
        )

    # === LAYER 1: Call LLM ===
    raw = _chat_json(
        system=eval_system,
        user=user_payload,
        temperature=0.3,
        model_name=llm_model,
        provider=llm_provider,
    )

    # === LAYER 2: Validate improvements ===
    raw_improvements = raw.get("improvements", [])
    validated_improvements = _validate_improvements(raw_improvements, sales_messages)

    discarded_count = len(raw_improvements) - len(validated_improvements)
    if discarded_count > 0:
        print(f"[evaluate_web] âš ï¸ Discarded {discarded_count}/{len(raw_improvements)} hallucinations")

    # === LAYER 3: Auto-retry if ALL discarded ===
    if raw_improvements and not validated_improvements:
        print("[evaluate_web] ðŸ”„ All improvements discarded â†’ Auto-retry...")
        retry_system = (
            EVALUATOR_SYSTEM_PROMPT_WEB
            + "\n\n== âš ï¸ Cáº¢NH BÃO NGHIÃŠM TRá»ŒNG Tá»ª Há»† THá»NG ==\n"
            "Láº¦N TRÆ¯á»šC Báº N ÄÃƒ TRáº¢ Vá»€ CÃ‚U Bá»ŠA (HALLUCINATION) â€” Táº¤T Cáº¢ Äá»€U Bá»Š LOáº I.\n"
            "Há»‡ thá»‘ng Ä‘Ã£ verify tá»«ng cÃ¢u báº¡n tráº£ vá» vÃ  KHÃ”NG TÃŒM THáº¤Y cÃ¢u nÃ o trong transcript thá»±c táº¿.\n"
            "QUY Táº®C TUYá»†T Äá»I Láº¦N NÃ€Y:\n"
            "- Báº¡n CHá»ˆ Ä‘Æ°á»£c trÃ­ch xuáº¥t NGUYÃŠN VÄ‚N tá»« pháº§n \"Sales: ...\" trong transcript.\n"
            "- KHÃ”NG Ä‘Æ°á»£c tá»± táº¡o cÃ¢u, KHÃ”NG paraphrase, KHÃ”NG suy luáº­n.\n"
            "- Náº¿u khÃ´ng cÃ³ cÃ¢u nÃ o cáº§n cáº£i thiá»‡n â†’ tráº£ Ä‘Ãºng: improvements: []"
        )
        retry_raw = _chat_json(
            system=retry_system,
            user=user_payload,
            temperature=0.3,
            model_name=llm_model,
            provider=llm_provider,
        )
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
                "reason": str(entry.get("reason", "KhÃ´ng cÃ³ nháº­n xÃ©t")),
            }
        else:
            rubric[key] = {"score": _clamp_int_score(entry, default=5), "reason": "KhÃ´ng cÃ³ nháº­n xÃ©t"}

    # Compute overall score (0-100 scale)
    scores = [v["score"] for v in rubric.values()]
    overall_score = round(sum(scores) / len(scores) * 10, 1) if scores else 0.0

    # Ensure top_3_tips has exactly 3 items
    tips = list(raw.get("top_3_tips", []))[:3]
    while len(tips) < 3:
        tips.append("Luyá»‡n táº­p thÃªm Ä‘á»ƒ cáº£i thiá»‡n ká»¹ nÄƒng bÃ¡n hÃ ng.")

    return {
        "overall_score": overall_score,
        "rubric_breakdown": rubric,
        "improvements": validated_improvements,
        "top_3_tips": tips,
    }



def generate_scenario_from_doc(
    *,
    document_contents: str,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> dict[str, Any]:
    """Auto-generate scenario details from an uploaded document."""
    user_payload = (
        "ÄÃ¢y lÃ  ná»™i dung tÃ i liá»‡u cá»§a tÃ´i:\n"
        "--- BEGIN DOCUMENT ---\n"
        f"{document_contents}\n"
        "--- END DOCUMENT ---\n\n"
        "HÃ£y táº¡o ká»‹ch báº£n JSON theo hÆ°á»›ng dáº«n."
    )
    
    data = _chat_json(
        system=GENERATE_SCENARIO_SYSTEM_PROMPT,
        user=user_payload,
        temperature=0.7,
        model_name=llm_model,
        provider=llm_provider,
    )
    
    return {
        "title": str(data.get("title", "Ká»‹ch báº£n tá»± Ä‘á»™ng")),
        "description": str(data.get("description", "KhÃ´ng cÃ³ mÃ´ táº£")),
        "company_context": str(data.get("company_context", "KhÃ´ng cÃ³ tÃ³m táº¯t")),
        "customer_persona": str(data.get("customer_persona", "friendly_indecisive")),
    }


async def generate_customer_turn_voice_stream(
    *,
    llm_provider: str | None = None,
    llm_model: str | None = None,
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
        time_instruction = f" CHá»ˆ CÃ’N {time_remaining_seconds} GIÃ‚Y, hÃ£y chá»§ Ä‘á»™ng nháº¯c Ä‘áº¿n viá»‡c sáº¯p háº¿t thá»i gian vÃ  háº¹n lá»‹ch tiáº¿p theo Ä‘á»ƒ chá»‘t láº¡i cuá»™c gá»i."

    system = (
        f"Báº¡n lÃ  khÃ¡ch hÃ ng trong cuá»™c gá»i luyá»‡n sales B2B. Ká»‹ch báº£n: {scenario_title}.\n"
        f"TÃ­nh cÃ¡ch: {customer_persona}.\n"
        f"{time_instruction}\n\n"
        "QUY Táº®C PHáº¢N Há»’I QUA GIá»ŒNG NÃ“I:\n"
        "- Pháº£n há»“i cá»±c ká»³ tá»± nhiÃªn, cÃ³ cáº£m xÃºc vÃ  Ä‘iá»‡u bá»™ nhÆ° ngÆ°á»i tháº­t Ä‘ang nghe Ä‘iá»‡n thoáº¡i.\n"
        "- Äá»™ dÃ i vá»«a pháº£i (khoáº£ng 2-4 cÃ¢u), Ä‘á»§ Ä‘á»ƒ pháº£n biá»‡n logic hoáº·c thá»ƒ hiá»‡n thÃ¡i Ä‘á»™, nhÆ°ng khÃ´ng thao thao báº¥t tuyá»‡t.\n"
        "- HÃ£y dÃ¹ng cÃ¡c tá»« ngá»¯ Ä‘á»i thÆ°á»ng, thá»‰nh thoáº£ng cÃ³ thá»ƒ ngáº­p ngá»«ng (á», á»«m) hoáº·c cáº¯t ngang náº¿u bá»±c mÃ¬nh.\n"
        "- Tuyá»‡t Ä‘á»‘i KHÃ”NG gáº¡ch Ä‘áº§u dÃ²ng, KHÃ”NG dÃ¹ng markdown, KHÃ”NG dÃ¹ng vÄƒn viáº¿t.\n"
        "- KHÃ”NG dÃ¹ng JSON, chá»‰ tráº£ vá» chá»¯.\n"
    )
    
    if company_context:
        system += f"\nSáº£n pháº©m Sales Ä‘ang bÃ¡n:\n{company_context[:300]}\n(DÃ¹ng thÃ´ng tin nÃ y Ä‘á»ƒ báº¯t báº», khÃ´ng tá»± khai ra).\n"

    if current_turn >= max_turns:
        system += "\nLÆ¯á»¢T CUá»I: ÄÆ°a ra quyáº¿t Ä‘á»‹nh (chá»‘t deal / tá»« chá»‘i) má»™t cÃ¡ch dá»©t khoÃ¡t.\n"

    transcript_lines: list[str] = []
    # Take only the last 4 turns to reduce prompt context and speed up TTFT (Time To First Token)
    recent_history = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
    for m in recent_history:
        role = "Sales rep" if m["role"] == "user" else "Buyer"
        transcript_lines.append(f"{role}: {m['content']}")
    transcript = "\n".join(transcript_lines) if transcript_lines else "(no prior turns)"

    user_payload = (
        f"Lá»‹ch sá»­:\n{transcript}\n\n"
        f"Sales vá»«a nÃ³i:\n{last_user_message}\n\n"
        "HÃ£y pháº£n há»“i ngáº¯n gá»n:"
    )

    model_name, provider = _resolve_model_and_provider(
        model_name=llm_model,
        provider=llm_provider,
        default_model=VOICE_LLM_MODEL,
        default_provider=VOICE_LLM_PROVIDER,
    )

    # Try selected provider, fallback on auth errors.
    priority = ["gpt", "gemini", "grok", "groq", "openrouter"]
    tried: set[str] = set()
    current_provider = provider
    while True:
        client = _selected_async_client(current_provider)
        try:
            response_stream = await client.chat.completions.create(
                model=model_name,
                temperature=0.6,
                max_tokens=80,  # Prevent long rambling
                stream=True,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_payload},
                ],
            )
            break
        except Exception as exc:
            is_auth = any(
                m in str(exc) or m in type(exc).__name__
                for m in ["401", "403", "AuthenticationError", "PermissionDenied"]
            )
            if not is_auth:
                raise
            tried.add(current_provider)
            next_provider = next((p for p in priority if p not in tried), None)
            if next_provider:
                api_key, _, next_model, _ = _provider_config(next_provider)
                if api_key:
                    print(f"[llm] voice fallback: {current_provider} → {next_provider}")
                    model_name = next_model
                    current_provider = next_provider
                    continue
            raise RuntimeError(
                f"All LLM providers failed (auth error). Tried: {tried}"
            ) from exc

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

