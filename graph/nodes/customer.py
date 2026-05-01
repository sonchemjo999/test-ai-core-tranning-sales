"""Customer persona node — simulates the buyer in a B2B sales role-play."""

from __future__ import annotations

from llm.llm_client import generate_customer_turn
from core.state import MessageTurn, SalesSessionState


def customer_persona_node(state: SalesSessionState) -> dict:
    """Simulate buyer reply and append the latest user + assistant turns."""
    raw = state.get("last_user_message", "") or ""
    last_user = raw.strip()
    if not last_user:
        return {
            "customer_reply": (
                "Hệ thống không nhận được nội dung. "
                "Vui lòng gửi một tin nhắn không rỗng."
            ),
            "graph_trace": ["customer_persona"],
        }

    prev_turn = int(state.get("turn_count") or 0)
    max_turns = int(state.get("max_turns") or 12)

    prior: list[MessageTurn] = list(state.get("conversation_history") or [])
    sim_state: SalesSessionState = {
        **state,
        "conversation_history": prior,
        "turn_count": prev_turn,
    }
    out = generate_customer_turn(sim_state)
    customer_text = (out.get("customer_message") or "").strip() or "…"
    should_end = bool(out.get("session_should_end"))
    end_reason = out.get("end_reason")
    if isinstance(end_reason, str):
        end_reason = end_reason.strip() or None
    else:
        end_reason = None

    new_turn = prev_turn + 1
    user_turn: MessageTurn = {"role": "user", "content": last_user}
    asst_turn: MessageTurn = {"role": "assistant", "content": customer_text}

    # Turn cap forces evaluation after this reply
    if new_turn >= max_turns:
        should_end = True
        end_reason = end_reason or "max_turns_reached"

    return {
        "conversation_history": [user_turn, asst_turn],
        "turn_count": new_turn,
        "customer_reply": customer_text,
        "session_should_end": should_end,
        "end_reason": end_reason,
        "current_status": "chatting",
        "graph_trace": ["customer_persona"],
    }
