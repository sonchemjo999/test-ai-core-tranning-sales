"""
CLI chat -- interactive terminal loop for the Sale Train Agent.

Run:
    set PYTHONPATH=src
    python -m cli.cli_chat
"""

from __future__ import annotations

import sys
import io
import json
import urllib.request
import urllib.error

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")

from graph.graph import build_sales_graph
from core.state import initial_sales_state

# --- Base URL for the FastAPI backend (for retry / next-level) ---
API_BASE = "http://127.0.0.1:8000"

# -- Available options (MVP: 3 scenarios x 3 personas) --------------------
SCENARIOS = {
    "1": "first_30_seconds",
    "2": "price_objection",
    "3": "closing",
}

PERSONAS = {
    "1": "friendly_indecisive",
    "2": "detail_oriented",
    "3": "busy_skeptic",
}


def _pick(label: str, options: dict[str, str]) -> str:
    print(f"\n-- {label} --")
    for k, v in options.items():
        print(f"  [{k}] {v}")
    while True:
        choice = input("Chon (so): ").strip()
        if choice in options:
            return options[choice]
        print("  [!] Lua chon khong hop le, thu lai.")


def _api_post(path: str) -> dict:
    """HTTP POST to the FastAPI backend (no request body needed)."""
    url = f"{API_BASE}{path}"
    req = urllib.request.Request(url, data=b"", method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"  [!] API lỗi {e.code}: {body}")
        return {}
    except urllib.error.URLError as e:
        print(f"  [!] Không kết nối được API ({e.reason}). Đảm bảo FastAPI đang chạy.")
        return {}


def _run_session(scenario: str, persona: str, max_turns: int, session_id: str) -> None:
    """Run one complete role-play session (chat loop + evaluation display)."""
    state = initial_sales_state(
        session_id=session_id,
        scenario=scenario,
        persona=persona,
        max_turns=max_turns,
    )
    graph = build_sales_graph()

    print("\n" + "-" * 60)
    print(f"  Scenario : {scenario}")
    print(f"  Persona  : {persona}")
    print(f"  Max turns: {max_turns}")
    print("-" * 60)
    print("Bat dau role-play! Go 'quit' de thoat.\n")

    while True:
        user_msg = input("[Sales Rep] > ").strip()
        if not user_msg:
            print("  (tin nhan trong, thu lai)")
            continue
        if user_msg.lower() in ("quit", "exit", "q"):
            print("\n[*] Da thoat. Bye!")
            sys.exit(0)

        state["last_user_message"] = user_msg
        state = graph.invoke(state)

        reply = state.get("customer_reply", "")
        turn = state.get("turn_count", 0)
        print(f"[Buyer] (turn {turn}/{max_turns}): {reply}\n")

        if state.get("current_status") == "completed":
            print("=" * 60)
            print("  [EVALUATION] SESSION KET THUC -- DANH GIA")
            print("=" * 60)
            ev = state.get("evaluation_results")
            if ev:
                print(f"  Hieu nhu cau       : {ev.get('understanding_needs')}/10")
                print(f"  Cau truc tra loi   : {ev.get('response_structure')}/10")
                print(f"  Xu ly phan doi     : {ev.get('objection_handling')}/10")
                print(f"  Thuyet phuc        : {ev.get('persuasiveness')}/10")
                print(f"  Buoc tiep theo     : {ev.get('next_steps')}/10")
                overall: float = ev.get("overall_score", 0.0)
                print(f"  -- Overall Score   : {overall}/10")
                print()
                strengths = ev.get("strengths", [])
                if strengths:
                    print("  [+] Diem manh:")
                    for s in strengths:
                        print(f"      - {s}")
                mistakes = ev.get("key_mistakes", [])
                if mistakes:
                    print("  [-] Sai lam chinh:")
                    for m in mistakes:
                        print(f"      - {m}")
                better = ev.get("suggested_better_answer", "")
                if better:
                    print(f"\n  [TIP] Goi y cau tra loi tot hon:\n      {better}")
            else:
                overall = 0.0
                print("  (Khong co du lieu danh gia)")

            print("\n" + "=" * 60)
            end_reason = state.get("end_reason", "N/A")
            print(f"  Ly do ket thuc: {end_reason}")
            print("=" * 60)

            # ── Gamification loop ────────────────────────────────────
            if overall < 7.0:
                print(f"\n  Ban dat {overall}/10.")
                choice = input("  Muon RETRY de cai thien diem? (y/n): ").strip().lower()
                if choice == "y":
                    # Use API if available; otherwise reset state directly
                    api_resp = _api_post(f"/session/retry/{session_id}")
                    if api_resp:
                        print(f"  {api_resp.get('message', 'Session reset!')}")
                    # Restart the session in-process (same scenario + persona)
                    _run_session(scenario, persona, max_turns, session_id)
                else:
                    print("\n  [*] Cam on da luyen tap. Hen gap lai!")
            else:
                print(f"\n  Great job! Ban dat {overall}/10.")
                choice = input("  Muon NEXT LEVEL voi khach hang kho hon? (y/n): ").strip().lower()
                if choice == "y":
                    api_resp = _api_post(f"/session/next-level/{session_id}")
                    if api_resp and api_resp.get("new_persona"):
                        new_persona = api_resp["new_persona"]
                        print(f"  {api_resp.get('message', f'New persona: {new_persona}')}")
                        _run_session(scenario, new_persona, max_turns, session_id)
                    else:
                        # Fallback: max level reached or API down — pick manually
                        print("  Ban da dat level cao nhat hoac API khong hoat dong.")
                        print("  Chuc mung! Ban da hoan thanh tat ca cac cap do.")
                else:
                    print("\n  [*] Xuat sac! Hen gap lai!")
            # ── End gamification loop ────────────────────────────────
            return  # Done with this session


def main() -> None:
    print("=" * 60)
    print("  [*] SALE TRAIN AGENT -- Terminal Chat")
    print("=" * 60)

    scenario = _pick("Scenario", SCENARIOS)
    persona = _pick("Buyer Persona", PERSONAS)

    max_turns_input = input("\nMax turns (Enter = 12): ").strip()
    max_turns = int(max_turns_input) if max_turns_input.isdigit() else 12

    session_id = "cli-session"
    _run_session(scenario, persona, max_turns, session_id)


if __name__ == "__main__":
    main()
