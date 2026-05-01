"""
LangGraph state for the sales training simulator.

Uses TypedDict + Annotated reducers for append-only fields (history, trace).
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict


class MessageTurn(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class EvaluationResults(TypedDict, total=False):
    """Structured feedback from _evaluation_node (aligned with schemas.EvaluationPayload)."""

    understanding_needs: int
    response_structure: int
    objection_handling: int
    persuasiveness: int
    next_steps: int
    overall_score: float
    strengths: list[str]
    key_mistakes: list[str]
    suggested_better_answer: str


class SalesSessionState(TypedDict, total=False):
    """
    Single session state for the compiled graph.

    - conversation_history: all user/customer turns (customer = assistant role).
    - current_status: chatting until we branch to evaluation, then completed.
    - evaluation_results: filled only after EvaluationNode runs.
    """

    session_id: str
    scenario: str
    persona: str
    conversation_history: Annotated[list[MessageTurn], operator.add]
    turn_count: int
    max_turns: int
    current_status: Literal["chatting", "evaluating", "completed"]
    last_user_message: str
    customer_reply: str
    session_should_end: bool
    end_reason: str | None
    evaluation_results: EvaluationResults | None
    graph_trace: Annotated[list[str], operator.add]


def initial_sales_state(
    *,
    session_id: str,
    scenario: str,
    persona: str,
    max_turns: int = 12,
) -> SalesSessionState:
    """Baseline state for POST /init-session (before any user chat)."""
    return {
        "session_id": session_id,
        "scenario": scenario,
        "persona": persona,
        "conversation_history": [],
        "turn_count": 0,
        "max_turns": max_turns,
        "current_status": "chatting",
        "last_user_message": "",
        "customer_reply": "",
        "session_should_end": False,
        "end_reason": None,
        "evaluation_results": None,
        "graph_trace": [],
    }
