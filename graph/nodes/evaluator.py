"""Evaluation node — rubric-based feedback on the sales conversation."""

from __future__ import annotations

from llm.llm_client import generate_evaluation
from core.state import SalesSessionState


def evaluation_node(state: SalesSessionState) -> dict:
    """Rubric-based feedback JSON."""
    results = generate_evaluation(state)
    return {
        "current_status": "completed",
        "evaluation_results": results,
        "graph_trace": ["evaluation"],
    }
