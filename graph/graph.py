"""
LangGraph StateGraph: CustomerPersona -> (conditional) -> Evaluation.

Invoke with a SalesSessionState that includes last_user_message for each /chat turn.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from core.state import SalesSessionState
from graph.nodes.customer import customer_persona_node
from graph.nodes.evaluator import evaluation_node


def _route_after_customer(state: SalesSessionState) -> Literal["evaluate", "end"]:
    if state.get("session_should_end"):
        return "evaluate"
    return "end"


def build_sales_graph():
    """Compile and return the LangGraph runnable."""
    graph = StateGraph(SalesSessionState)
    graph.add_node("customer_persona", customer_persona_node)
    graph.add_node("evaluation", evaluation_node)

    graph.add_edge(START, "customer_persona")
    graph.add_conditional_edges(
        "customer_persona",
        _route_after_customer,
        {"evaluate": "evaluation", "end": END},
    )
    graph.add_edge("evaluation", END)
    return graph.compile()
