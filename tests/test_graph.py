"""
Tests for LangGraph State Machine Routing.
"""

from unittest.mock import patch

from core.state import SalesSessionState
from graph.graph import build_sales_graph


def test_graph_routing_chatting():
    """Test happy path where conversation continues."""
    state: SalesSessionState = {
        "turn_count": 2,
        "max_turns": 10,
        "last_user_message": "Hello",
        "conversation_history": [],
    }
    
    mock_customer_response = {
        "customer_message": "Hi there.",
        "session_should_end": False,
        "end_reason": None,
    }
    
    with patch("graph.nodes.customer.generate_customer_turn", return_value=mock_customer_response):
        graph = build_sales_graph()
        result = graph.invoke(state)
        
    assert result["current_status"] == "chatting"
    assert result["turn_count"] == 3
    assert result["session_should_end"] is False
    assert "evaluation" not in result.get("graph_trace", [])


def test_graph_routing_max_turns():
    """Test routing gracefully ends session at max_turns."""
    state: SalesSessionState = {
        "turn_count": 9,
        "max_turns": 10,
        "last_user_message": "Final try.",
        "conversation_history": [],
    }
    
    mock_customer_response = {
        "customer_message": "Goodbye.",
        "session_should_end": False,
        "end_reason": None,
    }
    
    mock_evaluation_response = {
        "overall_score": 8,
        "understanding_needs": 8,
        "response_structure": 8,
        "objection_handling": 8,
        "persuasiveness": 8,
        "next_steps": 8,
        "strengths": [],
        "key_mistakes": [],
        "suggested_better_answer": "...",
    }
    
    with patch("graph.nodes.customer.generate_customer_turn", return_value=mock_customer_response), \
         patch("graph.nodes.evaluator.generate_evaluation", return_value=mock_evaluation_response):
        
        graph = build_sales_graph()
        result = graph.invoke(state)
        
    assert result["current_status"] == "completed"
    assert result["turn_count"] == 10
    assert result["session_should_end"] is True
    assert result["end_reason"] == "max_turns_reached"
    assert "evaluation" in result.get("graph_trace", [])


def test_graph_routing_customer_ends_session():
    """Test routing ends session if LLM sets session_should_end=True."""
    state: SalesSessionState = {
        "turn_count": 5,
        "max_turns": 10,
        "last_user_message": "I will send the contract now.",
        "conversation_history": [],
    }
    
    mock_customer_response = {
        "customer_message": "Sounds good.",
        "session_should_end": True,
        "end_reason": "deal_closed",
    }
    
    mock_evaluation_response = {
        "overall_score": 10,
        "understanding_needs": 10,
        "response_structure": 10,
        "objection_handling": 10,
        "persuasiveness": 10,
        "next_steps": 10,
        "strengths": [],
        "key_mistakes": [],
        "suggested_better_answer": "...",
    }
    
    with patch("graph.nodes.customer.generate_customer_turn", return_value=mock_customer_response), \
         patch("graph.nodes.evaluator.generate_evaluation", return_value=mock_evaluation_response):
        
        graph = build_sales_graph()
        result = graph.invoke(state)
        
    assert result["current_status"] == "completed"
    assert result["session_should_end"] is True
    assert result["end_reason"] == "deal_closed"
    assert "evaluation" in result.get("graph_trace", [])
