"""
Tests for Prompt Templates and Builders.
"""

from llm.prompts import CUSTOMER_SYSTEM_TEMPLATE_WEB


def test_customer_system_template_web_formatting():
    """Test that customer web prompt formats correctly."""
    formatted = CUSTOMER_SYSTEM_TEMPLATE_WEB.format(
        scenario_title="Bán dự án",
        scenario_description="Khách đang do dự",
        customer_persona="Khách bận rộn",
        ai_tone_instruction="Thái độ: Lạnh lùng",
        follow_up_depth_instruction="Truy vấn sâu",
        time_instruction="Hết giờ"
    )
    
    assert "Bán dự án" in formatted
    assert "Khách đang do dự" in formatted
    assert "Thái độ: Lạnh lùng" in formatted
    assert "Truy vấn sâu" in formatted
