"""
Tests for LLM Client.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from llm.llm_client import generate_customer_turn_voice_stream


@pytest.mark.asyncio
class TestLLMClient:
    
    @patch("llm.llm_client._async_client")
    async def test_generate_customer_turn_voice_stream(self, mock_async_client_factory):
        """Test streaming from OpenAI/Groq yields text correctly."""
        
        # Fake chunks from LLM
        class FakeChunk:
            def __init__(self, text):
                self.choices = [type("Choice", (), {"delta": type("Delta", (), {"content": text})})]
        
        fake_chunks = [
            FakeChunk("Xin "),
            FakeChunk("chào "),
            FakeChunk("bạn.")
        ]
        
        # Mock client behavior
        mock_client = AsyncMock()
        mock_create = AsyncMock()
        
        async def fake_stream(*args, **kwargs):
            for chunk in fake_chunks:
                yield chunk
                
        mock_create.side_effect = fake_stream
        mock_client.chat.completions.create = mock_create
        
        # Make the factory return our mock client
        mock_async_client_factory.return_value = mock_client
        
        # Call function with required kwargs
        chunks = []
        async_gen = generate_customer_turn_voice_stream(
            customer_persona="Test Persona",
            scenario_title="Test Scenario",
            scenario_description="Test Desc",
            company_context="Test Context",
            conversation_history=[],
            last_user_message="Hello",
            current_turn=1,
            max_turns=10
        )
        async for chunk in async_gen:
            chunks.append(chunk)
            
        assert "".join(chunks) == "Xin chào bạn."
        mock_create.assert_called_once()
