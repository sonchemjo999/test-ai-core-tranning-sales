"""
Tests for Cartesia TTS streaming module.
"""

import asyncio
import base64
import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from tools.tts_cartesia import generate_tts_cartesia_stream, generate_tts_cartesia_base64


@pytest.mark.asyncio
class TestCartesiaTTS:
    
    @patch("tools.tts_cartesia.CARTESIA_API_KEY", "test-key")
    @patch("tools.tts_cartesia.CARTESIA_FEMALE_VOICE_ID", "test-voice")
    async def test_generate_tts_cartesia_stream_success(self):
        """Test streaming yields correctly parsed base64 chunks."""
        
        # Fake SSE chunks
        fake_chunks = [
            'data: {"data": "chunk1_base64"}',
            'data: {"data": "chunk2_base64"}',
            'data: [DONE]'
        ]
        
        # Mock response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        
        # Mock aiter_lines to yield lines asynchronously
        async def fake_aiter_lines():
            for line in fake_chunks:
                yield line
                
        mock_response.aiter_lines = fake_aiter_lines
        
        # Mock client context manager
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        
        # Mock client.stream context manager
        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)
        
        with patch("tools.tts_cartesia.httpx.AsyncClient", return_value=mock_client):
            chunks = []
            async for chunk in generate_tts_cartesia_stream("Xin chào"):
                chunks.append(chunk)
                
        assert len(chunks) == 2
        assert chunks[0] == "chunk1_base64"
        assert chunks[1] == "chunk2_base64"

    @patch("tools.tts_cartesia.CARTESIA_API_KEY", "test-key")
    @patch("tools.tts_cartesia.CARTESIA_FEMALE_VOICE_ID", "test-voice")
    async def test_generate_tts_cartesia_stream_error(self):
        """Test streaming handles HTTP errors gracefully without raising."""
        mock_response = AsyncMock()
        mock_response.status_code = 401
        mock_response.aread = AsyncMock(return_value=b"Unauthorized")
        
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        
        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)
        
        with patch("tools.tts_cartesia.httpx.AsyncClient", return_value=mock_client):
            chunks = []
            async for chunk in generate_tts_cartesia_stream("Xin chào"):
                chunks.append(chunk)
                
        assert len(chunks) == 0

    @patch("tools.tts_cartesia.CARTESIA_API_KEY", "")
    async def test_no_api_key_returns_empty(self):
        """Test missing API key yields nothing."""
        chunks = []
        async for chunk in generate_tts_cartesia_stream("Xin chào"):
            chunks.append(chunk)
        assert len(chunks) == 0

    @patch("tools.tts_cartesia.CARTESIA_API_KEY", "test-key")
    @patch("tools.tts_cartesia.CARTESIA_FEMALE_VOICE_ID", "test-voice")
    async def test_generate_tts_cartesia_base64_success(self):
        """Test full assembly base64 returns correct data URI."""
        fake_pcm = b"\x00\x01\x02\x03"
        b64_encoded = base64.b64encode(fake_pcm).decode("utf-8")
        
        # We can just mock the generator to return our fake chunk
        async def mock_generator(text):
            yield b64_encoded
            
        with patch("tools.tts_cartesia.generate_tts_cartesia_stream", side_effect=mock_generator):
            uri, fmt = await generate_tts_cartesia_base64("Xin chào")
            
        assert fmt == "pcm16"
        assert uri == f"data:audio/pcm;base64,{b64_encoded}"

    @patch("tools.tts_cartesia.CARTESIA_API_KEY", "test-key")
    @patch("tools.tts_cartesia.CARTESIA_FEMALE_VOICE_ID", "test-voice")
    async def test_generate_tts_cartesia_base64_empty(self):
        """Test full assembly gracefully handles empty chunks."""
        # Yield nothing
        async def mock_generator(text):
            if False: yield ""
            
        with patch("tools.tts_cartesia.generate_tts_cartesia_stream", side_effect=mock_generator):
            uri, fmt = await generate_tts_cartesia_base64("Xin chào")
            
        assert uri is None
        assert fmt == "mp3"
