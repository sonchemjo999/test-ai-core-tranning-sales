"""
Simli Avatar Manager — manages Simli WebRTC avatar sessions per session_id.

This module acts as a bridge between the FastAPI voice pipeline and Simli's
AI-powered video avatar service. When a phone app connects to /ws/avatar/{session_id},
the backend:
1. Authenticates the connection via HMAC ticket
2. Starts a Simli avatar session (male/female based on gender param)
3. Relays audio from TTS pipeline to Simli
4. Receives video frames from Simli and forwards them to the phone via WebSocket

This way, the phone app receives an animated avatar without needing the Simli SDK.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from core.config import (
    SIMLI_API_KEY,
    SIMLI_ENABLED,
    SIMLI_FEMALE_FACE_ID,
    SIMLI_MALE_FACE_ID,
)

logger = logging.getLogger(__name__)

# Default face IDs (can be overridden via environment)
DEFAULT_MALE_FACE_ID = "dd10cb5a-d31d-4f12-b69f-6db3383c006e"
DEFAULT_FEMALE_FACE_ID = "cace3ef7-a4c4-425d-a8cf-a5358eb0c427"

# Fallback IDs if env vars not set
_MALE_FACE_ID = SIMLI_MALE_FACE_ID or DEFAULT_MALE_FACE_ID
_FEMALE_FACE_ID = SIMLI_FEMALE_FACE_ID or DEFAULT_FEMALE_FACE_ID


@dataclass
class SimliSession:
    """Holds state for a single Simli avatar session."""

    session_id: str
    gender: str
    face_id: str
    client: Any | None = None
    is_connected: bool = False
    video_frame_callback: Callable[[bytes], None] | None = None
    error_callback: Callable[[str], None] | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SimliAvatarManager:
    """
    Singleton manager for Simli avatar sessions.

    Handles:
    - Starting/stopping Simli WebRTC connections per session_id
    - Sending PCM16 audio chunks to Simli for lip-sync animation
    - Receiving video frames from Simli and routing to the phone WebSocket
    - Cleanup on disconnect
    """

    _instance: SimliAvatarManager | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> SimliAvatarManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        self._sessions: dict[str, SimliSession] = {}
        self._sessions_lock = asyncio.Lock()

        self._simli_available = False
        self._simli_client_cls: Any = None
        self._simli_config_cls: Any = None

        self._check_simli_available()

    def _check_simli_available(self) -> None:
        """Check if simli-ai package is installed and API key is configured."""
        if not SIMLI_ENABLED:
            logger.info("[SimliAvatar] Disabled via SIMLI_ENABLED=false")
            return

        if not SIMLI_API_KEY:
            logger.warning("[SimliAvatar] SIMLI_API_KEY not set — avatar streaming disabled")
            return

        try:
            from simli import SimliClient, SimliConfig

            self._simli_client_cls = SimliClient
            self._simli_config_cls = SimliConfig
            self._simli_available = True
            logger.info("[SimliAvatar] simli-ai package loaded successfully")
        except ImportError:
            logger.warning(
                "[SimliAvatar] simli-ai not installed — avatar streaming disabled. "
                "Run: pip install simli-ai"
            )
        except Exception as e:
            logger.error(f"[SimliAvatar] Error loading simli-ai: {e}")

    @property
    def is_available(self) -> bool:
        """Check if Simli avatar is available and enabled."""
        return self._simli_available and SIMLI_ENABLED and bool(SIMLI_API_KEY)

    def _get_face_id(self, gender: str) -> str:
        """Get the appropriate face ID based on gender."""
        gender = gender.lower() if gender else "male"
        if gender in ("female", "f", "nu"):
            return _FEMALE_FACE_ID
        return _MALE_FACE_ID

    async def start_session(
        self,
        session_id: str,
        gender: str = "male",
        video_frame_callback: Callable[[bytes], None] | None = None,
        error_callback: Callable[[str], None] | None = None,
    ) -> SimliSession:
        """
        Start a new Simli avatar session.

        Args:
            session_id: Unique session identifier
            gender: "male" or "female" to select the appropriate avatar face
            video_frame_callback: Called when a video frame is received from Simli
            error_callback: Called when an error occurs

        Returns:
            SimliSession object for this session

        Raises:
            RuntimeError: If Simli is not available
        """
        if not self.is_available:
            raise RuntimeError(
                "Simli avatar is not available. "
                "Ensure SIMLI_ENABLED=true and SIMLI_API_KEY is set."
            )

        async with self._sessions_lock:
            if session_id in self._sessions:
                await self.stop_session(session_id)

            face_id = self._get_face_id(gender)
            session = SimliSession(
                session_id=session_id,
                gender=gender,
                face_id=face_id,
                video_frame_callback=video_frame_callback,
                error_callback=error_callback,
            )

            try:
                await self._connect_to_simli(session)
                self._sessions[session_id] = session
                logger.info(
                    f"[SimliAvatar] Session started: {session_id}, gender={gender}, face_id={face_id}"
                )
                return session
            except Exception as e:
                logger.error(f"[SimliAvatar] Failed to start session {session_id}: {e}")
                if error_callback:
                    try:
                        error_callback(str(e))
                    except Exception:
                        pass
                raise

    async def _connect_to_simli(self, session: SimliSession) -> None:
        """Establish WebRTC connection to Simli."""
        if not self._simli_available or not self._simli_client_cls:
            raise RuntimeError("Simli client not initialized")

        try:
            config = self._simli_config_cls(faceId=session.face_id)
            client = self._simli_client_cls(api_key=SIMLI_API_KEY, config=config)

            if hasattr(client, "set_video_frame_callback"):
                if session.video_frame_callback:
                    client.set_video_frame_callback(
                        lambda frame: self._handle_video_frame(session.session_id, frame)
                    )

            await client.start()
            session.client = client
            session.is_connected = True
            logger.info(f"[SimliAvatar] Connected to Simli for session {session.session_id}")

        except Exception as e:
            session.is_connected = False
            logger.error(f"[SimliAvatar] Connection failed for {session.session_id}: {e}")
            raise

    def _handle_video_frame(self, session_id: str, frame: bytes) -> None:
        """Handle incoming video frame from Simli."""
        try:
            session = self._sessions.get(session_id)
            if session and session.video_frame_callback:
                session.video_frame_callback(frame)
        except Exception as e:
            logger.error(f"[SimliAvatar] Error handling video frame for {session_id}: {e}")

    async def send_audio(self, session_id: str, audio_bytes: bytes) -> bool:
        """
        Send PCM16 audio data to Simli for lip-sync animation.

        Args:
            session_id: Session identifier
            audio_bytes: Raw PCM16 audio data (16-bit, 16kHz mono expected)

        Returns:
            True if audio was sent successfully, False otherwise
        """
        async with self._sessions_lock:
            session = self._sessions.get(session_id)

        if not session:
            logger.warning(f"[SimliAvatar] send_audio: session not found: {session_id}")
            return False

        if not session.is_connected or not session.client:
            logger.warning(f"[SimliAvatar] send_audio: session not connected: {session_id}")
            return False

        try:
            if hasattr(session.client, "send_audio"):
                await session.client.send_audio(audio_bytes)
            elif hasattr(session.client, "send"):
                await session.client.send(audio_bytes)
            else:
                await session.client.send(audio_bytes)
            return True
        except Exception as e:
            logger.error(f"[SimliAvatar] send_audio failed for {session_id}: {e}")
            return False

    async def send_text(self, session_id: str, text: str) -> bool:
        """
        Send text to Simli for avatar animation (if supported).

        Args:
            session_id: Session identifier
            text: Text to speak

        Returns:
            True if text was sent successfully, False otherwise
        """
        async with self._sessions_lock:
            session = self._sessions.get(session_id)

        if not session or not session.is_connected or not session.client:
            return False

        try:
            if hasattr(session.client, "send_text"):
                await session.client.send_text(text)
                return True
            return False
        except Exception as e:
            logger.error(f"[SimliAvatar] send_text failed for {session_id}: {e}")
            return False

    async def stop_session(self, session_id: str) -> None:
        """Stop and clean up a Simli avatar session."""
        async with self._sessions_lock:
            session = self._sessions.pop(session_id, None)

        if not session:
            logger.debug(f"[SimliAvatar] stop_session: session not found: {session_id}")
            return

        if session.client:
            try:
                if hasattr(session.client, "stop"):
                    await session.client.stop()
                elif hasattr(session.client, "close"):
                    await session.client.close()
                logger.info(f"[SimliAvatar] Session stopped: {session_id}")
            except Exception as e:
                logger.error(f"[SimliAvatar] Error stopping session {session_id}: {e}")
            finally:
                session.client = None
                session.is_connected = False

    async def get_session(self, session_id: str) -> SimliSession | None:
        """Get a session by ID."""
        async with self._sessions_lock:
            return self._sessions.get(session_id)

    async def has_active_session(self, session_id: str) -> bool:
        """Check if a session exists and is connected."""
        session = await self.get_session(session_id)
        return session is not None and session.is_connected

    async def stop_all_sessions(self) -> None:
        """Stop all active sessions. Used during graceful shutdown."""
        session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            await self.stop_session(session_id)
        logger.info("[SimliAvatar] All sessions stopped")


def get_simli_manager() -> SimliAvatarManager:
    """Get the singleton SimliAvatarManager instance."""
    return SimliAvatarManager()


async def simli_send_audio_from_url(session_id: str, audio_url: str) -> bool:
    """
    Fetch audio from URL and send it to Simli.

    This is useful when the TTS pipeline returns an audio URL (e.g., from FPT.AI)
    and we want to also send that audio to Simli for avatar animation.

    Args:
        session_id: Session identifier
        audio_url: URL to fetch audio from

    Returns:
        True if audio was fetched and sent successfully
    """
    import httpx

    manager = get_simli_manager()

    if not await manager.has_active_session(session_id):
        return False

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(audio_url)
            response.raise_for_status()
            audio_bytes = response.content

        return await manager.send_audio(session_id, audio_bytes)

    except Exception as e:
        logger.error(f"[SimliAvatar] Failed to fetch/send audio for {session_id}: {e}")
        return False


def encode_video_frame_for_websocket(frame: bytes) -> str:
    """
    Encode video frame bytes for WebSocket transmission.

    Returns a base64-encoded string that can be sent via WebSocket JSON.
    """
    return base64.b64encode(frame).decode("utf-8")
