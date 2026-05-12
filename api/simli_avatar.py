"""
Simli Avatar Manager — manages Simli WebRTC avatar sessions per session_id.

This module acts as a bridge between the FastAPI voice pipeline and Simli's
AI-powered video avatar service. When a phone app connects to /ws/avatar/{session_id},
the backend:
1. Authenticates the connection via HMAC ticket
2. Starts a Simli avatar session (male/female based on gender param)
3. Relays audio from TTS pipeline to Simli
4. Receives video frames from Simli (via polling) and forwards them to the phone via WebSocket

Simli SDK reference:
  pip install simli-ai
  Docs: https://docs.simli.ai
"""

from __future__ import annotations

import asyncio
import base64
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from core.config import (
    SIMLI_API_KEY,
    SIMLI_ENABLED,
    SIMLI_FEMALE_FACE_ID,
    SIMLI_MALE_FACE_ID,
)

logger = logging.getLogger(__name__)


def _simli_log(message: str, *args: Any) -> None:
    formatted = message % args if args else message
    print(f"[SimliAvatar] {formatted}", flush=True)
    logger.info(formatted)


DEFAULT_MALE_FACE_ID = "dd10cb5a-d31d-4f12-b69f-6db3383c006e"
DEFAULT_FEMALE_FACE_ID = "cace3ef7-a4c4-425d-a8cf-a5358eb0c427"

_MALE_FACE_ID = SIMLI_MALE_FACE_ID or DEFAULT_MALE_FACE_ID
_FEMALE_FACE_ID = SIMLI_FEMALE_FACE_ID or DEFAULT_FEMALE_FACE_ID

# Lazy SDK references — loaded on first use to avoid crash if package not installed
_simli_client_cls: type | None = None
_simli_config_cls: type | None = None


def _load_simli_sdk() -> bool:
    """Try to import simli-ai SDK. Returns True if available, False otherwise."""
    global _simli_client_cls, _simli_config_cls
    if _simli_client_cls is not None:
        return True
    try:
        from simli import SimliClient, SimliConfig

        _simli_client_cls = SimliClient
        _simli_config_cls = SimliConfig
        return True
    except ImportError:
        logger.warning(
            "[SimliAvatar] simli-ai package not installed — avatar streaming disabled. "
            "Run: pip install simli-ai"
        )
        return False
    except Exception as e:
        logger.error(f"[SimliAvatar] Error loading simli-ai: {e}")
        return False


@dataclass
class SimliSession:
    """Holds state for a single Simli avatar session."""

    session_id: str
    gender: str
    face_id: str
    client: Any | None = None
    is_connected: bool = False
    video_frame_callback: Callable[[Any], None] | None = None
    error_callback: Callable[[str], None] | None = None
    _frame_reader_task: asyncio.Task | None = None
    _last_frame_at: float = 0.0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def _cancel_frame_reader(self) -> None:
        if self._frame_reader_task:
            self._frame_reader_task.cancel()
            try:
                await self._frame_reader_task
            except asyncio.CancelledError:
                pass
            self._frame_reader_task = None


class SimliAvatarManager:
    """
    Singleton manager for Simli avatar sessions.

    Handles:
    - Starting/stopping Simli WebRTC connections per session_id
    - Sending PCM16 audio chunks to Simli for lip-sync animation
    - Receiving video frames from Simli (via polling) and routing to the phone WebSocket
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
        self._check_simli_available()

    def _check_simli_available(self) -> None:
        """Check if simli-ai package is installed and API key is configured."""
        if not SIMLI_ENABLED:
            _simli_log("disabled via SIMLI_ENABLED=false")
            return

        if not SIMLI_API_KEY:
            _simli_log("SIMLI_API_KEY not set; avatar streaming disabled")
            return

        if _load_simli_sdk():
            self._simli_available = True
            _simli_log("simli-ai package loaded successfully")

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
        video_frame_callback: Callable[[Any], None] | None = None,
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
                _simli_log(
                    "starting session=%s gender=%s face_id=%s",
                    session_id,
                    gender,
                    face_id,
                )
                await self._connect_to_simli(session)
                self._sessions[session_id] = session
                _simli_log(
                    "session started=%s gender=%s face_id=%s",
                    session_id,
                    gender,
                    face_id,
                )
                return session
            except Exception as e:
                _simli_log("failed to start session=%s error=%s", session_id, e)
                logger.exception("[SimliAvatar] start_session traceback %s", session_id)
                if error_callback:
                    try:
                        error_callback(str(e))
                    except Exception:
                        pass
                raise

    async def _connect_to_simli(self, session: SimliSession) -> None:
        """Establish WebRTC connection to Simli and start frame reader."""
        if not self._simli_available:
            raise RuntimeError("Simli client not available")

        if not _load_simli_sdk():
            raise RuntimeError("simli-ai package not installed")

        _simli_log(
            "connecting to Simli session=%s face_id=%s",
            session.session_id,
            session.face_id,
        )
        config = _simli_config_cls(faceId=session.face_id)
        client = _simli_client_cls(api_key=SIMLI_API_KEY, config=config)

        await client.start()
        session.client = client
        session.is_connected = True
        _simli_log(
            "client started session=%s methods=%s",
            session.session_id,
            [m for m in dir(client) if not m.startswith("_")][:30],
        )

        # Start background task to poll frames from Simli and forward via callback
        session._frame_reader_task = asyncio.create_task(
            self._frame_reader_loop(session)
        )

        _simli_log("connected to Simli session=%s", session.session_id)

    async def _frame_reader_loop(self, session: SimliSession) -> None:
        """
        Background task that continuously polls video frames from Simli's WebRTC stream
        and forwards them to the registered callback.

        The Simli SDK provides frames via getVideoStreamIterator() or getNextVideoFrame().
        We poll in a loop and invoke the callback for each received frame.
        """
        _simli_log("frame reader started session=%s", session.session_id)
        client = session.client
        min_frame_interval = 1 / 30

        try:
            # Try the async iterator API first (preferred)
            if hasattr(client, "getVideoStreamIterator"):
                async for frame in client.getVideoStreamIterator(targetFormat="rgb"):
                    now = time.perf_counter()
                    if now - session._last_frame_at < min_frame_interval:
                        continue
                    session._last_frame_at = now
                    if session.video_frame_callback:
                        try:
                            # Frame may be a PyAV VideoFrame — extract raw bytes
                            frame_payload = _extract_frame_payload(frame)
                            if frame_payload is not None:
                                session.video_frame_callback(frame_payload)
                        except Exception as e:
                            logger.error(
                                f"[SimliAvatar] Callback error for {session.session_id}: {e}"
                            )
            # Fallback: poll with getNextVideoFrame
            elif hasattr(client, "getNextVideoFrame"):
                while True:
                    try:
                        frame = await client.getNextVideoFrame()
                        now = time.perf_counter()
                        if now - session._last_frame_at < min_frame_interval:
                            await asyncio.sleep(0)
                            continue
                        session._last_frame_at = now
                        if frame and session.video_frame_callback:
                            frame_payload = _extract_frame_payload(frame)
                            if frame_payload is not None:
                                session.video_frame_callback(frame_payload)
                    except StopIteration:
                        break
                    except Exception as e:
                        logger.error(
                            f"[SimliAvatar] getNextVideoFrame error for {session.session_id}: {e}"
                        )
                        await asyncio.sleep(0.1)
            else:
                _simli_log(
                    "no frame iterator method session=%s available=%s",
                    session.session_id,
                    [m for m in dir(client) if not m.startswith("_")],
                )
        except asyncio.CancelledError:
            _simli_log("frame reader cancelled session=%s", session.session_id)
        except Exception as e:
            _simli_log("frame reader error session=%s error=%s", session.session_id, e)
            logger.exception("[SimliAvatar] frame reader traceback %s", session.session_id)
            if session.error_callback:
                try:
                    session.error_callback(str(e))
                except Exception:
                    pass
        finally:
            _simli_log("frame reader stopped session=%s", session.session_id)

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
            _simli_log("send_audio session not found=%s bytes=%s", session_id, len(audio_bytes))
            return False

        if not session.is_connected or not session.client:
            _simli_log("send_audio session not connected=%s bytes=%s", session_id, len(audio_bytes))
            return False

        try:
            # Try send() method (standard Simli SDK method)
            if hasattr(session.client, "send"):
                await session.client.send(audio_bytes)
            elif hasattr(session.client, "send_audio"):
                await session.client.send_audio(audio_bytes)
            else:
                _simli_log("no send method on client session=%s", session_id)
                return False
            return True
        except Exception as e:
            _simli_log("send_audio failed session=%s error=%s", session_id, e)
            return False

    async def send_silence(self, session_id: str) -> bool:
        async with self._sessions_lock:
            session = self._sessions.get(session_id)

        if not session or not session.is_connected or not session.client:
            _simli_log("send_silence session not ready=%s", session_id)
            return False

        try:
            if hasattr(session.client, "sendSilence"):
                result = session.client.sendSilence()
                if hasattr(result, "__await__"):
                    await result
                _simli_log("send_silence ok session=%s method=sendSilence", session_id)
                return True
            _simli_log("send_silence method missing session=%s", session_id)
            return False
        except Exception as e:
            _simli_log("send_silence failed session=%s error=%s", session_id, e)
            return False

    async def send_text(self, session_id: str, text: str) -> bool:
        """Send text to Simli for avatar animation (if supported)."""
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

        # Cancel frame reader first
        await session._cancel_frame_reader()

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


# ── Helpers ────────────────────────────────────────────────────────────────


def _extract_frame_payload(frame: Any) -> Any | None:
    """
    Normalize a Simli video frame for the WebRTC bridge.

    The web app uses Simli's LiveKit client directly. On phone we proxy through
    aiortc, so keep PyAV/numpy frames intact and let the WebRTC track convert
    them. This avoids an opencv-python dependency and avoids sending raw plane
    bytes as if they were JPEG frames.
    """
    if frame is None:
        return None

    if isinstance(frame, bytes):
        return frame

    if hasattr(frame, "reformat"):
        return frame

    try:
        if hasattr(frame, "to_ndarray"):
            try:
                return frame.to_ndarray(format="rgb24")
            except TypeError:
                return frame.to_ndarray()
    except Exception:
        pass

    if hasattr(frame, "ndim") and hasattr(frame, "shape"):
        return frame

    logger.warning(f"[SimliAvatar] Could not normalize frame type: {type(frame)}")
    return None


def _extract_frame_bytes(frame: Any) -> bytes | None:
    """
    Extract raw bytes from a Simli video frame.

    The Simli SDK returns PyAV VideoFrame objects. We need to extract
    the raw pixel data as bytes for transmission over WebSocket.

    Supports multiple frame formats:
    - VideoFrame with .to_ndarray() (NumPy array)
    - VideoFrame with .planes[0] (raw plane data)
    - Already a bytes object
    """
    if frame is None:
        return None

    # Already bytes
    if isinstance(frame, bytes):
        return frame

    try:
        # PyAV VideoFrame — .to_ndarray() returns (H, W, 3) RGB numpy array
        if hasattr(frame, "to_ndarray"):
            import numpy as np

            arr = frame.to_ndarray()
            # Convert RGB to BGR if needed (OpenCV convention)
            if arr.ndim == 3 and arr.shape[2] == 3:
                arr = arr[:, :, ::-1]
            # Encode as JPEG for WebSocket transmission
            import cv2

            _, jpeg_bytes = cv2.imencode(".jpg", arr)
            return jpeg_bytes.tobytes()
    except Exception:
        pass

    try:
        # PyAV VideoFrame — raw plane bytes
        if hasattr(frame, "planes") and frame.planes:
            plane = frame.planes[0]
            return bytes(plane)
    except Exception:
        pass

    try:
        # Direct bytes attribute
        if hasattr(frame, "to_bytes"):
            return frame.to_bytes()
    except Exception:
        pass

    logger.warning(f"[SimliAvatar] Could not extract bytes from frame type: {type(frame)}")
    return None


def encode_video_frame_for_websocket(frame: bytes) -> str:
    """
    Encode video frame bytes for WebSocket transmission.

    Returns a base64-encoded string that can be sent via WebSocket JSON.
    """
    return base64.b64encode(frame).decode("utf-8")
