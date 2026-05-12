"""WebRTC signaling agent for phone avatar video.

The phone app owns the WebRTC offer and sends PCM16 audio through a data
channel. This agent answers the offer, forwards audio to Simli, and exposes a
server-side video track fed by Simli frames.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from fractions import Fraction
from typing import Any

from fastapi import WebSocket

from api.simli_avatar import get_simli_manager

logger = logging.getLogger(__name__)


def _load_aiortc() -> dict[str, Any]:
    try:
        from aiortc import (
            RTCConfiguration,
            RTCIceCandidate,
            RTCIceServer,
            RTCPeerConnection,
            RTCSessionDescription,
            VideoStreamTrack,
        )
        from aiortc.mediastreams import MediaStreamError
        from aiortc.sdp import candidate_from_sdp, candidate_to_sdp
        import av
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "aiortc is not available. Install aiortc>=1.9.0"
        ) from exc

    return {
        "RTCConfiguration": RTCConfiguration,
        "RTCIceCandidate": RTCIceCandidate,
        "RTCIceServer": RTCIceServer,
        "RTCPeerConnection": RTCPeerConnection,
        "RTCSessionDescription": RTCSessionDescription,
        "VideoStreamTrack": VideoStreamTrack,
        "MediaStreamError": MediaStreamError,
        "candidate_from_sdp": candidate_from_sdp,
        "candidate_to_sdp": candidate_to_sdp,
        "av": av,
    }


_rtc = _load_aiortc()


class AvatarMediaStreamTrack(_rtc["VideoStreamTrack"]):
    """Video track backed by frames pushed from Simli."""

    kind = "video"

    def __init__(self, max_queue_size: int = 2) -> None:
        super().__init__()
        self._queue: asyncio.Queue[Any | None] = asyncio.Queue(maxsize=max_queue_size)
        self._timestamp = 0
        self._time_base = Fraction(1, 90000)
        self._frame_duration = 3000  # 90kHz / 30fps
        self._closed = False

    def enqueue_frame(self, frame: Any) -> None:
        if self._closed or frame is None:
            return

        while self._queue.full():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
        self._queue.put_nowait(frame)

    def close(self) -> None:
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    async def recv(self) -> Any:
        if self._closed:
            raise _rtc["MediaStreamError"]

        frame_payload = await self._queue.get()
        if frame_payload is None:
            raise _rtc["MediaStreamError"]

        frame = self._coerce_video_frame(frame_payload)
        self._timestamp += self._frame_duration
        frame.pts = self._timestamp
        frame.time_base = self._time_base
        return frame

    def _coerce_video_frame(self, frame_payload: Any) -> Any:
        if hasattr(frame_payload, "reformat"):
            return frame_payload.reformat(format="yuv420p")

        if hasattr(frame_payload, "ndim") and hasattr(frame_payload, "shape"):
            av = _rtc["av"]
            ndim = getattr(frame_payload, "ndim", None)
            shape = getattr(frame_payload, "shape", ())
            if ndim == 2:
                return av.VideoFrame.from_ndarray(
                    frame_payload,
                    format="gray",
                ).reformat(format="yuv420p")
            if ndim == 3 and len(shape) >= 3:
                channels = shape[2]
                if channels == 3:
                    return av.VideoFrame.from_ndarray(
                        frame_payload,
                        format="rgb24",
                    ).reformat(format="yuv420p")
                if channels == 4:
                    return av.VideoFrame.from_ndarray(
                        frame_payload,
                        format="rgba",
                    ).reformat(format="yuv420p")

        if isinstance(frame_payload, bytes):
            return self._decode_encoded_frame(frame_payload)

        raise ValueError(f"Unsupported avatar frame payload: {type(frame_payload)}")

    def _decode_encoded_frame(self, frame_bytes: bytes) -> Any:
        av = _rtc["av"]
        try:
            container = av.open(io.BytesIO(frame_bytes))
        except Exception:
            container = av.open(io.BytesIO(frame_bytes), format="mjpeg")
        with container:
            for frame in container.decode(video=0):
                return frame.reformat(format="yuv420p")
        raise ValueError("Could not decode avatar video frame")


class AvatarWebRTCAgent:
    """Owns one WebRTC peer connection for a phone avatar session."""

    def __init__(self, session_id: str, gender: str = "male") -> None:
        self.session_id = session_id
        self.gender = gender
        self._pc: Any | None = None
        self._video_track = AvatarMediaStreamTrack()
        self._simli = get_simli_manager()
        self._started = False
        self._simli_started = False
        self._simli_task: asyncio.Task | None = None
        self._pending_audio_chunks: list[bytes] = []
        self._closed = False
        self._ice_send_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def start(self) -> None:
        if self._started:
            return

        ice_server = _rtc["RTCIceServer"](urls=["stun:stun.l.google.com:19302"])
        config = _rtc["RTCConfiguration"](iceServers=[ice_server])
        self._pc = _rtc["RTCPeerConnection"](configuration=config)
        self._pc.addTrack(self._video_track)

        @self._pc.on("datachannel")
        def on_datachannel(channel: Any) -> None:
            logger.info("[AvatarWebRTC] DataChannel opened: %s", channel.label)

            @channel.on("message")
            def on_message(message: Any) -> None:
                if isinstance(message, str):
                    try:
                        payload = json.loads(message)
                    except json.JSONDecodeError:
                        payload = {}
                    if payload.get("type") == "ping":
                        channel.send(json.dumps({"type": "pong", "ts": time.time()}))
                    return

                if isinstance(message, bytearray):
                    message = bytes(message)
                if isinstance(message, bytes):
                    asyncio.create_task(self._send_audio(message))

        @self._pc.on("icecandidate")
        async def on_icecandidate(candidate: Any) -> None:
            await self._ice_send_queue.put(self._candidate_to_json(candidate))

        @self._pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            state = getattr(self._pc, "connectionState", "unknown")
            logger.info("[AvatarWebRTC] connection state %s: %s", self.session_id, state)
            if state in ("failed", "closed", "disconnected"):
                await self.close()

        self._started = True

    def _ensure_simli_started(self) -> None:
        if self._simli_task is not None or self._closed:
            return
        self._simli_task = asyncio.create_task(self._start_simli_session())

    async def _start_simli_session(self) -> None:
        try:
            await self._simli.start_session(
                session_id=self.session_id,
                gender=self.gender,
                video_frame_callback=self._video_track.enqueue_frame,
                error_callback=lambda message: logger.error(
                    "[AvatarWebRTC] Simli error %s: %s", self.session_id, message
                ),
            )
            self._simli_started = True
            logger.info("[AvatarWebRTC] Simli started: %s", self.session_id)
            await self._flush_pending_audio()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("[AvatarWebRTC] Simli start failed %s: %s", self.session_id, exc)

    async def _send_audio(self, audio_bytes: bytes) -> None:
        if not self._simli_started:
            self._ensure_simli_started()
            self._pending_audio_chunks.append(audio_bytes)
            if len(self._pending_audio_chunks) > 16:
                self._pending_audio_chunks.pop(0)
            return
        await self._simli.send_audio(self.session_id, audio_bytes)

    async def _flush_pending_audio(self) -> None:
        chunks = list(self._pending_audio_chunks)
        self._pending_audio_chunks.clear()
        for chunk in chunks:
            await self._simli.send_audio(self.session_id, chunk)

    async def handle_offer(self, sdp: str, offer_type: str = "offer") -> dict[str, str]:
        if self._pc is None:
            await self.start()

        if "m=video" not in sdp:
            raise ValueError("Avatar WebRTC offer is missing a recvonly video transceiver")

        offer = _rtc["RTCSessionDescription"](sdp=sdp, type=offer_type)
        await self._pc.setRemoteDescription(offer)
        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)
        return {"type": self._pc.localDescription.type, "sdp": self._pc.localDescription.sdp}

    async def add_ice_candidate(self, payload: dict[str, Any]) -> None:
        if self._pc is None:
            return

        candidate_text = payload.get("candidate")
        if not candidate_text:
            await self._pc.addIceCandidate(None)
            return

        if candidate_text.startswith("candidate:"):
            candidate_text = candidate_text[len("candidate:") :]
        candidate = _rtc["candidate_from_sdp"](candidate_text)
        candidate.sdpMid = payload.get("sdpMid")
        candidate.sdpMLineIndex = payload.get("sdpMLineIndex")
        await self._pc.addIceCandidate(candidate)

    async def next_local_ice(self) -> dict[str, Any] | None:
        return await self._ice_send_queue.get()

    def _candidate_to_json(self, candidate: Any | None) -> dict[str, Any] | None:
        if candidate is None:
            return None
        return {
            "type": "ice",
            "candidate": "candidate:" + _rtc["candidate_to_sdp"](candidate),
            "sdpMid": candidate.sdpMid,
            "sdpMLineIndex": candidate.sdpMLineIndex,
        }

    async def run_signaling(self, websocket: WebSocket) -> None:
        await self.start()
        self._ensure_simli_started()

        async def send_local_ice() -> None:
            while not self._closed:
                candidate = await self.next_local_ice()
                if self._closed:
                    break
                await websocket.send_json(candidate or {"type": "ice_complete"})

        ice_task = asyncio.create_task(send_local_ice())
        try:
            await websocket.send_json({"type": "avatar_ready", "session_id": self.session_id})
            while True:
                payload = json.loads(await websocket.receive_text())
                message_type = payload.get("type")
                if message_type == "offer":
                    answer = await self.handle_offer(payload["sdp"], payload.get("type", "offer"))
                    await websocket.send_json({"type": "answer", **answer})
                    self._ensure_simli_started()
                elif message_type == "ice":
                    await self.add_ice_candidate(payload)
                elif message_type == "ping":
                    await websocket.send_json({"type": "pong", "ts": time.time()})
        finally:
            ice_task.cancel()
            await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._simli_task is not None:
            self._simli_task.cancel()
            try:
                await self._simli_task
            except asyncio.CancelledError:
                pass
            self._simli_task = None
        self._pending_audio_chunks.clear()
        self._simli_started = False
        self._video_track.close()
        await self._simli.stop_session(self.session_id)
        if self._pc is not None:
            await self._pc.close()
            self._pc = None
