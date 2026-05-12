"""WebRTC signaling agent for phone avatar video.

The phone app owns the WebRTC offer and sends PCM16 audio through a data
channel. This agent answers the offer, forwards audio to Simli, and exposes a
server-side video track fed by Simli frames.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import time
from fractions import Fraction
from typing import Any

from fastapi import WebSocket

from api.simli_avatar import get_simli_manager
from core.config import (
    WEBRTC_ICE_SERVERS_JSON,
    WEBRTC_STUN_URLS,
    WEBRTC_TURN_CREDENTIAL,
    WEBRTC_TURN_URLS,
    WEBRTC_TURN_USERNAME,
)

logger = logging.getLogger(__name__)


def _avatar_log(message: str, *args: Any) -> None:
    formatted = message % args if args else message
    print(f"[AvatarWebRTC] {formatted}", flush=True)
    logger.info(formatted)


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


def _configured_ice_servers() -> list[Any]:
    ice_servers: list[Any] = []

    for server in WEBRTC_ICE_SERVERS_JSON:
        urls = server.get("urls")
        if not urls:
            continue
        ice_servers.append(
            _rtc["RTCIceServer"](
                urls=urls,
                username=server.get("username"),
                credential=server.get("credential"),
            )
        )

    if ice_servers:
        return ice_servers

    for url in WEBRTC_STUN_URLS:
        ice_servers.append(_rtc["RTCIceServer"](urls=[url]))

    if WEBRTC_TURN_URLS:
        ice_servers.append(
            _rtc["RTCIceServer"](
                urls=WEBRTC_TURN_URLS,
                username=WEBRTC_TURN_USERNAME or None,
                credential=WEBRTC_TURN_CREDENTIAL or None,
            )
        )

    return ice_servers


def _ice_server_labels(ice_servers: list[Any]) -> list[str]:
    labels: list[str] = []
    for server in ice_servers:
        urls = getattr(server, "urls", None)
        if isinstance(urls, str):
            url_list = [urls]
        else:
            url_list = list(urls or [])
        for url in url_list:
            if url.startswith("turn:") or url.startswith("turns:"):
                labels.append(url.split("?", 1)[0] + "?transport=redacted")
            else:
                labels.append(url)
    return labels


def _sdp_summary(sdp: str | None) -> dict[str, Any]:
    if not sdp:
        return {"candidates": 0, "video_direction": None, "has_video": False}
    lines = sdp.splitlines()
    video_direction: str | None = None
    in_video = False
    for line in lines:
        if line.startswith("m="):
            in_video = line.startswith("m=video")
        if in_video and line in {
            "a=sendonly",
            "a=recvonly",
            "a=sendrecv",
            "a=inactive",
        }:
            video_direction = line[2:]
            break
    return {
        "candidates": sum(1 for line in lines if line.startswith("a=candidate:")),
        "video_direction": video_direction,
        "has_video": "m=video" in sdp,
    }


class AvatarMediaStreamTrack(_rtc["VideoStreamTrack"]):
    """Video track backed by frames pushed from Simli."""

    kind = "video"

    def __init__(self, session_id: str, max_queue_size: int = 2) -> None:
        super().__init__()
        self._session_id = session_id
        self._queue: asyncio.Queue[Any | None] = asyncio.Queue(maxsize=max_queue_size)
        self._timestamp = 0
        self._time_base = Fraction(1, 90000)
        self._frame_duration = 3000  # 90kHz / 30fps
        self._frame_interval = 1 / 30
        self._real_frame_count = 0
        self._placeholder_frame_count = 0
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
        self._real_frame_count += 1
        _avatar_log(
            "%s queued real video frame #%s type=%s queue=%s",
            self._session_id,
            self._real_frame_count,
            type(frame).__name__,
            self._queue.qsize(),
        )

    def close(self) -> None:
        self._closed = True
        _avatar_log(
            "%s video track close requested real_frames=%s placeholders=%s",
            self._session_id,
            self._real_frame_count,
            self._placeholder_frame_count,
        )
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    @property
    def real_frame_count(self) -> int:
        return self._real_frame_count

    async def recv(self) -> Any:
        if self._closed:
            raise _rtc["MediaStreamError"]

        await asyncio.sleep(self._frame_interval)
        try:
            frame_payload = await asyncio.wait_for(
                self._queue.get(),
                timeout=0.001,
            )
        except asyncio.TimeoutError:
            frame_payload = self._make_placeholder_frame()
            self._placeholder_frame_count += 1
            if (
                self._placeholder_frame_count <= 3
                or self._placeholder_frame_count % 90 == 0
            ):
                _avatar_log(
                    "%s sending placeholder video frame #%s while waiting for Simli",
                    self._session_id,
                    self._placeholder_frame_count,
                )

        if frame_payload is None:
            raise _rtc["MediaStreamError"]

        frame = self._coerce_video_frame(frame_payload)
        self._timestamp += self._frame_duration
        frame.pts = self._timestamp
        frame.time_base = self._time_base
        if self._real_frame_count <= 5 or self._real_frame_count % 30 == 0:
            _avatar_log(
                "%s sending video frame #%s real=%s placeholder=%s",
                self._session_id,
                self._real_frame_count + self._placeholder_frame_count,
                self._real_frame_count,
                self._placeholder_frame_count,
            )
        return frame

    def _make_placeholder_frame(self) -> Any:
        av = _rtc["av"]
        frame = av.VideoFrame(width=640, height=640, format="yuv420p")
        # yuv420p neutral gray: Y=16, U=128, V=128. U/V=0 renders as green.
        frame.planes[0].update(bytes([16]) * frame.planes[0].buffer_size)
        frame.planes[1].update(bytes([128]) * frame.planes[1].buffer_size)
        frame.planes[2].update(bytes([128]) * frame.planes[2].buffer_size)
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
        self._video_track = AvatarMediaStreamTrack(session_id=session_id)
        self._simli = get_simli_manager()
        self._started = False
        self._simli_started = False
        self._simli_task: asyncio.Task | None = None
        self._silence_task: asyncio.Task | None = None
        self._pending_audio_chunks: list[bytes] = []
        self._remote_track_tasks: list[asyncio.Task] = []
        self._closed = False
        self._local_ice_count = 0
        self._remote_ice_count = 0
        self._signal_message_count = 0
        self._audio_chunk_count = 0
        self._ice_send_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def start(self) -> None:
        if self._started:
            _avatar_log("%s start skipped; agent already started", self.session_id)
            return

        _avatar_log("%s creating peer connection gender=%s", self.session_id, self.gender)
        ice_servers = _configured_ice_servers()
        _avatar_log(
            "%s ICE servers configured: %s",
            self.session_id,
            _ice_server_labels(ice_servers),
        )
        config = _rtc["RTCConfiguration"](iceServers=ice_servers)
        self._pc = _rtc["RTCPeerConnection"](configuration=config)
        self._pc.addTrack(self._video_track)
        _avatar_log("%s outbound avatar video track added", self.session_id)

        @self._pc.on("datachannel")
        def on_datachannel(channel: Any) -> None:
            _avatar_log("%s data channel received label=%s", self.session_id, channel.label)

            @channel.on("open")
            def on_open() -> None:
                _avatar_log("%s data channel open label=%s", self.session_id, channel.label)

            @channel.on("close")
            def on_close() -> None:
                _avatar_log("%s data channel closed label=%s", self.session_id, channel.label)

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
                    self._audio_chunk_count += 1
                    if self._audio_chunk_count <= 3 or self._audio_chunk_count % 25 == 0:
                        _avatar_log(
                            "%s received avatar audio chunk #%s bytes=%s simli_started=%s pending=%s",
                            self.session_id,
                            self._audio_chunk_count,
                            len(message),
                            self._simli_started,
                            len(self._pending_audio_chunks),
                        )
                    asyncio.create_task(self._send_audio(message))

        @self._pc.on("icecandidate")
        async def on_icecandidate(candidate: Any) -> None:
            candidate_json = self._candidate_to_json(candidate)
            if candidate_json is None:
                _avatar_log("%s local ICE complete", self.session_id)
            else:
                self._local_ice_count += 1
                if self._local_ice_count <= 5 or self._local_ice_count % 10 == 0:
                    _avatar_log(
                        "%s local ICE candidate #%s mid=%s mline=%s",
                        self.session_id,
                        self._local_ice_count,
                        candidate_json.get("sdpMid"),
                        candidate_json.get("sdpMLineIndex"),
                    )
            await self._ice_send_queue.put(candidate_json)

        @self._pc.on("track")
        def on_track(track: Any) -> None:
            _avatar_log(
                "%s ignoring remote %s track",
                self.session_id,
                getattr(track, "kind", "unknown"),
            )
            self._remote_track_tasks.append(
                asyncio.create_task(self._drain_remote_track(track))
            )

        @self._pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            state = getattr(self._pc, "connectionState", "unknown")
            _avatar_log("%s peer connection state=%s", self.session_id, state)
            if state in ("failed", "closed", "disconnected"):
                await self.close()

        self._started = True
        _avatar_log("%s agent started", self.session_id)

    async def _drain_remote_track(self, track: Any) -> None:
        try:
            while not self._closed:
                await track.recv()
        except _rtc["MediaStreamError"]:
            _avatar_log(
                "%s remote %s track ended",
                self.session_id,
                getattr(track, "kind", "unknown"),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _avatar_log("%s remote track drain failed: %s", self.session_id, exc)

    def _ensure_simli_started(self) -> None:
        if self._simli_task is not None or self._closed:
            _avatar_log(
                "%s Simli start skipped task_exists=%s closed=%s started=%s",
                self.session_id,
                self._simli_task is not None,
                self._closed,
                self._simli_started,
            )
            return
        _avatar_log("%s scheduling Simli start gender=%s", self.session_id, self.gender)
        self._simli_task = asyncio.create_task(self._start_simli_session())

    async def _start_simli_session(self) -> None:
        try:
            _avatar_log("%s Simli start begin gender=%s", self.session_id, self.gender)
            await self._simli.start_session(
                session_id=self.session_id,
                gender=self.gender,
                video_frame_callback=self._video_track.enqueue_frame,
                error_callback=lambda message: logger.error(
                    "[AvatarWebRTC] Simli error %s: %s", self.session_id, message
                ),
            )
            self._simli_started = True
            _avatar_log("%s Simli started", self.session_id)
            self._start_silence_keepalive()
            await self._flush_pending_audio()
        except asyncio.CancelledError:
            _avatar_log("%s Simli start cancelled", self.session_id)
            raise
        except Exception as exc:
            _avatar_log("%s Simli start failed: %s", self.session_id, exc)
            logger.exception("[AvatarWebRTC] Simli start traceback for %s", self.session_id)

    def _start_silence_keepalive(self) -> None:
        if self._silence_task is not None or self._closed:
            return
        self._silence_task = asyncio.create_task(self._silence_keepalive_loop())

    async def _silence_keepalive_loop(self) -> None:
        try:
            while not self._closed and self._video_track.real_frame_count == 0:
                _avatar_log(
                    "%s sending Simli silence keepalive real_frames=%s",
                    self.session_id,
                    self._video_track.real_frame_count,
                )
                ok = await self._simli.send_silence(self.session_id)
                if not ok:
                    ok = await self._simli.send_audio(self.session_id, bytes(32000))
                    _avatar_log("%s fallback zero PCM keepalive sent ok=%s", self.session_id, ok)
                await asyncio.sleep(2)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _avatar_log("%s silence keepalive failed: %s", self.session_id, exc)

    async def _send_audio(self, audio_bytes: bytes) -> None:
        if not self._simli_started:
            self._ensure_simli_started()
            self._pending_audio_chunks.append(audio_bytes)
            if len(self._pending_audio_chunks) > 16:
                self._pending_audio_chunks.pop(0)
                _avatar_log("%s dropped oldest pending avatar audio chunk", self.session_id)
            return
        ok = await self._simli.send_audio(self.session_id, audio_bytes)
        if not ok:
            _avatar_log("%s Simli send_audio returned false bytes=%s", self.session_id, len(audio_bytes))

    async def _flush_pending_audio(self) -> None:
        chunks = list(self._pending_audio_chunks)
        self._pending_audio_chunks.clear()
        _avatar_log("%s flushing %s pending avatar audio chunks", self.session_id, len(chunks))
        for chunk in chunks:
            ok = await self._simli.send_audio(self.session_id, chunk)
            if not ok:
                _avatar_log("%s pending avatar audio send failed bytes=%s", self.session_id, len(chunk))

    async def handle_offer(self, sdp: str, offer_type: str = "offer") -> dict[str, str]:
        if self._pc is None:
            await self.start()

        _avatar_log(
            "%s handling offer type=%s sdp_len=%s has_video=%s has_audio=%s has_data=%s",
            self.session_id,
            offer_type,
            len(sdp),
            "m=video" in sdp,
            "m=audio" in sdp,
            "m=application" in sdp,
        )
        if "m=video" not in sdp:
            raise ValueError("Avatar WebRTC offer is missing a recvonly video transceiver")

        offer = _rtc["RTCSessionDescription"](sdp=sdp, type=offer_type)
        await self._pc.setRemoteDescription(offer)
        _avatar_log("%s remote description applied", self.session_id)
        answer = await self._pc.createAnswer()
        await self._pc.setLocalDescription(answer)
        answer_sdp = self._pc.localDescription.sdp
        summary = _sdp_summary(answer_sdp)
        _avatar_log(
            "%s local answer ready type=%s sdp_len=%s has_video=%s video_direction=%s candidates=%s",
            self.session_id,
            self._pc.localDescription.type,
            len(answer_sdp or ""),
            summary["has_video"],
            summary["video_direction"],
            summary["candidates"],
        )
        return {"type": self._pc.localDescription.type, "sdp": answer_sdp}

    async def add_ice_candidate(self, payload: dict[str, Any]) -> None:
        if self._pc is None:
            _avatar_log("%s ignoring remote ICE because pc is not ready", self.session_id)
            return

        candidate_text = payload.get("candidate")
        if not candidate_text:
            _avatar_log("%s remote ICE complete", self.session_id)
            await self._pc.addIceCandidate(None)
            return

        if candidate_text.startswith("candidate:"):
            candidate_text = candidate_text[len("candidate:") :]
        candidate = _rtc["candidate_from_sdp"](candidate_text)
        candidate.sdpMid = payload.get("sdpMid")
        candidate.sdpMLineIndex = payload.get("sdpMLineIndex")
        await self._pc.addIceCandidate(candidate)
        self._remote_ice_count += 1
        if self._remote_ice_count <= 5 or self._remote_ice_count % 10 == 0:
            _avatar_log(
                "%s remote ICE candidate #%s mid=%s mline=%s",
                self.session_id,
                self._remote_ice_count,
                candidate.sdpMid,
                candidate.sdpMLineIndex,
            )

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
        _avatar_log("%s signaling run begin gender=%s", self.session_id, self.gender)
        await self.start()
        self._ensure_simli_started()

        async def send_local_ice() -> None:
            while not self._closed:
                candidate = await self.next_local_ice()
                if self._closed:
                    break
                if candidate is None:
                    _avatar_log("%s sending ICE complete to phone", self.session_id)
                else:
                    _avatar_log(
                        "%s sending local ICE to phone mid=%s mline=%s",
                        self.session_id,
                        candidate.get("sdpMid"),
                        candidate.get("sdpMLineIndex"),
                    )
                await websocket.send_json(candidate or {"type": "ice_complete"})

        ice_task = asyncio.create_task(send_local_ice())
        try:
            await websocket.send_json({"type": "avatar_ready", "session_id": self.session_id})
            _avatar_log("%s sent avatar_ready", self.session_id)
            while True:
                raw = await websocket.receive_text()
                self._signal_message_count += 1
                _avatar_log(
                    "%s received signaling message #%s bytes=%s",
                    self.session_id,
                    self._signal_message_count,
                    len(raw),
                )
                payload = json.loads(raw)
                message_type = payload.get("type")
                _avatar_log("%s signaling type=%s", self.session_id, message_type)
                if message_type == "offer":
                    answer = await self.handle_offer(payload["sdp"], payload.get("type", "offer"))
                    await websocket.send_json({"type": "answer", **answer})
                    _avatar_log("%s sent answer to phone", self.session_id)
                    self._ensure_simli_started()
                elif message_type == "ice":
                    await self.add_ice_candidate(payload)
                elif message_type == "ping":
                    await websocket.send_json({"type": "pong", "ts": time.time()})
                    _avatar_log("%s sent pong", self.session_id)
                else:
                    _avatar_log("%s ignored unknown signaling type=%s", self.session_id, message_type)
        except Exception as exc:
            _avatar_log("%s signaling loop exception: %s", self.session_id, exc)
            raise
        finally:
            _avatar_log("%s signaling run closing", self.session_id)
            ice_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ice_task
            await self.close()

    async def close(self) -> None:
        if self._closed:
            _avatar_log("%s close skipped; already closed", self.session_id)
            return
        _avatar_log(
            "%s close begin local_ice=%s remote_ice=%s signals=%s audio_chunks=%s simli_started=%s",
            self.session_id,
            self._local_ice_count,
            self._remote_ice_count,
            self._signal_message_count,
            self._audio_chunk_count,
            self._simli_started,
        )
        self._closed = True
        if self._simli_task is not None:
            self._simli_task.cancel()
            try:
                await self._simli_task
            except asyncio.CancelledError:
                pass
            self._simli_task = None
        if self._silence_task is not None:
            self._silence_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._silence_task
            self._silence_task = None
        for task in self._remote_track_tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._remote_track_tasks.clear()
        self._pending_audio_chunks.clear()
        self._simli_started = False
        self._video_track.close()
        await self._simli.stop_session(self.session_id)
        if self._pc is not None:
            await self._pc.close()
            self._pc = None
        _avatar_log("%s close complete", self.session_id)
