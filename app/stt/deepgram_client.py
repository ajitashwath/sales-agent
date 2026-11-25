"""Streaming speech-to-text wrapper around the Deepgram Python SDK (v7).

The SDK owns a websocket to Deepgram; this class hides the lifecycle:

    stt = DeepgramStreamingSTT(on_partial=..., on_final=...)
    await stt.start()            # waits (briefly) for the connection
    await stt.send_audio(chunk)  # raw PCM chunks from the client
    await stt.finish()           # flush + close at end of call

Deepgram v7 uses `client.listen.v1.connect(...)` with event callbacks. We run
`start_listening()` in its own task because it blocks until the socket closes.

Audio format is configured in app/config.py: 16-bit PCM, 16 kHz, mono by
default. Partial transcripts are forwarded for UI feedback; final transcripts
trigger an agent turn. Voice-activity events (`vad_events=True`) are used for
barge-in: `SpeechStarted` tells the call handler the caller is talking, so it
can stop Riya mid-reply.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType

from app.config import get_settings

logger = logging.getLogger(__name__)

TranscriptHandler = Callable[[str], Awaitable[None]]
SpeechStartHandler = Callable[[], Awaitable[None]]


class DeepgramStreamingSTT:
    """One Deepgram live-transcription connection for one call."""

    def __init__(
        self,
        on_partial: TranscriptHandler,
        on_final: TranscriptHandler,
        on_error: TranscriptHandler | None = None,
        on_speech_started: SpeechStartHandler | None = None,
    ) -> None:
        self._settings = get_settings()
        self._on_partial = on_partial
        self._on_final = on_final
        self._on_error = on_error
        self._on_speech_started = on_speech_started

        self._client: AsyncDeepgramClient | None = None
        self._connection = None
        self._listen_task: asyncio.Task | None = None
        self._keepalive_task: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._closed = False
        self.last_error: str | None = None

    @property
    def connected(self) -> bool:
        return self._connection is not None

    async def start(self) -> None:
        """Open the Deepgram websocket. Failures are logged, never raised."""
        if not self._settings.deepgram_api_key:
            self.last_error = "DEEPGRAM_API_KEY is not configured"
            logger.warning(self.last_error)
            return

        self._ready.clear()
        self._closed = False
        self._client = AsyncDeepgramClient(api_key=self._settings.deepgram_api_key)
        self._listen_task = asyncio.create_task(self._run(), name="deepgram-listen")

        try:
            # Do not block the call forever if Deepgram is unreachable.
            await asyncio.wait_for(self._ready.wait(), timeout=10)
        except asyncio.TimeoutError:
            self.last_error = "Timed out connecting to Deepgram"
            logger.error(self.last_error)

    async def _run(self) -> None:
        """Own the connection from open to close."""
        try:
            async with self._client.listen.v1.connect(
                model=self._settings.deepgram_model,
                language="multi",  # English + Hindi code-switching (nova-3)
                encoding=self._settings.stt_encoding,
                sample_rate=self._settings.audio_sample_rate,
                channels=1,
                interim_results=True,
                punctuate=True,
                smart_format=True,
                endpointing=300,
                utterance_end_ms=1000,
                vad_events=True,
            ) as connection:
                self._connection = connection
                connection.on(EventType.OPEN, self._handle_open)
                connection.on(EventType.MESSAGE, self._handle_message)
                connection.on(EventType.ERROR, self._handle_error)
                connection.on(EventType.CLOSE, self._handle_close)

                self._ready.set()
                self._keepalive_task = asyncio.create_task(self._keepalive_loop())

                # Blocks until the socket closes (or raises, handled by the SDK).
                await connection.start_listening()

        except Exception as exc:
            self.last_error = f"Deepgram connection failed: {exc}"
            logger.exception("Deepgram streaming connection failed")
        finally:
            self._connection = None
            self._ready.set()  # unblock start() even when connect failed
            if self._keepalive_task:
                self._keepalive_task.cancel()

    # -- SDK event callbacks (the SDK awaits coroutine callbacks) -------------
    async def _handle_open(self, _event) -> None:
        logger.info("Deepgram live connection open (model=%s)", self._settings.deepgram_model)

    async def _handle_close(self, _event) -> None:
        logger.info("Deepgram live connection closed")

    async def _handle_error(self, error) -> None:
        self.last_error = str(error)
        logger.error("Deepgram streaming error: %s", error)
        if self._on_error is not None:
            await self._safe_call(self._on_error, "Speech recognition is having trouble.")

    async def _handle_message(self, message) -> None:
        """Forward partial/final transcripts and the VAD speech-start event."""
        # Deepgram emits ListenV1SpeechStarted.type == "SpeechStarted" when it
        # hears the caller begin to talk (enabled by vad_events=True). It has no
        # transcript, so handle it before touching message.channel.
        if getattr(message, "type", None) == "SpeechStarted":
            await self._notify_speech_started()
            return

        try:
            transcript = (message.channel.alternatives[0].transcript or "").strip()
        except (AttributeError, IndexError, TypeError):
            # Metadata / UtteranceEnd events have no channel either.
            return

        if not transcript:
            return

        is_final = bool(getattr(message, "is_final", False))
        try:
            if is_final:
                await self._on_final(transcript)
            else:
                await self._on_partial(transcript)
        except Exception:
            logger.exception("Transcript handler failed")

    async def _notify_speech_started(self) -> None:
        """Best-effort barge-in signal; never let it break the listen loop."""
        if self._on_speech_started is None:
            return
        try:
            await self._on_speech_started()
        except Exception:
            logger.exception("Speech-start handler failed")

    async def _safe_call(self, handler: TranscriptHandler, text: str) -> None:
        try:
            await handler(text)
        except Exception:
            logger.exception("STT error handler failed")

    async def _keepalive_loop(self) -> None:
        """Deepgram closes idle streams; keep the socket warm during silence."""
        try:
            while self._connection is not None and not self._closed:
                await asyncio.sleep(5)
                if self._connection is None or self._closed:
                    return
                await self._connection.send_keep_alive()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug("Deepgram keepalive stopped", exc_info=True)

    # -- Public control -------------------------------------------------------
    async def send_audio(self, chunk: bytes) -> None:
        """Forward one binary audio chunk. Dropped if the stream is not open."""
        if not chunk or self._connection is None:
            return
        try:
            await self._connection.send_media(chunk)
        except Exception:
            logger.exception("Failed to forward audio to Deepgram")

    async def finish(self) -> None:
        """Flush remaining audio and close the Deepgram socket."""
        self._closed = True

        if self._connection is not None:
            try:
                await self._connection.send_finalize()  # get the last partial segment
                await asyncio.sleep(0.3)
                await self._connection.send_close_stream()
            except Exception:
                logger.debug("Error while closing Deepgram stream", exc_info=True)

        if self._keepalive_task:
            self._keepalive_task.cancel()

        if self._listen_task is not None:
            try:
                await asyncio.wait_for(self._listen_task, timeout=5)
            except asyncio.TimeoutError:
                self._listen_task.cancel()
                logger.warning("Deepgram listen task did not stop in time; cancelled")
            except Exception:
                logger.debug("Deepgram listen task ended with an error", exc_info=True)

        self._connection = None
