"""FastAPI app and the real-time voice WebSocket.

Flow per call:
  1. client connects to /ws/call/{session_id}
  2. binary audio chunks are forwarded to Deepgram streaming STT
  3. each final transcript triggers one Gemini agent turn (with tool calls)
  4. the reply text is sent to the client and streamed to ElevenLabs TTS,
     whose audio chunks are sent back over the same WebSocket
  5. on "end_call" (or disconnect) the transcript is summarised, attached to
     the lead record, and returned as a final JSON message

Barge-in: when the caller starts speaking (Deepgram VAD "SpeechStarted") while
Riya is thinking or speaking, the current reply's audio is cut within about one
chunk, the clip closes with "interrupted": true, and any turns queued during
that reply are dropped. The caller's next final transcript starts a fresh turn.

Message protocol (server -> client):
  {"type": "ready", ...}                       session is live
  {"type": "partial_transcript", "text": ...}  interim STT result
  {"type": "final_transcript", "text": ...}    final STT result (a turn started)
  {"type": "thinking"}                         agent is working
  {"type": "assistant_text", "text": ...}      reply to speak
  {"type": "audio_start", "format": "pcm_16000"} then binary audio frames,
  {"type": "audio_end", "interrupted": bool}

Control JSON frames can be interleaved between binary audio frames (a partial
transcript arriving while Riya is still speaking, for example); clients should
treat everything between audio_start and audio_end as one clip.
  {"type": "call_summary", "summary": {...}, "lead_id": 1|null}
  {"type": "error" | "warning", "message": ...}

Client -> server:
  binary: audio frames
  {"type": "text", "text": "..."}    optional typed turn (handy without a mic)
  {"type": "end_call"}               finish the call and get the summary
  {"type": "ping"}                   -> {"type": "pong"}
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import aclosing, asynccontextmanager, suppress

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.agent.gemini_agent import AgentUnavailable, run_agent_turn
from app.agent.memory import CallSession, drop_session, get_session
from app.config import get_settings
from app.crm.crud import list_leads, save_call_summary
from app.crm.models import init_db
from app.stt.deepgram_client import DeepgramStreamingSTT
from app.summary.call_summary import summarize_call
from app.tts.elevenlabs_client import ElevenLabsTTS, TTSError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Spoken when the agent or a downstream API fails mid-turn, so the caller hears
# something sensible and the WebSocket stays open.
AGENT_FALLBACK = (
    "Sorry, I'm having trouble thinking right now. Could you give me a moment and say that again?"
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Create the SQLite tables and flag missing configuration at startup."""
    init_db()
    settings = get_settings()
    missing = [
        name
        for name, value in {
            "DEEPGRAM_API_KEY": settings.deepgram_api_key,
            "GEMINI_API_KEY": settings.gemini_api_key,
            "ELEVENLABS_API_KEY": settings.elevenlabs_api_key,
        }.items()
        if not value
    ]
    if missing:
        logger.warning("Missing credentials: %s (affected features will degrade)", ", ".join(missing))
    logger.info("Sales voice agent ready (company=%s)", settings.company_name)
    yield


app = FastAPI(
    title="Autonomous AI Sales Voice Agent",
    version="0.1.0",
    description="Streaming voice sales agent: Deepgram STT -> Gemini + tools -> ElevenLabs TTS.",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    """Cheap liveness check."""
    settings = get_settings()
    return {
        "status": "ok",
        "deepgram_configured": bool(settings.deepgram_api_key),
        "gemini_configured": bool(settings.gemini_api_key),
        "elevenlabs_configured": bool(settings.elevenlabs_api_key),
    }


@app.get("/leads")
async def leads(limit: int = 50) -> dict:
    """Read-only view of the mock CRM (handy during a demo)."""
    return {"leads": [lead.to_dict() for lead in list_leads(limit=limit)]}


@app.websocket("/ws/call/{session_id}")
async def call_websocket(websocket: WebSocket, session_id: str) -> None:
    """One live sales call."""
    await websocket.accept()

    settings = get_settings()
    session = get_session(session_id)
    tts = ElevenLabsTTS()

    # Serialises individual outgoing frames: STT callbacks, agent replies and
    # audio chunks are produced by different tasks. The lock is taken per frame,
    # never for a whole audio clip, so a long reply cannot stall transcription.
    send_lock = asyncio.Lock()
    turn_queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def send_json(payload: dict) -> None:
        async with send_lock:
            try:
                await websocket.send_text(json.dumps(payload, ensure_ascii=False, default=str))
            except Exception:
                # The client usually went away; nothing useful to do here.
                logger.debug("Dropping message, websocket is closed", exc_info=True)

    async def send_bytes(chunk: bytes) -> None:
        async with send_lock:
            try:
                await websocket.send_bytes(chunk)
            except Exception:
                logger.debug("Dropping audio frame, websocket is closed", exc_info=True)

    async def on_partial(text: str) -> None:
        await send_json({"type": "partial_transcript", "text": text})

    async def on_final(text: str) -> None:
        await send_json({"type": "final_transcript", "text": text})
        await turn_queue.put(text)

    async def on_stt_error(message: str) -> None:
        await send_json({"type": "warning", "message": message})

    # --- Barge-in ---------------------------------------------------------
    # `generation` is bumped whenever the caller speaks over Riya. A reply only
    # keeps streaming audio while its generation is still current, so playback
    # stops within roughly one chunk (~100 ms) of the caller starting to talk.
    generation = 0
    riya_busy = False  # thinking or speaking: the only states she can be cut off in

    async def on_speech_started() -> None:
        """Deepgram VAD says the caller is talking, not just background noise."""
        nonlocal generation
        if not riya_busy:
            # Nothing to interrupt: their speech becomes an ordinary turn.
            return

        generation += 1
        dropped = _drain_queue(turn_queue)
        logger.info(
            "session=%s barge-in: cutting the reply, dropped %s queued turn(s)",
            session_id,
            dropped,
        )

    stt = DeepgramStreamingSTT(
        on_partial=on_partial,
        on_final=on_final,
        on_error=on_stt_error,
        on_speech_started=on_speech_started,
    )
    await stt.start()

    await send_json(
        {
            "type": "ready",
            "session_id": session_id,
            "sample_rate": settings.audio_sample_rate,
            "stt_connected": stt.connected,
            "tts_format": settings.elevenlabs_output_format,
        }
    )
    if not stt.connected:
        await send_json(
            {
                "type": "warning",
                "message": (
                    "Streaming speech-to-text is unavailable. "
                    'Send {"type": "text", "text": "..."} messages instead.'
                ),
            }
        )

    async def stream_reply_audio(reply: str, turn_generation: int) -> bool:
        """Send one TTS clip, one chunk at a time. Failures are non-fatal.

        Stops early if the caller barges in; `aclosing` then closes the HTTP
        response so ElevenLabs stops synthesising the rest of the clip.

        Returns True when barge-in cut the clip short, so the transcript can
        record that the caller only heard part of the reply.
        """
        interrupted = False
        await send_json({"type": "audio_start", "format": settings.elevenlabs_output_format})
        try:
            async with aclosing(tts.stream_speech(reply)) as audio:
                async for chunk in audio:
                    if turn_generation != generation:
                        interrupted = True
                        break
                    await send_bytes(chunk)
        except TTSError as exc:
            logger.error("Text-to-speech failed: %s", exc)
            await send_json({"type": "warning", "message": "Voice playback is temporarily unavailable."})
        except Exception:
            logger.exception("Streaming TTS audio failed")
        finally:
            # Always close the clip, so a client knows to stop playback now.
            await send_json({"type": "audio_end", "interrupted": interrupted})
        return interrupted

    async def handle_turn(user_text: str) -> None:
        """Full pipeline for one user utterance."""
        nonlocal riya_busy

        if not user_text.strip():
            return

        # The single turn_worker calls this, so turns are already serialised.
        turn_generation = generation
        riya_busy = True
        try:
            session.record_user(user_text)
            await send_json({"type": "thinking"})

            try:
                reply = await run_agent_turn(session, user_text)
            except AgentUnavailable:
                logger.error("Gemini is not configured; sending fallback reply")
                reply = AGENT_FALLBACK
            except Exception:
                logger.exception("Agent turn failed")
                reply = AGENT_FALLBACK

            if turn_generation != generation:
                # The caller interrupted while the agent was still thinking:
                # their new turn matters, not this reply.
                logger.info("session=%s dropping stale reply (barge-in)", session_id)
                return

            await send_json({"type": "assistant_text", "text": reply})
            interrupted = await stream_reply_audio(reply, turn_generation)
            # Recorded after playback, so the transcript says whether the caller
            # actually heard the whole reply (the summariser depends on it).
            session.record_agent(reply, interrupted=interrupted)
        finally:
            riya_busy = False

    async def turn_worker() -> None:
        """Consume final transcripts sequentially so replies stay ordered."""
        while True:
            user_text = await turn_queue.get()
            try:
                if user_text is None:
                    return
                await handle_turn(user_text)
            except Exception:
                logger.exception("Turn handling failed")
            finally:
                turn_queue.task_done()

    worker_task = asyncio.create_task(turn_worker(), name=f"turns-{session_id}")

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            audio_chunk = message.get("bytes")
            if audio_chunk:
                await stt.send_audio(audio_chunk)
                continue

            raw_text = message.get("text")
            if raw_text is None:
                continue

            try:
                payload = json.loads(raw_text)
            except json.JSONDecodeError:
                await send_json({"type": "error", "message": "Control messages must be JSON."})
                continue

            message_type = payload.get("type")
            if message_type == "end_call":
                logger.info("session=%s client requested end_call", session_id)
                break
            if message_type == "text":
                await turn_queue.put(str(payload.get("text", "")))
            elif message_type == "ping":
                await send_json({"type": "pong"})
            else:
                await send_json({"type": "error", "message": f"Unknown message type: {message_type!r}"})

    except WebSocketDisconnect:
        logger.info("session=%s client disconnected", session_id)
    except Exception:
        logger.exception("session=%s websocket loop failed", session_id)
    finally:
        # 1. Close STT (flushes any last final transcript into the queue).
        try:
            await stt.finish()
        except Exception:
            logger.exception("Failed to close Deepgram stream cleanly")

        # 2. Let queued turns finish before summarising.
        await turn_queue.put(None)
        try:
            await asyncio.wait_for(worker_task, timeout=120)
        except asyncio.TimeoutError:
            logger.warning("session=%s turn worker timed out; cancelling", session_id)
            worker_task.cancel()
            # Await the cancellation, otherwise the task is left pending and
            # Python complains about it once the connection is gone.
            with suppress(asyncio.CancelledError):
                await worker_task
        except Exception:
            logger.exception("Turn worker ended with an error")

        await tts.close()

        # 3. Summarise the call and attach it to the lead record (if any).
        summary = await _summarise_session(session)

        # 4. Best-effort final JSON message, then close.
        try:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "call_summary",
                        "session_id": session_id,
                        "lead_id": session.lead_id,
                        "summary": summary,
                    },
                    default=str,
                )
            )
            await websocket.close()
        except Exception:
            logger.debug("Could not send call summary (socket already closed)", exc_info=True)

        drop_session(session_id)
        logger.info(
            "session=%s call finished (turns=%s lead_id=%s)",
            session_id,
            len(session.transcript),
            session.lead_id,
        )


async def _summarise_session(session: CallSession) -> dict:
    """Summarise and persist, tolerating a failed summariser or missing lead."""
    summary = await summarize_call(session)

    if session.lead_id is not None:
        try:
            save_call_summary(session.lead_id, summary)
        except Exception:
            logger.exception("Failed to attach summary to lead %s", session.lead_id)

    return summary


def _drain_queue(queue: asyncio.Queue) -> int:
    """Discard everything still queued (turns captured during an interrupted reply).

    Returns how many items were dropped. `task_done()` keeps the queue's internal
    unfinished-task counter balanced, so a later `join()` (if any) cannot hang.
    """
    dropped = 0
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return dropped
        queue.task_done()
        dropped += 1
