"""Streamlit front end for the AI sales voice agent ("Riya").

Start the backend first, then this UI:

    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
    streamlit run streamlit_app.py

How it works: the browser only talks to Streamlit. Streamlit keeps one WebSocket
to the FastAPI app on a background thread, because Streamlit re-runs this script
top to bottom on every interaction and a live connection cannot live in the
script's own call stack.

Typed turns go up as {"type": "text"}; reply audio comes back as binary frames
between audio_start and audio_end, which we wrap in a WAV header so the browser
can play the 16 kHz PCM directly.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import queue
import threading
import time
import uuid
import wave

import httpx
import streamlit as st
import websockets

DEFAULT_HTTP_URL = os.getenv("AGENT_HTTP_URL", "http://127.0.0.1:8000")
TURN_TIMEOUT_SECONDS = 150
SUMMARY_TIMEOUT_SECONDS = 120


# ---------------------------------------------------------------------------
# WebSocket client (background thread, so it survives Streamlit reruns)
# ---------------------------------------------------------------------------
class CallClient:
    """One WebSocket to /ws/call/{session_id}, with a thread-safe event queue."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.events: queue.Queue[dict] = queue.Queue()
        self._ws = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._connected = threading.Event()
        self._error: str | None = None

    # -- lifecycle ---------------------------------------------------------
    def connect(self, timeout: float = 10.0) -> None:
        """Open the connection; raises RuntimeError with a readable message."""
        asyncio.run_coroutine_threadsafe(self._open(), self._loop)
        if not self._connected.wait(timeout):
            self._error = "Timed out connecting to the agent."
        if self._error:
            raise RuntimeError(self._error)

    async def _open(self) -> None:
        try:
            self._ws = await websockets.connect(self.url, max_size=None)
        except Exception as exc:  # server down, wrong URL, ...
            self._error = f"Could not connect to {self.url}: {exc}"
            self._connected.set()
            return
        self._connected.set()
        asyncio.create_task(self._pump())

    async def _pump(self) -> None:
        """Forward every server frame into the thread-safe queue."""
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    self.events.put({"type": "audio", "bytes": message})
                else:
                    self.events.put(json.loads(message))
        except Exception as exc:
            self.events.put({"type": "closed", "message": f"Connection lost: {exc}"})
        else:
            self.events.put({"type": "closed", "message": "Connection closed"})

    # -- sending -----------------------------------------------------------
    def send_json(self, payload: dict) -> None:
        self._send(json.dumps(payload, ensure_ascii=False))

    def send_bytes(self, data: bytes) -> None:
        self._send(data)

    def _send(self, data) -> None:
        if self._ws is None:
            raise RuntimeError("The call is not connected.")
        asyncio.run_coroutine_threadsafe(self._ws.send(data), self._loop)

    # -- receiving ---------------------------------------------------------
    def next_event(self, timeout: float = 1.0) -> dict | None:
        """Next server event, or None if nothing arrived within `timeout`."""
        try:
            return self.events.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        if self._ws is not None:
            with contextlib.suppress(Exception):
                asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop).result(timeout=5)
            self._ws = None
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2)
        if not self._loop.is_running():
            with contextlib.suppress(Exception):
                self._loop.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def http_to_ws_url(base_url: str, session_id: str) -> str:
    base = base_url.strip().rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://") :]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://") :]
    return f"{base}/ws/call/{session_id}"


def pcm_to_wav(pcm: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV header so <audio> can play it."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def audio_format(tts_format: str) -> tuple[str, int]:
    """Map the server's TTS format to (mime type, sample rate)."""
    if tts_format.startswith("pcm_"):
        with contextlib.suppress(ValueError):
            return "audio/wav", int(tts_format.split("_", 1)[1])
        return "audio/wav", 16000
    return "audio/mpeg", 0


def play_audio(data: bytes, mime: str) -> None:
    """Play a clip, autoplaying when the installed Streamlit supports it."""
    if not data:
        return
    try:
        st.audio(data, format=mime, autoplay=True)
    except TypeError:  # older Streamlit without autoplay
        st.audio(data, format=mime)


def backend_health(base_url: str) -> dict | None:
    try:
        return httpx.get(f"{base_url.rstrip('/')}/health", timeout=3).json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Call actions
# ---------------------------------------------------------------------------
def start_call(base_url: str) -> None:
    session_id = f"st-{uuid.uuid4().hex[:8]}"
    client = CallClient(http_to_ws_url(base_url, session_id))
    try:
        client.connect()
    except RuntimeError as exc:
        st.error(str(exc))
        return

    ready: dict | None = None
    deadline = time.time() + 30
    while time.time() < deadline:
        event = client.next_event(timeout=1)
        if event is None:
            continue
        if event.get("type") == "ready":
            ready = event
            break
        if event.get("type") == "closed":
            st.error(event.get("message", "Connection closed"))
            client.close()
            return

    st.session_state.client = client
    st.session_state.session_id = session_id
    st.session_state.messages = []
    st.session_state.summary = None
    st.session_state.lead_id = None
    st.session_state.tts_format = (ready or {}).get("tts_format") or "pcm_16000"
    st.session_state.sample_rate = int((ready or {}).get("sample_rate") or 16000)
    st.session_state.stt_connected = bool((ready or {}).get("stt_connected"))


def run_turn(client: CallClient, prompt: str) -> dict:
    """Send one typed turn and collect the reply text plus its audio."""
    client.send_json({"type": "text", "text": prompt})

    text = ""
    audio = bytearray()
    interrupted = False
    warnings: list[str] = []

    box = st.status("Riya is thinking…", expanded=False)
    deadline = time.time() + TURN_TIMEOUT_SECONDS
    while time.time() < deadline:
        event = client.next_event(timeout=1)
        if event is None:
            continue

        kind = event.get("type")
        if kind == "thinking":
            box.update(label="Riya is thinking…")
        elif kind == "partial_transcript":
            box.update(label=f"hearing: {event.get('text', '')[:70]}")
        elif kind == "final_transcript":
            # Just our own typed words echoed back by the server.
            continue
        elif kind == "assistant_text":
            text = event.get("text", "")
            box.update(label=text[:90] or "Riya is replying…")
        elif kind == "audio":
            audio.extend(event["bytes"])
        elif kind == "audio_end":
            interrupted = bool(event.get("interrupted"))
            box.update(label="Reply ready", state="complete")
            return {"text": text, "audio": bytes(audio), "interrupted": interrupted, "warnings": warnings}
        elif kind in {"warning", "error"}:
            warnings.append(str(event.get("message", "")))
        elif kind == "closed":
            box.update(label="Connection closed", state="error")
            st.session_state.client = None
            return {"text": text or "(connection closed)", "audio": bytes(audio), "interrupted": interrupted, "warnings": warnings}

    box.update(label="No reply before the timeout", state="error")
    return {"text": text or "(no reply)", "audio": bytes(audio), "interrupted": interrupted, "warnings": warnings}


def end_call() -> None:
    client: CallClient | None = st.session_state.client
    if client is None:
        return

    with st.spinner("Ending the call and summarising…"):
        with contextlib.suppress(RuntimeError):
            client.send_json({"type": "end_call"})

        payload: dict | None = None
        deadline = time.time() + SUMMARY_TIMEOUT_SECONDS
        while time.time() < deadline:
            event = client.next_event(timeout=1)
            if event is None:
                continue
            if event.get("type") == "call_summary":
                payload = event
                break
            if event.get("type") == "closed":
                break

    st.session_state.summary = (payload or {}).get("summary")
    st.session_state.lead_id = (payload or {}).get("lead_id")
    client.close()
    st.session_state.client = None
    st.rerun()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_summary() -> None:
    summary = st.session_state.summary
    if not summary:
        return

    with st.container(border=True):
        st.subheader("Call summary")
        st.markdown(f"**Intent** — {summary.get('lead_intent', 'n/a')}")

        objections = summary.get("objections") or []
        st.markdown(f"**Objections** — {', '.join(objections) if objections else 'none'}")

        left, right = st.columns(2)
        left.markdown(f"**Outcome** — {summary.get('outcome', 'n/a')}")
        right.markdown(f"**Next action** — {summary.get('next_action', 'n/a')}")

        lead_id = st.session_state.lead_id
        st.caption(f"Lead: {f'#{lead_id}' if lead_id else 'not captured in the CRM'}")
        if summary.get("error"):
            st.warning(f"Summary fell back to the deterministic version: {summary['error']}")


def render_history() -> None:
    tts_format = st.session_state.get("tts_format", "pcm_16000")
    mime, rate = audio_format(tts_format)

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🎧"):
            st.markdown(message["text"])
            if message.get("interrupted"):
                st.caption("Cut off — you started talking over this reply.")
            for warning in message.get("warnings", []):
                st.warning(warning)
            audio = message.get("audio") or b""
            if audio:
                data = pcm_to_wav(audio, rate) if mime == "audio/wav" else audio
                play_audio(data, mime)


def main() -> None:
    st.set_page_config(page_title="Riya · sales voice agent", page_icon="🎧", layout="centered")

    st.session_state.setdefault("client", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("summary", None)
    st.session_state.setdefault("lead_id", None)
    st.session_state.setdefault("session_id", None)
    st.session_state.setdefault("tts_format", "pcm_16000")
    st.session_state.setdefault("sample_rate", 16000)
    st.session_state.setdefault("stt_connected", False)

    client: CallClient | None = st.session_state.client

    # -- sidebar ---------------------------------------------------------
    with st.sidebar:
        st.title("StockPilot")
        st.caption("Riya · autonomous sales agent")

        base_url = st.text_input("Backend URL", DEFAULT_HTTP_URL)
        health = backend_health(base_url)

        if health:
            st.success("Backend reachable")
            st.caption(
                " · ".join(
                    [
                        f"STT {'on' if health.get('deepgram_configured') else 'off'}",
                        f"Gemini {'on' if health.get('gemini_configured') else 'off'}",
                        f"TTS {'on' if health.get('elevenlabs_configured') else 'off'}",
                    ]
                )
            )
        else:
            st.error("Backend not reachable")
            st.code(
                "python -m uvicorn app.main:app --host 127.0.0.1 --port 8000",
                language="powershell",
            )

        st.divider()

        if client is None:
            if st.button("Start call", type="primary", use_container_width=True, disabled=health is None):
                start_call(base_url)
                st.rerun()
        else:
            st.markdown(f"**Session** `{st.session_state.session_id}`")
            st.caption(
                f"STT {'connected' if st.session_state.stt_connected else 'unavailable'} · "
                f"audio {st.session_state.tts_format}"
            )
            if st.button("End call & summarise", use_container_width=True):
                end_call()

        st.divider()
        st.caption("Typed turns keep the demo usable without a microphone; every reply is streamed back as audio.")

    # -- main ------------------------------------------------------------
    st.title("Talk to Riya")
    st.caption(
        "Sales demo for StockPilot. Ask about pricing, features or integrations, "
        "or book a demo — the agent calls tools and the CRM as it goes."
    )

    if client is None and not st.session_state.messages:
        st.info("Start a call from the sidebar to begin.")
        render_summary()
        return

    render_history()
    render_summary()

    prompt = st.chat_input("Type your message…", disabled=client is None)
    if prompt and client is not None:
        st.session_state.messages.append({"role": "user", "text": prompt})
        result = run_turn(client, prompt)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": result["text"],
                "audio": result["audio"],
                "interrupted": result["interrupted"],
                "warnings": result["warnings"],
            }
        )
        st.rerun()


if __name__ == "__main__":
    main()
