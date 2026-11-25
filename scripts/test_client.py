"""Simple WebSocket test client for the sales voice agent.

Examples (server running on http://127.0.0.1:8000):

    # Type instead of speaking (works with no microphone):
    python scripts/test_client.py --mode text

    # Stream a 16 kHz mono 16-bit WAV file:
    python scripts/test_client.py --mode wav --wav sample.wav

    # Live microphone (needs sounddevice + a working input device):
    python scripts/test_client.py --mode mic

Received audio is played back through sounddevice when the server's TTS format
is pcm_16000 (the default). Use --no-audio to stay silent.

Convert any recording to the expected format:
    ffmpeg -i input.m4a -ar 16000 -ac 1 -sample_fmt s16 sample.wav
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import wave
from pathlib import Path

import websockets

SAMPLE_RATE = 16000
CHUNK_BYTES = 3200  # 100 ms of 16-bit mono audio at 16 kHz
DEFAULT_URL = "ws://127.0.0.1:8000/ws/call/demo-session-1"


class AudioPlayer:
    """Plays raw 16-bit PCM chunks as they arrive."""

    def __init__(self, enabled: bool) -> None:
        self._stream = None
        if not enabled:
            return
        try:
            import sounddevice as sd

            self._stream = sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16")
            self._stream.start()
        except Exception as exc:  # no sound card, missing PortAudio, ...
            print(f"[client] audio playback disabled: {exc}", file=sys.stderr)

    def write(self, data: bytes) -> None:
        if self._stream is not None:
            self._stream.write(data)

    def close(self) -> None:
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.stop()
                self._stream.close()
            self._stream = None


async def receive_loop(ws, player: AudioPlayer, summary_received: asyncio.Event) -> None:
    """Print server events and play incoming audio."""
    try:
        async for message in ws:
            if isinstance(message, bytes):
                player.write(message)
                continue

            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                print(f"[server] {message}")
                continue

            message_type = payload.get("type")

            if message_type == "partial_transcript":
                print(f"\r  ... {payload.get('text', '')}", end="", flush=True)
            elif message_type == "final_transcript":
                print(f"\rLead: {payload.get('text', '')}")
            elif message_type == "assistant_text":
                print(f"Riya: {payload.get('text', '')}")
            elif message_type == "audio_start":
                print("[client] streaming reply audio...")
            elif message_type == "audio_end":
                print("[client] reply audio finished")
            elif message_type == "call_summary":
                print("\n=== CALL SUMMARY ===")
                print(json.dumps(payload.get("summary", payload), indent=2, ensure_ascii=False))
                print(f"lead_id: {payload.get('lead_id')}")
                summary_received.set()
            elif message_type == "ready":
                print(f"[client] connected (session={payload.get('session_id')}, "
                      f"stt={payload.get('stt_connected')}, audio={payload.get('tts_format')})")
            elif message_type in {"warning", "error"}:
                print(f"[server:{message_type}] {payload.get('message')}")
            else:
                print(f"[server] {json.dumps(payload, ensure_ascii=False)}")
    except websockets.ConnectionClosed:
        print("\n[client] connection closed")
    except Exception as exc:
        print(f"\n[client] receive loop stopped: {exc}", file=sys.stderr)


async def stream_wav(ws, path: Path) -> None:
    """Send a WAV file in real time, chunk by chunk."""
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != SAMPLE_RATE:
            raise SystemExit(
                "WAV must be mono, 16-bit, 16 kHz. Got "
                f"{wav.getnchannels()} channel(s), {wav.getsampwidth() * 8}-bit, "
                f"{wav.getframerate()} Hz.\n"
                "Convert with: ffmpeg -i input.wav -ar 16000 -ac 1 -sample_fmt s16 sample.wav"
            )

        print(f"[client] streaming {path} ...")
        while frames := wav.readframes(CHUNK_BYTES // 2):
            await ws.send(frames)
            await asyncio.sleep(len(frames) / (SAMPLE_RATE * 2))  # real-time pacing


async def run_mic(ws, stop: asyncio.Event) -> None:
    """Stream microphone audio until `stop` is set."""
    try:
        import sounddevice as sd
    except Exception as exc:
        raise SystemExit(f"sounddevice is required for mic mode: {exc}")

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes] = asyncio.Queue()

    def callback(indata, _frames, _time_info, status) -> None:
        if status:
            print(f"[client] mic status: {status}", file=sys.stderr)
        loop.call_soon_threadsafe(queue.put_nowait, bytes(indata))

    print("[client] microphone live - speak now")
    try:
        stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=1600,  # 100 ms
            callback=callback,
        )
    except Exception as exc:
        raise SystemExit(f"Could not open the microphone: {exc}")

    with stream:
        while not stop.is_set():
            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            await ws.send(chunk)


async def run_text(ws) -> None:
    """Type turns instead of speaking (useful in CI and on machines without a mic)."""
    print("[client] type a message and press Enter. Type /end to finish the call.")
    while True:
        try:
            line = await asyncio.to_thread(input, "You: ")
        except EOFError:
            return
        line = line.strip()
        if not line:
            continue
        if line.lower() in {"/end", "quit", "exit"}:
            return
        await ws.send(json.dumps({"type": "text", "text": line}))


async def wait_for_enter(stop: asyncio.Event) -> None:
    with contextlib.suppress(EOFError):
        await asyncio.to_thread(input, "\nPress Enter to end the call...\n")
    stop.set()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test client for the sales voice agent WebSocket.")
    parser.add_argument("--url", default=DEFAULT_URL, help="WebSocket URL")
    parser.add_argument("--mode", choices=["text", "wav", "mic"], default="text")
    parser.add_argument("--wav", help="Path to a 16 kHz mono 16-bit WAV file (--mode wav)")
    parser.add_argument("--no-audio", action="store_true", help="Do not play received audio")
    args = parser.parse_args()

    if args.mode == "wav" and not args.wav:
        parser.error("--mode wav requires --wav PATH")

    player = AudioPlayer(enabled=not args.no_audio)
    summary_received = asyncio.Event()

    try:
        async with websockets.connect(args.url, max_size=None) as ws:
            receiver = asyncio.create_task(receive_loop(ws, player, summary_received))
            stop = asyncio.Event()

            try:
                if args.mode == "text":
                    await run_text(ws)
                elif args.mode == "wav":
                    await stream_wav(ws, Path(args.wav))
                    await asyncio.sleep(1.5)  # let the last final transcript land
                else:
                    await asyncio.gather(run_mic(ws, stop), wait_for_enter(stop))
            except (KeyboardInterrupt, asyncio.CancelledError):
                stop.set()

            print("[client] ending call, waiting for the summary...")
            await ws.send(json.dumps({"type": "end_call"}))

            try:
                await asyncio.wait_for(summary_received.wait(), timeout=90)
            except asyncio.TimeoutError:
                print("[client] no summary received before timeout", file=sys.stderr)

            receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receiver
    finally:
        player.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[client] stopped")
