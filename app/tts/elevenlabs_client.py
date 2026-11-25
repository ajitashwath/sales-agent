"""Streaming text-to-speech via the ElevenLabs HTTP API.

We call the streaming endpoint with httpx and yield raw audio bytes as they
arrive, so the caller can forward them over the WebSocket immediately instead of
waiting for the whole clip.

Default output is `pcm_16000` (16-bit little-endian mono, 16 kHz) because the
test client plays it directly with sounddevice without an MP3 decoder. Set
ELEVENLABS_OUTPUT_FORMAT=mp3_44100_128 for browser clients.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"


class TTSError(RuntimeError):
    """Raised when ElevenLabs cannot synthesise audio."""


class ElevenLabsTTS:
    """Thin async wrapper around /text-to-speech/{voice_id}/stream."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = httpx.AsyncClient(
            base_url=ELEVENLABS_BASE_URL,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    async def stream_speech(self, text: str) -> AsyncIterator[bytes]:
        """Yield audio chunks for `text`.

        Raises TTSError on configuration/HTTP problems; the WebSocket handler
        catches it so the call continues (the transcript is already delivered).
        """
        text = (text or "").strip()
        if not text:
            return

        if not self._settings.elevenlabs_api_key:
            raise TTSError("ELEVENLABS_API_KEY is not set")

        url = f"/text-to-speech/{self._settings.elevenlabs_voice_id}/stream"
        params = {"output_format": self._settings.elevenlabs_output_format}
        payload = {
            "text": text,
            "model_id": self._settings.elevenlabs_model_id,
            "voice_settings": {
                "stability": 0.4,
                "similarity_boost": 0.8,
                "style": 0.0,
                "use_speaker_boost": True,
            },
        }

        try:
            async with self._client.stream(
                "POST",
                url,
                params=params,
                headers={"xi-api-key": self._settings.elevenlabs_api_key},
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise TTSError(f"ElevenLabs returned {response.status_code}: {body[:300]}")

                async for chunk in response.aiter_bytes(chunk_size=4096):
                    if chunk:
                        yield chunk

        except httpx.HTTPError as exc:
            logger.exception("ElevenLabs request failed")
            raise TTSError(f"ElevenLabs request failed: {exc}") from exc

    async def close(self) -> None:
        await self._client.aclose()
