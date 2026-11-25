"""Application configuration.

All secrets/knobs come from environment variables, loaded from a local `.env`
file via python-dotenv. Nothing secret is ever hardcoded in the repo.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Project root = the folder that contains `app/`, `data/`, `requirements.txt` ...
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env once, before pydantic-settings reads os.environ.
# override=False keeps real environment variables (e.g. in CI/production) winning.
load_dotenv(PROJECT_ROOT / ".env", override=False)


class Settings(BaseSettings):
    """Typed view of the environment. Missing optional values fall back to ""."""

    # --- Deepgram (streaming STT) ---
    deepgram_api_key: str = ""
    deepgram_model: str = "nova-3"
    # 16 kHz mono linear16 PCM is what the test client streams; browsers will
    # usually send something else, so keep it configurable.
    stt_encoding: str = "linear16"
    audio_sample_rate: int = 16000

    # --- Gemini (the conversational brain) ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_temperature: float = 0.4

    # --- ElevenLabs (streaming TTS) ---
    elevenlabs_api_key: str = ""
    # "Sarah" (premade, works on free plans). Some library voices need a paid plan.
    elevenlabs_voice_id: str = "EXAVITQu4vr4xnSDxMaL"
    elevenlabs_model_id: str = "eleven_turbo_v2_5"
    # pcm_16000 = raw 16-bit little-endian PCM, easy for the test client to play.
    # Use mp3_44100_128 if your web client decodes MP3 in the browser.
    elevenlabs_output_format: str = "pcm_16000"

    # --- Mock CRM ---
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'sales_agent.db').as_posix()}"

    # --- Chroma (local persistent vector store) ---
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"
    chroma_collection: str = "sales_knowledge_base"
    knowledge_base_dir: Path = PROJECT_ROOT / "data" / "knowledge_base"

    # --- Google Calendar ---
    google_calendar_credentials_path: str = "credentials/google-service-account.json"
    google_calendar_id: str = "primary"
    google_calendar_timezone: str = "Asia/Kolkata"
    # Optional: Workspace domain user the service account should impersonate
    # (needed because a bare service account has its own empty calendar).
    google_calendar_delegated_user: str = ""

    # --- Persona / business context used in the system prompt ---
    company_name: str = "StockPilot"
    company_product: str = "cloud-based inventory software for small retailers"

    # --- Prototype limits ---
    max_history_messages: int = 24  # how much conversation memory to keep per call
    max_tool_rounds: int = 4  # Gemini tool-call loops per user turn
    rag_top_k: int = 3

    @property
    def credentials_path(self) -> Path:
        """Calendar credentials path resolved against the project root."""
        path = Path(self.google_calendar_credentials_path)
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache
def get_settings() -> Settings:
    """Settings are constructed once and reused (cheap + consistent)."""
    return Settings()
