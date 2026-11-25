"""Per-call conversation memory.

Prototype: a plain in-memory dict keyed by session_id.
Production: replace SESSIONS with Redis (one key per session_id, TTL a few
hours) so multiple API workers share state and a restart does not lose calls.

LangChain tools are plain functions without access to the WebSocket handler, so
the "currently active" session is exposed through a ContextVar. The agent sets
it before invoking tools; `log_turn_metadata` reads it to update live scoring.

Ownership: the agent (gemini_agent) owns `history` (LangChain messages), while
the WebSocket handler owns `transcript` (what was actually said, for the
summary). Turns are serialised by the caller's single worker task, so no locks
are needed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from contextvars import ContextVar
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage

Sentiment = Literal["positive", "neutral", "negative"]
Qualification = Literal["hot", "warm", "cold"]


@dataclass
class CallSession:
    """Everything we know about one live call."""

    session_id: str
    history: list[BaseMessage] = field(default_factory=list)  # LangChain message list
    transcript: list[dict] = field(default_factory=list)  # [{"speaker": "user"|"agent", "text": ...}]
    sentiment: str = "neutral"
    qualification: str = "warm"
    lead_id: int | None = None  # set by the save_lead tool

    def record_user(self, text: str) -> None:
        """Add the lead's utterance to the transcript (history is the agent's)."""
        self.transcript.append({"speaker": "user", "text": text})

    def record_agent(self, text: str) -> None:
        """Add Riya's spoken reply to the transcript (history is the agent's)."""
        self.transcript.append({"speaker": "agent", "text": text})

    def transcript_text(self) -> str:
        """Flat transcript for the summariser prompt."""
        if not self.transcript:
            return ""
        lines = []
        for turn in self.transcript:
            who = "Lead" if turn["speaker"] == "user" else "Riya"
            lines.append(f"{who}: {turn['text']}")
        return "\n".join(lines)


    def recent_history(self, max_messages: int) -> list[BaseMessage]:
        """Trim history so long calls do not blow up the prompt.

        The window is moved forward to a user message, so the model never sees a
        conversation that starts with its own reply.
        """
        window = self.history[-max_messages:]
        for index, message in enumerate(window):
            if isinstance(message, HumanMessage):
                return window[index:]
        return window


# session_id -> CallSession. In-memory only: see module docstring.
SESSIONS: dict[str, CallSession] = {}

# The session whose tools are currently executing (set by gemini_agent).
ACTIVE_SESSION: ContextVar[CallSession | None] = ContextVar("active_session", default=None)


def get_session(session_id: str) -> CallSession:
    """Fetch or create the session for this call."""
    session = SESSIONS.get(session_id)
    if session is None:
        session = CallSession(session_id=session_id)
        SESSIONS[session_id] = session
    return session


def drop_session(session_id: str) -> None:
    """Remove a finished call from memory."""
    SESSIONS.pop(session_id, None)


def coerce_label(value: str, allowed: tuple[str, ...], default: str) -> str:
    """Normalise loose LLM output ("Positive", "hot lead") to a known label."""
    text = (value or "").strip().lower()
    for label in allowed:
        if label in text:
            return label
    return default
