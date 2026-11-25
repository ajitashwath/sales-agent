"""End-of-call summarisation ("business insights").

Takes the full transcript, asks Gemini (through LangChain structured output)
for a small JSON object and returns it as a plain dict:

    {"lead_intent": str, "objections": [str], "outcome": str, "next_action": str}

Never raises: if Gemini is unavailable the caller still gets a usable dict with
an "error" field, so the WebSocket can close cleanly.
"""

from __future__ import annotations

import asyncio
import logging

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.agent.gemini_agent import AgentUnavailable, get_llm
from app.agent.memory import CallSession
from app.agent.prompts import render_summary_prompt

logger = logging.getLogger(__name__)

# The client is already gone by the time we summarise, so never let a stuck
# request keep the call alive.
SUMMARY_TIMEOUT_SECONDS = 45


class CallSummary(BaseModel):
    """Schema Gemini must fill in (used by with_structured_output)."""

    lead_intent: str = Field(description="What the lead actually wants, in one short sentence.")
    objections: list[str] = Field(
        default_factory=list,
        description="Short list of objections or concerns raised. Empty list if none.",
    )
    outcome: str = Field(description="What happened by the end of the call.")
    next_action: str = Field(description="The single most useful next step for the sales rep.")


def transcript_to_text(session: CallSession) -> str:
    """Flat 'Lead: ... / Riya: ...' transcript (also handy for debugging)."""
    return session.transcript_text()


async def summarize_call(session: CallSession) -> dict:
    """Summarise a call. Returns a dict matching CallSummary (plus optional error)."""
    transcript = session.transcript_text()

    if not transcript.strip():
        return {
            "lead_intent": "No conversation took place.",
            "objections": [],
            "outcome": "Empty call",
            "next_action": "No follow-up needed.",
        }

    try:
        llm = get_llm()  # raises AgentUnavailable when GEMINI_API_KEY is missing
        # with_structured_output asks Gemini for a response_schema matching CallSummary.
        structured_llm = llm.with_structured_output(CallSummary)
        async with asyncio.timeout(SUMMARY_TIMEOUT_SECONDS):
            result: CallSummary = await structured_llm.ainvoke(
                [HumanMessage(content=render_summary_prompt(transcript))]
            )
        return result.model_dump()

    except AgentUnavailable:
        logger.warning("Skipping Gemini summary: GEMINI_API_KEY is not configured")
        return _fallback_summary(session, "GEMINI_API_KEY is not configured")
    except TimeoutError:
        logger.warning("Gemini summary timed out after %ss", SUMMARY_TIMEOUT_SECONDS)
        return _fallback_summary(session, "summarisation timed out")
    except Exception as exc:
        logger.exception("Call summarisation failed")
        return _fallback_summary(session, str(exc))


def _fallback_summary(session: CallSession, error: str) -> dict:
    """Deterministic placeholder so the call record is never empty."""
    turns = len(session.transcript)
    return {
        "lead_intent": "Summary unavailable - review the transcript.",
        "objections": [],
        "outcome": f"Call ended after {turns} transcript turn(s).",
        "next_action": "Review the transcript manually and follow up.",
        "error": error,
    }
