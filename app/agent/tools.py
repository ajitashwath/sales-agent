"""LangChain tools that Gemini can call during a sales call.

The docstrings are the tool descriptions the model reads, so they are written
for an LLM: what the tool does, when to use it, and what it returns.

Design notes:
- Tools are synchronous; the agent executes them with asyncio.to_thread so
  blocking SDKs (Chroma, Google API, SQLite) never stall the event loop.
- Tools return plain strings, because the result is fed back to the model and
  may end up being spoken.
- Tools never raise: an external failure becomes a readable message, which the
  model can turn into a graceful response.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from app.agent.memory import ACTIVE_SESSION, coerce_label
from app.calendar_integration import google_calendar
from app.crm import crud
from app.rag.retriever import search_knowledge

logger = logging.getLogger(__name__)


@tool
def knowledge_search(query: str) -> str:
    """Search the company knowledge base for product, pricing, feature,
    integration or policy information.

    Use this whenever the lead asks about what the product does, how much it
    costs, what it integrates with, or company policies. Never invent product
    details: if the knowledge base does not contain the answer, say so and offer
    a human follow-up.

    Args:
        query: A short natural-language search query, e.g. "pricing for 5 stores".

    Returns:
        The top matching knowledge base chunks with their source file names, or
        a message saying nothing was found.
    """
    return search_knowledge(query)


@tool
def save_lead(
    name: str,
    contact: str = "",
    need: str = "",
    budget: str = "",
    timeline: str = "",
    qualification: str = "warm",
    sentiment: str = "neutral",
) -> str:
    """Create or update the lead record in the CRM.

    Call this as soon as you learn the lead's name, and call it again as you
    learn more (contact details, need, budget, timeline). It updates the same
    lead rather than creating duplicates, so pass whatever you know and leave
    the rest empty.

    Args:
        name: The lead's name.
        contact: Phone number or email, if known.
        need: The problem they are trying to solve, in their own words.
        budget: Their stated budget or budget range.
        timeline: When they want to start or buy.
        qualification: One of "hot", "warm" or "cold".
        sentiment: One of "positive", "neutral" or "negative".

    Returns:
        A short confirmation containing the CRM record id.
    """
    try:
        lead = crud.create_or_update_lead(
            name=name,
            contact=contact,
            need=need,
            budget=budget,
            timeline=timeline,
            qualification=coerce_label(qualification, ("hot", "warm", "cold"), "warm"),
            sentiment=coerce_label(sentiment, ("positive", "neutral", "negative"), "neutral"),
        )
    except Exception:
        logger.exception("save_lead failed")
        return "Sorry, I couldn't save those details right now."

    session = ACTIVE_SESSION.get()
    if session is not None:
        session.lead_id = lead.id
        session.sentiment = lead.sentiment
        session.qualification = lead.qualification

    return f"Saved lead #{lead.id} ({lead.name}) to the CRM."


@tool
def check_calendar_availability(date_range: str) -> str:
    """Check the sales calendar for open 30-minute demo slots.

    Use this before offering times. Accepts natural ranges such as "tomorrow",
    "next week", "2026-12-01" or "2026-12-01 to 2026-12-05".

    Args:
        date_range: The range to check. Prefer ISO dates ("YYYY-MM-DD") or
            "YYYY-MM-DD to YYYY-MM-DD" so the range is unambiguous.

    Returns:
        A readable list of open slots per day. If no calendar credentials are
        configured, clearly-marked mock slots are returned for the demo.
    """
    try:
        return google_calendar.check_availability(date_range)
    except Exception:
        logger.exception("check_calendar_availability failed")
        return "Sorry, I'm having trouble checking the calendar right now."


@tool
def book_appointment(name: str, contact: str, datetime_str: str) -> str:
    """Book a confirmed 30-minute demo on the sales calendar.

    Only call this after the lead has explicitly agreed to a specific date and
    time. If the tool reports a conflict, apologise and offer another slot.

    Args:
        name: The lead's name.
        contact: Phone number or email for the invite/description.
        datetime_str: The agreed start time, ideally ISO like
            "2026-12-01 15:00".

    Returns:
        A confirmation message, a conflict message, or a graceful error.
    """
    try:
        return google_calendar.book_slot(name=name, contact=contact, datetime_str=datetime_str)
    except Exception:
        logger.exception("book_appointment failed")
        return "Sorry, I'm having trouble booking that right now."


@tool
def log_turn_metadata(sentiment: str, qualification: str) -> str:
    """Record your silent scoring of the current user turn.

    Call this once after each user turn. These scores are NOT spoken; they feed
    the business insights dashboard and your own tone/next-step decisions.

    Args:
        sentiment: "positive", "neutral" or "negative".
        qualification: "hot", "warm" or "cold".

    Returns:
        The string "logged".
    """
    session = ACTIVE_SESSION.get()
    if session is None:
        # Tool called outside a live call (e.g. a unit test): nothing to update.
        logger.debug("log_turn_metadata called with no active session")
        return "logged"

    session.sentiment = coerce_label(sentiment, ("positive", "neutral", "negative"), session.sentiment)
    session.qualification = coerce_label(qualification, ("hot", "warm", "cold"), session.qualification)
    logger.debug(
        "session=%s scoring sentiment=%s qualification=%s",
        session.session_id,
        session.sentiment,
        session.qualification,
    )
    return "logged"


# The list Gemini is bound to, and a name -> tool lookup for execution.
TOOLS = [knowledge_search, save_lead, check_calendar_availability, book_appointment, log_turn_metadata]
TOOLS_BY_NAME = {tool_.name: tool_ for tool_ in TOOLS}
