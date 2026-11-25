"""Gemini (via LangChain) sales agent with tool calling.

One user turn looks like:
    history + system prompt -> Gemini -> (tool calls -> tool results -> Gemini)*
    -> final plain-text reply (which the caller sends to TTS).

Conversation memory lives in app/agent/memory.py (in-memory dict for the
prototype; Redis in production).
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import ValidationError

from app.agent.memory import ACTIVE_SESSION, CallSession
from app.agent.prompts import build_system_prompt
from app.agent.tools import TOOLS, TOOLS_BY_NAME
from app.config import get_settings

logger = logging.getLogger(__name__)

_llm: ChatGoogleGenerativeAI | None = None


class AgentUnavailable(RuntimeError):
    """Raised when the conversational brain cannot be used at all."""


def get_llm() -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model once per process."""
    global _llm

    settings = get_settings()
    if not settings.gemini_api_key:
        raise AgentUnavailable("GEMINI_API_KEY is not configured")

    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=settings.gemini_temperature,
            max_retries=2,
            timeout=60,
        )
    return _llm


def _message_text(message: AIMessage) -> str:
    """Flatten LangChain's content (string or content blocks) into plain text."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        return " ".join(part for part in parts if part)
    return str(content or "")


def _clean_for_voice(text: str) -> str:
    """Strip markdown that sounds wrong when spoken by TTS."""
    text = text or ""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)  # code blocks
    text = re.sub(r"[*_#>`]+", "", text)  # emphasis / headings / quotes
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)  # bullet markers
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # markdown links
    text = re.sub(r"\s+", " ", text).strip()
    return text


async def _execute_tool(tool_call: dict) -> str:
    """Run one tool call, converting any failure into a speakable message."""
    name = tool_call.get("name") or ""
    args = tool_call.get("args") or {}

    tool_ = TOOLS_BY_NAME.get(name)
    if tool_ is None:
        logger.warning("Model requested unknown tool %r", name)
        return f"Unknown tool: {name}."

    try:
        # to_thread keeps blocking SDK calls (Chroma, Google API, SQLite) off the loop.
        result = await asyncio.to_thread(tool_.invoke, args)
        return str(result)
    except ValidationError as exc:
        # The model produced arguments that do not match the schema. Tell it so
        # it can correct itself on the next round instead of dying here.
        logger.warning("Bad arguments for tool %s: %s | args=%r", name, exc, args)
        return (
            f"Invalid arguments for {name}: {exc.error_count()} field(s) failed validation. "
            "Call it again with exactly the documented parameters, or carry on without it."
        )
    except Exception:
        logger.exception("Tool %s failed", name)
        return "Sorry, I'm having trouble doing that right now."


async def run_agent_turn(session: CallSession, user_text: str) -> str:
    """Run one conversational turn and return the reply text (voice-ready).

    Mutates `session.history` only; the caller owns the transcript.
    Raises AgentUnavailable if Gemini is not configured.
    """
    settings = get_settings()
    llm = get_llm()  # raises AgentUnavailable when the key is missing
    system_prompt = build_system_prompt(
        company_name=settings.company_name,
        company_product=settings.company_product,
        timezone=settings.google_calendar_timezone,
        today=datetime.now().date(),
    )

    # Record the user's turn before anything can fail, so the transcript and
    # summary remain useful even if the model errors out.
    session.history.append(HumanMessage(content=user_text))

    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        *session.recent_history(settings.max_history_messages),
    ]

    token = ACTIVE_SESSION.set(session)  # lets tools update live session state
    try:
        llm_with_tools = llm.bind_tools(TOOLS)

        for round_number in range(settings.max_tool_rounds):
            ai_message: AIMessage = await llm_with_tools.ainvoke(messages)
            messages.append(ai_message)

            if not ai_message.tool_calls:
                reply = _clean_for_voice(_message_text(ai_message))
                if not reply:
                    reply = "Sorry, could you say that again?"
                session.history.append(AIMessage(content=reply))
                return reply

            logger.info(
                "session=%s round=%s tool_calls=%s",
                session.session_id,
                round_number + 1,
                [call.get("name") for call in ai_message.tool_calls],
            )

            for call in ai_message.tool_calls:
                result = await _execute_tool(call)
                messages.append(ToolMessage(content=result, tool_call_id=call.get("id") or ""))

        # Safety valve: the model kept calling tools. Ask for a plain answer.
        logger.warning("session=%s hit max_tool_rounds=%s", session.session_id, settings.max_tool_rounds)
        final = await llm.ainvoke(
            [
                *messages,
                HumanMessage(
                    content="Stop using tools and reply to the lead now in one or two short spoken sentences."
                ),
            ]
        )
        reply = _clean_for_voice(_message_text(final)) or "Sorry, could you repeat that?"
        session.history.append(AIMessage(content=reply))
        return reply

    finally:
        ACTIVE_SESSION.reset(token)
