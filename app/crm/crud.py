"""CRUD helpers for the mock CRM.

Kept deliberately small: upsert a lead, read one back, list them.
Every function owns its session so callers (LangChain tools, FastAPI) do not
have to manage transactions.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from sqlalchemy import select

from app.crm.models import Lead, SessionLocal

logger = logging.getLogger(__name__)


def create_or_update_lead(
    name: str,
    contact: str = "",
    need: str = "",
    budget: str = "",
    timeline: str = "",
    qualification: str = "warm",
    sentiment: str = "neutral",
) -> Lead:
    """Insert a new lead or update an existing one.

    Matching order: same `contact` (phone/email) first, then same `name`
    (most recently updated). Only non-empty values overwrite existing data, so
    the agent can call this repeatedly as it learns more during the call.
    """
    with SessionLocal() as db:
        lead: Lead | None = None

        if contact:
            lead = db.scalar(select(Lead).where(Lead.contact == contact).order_by(Lead.updated_at.desc()))
        if lead is None and name:
            lead = db.scalar(
                select(Lead).where(Lead.name.ilike(name)).order_by(Lead.updated_at.desc())
            )

        if lead is None:
            lead = Lead(name=name or "Unknown caller")
            db.add(lead)

        # Never clobber known facts with empty strings or "unknown" placeholders.
        updates = {
            "name": name,
            "contact": contact,
            "need": need,
            "budget": budget,
            "timeline": timeline,
            "qualification": qualification,
            "sentiment": sentiment,
        }
        for field, value in updates.items():
            value = (value or "").strip()
            if value and value.lower() not in {"unknown", "n/a", "not provided", "none"}:
                setattr(lead, field, value)

        lead.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(lead)

        logger.info("Lead upserted: id=%s name=%s contact=%s", lead.id, lead.name, lead.contact)
        return lead


def get_lead(lead_id: int | None = None, contact: str | None = None, name: str | None = None) -> Lead | None:
    """Look a lead up by id, contact, or (most recent) name."""
    with SessionLocal() as db:
        if lead_id is not None:
            return db.get(Lead, lead_id)
        if contact:
            return db.scalar(select(Lead).where(Lead.contact == contact).order_by(Lead.updated_at.desc()))
        if name:
            return db.scalar(select(Lead).where(Lead.name.ilike(name)).order_by(Lead.updated_at.desc()))
        return None


def list_leads(limit: int = 50) -> list[Lead]:
    """Most recently updated leads first."""
    with SessionLocal() as db:
        return list(db.scalars(select(Lead).order_by(Lead.updated_at.desc()).limit(limit)))


def save_call_summary(lead_id: int, summary: dict) -> Lead | None:
    """Attach the end-of-call summary to a lead record.

    `summary` is the structured dict produced by summary/call_summary.py; we store
    a readable one-liner plus the raw JSON so a dashboard can parse it later.
    """
    with SessionLocal() as db:
        lead = db.get(Lead, lead_id)
        if lead is None:
            logger.warning("Cannot attach summary: lead %s not found", lead_id)
            return None

        lead.summary_json = json.dumps(summary, ensure_ascii=False)
        lead.summary = (
            f"Intent: {summary.get('lead_intent', 'n/a')} | "
            f"Objections: {', '.join(summary.get('objections', []) or []) or 'none'} | "
            f"Outcome: {summary.get('outcome', 'n/a')} | "
            f"Next action: {summary.get('next_action', 'n/a')}"
        )
        lead.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(lead)
        return lead
