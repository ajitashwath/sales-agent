"""SQLAlchemy models for the mock CRM.

One table is enough for the prototype: `leads`. The engine/session factory live
here too so the rest of the app has a single import for DB access.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import get_settings

settings = get_settings()

# check_same_thread=False: SQLite is touched from the FastAPI event loop and
# from worker threads (LangChain tools run in a thread pool).
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def utcnow() -> datetime:
    """Timezone-aware UTC now (avoids the deprecated datetime.utcnow)."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Lead(Base):
    """A sales lead captured during (or after) a call."""

    __tablename__ = "leads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), default="", index=True)
    contact: Mapped[str] = mapped_column(String(200), default="", index=True)  # phone/email
    need: Mapped[str] = mapped_column(Text, default="")
    budget: Mapped[str] = mapped_column(String(120), default="")
    timeline: Mapped[str] = mapped_column(String(120), default="")
    qualification: Mapped[str] = mapped_column(String(20), default="warm")  # hot/warm/cold
    sentiment: Mapped[str] = mapped_column(String(20), default="neutral")  # positive/neutral/negative

    # Filled in when the call ends (see summary/call_summary.py).
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    def to_dict(self) -> dict:
        """Plain dict for JSON responses / tool output."""
        return {
            "id": self.id,
            "name": self.name,
            "contact": self.contact,
            "need": self.need,
            "budget": self.budget,
            "timeline": self.timeline,
            "qualification": self.qualification,
            "sentiment": self.sentiment,
            "summary": self.summary,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


def init_db() -> None:
    """Create tables if they do not exist yet. Called on FastAPI startup."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """Convenience for scripts: `with get_db() as db:`."""
    return SessionLocal()
