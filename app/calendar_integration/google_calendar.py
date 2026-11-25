"""Google Calendar integration (service account) for availability + booking.

Prototype choices, documented in the README:
- Auth uses a service-account JSON file. A service account has its own calendar,
  so for a real calendar either share that calendar with the service-account
  email (view + edit), or set GOOGLE_CALENDAR_DELEGATED_USER for a Workspace
  domain-wide delegation setup.
- If credentials are missing or the API fails, we degrade to mock data so the
  live demo never breaks: check_availability returns synthetic slots and
  book_slot confirms a mock booking.

All Google API calls are synchronous (google-api-python-client). FastAPI callers
should run them via asyncio.to_thread, which the agent tools do.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from app.config import get_settings

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/calendar"]

# Cached service + whether we already tried to build it (avoid retrying a bad
# credentials file on every single tool call).
_service = None
_service_attempted = False

BUSINESS_START = time(9, 0)
BUSINESS_END = time(17, 0)
SLOT_MINUTES = 30
MAX_DAYS = 5
SLOTS_PER_DAY = 3


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def _get_service():
    """Build (once) and return an authorized Calendar API client, or None."""
    global _service, _service_attempted

    if _service_attempted:
        return _service
    _service_attempted = True

    settings = get_settings()
    credentials_path = settings.credentials_path

    if not credentials_path.exists():
        logger.warning(
            "Google Calendar credentials not found at %s - using mock availability/booking.",
            credentials_path,
        )
        return None

    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        credentials = service_account.Credentials.from_service_account_file(
            str(credentials_path), scopes=SCOPES
        )
        if settings.google_calendar_delegated_user:
            # Only valid with Workspace domain-wide delegation.
            credentials = credentials.with_subject(settings.google_calendar_delegated_user)

        _service = build("calendar", "v3", credentials=credentials, cache_discovery=False)
        logger.info("Google Calendar service initialised (calendar_id=%s)", settings.google_calendar_id)
    except Exception:
        logger.exception("Failed to initialise Google Calendar service - falling back to mocks")
        _service = None

    return _service


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tz() -> ZoneInfo:
    return ZoneInfo(get_settings().google_calendar_timezone)


def _parse_date_range(date_range: str) -> tuple[date, date]:
    """Accept 'today', 'tomorrow', 'this week', 'next week', 'YYYY-MM-DD' or
    'YYYY-MM-DD to YYYY-MM-DD'. Defaults to the next 3 days when unsure."""
    today = datetime.now(_tz()).date()
    text = (date_range or "").strip().lower()

    if not text or text in {"today", "now"}:
        return today, today
    if text == "tomorrow":
        start = today + timedelta(days=1)
        return start, start
    if "next week" in text:
        start = today + timedelta(days=(7 - today.weekday()))
        return start, start + timedelta(days=4)
    if "this week" in text:
        end = today + timedelta(days=(4 - today.weekday()))  # through Friday
        return today, max(today, end)
    if "next" in text and "day" in text:
        match = re.search(r"\d+", text)
        days = int(match.group()) if match else 3
        return today, today + timedelta(days=max(days, 1) - 1)

    # Explicit ISO ranges: "2026-12-01 to 2026-12-05"
    found = re.findall(r"\d{4}-\d{2}-\d{2}", text)
    if len(found) >= 2:
        return date.fromisoformat(found[0]), date.fromisoformat(found[1])
    if len(found) == 1:
        parsed = date.fromisoformat(found[0])
        return parsed, parsed

    logger.info("Could not parse date_range=%r - defaulting to the next 3 days", date_range)
    return today, today + timedelta(days=2)


def _parse_datetime(datetime_str: str) -> datetime | None:
    """Parse common formats an LLM might produce. Returns tz-aware datetime."""
    tz = _tz()
    text = (datetime_str or "").strip().replace("Z", "+00:00")
    if not text:
        return None

    # ISO first ("2026-12-01T15:00:00", "2026-12-01 15:00").
    try:
        parsed = datetime.fromisoformat(text)
        return parsed.replace(tzinfo=tz) if parsed.tzinfo is None else parsed.astimezone(tz)
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d %I:%M %p", "%d %b %Y %H:%M", "%d %B %Y %H:%M", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=tz)
        except ValueError:
            continue

    return None


def _format_slot(dt: datetime) -> str:
    return dt.strftime("%a %d %b, %I:%M %p")


def _mock_availability(start_day: date, end_day: date) -> str:
    """Synthetic slots so the demo works without Google credentials."""
    tz = _tz()
    lines = []
    day = start_day
    while day <= end_day and len(lines) < MAX_DAYS:
        slots = []
        for hour, minute in ((10, 0), (14, 0), (16, 0)):
            slot = datetime.combine(day, time(hour, minute), tzinfo=tz)
            slots.append(slot.strftime("%I:%M %p"))
        lines.append(f"{day.strftime('%a %d %b')}: " + ", ".join(slots))
        day += timedelta(days=1)

    return "(mock availability - Google Calendar is not configured)\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def check_availability(date_range: str) -> str:
    """Return readable open demo slots for the requested range (max ~5 days)."""
    start_day, end_day = _parse_date_range(date_range)
    end_day = min(end_day, start_day + timedelta(days=MAX_DAYS - 1))
    tz = _tz()
    service = _get_service()
    settings = get_settings()

    if service is None:
        return _mock_availability(start_day, end_day)

    try:
        time_min = datetime.combine(start_day, BUSINESS_START, tzinfo=tz)
        time_max = datetime.combine(end_day + timedelta(days=1), time(0, 0), tzinfo=tz)

        freebusy = (
            service.freebusy()
            .query(
                body={
                    "timeMin": time_min.isoformat(),
                    "timeMax": time_max.isoformat(),
                    "timeZone": settings.google_calendar_timezone,
                    "items": [{"id": settings.google_calendar_id}],
                }
            )
            .execute()
        )

        calendars = freebusy.get("calendars", {})
        busy_raw = calendars.get(settings.google_calendar_id, {}).get("busy", [])
        busy = [
            (datetime.fromisoformat(entry["start"]), datetime.fromisoformat(entry["end"]))
            for entry in busy_raw
        ]

        lines = []
        day = start_day
        now = datetime.now(tz)
        while day <= end_day and len(lines) < MAX_DAYS:
            slots: list[str] = []
            slot_start = datetime.combine(day, BUSINESS_START, tzinfo=tz)
            day_end = datetime.combine(day, BUSINESS_END, tzinfo=tz)
            while slot_start + timedelta(minutes=SLOT_MINUTES) <= day_end:
                slot_end = slot_start + timedelta(minutes=SLOT_MINUTES)
                overlaps = any(
                    slot_start < busy_end and slot_end > busy_start
                    for busy_start, busy_end in busy
                )
                if slot_start > now and not overlaps:
                    slots.append(slot_start.strftime("%I:%M %p"))
                    if len(slots) >= SLOTS_PER_DAY:
                        break
                slot_start = slot_end
            lines.append(f"{day.strftime('%a %d %b')}: " + (", ".join(slots) or "fully booked"))
            day += timedelta(days=1)

        return "\n".join(lines) if lines else "No open slots found in that range."

    except Exception:
        logger.exception("Google Calendar availability check failed")
        return _mock_availability(start_day, end_day)


def book_slot(name: str, contact: str, datetime_str: str) -> str:
    """Book a 30-minute demo. Returns a spoken-friendly confirmation or conflict."""
    start = _parse_datetime(datetime_str)
    if start is None:
        return (
            "I couldn't understand that date and time. "
            "Could you give it to me as something like 2026-12-01 15:00?"
        )

    tz = _tz()
    start = start.astimezone(tz)
    if start <= datetime.now(tz):
        return (
            f"{_format_slot(start)} is in the past. "
            "Which upcoming slot would work for you?"
        )
    end = start + timedelta(minutes=SLOT_MINUTES)
    settings = get_settings()
    service = _get_service()

    if service is None:
        # Mock fallback: the demo keeps working without credentials.
        logger.info("[MOCK] Booking for %s (%s) at %s", name, contact, start.isoformat())
        return (
            f"[MOCK BOOKING] Confirmed a demo for {name} on {_format_slot(start)} "
            f"({settings.google_calendar_timezone}). Google Calendar is not configured, "
            f"so this booking was not written to a real calendar."
        )

    try:
        # Re-check the specific slot so we do not double-book.
        freebusy = (
            service.freebusy()
            .query(
                body={
                    "timeMin": start.isoformat(),
                    "timeMax": end.isoformat(),
                    "timeZone": settings.google_calendar_timezone,
                    "items": [{"id": settings.google_calendar_id}],
                }
            )
            .execute()
        )
        busy = freebusy.get("calendars", {}).get(settings.google_calendar_id, {}).get("busy", [])
        if busy:
            return (
                f"That slot ({_format_slot(start)}) is already taken. "
                "Would you like to pick another time from the available slots?"
            )

        event = {
            "summary": f"Demo call - {name}",
            "description": (
                f"Demo booked by the AI sales voice agent.\n"
                f"Contact: {contact}\n"
                f"Booked at: {datetime.now(tz).isoformat()}"
            ),
            "start": {"dateTime": start.isoformat(), "timeZone": settings.google_calendar_timezone},
            "end": {"dateTime": end.isoformat(), "timeZone": settings.google_calendar_timezone},
        }
        created = service.events().insert(calendarId=settings.google_calendar_id, body=event).execute()
        logger.info("Booked event %s at %s", created.get("id"), start.isoformat())
        return f"Confirmed. Demo booked for {_format_slot(start)} ({settings.google_calendar_timezone})."

    except Exception:
        logger.exception("Google Calendar booking failed")
        return "Sorry, I'm having trouble booking that right now. Could we try another slot?"
