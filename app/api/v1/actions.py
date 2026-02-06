from datetime import datetime

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field

from app.core.security import check_api_key
from app.core.config import settings
import logging

from app.integrations.google_calendar import create_booking
from app.integrations.email import send_admin_booking_notice

router = APIRouter()
logger = logging.getLogger(__name__)


class CalendarBookingRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(min_length=3, max_length=200)
    start: str
    end: str
    timezone: str | None = None
    title: str | None = None
    notes: str | None = None


class CalendarAvailabilityRequest(BaseModel):
    start: str
    end: str
    timezone: str | None = None


def _auth(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    check_api_key(x_api_key, None)


@router.post("/actions/calendar/book")
def book_calendar(payload: CalendarBookingRequest, _: None = Depends(_auth)):
    if not settings.google_calendar_id:
        raise HTTPException(status_code=500, detail="Google Calendar not configured.")

    # Basic validation for ordering
    try:
        start_dt = datetime.fromisoformat(payload.start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(payload.end.replace("Z", "+00:00"))
        if end_dt <= start_dt:
            raise HTTPException(status_code=400, detail="End time must be after start time.")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid datetime format.")

    payload_data = payload.model_dump()
    result = create_booking(payload_data)
    if result.get("status") == "busy":
        raise HTTPException(status_code=409, detail="Selected time is not available.")
    try:
        email_sent = send_admin_booking_notice(payload_data)
    except Exception as exc:  # noqa: BLE001 - do not block booking
        logger.exception("Admin booking email failed: %s", exc)
        email_sent = False
    response = dict(result)
    response["email_sent"] = email_sent
    return response


@router.post("/actions/calendar/availability")
def check_availability(payload: CalendarAvailabilityRequest, _: None = Depends(_auth)):
    if not settings.google_calendar_id:
        raise HTTPException(status_code=500, detail="Google Calendar not configured.")

    try:
        start_dt = datetime.fromisoformat(payload.start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(payload.end.replace("Z", "+00:00"))
        if end_dt <= start_dt:
            raise HTTPException(status_code=400, detail="End time must be after start time.")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid datetime format.")

    from app.integrations.google_calendar import check_busy

    busy = check_busy(payload.start, payload.end, payload.timezone or settings.google_calendar_timezone)
    return {"available": not busy}
