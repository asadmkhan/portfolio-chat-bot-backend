from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from app.core.config import settings

SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.freebusy",
]


def _get_credentials():
    if settings.google_service_account_json:
        info = json.loads(settings.google_service_account_json)
        creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    elif settings.google_service_account_file:
        creds = service_account.Credentials.from_service_account_file(
            settings.google_service_account_file, scopes=SCOPES
        )
    else:
        raise RuntimeError("Google service account credentials are missing.")

    if settings.google_impersonate_user:
        creds = creds.with_subject(settings.google_impersonate_user)
    return creds


def _service():
    creds = _get_credentials()
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


def _ensure_iso(dt: str) -> str:
    # Accepts ISO strings; ensures timezone-aware by appending Z if missing
    if dt.endswith("Z") or "+" in dt:
        return dt
    return dt + "Z"


def check_busy(start: str, end: str, timezone_name: str) -> bool:
    service = _service()
    body = {
        "timeMin": _ensure_iso(start),
        "timeMax": _ensure_iso(end),
        "timeZone": timezone_name,
        "items": [{"id": settings.google_calendar_id}],
    }
    result = service.freebusy().query(body=body).execute()
    busy = result.get("calendars", {}).get(settings.google_calendar_id, {}).get("busy", [])
    return len(busy) > 0


def create_event(payload: dict[str, Any]) -> dict[str, Any]:
    service = _service()
    tz = payload.get("timezone") or settings.google_calendar_timezone

    event = {
        "summary": payload.get("title") or f"Call with {payload.get('name')}",
        "description": (
            "Name: {name}\nEmail: {email}\nNotes: {notes}"
        ).format(
            name=payload.get("name"),
            email=payload.get("email"),
            notes=payload.get("notes") or "",
        ),
        "start": {"dateTime": _ensure_iso(payload.get("start")), "timeZone": tz},
        "end": {"dateTime": _ensure_iso(payload.get("end")), "timeZone": tz},
        "guestsCanInviteOthers": False,
        "guestsCanModify": False,
        "guestsCanSeeOtherGuests": False,
    }

    created = (
        service.events()
        .insert(
            calendarId=settings.google_calendar_id,
            body=event,
            sendUpdates="none",
        )
        .execute()
    )
    return created


def create_booking(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        if check_busy(payload["start"], payload["end"], payload.get("timezone") or settings.google_calendar_timezone):
            return {"status": "busy"}
        event = create_event(payload)
        return {
            "status": "booked",
            "eventId": event.get("id"),
            "htmlLink": event.get("htmlLink"),
        }
    except HttpError as exc:
        raise RuntimeError(f"Google Calendar API error: {exc}") from exc
