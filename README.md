## Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

## Configure
copy .env.example .env
# fill OPENAI_API_KEY
# (optional) set API_KEY to protect /v1/chat/stream
# (optional) adjust RATE_LIMIT, RATE_LIMIT_ENABLED, LOG_LEVEL, SENTRY_DSN

## Ingest
python -m scripts.ingest

## Optional: Website Snapshot Ingest
python -m scripts.fetch_site --lang en --render
python -m scripts.fetch_site --lang de --render
python -m scripts.ingest

## Calendar Booking (Google Calendar)
Set the following environment variables:
- GOOGLE_CALENDAR_ID (calendar ID or shared calendar address)
- GOOGLE_SERVICE_ACCOUNT_JSON (service account JSON string) or GOOGLE_SERVICE_ACCOUNT_FILE (path)
- GOOGLE_CALENDAR_TIMEZONE (e.g. Europe/Berlin)
- (Optional) GOOGLE_IMPERSONATE_USER for Workspace domain-wide delegation

Endpoint:
POST /v1/actions/calendar/book

Payload example:
{
  "name": "Jane Doe",
  "email": "jane@company.com",
  "start": "2026-02-10T10:00:00+01:00",
  "end": "2026-02-10T10:30:00+01:00",
  "timezone": "Europe/Berlin",
  "title": "Intro Call",
  "notes": "Looking to discuss a senior role."
}

## Run
python -m uvicorn app.main:app --reload
