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

## Run
python -m uvicorn app.main:app --reload
