## Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

## Configure
copy .env.example .env
# fill OPENAI_API_KEY

## Ingest
python -m scripts.ingest

## Run
python -m uvicorn app.main:app --reload
