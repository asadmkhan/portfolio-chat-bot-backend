from fastapi import APIRouter, Depends, Header, Query

from app.core.security import check_api_key
from app.analytics import db as analytics_db

router = APIRouter()


def _auth(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    check_api_key(x_api_key, None)


@router.get("/analytics/summary")
def summary(_: None = Depends(_auth)):
    return analytics_db.get_summary()


@router.get("/analytics/latest")
def latest(
    limit: int = Query(default=20, ge=1, le=200),
    _: None = Depends(_auth),
):
    return analytics_db.get_latest(limit=limit)


@router.get("/analytics/top-questions")
def top_questions(
    limit: int = Query(default=10, ge=1, le=100),
    _: None = Depends(_auth),
):
    return analytics_db.get_top_questions(limit=limit)


@router.get("/analytics/feedback")
def feedback(
    limit: int = Query(default=20, ge=1, le=200),
    _: None = Depends(_auth),
):
    return analytics_db.get_feedback(limit=limit)
