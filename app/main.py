import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler
import sentry_sdk

from app.api.v1.health import router as health_router
from app.api.v1.chat import router as chat_router
from app.api.v1.analytics import router as analytics_router
from app.api.v1.actions import router as actions_router
from app.core.cors import DEV_ALLOWED_ORIGINS
from app.core.rate_limit import limiter
from app.core.config import settings
from dotenv import load_dotenv
from app.core.lifespan import lifespan

load_dotenv()
logging.basicConfig(level=settings.log_level, format="%(message)s")
if settings.sentry_dsn:
    sentry_sdk.init(dsn=settings.sentry_dsn)

app = FastAPI(title="Portfolio Chat Bot API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=DEV_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.include_router(health_router, prefix="/v1", tags=["Health"])
app.include_router(chat_router, prefix="/v1", tags=["Chat"])
app.include_router(analytics_router, prefix="/v1", tags=["Analytics"])
app.include_router(actions_router, prefix="/v1", tags=["Actions"])
