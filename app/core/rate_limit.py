from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings

limiter = Limiter(key_func=get_remote_address)


def rate_limit():
    if settings.rate_limit_enabled:
        return limiter.limit(settings.rate_limit)

    def decorator(func):
        return func

    return decorator
