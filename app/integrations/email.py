from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


def _smtp_ready() -> bool:
    return bool(settings.smtp_host and settings.admin_notify_email)


def _smtp_password() -> str | None:
    if not settings.smtp_password:
        return None
    # Gmail app passwords are often copied with spaces every 4 chars.
    return settings.smtp_password.replace(" ", "")


def _smtp_login_if_needed(server: smtplib.SMTP) -> None:
    if settings.smtp_user and _smtp_password():
        server.login(settings.smtp_user, _smtp_password())


def _send_via_smtp_with(host: str, port: int, use_tls: bool, msg: EmailMessage, context: ssl.SSLContext) -> None:
    if use_tls:
        with smtplib.SMTP(host, port, timeout=15) as server:
            server.starttls(context=context)
            _smtp_login_if_needed(server)
            server.send_message(msg)
        return

    with smtplib.SMTP_SSL(host, port, context=context, timeout=15) as server:
        _smtp_login_if_needed(server)
        server.send_message(msg)


def send_admin_booking_notice(payload: dict[str, Any]) -> bool:
    if not _smtp_ready():
        logger.info("Admin email notification is not configured; skipping.")
        return False

    recipient = settings.admin_notify_email
    sender = settings.smtp_from or settings.smtp_user or recipient

    msg = EmailMessage()
    msg["Subject"] = f"New Booking: {payload.get('name') or 'Unknown'}"
    msg["From"] = sender
    msg["To"] = recipient

    body = "\n".join(
        [
            f"Name: {payload.get('name')}",
            f"Email: {payload.get('email')}",
            f"Start: {payload.get('start')}",
            f"End: {payload.get('end')}",
            f"Timezone: {payload.get('timezone')}",
            f"Title: {payload.get('title')}",
            f"Notes: {payload.get('notes') or ''}",
        ]
    ).strip()
    msg.set_content(body)

    context = ssl.create_default_context()
    primary_mode = "STARTTLS" if settings.smtp_use_tls else "SSL"
    try:
        _send_via_smtp_with(
            host=settings.smtp_host or "",
            port=settings.smtp_port,
            use_tls=settings.smtp_use_tls,
            msg=msg,
            context=context,
        )
        return True
    except Exception as exc:  # noqa: BLE001 - keep booking flow alive
        logger.exception(
            "Admin email via SMTP failed (host=%s port=%s mode=%s): %s",
            settings.smtp_host,
            settings.smtp_port,
            primary_mode,
            exc,
        )

    if not settings.smtp_fallback_ssl:
        return False

    fallback_host = settings.smtp_host or ""
    fallback_port = 465 if settings.smtp_use_tls else 587
    fallback_tls = not settings.smtp_use_tls
    fallback_mode = "STARTTLS" if fallback_tls else "SSL"
    try:
        _send_via_smtp_with(
            host=fallback_host,
            port=fallback_port,
            use_tls=fallback_tls,
            msg=msg,
            context=context,
        )
        logger.info(
            "Admin email sent with SMTP fallback (host=%s port=%s mode=%s).",
            fallback_host,
            fallback_port,
            fallback_mode,
        )
        return True
    except Exception as exc:  # noqa: BLE001 - keep booking flow alive
        logger.exception(
            "Admin email SMTP fallback failed (host=%s port=%s mode=%s): %s",
            fallback_host,
            fallback_port,
            fallback_mode,
            exc,
        )
        return False
