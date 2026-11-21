"""Retry helpers for transient Gemini model-overload errors."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


def is_model_overloaded(exc: Exception) -> bool:
    """Return True only for overload / temporary-unavailable style errors."""
    parts = [str(exc)]
    for attribute in ("message", "status", "code"):
        value = getattr(exc, attribute, None)
        if value is not None:
            parts.append(str(value))
    message = " ".join(parts).lower()
    status = str(getattr(exc, "status", "") or "").lower()
    code = str(getattr(exc, "code", "") or "")

    if code == "503" or status in {"unavailable", "service_unavailable"}:
        return True

    overload_markers = (
        "model is overloaded",
        "model overloaded",
        "overloaded",
        "high demand",
        "temporarily unavailable",
        "service unavailable",
        "please try again later",
        "try again in a moment",
        "model is busy",
    )
    return any(marker in message for marker in overload_markers)


async def generate_content_with_retry(
    gemini_client: Any,
    *,
    operation_name: str,
    max_retries: int = 10,
    delay_seconds: float = 1.0,
    **kwargs: Any,
) -> Any:
    """Retry model-overload errors with a fixed delay.

    ``max_retries`` counts retries after the initial request. Other errors are
    raised immediately so invalid keys, bad requests, and quota errors are not
    repeatedly sent to the API.
    """
    retries = max(0, int(max_retries))
    delay = max(0.0, float(delay_seconds))
    attempt = 0

    while True:
        try:
            return await gemini_client.models.generate_content(**kwargs)
        except Exception as exc:
            if not is_model_overloaded(exc) or attempt >= retries:
                raise
            attempt += 1
            logger.warning(
                "Gemini перегружена во время '%s'; повтор %s/%s через %.1f сек. (%s)",
                operation_name,
                attempt,
                retries,
                delay,
                exc,
            )
            await asyncio.sleep(delay)
