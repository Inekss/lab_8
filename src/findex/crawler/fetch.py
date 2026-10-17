from __future__ import annotations

import asyncio
import email.utils
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

DEFAULT_USER_AGENT = (
    "findex-crawler/0.1 (+https://github.com/Inekss/lab_6; contact@devhelm.io)"
)

MAX_ATTEMPTS = 3
BACKOFF_BASE = 0.5
BACKOFF_CAP = 8.0


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    status: int | None
    nbytes: int
    elapsed: float
    body: bytes = b""
    error: str | None = None
    final_url: str | None = None
    content_type: str | None = None


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def _backoff(attempt: int, retry_after: str | None = None) -> float:
    ra = _retry_after_seconds(retry_after)
    if ra is not None:
        return ra + random.uniform(0.0, 0.25)
    return min(BACKOFF_CAP, BACKOFF_BASE * (2**attempt)) + random.uniform(0.0, 1.0)


def _transient(status: int | None, exc: BaseException | None) -> bool:
    if isinstance(exc, (httpx.TimeoutException, TimeoutError, asyncio.TimeoutError)):
        return True
    if isinstance(exc, httpx.TransportError):
        return True
    return status == 429 or (status is not None and 500 <= status <= 599)


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_attempts: int = MAX_ATTEMPTS,
) -> FetchResult:
    last = FetchResult(url=url, status=None, nbytes=0, elapsed=0.0, error="fetch failed")

    for attempt in range(max_attempts):
        t0 = time.perf_counter()
        try:
            async with asyncio.timeout(timeout):
                resp = await client.get(url)
            elapsed = time.perf_counter() - t0
            body = resp.content
            status = resp.status_code
            last = FetchResult(
                url=url,
                status=status,
                nbytes=len(body),
                elapsed=elapsed,
                body=body,
                error=None if status < 400 else f"HTTP {status}",
                final_url=str(resp.url),
                content_type=resp.headers.get("content-type"),
            )
            if _transient(status, None) and attempt + 1 < max_attempts:
                await asyncio.sleep(_backoff(attempt, resp.headers.get("Retry-After")))
                continue
            return last
        except (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException) as exc:
            last = FetchResult(
                url=url,
                status=None,
                nbytes=0,
                elapsed=time.perf_counter() - t0,
                error=f"timeout: {exc}",
            )
            if attempt + 1 < max_attempts:
                await asyncio.sleep(_backoff(attempt))
        except httpx.HTTPError as exc:
            last = FetchResult(
                url=url,
                status=None,
                nbytes=0,
                elapsed=time.perf_counter() - t0,
                error=f"{type(exc).__name__}: {exc}",
            )
            if _transient(None, exc) and attempt + 1 < max_attempts:
                await asyncio.sleep(_backoff(attempt))
            else:
                break

    return last
