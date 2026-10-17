from __future__ import annotations

import asyncio

import httpx

import findex.crawler.fetch as fetch_mod
from findex.crawler.fetch import fetch


def _run(coro):
    return asyncio.run(coro)


def test_fetch_ok() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"hello")

    async def main() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            fr = await fetch(client, "https://example.com/")
        assert fr.status == 200
        assert fr.body == b"hello"
        assert fr.error is None
        assert fr.nbytes == 5

    _run(main())


def test_no_retry_on_404() -> None:
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return httpx.Response(404, content=b"missing")

    async def main() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            fr = await fetch(client, "https://example.com/x")
        assert fr.status == 404
        assert hits["n"] == 1
        assert fr.error == "HTTP 404"

    _run(main())


def test_retry_5xx_then_ok() -> None:
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        if hits["n"] < 3:
            return httpx.Response(503, content=b"busy")
        return httpx.Response(200, content=b"ok")

    async def main() -> None:
        async def no_sleep(_delay: float) -> None:
            return None

        orig = fetch_mod.asyncio.sleep
        fetch_mod.asyncio.sleep = no_sleep  # type: ignore[method-assign]
        try:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                fr = await fetch(client, "https://example.com/", max_attempts=3)
        finally:
            fetch_mod.asyncio.sleep = orig  # type: ignore[method-assign]
        assert fr.status == 200
        assert hits["n"] == 3
        assert fr.body == b"ok"

    _run(main())


def test_retry_after_honored() -> None:
    hits = {"n": 0}
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        if hits["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "1"}, content=b"slow")
        return httpx.Response(200, content=b"ok")

    async def main() -> None:
        async def fake_sleep(delay: float) -> None:
            slept.append(delay)

        orig = fetch_mod.asyncio.sleep
        fetch_mod.asyncio.sleep = fake_sleep  # type: ignore[method-assign]
        try:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                fr = await fetch(client, "https://example.com/r")
        finally:
            fetch_mod.asyncio.sleep = orig  # type: ignore[method-assign]
        assert fr.status == 200
        assert hits["n"] == 2
        assert slept
        assert slept[0] >= 1.0

    _run(main())


def test_timeout_retries() -> None:
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        raise httpx.ReadTimeout("slow")

    async def main() -> None:
        async def no_sleep(_delay: float) -> None:
            return None

        orig = fetch_mod.asyncio.sleep
        fetch_mod.asyncio.sleep = no_sleep  # type: ignore[method-assign]
        try:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                fr = await fetch(client, "https://example.com/t", max_attempts=3, timeout=1.0)
        finally:
            fetch_mod.asyncio.sleep = orig  # type: ignore[method-assign]
        assert fr.status is None
        assert hits["n"] == 3
        assert fr.error is not None

    _run(main())
