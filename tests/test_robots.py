from __future__ import annotations

import asyncio

import httpx

from findex.crawler.robots import RobotsCache


def test_robots_cache_disallow_and_allow() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            body = "User-agent: *\nDisallow: /private\nAllow: /\n"
            return httpx.Response(200, text=body)
        return httpx.Response(404)

    async def main() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            cache = RobotsCache(client, user_agent="findex-test")
            assert await cache.allowed("https://example.com/public")
            assert not await cache.allowed("https://example.com/private/x")
            # second call hits cache (robots fetched once)
            assert await cache.allowed("https://example.com/")

    asyncio.run(main())


def test_missing_robots_allows_all() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    async def main() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            cache = RobotsCache(client)
            assert await cache.allowed("https://example.com/anything")

    asyncio.run(main())
