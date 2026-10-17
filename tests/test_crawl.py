from __future__ import annotations

import asyncio

import httpx

from findex.crawler.crawl import CrawlStats, crawl


def test_crawl_respects_max_pages_and_dedupes() -> None:
    pages_html = {
        "/": (
            b"<html><head><title>Home</title></head>"
            b'<body><a href="/a">A</a><a href="/b">B</a></body></html>'
        ),
        "/a": (
            b"<html><head><title>A</title></head>"
            b'<body>page a <a href="/c">C</a></body></html>'
        ),
        "/b": b"<html><head><title>B</title></head><body>page b</body></html>",
        "/c": b"<html><head><title>C</title></head><body>page c</body></html>",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path or "/"
        if path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        body = pages_html.get(path)
        if body is None:
            return httpx.Response(404, content=b"nope")
        return httpx.Response(200, content=body, headers={"content-type": "text/html"})

    async def main() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="https://docs.example.com",
            follow_redirects=True,
        ) as client:
            stats = CrawlStats()
            pages = [
                p
                async for p in crawl(
                    ["https://docs.example.com/"],
                    max_pages=3,
                    concurrency=2,
                    per_host=2,
                    allowed_domains=["docs.example.com"],
                    host_delay=0.0,
                    stats=stats,
                    client=client,
                )
            ]
        assert len(pages) == 3
        assert len({p.url for p in pages}) == 3
        assert stats.pages == 3

    asyncio.run(main())
