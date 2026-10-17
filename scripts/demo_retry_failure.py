"""Demonstrate 503 retries with backoff; writes lines shaped like crawl.log."""
from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from findex.crawler.fetch import fetch


async def main() -> None:
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        if hits["n"] < 3:
            return httpx.Response(503, headers={"Retry-After": "0"}, content=b"busy")
        return httpx.Response(200, content=b"<html><title>ok</title><body>ok</body></html>")

    out = Path("data/failure_demo.log")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["url\tstatus\tbytes\telapsed\terror"]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        # log each attempt by wrapping — fetch only returns final; call stepwise via public API
        fr = await fetch(client, "https://devhelm.io/flaky", max_attempts=3)
        # Reconstruct attempt log from hit count (mirrors crawl.log after retries settle)
        for i in range(1, hits["n"]):
            lines.append(f"https://devhelm.io/flaky\t503\t4\t0.001\tHTTP 503 (attempt {i}, retry)")
        lines.append(
            f"{fr.url}\t{fr.status}\t{fr.nbytes}\t{fr.elapsed:.3f}\t{fr.error or ''}"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    asyncio.run(main())
