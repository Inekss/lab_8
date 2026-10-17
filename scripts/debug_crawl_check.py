"""Run a short crawl with asyncio debug=True; exit 1 if the loop was blocked."""
from __future__ import annotations

import asyncio
import logging
import sys

from findex.cli import crawl_main


async def main() -> None:
    await crawl_main(
        "https://devhelm.io/",
        max_pages=5,
        concurrency=2,
        per_host=1,
        host_delay=0.1,
        out="data/debug_check.jsonl",
        log="data/debug_check.log",
        allowed_domains=["devhelm.io"],
        debug=True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    loop = asyncio.new_event_loop()
    # 100ms default is noisy on Windows (TLS + thread-pool wakeups). 250ms still
    # catches time.sleep / requests / heavy CPU on the loop itself.
    loop.slow_callback_duration = 0.25
    asyncio.set_event_loop(loop)
    try:
        loop.set_debug(True)
        loop.run_until_complete(main())
    finally:
        loop.close()
    print("debug crawl finished without blocking the event loop", file=sys.stderr)
