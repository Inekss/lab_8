"""Compare blocking async def vs plain def for 20 concurrent /search calls.

Uses a short ``time.sleep`` (releases the GIL) to make event-loop blocking
obvious — same experiment as the lab notes. Pure-Python BM25 is GIL-bound, so
threads alone barely help there; that is why the README also measures multiple
uvicorn workers.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Query
from httpx import ASGITransport, AsyncClient

from findex.models import DocMeta, Index, Posting
from findex.score import get_scorer
from findex.search import search as run_search

SLEEP_S = 0.05  # 20 concurrent → ~1.0s if serialized on the loop, ~0.05s if pooled


def _index() -> Index:
    return Index(
        {
            "harry": [Posting(0, 2, (0, 5))],
            "potter": [Posting(0, 1, (1,))],
        },
        {0: 10},
        {0: DocMeta("mem://0", "alpha")},
    )


def build(mode: Literal["async", "sync"]) -> FastAPI:
    index = _index()
    app = FastAPI()

    if mode == "async":

        @app.get("/search")
        async def search_async(q: str = Query("harry")):
            time.sleep(SLEEP_S)  # blocks the event loop
            hits = run_search(index, q, get_scorer("bm25"), k=10)
            return {"n": len(hits)}

    else:

        @app.get("/search")
        def search_sync(q: str = Query("harry")):
            time.sleep(SLEEP_S)  # runs in FastAPI's thread pool
            hits = run_search(index, q, get_scorer("bm25"), k=10)
            return {"n": len(hits)}

    return app


async def hammer(app: FastAPI, n: int = 20) -> float:
    transport = ASGITransport(app=app)
    t0 = time.perf_counter()
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        responses = await asyncio.gather(
            *(client.get("/search", params={"q": "harry"}) for _ in range(n))
        )
    for r in responses:
        assert r.status_code == 200, r.text
    return time.perf_counter() - t0


def main() -> None:
    t_async = asyncio.run(hammer(build("async"), 20))
    t_sync = asyncio.run(hammer(build("sync"), 20))
    line_a = f"async def + time.sleep({SLEEP_S}), 20 concurrent: {t_async:.4f}s"
    line_b = f"def + time.sleep({SLEEP_S}),          20 concurrent: {t_sync:.4f}s"
    print(line_a)
    print(line_b)
    out = Path("data/bench_handlers.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(line_a + "\n" + line_b + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
