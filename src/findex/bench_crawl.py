from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

from findex.crawler.crawl import CrawlStats, crawl
from findex.crawler.urlcanon import canonicalize

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def peak_rss_mib(fn):
    box = [0.0]
    stop = threading.Event()

    def watch() -> None:
        proc = psutil.Process(os.getpid()) if psutil else None
        while not stop.is_set():
            if proc is not None:
                box[0] = max(box[0], proc.memory_info().rss / (1024 * 1024))
            stop.wait(0.05)

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    try:
        result = fn()
    finally:
        stop.set()
        t.join(timeout=1)
    return result, box[0]


async def run_once(
    seed: str,
    *,
    max_pages: int,
    concurrency: int,
    per_host: int,
    host_delay: float,
    out: Path,
    log: Path,
) -> tuple[CrawlStats, float]:
    stats = CrawlStats()
    domain = (urlsplit(seed).hostname or "").lower()
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()
    with out.open("w", encoding="utf-8") as fh, log.open("w", encoding="utf-8") as lf:
        lf.write("url\tstatus\tbytes\telapsed\terror\n")
        async for page in crawl(
            [canonicalize(seed)],
            max_pages=max_pages,
            concurrency=concurrency,
            per_host=per_host,
            allowed_domains=[domain],
            host_delay=host_delay,
            stats=stats,
        ):
            fh.write(
                json.dumps(
                    {
                        "url": page.url,
                        "title": page.title,
                        "text": page.text,
                        "fetched_at": page.fetched_at,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            while stats.log_lines:
                lf.write(stats.log_lines.pop(0) + "\n")
        while stats.log_lines:
            lf.write(stats.log_lines.pop(0) + "\n")
    wall = perf_counter() - t0
    return stats, wall


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark findex crawl concurrency")
    p.add_argument("--seed", default="https://devhelm.io/")
    p.add_argument("--max-pages", type=int, default=200)
    p.add_argument("--per-host", type=int, default=2)
    p.add_argument("--host-delay", type=float, default=0.25)
    p.add_argument("--concurrencies", default="1,5,20")
    p.add_argument("--out-dir", default="data/bench")
    args = p.parse_args()

    concurrencies = [int(x) for x in args.concurrencies.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"| Concurrency | Pages | Wall (s) | Pages/s | Errors | Peak RSS |\n"
        f"|---|---|---|---|---|---|"
    )
    for c in concurrencies:
        out = out_dir / f"crawl_c{c}.jsonl"
        log = out_dir / f"crawl_c{c}.log"

        def job(c=c, out=out, log=log):
            return asyncio.run(
                run_once(
                    args.seed,
                    max_pages=args.max_pages,
                    concurrency=c,
                    per_host=args.per_host,
                    host_delay=args.host_delay,
                    out=out,
                    log=log,
                )
            )

        (stats, wall), rss = peak_rss_mib(job)
        pps = stats.pages / wall if wall > 0 else 0.0
        print(
            f"| {c} | {stats.pages} | {wall:.2f} | {pps:.2f} | {stats.errors} | {rss:.1f} MiB |"
        )


if __name__ == "__main__":
    main()
