from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from typing import Optional
from urllib.parse import urlsplit

from rich.live import Live
from rich.table import Table

from findex.crawler.crawl import CrawlStats, crawl
from findex.crawler.fetch import DEFAULT_USER_AGENT
from findex.crawler.urlcanon import canonicalize
from findex.documents import CorpusReader
from findex.embed import build_embeddings, load_embeddings
from findex.models import DocMeta, Index, Posting
from findex.score import BM25, TfIdf
from findex.search import SearchMode, search
from findex.store import load_pickle, timed_save_load
from findex.tokenizer import Tokenizer


def _domain_from_url(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def _make_stats_table(stats: CrawlStats) -> Table:
    table = Table(title="findex crawl")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("pages", str(stats.pages))
    table.add_row("pages/s", f"{stats.pages_per_sec:.2f}")
    table.add_row("in-flight", str(stats.in_flight))
    table.add_row("errors", str(stats.errors))
    table.add_row("queue", str(stats.queue_depth))
    return table


def _flush_log(log_fh, stats: CrawlStats, visited: list[str]) -> None:
    while stats.log_lines:
        line = stats.log_lines.pop(0)
        log_fh.write(line + "\n")
        visited.append(line.split("\t", 1)[0])
    log_fh.flush()


async def crawl_main(
    seeds: list[str] | str,
    *,
    max_pages: int = 500,
    concurrency: int = 10,
    per_host: int = 2,
    host_delay: float = 0.25,
    timeout: float = 10.0,
    out: Path | str = "data/crawl.jsonl",
    log: Path | str = "crawl.log",
    allowed_domains: list[str] | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
    debug: bool = False,
) -> CrawlStats:
    if isinstance(seeds, str):
        seeds = [seeds]
    seeds = [canonicalize(u) for u in seeds]
    allowed = allowed_domains or [_domain_from_url(s) for s in seeds]

    out_path = Path(out)
    log_path = Path(log)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    stats = CrawlStats()
    visited: list[str] = []
    saved: list[str] = []

    with out_path.open("w", encoding="utf-8") as fh, log_path.open("w", encoding="utf-8") as log_fh:
        log_fh.write("url\tstatus\tbytes\telapsed\terror\n")
        # rich.Live can block the loop on Windows terminals; skip under asyncio debug
        live_cm = (
            nullcontext(None)
            if debug
            else Live(_make_stats_table(stats), refresh_per_second=4)
        )
        with live_cm as live:
            async for page in crawl(
                seeds,
                max_pages=max_pages,
                concurrency=concurrency,
                per_host=per_host,
                allowed_domains=allowed,
                host_delay=host_delay,
                timeout=timeout,
                user_agent=user_agent,
                stats=stats,
            ):
                saved.append(page.url)
                line = json.dumps(
                    {
                        "url": page.url,
                        "title": page.title,
                        "text": page.text,
                        "fetched_at": page.fetched_at,
                    },
                    ensure_ascii=False,
                )

                def _write(line: str = line) -> None:
                    fh.write(line + "\n")
                    fh.flush()
                    _flush_log(log_fh, stats, visited)

                await asyncio.to_thread(_write)
                if live is not None:
                    live.update(_make_stats_table(stats))
            if live is not None:
                live.update(_make_stats_table(stats))
        _flush_log(log_fh, stats, visited)

    print(
        f"wrote {stats.pages} pages to {out_path} "
        f"({stats.pages_per_sec:.2f} pages/s, errors={stats.errors})",
        file=sys.stderr,
    )
    saved_set = set(saved)
    print("visited:", file=sys.stderr)
    for url in dict.fromkeys(visited):
        mark = "saved" if url in saved_set else "fetch"
        print(f"  [{mark}] {url}", file=sys.stderr)
    return stats


def cmd_crawl(args: argparse.Namespace) -> int:
    kwargs = dict(
        seeds=args.seeds,
        max_pages=args.max_pages,
        concurrency=args.concurrency,
        per_host=args.per_host,
        host_delay=args.host_delay,
        timeout=args.timeout,
        out=args.out,
        log=args.log,
        allowed_domains=args.allowed_domain,
        user_agent=args.user_agent,
        debug=args.debug,
    )
    if not args.debug:
        asyncio.run(crawl_main(**kwargs), debug=False)
        return 0
    loop = asyncio.new_event_loop()
    # 100ms default is noisy on Windows (TLS + to_thread); 250ms still flags real blocks
    loop.slow_callback_duration = 0.25
    asyncio.set_event_loop(loop)
    try:
        loop.set_debug(True)
        loop.run_until_complete(crawl_main(**kwargs))
        return 0
    finally:
        loop.close()


def cmd_index(args: argparse.Namespace) -> int:
    import tracemalloc

    reader = CorpusReader(args.corpus)
    tok = Tokenizer()
    postings: dict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    tracemalloc.start()
    t0 = perf_counter()
    for doc in reader.iter_documents_meta():
        counts: Counter[str] = Counter()
        pos: dict[str, list[int]] = defaultdict(list)
        n = 0
        for i, term in enumerate(tok.tokenize(doc.text)):
            counts[term] += 1
            n += 1
            if args.positions:
                pos[term].append(i)
        doc_lengths[doc.doc_id] = n
        doc_meta[doc.doc_id] = DocMeta(str(doc.path), doc.title)
        for term, tf in counts.items():
            positions = tuple(pos[term]) if args.positions else ()
            postings[term].append(Posting(doc.doc_id, tf, positions))
    index = Index(dict(postings), doc_lengths, doc_meta)
    index.materialize_numpy()
    elapsed = perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"{index!r} time={elapsed:.4f}s peak={peak / 1024:.1f}KiB")
    timed_save_load(index, Path(args.out_pkl), Path(args.out_json))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    index = load_pickle(Path(args.index))
    scorer = TfIdf() if args.scorer == "tfidf" else BM25()
    embeddings = None
    mode: SearchMode = args.mode
    if mode in ("semantic", "hybrid"):
        emb_path = Path(args.embeddings)
        if not emb_path.is_file():
            print(f"embeddings not found: {emb_path}", file=sys.stderr)
            return 1
        embeddings = load_embeddings(emb_path)
    for q in args.queries:
        hits = search(
            index,
            q,
            scorer,
            k=args.k,
            backend=args.backend,
            mode=mode,
            embeddings=embeddings,
            scorer_name=args.scorer,
        )
        print(f"query: {q}  mode={mode}")
        for h in hits:
            snip = h.snippet[:80] if h.snippet else ""
            print(f"  {h.score:.4f}  {h.title}  {snip}")
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    index = load_pickle(Path(args.index))
    store = build_embeddings(
        index,
        model_name=args.model,
        out=Path(args.out),
        batch_size=args.batch_size,
    )
    print(f"embeddings n_chunks={store.n_chunks} dim={store.dim} → {args.out}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import os

    import uvicorn

    if args.index is not None:
        os.environ["INDEX_PATH"] = str(Path(args.index).resolve())
    if args.embeddings is not None:
        os.environ["EMBEDDINGS_PATH"] = str(Path(args.embeddings).resolve())
    # drop cached settings so INDEX_PATH is re-read
    from findex.web.settings import get_settings

    get_settings.cache_clear()
    uvicorn.run(
        "findex.web:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level.lower(),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="findex", description="findex search engine CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("crawl", help="async polite web crawl → JSONL")
    c.add_argument("seeds", nargs="+", help="seed URL(s)")
    c.add_argument("--max-pages", type=int, default=500)
    c.add_argument("--concurrency", type=int, default=10)
    c.add_argument("--per-host", type=int, default=2)
    c.add_argument("--host-delay", type=float, default=0.25)
    c.add_argument("--timeout", type=float, default=10.0)
    c.add_argument("--out", default="data/crawl.jsonl")
    c.add_argument("--log", default="crawl.log")
    c.add_argument("--allowed-domain", action="append", default=None)
    c.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    c.add_argument("--debug", action="store_true", help="asyncio debug mode")
    c.set_defaults(func=cmd_crawl)

    i = sub.add_parser("index", help="index corpus (txt dir or crawl.jsonl)")
    i.add_argument("--corpus", required=True, help="directory of *.txt or a .jsonl file")
    i.add_argument("--out-pkl", default="index.pkl")
    i.add_argument("--out-json", default="index.json")
    i.add_argument("--positions", action="store_true", default=True)
    i.add_argument("--no-positions", action="store_false", dest="positions")
    i.set_defaults(func=cmd_index)

    s = sub.add_parser("search", help="search a saved index")
    s.add_argument("--index", default="index.pkl")
    s.add_argument("queries", nargs="+")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--scorer", choices=("bm25", "tfidf"), default="bm25")
    s.add_argument("--backend", choices=("numpy", "loop"), default="numpy")
    s.add_argument(
        "--mode",
        choices=("keyword", "semantic", "hybrid"),
        default="keyword",
    )
    s.add_argument("--embeddings", default="embeddings.npy")
    s.set_defaults(func=cmd_search)

    e = sub.add_parser("embed", help="chunk docs and build embedding matrix (.npy)")
    e.add_argument("--index", default="index.pkl")
    e.add_argument("--out", default="embeddings.npy")
    e.add_argument("--model", default="all-MiniLM-L6-v2")
    e.add_argument("--batch-size", type=int, default=32)
    e.set_defaults(func=cmd_embed)

    v = sub.add_parser("serve", help="run HTTP API + web UI (uvicorn)")
    v.add_argument("--index", default=None, help="index file; sets INDEX_PATH")
    v.add_argument(
        "--embeddings",
        default=None,
        help="embeddings .npy; sets EMBEDDINGS_PATH",
    )
    v.add_argument("--host", default="0.0.0.0")
    v.add_argument("--port", type=int, default=8000)
    v.add_argument("--workers", type=int, default=1)
    v.add_argument("--log-level", default="info")
    v.set_defaults(func=cmd_serve)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
