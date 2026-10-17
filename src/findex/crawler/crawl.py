from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlsplit

import httpx

from findex.crawler.fetch import DEFAULT_USER_AGENT, FetchResult, fetch
from findex.crawler.parse import parse_html
from findex.crawler.robots import RobotsCache
from findex.crawler.urlcanon import canonicalize


@dataclass(frozen=True, slots=True)
class Page:
    url: str
    title: str
    text: str
    fetched_at: str


@dataclass
class CrawlStats:
    pages: int = 0
    errors: int = 0
    in_flight: int = 0
    queue_depth: int = 0
    started_at: float = field(default_factory=time.perf_counter)
    log_lines: list[str] = field(default_factory=list)

    @property
    def pages_per_sec(self) -> float:
        return self.pages / max(1e-9, time.perf_counter() - self.started_at)


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


class Crawler:
    def __init__(
        self,
        seeds: Iterable[str],
        *,
        max_pages: int = 500,
        concurrency: int = 10,
        per_host: int = 2,
        allowed_domains: Iterable[str] | None = None,
        host_delay: float = 0.25,
        timeout: float = 10.0,
        user_agent: str = DEFAULT_USER_AGENT,
        stats: CrawlStats | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.max_pages = max_pages
        self.concurrency = max(1, concurrency)
        self.per_host = max(1, per_host)
        self.host_delay = host_delay
        self.timeout = timeout
        self.user_agent = user_agent
        self.stats = stats or CrawlStats()
        self.allowed = {d.lower().lstrip(".") for d in (allowed_domains or [])}
        self.client = client
        self.owns_client = client is None
        self.seeds = list(seeds)

        self.frontier: asyncio.Queue[str] = asyncio.Queue()
        self.out: asyncio.Queue[Page | None] = asyncio.Queue()
        self.seen: set[str] = set()
        self.stop = asyncio.Event()
        self.pages_done = 0
        self.pending = 0
        self.global_sem = asyncio.Semaphore(self.concurrency)
        self.host_sems: dict[str, asyncio.Semaphore] = {}
        self.host_next: dict[str, float] = {}
        self.robots: RobotsCache | None = None

    def _domain_ok(self, url: str) -> bool:
        host = _host(url)
        if not host:
            return False
        if not self.allowed:
            return True
        return host in self.allowed or any(host.endswith("." + d) for d in self.allowed)

    def enqueue(self, url: str) -> None:
        if self.stop.is_set() or self.pages_done >= self.max_pages:
            return
        url = canonicalize(url)
        if url in self.seen or not self._domain_ok(url):
            return
        self.seen.add(url)
        self.pending += 1
        self.frontier.put_nowait(url)
        self.stats.queue_depth = self.frontier.qsize()

    def _log(self, fr: FetchResult) -> None:
        status = fr.status if fr.status is not None else "-"
        self.stats.log_lines.append(
            f"{fr.url}\t{status}\t{fr.nbytes}\t{fr.elapsed:.3f}\t{fr.error or ''}"
        )

    async def _polite_fetch(self, url: str) -> FetchResult:
        assert self.client is not None
        host = _host(url)
        sem = self.host_sems.setdefault(host, asyncio.Semaphore(self.per_host))
        async with self.global_sem, sem:
            wait = self.host_next.get(host, 0.0) - time.perf_counter()
            if wait > 0:
                await asyncio.sleep(wait)
            self.stats.in_flight += 1
            try:
                return await fetch(self.client, url, timeout=self.timeout)
            finally:
                self.stats.in_flight -= 1
                self.host_next[host] = time.perf_counter() + self.host_delay

    async def _handle(self, url: str) -> None:
        assert self.robots is not None
        if self.stop.is_set() or self.pages_done >= self.max_pages:
            return

        if not await self.robots.allowed(url):
            self.stats.errors += 1
            self._log(FetchResult(url=url, status=None, nbytes=0, elapsed=0.0, error="robots disallowed"))
            return

        fr = await self._polite_fetch(url)
        self._log(fr)
        if fr.status is None or fr.status >= 400 or not fr.body:
            if fr.status is None or fr.status >= 400:
                self.stats.errors += 1
            return

        base = fr.final_url or url
        title, text, links = await asyncio.to_thread(parse_html, fr.body, base)
        page = Page(
            url=canonicalize(base),
            title=title or canonicalize(base),
            text=text,
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )

        if self.pages_done >= self.max_pages or self.stop.is_set():
            return
        self.pages_done += 1
        self.stats.pages = self.pages_done
        await self.out.put(page)

        if self.pages_done >= self.max_pages:
            self.stop.set()
            return
        for link in links:
            self.enqueue(link)

    async def _worker(self) -> None:
        while not self.stop.is_set():
            try:
                url = await asyncio.wait_for(self.frontier.get(), timeout=0.15)
            except TimeoutError:
                if self.pending <= 0:
                    self.stop.set()
                    return
                continue
            try:
                self.stats.queue_depth = self.frontier.qsize()
                await self._handle(url)
            finally:
                self.pending -= 1
                self.frontier.task_done()

    async def pages(self) -> AsyncIterator[Page]:
        if self.client is None:
            self.client = httpx.AsyncClient(
                headers={"User-Agent": self.user_agent},
                follow_redirects=True,
                timeout=self.timeout,
            )
        self.robots = RobotsCache(self.client, user_agent=self.user_agent)

        for seed in self.seeds:
            self.enqueue(seed)

        async def run_pool() -> None:
            try:
                async with asyncio.TaskGroup() as tg:
                    for _ in range(self.concurrency):
                        tg.create_task(self._worker())
            finally:
                self.stop.set()
                await self.out.put(None)

        producer = asyncio.create_task(run_pool())
        yielded = 0
        try:
            while True:
                item = await self.out.get()
                if item is None:
                    break
                yield item
                yielded += 1
                if yielded >= self.max_pages:
                    self.stop.set()
                    break
        finally:
            self.stop.set()
            if not producer.done():
                producer.cancel()
                try:
                    await producer
                except (asyncio.CancelledError, ExceptionGroup, Exception):
                    pass
            if self.owns_client and self.client is not None:
                await self.client.aclose()


async def crawl(
    seeds: Iterable[str],
    *,
    max_pages: int = 500,
    concurrency: int = 10,
    per_host: int = 2,
    allowed_domains: Iterable[str] | None = None,
    host_delay: float = 0.25,
    timeout: float = 10.0,
    user_agent: str = DEFAULT_USER_AGENT,
    stats: CrawlStats | None = None,
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[Page]:
    crawler = Crawler(
        seeds,
        max_pages=max_pages,
        concurrency=concurrency,
        per_host=per_host,
        allowed_domains=allowed_domains,
        host_delay=host_delay,
        timeout=timeout,
        user_agent=user_agent,
        stats=stats,
        client=client,
    )
    async for page in crawler.pages():
        yield page
