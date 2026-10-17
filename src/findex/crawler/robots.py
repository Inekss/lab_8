from __future__ import annotations

import asyncio
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from findex.crawler.fetch import DEFAULT_USER_AGENT


class RobotsCache:
    def __init__(self, client: httpx.AsyncClient, user_agent: str = DEFAULT_USER_AGENT) -> None:
        self._client = client
        self._ua = user_agent
        self._cache: dict[str, RobotFileParser] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def _load(self, host: str, scheme: str) -> RobotFileParser:
        rp = RobotFileParser()
        try:
            resp = await self._client.get(f"{scheme}://{host}/robots.txt", timeout=10.0)
            lines = resp.text.splitlines() if resp.status_code < 400 else []
        except httpx.HTTPError:
            lines = []
        rp.parse(lines)
        return rp

    async def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if not host:
            return False
        if host not in self._cache:
            lock = self._locks.setdefault(host, asyncio.Lock())
            async with lock:
                if host not in self._cache:
                    scheme = (parts.scheme or "https").lower()
                    self._cache[host] = await self._load(host, scheme)
        return self._cache[host].can_fetch(self._ua, url)
