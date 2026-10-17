from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def canonicalize(url: str) -> str:
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    netloc = f"{host}:{parts.port}" if parts.port else host
    path = parts.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((parts.scheme.lower(), netloc, path, query, ""))
