__all__ = [
    "DEFAULT_USER_AGENT",
    "FetchResult",
    "Page",
    "crawl",
    "fetch",
]


def __getattr__(name: str):
    # keep submodule name `fetch` — eager import would shadow it with the function
    if name in ("DEFAULT_USER_AGENT", "FetchResult", "fetch"):
        from findex.crawler import fetch as fetch_module

        return getattr(fetch_module, name)
    if name in ("Page", "crawl"):
        from findex.crawler import crawl as crawl_module

        return getattr(crawl_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
