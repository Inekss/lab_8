from findex.crawler.urlcanon import canonicalize


def test_lowercase_scheme_host_and_strip_fragment() -> None:
    assert (
        canonicalize("HTTPS://Example.COM/Path#section")
        == "https://example.com/Path"
    )


def test_sort_query_and_trailing_slash() -> None:
    assert (
        canonicalize("https://ex.com/a/?b=2&a=1")
        == "https://ex.com/a?a=1&b=2"
    )
    assert canonicalize("https://ex.com/a/") == "https://ex.com/a"
    assert canonicalize("https://ex.com/") == "https://ex.com/"
