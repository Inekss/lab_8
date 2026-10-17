from findex.crawler.parse import extract_links, extract_title_text


def test_extract_links_and_title() -> None:
    html = (
        b"<html><head><title>Home</title></head>"
        b'<body><a href="/a">A</a><a href="/b">B</a></body></html>'
    )
    title, text = extract_title_text(html)
    assert title == "Home"
    assert "A" in text and "B" in text
    links = extract_links(html, "https://docs.example.com/")
    assert "https://docs.example.com/a" in links
    assert "https://docs.example.com/b" in links
