from __future__ import annotations

from urllib.parse import urljoin, urlsplit

from selectolax.lexbor import LexborHTMLParser


def parse_html(html: str | bytes, base_url: str) -> tuple[str, str, list[str]]:
    tree = LexborHTMLParser(html)
    title_node = tree.css_first("title")
    title = title_node.text(strip=True) if title_node else ""

    links: list[str] = []
    for node in tree.css("a[href]"):
        href = (node.attributes.get("href") or "").strip()
        if not href or href.startswith(("#", "mailto:", "javascript:", "tel:")):
            continue
        absolute = urljoin(base_url, href)
        if urlsplit(absolute).scheme.lower() in ("http", "https"):
            links.append(absolute)

    for sel in ("script", "style", "noscript"):
        for node in tree.css(sel):
            node.decompose()
    body = tree.body
    text = (body if body is not None else tree).text(separator=" ", strip=True)
    return title or "", text, links


def extract_links(html: str | bytes, base_url: str) -> list[str]:
    return parse_html(html, base_url)[2]


def extract_title_text(html: str | bytes) -> tuple[str, str]:
    title, text, _ = parse_html(html, "")
    return title, text
