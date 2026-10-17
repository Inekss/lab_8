from __future__ import annotations

from pathlib import Path
from time import perf_counter

from findex.documents import read_text
from findex.embed import EmbeddingStore
from findex.models import Index, SearchResult
from findex.score import ScorerName, get_scorer
from findex.search import SearchMode, search as run_search
from findex.web.schemas import DocumentOut, SearchResponse, SearchResultOut

MAX_HITS = 100


def paginated_search(
    index: Index,
    query: str,
    *,
    k: int = 10,
    scorer: ScorerName = "bm25",
    page: int = 1,
    page_size: int = 10,
    mode: SearchMode = "keyword",
    embeddings: EmbeddingStore | None = None,
) -> SearchResponse:
    """Run search and return one page of results."""
    size = min(max(k, 1), page_size)
    t0 = perf_counter()
    hits = run_search(
        index,
        query,
        get_scorer(scorer),
        k=MAX_HITS,
        mode=mode,
        embeddings=embeddings,
        scorer_name=scorer,
        backend="numpy",
    )
    took_ms = (perf_counter() - t0) * 1000.0
    total = len(hits)
    start = (page - 1) * size
    window = hits[start : start + size]
    return SearchResponse(
        query=query,
        total=total,
        page=page,
        page_size=size,
        took_ms=round(took_ms, 3),
        mode=mode,
        results=[_to_out(h) for h in window],
    )


def _to_out(hit: SearchResult) -> SearchResultOut:
    return SearchResultOut(
        doc_id=hit.doc_id,
        title=hit.title,
        score=hit.score,
        snippet=hit.snippet,
    )


def get_document(index: Index, doc_id: int) -> DocumentOut | None:
    meta = index.doc_meta.get(doc_id)
    if meta is None:
        return None
    text = _load_text(meta.path)
    return DocumentOut(doc_id=doc_id, title=meta.title, path=meta.path, text=text)


def _load_text(path_str: str) -> str:
    path = Path(path_str)
    if path.is_file():
        return read_text(path)
    return ""


def highlight_snippet(snippet: str) -> str:
    """Turn Lab-3 ``[term]`` markers into HTML ``<mark>`` tags."""
    out: list[str] = []
    i = 0
    while i < len(snippet):
        if snippet[i] == "[":
            end = snippet.find("]", i + 1)
            if end > i:
                out.append("<mark>")
                out.append(_escape(snippet[i + 1 : end]))
                out.append("</mark>")
                i = end + 1
                continue
        out.append(_escape(snippet[i]))
        i += 1
    return "".join(out)


def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
