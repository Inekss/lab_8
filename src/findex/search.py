from __future__ import annotations

import functools
import logging
import re
from pathlib import Path
from time import perf_counter
from typing import Literal, Optional

import numpy as np

from findex.deco import log, timed
from findex.documents import read_text
from findex.embed import EmbeddingStore, encode_query
from findex.index import OUT_PKL
from findex.models import Index, SearchResult
from findex.query import parse, query_terms
from findex.rrf import reciprocal_rank_fusion
from findex.score import (
    BM25,
    TfIdf,
    accumulate_bm25,
    accumulate_tfidf,
    top_k_scores,
)
from findex.store import open_index

QUERY = '(quirrell OR buckbeak) AND harry NOT lockhart "privet drive"'
K = 5
# hand labels: the book that query is about, must show up in the top 5
EVAL = (
    ("quirrell", "harry_potter_1"),
    ("flamel", "harry_potter_1"),
    ("firenze", "harry_potter_1"),
    ("basilisk", "harry_potter_2"),
    ("lockhart", "harry_potter_2"),
    ("dobby", "harry_potter_2"),
    ('"chamber of secrets"', "harry_potter_2"),
    ("buckbeak", "harry_potter_3"),
    ("lupin", "harry_potter_3"),
    ("pettigrew", "harry_potter_3"),
)

EVAL_PARAPHRASE = (
    ("turbaned teacher who shared a body with the dark lord", "harry_potter_1"),
    ("alchemist who created a stone that grants endless life", "harry_potter_1"),
    ("enormous serpent hidden in the school's underground pipes", "harry_potter_2"),
    ("tiny magical servant who stole letters and warned of a plot", "harry_potter_2"),
    ("innocent half-eagle creature sentenced to be killed by officials", "harry_potter_3"),
)

ENGINE = "merge"  # merge | set
BENCH_REPEATS = 200000

SearchMode = Literal["keyword", "semantic", "hybrid"]
ScoreBackend = Literal["loop", "numpy"]


def merge_and(a: list[int], b: list[int]) -> list[int]:
    i = j = 0
    out: list[int] = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def merge_or(a: list[int], b: list[int]) -> list[int]:
    i = j = 0
    out: list[int] = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def merge_not(a: list[int], b: list[int]) -> list[int]:
    i = j = 0
    out: list[int] = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            j += 1
    out.extend(a[i:])
    return out


def set_and(a: list[int], b: list[int]) -> list[int]:
    return sorted(set(a) & set(b))


def set_or(a: list[int], b: list[int]) -> list[int]:
    return sorted(set(a) | set(b))


def set_not(a: list[int], b: list[int]) -> list[int]:
    return sorted(set(a) - set(b))


_OPS = {
    "merge": {"AND": merge_and, "OR": merge_or, "NOT": merge_not},
    "set": {"AND": set_and, "OR": set_or, "NOT": set_not},
}


def _ids(index: Index, term: str) -> list[int]:
    return [p.doc_id for p in index.postings.get(term, ())]


def boolean_search(index: Index, query: str, engine: str) -> list[int]:
    universe = sorted(index.doc_meta)
    acc: Optional[list[int]] = None
    op = "AND"
    fn = _OPS[engine]
    for raw in query.split():
        word = raw.upper()
        if word in ("AND", "OR", "NOT"):
            op = word
            continue
        docs = _ids(index, raw.lower())
        if acc is None:
            acc = fn["NOT"](universe, docs) if op == "NOT" else docs
        else:
            acc = fn[op](acc, docs)
        op = "AND"
    return [] if acc is None else acc


def _freq(index: Index, term: str) -> int:
    return sum(p.tf for p in index.postings[term])


def bench(index: Index) -> None:
    if len(index.postings) < 2:
        print("bench skipped")
        return
    ranked = sorted(index.postings, key=lambda t: (_freq(index, t), t))
    pairs = (("rare", ranked[:2]), ("common", ranked[-2:]))
    for label, (left, right) in pairs:
        query = f"{left} {right}"
        for engine in ("merge", "set"):
            t0 = perf_counter()
            for _ in range(BENCH_REPEATS):
                boolean_search(index, query, engine)
            avg = (perf_counter() - t0) / BENCH_REPEATS
            print(
                f"{label} {left!r} tf={_freq(index, left)} "
                f"{right!r} tf={_freq(index, right)} {engine} {avg:.8f}s"
            )


def _matcher(index: Index):
    fn = getattr(index, "_match_ids", None)
    if fn is not None:
        return fn

    @functools.lru_cache(maxsize=256)
    def match_ids(query: str) -> tuple[int, ...]:
        log.debug("cache miss %s", query)
        return tuple(sorted(parse(query).evaluate(index)))

    index._match_ids = match_ids
    return match_ids


_TEXT_CACHE: dict[str, str] = {}


def _doc_text(path: Path) -> str:
    key = str(path)
    text = _TEXT_CACHE.get(key)
    if text is None:
        text = read_text(path) if path.is_file() else ""
        _TEXT_CACHE[key] = text
    return text


def _snippet(index: Index, doc_id: int, terms: list[str]) -> str:
    path = Path(index.doc_meta[doc_id].path)
    text = _doc_text(path)
    if not text:
        return ""
    folded = text.lower()
    at = -1
    hit = ""
    for term in sorted(set(terms), key=lambda t: (index.df(t) or 10**9, t)):
        found = folded.find(term)
        if found >= 0:
            at, hit = found, term
            break
    if at < 0:
        window = text[:160]
    else:
        window = text[max(0, at - 80) : at + len(hit) + 80]
    for term in sorted(set(terms), key=len, reverse=True):
        if not term:
            continue
        window = re.sub(
            rf"(?i)(?<!\w){re.escape(term)}(?!\w)",
            lambda m: f"[{m.group(0)}]",
            window,
        )
    return window.replace("\n", " ")


def _score_size(index: Index) -> int:
    return int(index.doc_len_arr.shape[0])


def score_query_loop(
    index: Index,
    query: str,
    scorer,
    *,
    candidates: set[int] | None = None,
) -> dict[int, float]:
    """Lab-3 Python loop over Posting objects."""
    node = parse(query)
    terms = query_terms(node)
    if candidates is None:
        candidates = set(_matcher(index)(query))
    scores: dict[int, float] = {}
    for term in terms:
        if term not in index:
            continue
        for posting in index[term]:
            if posting.doc_id not in candidates:
                continue
            scores[posting.doc_id] = scores.get(posting.doc_id, 0.0) + scorer(
                term, posting, index
            )
    return scores


def score_query_numpy(
    index: Index,
    query: str,
    scorer_name: str = "bm25",
    *,
    candidates: set[int] | None = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized BM25 / TF-IDF → ``(scores, touched)`` dense arrays."""
    index.materialize_numpy()
    node = parse(query)
    terms = query_terms(node)
    if candidates is None:
        candidates = set(_matcher(index)(query))
    n = _score_size(index)
    scores = np.zeros(n, dtype=np.float64)
    touched = np.zeros(n, dtype=bool)
    for term in terms:
        if term not in index:
            continue
        if scorer_name == "tfidf":
            accumulate_tfidf(
                scores, index, term, candidates=candidates, touched=touched
            )
        else:
            accumulate_bm25(
                scores, index, term, k1=k1, b=b, candidates=candidates, touched=touched
            )
    return scores, touched


def ranked_doc_ids_loop(
    index: Index,
    query: str,
    scorer,
    k: int,
) -> list[tuple[int, float]]:
    scores = score_query_loop(index, query, scorer)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
    return [(doc_id, score) for doc_id, score in ordered]


def ranked_doc_ids_numpy(
    index: Index,
    query: str,
    scorer_name: str,
    k: int,
) -> list[tuple[int, float]]:
    scores, touched = score_query_numpy(index, query, scorer_name)
    return top_k_scores(scores, k, touched=touched)


def semantic_search(
    index: Index,
    query: str,
    store: EmbeddingStore,
    *,
    k: int = 10,
    with_snippet: bool = True,
) -> list[SearchResult]:
    q = encode_query(query, store.model_name)
    # mmap @ vector may copy a row slice; still much cheaper than torch
    sims = np.asarray(store.vectors @ q, dtype=np.float64)
    doc_ids = np.asarray(store.chunk_doc_ids, dtype=np.int32)
    # best chunk score per document (vectorized scatter-max)
    n_docs = int(doc_ids.max()) + 1 if doc_ids.size else 0
    best = np.full(n_docs, -np.inf, dtype=np.float64)
    np.maximum.at(best, doc_ids, sims)
    scored = np.flatnonzero(np.isfinite(best) & (best > -np.inf))
    if scored.size == 0:
        return []
    if scored.size <= k:
        chosen = scored
    else:
        part = np.argpartition(best[scored], -k)[-k:]
        chosen = scored[part]
    order = np.lexsort((chosen, -best[chosen]))
    chosen = chosen[order]
    out: list[SearchResult] = []
    for doc_id in chosen:
        did = int(doc_id)
        if did not in index.doc_meta:
            continue
        snip = _snippet(index, did, []) if with_snippet else ""
        out.append(SearchResult(float(best[did]), did, index.doc_meta[did].title, snip))
    return out


@timed
def search(
    index: Index,
    query: str,
    scorer=None,
    k: int = 10,
    *,
    backend: ScoreBackend = "numpy",
    mode: SearchMode = "keyword",
    embeddings: EmbeddingStore | None = None,
    scorer_name: str = "bm25",
    rrf_k: int = 60,
    with_snippet: bool = True,
) -> list[SearchResult]:
    if scorer is None:
        scorer = BM25() if scorer_name == "bm25" else TfIdf()

    if mode == "semantic":
        if embeddings is None:
            raise ValueError("semantic mode requires embeddings")
        return semantic_search(
            index, query, embeddings, k=k, with_snippet=with_snippet
        )

    if mode == "hybrid":
        if embeddings is None:
            raise ValueError("hybrid mode requires embeddings")
        kw = search(
            index,
            query,
            scorer,
            k=max(k, 20),
            backend=backend,
            mode="keyword",
            scorer_name=scorer_name,
            with_snippet=False,
        )
        sem = semantic_search(
            index, query, embeddings, k=max(k, 20), with_snippet=False
        )
        fused = reciprocal_rank_fusion(
            [[h.doc_id for h in kw], [h.doc_id for h in sem]],
            k=rrf_k,
        )[:k]
        node = parse(query)
        terms = query_terms(node)
        out: list[SearchResult] = []
        for doc_id, score in fused:
            if doc_id not in index.doc_meta:
                continue
            snip = _snippet(index, doc_id, terms) if with_snippet else ""
            out.append(SearchResult(score, doc_id, index.doc_meta[doc_id].title, snip))
        return out

    # keyword
    name = scorer_name
    if isinstance(scorer, TfIdf):
        name = "tfidf"
    elif isinstance(scorer, BM25):
        name = "bm25"

    fn = _matcher(index)
    before = fn.cache_info().hits
    _ = fn(query)
    if fn.cache_info().hits > before:
        log.debug("cache hit %s", query)

    if backend == "loop":
        ranked = ranked_doc_ids_loop(index, query, scorer, k)
    else:
        ranked = ranked_doc_ids_numpy(index, query, name, k)

    node = parse(query)
    terms = query_terms(node)
    results: list[SearchResult] = []
    for doc_id, score in ranked:
        snip = _snippet(index, doc_id, terms) if with_snippet else ""
        results.append(SearchResult(score, doc_id, index.doc_meta[doc_id].title, snip))
    return results


def precision_at_5(
    index: Index,
    *,
    embeddings: EmbeddingStore | None = None,
    modes: tuple[SearchMode, ...] = ("keyword",),
) -> None:
    pairs = EVAL + (EVAL_PARAPHRASE if embeddings is not None else ())
    header = f"{'query':<56}" + "".join(f" {m:>10}" for m in modes)
    print(header)
    totals = {m: 0.0 for m in modes}
    for query, title in pairs:
        row = []
        for mode in modes:
            hits = search(
                index,
                query,
                BM25(),
                K,
                mode=mode,
                embeddings=embeddings,
                with_snippet=False,
            )
            p = sum(h.title == title for h in hits) / K
            totals[mode] += p
            row.append(p)
        print(f"{query:<56}" + "".join(f" {p:10.2f}" for p in row))
    n = len(pairs)
    print(f"{'mean':<56}" + "".join(f" {totals[m] / n:10.2f}" for m in modes))


def _show(index: Index, query: str, scorer) -> None:
    print(f"\n{query} ({type(scorer).__name__})")
    for rank, hit in enumerate(search(index, query, scorer, K), 1):
        print(f"{rank} {hit.score:.4f} {hit.title} {hit.snippet}")


def main() -> None:
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    with open_index(OUT_PKL) as index:
        _show(index, QUERY, BM25())
        _show(index, QUERY, BM25())
        print("cache", index._match_ids.cache_info())


if __name__ == "__main__":
    main()
