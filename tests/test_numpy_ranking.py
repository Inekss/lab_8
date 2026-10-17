"""NumPy scorer must match Lab-3 Python-loop ranking."""
from __future__ import annotations

from pathlib import Path

import pytest

from findex.models import DocMeta, Index, Posting
from findex.score import BM25, TfIdf
from findex.search import ranked_doc_ids_loop, ranked_doc_ids_numpy


def _index() -> Index:
    # lengths differ so BM25 length norm matters
    postings = {
        "alpha": [Posting(0, 3), Posting(1, 1), Posting(2, 2)],
        "beta": [Posting(0, 1), Posting(2, 5)],
        "gamma": [Posting(1, 4), Posting(3, 1)],
        "delta": [Posting(2, 1)],
    }
    lengths = {0: 10, 1: 40, 2: 20, 3: 5}
    meta = {i: DocMeta(f"{i}.txt", f"doc{i}") for i in range(4)}
    index = Index(postings, lengths, meta)
    index.materialize_numpy()
    return index


@pytest.mark.parametrize(
    "query",
    ["alpha", "beta", "alpha beta", "gamma", "alpha OR gamma", "delta"],
)
@pytest.mark.parametrize("scorer_name", ["bm25", "tfidf"])
def test_numpy_matches_loop_ranking(query: str, scorer_name: str) -> None:
    index = _index()
    scorer = BM25() if scorer_name == "bm25" else TfIdf()
    k = 5
    loop = ranked_doc_ids_loop(index, query, scorer, k)
    numpy = ranked_doc_ids_numpy(index, query, scorer_name, k)
    assert [d for d, _ in numpy] == [d for d, _ in loop]
    for (_, a), (_, b) in zip(numpy, loop):
        assert a == pytest.approx(b, rel=1e-9, abs=1e-9)


def test_numpy_matches_on_resource_corpus() -> None:
    corpus = Path(__file__).resolve().parents[1] / "src" / "resources"
    if not corpus.is_dir() or not any(corpus.glob("*.txt")):
        pytest.skip("corpus missing")
    from findex.documents import CorpusReader
    from findex.index import build
    from findex.tokenizer import Tokenizer

    index = build(CorpusReader(corpus), Tokenizer(), with_positions=False)
    for query in ("harry", "quirrell", "harry potter magic"):
        loop = ranked_doc_ids_loop(index, query, BM25(), 5)
        numpy = ranked_doc_ids_numpy(index, query, "bm25", 5)
        assert [d for d, _ in numpy] == [d for d, _ in loop]
