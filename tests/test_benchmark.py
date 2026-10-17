"""pytest-benchmark: catch regressions on search / build_index."""
from __future__ import annotations

from pathlib import Path

import pytest

from findex.documents import CorpusReader
from findex.index import build
from findex.models import DocMeta, Index, Posting
from findex.score import BM25
from findex.search import search
from findex.tokenizer import Tokenizer


def _tiny() -> Index:
    postings = {
        "harry": [Posting(0, 2), Posting(1, 1)],
        "potter": [Posting(0, 1), Posting(1, 1)],
        "magic": [Posting(1, 3)],
    }
    index = Index(
        postings,
        {0: 5, 1: 8},
        {0: DocMeta("0.txt", "a"), 1: DocMeta("1.txt", "b")},
    )
    index.materialize_numpy()
    return index


def test_bench_search(benchmark) -> None:
    index = _tiny()
    scorer = BM25()

    def run():
        return search(index, "harry potter", scorer, k=5, with_snippet=False)

    result = benchmark(run)
    assert len(result) >= 1


def test_bench_build_index(benchmark, tmp_path: Path) -> None:
    # tiny synthetic corpus so CI stays fast
    for i, text in enumerate(["alpha beta gamma", "beta delta", "gamma harry potter"]):
        (tmp_path / f"d{i}.txt").write_text(text, encoding="utf-8")

    def run():
        return build(CorpusReader(tmp_path), Tokenizer(), with_positions=False)

    index = benchmark(run)
    assert index.num_docs == 3
