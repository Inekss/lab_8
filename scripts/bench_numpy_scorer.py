"""Compare Lab-3 loop scorer vs NumPy (no snippet I/O).

Writes markdown table to docs/numpy_speedup.md
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from findex.documents import CorpusReader
from findex.index import build
from findex.models import DocMeta, Index, Posting
from findex.score import BM25, accumulate_bm25, top_k_scores
from findex.search import ranked_doc_ids_loop, ranked_doc_ids_numpy
from findex.tokenizer import Tokenizer


def median_ms(fn, repeats: int = 31, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = perf_counter()
        fn()
        samples.append((perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def _content_terms(index: Index) -> list[str]:
    return [
        t
        for t in index.postings
        if t.isalpha() and len(t) > 2
    ]


def synthetic_long_postings(n: int = 50_000) -> Index:
    """One common term over many docs — shows NumPy win when lists are long."""
    doc_ids = list(range(n))
    postings = {
        "common": [Posting(i, 1 + (i % 5)) for i in doc_ids],
        "rare": [Posting(0, 1)],
    }
    lengths = {i: 100 + (i % 50) for i in doc_ids}
    meta = {i: DocMeta(f"{i}.txt", f"d{i}") for i in doc_ids}
    index = Index(postings, lengths, meta)
    index.materialize_numpy()
    return index


def main() -> int:
    corpus = ROOT / "src" / "resources"
    index = build(CorpusReader(corpus), Tokenizer(), with_positions=False)
    terms = sorted(_content_terms(index), key=lambda t: (index.df(t), t))
    rare = terms[0]
    common = terms[-1]
    queries = [
        ("common term (8-doc corpus)", common),
        ("rare term (8-doc corpus)", rare),
        ("3-term query (8-doc corpus)", "harry potter magic"),
    ]
    scorer = BM25()
    rows = []
    for label, q in queries:
        matched = len(ranked_doc_ids_loop(index, q, scorer, k=10_000))
        loop_ms = median_ms(lambda q=q: ranked_doc_ids_loop(index, q, scorer, 10))
        np_ms = median_ms(lambda q=q: ranked_doc_ids_numpy(index, q, "bm25", 10))
        speedup = loop_ms / np_ms if np_ms > 0 else float("inf")
        rows.append((label, q, matched, loop_ms, np_ms, speedup))

    # synthetic: long postings list
    big = synthetic_long_postings(50_000)
    loop_ms = median_ms(lambda: ranked_doc_ids_loop(big, "common", BM25(), 10))
    np_ms = median_ms(lambda: ranked_doc_ids_numpy(big, "common", "bm25", 10))
    rows.append(
        (
            "common term (50k synthetic docs)",
            "common",
            50_000,
            loop_ms,
            np_ms,
            loop_ms / np_ms if np_ms > 0 else float("inf"),
        )
    )

    lines = [
        "# NumPy scorer speedup (Lab 08)",
        "",
        f"Corpus: `{corpus}` · docs={index.num_docs} · terms={len(index)}",
        "",
        "| Query | Term(s) | Docs matched | Lab 3 loop (ms) | NumPy (ms) | Speedup |",
        "|---|---|---:|---:|---:|---:|",
    ]
    best = 0.0
    for label, q, matched, loop_ms, np_ms, speedup in rows:
        best = max(best, speedup)
        lines.append(
            f"| {label} | `{q}` | {matched} | {loop_ms:.3f} | {np_ms:.3f} | {speedup:.1f}x |"
        )
    lines += [
        "",
        f"Largest speedup in this table: **{best:.1f}x**.",
        "",
        "## Notes",
        "",
        "- Measured scorer + top-k only (no snippet disk I/O).",
        "- On the real 8-document corpus, posting lists are tiny: NumPy setup cost "
        "dominates and speedup can be **below 1x**. That is expected.",
        "- On a 50k-doc synthetic common term, the Lab-3 Python loop over `Posting` "
        "objects is the bottleneck and NumPy wins clearly.",
        "",
    ]
    out = ROOT / "docs" / "numpy_speedup.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} best={best:.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
