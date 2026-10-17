from __future__ import annotations

import math
from typing import Literal, Protocol

import numpy as np

from findex.models import Index, Posting

type ScorerName = Literal["bm25", "tfidf"]


class Scorer(Protocol):
    def __call__(self, term: str, posting: Posting, index: Index) -> float: ...


class TfIdf:
    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        idf = math.log(index.num_docs / index.df(term))
        return (1 + math.log(posting.tf)) * idf


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        idf = math.log(index.num_docs / index.df(term))
        dl = index.doc_length(posting.doc_id)
        norm = 1 - self.b + self.b * dl / index.avg_doc_length
        tf = posting.tf
        return idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)


def get_scorer(name: ScorerName) -> Scorer:
    if name == "tfidf":
        return TfIdf()
    return BM25()


def accumulate_bm25(
    scores: np.ndarray,
    index: Index,
    term: str,
    *,
    k1: float = 1.5,
    b: float = 0.75,
    candidates: set[int] | None = None,
    touched: np.ndarray | None = None,
) -> None:
    """Add BM25 contribution for ``term`` into dense ``scores`` (Lab-3 formula)."""
    doc_ids, tfs = index.np_postings(term)
    if doc_ids.size == 0:
        return
    if candidates is not None:
        cand = np.fromiter(candidates, dtype=np.int32, count=len(candidates))
        mask = np.isin(doc_ids, cand)
        if not mask.any():
            return
        doc_ids = doc_ids[mask]
        tfs = tfs[mask]
    idf = math.log(index.num_docs / index.df(term))
    tfs_f = tfs.astype(np.float64)
    doc_len = index.doc_len_arr[doc_ids].astype(np.float64)
    norm = k1 * (1.0 - b + b * doc_len / index.avg_doc_length)
    scores[doc_ids] += idf * tfs_f * (k1 + 1.0) / (tfs_f + norm)
    if touched is not None:
        touched[doc_ids] = True


def accumulate_tfidf(
    scores: np.ndarray,
    index: Index,
    term: str,
    *,
    candidates: set[int] | None = None,
    touched: np.ndarray | None = None,
) -> None:
    doc_ids, tfs = index.np_postings(term)
    if doc_ids.size == 0:
        return
    if candidates is not None:
        cand = np.fromiter(candidates, dtype=np.int32, count=len(candidates))
        mask = np.isin(doc_ids, cand)
        if not mask.any():
            return
        doc_ids = doc_ids[mask]
        tfs = tfs[mask]
    idf = math.log(index.num_docs / index.df(term))
    tfs_f = tfs.astype(np.float64)
    scores[doc_ids] += (1.0 + np.log(tfs_f)) * idf
    if touched is not None:
        touched[doc_ids] = True


def top_k_scores(
    scores: np.ndarray,
    k: int,
    *,
    touched: np.ndarray | None = None,
) -> list[tuple[int, float]]:
    """Return ``[(doc_id, score), ...]`` best-first via ``np.argpartition``.

    Tie-break: lower ``doc_id`` first (same as stable Lab-3 path).
    ``touched`` marks docs that received a posting (even when idf=0 → score 0).
    """
    if touched is not None:
        nz = np.flatnonzero(touched)
    else:
        nz = np.flatnonzero(scores)
    if nz.size == 0:
        return []
    if nz.size <= k:
        chosen = nz
    else:
        part = np.argpartition(scores[nz], -k)[-k:]
        chosen = nz[part]
    order = np.lexsort((chosen, -scores[chosen]))
    chosen = chosen[order]
    return [(int(i), float(scores[i])) for i in chosen]


def sanity(index: Index) -> None:
    rare, common = "quirrell", "norbert"
    bm = BM25()
    print(
        f"rare {rare} df={index.df(rare)} w={bm(rare, Posting(0, 1), index):.4f}  "
        f"common {common} df={index.df(common)} w={bm(common, Posting(0, 1), index):.4f}"
    )
    for tf in (1, 2, 20, 21):
        w = bm(rare, Posting(0, tf), index)
        before = bm(rare, Posting(0, tf - 1), index) if tf > 1 else 0.0
        print(f"tf={tf} w={w:.4f} step={w - before:.4f}")
    short = min(index.doc_lengths, key=lambda doc_id: index.doc_lengths[doc_id])
    long = max(index.doc_lengths, key=lambda doc_id: index.doc_lengths[doc_id])
    flat = BM25(b=0)
    for doc_id in (short, long):
        one = Posting(doc_id, 1)
        print(
            f"len={index.doc_length(doc_id)} {index.doc_meta[doc_id].title} "
            f"b0.75={bm(rare, one, index):.4f} b0={flat(rare, one, index):.4f}"
        )
