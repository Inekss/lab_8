from __future__ import annotations

import functools
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class StatsResult:
    n_docs: int
    n_tokens: int
    vocab_size: int
    top50: list[tuple[str, int]]
    elapsed_s: float
    peak_kib: float


@dataclass(frozen=True)
class PostingPlain:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    path: str
    title: str


@dataclass(frozen=True, slots=True)
class PartialIndex:
    postings: dict[str, list[Posting]]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]


@dataclass(order=True, frozen=True, slots=True)
class SearchResult:
    score: float
    doc_id: int
    title: str = field(compare=False)
    snippet: str = field(compare=False)


class Index(Mapping[str, list[Posting]]):
    # not frozen: cached_property / numpy arrays write onto the instance
    def __init__(
        self,
        postings: dict[str, list[Posting]],
        doc_lengths: dict[int, int],
        doc_meta: dict[int, DocMeta],
    ) -> None:
        self.postings = postings
        self.doc_lengths = doc_lengths
        self.doc_meta = doc_meta
        self._match_ids: Optional[Callable[[str], tuple[int, ...]]] = None
        self._doc_len_arr: Any = None
        self._np_doc_ids: Optional[dict[str, Any]] = None
        self._np_tfs: Optional[dict[str, Any]] = None

    def __getitem__(self, term: str) -> list[Posting]:
        return self.postings[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self.postings)

    def __len__(self) -> int:
        return len(self.postings)

    def __repr__(self) -> str:
        return f"Index(terms={len(self):_}, docs={self.num_docs:_})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Index):
            return False
        return (
            self.postings == other.postings
            and self.doc_lengths == other.doc_lengths
            and self.doc_meta == other.doc_meta
        )

    @property
    def num_docs(self) -> int:
        return len(self.doc_lengths)

    @functools.cached_property
    def avg_doc_length(self) -> float:
        return sum(self.doc_lengths.values()) / self.num_docs

    def doc_length(self, doc_id: int) -> int:
        return self.doc_lengths[doc_id]

    def df(self, term: str) -> int:
        return len(self[term]) if term in self else 0

    def materialize_numpy(self) -> None:
        """Build int32 posting / doc-length arrays (idempotent)."""
        # old pickles predate these attrs
        if not hasattr(self, "_doc_len_arr"):
            self._doc_len_arr = None
            self._np_doc_ids = None
            self._np_tfs = None
        if self._doc_len_arr is not None:
            return
        import numpy as np

        max_id = max(self.doc_lengths) if self.doc_lengths else -1
        doc_len = np.zeros(max_id + 1, dtype=np.int32)
        for doc_id, length in self.doc_lengths.items():
            doc_len[doc_id] = length
        np_doc_ids: dict[str, np.ndarray] = {}
        np_tfs: dict[str, np.ndarray] = {}
        for term, plist in self.postings.items():
            np_doc_ids[term] = np.fromiter(
                (p.doc_id for p in plist), dtype=np.int32, count=len(plist)
            )
            np_tfs[term] = np.fromiter((p.tf for p in plist), dtype=np.int32, count=len(plist))
        self._doc_len_arr = doc_len
        self._np_doc_ids = np_doc_ids
        self._np_tfs = np_tfs

    @property
    def doc_len_arr(self) -> Any:
        self.materialize_numpy()
        assert self._doc_len_arr is not None
        return self._doc_len_arr

    def np_postings(self, term: str) -> tuple[Any, Any]:
        import numpy as np

        self.materialize_numpy()
        assert self._np_doc_ids is not None and self._np_tfs is not None
        empty = np.empty(0, dtype=np.int32)
        return (
            self._np_doc_ids.get(term, empty),
            self._np_tfs.get(term, empty),
        )
