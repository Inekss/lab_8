from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    *,
    k: int = 60,
) -> list[tuple[int, float]]:
    """Fuse ranked doc-id lists with RRF: score = Σ 1/(k + rank).

    Returns ``[(doc_id, score), ...]`` best-first. Ties broken by lower doc_id.
    """
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
