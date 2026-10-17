from findex.rrf import reciprocal_rank_fusion


def test_rrf_prefers_docs_high_in_both() -> None:
    a = [10, 20, 30]
    b = [20, 10, 40]
    fused = reciprocal_rank_fusion([a, b], k=60)
    ids = [doc_id for doc_id, _ in fused]
    assert ids[0] in (10, 20)
    assert set(ids) == {10, 20, 30, 40}


def test_rrf_scores_decrease() -> None:
    fused = reciprocal_rank_fusion([[1, 2, 3]], k=60)
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)
