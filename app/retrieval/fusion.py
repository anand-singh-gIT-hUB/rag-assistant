"""
app/retrieval/fusion.py
────────────────────────
Reciprocal Rank Fusion (RRF) for merging dense and sparse retrieval results.

Formula: RRF_score(d) = Σ 1 / (k + rank(d))
where k=60 is a smoothing constant that reduces the impact of high ranks.

Reference: Cormack, Clarke & Buettcher (SIGIR 2009).
"""
from __future__ import annotations

from typing import Any


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """
    Merge two ranked lists with RRF.

    Args:
        dense_results:  Ranked list from the dense (semantic) retriever.
        sparse_results: Ranked list from the sparse (BM25) retriever.
        k:              RRF smoothing constant (default 60, per the original paper).

    Returns:
        A single deduplicated list sorted by RRF score descending.
        Each item retains the original chunk data; an extra ``rrf_score`` key
        is added and the ``score`` key is overwritten with the RRF value.
    """
    # chunk_id → accumulated RRF score
    rrf_scores: dict[str, float] = {}
    # chunk_id → chunk dict (keep first-seen copy; metadata is identical)
    chunks_by_id: dict[str, dict[str, Any]] = {}

    for ranked_list in (dense_results, sparse_results):
        for rank, chunk in enumerate(ranked_list, start=1):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in chunks_by_id:
                chunks_by_id[cid] = dict(chunk)  # shallow copy

    # Attach RRF score and sort
    merged: list[dict[str, Any]] = []
    for cid, rrf_score in rrf_scores.items():
        chunk = chunks_by_id[cid]
        chunk["score"] = rrf_score        # overwrite with fused score
        chunk["rrf_score"] = rrf_score    # extra key for transparency
        merged.append(chunk)

    merged.sort(key=lambda x: x["rrf_score"], reverse=True)
    return merged
