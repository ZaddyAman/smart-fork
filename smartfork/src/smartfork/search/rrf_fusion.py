"""Reciprocal Rank Fusion (RRF) for combining multiple ranked lists (v2).

RRF combines BM25 and vector search rankings into a single unified ranking.
It only uses rank positions, not raw scores — so it's immune to the scale
differences between BM25 scores and cosine similarity scores.

Formula: RRF_score(session) = Σ 1/(k + rank_i)
where k = 60 (standard constant), rank_i = position in each signal's list

Why RRF over weighted average (v1 approach):
- No score normalization needed (BM25 and cosine are on different scales)
- Robust to outliers (one signal can't dominate by having larger values)
- Proven to outperform linear combination in IR benchmarks
- Simple, no hyperparameter tuning beyond k
"""

from typing import List, Tuple, Dict


def rrf_fuse(rankings: List[List[Tuple[str, float]]],
             k: int = 60,
             top_n: int = 10) -> List[Tuple[str, float]]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion.
    
    Args:
        rankings: List of ranked lists. Each list contains (session_id, score) tuples
                  sorted by score descending. The score values are ignored —
                  only the rank position matters.
        k: RRF constant. Standard value is 60. Higher values reduce the
           influence of high-ranking items.
        top_n: Number of results to return
    
    Returns:
        Unified ranked list: [(session_id, rrf_score), ...] sorted by RRF score descending
    
    Example:
        bm25_results = [("s1", 5.2), ("s2", 3.1), ("s3", 1.0)]
        vector_results = [("s2", 0.95), ("s1", 0.87), ("s4", 0.72)]
        fused = rrf_fuse([bm25_results, vector_results])
        # s2 is rank 1 in vector, rank 2 in BM25 → highest RRF
        # s1 is rank 1 in BM25, rank 2 in vector → also high RRF
    """
    rrf_scores: Dict[str, float] = {}
    
    for ranking in rankings:
        for rank, (session_id, _score) in enumerate(ranking, start=1):
            if session_id not in rrf_scores:
                rrf_scores[session_id] = 0.0
            rrf_scores[session_id] += 1.0 / (k + rank)
    
    # Sort by RRF score descending
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_n]


def rrf_fuse_weighted(rankings: List[List[Tuple[str, float]]],
                       weights: List[float] = None,
                       k: int = 60,
                       top_n: int = 10) -> List[Tuple[str, float]]:
    """Weighted RRF — allows boosting certain signals.
    
    Same as rrf_fuse but each ranking list gets a weight multiplier.
    Useful for intent-aware boosting (e.g., boost reasoning_docs for
    decision_hunting queries).
    
    Args:
        rankings: List of ranked lists
        weights: Weight for each ranking list (default: all 1.0)
        k: RRF constant
        top_n: Results to return
    
    Returns:
        Weighted unified ranked list
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    
    if len(weights) != len(rankings):
        raise ValueError(f"Expected {len(rankings)} weights, got {len(weights)}")
    
    rrf_scores: Dict[str, float] = {}
    
    for weight, ranking in zip(weights, rankings):
        for rank, (session_id, _score) in enumerate(ranking, start=1):
            if session_id not in rrf_scores:
                rrf_scores[session_id] = 0.0
            rrf_scores[session_id] += weight * (1.0 / (k + rank))
    
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_n]
