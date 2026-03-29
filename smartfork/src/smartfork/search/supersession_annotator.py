"""Supersession annotator for search results (v2.1).

Adds visual indicators and score boosts to search results based on
supersession relationships. Follows supersession chains to find the
latest session.

Key design decisions:
- ADDITIVE boost (+0.15) not multiplicative (×1.1) for clear ranking separation
- Chain following with max depth 5 to prevent infinite loops
- NEVER silently replaces results — only adds visual indicators
"""

from typing import List, Optional
from loguru import logger

from ..database.models import ResultCard
from ..database.metadata_store import MetadataStore


def get_latest_in_chain(
    session_id: str,
    store: MetadataStore,
    depth: int = 0,
) -> str:
    """Follow supersession chain to find the latest session.
    
    When multiple sessions supersede the same session, picks the one with
    highest confidence (or most recent detection if confidence is tied).
    
    Args:
        session_id: Starting session ID
        store: MetadataStore instance
        depth: Current recursion depth (prevents infinite loops)
    
    Returns:
        Session ID of the latest session in the chain
    """
    if depth > 5:  # Prevent infinite loops
        return session_id
    
    superseding = store.get_superseding_sessions(session_id)
    if not superseding:
        return session_id  # This IS the latest
    
    # Already sorted by confidence DESC, detected_at DESC
    # superseding is List[Tuple[session_id, confidence, detected_at]]
    return get_latest_in_chain(superseding[0][0], store, depth + 1)


def annotate_supersession(
    results: List[ResultCard],
    store: MetadataStore,
    boost_amount: float = 0.15,
) -> List[ResultCard]:
    """Add supersession labels and boost to search results.
    
    Uses ADDITIVE boost (not multiplicative) to ensure superseding
    sessions rank above non-superseding ones with similar scores.
    
    Boost math:
        - Original score 0.56 + 0.15 = 0.71 (beats 0.62 non-superseding)
        - Original score 0.88 + 0.15 = 1.03 (capped at 1.0)
    
    NEVER silently replaces results — only adds visual indicators.
    
    Args:
        results: List of ResultCard objects from search
        store: MetadataStore instance
        boost_amount: Additive boost for superseding sessions (default 0.15)
    
    Returns:
        Cards with updated why_matched and match_score, sorted by score DESC
    """
    for card in results:
        # Check if this session has been superseded
        superseding = store.get_superseding_sessions(card.session_id)
        if superseding:
            superseding_id = superseding[0][0][:8]
            card.why_matched += f" | ⚠️ Superseded by {superseding_id}..."
        
        # Check if this session supersedes others
        superseded = store.get_superseded_by(card.session_id)
        if superseded:
            card.why_matched += f" | 🔄 Fixes earlier attempt"
            # Additive boost: +0.15 ensures clear ranking separation
            card.match_score = min(card.match_score + boost_amount, 1.0)
    
    return sorted(results, key=lambda c: -c.match_score)
