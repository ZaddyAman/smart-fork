"""Supersession boost strategies for search result ranking.

The core problem: superseding sessions should rank higher than non-superseding
ones with similar scores, but not so high that poor matches beat good ones.

Strategy comparison:
- Multiplicative (1.1x): 0.56 -> 0.616, 0.90 -> 0.99. Weak at low scores.
- Additive (+0.15): 0.56 -> 0.71, 0.90 -> 1.05. Ceiling problem, clusters at top.
- Diminishing (0.15*(1-score)): 0.56 -> 0.626, 0.90 -> 0.915. Too weak at low scores.

Hybrid: floor boost + multiplicative
- 0.56 -> max(0.56 + 0.10, 0.56 * 1.15) = max(0.66, 0.644) = 0.66
- 0.75 -> max(0.75 + 0.10, 0.75 * 1.15) = max(0.85, 0.8625) = 0.8625
- 0.90 -> max(0.90 + 0.10, 0.90 * 1.15) = max(1.00, 1.035) = 1.0 (capped)

This guarantees a minimum boost of 0.10 while scaling with the original score.
"""


def compute_boosted_score(
    original_score: float,
    floor_boost: float = 0.10,
    multi_boost: float = 1.15,
    ceiling: float = 1.0,
) -> float:
    """Compute boosted score using floor + multiplicative strategy.
    
    Args:
        original_score: Base match score (0.0 - 1.0)
        floor_boost: Minimum absolute boost
        multi_boost: Multiplicative factor
        ceiling: Maximum score cap
    
    Returns:
        Boosted score, capped at ceiling
    
    Examples:
        >>> compute_boosted_score(0.56)  # Low score
        0.66
        >>> compute_boosted_score(0.75)  # Medium score  
        0.8625
        >>> compute_boosted_score(0.90)  # High score
        1.0
        >>> compute_boosted_score(0.95)  # Very high score
        1.0
    """
    floor_result = original_score + floor_boost
    multi_result = original_score * multi_boost
    return min(max(floor_result, multi_result), ceiling)


# Verification of the boost strategy
if __name__ == "__main__":
    print("Boost Strategy Comparison")
    print("=" * 60)
    print(f"{'Original':>10} | {'Floor+Multi':>12} | {'1.1x Mult':>10} | {'Diminishing':>12}")
    print("-" * 60)
    
    for score in [0.40, 0.50, 0.56, 0.62, 0.64, 0.75, 0.85, 0.90, 0.95]:
        hybrid = compute_boosted_score(score)
        mult_1_1 = min(score * 1.1, 1.0)
        diminishing = min(score + 0.15 * (1 - score), 1.0)
        
        print(f"{score:>10.2f} | {hybrid:>12.4f} | {mult_1_1:>10.4f} | {diminishing:>12.4f}")
    
    print("\nKey insight: At score 0.56, hybrid gives +0.10 (0.66)")
    print("This beats a non-superseding session at 0.64 by 0.02")
    print("Old diminishing: 0.56 → 0.626, loses to 0.64 by 0.014")
