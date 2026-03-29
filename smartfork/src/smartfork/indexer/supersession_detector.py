"""Supersession detection for SmartFork v2.1.

Detects when newer sessions correct or supersede older sessions.
Uses pre-stored embeddings from ChromaDB for zero Ollama calls at detection time.

Key design decisions:
- Domain-specific similarity thresholds (0.85 for project-scoped sessions)
- Sentence-boundary negation handling for resolution detection
- Edge case guards: unknown_project, short tasks, self-supersession
"""

import re
from typing import List, Tuple, Dict, Optional
import numpy as np
from loguru import logger

from ..database.models import SessionDocument


# ═══════════════════════════════════════════════════════════════════════════════
# SIMILARITY THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

# Sessions in the same project share context, so need higher thresholds
SIMILARITY_THRESHOLDS = {
    "error_recall":     0.82,  # Bugs need high precision
    "decision_hunting": 0.80,  # Decisions are semantically distinct
    "default":          0.85,  # Conservative default for project-scoped
}


# ═══════════════════════════════════════════════════════════════════════════════
# RESOLUTION DETECTION PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

# Resolution signals requiring object context (reduces false positives)
RESOLUTION_PATTERNS: List[Tuple[str, float]] = [
    (r'\bfixed\s+(?:the|this|that|it|bug|error|issue|problem|crash)', 1.0),
    (r'\bresolved\s+(?:the|this|that|it|bug|error|issue|problem)', 1.0),
    (r'\bworking\s+now\b', 1.0),
    (r'\bsuccessfully\s+(?:implemented|completed|fixed|resolved|tested)\b', 1.0),
    (r'\bnow\s+works?\b', 1.0),
    (r'\bpasses?\s+(?:the|all)\s+tests?\b', 1.0),
]

# Problem signals
PROBLEM_PATTERNS: List[Tuple[str, float]] = [
    (r'\bstill\s+broken\b', 1.0),
    (r'\bnot\s+working\b', 1.0),
    (r'\bfailed\s+(?:to|again)\b', 1.0),
    (r'\bissue\s+persists\b', 1.0),
    (r'\bcouldn\'?t\s+(?:fix|resolve|implement)\b', 1.0),
    (r'\bunable\s+to\s+(?:fix|resolve|implement)\b', 1.0),
    (r'\bTODO\b', 0.5),
    (r'\bneed\s+to\s+(?:fix|resolve|implement)\b', 0.5),
]

# Negation words for sentence-level checking
NEGATION_WORDS = {"not", "no", "without", "isn't", "don't", "doesn't", "wasn't", "hasn't", "never"}

# Error keywords for counting in task_raw
ERROR_KEYWORDS = ["error", "bug", "fix", "crash", "broken", "failed", "traceback"]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
    
    Returns:
        Cosine similarity score between 0.0 and 1.0
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using period/question/exclamation boundaries.
    
    Args:
        text: Text to split
    
    Returns:
        List of sentence strings
    """
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def is_negated_in_sentence(keyword: str, sentence: str) -> bool:
    """Check if keyword is negated within its sentence.
    
    Uses sentence boundaries instead of character offset for robustness.
    
    Args:
        keyword: The keyword to check (e.g., "error", "bug")
        sentence: The sentence containing the keyword
    
    Returns:
        True if keyword is negated in this sentence
    """
    idx = sentence.lower().find(keyword.lower())
    if idx == -1:
        return False
    
    # Get text before keyword in same sentence
    prefix = sentence[:idx].split()[-3:]  # Last 3 words before keyword
    return any(w in NEGATION_WORDS for w in prefix)


def sentence_has_negation_before_phrase(sentence: str, phrase: str) -> bool:
    """Check if a phrase is preceded by negation in the same sentence.
    
    Handles: "not working now", "isn't fixed yet", etc.
    
    Args:
        sentence: The sentence to check
        phrase: The phrase to check for negation before
    
    Returns:
        True if phrase is negated
    """
    idx = sentence.find(phrase.lower())
    if idx == -1:
        return False
    
    # Get text before phrase
    prefix = sentence[:idx].split()[-3:]  # Last 3 words before phrase
    return any(w in NEGATION_WORDS for w in prefix)


def load_embeddings_from_chromadb(vector_index, project_name: str) -> Dict[str, np.ndarray]:
    """Load task embeddings from ChromaDB for a specific project.
    
    No pagination needed for typical project sizes (< 1000 sessions).
    Memory: 200 sessions * 512 dims * 4 bytes = 400KB (trivial).
    
    Args:
        vector_index: VectorIndex instance with ChromaDB collections
        project_name: Project to load embeddings for
    
    Returns:
        Dict mapping session_id to embedding vector
    """
    try:
        results = vector_index.task_collection.get(
            where={"project_name": project_name},
            include=["embeddings", "metadatas"]
        )
        embeddings = {}
        for meta, emb in zip(results.get("metadatas", []), results.get("embeddings", [])):
            if meta and emb:
                session_id = meta.get("session_id")
                if session_id:
                    embeddings[session_id] = np.array(emb, dtype=np.float32)
        return embeddings
    except Exception as e:
        logger.warning(f"Failed to load embeddings from ChromaDB: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# SUPERSESSION DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_supersession(
    new_session: SessionDocument,
    new_embedding: np.ndarray,
    existing_sessions: List[SessionDocument],
    stored_embeddings: Dict[str, np.ndarray],
    domain_hint: str = "default",
) -> List[Tuple[str, float]]:
    """Detect which sessions are superseded by the new session.
    
    Uses pre-stored embeddings from ChromaDB for zero Ollama calls.
    Applies domain-specific similarity thresholds for project-scoped sessions.
    
    Args:
        new_session: The new session being indexed
        new_embedding: Embedding vector for the new session
        existing_sessions: List of existing SessionDocuments
        stored_embeddings: Dict mapping session_id to embedding vector
        domain_hint: Intent type for threshold selection
    
    Returns:
        List of (superseded_session_id, confidence) pairs, sorted by confidence DESC
    """
    # Edge case: skip unknown_project sessions (can't reliably detect without project context)
    if new_session.project_name == "unknown_project":
        return []
    
    # Edge case: skip very short task descriptions (high-variance embeddings)
    if len(new_session.task_raw.split()) < 5:
        return []
    
    threshold = SIMILARITY_THRESHOLDS.get(domain_hint, SIMILARITY_THRESHOLDS["default"])
    
    # Filter candidates: same project, overlapping domains, older timestamp, different session
    candidates = [
        s for s in existing_sessions
        if s.project_name == new_session.project_name
        and s.session_id != new_session.session_id  # Edge case: self-supersession guard
        and set(s.domains) & set(new_session.domains)
        and s.session_start < new_session.session_start
        and s.session_id in stored_embeddings
    ]
    
    links = []
    for candidate in candidates:
        similarity = cosine_similarity(new_embedding, stored_embeddings[candidate.session_id])
        if similarity > threshold:
            links.append((candidate.session_id, float(similarity)))
    
    return sorted(links, key=lambda x: -x[1])


# ═══════════════════════════════════════════════════════════════════════════════
# RESOLUTION STATUS DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def count_error_signals(text: str) -> int:
    """Count error-related keywords in text with sentence-boundary negation handling.
    
    Args:
        text: Text to analyze (usually task_raw)
    
    Returns:
        Count of non-negated error keywords
    """
    sentences = split_into_sentences(text.lower())
    
    count = 0
    for sentence in sentences:
        for keyword in ERROR_KEYWORDS:
            if keyword in sentence and not is_negated_in_sentence(keyword, sentence):
                count += 1
                break  # Count each sentence once, not each keyword
    
    return count


def detect_resolution_from_reasoning(reasoning_text: str) -> Tuple[float, float]:
    """Detect resolution and problem signals from reasoning text.
    
    Uses sentence-boundary detection to handle negation correctly.
    
    Args:
        reasoning_text: The reasoning block to analyze (usually last one)
    
    Returns:
        Tuple of (resolution_score, problem_score)
    """
    text_lower = reasoning_text.lower()
    sentences = split_into_sentences(text_lower)
    
    resolution_score = 0.0
    problem_score = 0.0
    
    for sentence in sentences:
        # Check resolution patterns in this sentence
        for pattern, weight in RESOLUTION_PATTERNS:
            match = re.search(pattern, sentence)
            if match:
                matched_phrase = match.group()
                # Skip if the matched phrase is negated
                if not sentence_has_negation_before_phrase(sentence, matched_phrase):
                    resolution_score += weight
        
        # Check problem patterns in this sentence
        for pattern, weight in PROBLEM_PATTERNS:
            if re.search(pattern, sentence):
                problem_score += weight
    
    return resolution_score, problem_score


def detect_resolution_status(doc: SessionDocument) -> Tuple[str, int]:
    """Detect resolution status from LAST reasoning block.
    
    Uses sentence-boundary negation handling and context-aware patterns.
    
    Args:
        doc: SessionDocument to analyze
    
    Returns:
        Tuple of (status, error_count) where status is one of:
        "solved", "partial", "ongoing", "unknown"
    """
    # Count error signals in task_raw (with sentence-boundary negation)
    error_count = count_error_signals(doc.task_raw)
    
    # Check last reasoning block for resolution signals
    if not doc.reasoning_docs:
        return "unknown", error_count
    
    last_block = doc.reasoning_docs[-1]
    resolution_score, problem_score = detect_resolution_from_reasoning(last_block)
    
    if resolution_score > problem_score:
        return "solved", error_count
    elif problem_score > 0:
        return "partial", error_count
    elif doc.edit_count > 3:
        return "ongoing", error_count
    return "unknown", error_count
