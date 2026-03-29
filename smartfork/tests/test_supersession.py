"""Tests for v2.1 supersession detection and resolution tagging.

Tests:
- Supersession detection with domain-specific thresholds
- Resolution status detection with sentence-boundary negation
- Edge case guards: unknown_project, short tasks, self-supersession
- Chain following for supersession relationships
- Additive boost in search results
"""

import numpy as np
import tempfile
from pathlib import Path

import pytest

from src.smartfork.database.models import SessionDocument
from src.smartfork.database.metadata_store import MetadataStore
from src.smartfork.indexer.supersession_detector import (
    detect_supersession, detect_resolution_status, count_error_signals,
    detect_resolution_from_reasoning, cosine_similarity,
    split_into_sentences, is_negated_in_sentence, sentence_has_negation_before_phrase,
    SIMILARITY_THRESHOLDS
)
from src.smartfork.search.supersession_annotator import (
    annotate_supersession, get_latest_in_chain
)
from src.smartfork.ui.result_card import build_result_card


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: Create test SessionDocument
# ═══════════════════════════════════════════════════════════════════════════════

def make_session(
    session_id: str,
    project_name: str = "TestProject",
    task_raw: str = "Fix the login bug",
    domains: list = None,
    session_start: int = 1000,
    edit_count: int = 0,
    reasoning_docs: list = None,
) -> SessionDocument:
    """Helper to create a test SessionDocument."""
    return SessionDocument(
        session_id=session_id,
        project_name=project_name,
        task_raw=task_raw,
        task_doc=task_raw,
        domains=domains or ["auth"],
        session_start=session_start,
        edit_count=edit_count,
        reasoning_docs=reasoning_docs or [],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SUPERSESSION DETECTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSupersessionDetection:
    
    def test_basic_supersession_detection(self):
        """Test basic supersession: Session A → Session B (B supersedes A)."""
        session_a = make_session("session_a", session_start=1000, task_raw="Fix login error in auth.py")
        session_b = make_session("session_b", session_start=2000, task_raw="Fixed login error in auth.py")
        
        # Create embeddings with high similarity
        emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb_b = np.array([0.95, 0.1, 0.0], dtype=np.float32)  # Very similar
        
        stored_embeddings = {"session_a": emb_a}
        
        links = detect_supersession(
            new_session=session_b,
            new_embedding=emb_b,
            existing_sessions=[session_a],
            stored_embeddings=stored_embeddings,
        )
        
        assert len(links) == 1
        assert links[0][0] == "session_a"
        assert links[0][1] > 0.85  # Should exceed threshold
    
    def test_different_projects_not_linked(self):
        """Sessions from different projects should not be linked."""
        session_a = make_session("session_a", project_name="ProjectA", session_start=1000)
        session_b = make_session("session_b", project_name="ProjectB", session_start=2000)
        
        emb_a = np.array([1.0, 0.0], dtype=np.float32)
        emb_b = np.array([1.0, 0.0], dtype=np.float32)  # Identical
        
        links = detect_supersession(
            new_session=session_b,
            new_embedding=emb_b,
            existing_sessions=[session_a],
            stored_embeddings={"session_a": emb_a},
        )
        
        assert len(links) == 0  # Different projects
    
    def test_different_domains_not_linked(self):
        """Sessions with no domain overlap should not be linked."""
        session_a = make_session("session_a", domains=["auth"], session_start=1000)
        session_b = make_session("session_b", domains=["frontend"], session_start=2000)
        
        emb_a = np.array([1.0, 0.0], dtype=np.float32)
        emb_b = np.array([1.0, 0.0], dtype=np.float32)  # Identical
        
        links = detect_supersession(
            new_session=session_b,
            new_embedding=emb_b,
            existing_sessions=[session_a],
            stored_embeddings={"session_a": emb_a},
        )
        
        assert len(links) == 0  # No domain overlap
    
    def test_unknown_project_skipped(self):
        """Sessions with unknown_project should be skipped."""
        session = make_session("session_a", project_name="unknown_project")
        emb = np.array([1.0, 0.0], dtype=np.float32)
        
        links = detect_supersession(
            new_session=session,
            new_embedding=emb,
            existing_sessions=[],
            stored_embeddings={},
        )
        
        assert len(links) == 0
    
    def test_short_task_skipped(self):
        """Sessions with very short tasks (< 5 words) should be skipped."""
        session = make_session("session_a", task_raw="fix bug")  # Only 2 words
        emb = np.array([1.0, 0.0], dtype=np.float32)
        
        links = detect_supersession(
            new_session=session,
            new_embedding=emb,
            existing_sessions=[],
            stored_embeddings={},
        )
        
        assert len(links) == 0
    
    def test_self_supersession_prevented(self):
        """A session cannot supersede itself."""
        session = make_session("session_a", task_raw="Fix the login error properly this time")
        emb = np.array([1.0, 0.0], dtype=np.float32)
        
        links = detect_supersession(
            new_session=session,
            new_embedding=emb,
            existing_sessions=[session],  # Same session in existing
            stored_embeddings={"session_a": emb},
        )
        
        assert len(links) == 0
    
    def test_older_session_required(self):
        """Only older sessions can be superseded."""
        session_old = make_session("session_old", session_start=1000)
        session_new = make_session("session_new", session_start=2000)
        
        emb = np.array([1.0, 0.0], dtype=np.float32)
        
        # Try to supersede newer session from older
        links = detect_supersession(
            new_session=session_old,  # Older
            new_embedding=emb,
            existing_sessions=[session_new],  # Newer
            stored_embeddings={"session_new": emb},
        )
        
        assert len(links) == 0  # Can't supersede newer session


# ═══════════════════════════════════════════════════════════════════════════════
# RESOLUTION STATUS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestResolutionStatus:
    
    def test_solved_with_fixed_the(self):
        """'Fixed the bug' should be detected as solved."""
        doc = make_session(
            "test_session",
            reasoning_docs=["We finally fixed the bug in the auth module."],
        )
        status, error_count = detect_resolution_status(doc)
        assert status == "solved"
    
    def test_solved_with_working_now(self):
        """'Working now' should be detected as solved."""
        doc = make_session(
            "test_session",
            reasoning_docs=["The error is fixed and working now."],
        )
        status, error_count = detect_resolution_status(doc)
        assert status == "solved"
    
    def test_not_working_not_detected_as_solved(self):
        """'Not working' should NOT be detected as solved."""
        doc = make_session(
            "test_session",
            reasoning_docs=["Still not working after the changes."],
        )
        status, error_count = detect_resolution_status(doc)
        assert status != "solved"
    
    def test_fixed_positioning_false_positive(self):
        """'Fixed positioning' (no object) should NOT be detected as solved."""
        doc = make_session(
            "test_session",
            reasoning_docs=["I tried fixed positioning but it didn't work."],
        )
        status, error_count = detect_resolution_status(doc)
        assert status != "solved"
    
    def test_resolved_to_approach_false_positive(self):
        """'Resolved to try' (no object) should NOT be detected as solved."""
        doc = make_session(
            "test_session",
            reasoning_docs=["I resolved to try a different approach."],
        )
        status, error_count = detect_resolution_status(doc)
        assert status != "solved"
    
    def test_partial_with_problem_signals(self):
        """Session with problem signals but no resolution should be partial."""
        doc = make_session(
            "test_session",
            reasoning_docs=["Still broken after multiple attempts."],
        )
        status, error_count = detect_resolution_status(doc)
        assert status == "partial"
    
    def test_ongoing_with_edits(self):
        """Session with many edits but no resolution signal should be ongoing."""
        doc = make_session(
            "test_session",
            edit_count=5,
            reasoning_docs=["Need to try another approach."],
        )
        status, error_count = detect_resolution_status(doc)
        assert status == "ongoing"
    
    def test_unknown_without_reasoning(self):
        """Session without reasoning docs should be unknown."""
        doc = make_session("test_session", reasoning_docs=[])
        status, error_count = detect_resolution_status(doc)
        assert status == "unknown"
    
    def test_negated_error_not_counted(self):
        """'Not an error' should not count as an error signal."""
        doc = make_session(
            "test_session",
            task_raw="This was previously not an error.",
        )
        status, error_count = detect_resolution_status(doc)
        assert error_count == 0


# ═══════════════════════════════════════════════════════════════════════════════
# NEGATION HANDLING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestNegationHandling:
    
    def test_split_into_sentences(self):
        text = "First sentence. Second sentence! Third sentence?"
        sentences = split_into_sentences(text)
        assert len(sentences) == 3
    
    def test_is_negated_in_sentence(self):
        sentence = "This is not an error"
        assert is_negated_in_sentence("error", sentence) is True
        
        sentence2 = "This is an error"
        assert is_negated_in_sentence("error", sentence2) is False
    
    def test_sentence_has_negation_before_phrase(self):
        sentence = "It is not working now"
        assert sentence_has_negation_before_phrase(sentence, "working now") is True
        
        sentence2 = "It is working now"
        assert sentence_has_negation_before_phrase(sentence2, "working now") is False


# ═══════════════════════════════════════════════════════════════════════════════
# CHAIN FOLLOWING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestChainFollowing:
    
    def test_chain_following_single(self):
        """Single supersession: A → B, should return B."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MetadataStore(Path(tmpdir) / "test.db")
            
            session_a = make_session("session_a", session_start=1000)
            session_b = make_session("session_b", session_start=2000)
            
            store.upsert_session(session_a)
            store.upsert_session(session_b)
            store.insert_supersession_link("session_b", "session_a", 0.9)
            
            latest = get_latest_in_chain("session_a", store)
            assert latest == "session_b"
            
            store.close()
    
    def test_chain_following_multiple(self):
        """Chain: A → B → C, should return C."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MetadataStore(Path(tmpdir) / "test.db")
            
            session_a = make_session("session_a", session_start=1000)
            session_b = make_session("session_b", session_start=2000)
            session_c = make_session("session_c", session_start=3000)
            
            store.upsert_session(session_a)
            store.upsert_session(session_b)
            store.upsert_session(session_c)
            
            store.insert_supersession_link("session_b", "session_a", 0.9)
            store.insert_supersession_link("session_c", "session_b", 0.9)
            
            latest = get_latest_in_chain("session_a", store)
            assert latest == "session_c"
            
            store.close()
    
    def test_no_chain_returns_self(self):
        """Session with no supersession should return itself."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MetadataStore(Path(tmpdir) / "test.db")
            
            session = make_session("session_x")
            store.upsert_session(session)
            
            latest = get_latest_in_chain("session_x", store)
            assert latest == "session_x"
            
            store.close()


# ═══════════════════════════════════════════════════════════════════════════════
# ADDITIVE BOOST TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdditiveBoost:
    
    def test_additive_boost_applied(self):
        """Superseding sessions should get additive boost."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MetadataStore(Path(tmpdir) / "test.db")
            
            session_a = make_session("session_a", session_start=1000)
            session_b = make_session("session_b", session_start=2000)
            
            store.upsert_session(session_a)
            store.upsert_session(session_b)
            store.insert_supersession_link("session_b", "session_a", 0.9)
            
            # Create result cards
            card_a = build_result_card(session_a, match_score=0.56)
            card_b = build_result_card(session_b, match_score=0.55)  # Lower initially
            
            cards = annotate_supersession([card_a, card_b], store, boost_amount=0.15)
            
            # Session B should have boost applied (0.55 + 0.15 = 0.70)
            # Session A should not have boost
            b_card = next(c for c in cards if c.session_id == "session_b")
            a_card = next(c for c in cards if c.session_id == "session_a")
            
            assert b_card.match_score >= 0.70  # Boosted
            assert a_card.match_score == 0.56  # Not boosted
            
            store.close()
    
    def test_boost_ceiling(self):
        """Boost should not exceed 1.0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MetadataStore(Path(tmpdir) / "test.db")
            
            session_a = make_session("session_a", session_start=1000)
            session_b = make_session("session_b", session_start=2000)
            
            store.upsert_session(session_a)
            store.upsert_session(session_b)
            store.insert_supersession_link("session_b", "session_a", 0.9)
            
            card_b = build_result_card(session_b, match_score=0.95)
            cards = annotate_supersession([card_b], store, boost_amount=0.15)
            
            b_card = cards[0]
            assert b_card.match_score <= 1.0  # Capped at 1.0
            
            store.close()


# ═══════════════════════════════════════════════════════════════════════════════
# COSINE SIMILARITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCosineSimilarity:
    
    def test_identical_vectors(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert abs(cosine_similarity(a, b) - 1.0) < 0.001
    
    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert abs(cosine_similarity(a, b) - 0.0) < 0.001
    
    def test_similar_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.9, 0.435], dtype=np.float32)
        sim = cosine_similarity(a, b)
        assert sim > 0.8
    
    def test_zero_vector(self):
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0
