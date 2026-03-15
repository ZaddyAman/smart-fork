"""Phase 3 verification tests — query decomposition and proposition extraction.

All tests run WITHOUT needing Ollama/Anthropic/OpenAI since they test
the rule-based fallback paths that work offline.
"""

import pytest

from src.smartfork.search.query_decomposer import (
    QueryDecomposer, get_vector_weights, INTENT_VECTOR_WEIGHTS
)
from src.smartfork.intelligence.proposition_extractor import PropositionExtractor
from src.smartfork.database.models import SessionDocument, QueryDecomposition


class TestQueryDecomposerRuleBased:
    """Tests the rule-based fallback (no LLM needed)."""
    
    @pytest.fixture
    def decomposer(self):
        return QueryDecomposer(llm=None)  # No LLM — forces rule-based
    
    def test_decision_hunting(self, decomposer):
        result = decomposer.decompose("why did I choose JWT for auth?")
        assert result.intent == "decision_hunting"
    
    def test_implementation_lookup(self, decomposer):
        result = decomposer.decompose("how did I implement the search engine?")
        assert result.intent == "implementation_lookup"
    
    def test_error_recall(self, decomposer):
        result = decomposer.decompose("fix the bug in the auth module")
        assert result.intent == "error_recall"
    
    def test_file_lookup(self, decomposer):
        result = decomposer.decompose("changes to auth.py")
        assert result.intent == "file_lookup"
        assert result.file_hint == "auth.py"
    
    def test_temporal_lookup(self, decomposer):
        result = decomposer.decompose("sessions from last week")
        assert result.intent == "temporal_lookup"
        assert result.time_hint == "last_week"
    
    def test_temporal_days_ago(self, decomposer):
        result = decomposer.decompose("what did I do 3 days ago")
        assert result.time_hint == "3_days_ago"
    
    def test_pattern_hunting(self, decomposer):
        result = decomposer.decompose("all sessions with auth")
        assert result.intent == "pattern_hunting"
    
    def test_vague_memory(self, decomposer):
        result = decomposer.decompose("auth stuff")
        assert result.intent == "vague_memory"
    
    def test_project_extraction(self, decomposer):
        result = decomposer.decompose("auth decisions in BharatLawAI")
        assert result.project == "BharatLawAI"
    
    def test_empty_query(self, decomposer):
        result = decomposer.decompose("")
        assert result.intent == "vague_memory"
    
    def test_tech_terms_extracted(self, decomposer):
        result = decomposer.decompose("JWT authentication for the API endpoint")
        assert len(result.tech_terms) > 0


class TestVectorWeights:
    
    def test_all_intents_have_weights(self):
        intents = [
            "decision_hunting", "implementation_lookup", "error_recall",
            "file_lookup", "temporal_lookup", "pattern_hunting", "vague_memory"
        ]
        for intent in intents:
            weights = get_vector_weights(intent)
            assert "task" in weights
            assert "summary" in weights
            assert "reasoning" in weights
    
    def test_decision_hunting_boosts_reasoning(self):
        weights = get_vector_weights("decision_hunting")
        assert weights["reasoning"] >= weights["task"]
        assert weights["reasoning"] >= weights["summary"]
    
    def test_implementation_lookup_boosts_task(self):
        weights = get_vector_weights("implementation_lookup")
        assert weights["task"] >= weights["reasoning"]


class TestPropositionExtractor:
    
    @pytest.fixture
    def extractor(self):
        return PropositionExtractor(llm=None)  # No LLM — rule-based only
    
    def test_task_proposition(self, extractor):
        doc = SessionDocument(
            session_id="s1",
            project_name="BharatLawAI",
            task_raw="implement JWT authentication for the API",
            files_edited=["backend/api/auth.py"],
        )
        props = extractor.extract(doc)
        assert len(props) >= 1
        assert any("JWT" in p for p in props)
    
    def test_file_proposition(self, extractor):
        doc = SessionDocument(
            session_id="s1",
            project_name="BharatLawAI",
            task_raw="fix auth",
            files_edited=["backend/api/auth.py", "backend/db/models.py"],
        )
        props = extractor.extract(doc)
        assert any("auth.py" in p for p in props)
    
    def test_tech_detection(self, extractor):
        doc = SessionDocument(
            session_id="s1",
            project_name="BharatLawAI",
            task_raw="set up ChromaDB for RAG pipeline",
            files_edited=[],
        )
        props = extractor.extract(doc)
        assert any("ChromaDB" in p for p in props)
    
    def test_empty_task(self, extractor):
        doc = SessionDocument(
            session_id="s1",
            project_name="unknown_project",
            task_raw="",
            files_edited=[],
        )
        props = extractor.extract(doc)
        assert len(props) == 0
