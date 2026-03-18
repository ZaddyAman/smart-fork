"""Phase 2 verification tests for retrieval pipeline components.

Tests contextual_chunker, bm25_index, rrf_fusion (without needing Ollama).
Vector index and embedder tests are mock-based to avoid external dependencies.
"""

import json
import tempfile
from pathlib import Path
from typing import List

import pytest

from src.smartfork.database.models import SessionDocument, VectorResult
from src.smartfork.indexer.contextual_chunker import ContextualChunker
from src.smartfork.search.bm25_index import BM25Index
from src.smartfork.search.rrf_fusion import rrf_fuse, rrf_fuse_weighted
from src.smartfork.database.metadata_store import MetadataStore


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXTUAL CHUNKER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestContextualChunker:
    
    @pytest.fixture
    def chunker(self):
        return ContextualChunker()
    
    @pytest.fixture
    def sample_doc(self):
        return SessionDocument(
            session_id="test_session_001",
            project_name="BharatLawAI",
            project_root="d:/Indian Legal Assistant",
            session_start=1757163992341,
            session_end=1757164800416,
            duration_minutes=13.5,
            files_edited=["backend/api/auth.py", "backend/db/models.py", "backend/rag/engine.py"],
            task_raw="implement JWT authentication for the API with refresh tokens",
            reasoning_docs=[
                "Decided to use JWT for stateless auth because ChromaDB lacks session support.",
                "Short reasoning block here.",
            ],
        )
    
    def test_context_header_contains_project(self, chunker, sample_doc):
        header = chunker.build_context_header(sample_doc)
        assert "BharatLawAI" in header
    
    def test_context_header_contains_date(self, chunker, sample_doc):
        header = chunker.build_context_header(sample_doc)
        # Should contain a date string
        assert "2025" in header or "202" in header
    
    def test_context_header_contains_task(self, chunker, sample_doc):
        header = chunker.build_context_header(sample_doc)
        assert "JWT" in header or "implement" in header
    
    def test_context_header_contains_files(self, chunker, sample_doc):
        header = chunker.build_context_header(sample_doc)
        assert "auth.py" in header
    
    def test_task_doc_has_header(self, chunker, sample_doc):
        task_doc = chunker.build_task_doc(sample_doc)
        assert task_doc.startswith("[Project:")
        assert "JWT authentication" in task_doc
    
    def test_summary_doc_empty_when_no_summary(self, chunker, sample_doc):
        result = chunker.build_summary_doc(sample_doc)
        assert result == ""
    
    def test_summary_doc_with_summary(self, chunker, sample_doc):
        sample_doc.summary_doc = "Implemented JWT auth for BharatLawAI."
        result = chunker.build_summary_doc(sample_doc)
        assert "[Project:" in result
        assert "Implemented JWT" in result
    
    def test_reasoning_docs_have_headers(self, chunker, sample_doc):
        docs = chunker.build_reasoning_docs(sample_doc)
        assert len(docs) >= 2
        for doc in docs:
            assert isinstance(doc, dict)
            assert doc["text"].startswith("[Project:")
            assert doc["parent_id"] is not None
            assert doc["chunk_index"] >= 0
            assert doc["full_raw_text"] is not None
    
    def test_long_reasoning_split(self, chunker, sample_doc):
        # Create a very long reasoning block
        long_text = "This is a reasoning sentence about auth. " * 200
        sample_doc.reasoning_docs = [long_text]
        
        docs = chunker.build_reasoning_docs(sample_doc)
        # Should be split into multiple chunks
        assert len(docs) > 1
        # In a variable split, children of the same block should share a parent_id
        parent_ids = [d["parent_id"] for d in docs]
        assert len(set(parent_ids)) == 1
        for doc in docs:
            assert doc["text"].startswith("[Project:")
            assert doc["full_raw_text"] == long_text
    
    def test_empty_reasoning_docs(self, chunker, sample_doc):
        sample_doc.reasoning_docs = []
        docs = chunker.build_reasoning_docs(sample_doc)
        assert len(docs) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# BM25 INDEX TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestBM25Index:
    
    @pytest.fixture
    def store_with_data(self, tmp_path):
        store = MetadataStore(tmp_path / "test.db")
        
        docs = [
            SessionDocument(
                session_id="s1",
                project_name="BharatLawAI",
                task_raw="implement JWT authentication for API",
                files_edited=["backend/api/auth.py", "backend/db/models.py"],
                domains=["auth", "backend"],
            ),
            SessionDocument(
                session_id="s2",
                project_name="SmartFork",
                task_raw="fix search ranking algorithm",
                files_edited=["smartfork/search/hybrid.py"],
                domains=["backend", "rag"],
            ),
            SessionDocument(
                session_id="s3",
                project_name="BharatLawAI",
                task_raw="create React frontend dashboard for analytics",
                files_edited=["frontend/src/Dashboard.tsx", "frontend/src/App.tsx"],
                domains=["frontend"],
            ),
        ]
        
        for doc in docs:
            store.upsert_session(doc)
        
        yield store
        store.close()
    
    def test_build_from_metadata(self, store_with_data):
        index = BM25Index()
        count = index.build_from_metadata(store_with_data)
        assert count == 3
    
    def test_search_by_filename(self, store_with_data):
        index = BM25Index()
        index.build_from_metadata(store_with_data)
        
        results = index.search_text("auth.py")
        assert len(results) > 0
        assert results[0][0] == "s1"  # Should find auth-related session
    
    def test_search_by_keyword(self, store_with_data):
        index = BM25Index()
        index.build_from_metadata(store_with_data)
        
        results = index.search_text("JWT authentication")
        assert len(results) > 0
        assert results[0][0] == "s1"
    
    def test_search_by_project(self, store_with_data):
        index = BM25Index()
        index.build_from_metadata(store_with_data)
        
        results = index.search_text("SmartFork")
        assert len(results) > 0
        assert results[0][0] == "s2"
    
    def test_search_with_candidate_filter(self, store_with_data):
        index = BM25Index()
        index.build_from_metadata(store_with_data)
        
        # Only search within s2 and s3
        results = index.search_text("auth", candidate_ids=["s2", "s3"])
        # s1 should not be in results even though it matches best
        session_ids = [r[0] for r in results]
        assert "s1" not in session_ids
    
    def test_search_empty_query(self, store_with_data):
        index = BM25Index()
        index.build_from_metadata(store_with_data)
        results = index.search_text("")
        assert len(results) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# RRF FUSION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRRFFusion:
    
    def test_basic_fusion(self):
        bm25_results = [("s1", 5.2), ("s2", 3.1), ("s3", 1.0)]
        vector_results = [("s2", 0.95), ("s1", 0.87), ("s4", 0.72)]
        
        fused = rrf_fuse([bm25_results, vector_results])
        
        # s1 and s2 should be top since they appear in both lists
        top_ids = [r[0] for r in fused[:2]]
        assert "s1" in top_ids
        assert "s2" in top_ids
    
    def test_fusion_preserves_all_sessions(self):
        r1 = [("s1", 5.0), ("s2", 3.0)]
        r2 = [("s3", 0.9), ("s4", 0.5)]
        
        fused = rrf_fuse([r1, r2])
        fused_ids = [r[0] for r in fused]
        
        # All 4 sessions should appear
        assert "s1" in fused_ids
        assert "s2" in fused_ids
        assert "s3" in fused_ids
        assert "s4" in fused_ids
    
    def test_fusion_rank_1_in_both_wins(self):
        # s1 is rank 1 in both — should have highest RRF score
        r1 = [("s1", 10.0), ("s2", 5.0)]
        r2 = [("s1", 0.99), ("s3", 0.50)]
        
        fused = rrf_fuse([r1, r2])
        assert fused[0][0] == "s1"
    
    def test_fusion_top_n(self):
        r1 = [("s1", 5.0), ("s2", 4.0), ("s3", 3.0), ("s4", 2.0), ("s5", 1.0)]
        fused = rrf_fuse([r1], top_n=3)
        assert len(fused) == 3
    
    def test_fusion_empty_lists(self):
        fused = rrf_fuse([[], []])
        assert len(fused) == 0
    
    def test_weighted_fusion(self):
        bm25_results = [("s1", 5.0), ("s2", 3.0)]
        vector_results = [("s2", 0.95), ("s1", 0.80)]
        
        # Boost BM25 heavily — s1 should win since it's rank 1 in boosted BM25
        fused = rrf_fuse_weighted(
            [bm25_results, vector_results],
            weights=[3.0, 1.0]
        )
        assert fused[0][0] == "s1"
    
    def test_weighted_fusion_boost_vector(self):
        bm25_results = [("s1", 5.0), ("s2", 3.0)]
        vector_results = [("s2", 0.95), ("s1", 0.80)]
        
        # Boost vector heavily — s2 should win since it's rank 1 in boosted vector
        fused = rrf_fuse_weighted(
            [bm25_results, vector_results],
            weights=[1.0, 3.0]
        )
        assert fused[0][0] == "s2"
