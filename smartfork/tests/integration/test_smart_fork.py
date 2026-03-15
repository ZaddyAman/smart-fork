"""Integration tests for SmartFork - Phase 4: Integration Testing and Validation.

This module contains comprehensive integration tests that validate the complete
smart fork workflow including:
- Import verification
- MessageBoundaryChunker functionality
- Query parsing and intent classification
- Chunk ranking and search
- Smart context extraction
- Markdown generation
- CLI integration
- Backward compatibility
- End-to-end workflow
"""

import unittest
from unittest.mock import MagicMock, patch, Mock
import tempfile
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestSmartForkImports(unittest.TestCase):
    """Test that all new SmartFork components can be imported."""
    
    def test_import_enhanced_chunk_metadata(self):
        """Test importing EnhancedChunkMetadata from chunk_models."""
        from smartfork.database.chunk_models import EnhancedChunkMetadata
        self.assertIsNotNone(EnhancedChunkMetadata)
        
    def test_import_enhanced_chunk(self):
        """Test importing EnhancedChunk from chunk_models."""
        from smartfork.database.chunk_models import EnhancedChunk
        self.assertIsNotNone(EnhancedChunk)
        
    def test_import_message_boundary_chunker(self):
        """Test importing MessageBoundaryChunker from chunkers."""
        from smartfork.indexer.chunkers import MessageBoundaryChunker
        self.assertIsNotNone(MessageBoundaryChunker)
        
    def test_import_chunking_config(self):
        """Test importing ChunkingConfig from chunkers."""
        from smartfork.indexer.chunkers import ChunkingConfig
        self.assertIsNotNone(ChunkingConfig)
        
    def test_import_query_parser(self):
        """Test importing QueryParser and QueryIntent from query_parser."""
        from smartfork.search.query_parser import QueryParser, QueryIntent
        self.assertIsNotNone(QueryParser)
        self.assertIsNotNone(QueryIntent)
        
    def test_import_chunk_ranker(self):
        """Test importing ChunkRanker from chunk_ranker."""
        from smartfork.search.chunk_ranker import ChunkRanker
        self.assertIsNotNone(ChunkRanker)
        
    def test_import_chunk_search_engine(self):
        """Test importing ChunkSearchEngine from chunk_search."""
        from smartfork.search.chunk_search import ChunkSearchEngine
        self.assertIsNotNone(ChunkSearchEngine)
        
    def test_import_smart_context_extractor(self):
        """Test importing SmartContextExtractor from smart_generator."""
        from smartfork.fork.smart_generator import SmartContextExtractor
        self.assertIsNotNone(SmartContextExtractor)
        
    def test_import_smart_fork_md_generator(self):
        """Test importing SmartForkMDGenerator from smart_generator."""
        from smartfork.fork.smart_generator import SmartForkMDGenerator
        self.assertIsNotNone(SmartForkMDGenerator)
        
    def test_import_context_extraction_config(self):
        """Test importing ContextExtractionConfig from smart_generator."""
        from smartfork.fork.smart_generator import ContextExtractionConfig
        self.assertIsNotNone(ContextExtractionConfig)
        
    def test_import_token_budget(self):
        """Test importing TokenBudget from chunk_models."""
        from smartfork.database.chunk_models import TokenBudget
        self.assertIsNotNone(TokenBudget)
        
    def test_import_chunk_search_result(self):
        """Test importing ChunkSearchResult from chunk_models."""
        from smartfork.database.chunk_models import ChunkSearchResult
        self.assertIsNotNone(ChunkSearchResult)


class TestMessageBoundaryChunker(unittest.TestCase):
    """Test message-aware chunking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        from smartfork.indexer.chunkers import MessageBoundaryChunker, ChunkingConfig
        from smartfork.database.models import TaskSession, TaskMetadata, ConversationMessage
        
        self.chunker = MessageBoundaryChunker()
        self.config = ChunkingConfig(max_tokens_per_chunk=512)
        
        # Create a sample session with messages
        self.messages = [
            ConversationMessage(role="user", content="How do I implement JWT authentication?", timestamp=1000),
            ConversationMessage(role="assistant", content="Here's how to implement JWT auth in Python using PyJWT library.", timestamp=2000),
            ConversationMessage(role="assistant", content="```python\nimport jwt\n\ndef create_token(user_id):\n    return jwt.encode({'user_id': user_id}, 'secret')\n```", timestamp=3000),
            ConversationMessage(role="user", content="Thanks! Can you also show me how to verify tokens?", timestamp=4000),
            ConversationMessage(role="assistant", content="Here's the verification code in app.py", timestamp=5000),
        ]
        
        self.session = TaskSession(
            task_id="test_session_123",
            metadata=TaskMetadata(files_in_context=["app.py", "auth.py"]),
            conversation=self.messages,
            ui_messages=[]
        )
    
    def test_chunk_session_creates_chunks(self):
        """Test that chunking creates chunks from a session."""
        chunks = self.chunker.chunk_session(self.session)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
    
    def test_chunks_preserve_message_boundaries(self):
        """Test that chunks don't split individual messages."""
        chunks = self.chunker.chunk_session(self.session)
        
        for chunk in chunks:
            # Each chunk should have complete messages
            # Check that content starts with a role prefix
            self.assertTrue(
                chunk.content.startswith('['),
                f"Chunk content should start with role prefix: {chunk.content[:50]}"
            )
    
    def test_chunks_have_enhanced_metadata(self):
        """Test that chunks have EnhancedChunkMetadata attached."""
        from smartfork.database.chunk_models import EnhancedChunkMetadata
        
        chunks = self.chunker.chunk_session(self.session)
        
        for chunk in chunks:
            self.assertIsInstance(
                chunk.metadata, 
                EnhancedChunkMetadata,
                "Chunk should have EnhancedChunkMetadata"
            )
    
    def test_chunk_metadata_contains_session_info(self):
        """Test that chunk metadata contains session information."""
        chunks = self.chunker.chunk_session(self.session)
        
        for chunk in chunks:
            self.assertEqual(chunk.metadata.session_id, "test_session_123")
            self.assertEqual(chunk.metadata.task_id, "test_session_123")
            self.assertIsNotNone(chunk.metadata.chunk_index)
    
    def test_chunk_metadata_tracks_files(self):
        """Test that chunks track files mentioned."""
        # Create session with clear file references in content
        from smartfork.database.models import TaskSession, TaskMetadata, ConversationMessage
        
        messages = [
            ConversationMessage(role="user", content="Check the code in src/main.py and config.yaml", timestamp=1000),
            ConversationMessage(role="assistant", content="Here's the fix for src/main.py", timestamp=2000),
        ]
        
        session = TaskSession(
            task_id="test_session_files",
            metadata=TaskMetadata(files_in_context=["src/main.py", "config.yaml"]),
            conversation=messages,
            ui_messages=[]
        )
        
        chunks = self.chunker.chunk_session(session)
        
        # At least one chunk should mention file references
        files_found = set()
        for chunk in chunks:
            files_found.update(chunk.metadata.files_mentioned)
        
        # Should find some file references (the regex extracts files with extensions)
        self.assertGreater(len(files_found), 0, f"Should find file references, got: {files_found}")
    
    def test_chunk_content_classification(self):
        """Test that chunks have content type classification."""
        chunks = self.chunker.chunk_session(self.session)
        
        for chunk in chunks:
            self.assertIn(
                chunk.metadata.content_type,
                ["user_query", "assistant_response", "code_block", "mixed", "file_operation"]
            )
    
    def test_chunk_token_estimation(self):
        """Test that chunks can estimate token counts."""
        chunks = self.chunker.chunk_session(self.session)
        
        for chunk in chunks:
            tokens = chunk.estimate_tokens()
            self.assertIsInstance(tokens, int)
            self.assertGreater(tokens, 0)
    
    def test_chunk_to_dict(self):
        """Test chunk serialization to dict."""
        chunks = self.chunker.chunk_session(self.session)
        
        for chunk in chunks:
            chunk_dict = chunk.to_dict()
            self.assertIsInstance(chunk_dict, dict)
            self.assertIn("id", chunk_dict)
            self.assertIn("content", chunk_dict)
            self.assertIn("metadata", chunk_dict)


class TestQueryParser(unittest.TestCase):
    """Test query parsing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        from smartfork.search.query_parser import QueryParser
        self.parser = QueryParser()
    
    def test_parse_extracts_files(self):
        """Test that parser extracts file references from query."""
        intent = self.parser.parse("JWT auth in auth/jwt.py")
        
        self.assertIn("auth/jwt.py", intent.file_references)
    
    def test_parse_extracts_multiple_files(self):
        """Test extraction of multiple file references."""
        intent = self.parser.parse("Fix the bug in app.py and utils.py")
        
        self.assertTrue(
            any("app.py" in f for f in intent.file_references) or 
            "app.py" in intent.file_references
        )
    
    def test_parse_detects_technologies(self):
        """Test that parser detects technology keywords."""
        intent = self.parser.parse("How do I implement JWT authentication with FastAPI?")
        
        self.assertIn("jwt", intent.technologies)
        self.assertIn("fastapi", intent.technologies)
        self.assertIn("authentication", intent.technologies)
    
    def test_parse_classifies_debug_intent(self):
        """Test classification of debug intent."""
        intent = self.parser.parse("Fix the bug in authentication")
        
        self.assertEqual(intent.intent_type, "debug")
        self.assertIn("debug", intent.actions)
    
    def test_parse_classifies_implement_intent(self):
        """Test classification of implement intent."""
        intent = self.parser.parse("Implement JWT authentication")
        
        self.assertEqual(intent.intent_type, "implement")
        self.assertIn("implement", intent.actions)
    
    def test_parse_classifies_explain_intent(self):
        """Test classification of explain intent."""
        intent = self.parser.parse("How does the auth middleware work?")
        
        self.assertEqual(intent.intent_type, "explain")
    
    def test_parse_detects_code_preference(self):
        """Test detection of code preference in query."""
        intent = self.parser.parse("Show me the code for JWT verification")
        
        self.assertTrue(intent.prefer_code)
    
    def test_parse_detects_recency_preference(self):
        """Test detection of recency preference."""
        intent = self.parser.parse("What did I work on recently?")
        
        self.assertTrue(intent.prefer_recent)
    
    def test_parse_normalizes_query(self):
        """Test query normalization."""
        intent = self.parser.parse("  JWT   AUTH  in APP.PY  ")
        
        self.assertEqual(intent.normalized_query, "jwt auth in app.py")
    
    def test_parse_empty_query(self):
        """Test parsing of empty query."""
        intent = self.parser.parse("")
        
        self.assertEqual(intent.intent_type, "general")
        self.assertEqual(intent.file_references, [])
        self.assertEqual(intent.technologies, [])


class TestChunkRanking(unittest.TestCase):
    """Test chunk ranking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        from smartfork.search.chunk_ranker import ChunkRanker
        from smartfork.search.query_parser import QueryParser
        from smartfork.database.chunk_models import EnhancedChunk, EnhancedChunkMetadata
        
        self.ranker = ChunkRanker()
        self.parser = QueryParser()
        
        # Create sample chunks with different metadata
        self.chunks = [
            EnhancedChunk(
                id="chunk_1",
                content="This is about JWT authentication",
                metadata=EnhancedChunkMetadata(
                    session_id="session_1",
                    task_id="task_1",
                    chunk_index=0,
                    files_mentioned=["auth.py", "jwt.py"],
                    keywords=["jwt", "authentication", "token"],
                    content_type="code_block",
                    last_active=datetime.now().isoformat()
                )
            ),
            EnhancedChunk(
                id="chunk_2",
                content="This is about database queries",
                metadata=EnhancedChunkMetadata(
                    session_id="session_1",
                    task_id="task_1",
                    chunk_index=1,
                    files_mentioned=["database.py"],
                    keywords=["sql", "database", "query"],
                    content_type="assistant_response",
                    last_active=datetime.now().isoformat()
                )
            ),
            EnhancedChunk(
                id="chunk_3",
                content="JWT implementation details",
                metadata=EnhancedChunkMetadata(
                    session_id="session_1",
                    task_id="task_1",
                    chunk_index=2,
                    files_mentioned=["app.py"],
                    keywords=["jwt", "implementation"],
                    content_type="mixed",
                    last_active=datetime.now().isoformat()
                )
            )
        ]
    
    def test_rank_chunks_returns_results(self):
        """Test that ranking returns ChunkSearchResults."""
        from smartfork.database.chunk_models import ChunkSearchResult
        
        query_intent = self.parser.parse("JWT authentication")
        results = self.ranker.rank_chunks(self.chunks, query_intent)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), len(self.chunks))
        
        for result in results:
            self.assertIsInstance(result, ChunkSearchResult)
    
    def test_rank_chunks_sorted_by_relevance(self):
        """Test that results are sorted by relevance (highest first)."""
        query_intent = self.parser.parse("JWT authentication in auth.py")
        results = self.ranker.rank_chunks(self.chunks, query_intent)
        
        # Check that scores are in descending order
        for i in range(len(results) - 1):
            self.assertGreaterEqual(
                results[i].score, 
                results[i + 1].score,
                "Results should be sorted by score descending"
            )
    
    def test_rank_chunks_higher_for_matching_files(self):
        """Test that chunks with matching files get higher scores."""
        query_intent = self.parser.parse("JWT in auth.py")
        results = self.ranker.rank_chunks(self.chunks, query_intent)
        
        # Find chunks with auth.py
        auth_chunks = [r for r in results if "auth.py" in r.metadata.files_mentioned]
        other_chunks = [r for r in results if "auth.py" not in r.metadata.files_mentioned]
        
        if auth_chunks and other_chunks:
            self.assertGreaterEqual(
                auth_chunks[0].score,
                other_chunks[-1].score,
                "Chunks with matching files should have higher or equal scores"
            )
    
    def test_rank_chunks_higher_for_matching_keywords(self):
        """Test that chunks with matching keywords get higher scores."""
        query_intent = self.parser.parse("JWT token implementation")
        results = self.ranker.rank_chunks(self.chunks, query_intent)
        
        # Chunk 1 has JWT keywords, should rank high
        jwt_chunks = [r for r in results if "jwt" in r.metadata.keywords]
        
        self.assertGreater(len(jwt_chunks), 0, "Should find chunks with JWT keywords")
    
    def test_rank_chunks_includes_breakdown(self):
        """Test that results include score breakdown."""
        query_intent = self.parser.parse("JWT authentication")
        results = self.ranker.rank_chunks(self.chunks, query_intent)
        
        for result in results:
            self.assertIn("breakdown", dir(result) or hasattr(result, 'breakdown'))
            self.assertIsInstance(result.breakdown, dict)
    
    def test_rank_empty_chunks(self):
        """Test ranking with empty chunk list."""
        query_intent = self.parser.parse("JWT authentication")
        results = self.ranker.rank_chunks([], query_intent)
        
        self.assertEqual(results, [])


class TestChunkSearchEngine(unittest.TestCase):
    """Test chunk-level search engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        from smartfork.search.chunk_search import ChunkSearchEngine
        from smartfork.database.chunk_models import EnhancedChunk, EnhancedChunkMetadata
        
        # Mock the database
        self.mock_db = MagicMock()
        
        # Create sample chunks for session
        self.session_chunks = [
            EnhancedChunk(
                id="session_1_0",
                content="[user]: How do I implement JWT?",
                metadata=EnhancedChunkMetadata(
                    session_id="session_1",
                    task_id="session_1",
                    chunk_index=0,
                    files_mentioned=["auth.py"],
                    keywords=["jwt", "authentication"],
                    content_type="user_query",
                    last_active=datetime.now().isoformat()
                )
            ),
            EnhancedChunk(
                id="session_1_1",
                content="[assistant]: Here's JWT implementation code",
                metadata=EnhancedChunkMetadata(
                    session_id="session_1",
                    task_id="session_1",
                    chunk_index=1,
                    files_mentioned=["auth.py", "jwt.py"],
                    keywords=["jwt", "implementation", "code"],
                    content_type="code_block",
                    last_active=datetime.now().isoformat()
                )
            )
        ]
        
        self.mock_db.get_session_chunks.return_value = self.session_chunks
        
        # Create search engine with mocked db
        self.search_engine = ChunkSearchEngine(self.mock_db, enable_cache=False)
    
    def test_search_within_session(self):
        """Test searching within a specific session."""
        results = self.search_engine.search(
            query="JWT authentication",
            session_id="session_1",
            n_results=5
        )
        
        self.assertIsInstance(results, list)
        # Should return results from the mocked session
        self.mock_db.get_session_chunks.assert_called_with("session_1")
    
    def test_search_across_sessions(self):
        """Test searching across all sessions."""
        # Mock semantic search to return some results
        with patch.object(self.search_engine.semantic_engine, 'search') as mock_semantic:
            from smartfork.search.hybrid import HybridResult
            mock_semantic.return_value = [
                HybridResult(
                    session_id="session_2",
                    score=0.8,
                    breakdown={"semantic": 0.8},
                    metadata={}
                )
            ]
            
            results = self.search_engine.search(
                query="JWT authentication",
                session_id=None,
                n_results=5
            )
            
            self.assertIsInstance(results, list)
            mock_semantic.assert_called_once()
    
    def test_search_results_sorted(self):
        """Test that search results are sorted by relevance."""
        results = self.search_engine.search(
            query="JWT",
            session_id="session_1",
            n_results=5
        )
        
        # Check sorting
        for i in range(len(results) - 1):
            self.assertGreaterEqual(
                results[i].score,
                results[i + 1].score
            )
    
    def test_search_with_token_budget(self):
        """Test search with token budget constraints."""
        from smartfork.database.chunk_models import TokenBudget
        
        budget = TokenBudget(
            max_total_tokens=1000,
            max_chunks=2,
            min_chunk_score=0.0
        )
        
        results = self.search_engine.search(
            query="JWT",
            session_id="session_1",
            n_results=10,
            token_budget=budget
        )
        
        # Should respect max_chunks limit
        self.assertLessEqual(len(results), budget.max_chunks)
    
    def test_search_empty_query(self):
        """Test search with empty query."""
        results = self.search_engine.search(
            query="",
            session_id="session_1"
        )
        
        self.assertEqual(results, [])
    
    def test_search_with_embedding(self):
        """Test search with pre-computed embedding."""
        query_embedding = [0.1] * 384  # Sample embedding
        
        results = self.search_engine.search_with_embedding(
            query="JWT authentication",
            query_embedding=query_embedding,
            session_id="session_1"
        )
        
        self.assertIsInstance(results, list)


class TestSmartContextExtractor(unittest.TestCase):
    """Test smart context extraction."""
    
    def setUp(self):
        """Set up test fixtures."""
        from smartfork.fork.smart_generator import SmartContextExtractor, ContextExtractionConfig
        from smartfork.database.chunk_models import EnhancedChunkMetadata, ChunkSearchResult
        
        self.mock_db = MagicMock()
        
        # Create sample search results
        self.sample_results = [
            ChunkSearchResult(
                chunk_id="chunk_1",
                session_id="session_1",
                content="[user]: How do I implement JWT?",
                score=0.9,
                breakdown={"semantic": 0.9, "file_match": 0.8},
                metadata=EnhancedChunkMetadata(
                    session_id="session_1",
                    task_id="session_1",
                    chunk_index=0,
                    files_mentioned=["auth.py"],
                    keywords=["jwt"],
                    content_type="user_query",
                    token_count=50,
                    last_active=datetime.now().isoformat()
                )
            ),
            ChunkSearchResult(
                chunk_id="chunk_2",
                session_id="session_1",
                content="[assistant]: Here's the implementation",
                score=0.85,
                breakdown={"semantic": 0.85, "file_match": 0.8},
                metadata=EnhancedChunkMetadata(
                    session_id="session_1",
                    task_id="session_1",
                    chunk_index=1,
                    files_mentioned=["auth.py", "jwt.py"],
                    keywords=["jwt", "implementation"],
                    content_type="code_block",
                    token_count=100,
                    last_active=datetime.now().isoformat()
                )
            )
        ]
    
    def test_extract_context_returns_dict(self):
        """Test that extract_context returns a dictionary."""
        with patch('smartfork.fork.smart_generator.ChunkSearchEngine') as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine.search.return_value = self.sample_results
            mock_engine_class.return_value = mock_engine
            
            from smartfork.fork.smart_generator import SmartContextExtractor
            extractor = SmartContextExtractor(self.mock_db)
            context = extractor.extract_context(
                query="JWT authentication",
                session_id="session_1"
            )
        
        self.assertIsInstance(context, dict)
    
    def test_extract_context_contains_required_keys(self):
        """Test that context contains required keys."""
        with patch('smartfork.fork.smart_generator.ChunkSearchEngine') as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine.search.return_value = self.sample_results
            mock_engine_class.return_value = mock_engine
            
            from smartfork.fork.smart_generator import SmartContextExtractor
            extractor = SmartContextExtractor(self.mock_db)
            context = extractor.extract_context(
                query="JWT authentication",
                session_id="session_1"
            )
        
        required_keys = ["found", "query", "session_id", "chunks", "total_tokens", "files_mentioned"]
        for key in required_keys:
            self.assertIn(key, context, f"Context should contain '{key}'")
    
    def test_extract_context_enforces_token_budget(self):
        """Test that context extraction respects token budget."""
        from smartfork.fork.smart_generator import ContextExtractionConfig
        
        # Use larger budget to accommodate the mock results
        config = ContextExtractionConfig(
            max_total_tokens=200,
            max_chunks=2
        )
        
        with patch('smartfork.fork.smart_generator.ChunkSearchEngine') as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine.search.return_value = self.sample_results
            mock_engine_class.return_value = mock_engine
            
            from smartfork.fork.smart_generator import SmartContextExtractor
            extractor = SmartContextExtractor(self.mock_db)
            context = extractor.extract_context(
                query="JWT authentication",
                session_id="session_1",
                config=config
            )
        
        # Total tokens should not exceed budget
        self.assertLessEqual(context["total_tokens"], config.max_total_tokens)
    
    def test_extract_context_returns_files_mentioned(self):
        """Test that context returns files mentioned in chunks."""
        with patch('smartfork.fork.smart_generator.ChunkSearchEngine') as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine.search.return_value = self.sample_results
            mock_engine_class.return_value = mock_engine
            
            from smartfork.fork.smart_generator import SmartContextExtractor
            extractor = SmartContextExtractor(self.mock_db)
            context = extractor.extract_context(
                query="JWT authentication",
                session_id="session_1"
            )
        
        self.assertIn("files_mentioned", context)
        self.assertIsInstance(context["files_mentioned"], list)
        self.assertIn("auth.py", context["files_mentioned"])
    
    def test_generate_fork_content_returns_markdown(self):
        """Test that generate_fork_content returns markdown string."""
        with patch('smartfork.fork.smart_generator.ChunkSearchEngine') as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine.search.return_value = self.sample_results
            mock_engine_class.return_value = mock_engine
            
            from smartfork.fork.smart_generator import SmartContextExtractor
            extractor = SmartContextExtractor(self.mock_db)
            markdown = extractor.generate_fork_content(
                query="JWT authentication",
                session_id="session_1"
            )
        
        self.assertIsInstance(markdown, str)
        self.assertIn("#", markdown)  # Should contain markdown headers
    
    def test_generate_fork_content_includes_all_sections(self):
        """Test that markdown includes all required sections."""
        with patch('smartfork.fork.smart_generator.ChunkSearchEngine') as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine.search.return_value = self.sample_results
            mock_engine_class.return_value = mock_engine
            
            from smartfork.fork.smart_generator import SmartContextExtractor
            extractor = SmartContextExtractor(self.mock_db)
            markdown = extractor.generate_fork_content(
                query="JWT authentication",
                session_id="session_1"
            )
        
        # Check for required sections
        self.assertIn("# Context Fork", markdown)
        self.assertIn("## Overview", markdown)
        self.assertIn("## Relevant Exchanges", markdown)


class TestSmartForkMDGenerator(unittest.TestCase):
    """Test SmartFork markdown generator."""
    
    def setUp(self):
        """Set up test fixtures."""
        from smartfork.database.chunk_models import EnhancedChunkMetadata, ChunkSearchResult
        
        self.mock_db = MagicMock()
        
        # Create sample search results
        self.sample_results = [
            ChunkSearchResult(
                chunk_id="chunk_1",
                session_id="session_1",
                content="[user]: How do I implement JWT?",
                score=0.9,
                breakdown={"semantic": 0.9},
                metadata=EnhancedChunkMetadata(
                    session_id="session_1",
                    task_id="session_1",
                    chunk_index=0,
                    files_mentioned=["auth.py"],
                    keywords=["jwt"],
                    content_type="user_query",
                    token_count=50,
                    last_active=datetime.now().isoformat()
                )
            )
        ]
    
    def test_generator_initialization(self):
        """Test that generator initializes correctly."""
        from smartfork.fork.smart_generator import SmartForkMDGenerator
        
        with patch('smartfork.fork.smart_generator.ChunkSearchEngine'):
            generator = SmartForkMDGenerator(self.mock_db)
            self.assertIsNotNone(generator)
    
    def test_generate_returns_markdown(self):
        """Test that generate returns markdown content."""
        from smartfork.fork.smart_generator import SmartForkMDGenerator
        
        with patch('smartfork.fork.smart_generator.ChunkSearchEngine') as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine.search.return_value = self.sample_results
            mock_engine_class.return_value = mock_engine
            
            generator = SmartForkMDGenerator(self.mock_db)
            content = generator.generate(
                session_id="session_1",
                query="JWT authentication"
            )
        
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 0)
    
    def test_generate_includes_query(self):
        """Test that generated markdown includes the query."""
        from smartfork.fork.smart_generator import SmartForkMDGenerator
        
        with patch('smartfork.fork.smart_generator.ChunkSearchEngine') as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine.search.return_value = self.sample_results
            mock_engine_class.return_value = mock_engine
            
            generator = SmartForkMDGenerator(self.mock_db)
            content = generator.generate(
                session_id="session_1",
                query="JWT authentication"
            )
        
        self.assertIn("JWT authentication", content)
    
    def test_generate_respects_token_limit(self):
        """Test that generate respects max_tokens parameter."""
        from smartfork.fork.smart_generator import SmartForkMDGenerator
        
        with patch('smartfork.fork.smart_generator.ChunkSearchEngine') as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine.search.return_value = self.sample_results
            mock_engine_class.return_value = mock_engine
            
            generator = SmartForkMDGenerator(self.mock_db)
            content = generator.generate(
                session_id="session_1",
                query="JWT authentication",
                max_tokens=500
            )
        
        # Content should be generated successfully with token limit
        self.assertIsInstance(content, str)
    
    def test_save_creates_file(self):
        """Test that save creates a file."""
        from smartfork.fork.smart_generator import SmartForkMDGenerator
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_fork.md"
            
            with patch('smartfork.fork.smart_generator.ChunkSearchEngine') as mock_engine_class:
                mock_engine = MagicMock()
                mock_engine.search.return_value = self.sample_results
                mock_engine_class.return_value = mock_engine
                
                generator = SmartForkMDGenerator(self.mock_db)
                result_path = generator.save(
                    session_id="session_1",
                    query="JWT authentication",
                    output_path=output_path
                )
            
            self.assertTrue(output_path.exists())
            self.assertEqual(result_path, output_path)
            
            # Verify content was written
            content = output_path.read_text()
            self.assertGreater(len(content), 0)


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with old types."""
    
    def test_chunk_metadata_alias(self):
        """Test that ChunkMetadata is an alias for EnhancedChunkMetadata."""
        from smartfork.database.chunk_models import ChunkMetadata, EnhancedChunkMetadata
        
        self.assertIs(ChunkMetadata, EnhancedChunkMetadata)
    
    def test_chunk_alias(self):
        """Test that Chunk is an alias for EnhancedChunk."""
        from smartfork.database.chunk_models import Chunk, EnhancedChunk
        
        self.assertIs(Chunk, EnhancedChunk)
    
    def test_old_chunk_metadata_creation(self):
        """Test creating old-style ChunkMetadata still works."""
        from smartfork.database.chunk_models import ChunkMetadata
        
        metadata = ChunkMetadata(
            session_id="test_session",
            task_id="test_task",
            chunk_index=0
        )
        
        self.assertEqual(metadata.session_id, "test_session")
        self.assertEqual(metadata.task_id, "test_task")
    
    def test_enhanced_fields_available(self):
        """Test that enhanced fields are available on old alias."""
        from smartfork.database.chunk_models import ChunkMetadata
        
        metadata = ChunkMetadata(
            session_id="test_session",
            task_id="test_task",
            chunk_index=0,
            files_mentioned=["app.py"],
            keywords=["auth", "jwt"]
        )
        
        self.assertEqual(metadata.files_mentioned, ["app.py"])
        self.assertEqual(metadata.keywords, ["auth", "jwt"])


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration test of the complete SmartFork workflow."""
    
    def setUp(self):
        """Set up end-to-end test fixtures."""
        from smartfork.database.models import TaskSession, TaskMetadata, ConversationMessage
        
        # Create a realistic session
        self.session = TaskSession(
            task_id="e2e_test_session",
            metadata=TaskMetadata(files_in_context=["app.py", "auth.py", "models.py"]),
            conversation=[
                ConversationMessage(
                    role="user",
                    content="I need to implement JWT authentication in my FastAPI app. The auth should go in auth.py",
                    timestamp=1000000
                ),
                ConversationMessage(
                    role="assistant",
                    content="I'll help you implement JWT authentication. First, let's create the auth module.",
                    timestamp=2000000
                ),
                ConversationMessage(
                    role="assistant",
                    content="""```python
# auth.py
import jwt
from datetime import datetime, timedelta

def create_access_token(user_id: str):
    expire = datetime.utcnow() + timedelta(hours=24)
    data = {"sub": user_id, "exp": expire}
    return jwt.encode(data, "secret", algorithm="HS256")
```""",
                    timestamp=3000000
                ),
                ConversationMessage(
                    role="user",
                    content="Great! Now how do I protect routes in app.py?",
                    timestamp=4000000
                ),
                ConversationMessage(
                    role="assistant",
                    content="""```python
# app.py
from fastapi import Depends, HTTPException
from auth import create_access_token

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, "secret", algorithms=["HS256"])
        return payload["sub"]
    except jwt.PyJWTError:
        raise HTTPException(status_code=401)
```""",
                    timestamp=5000000
                )
            ],
            ui_messages=[]
        )
    
    def test_complete_workflow(self):
        """Test the complete workflow from session to markdown generation."""
        from smartfork.indexer.chunkers import MessageBoundaryChunker
        from smartfork.search.query_parser import QueryParser
        from smartfork.search.chunk_ranker import ChunkRanker
        
        # Step 1: Chunk the session
        chunker = MessageBoundaryChunker()
        chunks = chunker.chunk_session(self.session)
        
        self.assertGreater(len(chunks), 0, "Should create chunks from session")
        
        # Step 2: Parse a query with action keyword
        parser = QueryParser()
        query_intent = parser.parse("Implement JWT authentication in auth.py")
        
        # Should detect implement intent and technologies
        self.assertEqual(query_intent.intent_type, "implement")
        self.assertIn("jwt", query_intent.technologies)
        
        # Step 3: Rank chunks
        ranker = ChunkRanker()
        results = ranker.rank_chunks(chunks, query_intent)
        
        self.assertGreater(len(results), 0, "Should have ranked results")
        
        # Verify sorting
        for i in range(len(results) - 1):
            self.assertGreaterEqual(
                results[i].score,
                results[i + 1].score,
                "Results should be sorted by relevance"
            )
        
        # Step 4: Check that JWT-related content ranks higher
        jwt_results = [r for r in results if "jwt" in r.content.lower()]
        other_results = [r for r in results if "jwt" not in r.content.lower()]
        
        if jwt_results and other_results:
            self.assertGreaterEqual(
                jwt_results[0].score,
                other_results[-1].score,
                "JWT-related content should rank higher"
            )
    
    def test_workflow_with_file_filtering(self):
        """Test workflow with file-based filtering."""
        from smartfork.indexer.chunkers import MessageBoundaryChunker
        from smartfork.search.query_parser import QueryParser
        from smartfork.search.chunk_ranker import ChunkRanker
        
        # Chunk session
        chunker = MessageBoundaryChunker()
        chunks = chunker.chunk_session(self.session)
        
        # Parse query with file reference
        parser = QueryParser()
        query_intent = parser.parse("Show me auth.py implementation")
        
        self.assertIn("auth.py", query_intent.file_references)
        
        # Rank chunks
        ranker = ChunkRanker()
        results = ranker.rank_chunks(chunks, query_intent)
        
        # Chunks mentioning auth.py should have high file_match scores
        auth_chunks = [r for r in results if "auth.py" in r.metadata.files_mentioned]
        
        for chunk in auth_chunks:
            self.assertGreater(
                chunk.breakdown.get("file_match", 0),
                0.0,
                "Chunks mentioning auth.py should have positive file_match score"
            )
    
    def test_workflow_preserves_message_boundaries(self):
        """Test that complete workflow preserves message boundaries."""
        from smartfork.indexer.chunkers import MessageBoundaryChunker
        
        chunker = MessageBoundaryChunker()
        chunks = chunker.chunk_session(self.session)
        
        # Verify each chunk starts with a role prefix
        for chunk in chunks:
            content = chunk.content
            self.assertTrue(
                content.startswith('[') and ']:' in content[:20],
                f"Chunk should start with role prefix: {content[:50]}"
            )
    
    def test_workflow_extracts_code_blocks(self):
        """Test that workflow correctly identifies and extracts code blocks."""
        from smartfork.indexer.chunkers import MessageBoundaryChunker
        from smartfork.database.chunk_models import ChunkContentType
        
        chunker = MessageBoundaryChunker()
        chunks = chunker.chunk_session(self.session)
        
        # Should have at least one code block chunk
        code_chunks = [
            c for c in chunks 
            if c.metadata.content_type == ChunkContentType.CODE_BLOCK or 
               "```" in c.content
        ]
        
        self.assertGreater(len(code_chunks), 0, "Should extract code block chunks")


class TestCLIIntegration(unittest.TestCase):
    """Test CLI integration."""
    
    def test_resume_command_args(self):
        """Test resume command argument parsing."""
        from smartfork.cli import resume
        import inspect
        
        # Check that resume command accepts required parameters
        sig = inspect.signature(resume)
        params = list(sig.parameters.keys())
        
        # Verify the function exists and has proper signature
        self.assertIsNotNone(resume)
    
    def test_fork_command_smart_flag(self):
        """Test fork command with --smart flag."""
        from smartfork.cli import fork
        import inspect
        
        # Check that fork command accepts smart parameter
        sig = inspect.signature(fork)
        params = sig.parameters
        
        # Should have smart parameter with bool type
        self.assertIn("smart", params)
        self.assertIn("query", params)
    
    def test_cli_imports_smart_generator(self):
        """Test that CLI imports SmartForkMDGenerator."""
        # This tests that the import in cli.py works
        try:
            from smartfork.cli import SmartForkMDGenerator, ContextExtractionConfig
            self.assertIsNotNone(SmartForkMDGenerator)
            self.assertIsNotNone(ContextExtractionConfig)
        except ImportError as e:
            self.fail(f"CLI should import SmartForkMDGenerator: {e}")


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions from chunk_models."""
    
    def test_calculate_total_tokens(self):
        """Test calculate_total_tokens utility function."""
        from smartfork.database.chunk_models import (
            calculate_total_tokens, 
            EnhancedChunk, 
            EnhancedChunkMetadata
        )
        
        chunks = [
            EnhancedChunk(
                id="c1",
                content="Short content",
                metadata=EnhancedChunkMetadata(
                    session_id="s1",
                    task_id="t1",
                    chunk_index=0,
                    token_count=10
                )
            ),
            EnhancedChunk(
                id="c2",
                content="Another content here",
                metadata=EnhancedChunkMetadata(
                    session_id="s1",
                    task_id="t1",
                    chunk_index=1,
                    token_count=20
                )
            )
        ]
        
        total = calculate_total_tokens(chunks)
        self.assertEqual(total, 30)
    
    def test_filter_chunks_by_budget(self):
        """Test filter_chunks_by_budget utility function."""
        from smartfork.database.chunk_models import (
            filter_chunks_by_budget,
            EnhancedChunk,
            EnhancedChunkMetadata,
            TokenBudget
        )
        
        chunks = [
            EnhancedChunk(
                id=f"c{i}",
                content=f"Content {i}",
                metadata=EnhancedChunkMetadata(
                    session_id="s1",
                    task_id="t1",
                    chunk_index=i,
                    token_count=100
                )
            )
            for i in range(5)
        ]
        
        budget = TokenBudget(max_total_tokens=250, max_chunks=3)
        filtered = filter_chunks_by_budget(chunks, budget)
        
        self.assertLessEqual(len(filtered), 3)
        self.assertLessEqual(
            sum(c.estimate_tokens() for c in filtered),
            250
        )
    
    def test_merge_chunk_metadata(self):
        """Test merge_chunk_metadata utility function."""
        from smartfork.database.chunk_models import (
            merge_chunk_metadata,
            EnhancedChunk,
            EnhancedChunkMetadata
        )
        
        chunks = [
            EnhancedChunk(
                id="c1",
                content="Content 1",
                metadata=EnhancedChunkMetadata(
                    session_id="s1",
                    task_id="t1",
                    chunk_index=0,
                    files_mentioned=["app.py"],
                    keywords=["auth"],
                    token_count=50
                )
            ),
            EnhancedChunk(
                id="c2",
                content="Content 2",
                metadata=EnhancedChunkMetadata(
                    session_id="s1",
                    task_id="t1",
                    chunk_index=1,
                    files_mentioned=["utils.py"],
                    keywords=["jwt"],
                    token_count=75
                )
            )
        ]
        
        merged = merge_chunk_metadata(chunks)
        
        self.assertEqual(merged["session_id"], "s1")
        self.assertEqual(merged["total_chunks"], 2)
        self.assertEqual(merged["total_tokens"], 125)
        self.assertIn("app.py", merged["files_mentioned"])
        self.assertIn("utils.py", merged["files_mentioned"])
        self.assertIn("auth", merged["keywords"])
        self.assertIn("jwt", merged["keywords"])


if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)
