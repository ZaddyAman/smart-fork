"""Chunk-level search engine for SmartFork.

This module provides the ChunkSearchEngine class which brings together
query parsing, semantic search, and chunk ranking to deliver intelligent
chunk-level retrieval within and across sessions.
"""

import time
import hashlib
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from loguru import logger

from ..database.chroma_db import ChromaDatabase
from ..database.chunk_models import (
    EnhancedChunk,
    ChunkSearchResult,
    TokenBudget
)
from .query_parser import QueryParser, QueryIntent
from .chunk_ranker import ChunkRanker
from .semantic import SemanticSearchEngine


@dataclass
class CacheEntry:
    """Cache entry with TTL support."""
    results: List[ChunkSearchResult]
    timestamp: float
    query_hash: str
    session_id: Optional[str]
    n_results: int
    
    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if cache entry has expired."""
        return (time.time() - self.timestamp) > ttl_seconds


class ChunkSearchEngine:
    """
    Main search engine for finding specific chunks within sessions.
    
    This engine combines query parsing, semantic search, and multi-signal
    chunk ranking to deliver highly relevant chunk-level search results.
    It supports both session-scoped searches (within a specific session)
    and global searches (across all sessions).
    
    Features:
        - Query intent parsing for smarter retrieval
        - Semantic similarity search for candidate selection
        - Multi-signal ranking (semantic, file match, keywords, recency)
        - Token budget management for context window constraints
        - Result caching with TTL for performance
    
    Example:
        >>> db = ChromaDatabase(Path("./chroma_db"))
        >>> engine = ChunkSearchEngine(db, enable_cache=True)
        >>> 
        >>> # Search within a specific session
        >>> results = engine.search(
        ...     "How does the auth middleware work?",
        ...     session_id="session_123",
        ...     n_results=5
        ... )
        >>> 
        >>> # Search across all sessions
        >>> global_results = engine.search(
        ...     "JWT authentication examples",
        ...     n_results=10
        ... )
    """
    
    # Cache TTL in seconds (5 minutes default)
    CACHE_TTL_SECONDS: float = 300.0
    
    # Default number of semantic candidates to retrieve
    DEFAULT_N_CANDIDATES: int = 50
    
    # Maximum chunks to retrieve from a single session
    MAX_SESSION_CHUNKS: int = 100
    
    def __init__(self, db: ChromaDatabase, enable_cache: bool = True):
        """
        Initialize the chunk search engine.
        
        Args:
            db: ChromaDatabase instance for data access
            enable_cache: Whether to enable result caching (default: True)
        """
        self.db = db
        self.query_parser = QueryParser()
        self.ranker = ChunkRanker()
        self.semantic_engine = SemanticSearchEngine(db)
        self.enable_cache = enable_cache
        self._cache: Dict[str, CacheEntry] = {}
        
        logger.info(f"ChunkSearchEngine initialized (cache={'enabled' if enable_cache else 'disabled'})")
    
    def search(
        self,
        query: str,
        session_id: Optional[str] = None,
        n_results: int = 10,
        token_budget: Optional[TokenBudget] = None
    ) -> List[ChunkSearchResult]:
        """
        Search for chunks matching the query.
        
        If session_id is provided, searches within that session only.
        If session_id is None, searches across all sessions using
        semantic search to find relevant candidates.
        
        The search process:
        1. Parse the query to extract intent and entities
        2. Get candidate chunks (session-scoped or global semantic search)
        3. Rank chunks using multi-signal scoring
        4. Apply token budget filtering if provided
        5. Return top N results
        
        Args:
            query: Search query string
            session_id: Optional session ID to scope search to
            n_results: Maximum number of results to return
            token_budget: Optional token budget for context constraints
            
        Returns:
            List of ChunkSearchResult sorted by relevance (highest first)
            
        Example:
            >>> # Search for authentication-related chunks
            >>> results = engine.search("JWT auth implementation")
            >>> 
            >>> # Search with token budget
            >>> budget = TokenBudget(max_total_tokens=1500, max_chunks=3)
            >>> results = engine.search("API error handling", token_budget=budget)
        """
        if not query or not isinstance(query, str):
            logger.warning("Empty or invalid query provided")
            return []
        
        # Check cache if enabled
        if self.enable_cache:
            cached_results = self._get_cached_results(query, session_id, n_results)
            if cached_results is not None:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return self._apply_token_budget(cached_results, token_budget)[:n_results]
        
        try:
            # Step 1: Parse query to extract intent
            query_intent = self.query_parser.parse(query)
            logger.debug(f"Parsed query intent: {query_intent.intent_type} "
                        f"(files: {query_intent.file_references}, "
                        f"techs: {query_intent.technologies})")
            
            # Step 2: Get candidate chunks
            if session_id:
                candidates = self._get_session_chunks(session_id)
                logger.debug(f"Retrieved {len(candidates)} chunks from session {session_id}")
            else:
                candidates = self._get_candidate_chunks_semantic(query, self.DEFAULT_N_CANDIDATES)
                logger.debug(f"Retrieved {len(candidates)} candidates from semantic search")
            
            if not candidates:
                logger.info(f"No candidate chunks found for query: {query[:50]}...")
                return []
            
            # Step 3: Rank chunks using multi-signal scoring
            # Note: We don't have query embedding here, so semantic score will use default
            ranked_results = self.ranker.rank_chunks(candidates, query_intent, query_embedding=None)
            logger.debug(f"Ranked {len(ranked_results)} chunks")
            
            # Step 4: Apply token budget if provided
            if token_budget:
                ranked_results = self._apply_token_budget(ranked_results, token_budget)
                logger.debug(f"After token budget filtering: {len(ranked_results)} chunks")
            
            # Step 5: Limit to top N results
            final_results = ranked_results[:n_results]
            
            # Cache results if enabled
            if self.enable_cache:
                self._cache_results(query, session_id, n_results, final_results)
            
            logger.info(f"Search returned {len(final_results)} results for query: {query[:50]}...")
            return final_results
            
        except Exception as e:
            logger.error(f"Search error for query '{query[:50]}...': {e}")
            return []
    
    def _get_session_chunks(self, session_id: str) -> List[EnhancedChunk]:
        """
        Get all chunks for a specific session.
        
        Retrieves chunks from ChromaDB with a maximum limit to prevent
        memory issues with very large sessions.
        
        Args:
            session_id: The session ID to retrieve chunks for
            
        Returns:
            List of EnhancedChunk objects for the session
            
        Raises:
            ValueError: If session_id is empty or invalid
        """
        if not session_id or not isinstance(session_id, str):
            raise ValueError("session_id must be a non-empty string")
        
        try:
            chunks = self.db.get_session_chunks(session_id)
            
            # Limit chunks to prevent memory issues
            if len(chunks) > self.MAX_SESSION_CHUNKS:
                logger.warning(f"Session {session_id} has {len(chunks)} chunks, "
                             f"limiting to {self.MAX_SESSION_CHUNKS}")
                chunks = chunks[:self.MAX_SESSION_CHUNKS]
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks for session {session_id}: {e}")
            return []
    
    def _get_candidate_chunks_semantic(
        self,
        query: str,
        n_candidates: int
    ) -> List[EnhancedChunk]:
        """
        Use ChromaDB semantic search to find candidate chunks across all sessions.
        
        This method uses the SemanticSearchEngine to perform vector similarity
        search and retrieve relevant chunks from the entire database.
        
        Args:
            query: The search query text
            n_candidates: Number of candidate chunks to retrieve
            
        Returns:
            List of EnhancedChunk objects from semantic search
        """
        try:
            # Use semantic engine to search
            search_results = self.semantic_engine.search(query, n_results=n_candidates)
            
            if not search_results:
                return []
            
            # Convert SearchResult to EnhancedChunk
            chunks: List[EnhancedChunk] = []
            for result in search_results:
                # Create minimal EnhancedChunk from search result
                # Note: This is a lightweight version - full chunk data would
                # require additional database lookup
                chunk = EnhancedChunk(
                    id=result.session_id,  # Use session_id as chunk id for now
                    content=result.content,
                    metadata=result.metadata if hasattr(result, 'metadata') else {}
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []
    
    def _apply_token_budget(
        self,
        results: List[ChunkSearchResult],
        budget: Optional[TokenBudget]
    ) -> List[ChunkSearchResult]:
        """
        Filter results to fit within token budget constraints.
        
        Applies budget constraints in order:
        1. Minimum score threshold
        2. Maximum chunks limit
        3. Maximum total tokens
        
        Args:
            results: List of ranked search results
            budget: TokenBudget constraints (None means no filtering)
            
        Returns:
            Filtered list of results within budget
        """
        if not budget or not budget.validate_budget():
            return results
        
        filtered: List[ChunkSearchResult] = []
        total_tokens = 0
        
        for result in results:
            # Check minimum score threshold
            if result.score < budget.min_chunk_score:
                continue
            
            # Check max chunks limit
            if len(filtered) >= budget.max_chunks:
                break
            
            # Calculate tokens for this result
            chunk_tokens = result.get_token_estimate()
            
            # Check total token budget
            if total_tokens + chunk_tokens > budget.max_total_tokens:
                # If we haven't added any chunks yet, add at least one
                # even if it exceeds budget (better than empty results)
                if not filtered:
                    filtered.append(result)
                break
            
            filtered.append(result)
            total_tokens += chunk_tokens
        
        logger.debug(f"Token budget filtering: {len(results)} -> {len(filtered)} "
                    f"({total_tokens}/{budget.max_total_tokens} tokens)")
        
        return filtered
    
    def _get_cache_key(
        self,
        query: str,
        session_id: Optional[str],
        n_results: int
    ) -> str:
        """
        Generate cache key for a search query.
        
        Creates a deterministic hash based on query parameters.
        
        Args:
            query: Search query string
            session_id: Optional session ID
            n_results: Number of results requested
            
        Returns:
            Cache key string
        """
        key_data = f"{query}:{session_id}:{n_results}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_results(
        self,
        query: str,
        session_id: Optional[str],
        n_results: int
    ) -> Optional[List[ChunkSearchResult]]:
        """
        Retrieve cached search results if available and not expired.
        
        Args:
            query: Search query string
            session_id: Optional session ID
            n_results: Number of results requested
            
        Returns:
            Cached results if valid, None otherwise
        """
        cache_key = self._get_cache_key(query, session_id, n_results)
        entry = self._cache.get(cache_key)
        
        if entry is None:
            return None
        
        if entry.is_expired(self.CACHE_TTL_SECONDS):
            # Remove expired entry
            del self._cache[cache_key]
            return None
        
        return entry.results
    
    def _cache_results(
        self,
        query: str,
        session_id: Optional[str],
        n_results: int,
        results: List[ChunkSearchResult]
    ) -> None:
        """
        Store search results in cache.
        
        Args:
            query: Search query string
            session_id: Optional session ID
            n_results: Number of results requested
            results: Search results to cache
        """
        cache_key = self._get_cache_key(query, session_id, n_results)
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        self._cache[cache_key] = CacheEntry(
            results=results,
            timestamp=time.time(),
            query_hash=query_hash,
            session_id=session_id,
            n_results=n_results
        )
        
        logger.debug(f"Cached {len(results)} results for query: {query[:50]}...")
    
    def clear_cache(self) -> None:
        """Clear all cached search results."""
        self._cache.clear()
        logger.info("Search cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_entries = len(self._cache)
        expired_entries = sum(
            1 for entry in self._cache.values()
            if entry.is_expired(self.CACHE_TTL_SECONDS)
        )
        
        return {
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "valid_entries": total_entries - expired_entries,
            "cache_enabled": self.enable_cache,
            "ttl_seconds": self.CACHE_TTL_SECONDS
        }
    
    def search_with_embedding(
        self,
        query: str,
        query_embedding: List[float],
        session_id: Optional[str] = None,
        n_results: int = 10,
        token_budget: Optional[TokenBudget] = None
    ) -> List[ChunkSearchResult]:
        """
        Search with a pre-computed query embedding for better semantic scoring.
        
        This method allows callers to provide their own query embedding,
        which will be used for semantic similarity scoring in the ranking
        phase. This can improve relevance when using custom embedding models.
        
        Args:
            query: Search query string (for parsing and display)
            query_embedding: Pre-computed embedding vector for the query
            session_id: Optional session ID to scope search to
            n_results: Maximum number of results to return
            token_budget: Optional token budget for context constraints
            
        Returns:
            List of ChunkSearchResult sorted by relevance
        """
        if not query or not isinstance(query, str):
            logger.warning("Empty or invalid query provided")
            return []
        
        try:
            # Parse query intent
            query_intent = self.query_parser.parse(query)
            
            # Get candidate chunks
            if session_id:
                candidates = self._get_session_chunks(session_id)
            else:
                candidates = self._get_candidate_chunks_semantic(query, self.DEFAULT_N_CANDIDATES)
            
            if not candidates:
                return []
            
            # Rank with provided embedding for better semantic scoring
            ranked_results = self.ranker.rank_chunks(
                candidates,
                query_intent,
                query_embedding=query_embedding
            )
            
            # Apply token budget if provided
            if token_budget:
                ranked_results = self._apply_token_budget(ranked_results, token_budget)
            
            return ranked_results[:n_results]
            
        except Exception as e:
            logger.error(f"Search with embedding error: {e}")
            return []
