"""Hybrid search engine combining semantic, keyword, recency, and path signals."""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import re
import hashlib
import time
from collections import defaultdict, OrderedDict
from functools import lru_cache
from loguru import logger

from rank_bm25 import BM25Okapi
import numpy as np

from ..database.chroma_db import ChromaDatabase
from ..database.models import HybridResult, SearchResult
from .semantic import SemanticSearchEngine


class TimedLRUCache:
    """LRU Cache with TTL support for search results."""
    
    def __init__(self, maxsize: int = 128, ttl: int = 300):
        """Initialize cache.
        
        Args:
            maxsize: Maximum number of items in cache
            ttl: Time-to-live in seconds
        """
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
    
    def _make_key(self, *args, **kwargs) -> str:
        """Create a cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, *args, **kwargs) -> Optional[Any]:
        """Get item from cache if it exists and is not expired."""
        key = self._make_key(*args, **kwargs)
        
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return value
            else:
                # Expired
                del self._cache[key]
        
        return None
    
    def set(self, value: Any, *args, **kwargs) -> None:
        """Set item in cache."""
        key = self._make_key(*args, **kwargs)
        
        # Remove oldest if at capacity
        if len(self._cache) >= self.maxsize:
            self._cache.popitem(last=False)
        
        self._cache[key] = (value, time.time())
        self._cache.move_to_end(key)
    
    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()
    
    def invalidate_session(self, session_id: str) -> None:
        """Invalidate cache entries containing a specific session."""
        keys_to_remove = []
        for key in list(self._cache.keys()):
            value, _ = self._cache[key]
            if isinstance(value, list):
                for item in value:
                    if hasattr(item, 'session_id') and item.session_id == session_id:
                        keys_to_remove.append(key)
                        break
        
        for key in keys_to_remove:
            del self._cache[key]


class RecencyScorer:
    """
    Exponential decay based on session age.
    Sessions from last 7 days get maximum score.
    Decay rate: 50% per month after 7 days.
    """
    
    MAX_AGE_DAYS = 7
    HALF_LIFE_DAYS = 30
    
    def score(self, last_active: Optional[datetime]) -> float:
        """Calculate recency score.
        
        Args:
            last_active: Last active datetime
            
        Returns:
            Score between 0.0 and 1.0
        """
        if not last_active:
            return 0.5  # Neutral score for unknown
        
        age_days = (datetime.now() - last_active).days
        
        if age_days <= self.MAX_AGE_DAYS:
            return 1.0
        
        # Exponential decay
        decay_days = age_days - self.MAX_AGE_DAYS
        return 0.5 ** (decay_days / self.HALF_LIFE_DAYS)


class PathMatcher:
    """
    Boosts sessions that worked on files in the current directory tree.
    """
    
    def score(self, session_paths: List[str], current_dir: str) -> float:
        """Calculate path match score.
        
        Args:
            session_paths: List of file paths from session
            current_dir: Current working directory
            
        Returns:
            Score between 0.0 and 1.0
        """
        if not current_dir or not session_paths:
            return 0.0
        
        try:
            current_parts = Path(current_dir).resolve().parts
        except Exception:
            return 0.0
        
        matches = 0
        valid_paths = 0
        
        for session_path in session_paths:
            try:
                session_parts = Path(session_path).resolve().parts
                valid_paths += 1
                
                # Count common path components
                common = 0
                for a, b in zip(current_parts, session_parts):
                    if a.lower() == b.lower():
                        common += 1
                    else:
                        break
                
                # Score based on overlap depth
                if common > 0:
                    matches += common / len(current_parts)
            except Exception:
                continue
        
        return min(matches / max(1, valid_paths), 1.0)


class BM25Search:
    """BM25 keyword search for sessions."""
    
    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.session_ids: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
    
    def build_index(self, sessions: Dict[str, str]) -> None:
        """Build BM25 index from session texts.
        
        Args:
            sessions: Dict mapping session_id to full text
        """
        self.session_ids = []
        self.tokenized_corpus = []
        
        for session_id, text in sessions.items():
            self.session_ids.append(session_id)
            # Tokenize: lowercase, split on non-alphanumeric, remove stop words
            tokens = self._tokenize(text)
            self.tokenized_corpus.append(tokens)
        
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        logger.info(f"BM25 index built with {len(self.session_ids)} sessions")
    
    def search(self, query: str, n_results: int = 20) -> List[SearchResult]:
        """Search using BM25.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        if not self.bm25:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize scores to 0-1 range
        max_score = max(scores) if max(scores) > 0 else 1
        normalized = [(idx, score / max_score) for idx, score in enumerate(scores)]
        
        # Sort by score
        normalized.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in normalized[:n_results]:
            if score > 0:
                results.append(SearchResult(
                    session_id=self.session_ids[idx],
                    content="",  # BM25 doesn't return content
                    score=score,
                    metadata={"source": "bm25"}
                ))
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Split on non-alphanumeric but preserve code identifiers
        # Keep camelCase and snake_case intact
        tokens = re.findall(r'[a-z]+|[a-z]+_[a-z]+_?[a-z]*|[a-z]+[a-z0-9]*[A-Z][a-zA-Z0-9]*', text)
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'
        }
        
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
        
        return tokens


class HybridSearchEngine:
    """
    Combines multiple search signals for optimal relevance.
    
    Final Score = (semantic * 0.50) + (bm25 * 0.25) + (recency * 0.15) + (path * 0.10)
    """
    
    WEIGHTS = {
        'semantic': 0.50,
        'keyword': 0.25,
        'recency': 0.15,
        'path': 0.10
    }
    
    # Default cache settings
    DEFAULT_CACHE_SIZE = 128
    DEFAULT_CACHE_TTL = 300  # 5 minutes
    
    def __init__(self, db: ChromaDatabase, enable_cache: bool = True, 
                 cache_size: int = DEFAULT_CACHE_SIZE, cache_ttl: int = DEFAULT_CACHE_TTL):
        """Initialize the hybrid search engine.
        
        Args:
            db: ChromaDatabase instance
            enable_cache: Whether to enable search result caching
            cache_size: Maximum number of cached results
            cache_ttl: Cache time-to-live in seconds
        """
        self.db = db
        self.semantic = SemanticSearchEngine(db)
        self.bm25 = BM25Search()
        self.recency = RecencyScorer()
        self.path_matcher = PathMatcher()
        
        # Cache for session metadata
        self._session_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Search result cache
        self._enable_cache = enable_cache
        self._search_cache: Optional[TimedLRUCache] = TimedLRUCache(maxsize=cache_size, ttl=cache_ttl) if enable_cache else None
        
        # Track last index build time
        self._index_built_at: Optional[float] = None
    
    def build_bm25_index(self, force_rebuild: bool = False) -> None:
        """Build BM25 index from all sessions in the database.
        
        Args:
            force_rebuild: If True, rebuild even if index already exists
        """
        # Skip rebuild if index is fresh (less than 5 minutes old)
        if not force_rebuild and self._index_built_at and self.bm25.bm25:
            age_seconds = time.time() - self._index_built_at
            if age_seconds < 300:  # 5 minutes
                logger.debug(f"BM25 index is fresh ({age_seconds:.0f}s old), skipping rebuild")
                return
        
        sessions = {}
        session_count = 0
        
        # Build index with batched metadata retrieval
        for session_id in self.db.get_unique_sessions():
            # Skip if metadata already cached
            if session_id in self._session_metadata and not force_rebuild:
                # Use cached metadata but still need to build BM25
                chunks = self.db.get_session_chunks(session_id)
                if chunks:
                    full_text = " ".join([c.content for c in chunks])
                    sessions[session_id] = full_text
            else:
                chunks = self.db.get_session_chunks(session_id)
                if chunks:
                    full_text = " ".join([c.content for c in chunks])
                    sessions[session_id] = full_text
                    
                    # Cache metadata
                    self._session_metadata[session_id] = {
                        "files_in_context": chunks[0].metadata.files_in_context,
                        "technologies": chunks[0].metadata.technologies,
                        "last_active": chunks[0].metadata.last_active
                    }
            
            session_count += 1
        
        self.bm25.build_index(sessions)
        self._index_built_at = time.time()
        logger.info(f"Hybrid search index built with {len(sessions)} sessions")
    
    def search(
        self,
        query: str,
        current_dir: Optional[str] = None,
        n_results: int = 10
    ) -> List[HybridResult]:
        """Perform hybrid search with caching support.
        
        Args:
            query: Search query
            current_dir: Current working directory for path matching
            n_results: Number of results to return
            
        Returns:
            List of HybridResult objects sorted by score
        """
        # Check cache first
        if self._enable_cache and self._search_cache:
            cached = self._search_cache.get(query, current_dir, n_results)
            if cached is not None:
                logger.debug(f"Search cache hit for query: {query[:30]}...")
                return cached
        
        # Build BM25 index if not already built
        if not self.bm25.bm25:
            self.build_bm25_index()
        
        # Get results from each component
        semantic_results = self.semantic.search(query, n_results=n_results * 2)
        bm25_results = self.bm25.search(query, n_results=n_results * 2)
        
        # Combine all session IDs
        all_session_ids = set()
        for r in semantic_results:
            all_session_ids.add(r.session_id)
        for r in bm25_results:
            all_session_ids.add(r.session_id)
        
        # Score each session
        combined = []
        for session_id in all_session_ids:
            # Get individual scores (0-1 range)
            sem_score = self._get_score(semantic_results, session_id)
            bm25_score = self._get_score(bm25_results, session_id)
            
            # Get metadata
            metadata = self._get_session_metadata(session_id)
            
            # Calculate recency score
            last_active = None
            if metadata.get("last_active"):
                try:
                    last_active = datetime.fromisoformat(metadata["last_active"])
                except:
                    pass
            rec_score = self.recency.score(last_active)
            
            # Calculate path match score
            path_score = 0.0
            if current_dir and metadata.get("files_in_context"):
                path_score = self.path_matcher.score(
                    metadata["files_in_context"],
                    current_dir
                )
            
            # Weighted combination
            final_score = (
                self.WEIGHTS['semantic'] * sem_score +
                self.WEIGHTS['keyword'] * bm25_score +
                self.WEIGHTS['recency'] * rec_score +
                self.WEIGHTS['path'] * path_score
            )
            
            combined.append(HybridResult(
                session_id=session_id,
                score=final_score,
                breakdown={
                    'semantic': sem_score,
                    'bm25': bm25_score,
                    'recency': rec_score,
                    'path': path_score
                },
                metadata=metadata
            ))
        
        # Sort by final score
        combined.sort(key=lambda x: x.score, reverse=True)
        results = combined[:n_results]
        
        # Cache the results
        if self._enable_cache and self._search_cache:
            self._search_cache.set(results, query, current_dir, n_results)
            logger.debug(f"Cached search results for query: {query[:30]}...")
        
        return results
    
    def invalidate_cache(self, session_id: Optional[str] = None) -> None:
        """Invalidate search cache.
        
        Args:
            session_id: If provided, only invalidate entries containing this session.
                       If None, clear entire cache.
        """
        if not self._search_cache:
            return
        
        if session_id:
            self._search_cache.invalidate_session(session_id)
            logger.debug(f"Invalidated cache for session: {session_id}")
        else:
            self._search_cache.clear()
            logger.debug("Cleared entire search cache")
        
        # Also mark index as needing rebuild
        self._index_built_at = None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._search_cache:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "size": len(self._search_cache._cache),
            "maxsize": self._search_cache.maxsize,
            "ttl": self._search_cache.ttl
        }
    
    def _get_score(self, results: List[SearchResult], session_id: str) -> float:
        """Get score for a session from results list."""
        for r in results:
            if r.session_id == session_id:
                return r.score
        return 0.0
    
    def _get_session_metadata(self, session_id: str) -> Dict[str, Any]:
        """Get cached metadata or fetch from DB."""
        if session_id not in self._session_metadata:
            chunks = self.db.get_session_chunks(session_id)
            if chunks:
                self._session_metadata[session_id] = {
                    "files_in_context": chunks[0].metadata.files_in_context,
                    "technologies": chunks[0].metadata.technologies,
                    "last_active": chunks[0].metadata.last_active
                }
            else:
                self._session_metadata[session_id] = {}
        
        return self._session_metadata[session_id]
