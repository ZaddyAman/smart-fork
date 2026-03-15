"""Semantic search engine for SmartFork."""

from typing import List, Optional
from pathlib import Path
from loguru import logger

from ..database.chroma_db import ChromaDatabase
from ..database.models import SearchResult


class SemanticSearchEngine:
    """Semantic search using vector embeddings."""
    
    def __init__(self, db: ChromaDatabase):
        """Initialize the search engine.
        
        Args:
            db: ChromaDatabase instance
        """
        self.db = db
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        files: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Search for sessions matching the query.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            files: Filter by files in context
            
        Returns:
            List of SearchResult objects
        """
        # Build filter conditions
        where = self._build_filter(files)
        
        # Search database using ChromaDB's built-in embedding
        results = self.db.search_by_text(query, n_results, where)
        
        # Deduplicate by session_id (keep highest scoring)
        seen_sessions = {}
        for r in results:
            if r.session_id not in seen_sessions or r.score > seen_sessions[r.session_id].score:
                seen_sessions[r.session_id] = r
        
        # Sort by score
        deduplicated = sorted(seen_sessions.values(), key=lambda x: x.score, reverse=True)
        
        logger.debug(f"Search for '{query}' returned {len(deduplicated)} unique sessions")
        return deduplicated[:n_results]
    
    def search_similar(
        self,
        session_id: str,
        n_results: int = 5
    ) -> List[SearchResult]:
        """Find sessions similar to a given session.
        
        Args:
            session_id: Session ID to find similar sessions for
            n_results: Number of results to return
            
        Returns:
            List of similar SearchResult objects
        """
        # Get chunks from the session
        chunks = self.db.get_session_chunks(session_id)
        
        if not chunks:
            logger.warning(f"No chunks found for session {session_id}")
            return []
        
        # Use first chunk as query
        query_text = chunks[0].content[:500]  # Truncate for efficiency
        
        # Search excluding the source session
        results = self.search(query_text, n_results=n_results + 1)
        
        # Filter out the source session
        filtered = [r for r in results if r.session_id != session_id]
        
        return filtered[:n_results]
    
    def _build_filter(
        self,
        files: Optional[List[str]]
    ) -> Optional[dict]:
        """Build ChromaDB filter conditions.
        
        Args:
            files: Files to filter by
            
        Returns:
            Filter dict or None
        """
        if files:
            # Check if any of the specified files are in context
            return {"files_in_context": {"$in": files}}
        
        return None
