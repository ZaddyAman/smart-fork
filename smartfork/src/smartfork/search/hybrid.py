"""Hybrid search engine combining semantic, keyword, recency, and path signals."""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import re
from collections import defaultdict
from loguru import logger

from rank_bm25 import BM25Okapi
import numpy as np

from ..database.chroma_db import ChromaDatabase
from ..database.models import HybridResult, SearchResult
from .semantic import SemanticSearchEngine


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
    
    def __init__(self, db: ChromaDatabase):
        """Initialize the hybrid search engine.
        
        Args:
            db: ChromaDatabase instance
        """
        self.db = db
        self.semantic = SemanticSearchEngine(db)
        self.bm25 = BM25Search()
        self.recency = RecencyScorer()
        self.path_matcher = PathMatcher()
        
        # Cache for session metadata
        self._session_metadata: Dict[str, Dict[str, Any]] = {}
    
    def build_bm25_index(self) -> None:
        """Build BM25 index from all sessions in the database."""
        sessions = {}
        
        for session_id in self.db.get_unique_sessions():
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
        
        self.bm25.build_index(sessions)
        logger.info(f"Hybrid search index built with {len(sessions)} sessions")
    
    def search(
        self,
        query: str,
        current_dir: Optional[str] = None,
        n_results: int = 10
    ) -> List[HybridResult]:
        """Perform hybrid search.
        
        Args:
            query: Search query
            current_dir: Current working directory for path matching
            n_results: Number of results to return
            
        Returns:
            List of HybridResult objects sorted by score
        """
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
        return combined[:n_results]
    
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
