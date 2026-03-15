"""Chunk ranking based on query relevance using multiple signals."""

from typing import List, Optional, Dict
from datetime import datetime
import math

from ..database.chunk_models import EnhancedChunk, ChunkSearchResult
from .query_parser import QueryIntent


class ChunkRanker:
    """
    Ranks chunks based on query relevance using multiple signals.
    
    This ranker combines multiple scoring signals to determine the relevance
    of chunks to a search query. Each signal contributes to a weighted final
    score that can be used to sort chunks by relevance.
    
    Signals:
        - Semantic similarity (50%): Vector similarity between query and chunk
        - File match (20%): Overlap between query file references and chunk files
        - Keyword match (15%): Overlap between query technologies and chunk keywords
        - Content type (10%): Preference for code blocks when query indicates code preference
        - Recency (5%): Exponential decay based on chunk age
    
    Example:
        >>> ranker = ChunkRanker()
        >>> results = ranker.rank_chunks(chunks, query_intent, query_embedding)
        >>> for result in results[:5]:
        ...     print(f"{result.chunk_id}: {result.score:.2f}")
    """
    
    # Weight configuration for each scoring signal
    WEIGHTS: Dict[str, float] = {
        'semantic': 0.50,
        'file_match': 0.20,
        'keyword_match': 0.15,
        'content_type': 0.10,
        'recency': 0.05
    }
    
    # Recency scoring constants
    RECENCY_MAX_AGE_DAYS: int = 7  # Chunks newer than this get max score
    RECENCY_HALF_LIFE_DAYS: int = 30  # Decay rate: 50% per month after max age
    
    def __init__(self):
        """Initialize the chunk ranker with default weights."""
        pass
    
    def rank_chunks(
        self,
        chunks: List[EnhancedChunk],
        query_intent: QueryIntent,
        query_embedding: Optional[List[float]] = None
    ) -> List[ChunkSearchResult]:
        """
        Rank chunks by relevance to query.
        
        Calculates a weighted score for each chunk based on multiple signals
        and returns results sorted by relevance (highest first).
        
        Args:
            chunks: List of candidate chunks to rank
            query_intent: Parsed query intent with extracted entities
            query_embedding: Optional pre-computed query embedding for semantic scoring
            
        Returns:
            List of ChunkSearchResult sorted by relevance score (descending)
            
        Example:
            >>> chunks = [chunk1, chunk2, chunk3]
            >>> intent = QueryParser().parse("fix auth in app.py")
            >>> results = ranker.rank_chunks(chunks, intent)
            >>> best_chunk = results[0] if results else None
        """
        if not chunks:
            return []
        
        results: List[ChunkSearchResult] = []
        
        for chunk in chunks:
            # Calculate individual signal scores
            scores = self._calculate_scores(chunk, query_intent, query_embedding)
            
            # Compute weighted final score
            final_score = sum(
                scores[signal] * weight
                for signal, weight in self.WEIGHTS.items()
            )
            
            results.append(ChunkSearchResult(
                chunk_id=chunk.id,
                session_id=chunk.metadata.session_id,
                content=chunk.content,
                score=final_score,
                breakdown=scores,
                metadata=chunk.metadata
            ))
        
        # Sort by score descending (highest relevance first)
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def _calculate_scores(
        self,
        chunk: EnhancedChunk,
        query_intent: QueryIntent,
        query_embedding: Optional[List[float]]
    ) -> Dict[str, float]:
        """
        Calculate individual signal scores for a chunk.
        
        Computes scores for each ranking signal:
        - semantic: Similarity between query and chunk embeddings
        - file_match: Overlap between query files and chunk files
        - keyword_match: Overlap between query technologies and chunk keywords
        - content_type: Boost for code blocks when query.prefer_code is True
        - recency: Exponential decay based on chunk age
        
        Args:
            chunk: The chunk to score
            query_intent: Parsed query intent
            query_embedding: Optional query embedding for semantic similarity
            
        Returns:
            Dictionary mapping signal names to scores (0.0 to 1.0)
        """
        scores: Dict[str, float] = {}
        
        # Semantic score
        scores['semantic'] = self._calculate_semantic_score(chunk, query_embedding)
        
        # File match score
        scores['file_match'] = self._calculate_file_match_score(chunk, query_intent)
        
        # Keyword match score
        scores['keyword_match'] = self._calculate_keyword_match_score(chunk, query_intent)
        
        # Content type score
        scores['content_type'] = self._calculate_content_type_score(chunk, query_intent)
        
        # Recency score
        scores['recency'] = self._calculate_recency_score(chunk)
        
        return scores
    
    def _calculate_semantic_score(
        self,
        chunk: EnhancedChunk,
        query_embedding: Optional[List[float]]
    ) -> float:
        """
        Calculate semantic similarity score.
        
        If a query embedding is provided and the chunk has an embedding,
        computes cosine similarity. Otherwise returns a default neutral score.
        
        Args:
            chunk: The chunk to score
            query_embedding: Optional query embedding vector
            
        Returns:
            Semantic similarity score (0.0 to 1.0)
        """
        if query_embedding and chunk.embedding:
            # Compute cosine similarity
            try:
                dot_product = sum(a * b for a, b in zip(query_embedding, chunk.embedding))
                query_norm = math.sqrt(sum(x * x for x in query_embedding))
                chunk_norm = math.sqrt(sum(x * x for x in chunk.embedding))
                
                if query_norm > 0 and chunk_norm > 0:
                    similarity = dot_product / (query_norm * chunk_norm)
                    # Normalize from [-1, 1] to [0, 1]
                    return (similarity + 1) / 2
            except (ValueError, TypeError):
                pass
        
        # Default neutral score if no embeddings available
        return 0.5
    
    def _calculate_file_match_score(
        self,
        chunk: EnhancedChunk,
        query_intent: QueryIntent
    ) -> float:
        """
        Calculate file match score based on file reference overlap.
        
        Compares files mentioned in the query with files mentioned in the chunk.
        Returns a normalized score based on the proportion of query files found
        in the chunk.
        
        Args:
            chunk: The chunk to score
            query_intent: Parsed query intent with file references
            
        Returns:
            File match score (0.0 to 1.0)
        """
        if not query_intent.file_references:
            # Neutral score if no files mentioned in query
            return 0.5
        
        chunk_files = set(chunk.metadata.files_mentioned)
        query_files = set(query_intent.file_references)
        
        if not chunk_files:
            # No files in chunk to match against
            return 0.0
        
        # Calculate overlap
        overlap = len(chunk_files.intersection(query_files))
        
        if overlap == 0:
            return 0.0
        
        # Normalize by number of query files (max 1.0)
        return min(overlap / len(query_files), 1.0)
    
    def _calculate_keyword_match_score(
        self,
        chunk: EnhancedChunk,
        query_intent: QueryIntent
    ) -> float:
        """
        Calculate keyword match score based on technology overlap.
        
        Compares technologies mentioned in the query with keywords and entities
        in the chunk. Returns a normalized score based on overlap.
        
        Args:
            chunk: The chunk to score
            query_intent: Parsed query intent with technologies
            
        Returns:
            Keyword match score (0.0 to 1.0)
        """
        if not query_intent.technologies:
            # Neutral score if no technologies mentioned in query
            return 0.5
        
        # Combine chunk keywords and entities
        chunk_keywords = set(
            (chunk.metadata.keywords or []) + 
            (chunk.metadata.entities or [])
        )
        
        # Normalize to lowercase for comparison
        chunk_keywords = {k.lower() for k in chunk_keywords}
        query_keywords = {t.lower() for t in query_intent.technologies}
        
        if not chunk_keywords:
            # No keywords in chunk to match against
            return 0.0
        
        # Calculate overlap
        overlap = len(chunk_keywords.intersection(query_keywords))
        
        if overlap == 0:
            return 0.0
        
        # Normalize by number of query keywords (max 1.0)
        return min(overlap / len(query_keywords), 1.0)
    
    def _calculate_content_type_score(
        self,
        chunk: EnhancedChunk,
        query_intent: QueryIntent
    ) -> float:
        """
        Calculate content type match score.
        
        When query indicates a preference for code (e.g., asking for implementation
        examples), boosts chunks containing code blocks.
        
        Args:
            chunk: The chunk to score
            query_intent: Parsed query intent with prefer_code flag
            
        Returns:
            Content type score (0.0 to 1.0)
        """
        if not query_intent.prefer_code:
            # Neutral score if no code preference
            return 0.5
        
        content_type = chunk.metadata.content_type
        
        if content_type == 'code_block':
            # Full boost for code blocks
            return 1.0
        elif content_type == 'mixed':
            # Partial boost for mixed content
            return 0.7
        else:
            # Reduced score for non-code content
            return 0.3
    
    def _calculate_recency_score(self, chunk: EnhancedChunk) -> float:
        """
        Calculate recency score with exponential decay.
        
        Chunks newer than RECENCY_MAX_AGE_DAYS get maximum score (1.0).
        Older chunks decay exponentially with RECENCY_HALF_LIFE_DAYS.
        
        Args:
            chunk: The chunk to score
            
        Returns:
            Recency score (0.0 to 1.0)
        """
        if not chunk.metadata.last_active:
            # Neutral score if no timestamp available
            return 0.5
        
        try:
            chunk_date = datetime.fromisoformat(chunk.metadata.last_active)
            now = datetime.now()
            age_days = (now - chunk_date).days
            
            if age_days <= self.RECENCY_MAX_AGE_DAYS:
                # Recent chunks get max score
                return 1.0
            
            # Exponential decay for older chunks
            decay_days = age_days - self.RECENCY_MAX_AGE_DAYS
            score = 0.5 ** (decay_days / self.RECENCY_HALF_LIFE_DAYS)
            
            # Ensure minimum score of 0.1 to avoid completely excluding old chunks
            return max(score, 0.1)
            
        except (ValueError, TypeError):
            # Invalid timestamp format
            return 0.5
    
    def get_score_breakdown(
        self,
        result: ChunkSearchResult
    ) -> Dict[str, Dict[str, float]]:
        """
        Get a detailed breakdown of how a score was computed.
        
        Useful for debugging and understanding why a chunk received
        its particular ranking.
        
        Args:
            result: A ChunkSearchResult with breakdown information
            
        Returns:
            Dictionary with raw scores, weights, and weighted contributions
        """
        breakdown = {
            'final_score': result.score,
            'signals': {}
        }
        
        for signal, raw_score in result.breakdown.items():
            weight = self.WEIGHTS.get(signal, 0.0)
            weighted_score = raw_score * weight
            
            breakdown['signals'][signal] = {
                'raw_score': raw_score,
                'weight': weight,
                'weighted_contribution': weighted_score,
                'percentage': (weighted_score / result.score * 100) if result.score > 0 else 0
            }
        
        return breakdown
