"""Enhanced chunk models with query-aware metadata for SmartFork."""

from typing import List, Dict, Any, Optional, Literal, Tuple
from datetime import datetime
from pydantic import BaseModel, Field


class MessageRange(BaseModel):
    """Range of messages included in a chunk."""
    start_index: int  # Index in conversation list
    end_index: int
    start_timestamp: Optional[int] = None
    end_timestamp: Optional[int] = None


class ChunkContentType:
    """Content type classification constants."""
    USER_QUERY = "user_query"
    ASSISTANT_RESPONSE = "assistant_response"
    CODE_BLOCK = "code_block"
    MIXED = "mixed"
    FILE_OPERATION = "file_operation"


class EnhancedChunkMetadata(BaseModel):
    """
    Rich metadata for query-aware chunk retrieval.
    
    This replaces the old ChunkMetadata with per-chunk file tracking
    and content classification for intelligent retrieval.
    """
    
    # Core identification
    session_id: str
    task_id: str
    chunk_index: int
    message_range: Optional[MessageRange] = None
    
    # Content classification
    primary_role: Literal["user", "assistant", "system", "tool"] = "assistant"
    content_type: Literal["user_query", "assistant_response", "code_block", "mixed", "file_operation"] = "mixed"
    
    # Content analysis (populated during indexing)
    summary: Optional[str] = None  # First sentence or extracted summary
    primary_topic: Optional[str] = None
    code_language: Optional[str] = None
    
    # Per-chunk file tracking (CRITICAL FEATURE)
    # These are files mentioned IN THIS SPECIFIC CHUNK, not the whole session
    files_mentioned: List[str] = Field(default_factory=list)
    files_modified: List[str] = Field(default_factory=list)
    
    # Token and size metrics for budget management
    token_count: int = 0
    char_count: int = 0
    
    # Temporal - specific to this chunk's messages
    last_active: Optional[str] = None  # ISO timestamp of last message in chunk
    
    # Search optimization
    keywords: List[str] = Field(default_factory=list)  # Extracted keywords
    entities: List[str] = Field(default_factory=list)  # Named entities
    
    class Config:
        # Allow extra fields for backward compatibility
        extra = "allow"


class EnhancedChunk(BaseModel):
    """Chunk with enhanced metadata for smart retrieval."""
    id: str
    content: str
    embedding: List[float] = Field(default_factory=list)
    metadata: EnhancedChunkMetadata
    
    def estimate_tokens(self) -> int:
        """
        Estimate token count for context budgeting.
        
        Uses metadata if available, otherwise estimates from content.
        GPT-style tokenization: ~0.75 tokens per word on average.
        """
        if self.metadata.token_count and self.metadata.token_count > 0:
            return self.metadata.token_count
        
        # Rough estimate: ~0.75 tokens per word
        word_count = len(self.content.split())
        return int(word_count * 0.75)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata.model_dump()
        }


class ChunkSearchResult(BaseModel):
    """Result from chunk-level search."""
    chunk_id: str
    session_id: str
    content: str
    score: float
    breakdown: Dict[str, float]  # How score was computed
    metadata: EnhancedChunkMetadata
    
    def to_context_string(self, max_chars: int = 1000) -> str:
        """
        Format for inclusion in context window.
        
        Args:
            max_chars: Maximum characters to include
            
        Returns:
            Formatted string with role prefix
        """
        role = self.metadata.primary_role
        content = self.content[:max_chars]
        if len(self.content) > max_chars:
            content += "..."
        return f"[{role}]: {content}"
    
    def get_token_estimate(self) -> int:
        """Get token estimate for budget management."""
        return self.metadata.token_count or int(len(self.content.split()) * 0.75)


class TokenBudget(BaseModel):
    """Token budget configuration for context extraction."""
    max_total_tokens: int = 2000
    max_chunks: int = 5
    min_chunk_score: float = 0.3
    prioritize_recent: bool = True
    include_code: bool = True
    summary_ratio: float = 0.3  # % of budget for summary
    
    def validate_budget(self) -> bool:
        """Validate budget configuration."""
        return (
            self.max_total_tokens > 0 and
            self.max_chunks > 0 and
            0 <= self.min_chunk_score <= 1 and
            0 <= self.summary_ratio <= 1
        )


# Backward compatibility aliases
# These allow existing code to work while we transition
ChunkMetadata = EnhancedChunkMetadata
Chunk = EnhancedChunk


# Utility functions for chunk manipulation
def calculate_total_tokens(chunks: List[EnhancedChunk]) -> int:
    """Calculate total token count for a list of chunks."""
    return sum(chunk.estimate_tokens() for chunk in chunks)


def filter_chunks_by_budget(
    chunks: List[EnhancedChunk], 
    budget: TokenBudget
) -> List[EnhancedChunk]:
    """
    Filter chunks to fit within token budget.
    
    Args:
        chunks: List of chunks to filter
        budget: Token budget constraints
        
    Returns:
        Filtered list of chunks within budget
    """
    filtered = []
    total_tokens = 0
    
    for chunk in chunks:
        chunk_tokens = chunk.estimate_tokens()
        
        # Check if adding this chunk would exceed budget
        if total_tokens + chunk_tokens > budget.max_total_tokens:
            break
        
        # Check max chunks limit
        if len(filtered) >= budget.max_chunks:
            break
        
        filtered.append(chunk)
        total_tokens += chunk_tokens
    
    return filtered


def merge_chunk_metadata(chunks: List[EnhancedChunk]) -> Dict[str, Any]:
    """
    Merge metadata from multiple chunks.
    
    Useful for generating session-level summaries from chunk data.
    
    Args:
        chunks: List of chunks to merge
        
    Returns:
        Merged metadata dictionary
    """
    if not chunks:
        return {}
    
    all_files = set()
    all_keywords = set()
    all_entities = set()
    total_tokens = 0
    
    for chunk in chunks:
        all_files.update(chunk.metadata.files_mentioned)
        all_keywords.update(chunk.metadata.keywords)
        all_entities.update(chunk.metadata.entities)
        total_tokens += chunk.metadata.token_count
    
    return {
        "session_id": chunks[0].metadata.session_id,
        "total_chunks": len(chunks),
        "total_tokens": total_tokens,
        "files_mentioned": sorted(list(all_files)),
        "keywords": sorted(list(all_keywords))[:20],  # Top 20
        "entities": sorted(list(all_entities))[:20],  # Top 20
        "time_range": {
            "start": min((c.metadata.last_active for c in chunks if c.metadata.last_active), default=None),
            "end": max((c.metadata.last_active for c in chunks if c.metadata.last_active), default=None)
        }
    }
