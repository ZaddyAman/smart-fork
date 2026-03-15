"""Database module for SmartFork."""

# Backward compatibility: Chunk and ChunkMetadata are now aliases for Enhanced versions
from .chroma_db import ChromaDatabase
from .models import (
    TaskMetadata, ConversationMessage, UIMessage, TaskSession,
    SearchResult, IndexingResult
)

# New enhanced chunk models (with backward compatibility)
from .chunk_models import (
    EnhancedChunk, EnhancedChunkMetadata,
    Chunk, ChunkMetadata,  # Aliases for backward compatibility
    MessageRange, ChunkContentType, ChunkSearchResult, TokenBudget
)

__all__ = [
    # Core database
    "ChromaDatabase",
    
    # Enhanced chunk models (new)
    "EnhancedChunk", "EnhancedChunkMetadata",
    "MessageRange", "ChunkContentType", "ChunkSearchResult", "TokenBudget",
    
    # Backward compatibility aliases
    "Chunk", "ChunkMetadata",
    
    # Legacy models
    "TaskMetadata", "ConversationMessage", "UIMessage",
    "TaskSession", "SearchResult", "IndexingResult"
]
