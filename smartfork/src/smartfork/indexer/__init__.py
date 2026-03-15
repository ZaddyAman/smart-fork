"""Indexer module for SmartFork."""

from .parser import KiloCodeParser
from .watcher import TranscriptWatcher
from .indexer import FullIndexer, IncrementalIndexer

# New message-aware chunkers
from .chunkers import (
    MessageBoundaryChunker,
    CodeAwareChunker,
    ChunkingConfig,
    create_chunker
)

__all__ = [
    # Core components
    "KiloCodeParser",
    "TranscriptWatcher",
    "FullIndexer",
    "IncrementalIndexer",
    
    # New chunking strategies
    "MessageBoundaryChunker",
    "CodeAwareChunker",
    "ChunkingConfig",
    "create_chunker",
]
