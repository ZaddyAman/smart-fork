"""Indexer for Kilo Code sessions."""

from pathlib import Path
from typing import List
from tqdm import tqdm
from loguru import logger

from ..database.chroma_db import ChromaDatabase
from ..database.models import TaskSession, IndexingResult
from ..database.chunk_models import EnhancedChunk, EnhancedChunkMetadata
from .parser import KiloCodeParser
from .chunkers import MessageBoundaryChunker, ChunkingConfig
from ..intelligence.titling import TitleGenerator, TitleManager


class FullIndexer:
    """Performs full re-indexing of sessions with message-aware chunking."""
    
    def __init__(
        self,
        db: ChromaDatabase,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        generate_titles: bool = True,
        batch_size: int = 100,
        chunking_strategy: str = "message_boundary"
    ):
        """Initialize the indexer.
        
        Args:
            db: ChromaDatabase instance
            chunk_size: Size of chunks in tokens/words
            chunk_overlap: Overlap between chunks (legacy, not used with message chunking)
            generate_titles: Whether to auto-generate session titles
            batch_size: Batch size for database operations
            chunking_strategy: Chunking strategy to use ("message_boundary" or "code_aware")
        """
        self.db = db
        self.parser = KiloCodeParser()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap  # Legacy, kept for compatibility
        self.generate_titles = generate_titles
        self.batch_size = batch_size
        self.title_manager = TitleManager(db) if generate_titles else None
        self._pending_chunks: List[EnhancedChunk] = []  # Buffer for batching
        
        # Initialize the new message-aware chunker
        chunking_config = ChunkingConfig(
            max_tokens_per_chunk=chunk_size,
            extract_keywords=True,
            extract_files=True
        )
        self.chunker = MessageBoundaryChunker(chunking_config)
    
    def index_all_sessions(self, tasks_path: Path, progress: bool = True) -> IndexingResult:
        """Index all sessions in the tasks directory.
        
        Args:
            tasks_path: Path to tasks directory
            progress: Show progress bar
            
        Returns:
            IndexingResult with counts
        """
        tasks_path = Path(tasks_path)
        
        if not tasks_path.exists():
            logger.error(f"Tasks path does not exist: {tasks_path}")
            return IndexingResult()
        
        # Get all session directories
        sessions = [
            item for item in tasks_path.iterdir()
            if item.is_dir() and (item / "api_conversation_history.json").exists()
        ]
        
        result = IndexingResult()
        
        iterator = tqdm(sessions, desc="Indexing sessions") if progress else sessions
        
        for session_path in iterator:
            try:
                # Flush on last session
                is_last = session_path == sessions[-1] if sessions else True
                chunks = self.index_session(session_path, flush=is_last)
                result.indexed += 1
                result.chunks_created += chunks
            except Exception as e:
                logger.error(f"Failed to index {session_path.name}: {e}")
                result.failed += 1
        
        # Finalize to flush any remaining chunks
        self.finalize()
        
        logger.info(f"Indexing complete: {result.indexed} indexed, {result.failed} failed, {result.chunks_created} chunks")
        return result
    
    def index_session(self, session_path: Path, flush: bool = False) -> int:
        """Index a single session with optional batching.
        
        Args:
            session_path: Path to session directory
            flush: If True, immediately flush pending chunks to DB
            
        Returns:
            Number of chunks created
        """
        task_id = session_path.name
        
        # Delete existing data for this session
        self.db.delete_session(task_id)
        
        # Parse session
        session = self.parser.parse_task_directory(session_path)
        if not session:
            logger.warning(f"Could not parse session: {task_id}")
            return 0
        
        # Chunk the conversation
        chunks = self._create_chunks(session)
        
        if not chunks:
            return 0
        
        # Add to pending buffer for batch processing
        self._pending_chunks.extend(chunks)
        
        # Flush if buffer is full or flush requested
        if flush or len(self._pending_chunks) >= self.batch_size:
            self._flush_pending_chunks()
        
        logger.debug(f"Indexed {len(chunks)} chunks for session {task_id[:8]}...")
        return len(chunks)
    
    def _flush_pending_chunks(self) -> None:
        """Flush pending chunks to database in batches."""
        if not self._pending_chunks:
            return
        
        # Store in database with batching
        self.db.add_chunks(self._pending_chunks, batch_size=self.batch_size)
        logger.debug(f"Flushed {len(self._pending_chunks)} pending chunks to database")
        self._pending_chunks.clear()
    
    def finalize(self) -> None:
        """Finalize indexing by flushing any remaining pending chunks."""
        self._flush_pending_chunks()
    
    def _create_chunks(self, session: TaskSession) -> List[EnhancedChunk]:
        """
        Create chunks from session conversation using message-aware chunking.
        
        Uses MessageBoundaryChunker which:
        - Preserves message boundaries (never splits messages)
        - Tracks files mentioned per chunk
        - Classifies content type (code/text/mixed)
        - Extracts keywords and entities
        
        Args:
            session: TaskSession to chunk
            
        Returns:
            List of EnhancedChunk objects with rich metadata
        """
        # Use the new message-aware chunker
        chunks = self.chunker.chunk_session(session)
        
        # Add session title if generation is enabled
        if self.title_manager and chunks:
            session_title = self.title_manager.generate_and_store_title(session)
            for chunk in chunks:
                chunk.metadata.session_title = session_title
        
        return chunks
    
    # Note: _split_text method removed - using MessageBoundaryChunker instead
    # The old word-based splitting is replaced with message-aware chunking
    # which preserves conversation structure and tracks per-chunk metadata


class IncrementalIndexer:
    """Performs incremental indexing of new/changed sessions."""
    
    def __init__(self, db: ChromaDatabase, chunk_size: int = 512):
        """Initialize the incremental indexer.
        
        Args:
            db: ChromaDatabase instance
            chunk_size: Maximum tokens per chunk for new sessions
        """
        self.db = db
        self.parser = KiloCodeParser()
        self.full_indexer = FullIndexer(db, chunk_size=chunk_size)
    
    def on_session_changed(self, session_id: str, session_path: Path) -> bool:
        """Handle a session change event.
        
        Args:
            session_id: Session ID
            session_path: Path to session directory
            
        Returns:
            True if indexed successfully
        """
        try:
            # For now, re-index the entire session
            # In future, we can implement partial updates
            chunks = self.full_indexer.index_session(session_path)
            logger.info(f"Incrementally indexed session {session_id[:8]}... ({chunks} chunks)")
            return True
        except Exception as e:
            logger.error(f"Failed to incrementally index {session_id}: {e}")
            return False
