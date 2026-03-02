"""Indexer for Kilo Code sessions."""

from pathlib import Path
from typing import List
from tqdm import tqdm
from loguru import logger

from ..database.chroma_db import ChromaDatabase
from ..database.models import TaskSession, Chunk, ChunkMetadata, IndexingResult
from .parser import KiloCodeParser


class FullIndexer:
    """Performs full re-indexing of sessions."""
    
    def __init__(
        self,
        db: ChromaDatabase,
        chunk_size: int = 512,
        chunk_overlap: int = 128
    ):
        """Initialize the indexer.
        
        Args:
            db: ChromaDatabase instance
            chunk_size: Size of chunks in tokens/words
            chunk_overlap: Overlap between chunks
        """
        self.db = db
        self.parser = KiloCodeParser()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
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
                chunks = self.index_session(session_path)
                result.indexed += 1
                result.chunks_created += chunks
            except Exception as e:
                logger.error(f"Failed to index {session_path.name}: {e}")
                result.failed += 1
        
        logger.info(f"Indexing complete: {result.indexed} indexed, {result.failed} failed, {result.chunks_created} chunks")
        return result
    
    def index_session(self, session_path: Path) -> int:
        """Index a single session.
        
        Args:
            session_path: Path to session directory
            
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
        
        # Store in database (without embeddings - ChromaDB will generate)
        self.db.add_chunks(chunks)
        
        logger.debug(f"Indexed {len(chunks)} chunks for session {task_id[:8]}...")
        return len(chunks)
    
    def _create_chunks(self, session: TaskSession) -> List[Chunk]:
        """Create chunks from session conversation.
        
        Args:
            session: TaskSession to chunk
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        full_text = session.get_full_text()
        
        if not full_text.strip():
            return chunks
        
        # Simple word-based chunking strategy
        text_chunks = self._split_text(full_text, self.chunk_size, self.chunk_overlap)
        
        # Detect technologies in the full text
        technologies = self.parser.detect_technologies(full_text)
        
        for idx, text in enumerate(text_chunks):
            chunk_id = f"{session.task_id}_{idx}"
            chunks.append(Chunk(
                id=chunk_id,
                content=text,
                metadata=ChunkMetadata(
                    session_id=session.task_id,
                    task_id=session.task_id,
                    chunk_index=idx,
                    files_in_context=session.metadata.files_in_context,
                    message_type="mixed",
                    technologies=technologies
                )
            ))
        
        return chunks
    
    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks.
        
        Args:
            text: Text to split
            chunk_size: Target chunk size in words
            overlap: Number of words to overlap
            
        Returns:
            List of text chunks
        """
        words = text.split()
        
        if len(words) <= chunk_size:
            return [text] if text.strip() else []
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            
            # Move forward with overlap
            start += chunk_size - overlap
            
            # Avoid infinite loop on small chunks
            if start >= end:
                break
        
        return chunks


class IncrementalIndexer:
    """Performs incremental indexing of new/changed sessions."""
    
    def __init__(self, db: ChromaDatabase):
        """Initialize the incremental indexer.
        
        Args:
            db: ChromaDatabase instance
        """
        self.db = db
        self.parser = KiloCodeParser()
        self.full_indexer = FullIndexer(db)
    
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
