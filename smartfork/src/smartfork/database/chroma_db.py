"""ChromaDB integration for SmartFork."""

import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from .models import Chunk, ChunkMetadata, SearchResult

# Default batch size for operations
DEFAULT_BATCH_SIZE = 100
MAX_BATCH_SIZE = 1000


class ChromaDatabase:
    """Manages ChromaDB connection and operations."""
    
    def __init__(self, db_path: Path):
        """Initialize ChromaDB client.
        
        Args:
            db_path: Path to ChromaDB storage directory
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.collection = self._get_or_create_collection()
        logger.info(f"ChromaDB initialized at {self.db_path}")
    
    def _get_or_create_collection(self):
        """Get or create the sessions collection."""
        return self.client.get_or_create_collection(
            name="sessions",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_chunks(self, chunks: List[Chunk], batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        """Add conversation chunks to the database with batching support.
        
        Args:
            chunks: List of Chunk objects to add
            batch_size: Number of chunks to add in each batch (default: 100)
        """
        import json
        
        if not chunks:
            return
        
        # Validate and clamp batch size
        batch_size = max(10, min(MAX_BATCH_SIZE, batch_size))
        
        total_added = 0
        
        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            ids = [chunk.id for chunk in batch]
            documents = [chunk.content for chunk in batch]
            embeddings = [chunk.embedding for chunk in batch if chunk.embedding]
            
            # Convert metadata to ChromaDB-compatible format (no lists, no None values)
            metadatas = []
            for chunk in batch:
                meta = chunk.metadata.model_dump()
                # Convert lists to JSON strings and remove None values
                cleaned_meta = {}
                for key, value in meta.items():
                    if value is None:
                        continue  # Skip None values
                    elif isinstance(value, list):
                        cleaned_meta[key] = json.dumps(value)
                    else:
                        cleaned_meta[key] = value
                metadatas.append(cleaned_meta)
            
            # If no embeddings provided, ChromaDB will generate them
            if embeddings and len(embeddings) == len(batch):
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
            else:
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
            
            total_added += len(batch)
            logger.debug(f"Added batch of {len(batch)} chunks ({total_added}/{len(chunks)})")
        
        logger.debug(f"Added {total_added} chunks to database in {len(chunks) // batch_size + 1} batches")
    
    def search(
        self, 
        query_embedding: List[float], 
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar chunks.
        
        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            where: Optional filter conditions
            
        Returns:
            List of SearchResult objects
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        return self._format_results(results)
    
    def search_by_text(
        self,
        query_text: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search using text query (ChromaDB generates embedding).
        
        Args:
            query_text: Text query
            n_results: Number of results to return
            where: Optional filter conditions
            
        Returns:
            List of SearchResult objects
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        return self._format_results(results)
    
    def _format_results(self, results: Dict[str, Any]) -> List[SearchResult]:
        """Format ChromaDB results into SearchResult objects."""
        search_results = []
        
        if not results or not results.get('ids'):
            return search_results
        
        ids = results['ids'][0]
        documents = results.get('documents', [[]])[0] or []
        distances = results.get('distances', [[]])[0] or []
        metadatas = results.get('metadatas', [[]])[0] or []
        
        for i, session_id in enumerate(ids):
            # Convert distance to similarity score (cosine distance -> similarity)
            # Distance is 0-2 for cosine, where 0 = identical, 2 = opposite
            score = 1.0 - (distances[i] / 2.0) if i < len(distances) else 0.0
            
            search_results.append(SearchResult(
                session_id=session_id.split('_')[0] if '_' in session_id else session_id,
                content=documents[i] if i < len(documents) else "",
                score=max(0.0, score),
                metadata=metadatas[i] if i < len(metadatas) else {}
            ))
        
        return search_results
    
    def delete_session(self, session_id: str) -> None:
        """Delete all chunks for a session.
        
        Args:
            session_id: Session ID to delete
        """
        self.collection.delete(
            where={"session_id": session_id}
        )
        logger.debug(f"Deleted session {session_id}")
    
    def get_session_chunks(self, session_id: str) -> List[Chunk]:
        """Get all chunks for a specific session.
        
        Args:
            session_id: Session ID to retrieve
            
        Returns:
            List of Chunk objects
        """
        import json
        
        results = self.collection.get(
            where={"session_id": session_id},
            include=["documents", "metadatas"]
        )
        
        chunks = []
        if results and results.get('ids'):
            for i, chunk_id in enumerate(results['ids']):
                meta_dict = results['metadatas'][i]
                # Deserialize JSON strings back to lists
                for key, value in meta_dict.items():
                    if isinstance(value, str) and value.startswith('['):
                        try:
                            meta_dict[key] = json.loads(value)
                        except json.JSONDecodeError:
                            pass
                metadata = ChunkMetadata(**meta_dict)
                chunks.append(Chunk(
                    id=chunk_id,
                    content=results['documents'][i],
                    metadata=metadata
                ))
        
        return chunks
    
    def get_session_count(self) -> int:
        """Get total number of indexed chunks."""
        return self.collection.count()
    
    def get_unique_sessions(self) -> List[str]:
        """Get list of unique session IDs in the database."""
        results = self.collection.get(include=[])
        if not results or not results.get('ids'):
            return []
        
        session_ids = set()
        for chunk_id in results['ids']:
            # Extract session_id from chunk_id (format: session_id_chunk_index)
            session_id = chunk_id.rsplit('_', 1)[0] if '_' in chunk_id else chunk_id
            session_ids.add(session_id)
        
        return list(session_ids)
    
    def reset(self) -> None:
        """Clear all data (use with caution)."""
        self.client.reset()
        self.collection = self._get_or_create_collection()
        logger.warning("Database reset complete")
