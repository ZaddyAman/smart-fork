"""Multi-collection ChromaDB vector index (v2).

Unlike v1 which had one collection with everything, v2 uses separate
collections per document type so different doc types can be weighted
differently at search time based on query intent.

Collections:
- v2_task_docs: task descriptions with contextual prefix
- v2_summary_docs: LLM-generated session summaries
- v2_reasoning_docs: AI reasoning/decision blocks
"""

import chromadb
from pathlib import Path
from typing import List, Optional, Dict, Any
from loguru import logger

from ..database.models import SessionDocument, VectorResult
from ..search.embedder import EmbeddingProvider
from ..indexer.contextual_chunker import ContextualChunker


# Collection names
TASK_COLLECTION = "v2_task_docs"
SUMMARY_COLLECTION = "v2_summary_docs"
REASONING_COLLECTION = "v2_reasoning_docs"


class VectorIndex:
    """Manages ChromaDB collections for v2 index schema.
    
    Three separate collections allow intent-aware weighting:
    - decision_hunting → boost reasoning_docs (0.6)
    - implementation_lookup → boost task_docs (0.5) 
    - vague_memory → boost summary_docs (0.6)
    
    Usage:
        index = VectorIndex(db_path, embedder)
        index.index_session(session_doc)
        results = index.search(query_embedding, "reasoning", session_ids=candidates)
    """
    
    def __init__(self, db_path: Path, embedder: EmbeddingProvider):
        """Initialize ChromaDB client with separate collections.
        
        Args:
            db_path: Path to ChromaDB storage directory
            embedder: EmbeddingProvider instance for generating embeddings
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.embedder = embedder
        self.chunker = ContextualChunker()
        
        # Get or create collections
        self.task_collection = self.client.get_or_create_collection(
            name=TASK_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
        self.summary_collection = self.client.get_or_create_collection(
            name=SUMMARY_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
        self.reasoning_collection = self.client.get_or_create_collection(
            name=REASONING_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.debug(f"VectorIndex initialized at {db_path}")
    
    def index_session(self, doc: SessionDocument) -> Dict[str, int]:
        """Index all embeddable documents for a session.
        
        Uses batch embedding for efficiency — one API call per doc type
        instead of one per chunk.
        
        Args:
            doc: SessionDocument with all fields populated
        
        Returns:
            Dict with counts: {"task": N, "summary": N, "reasoning": N}
        """
        counts = {"task": 0, "summary": 0, "reasoning": 0}
        
        # First, delete any existing data for this session
        self.delete_session(doc.session_id)
        
        # Build all documents first (CPU-only, fast)
        task_doc = self.chunker.build_task_doc(doc)
        summary_doc = self.chunker.build_summary_doc(doc)
        reasoning_docs = self.chunker.build_reasoning_docs(doc)
        
        # ── BATCH EMBED + STORE: task_doc ──
        if task_doc:
            try:
                embedding = self.embedder.embed(task_doc, "task_doc")
                self.task_collection.add(
                    ids=[f"{doc.session_id}_task_0"],
                    embeddings=[embedding],
                    documents=[task_doc],
                    metadatas=[{
                        "session_id": doc.session_id,
                        "doc_type": "task_doc",
                        "project_name": doc.project_name,
                        "chunk_index": 0,
                    }]
                )
                counts["task"] = 1
            except Exception as e:
                logger.warning(f"Failed to index task_doc for {doc.session_id}: {e}")
        
        # ── BATCH EMBED + STORE: summary_doc ──
        if summary_doc:
            try:
                embedding = self.embedder.embed(summary_doc, "summary_doc")
                self.summary_collection.add(
                    ids=[f"{doc.session_id}_summary_0"],
                    embeddings=[embedding],
                    documents=[summary_doc],
                    metadatas=[{
                        "session_id": doc.session_id,
                        "doc_type": "summary_doc",
                        "project_name": doc.project_name,
                        "chunk_index": 0,
                    }]
                )
                counts["summary"] = 1
            except Exception as e:
                logger.warning(f"Failed to index summary_doc for {doc.session_id}: {e}")
        
        # ── BATCH EMBED + STORE: reasoning_docs (the big win) ──
        if reasoning_docs:
            try:
                # Single batch embed call instead of N individual calls
                embeddings = self.embedder.embed_batch(reasoning_docs, "reasoning_doc")
                
                ids = [f"{doc.session_id}_reasoning_{i}" for i in range(len(reasoning_docs))]
                metadatas = [{
                    "session_id": doc.session_id,
                    "doc_type": "reasoning_doc",
                    "project_name": doc.project_name,
                    "chunk_index": i,
                } for i in range(len(reasoning_docs))]
                
                self.reasoning_collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=reasoning_docs,
                    metadatas=metadatas,
                )
                counts["reasoning"] = len(reasoning_docs)
            except Exception as e:
                logger.warning(f"Failed to batch index reasoning_docs for {doc.session_id}: {e}")
        
        logger.debug(
            f"Indexed session {doc.session_id}: "
            f"task={counts['task']}, summary={counts['summary']}, "
            f"reasoning={counts['reasoning']}"
        )
        
        return counts

    
    def search(self, query_embedding: List[float], doc_type: str = "task",
               session_ids: List[str] = None, n_results: int = 20) -> List[VectorResult]:
        """Search a specific collection for similar documents.
        
        Args:
            query_embedding: Query vector from embedder.embed_query()
            doc_type: "task", "summary", or "reasoning"
            session_ids: Optional list of session IDs to constrain search to
            n_results: Maximum number of results
        
        Returns:
            List of VectorResult objects sorted by score (highest first)
        """
        collection = self._get_collection(doc_type)
        if collection is None:
            return []
        
        # Build where filter for session_id constraint
        where = None
        if session_ids:
            if len(session_ids) == 1:
                where = {"session_id": session_ids[0]}
            else:
                where = {"session_id": {"$in": session_ids}}
        
        try:
            # Limit n_results to collection size to avoid ChromaDB errors
            collection_count = collection.count()
            effective_n = min(n_results, collection_count) if collection_count > 0 else 0
            
            if effective_n == 0:
                return []
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=effective_n,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            return self._format_results(results, doc_type)
        
        except Exception as e:
            logger.error(f"Vector search failed on {doc_type}: {e}")
            return []
    
    def search_all_collections(self, query_embedding: List[float],
                                session_ids: List[str] = None,
                                n_results: int = 20,
                                weights: Dict[str, float] = None) -> List[VectorResult]:
        """Search all three collections and combine results.
        
        Args:
            query_embedding: Query vector
            session_ids: Optional session ID filter
            n_results: Results per collection
            weights: Optional weight overrides {"task": W, "summary": W, "reasoning": W}
        
        Returns:
            Combined list of VectorResult objects
        """
        default_weights = {"task": 1.0, "summary": 1.0, "reasoning": 1.0}
        weights = weights or default_weights
        
        all_results = []
        
        for doc_type in ["task", "summary", "reasoning"]:
            if weights.get(doc_type, 0) > 0:
                results = self.search(query_embedding, doc_type, session_ids, n_results)
                # Apply weight to scores
                w = weights[doc_type]
                for r in results:
                    r.score *= w
                all_results.extend(results)
        
        # Sort by weighted score (highest first)
        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results
    
    def delete_session(self, session_id: str) -> None:
        """Remove a session from all collections.
        
        Args:
            session_id: Session ID to delete
        """
        for collection in [self.task_collection, self.summary_collection, 
                          self.reasoning_collection]:
            try:
                # Get IDs for this session
                results = collection.get(
                    where={"session_id": session_id},
                    include=[]
                )
                if results["ids"]:
                    collection.delete(ids=results["ids"])
            except Exception as e:
                logger.debug(f"Delete from collection failed (may be empty): {e}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get document counts per collection.
        
        Returns:
            Dict: {"task": N, "summary": N, "reasoning": N, "total": N}
        """
        task_count = self.task_collection.count()
        summary_count = self.summary_collection.count()
        reasoning_count = self.reasoning_collection.count()
        return {
            "task": task_count,
            "summary": summary_count,
            "reasoning": reasoning_count,
            "total": task_count + summary_count + reasoning_count,
        }
    
    def reset(self) -> None:
        """Delete all collections and recreate them. Use with caution."""
        self.client.delete_collection(TASK_COLLECTION)
        self.client.delete_collection(SUMMARY_COLLECTION)
        self.client.delete_collection(REASONING_COLLECTION)
        
        self.task_collection = self.client.get_or_create_collection(
            name=TASK_COLLECTION, metadata={"hnsw:space": "cosine"})
        self.summary_collection = self.client.get_or_create_collection(
            name=SUMMARY_COLLECTION, metadata={"hnsw:space": "cosine"})
        self.reasoning_collection = self.client.get_or_create_collection(
            name=REASONING_COLLECTION, metadata={"hnsw:space": "cosine"})
        
        logger.warning("VectorIndex reset — all collections cleared")
    
    def _get_collection(self, doc_type: str):
        """Get the ChromaDB collection for a document type."""
        mapping = {
            "task": self.task_collection,
            "task_doc": self.task_collection,
            "summary": self.summary_collection,
            "summary_doc": self.summary_collection,
            "reasoning": self.reasoning_collection,
            "reasoning_doc": self.reasoning_collection,
        }
        return mapping.get(doc_type)
    
    def _format_results(self, results: Dict[str, Any],
                         doc_type: str) -> List[VectorResult]:
        """Format ChromaDB query results into VectorResult objects.
        
        ChromaDB returns distances (lower = more similar for cosine).
        We convert to similarity scores (higher = more similar).
        """
        vector_results = []
        
        if not results or not results.get("ids") or not results["ids"][0]:
            return []
        
        ids = results["ids"][0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        for i, doc_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            doc = documents[i] if i < len(documents) else ""
            dist = distances[i] if i < len(distances) else 1.0
            
            # Convert cosine distance to similarity: similarity = 1 - distance
            similarity = max(0.0, 1.0 - dist)
            
            vector_results.append(VectorResult(
                session_id=meta.get("session_id", ""),
                doc_type=doc_type,
                content=doc,
                score=similarity,
                chunk_index=meta.get("chunk_index", 0),
            ))
        
        return vector_results
