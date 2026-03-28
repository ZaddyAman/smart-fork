"""ChromaDB-based vector index (v2).

Replaced Qdrant with ChromaDB for reliable Windows persistence.
Qdrant's embedded mode on Windows has a fatal bug: portalocker
imports msvcrt for file locking, but Python tears down msvcrt
during interpreter shutdown before the WAL can flush — data is
silently lost every time.

ChromaDB 1.0.20+ uses SQLite3 as its storage backend, which
persists immediately and has zero Windows file-locking issues.

Collections:
- v2_task_docs: task descriptions with contextual prefix
- v2_summary_docs: LLM-generated session summaries
- v2_reasoning_docs: AI reasoning/decision blocks
"""

import uuid
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from loguru import logger

import chromadb
from chromadb.config import Settings

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
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client (persistent, local)
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.embedder = embedder
        self.chunker = ContextualChunker()
        
        # Get embedding dimension from embedder
        self.dimensions = getattr(embedder, 'dimensions', 512)
        
        # Create or get collections
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
        
        self.noise_embeddings = []
        logger.debug(f"VectorIndex initialized at {db_path} (ChromaDB backend)")
    
    def _get_collection(self, doc_type: str):
        """Map doc_type to ChromaDB collection object."""
        mapping = {
            "task": self.task_collection,
            "task_doc": self.task_collection,
            "summary": self.summary_collection,
            "summary_doc": self.summary_collection,
            "reasoning": self.reasoning_collection,
            "reasoning_doc": self.reasoning_collection,
        }
        return mapping.get(doc_type, self.task_collection)

    def _get_noise_embeddings(self) -> List[List[float]]:
        """Lazy load noise prototype embeddings for cosine filtering."""
        if not self.noise_embeddings:
            prototypes = [
                "Here is a summary of the operations performed and files updated.",
                "Let me check the workspace to see what files are present.",
                "I will now run the command to verify the output.",
                "The following files have been modified successfully.",
                "I will write the updated content to the file now."
            ]
            try:
                self.noise_embeddings = self.embedder.embed_batch(prototypes, "reasoning_doc")
            except Exception as e:
                logger.warning(f"Failed to generate noise embeddings: {e}")
        return self.noise_embeddings
    
    def index_session(self, doc: SessionDocument, store=None) -> Dict[str, int]:
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
        reasoning_chunks = self.chunker.build_reasoning_docs(doc)
        
        # ── INDEX: task_doc ──
        if task_doc:
            try:
                embedding = self.embedder.embed(task_doc, "task_doc")
                doc_id = f"{doc.session_id}_task_0"
                self.task_collection.add(
                    ids=[doc_id],
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
        
        # ── INDEX: summary_doc ──
        if summary_doc:
            try:
                embedding = self.embedder.embed(summary_doc, "summary_doc")
                doc_id = f"{doc.session_id}_summary_0"
                self.summary_collection.add(
                    ids=[doc_id],
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
        
        # ── INDEX: reasoning_docs ──
        if reasoning_chunks:
            try:
                texts_to_embed = [chunk["text"] for chunk in reasoning_chunks]
                
                # Single batch embed call
                embeddings = self.embedder.embed_batch(texts_to_embed, "reasoning_doc")
                
                # Semantic Denoising - Cosine Pre-Filter
                noise_embs = self._get_noise_embeddings()
                
                valid_ids = []
                valid_embeddings = []
                valid_documents = []
                valid_metadatas = []
                seen_parents = set()
                
                for i, chunk in enumerate(reasoning_chunks):
                    emb = np.array(embeddings[i])
                    is_noise = False
                    
                    if noise_embs:
                        for n_emb in noise_embs:
                            n_emb_arr = np.array(n_emb)
                            norm_emb = np.linalg.norm(emb)
                            norm_n_emb = np.linalg.norm(n_emb_arr)
                            if norm_emb > 0 and norm_n_emb > 0:
                                similarity = np.dot(emb, n_emb_arr) / (norm_emb * norm_n_emb)
                                if similarity > 0.85:
                                    is_noise = True
                                    break
                    
                    if not is_noise:
                        doc_id = f"{doc.session_id}_reasoning_{chunk['chunk_index']}"
                        valid_ids.append(doc_id)
                        valid_embeddings.append(embeddings[i])
                        valid_documents.append(texts_to_embed[i])
                        valid_metadatas.append({
                            "session_id": doc.session_id,
                            "doc_type": "reasoning_doc",
                            "project_name": doc.project_name,
                            "chunk_index": chunk["chunk_index"],
                            "parent_id": chunk["parent_id"],
                        })
                        
                        if store:
                            pid = chunk["parent_id"]
                            if pid not in seen_parents:
                                store.insert_parent_chunk(
                                    parent_id=pid,
                                    session_id=doc.session_id,
                                    chunk_index=chunk["chunk_index"],
                                    full_raw_text=chunk["full_raw_text"]
                                )
                                seen_parents.add(pid)
                
                if valid_ids:
                    # ChromaDB handles batching internally, but we chunk
                    # to stay under any per-call limits
                    batch_size = 500
                    for b in range(0, len(valid_ids), batch_size):
                        end = b + batch_size
                        self.reasoning_collection.add(
                            ids=valid_ids[b:end],
                            embeddings=valid_embeddings[b:end],
                            documents=valid_documents[b:end],
                            metadatas=valid_metadatas[b:end],
                        )
                    counts["reasoning"] = len(valid_ids)
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
        
        # Build where filter for session_id constraint
        where_filter = None
        if session_ids:
            if len(session_ids) == 1:
                where_filter = {"session_id": session_ids[0]}
            else:
                where_filter = {"session_id": {"$in": session_ids}}
        
        try:
            # Clamp n_results to collection count to avoid ChromaDB errors
            count = collection.count()
            if count == 0:
                return []
            effective_n = min(n_results, count)
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=effective_n,
                where=where_filter,
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
        for collection in [self.task_collection, self.summary_collection, self.reasoning_collection]:
            try:
                # Check if there are any docs with this session_id before deleting
                existing = collection.get(
                    where={"session_id": session_id},
                    limit=1
                )
                if existing and existing["ids"]:
                    collection.delete(
                        where={"session_id": session_id}
                    )
            except Exception as e:
                logger.debug(f"Delete from {collection.name} failed: {e}")
    
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
        for name in [TASK_COLLECTION, SUMMARY_COLLECTION, REASONING_COLLECTION]:
            try:
                self.client.delete_collection(name)
            except Exception:
                pass
        
        # Recreate collections
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
        logger.warning("VectorIndex reset — all collections cleared")
    
    def _format_results(self, results: Dict[str, Any], doc_type: str) -> List[VectorResult]:
        """Format ChromaDB query results into VectorResult objects.
        
        ChromaDB returns cosine *distance* (0 = identical, 2 = opposite).
        We convert to similarity: score = 1.0 - (distance / 2.0).
        """
        vector_results = []
        
        if not results or not results.get("ids") or not results["ids"][0]:
            return vector_results
        
        ids = results["ids"][0]
        documents = results.get("documents", [[]])[0] or []
        distances = results.get("distances", [[]])[0] or []
        metadatas = results.get("metadatas", [[]])[0] or []
        
        for i in range(len(ids)):
            # Convert cosine distance to similarity score
            score = 1.0 - (distances[i] / 2.0) if i < len(distances) else 0.0
            
            meta = metadatas[i] if i < len(metadatas) else {}
            content = documents[i] if i < len(documents) else ""
            
            vector_results.append(VectorResult(
                session_id=meta.get("session_id", ""),
                doc_type=doc_type,
                content=content,
                score=max(0.0, score),
                chunk_index=meta.get("chunk_index", 0),
                parent_id=meta.get("parent_id")
            ))
        
        return vector_results
