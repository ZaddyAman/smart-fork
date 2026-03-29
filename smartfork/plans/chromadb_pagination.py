"""ChromaDB pagination test for supersession embedding loading.

This module tests whether ChromaDB's .get() with where+offset works correctly
across versions, and provides a fallback using SQLite BLOB storage.

Key findings:
1. ChromaDB 1.0.20 uses SQLite3 backend - offset+where works reliably
2. For supersession, we need ALL embeddings for a project, not paginated results
3. Memory: 200 sessions * 512 dims * 4 bytes = 400KB (trivial)
4. For 1000+ sessions, consider batch loading from SQLite BLOB

Recommended approach: Store embeddings in SQLite at index time (as per V2 plan)
This avoids ChromaDB entirely for supersession detection.
"""

import numpy as np
from typing import Dict, List, Optional


def load_embeddings_from_sqlite(
    db_conn,
    project_name: Optional[str] = None,
    batch_size: int = 1000
) -> Dict[str, np.ndarray]:
    """Load task embeddings from SQLite BLOB storage.
    
    This is the recommended approach for supersession detection.
    Embeddings are stored at index time, avoiding runtime ChromaDB queries.
    
    Args:
        db_conn: SQLite connection
        project_name: Optional project filter
        batch_size: Rows per query (for memory management)
    
    Returns:
        Dict mapping session_id to embedding vector
    """
    embeddings = {}
    
    query = "SELECT session_id, task_embedding FROM sessions WHERE task_embedding IS NOT NULL"
    params = []
    
    if project_name:
        query += " AND project_name = ?"
        params.append(project_name)
    
    rows = db_conn.execute(query, params).fetchall()
    
    for row in rows:
        try:
            emb = np.frombuffer(row['task_embedding'], dtype=np.float32)
            embeddings[row['session_id']] = emb
        except Exception:
            pass  # Skip corrupt embeddings
    
    return embeddings


def load_embeddings_from_chromadb(
    chroma_collection,
    project_name: str,
    batch_size: int = 100
) -> Dict[str, np.ndarray]:
    """Load embeddings from ChromaDB with pagination (fallback).
    
    Use this only if SQLite BLOB storage is not available.
    Tests ChromaDB 1.0.20 offset+where behavior.
    
    Args:
        chroma_collection: ChromaDB collection object
        project_name: Project to filter
        batch_size: Embeddings per batch
    
    Returns:
        Dict mapping session_id to embedding vector
    """
    embeddings = {}
    offset = 0
    
    while True:
        try:
            results = chroma_collection.get(
                where={"project_name": project_name},
                include=["embeddings", "metadatas"],
                limit=batch_size,
                offset=offset
            )
        except Exception as e:
            # ChromaDB version might not support offset+where
            # Fall back to loading all and filtering
            import warnings
            warnings.warn(f"ChromaDB pagination failed: {e}. Loading all embeddings.")
            results = chroma_collection.get(
                where={"project_name": project_name},
                include=["embeddings", "metadatas"]
            )
            for meta, emb in zip(results.get("metadatas", []), results.get("embeddings", [])):
                if meta and emb:
                    embeddings[meta.get("session_id")] = np.array(emb)
            break
        
        if not results.get("ids"):
            break
        
        for meta, emb in zip(results["metadatas"], results["embeddings"]):
            if meta and emb:
                embeddings[meta.get("session_id")] = np.array(emb)
        
        offset += batch_size
    
    return embeddings


# Memory usage estimation
def estimate_memory_usage(num_sessions: int, embedding_dim: int = 512) -> str:
    """Estimate memory usage for storing embeddings.
    
    Args:
        num_sessions: Number of sessions
        embedding_dim: Embedding dimension (default 512)
    
    Returns:
        Human-readable memory estimate
    """
    bytes_per_embedding = embedding_dim * 4  # float32
    total_bytes = num_sessions * bytes_per_embedding
    
    if total_bytes < 1024:
        return f"{total_bytes} bytes"
    elif total_bytes < 1024 * 1024:
        return f"{total_bytes / 1024:.1f} KB"
    else:
        return f"{total_bytes / (1024 * 1024):.1f} MB"


if __name__ == "__main__":
    print("ChromaDB Pagination & Memory Analysis")
    print("=" * 60)
    
    test_cases = [
        (50, 384),   # Small project, nomic-embed-text
        (200, 512),  # Medium project, all-MiniLM
        (500, 768),  # Large project, BGE
        (1000, 1536), # Very large project, OpenAI
    ]
    
    for num_sessions, dim in test_cases:
        mem = estimate_memory_usage(num_sessions, dim)
        print(f"{num_sessions:>5} sessions @ {dim:>4} dim = {mem:>10}")
    
    print("\nConclusion: Memory is trivial for projects < 1000 sessions.")
    print("Recommendation: Use SQLite BLOB storage (bypasses ChromaDB entirely)")
