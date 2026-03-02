"""Semantic clustering for session deduplication and organization."""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from ..config import get_config
from ..database.chroma_db import ChromaDatabase


class SemanticClustering:
    """Clusters sessions by semantic similarity using embeddings."""
    
    def __init__(self, min_cluster_size: int = 2, min_samples: int = 1):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        cfg = get_config()
        self.db = ChromaDatabase(cfg.chroma_db_path)
        
    def cluster_sessions(self) -> Dict[int, List[str]]:
        """Cluster sessions based on semantic similarity of content.
        
        Uses HDBSCAN if available, falls back to simple similarity-based clustering.
        """
        try:
            import hdbscan
            return self._cluster_with_hdbscan()
        except ImportError:
            logger.warning("HDBSCAN not available, using fallback clustering")
            return self._cluster_with_fallback()
            
    def _cluster_with_hdbscan(self) -> Dict[int, List[str]]:
        """Cluster using HDBSCAN algorithm."""
        import hdbscan
        
        # Get all embeddings from database
        collection = self.db.collection
        if collection.count() == 0:
            return {}
            
        results = collection.get(include=["embeddings", "metadatas"])
        embeddings = np.array(results["embeddings"])
        session_ids = [meta["session_id"] for meta in results["metadatas"]]
        
        if len(embeddings) < self.min_cluster_size:
            return {-1: list(set(session_ids))}  # All in one cluster
            
        # Perform clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean'
        )
        labels = clusterer.fit_predict(embeddings)
        
        # Group by cluster
        clusters = defaultdict(list)
        for session_id, label in zip(session_ids, labels):
            clusters[int(label)].append(session_id)
            
        return dict(clusters)
        
    def _cluster_with_fallback(self) -> Dict[int, List[str]]:
        """Fallback clustering using cosine similarity threshold."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        collection = self.db.collection
        if collection.count() == 0:
            return {}
            
        results = collection.get(include=["embeddings", "metadatas"])
        embeddings = np.array(results["embeddings"])
        session_ids = [meta["session_id"] for meta in results["metadatas"]]
        
        if len(embeddings) < 2:
            return {-1: list(set(session_ids))}
            
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Simple greedy clustering
        threshold = 0.7
        clustered = set()
        clusters = {}
        cluster_id = 0
        
        for i, session_id in enumerate(session_ids):
            if session_id in clustered:
                continue
                
            # Find all similar sessions
            cluster = [session_id]
            clustered.add(session_id)
            
            for j, other_id in enumerate(session_ids):
                if other_id not in clustered and similarity_matrix[i][j] > threshold:
                    cluster.append(other_id)
                    clustered.add(other_id)
                    
            if len(cluster) >= self.min_cluster_size:
                clusters[cluster_id] = cluster
                cluster_id += 1
            else:
                # Add to noise cluster
                if -1 not in clusters:
                    clusters[-1] = []
                clusters[-1].extend(cluster)
                
        return clusters
        
    def find_duplicates(self, threshold: float = 0.95) -> List[Tuple[str, str, float]]:
        """Find potential duplicate sessions."""
        from sklearn.metrics.pairwise import cosine_similarity
        
        collection = self.db.collection
        if collection.count() < 2:
            return []
            
        # Get session-level embeddings by averaging chunk embeddings
        results = collection.get(include=["embeddings", "metadatas"])
        
        session_embeddings = {}
        for embedding, metadata in zip(results["embeddings"], results["metadatas"]):
            session_id = metadata["session_id"]
            if session_id not in session_embeddings:
                session_embeddings[session_id] = []
            session_embeddings[session_id].append(embedding)
            
        # Average embeddings per session
        session_ids = list(session_embeddings.keys())
        session_vectors = []
        
        for session_id in session_ids:
            vectors = session_embeddings[session_id]
            avg_vector = np.mean(vectors, axis=0)
            session_vectors.append(avg_vector)
            
        session_vectors = np.array(session_vectors)
        
        # Find duplicates
        similarity_matrix = cosine_similarity(session_vectors)
        duplicates = []
        
        for i in range(len(session_ids)):
            for j in range(i + 1, len(session_ids)):
                sim = similarity_matrix[i][j]
                if sim > threshold:
                    duplicates.append((session_ids[i], session_ids[j], float(sim)))
                    
        return sorted(duplicates, key=lambda x: x[2], reverse=True)
        
    def get_cluster_summary(self, cluster_id: int, session_ids: List[str]) -> Dict:
        """Generate summary for a cluster."""
        # Get common technologies
        all_techs = set()
        all_files = set()
        
        for session_id in session_ids[:5]:  # Sample first 5
            try:
                results = self.db.collection.get(
                    where={"session_id": session_id},
                    include=["metadatas"]
                )
                
                for meta in results["metadatas"]:
                    techs = json.loads(meta.get("technologies", "[]"))
                    all_techs.update(techs)
                    
                    files = json.loads(meta.get("files_in_context", "[]"))
                    all_files.update(files)
                    
            except Exception:
                continue
                
        return {
            "cluster_id": cluster_id,
            "session_count": len(session_ids),
            "common_technologies": list(all_techs)[:10],
            "common_files": list(all_files)[:10],
            "sample_sessions": session_ids[:3]
        }


class SessionClusterer:
    """High-level interface for session clustering operations."""
    
    def __init__(self):
        self.clustering = SemanticClustering()
        
    def analyze_clusters(self) -> Dict:
        """Full cluster analysis with summaries."""
        clusters = self.clustering.cluster_sessions()
        duplicates = self.clustering.find_duplicates()
        
        cluster_summaries = []
        for cluster_id, session_ids in clusters.items():
            if cluster_id == -1:
                continue  # Skip noise cluster
            summary = self.clustering.get_cluster_summary(cluster_id, session_ids)
            cluster_summaries.append(summary)
            
        return {
            "total_clusters": len([c for c in clusters if c != -1]),
            "noise_sessions": len(clusters.get(-1, [])),
            "clusters": sorted(cluster_summaries, key=lambda x: x["session_count"], reverse=True),
            "potential_duplicates": len(duplicates),
            "duplicate_pairs": duplicates[:10]  # Top 10
        }
        
    def suggest_merge_candidates(self) -> List[Dict]:
        """Suggest sessions that could be merged."""
        duplicates = self.clustering.find_duplicates(threshold=0.90)
        
        suggestions = []
        for session_a, session_b, similarity in duplicates[:5]:
            suggestions.append({
                "sessions": [session_a, session_b],
                "similarity": similarity,
                "reason": "High semantic similarity",
                "recommended_action": "Review for potential merge"
            })
            
        return suggestions