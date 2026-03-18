"""RAPTOR Clustering Engine for cross-session synthesis (Phase 9).

Shrinks semantic vectors using UMAP (8-10 dimensions) and clusters them
using Gaussian Mixture Models (GMM) to form cross-session "Epics".
"""

from pathlib import Path
from loguru import logger
import numpy as np

from ..database.metadata_store import MetadataStore
from ..database.vector_index import VectorIndex

class ClusterEngine:
    def __init__(self, store: MetadataStore, vector_index: VectorIndex, model_dir: Path):
        self.store = store
        self.vector_index = vector_index
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.gmm_path = self.model_dir / "gmm_model.joblib"
        self.umap_path = self.model_dir / "umap_model.joblib"
        
    def run_clustering(self, force_full: bool = False) -> bool:
        """Run the clustering pipeline."""
        try:
            import joblib
            import umap
            from sklearn.mixture import GaussianMixture
        except Exception as e:
            logger.error(f"Missing clustering dependencies: {e}. Run pip install umap-learn scikit-learn joblib")
            return False
            
        sessions = self.store.get_all_sessions(limit=10000)
        
        # Session Count Guard Phase 9
        if len(sessions) < 15:
            logger.info(f"Session count ({len(sessions)}) below threshold (15). Skipping clustering.")
            return False
            
        # Get embeddings for ALL sessions using the summary collection
        doc_ids = [f"{s.session_id}_summary_0" for s in sessions]
        
        try:
            results = self.vector_index.summary_collection.get(ids=doc_ids, include=["embeddings"])
        except Exception as e:
            logger.warning(f"Failed to fetch embeddings: {e}")
            return False
            
        if not results or not results["embeddings"]:
            logger.warning("No embeddings found for clustering.")
            return False
            
        # Match embeddings to session IDs
        fetched_ids = results["ids"]
        fetched_embs = results["embeddings"]
        
        emb_dict = {fid: emb for fid, emb in zip(fetched_ids, fetched_embs)}
        
        valid_sessions = []
        matrix = []
        for s in sessions:
            eid = f"{s.session_id}_summary_0"
            if eid in emb_dict:
                valid_sessions.append(s)
                matrix.append(emb_dict[eid])
                
        if len(valid_sessions) < 15:
            logger.info("Not enough valid embeddings to cluster.")
            return False
            
        X = np.array(matrix)
        
        if self.gmm_path.exists() and self.umap_path.exists() and not force_full:
            # Incremental update logic: Nearest Centroid mapping via GMM Predict
            logger.info("Running incremental clustering mapping to existing centroids...")
            reducer = joblib.load(self.umap_path)
            gmm = joblib.load(self.gmm_path)
            
            X_reduced = reducer.transform(X)
            cluster_assignments = gmm.predict(X_reduced)
            
            self._update_sqlite(valid_sessions, cluster_assignments)
            return True
            
        # Full re-clustering
        logger.info(f"Running full UMAP+GMM clustering on {len(valid_sessions)} sessions...")
        # Reduce to 8-10 dimensions
        n_components = min(10, len(valid_sessions) - 2)
        n_clusters = max(2, len(valid_sessions) // 5)
        
        reducer = umap.UMAP(n_neighbors=5, min_dist=0.0, n_components=n_components, random_state=42)
        X_reduced = reducer.fit_transform(X)
        
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
        cluster_assignments = gmm.fit_predict(X_reduced)
        
        joblib.dump(reducer, self.umap_path)
        joblib.dump(gmm, self.gmm_path)
        
        self._update_sqlite(valid_sessions, cluster_assignments)
        logger.success("RAPTOR clustering complete.")
        return True
        
    def _update_sqlite(self, sessions, cluster_assignments):
        """Map assignments back to SQLite."""
        for i, doc in enumerate(sessions):
            cid = f"epic_{cluster_assignments[i]}"
            doc.cluster_id = cid
            self.store.conn.execute(
                "UPDATE sessions SET cluster_id = ? WHERE session_id = ?",
                (cid, doc.session_id)
            )
        self.store.conn.commit()
