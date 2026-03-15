"""BM25 keyword search index built over structured fields (v2).

Unlike v1 which built BM25 over full concatenated conversation text (huge, noisy),
v2 only indexes task_raw + files_edited + domains — compact, high-signal fields
where exact keyword matches matter most.

BM25 excels at proper nouns, filenames, version numbers, and error codes
that vector embeddings consistently fail at.
"""

import re
from typing import List, Optional, Tuple, Dict
from loguru import logger

from rank_bm25 import BM25Okapi

from ..database.metadata_store import MetadataStore


class BM25Index:
    """BM25 keyword search over structured session fields.
    
    Corpus per session = task_raw + ' '.join(files_edited) + ' '.join(domains)
    
    This gives BM25 access to the most important exact-match signals:
    - Task descriptions (user's original intent)
    - Filenames (exact file lookup queries)
    - Domains (topic keywords)
    
    Usage:
        index = BM25Index()
        index.build_from_metadata(store)
        results = index.search(["auth", "jwt"], candidate_ids=filtered_ids)
    """
    
    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.session_ids: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self._session_id_to_idx: Dict[str, int] = {}
    
    def build_from_metadata(self, store: MetadataStore) -> int:
        """Build BM25 corpus from all sessions in the metadata store.
        
        For each session, creates a single document:
            corpus_text = task_raw + ' ' + ' '.join(files_edited) + ' ' + ' '.join(domains)
        
        Args:
            store: MetadataStore instance with indexed sessions
        
        Returns:
            Number of documents in the corpus
        """
        sessions = store.get_all_sessions()
        
        self.session_ids = []
        self.tokenized_corpus = []
        self._session_id_to_idx = {}
        
        for doc in sessions:
            # Build corpus text from high-signal fields only
            parts = []
            
            if doc.task_raw:
                parts.append(doc.task_raw)
            
            if doc.files_edited:
                # Include both full paths and basenames for matching
                for f in doc.files_edited:
                    parts.append(f)
                    basename = f.split("/")[-1]
                    parts.append(basename)
            
            if doc.domains:
                parts.extend(doc.domains)
            
            if doc.project_name and doc.project_name != "unknown_project":
                parts.append(doc.project_name)
            
            # Include LLM-generated summary (high-quality search signal)
            if doc.summary_doc:
                parts.append(doc.summary_doc)
            
            corpus_text = " ".join(parts)
            tokens = self._tokenize(corpus_text)
            
            idx = len(self.session_ids)
            self.session_ids.append(doc.session_id)
            self.tokenized_corpus.append(tokens)
            self._session_id_to_idx[doc.session_id] = idx
        
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        logger.debug(f"BM25 index built with {len(self.session_ids)} documents")
        return len(self.session_ids)
    
    def search(self, query_terms: List[str], candidate_ids: List[str] = None,
               n_results: int = 20) -> List[Tuple[str, float]]:
        """Search BM25 index for matching sessions.
        
        Args:
            query_terms: List of search terms (already tokenized)
            candidate_ids: Optional list of session IDs to constrain search
            n_results: Maximum results to return
        
        Returns:
            List of (session_id, bm25_score) tuples sorted by score descending
        """
        if not self.bm25 or not query_terms:
            return []
        
        # Tokenize query terms
        query_tokens = []
        for term in query_terms:
            query_tokens.extend(self._tokenize(term))
        
        if not query_tokens:
            return []
        
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)
        
        # Build results
        results = []
        for idx, score in enumerate(scores):
            if score <= 0:
                continue
            
            session_id = self.session_ids[idx]
            
            # Apply candidate filter if provided
            if candidate_ids and session_id not in candidate_ids:
                continue
            
            results.append((session_id, float(score)))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n_results]
    
    def search_text(self, query: str, candidate_ids: List[str] = None,
                    n_results: int = 20) -> List[Tuple[str, float]]:
        """Search using a raw text query (will be tokenized).
        
        Convenience wrapper that accepts a text string instead of pre-tokenized terms.
        
        Args:
            query: Raw search query text
            candidate_ids: Optional session ID filter
            n_results: Maximum results
        
        Returns:
            List of (session_id, score) tuples
        """
        tokens = self._tokenize(query)
        return self.search(tokens, candidate_ids, n_results)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25.
        
        Strategy:
        - Lowercase
        - Split on whitespace and punctuation (except dots in filenames)
        - Remove very short tokens (< 2 chars)
        - Preserve filenames and technical terms
        """
        text = text.lower()
        
        # Split on whitespace and most punctuation, preserve dots in filenames
        tokens = re.findall(r'[a-z0-9_]+(?:\.[a-z0-9_]+)*', text)
        
        # Filter very short tokens (but keep filenames like "x.py")
        tokens = [t for t in tokens if len(t) >= 2]
        
        return tokens
