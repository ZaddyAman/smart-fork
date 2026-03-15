"""Contextual chunking for reasoning blocks (v2).

Prepends context headers to documents before embedding. This ensures every
chunk is self-contained and carries its session identity even when retrieved
in isolation (Anthropic's Contextual Retrieval technique).

Also handles splitting long reasoning blocks at sentence boundaries
with 1-2 sentence overlap.
"""

import re
from typing import List
from datetime import datetime

from ..database.models import SessionDocument


class ContextualChunker:
    """Builds contextually-prefixed documents for embedding.
    
    Every chunk gets a context header like:
        [Project: BharatLawAI | Session: 2025-09-06 | Task: Fix CSS layout | Files: Layout.tsx, ChatPanel.tsx]
    
    This ensures that when a reasoning chunk is retrieved in isolation,
    the LLM/user can immediately understand which project and session it belongs to.
    """
    
    MAX_REASONING_TOKENS = 500  # Split reasoning blocks exceeding this
    OVERLAP_SENTENCES = 2       # Sentence overlap between splits
    
    def build_context_header(self, doc: SessionDocument) -> str:
        """Build a context header string for a session document.
        
        Args:
            doc: SessionDocument with metadata fields
        
        Returns:
            Context header string like '[Project: X | Session: YYYY-MM-DD | Task: ... | Files: ...]'
        """
        # Format session date
        if doc.session_start:
            date_str = datetime.fromtimestamp(doc.session_start / 1000).strftime('%Y-%m-%d')
        else:
            date_str = "unknown date"
        
        # Truncate task to keep header compact
        task_short = doc.task_raw[:80].strip() if doc.task_raw else "no task description"
        if len(doc.task_raw) > 80:
            task_short += "..."
        
        # Top 3 edited files (basenames only for compactness)
        top_files = doc.files_edited[:3]
        files_str = ", ".join(
            f.split("/")[-1] for f in top_files
        ) if top_files else "no files"
        
        return (
            f"[Project: {doc.project_name} | "
            f"Session: {date_str} | "
            f"Task: {task_short} | "
            f"Files: {files_str}]"
        )
    
    def build_task_doc(self, doc: SessionDocument) -> str:
        """Build the task document for embedding.
        
        Combines context header + task_raw. This is the primary retrieval
        key for implementation_lookup and vague_memory intents.
        
        Args:
            doc: SessionDocument
        
        Returns:
            Context-prefixed task document string
        """
        header = self.build_context_header(doc)
        task_text = doc.task_raw.strip() if doc.task_raw else "no task description"
        return f"{header}\n\n{task_text}"
    
    def build_summary_doc(self, doc: SessionDocument) -> str:
        """Build the summary document for embedding.
        
        Combines context header + LLM-generated summary (from Phase 3).
        If no summary exists yet, returns empty string.
        
        Args:
            doc: SessionDocument
        
        Returns:
            Context-prefixed summary document, or empty string
        """
        if not doc.summary_doc:
            return ""
        
        header = self.build_context_header(doc)
        return f"{header}\n\n{doc.summary_doc}"
    
    def build_reasoning_docs(self, doc: SessionDocument) -> List[str]:
        """Build contextually-prefixed reasoning documents for embedding.
        
        For each reasoning block:
        1. Prepend context header
        2. If block > MAX_REASONING_TOKENS: split at sentence boundaries
           with OVERLAP_SENTENCES sentence overlap
        3. Each split chunk gets its own context header
        
        Args:
            doc: SessionDocument with reasoning_docs populated
        
        Returns:
            List of context-prefixed reasoning chunks ready for embedding
        """
        if not doc.reasoning_docs:
            return []
        
        header = self.build_context_header(doc)
        result = []
        
        for block in doc.reasoning_docs:
            block = block.strip()
            if not block:
                continue
            
            estimated_tokens = self._estimate_tokens(block)
            
            if estimated_tokens <= self.MAX_REASONING_TOKENS:
                # Short enough — embed as-is with header
                result.append(f"{header}\n\n{block}")
            else:
                # Too long — split at sentence boundaries
                chunks = self._split_at_sentences(block)
                for chunk in chunks:
                    result.append(f"{header}\n\n{chunk}")
        
        return result
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate: word_count * 1.3.
        
        This is approximate but fast. Exact tokenization would require
        the actual model's tokenizer which adds latency.
        """
        word_count = len(text.split())
        return int(word_count * 1.3)
    
    def _split_at_sentences(self, text: str) -> List[str]:
        """Split text at sentence boundaries with overlap.
        
        Strategy:
        1. Split text into sentences
        2. Accumulate sentences until token limit
        3. Start next chunk with OVERLAP_SENTENCES overlap
        
        Args:
            text: Long text to split
        
        Returns:
            List of text chunks
        """
        # Split into sentences (handles ., !, ?, and newline boundaries)
        sentences = re.split(r'(?<=[.!?])\s+|\n\n+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return [text]
        
        chunks = []
        i = 0
        
        while i < len(sentences):
            chunk_sentences = []
            chunk_tokens = 0
            
            while i < len(sentences):
                sent_tokens = self._estimate_tokens(sentences[i])
                if chunk_tokens + sent_tokens > self.MAX_REASONING_TOKENS and chunk_sentences:
                    break
                chunk_sentences.append(sentences[i])
                chunk_tokens += sent_tokens
                i += 1
            
            if chunk_sentences:
                chunks.append(" ".join(chunk_sentences))
            
            # Overlap: go back OVERLAP_SENTENCES for next chunk
            if i < len(sentences):
                i = max(i - self.OVERLAP_SENTENCES, len(chunks))
        
        return chunks if chunks else [text]
