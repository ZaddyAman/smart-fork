"""Advanced chunking strategies for conversation data."""

import re
from typing import List, Optional, Tuple, Dict, Set
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from collections import Counter

from ..database.models import ConversationMessage, TaskSession
from ..database.chunk_models import (
    EnhancedChunk, EnhancedChunkMetadata, MessageRange, ChunkContentType
)


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategy."""
    max_tokens_per_chunk: int = 512
    overlap_messages: int = 1  # Keep last N messages for context
    preserve_code_blocks: bool = True
    preserve_message_boundaries: bool = True
    extract_keywords: bool = True
    extract_files: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_tokens_per_chunk < 100:
            raise ValueError("max_tokens_per_chunk must be at least 100")
        if self.overlap_messages < 0:
            raise ValueError("overlap_messages must be non-negative")


class MessageBoundaryChunker:
    """
    Chunks conversations while preserving message boundaries.
    
    Unlike the naive word-based approach, this chunker:
    1. Accumulates complete messages until token limit
    2. Never splits a message across chunks
    3. Tracks which files mentioned in each chunk (per-chunk tracking!)
    4. Classifies content type (code vs text vs mixed)
    5. Extracts keywords and entities for better search
    
    Example:
        >>> chunker = MessageBoundaryChunker()
        >>> chunks = chunker.chunk_session(session)
        >>> print(f"Created {len(chunks)} chunks")
        >>> print(f"Files in chunk 0: {chunks[0].metadata.files_mentioned}")
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        
        # Compile regex patterns for efficiency
        self.code_pattern = re.compile(
            r'```(\w+)?\n(.*?)\n```', 
            re.DOTALL
        )
        self.file_pattern = re.compile(
            r'[\w\-./]+\.(py|js|ts|jsx|tsx|java|go|rs|cpp|c|h|yaml|yml|json|md|txt|sql|html|css|scss|vue|php|rb|swift|kt|scala|r|m|mm)',
            re.IGNORECASE
        )
        self.import_pattern = re.compile(
            r'(?:from|import)\s+([\w.]+)',
            re.IGNORECASE
        )
        
        # Common stop words for keyword extraction
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 'please',
            'help', 'need', 'want', 'like', 'know', 'think', 'see', 'get', 'use',
            'make', 'go', 'come', 'take', 'look', 'find', 'give', 'tell', 'ask'
        }
    
    def chunk_session(self, session: TaskSession) -> List[EnhancedChunk]:
        """
        Create message-aware chunks from a session.
        
        Args:
            session: TaskSession with conversation messages
            
        Returns:
            List of EnhancedChunks with rich metadata including per-chunk file tracking
            
        Raises:
            ValueError: If session has no conversation
        """
        chunks = []
        messages = session.conversation
        
        if not messages:
            return chunks
        
        i = 0
        while i < len(messages):
            chunk_messages, next_index = self._create_chunk(
                messages, i, session
            )
            
            if chunk_messages:
                chunk = self._build_chunk(
                    chunk_messages, 
                    len(chunks), 
                    session.task_id
                )
                chunks.append(chunk)
            
            i = next_index
        
        return chunks
    
    def _create_chunk(
        self, 
        messages: List[ConversationMessage], 
        start_idx: int,
        session: TaskSession
    ) -> Tuple[List[ConversationMessage], int]:
        """
        Accumulate messages into a single chunk without exceeding token limit.
        
        Strategy:
        1. Start at start_idx
        2. Add complete messages until token limit reached
        3. Never split a message - if one message exceeds limit, include it whole
        4. Return the chunk and next start index
        
        Args:
            messages: List of all messages
            start_idx: Starting index for this chunk
            session: TaskSession (for context)
            
        Returns:
            Tuple of (messages in chunk, next start index)
        """
        chunk_messages = []
        total_tokens = 0
        i = start_idx
        
        while i < len(messages):
            msg = messages[i]
            
            # Skip empty messages
            if not msg.content:
                i += 1
                continue
            
            msg_tokens = self._estimate_tokens(msg.content)
            
            # Check if adding this message would exceed limit
            if chunk_messages and (total_tokens + msg_tokens > self.config.max_tokens_per_chunk):
                # Would exceed limit, finalize chunk
                break
            
            chunk_messages.append(msg)
            total_tokens += msg_tokens
            i += 1
            
            # If this single message is huge (more than 1.5x limit), we include it anyway
            # It's better to have one oversized chunk than lose important data
            if msg_tokens > self.config.max_tokens_per_chunk * 1.5:
                i += 1
                break
        
        return chunk_messages, i
    
    def _build_chunk(
        self,
        messages: List[ConversationMessage],
        chunk_index: int,
        session_id: str
    ) -> EnhancedChunk:
        """
        Build an EnhancedChunk from a list of messages with rich metadata.
        
        Args:
            messages: List of messages in this chunk
            chunk_index: Index of this chunk in the session
            session_id: ID of the parent session
            
        Returns:
            EnhancedChunk with complete metadata
        """
        
        # Build content with role prefixes for context
        content_parts = []
        for msg in messages:
            role_prefix = f"[{msg.role}]"
            content_parts.append(f"{role_prefix}: {msg.content}")
        content = "\n\n".join(content_parts)
        
        # Analyze content
        files_mentioned = self._extract_files(content) if self.config.extract_files else set()
        content_type = self._classify_content(content)
        code_language = self._detect_code_language(content)
        primary_role = messages[0].role if messages else "unknown"
        
        # Extract summary (first user question or first sentence)
        summary = self._extract_summary(messages)
        
        # Get timestamps from messages
        timestamps = [m.timestamp for m in messages if m.timestamp]
        
        # Extract keywords and entities
        keywords = self._extract_keywords(content) if self.config.extract_keywords else []
        entities = list(files_mentioned)  # Files are key entities
        
        # Calculate metrics
        char_count = len(content)
        token_count = self._estimate_tokens(content)
        
        # Build message range
        start_ts = messages[0].timestamp if messages else None
        end_ts = messages[-1].timestamp if messages else None
        
        # Build metadata
        metadata = EnhancedChunkMetadata(
            session_id=session_id,
            task_id=session_id,
            chunk_index=chunk_index,
            message_range=MessageRange(
                start_index=0,  # Could be enhanced to track actual indices
                end_index=len(messages),
                start_timestamp=start_ts,
                end_timestamp=end_ts
            ),
            primary_role=primary_role,
            content_type=content_type,
            summary=summary,
            primary_topic=keywords[0] if keywords else None,
            code_language=code_language,
            files_mentioned=sorted(list(files_mentioned)),
            files_modified=[],  # Could be populated from task_metadata
            token_count=token_count,
            char_count=char_count,
            last_active=datetime.fromtimestamp(end_ts / 1000).isoformat() if end_ts else None,
            keywords=keywords[:20],  # Limit to top 20
            entities=entities[:20]
        )
        
        return EnhancedChunk(
            id=f"{session_id}_{chunk_index}",
            content=content,
            metadata=metadata
        )
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Uses a hybrid approach:
        - Natural language: ~0.75 tokens per word
        - Code: More token-dense, add penalty for code blocks
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        # Count words
        words = len(text.split())
        base_estimate = int(words * 0.75)
        
        # Add penalty for code blocks (more token-dense)
        code_blocks = len(self.code_pattern.findall(text))
        code_penalty = code_blocks * 50  # ~50 tokens per code block overhead
        
        return base_estimate + code_penalty
    
    def _extract_files(self, text: str) -> Set[str]:
        """
        Extract file paths mentioned in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Set of file paths found
        """
        matches = self.file_pattern.findall(text)
        return set(matches)
    
    def _classify_content(self, text: str) -> str:
        """
        Classify content type based on text analysis.
        
        Types:
        - file_operation: Mentions file creation/modification
        - code_block: Contains substantial code
        - mixed: Mix of code and text
        - user_query: User asking a question
        - assistant_response: Assistant's answer
        
        Args:
            text: Text to classify
            
        Returns:
            Content type string
        """
        text_lower = text.lower()
        
        # Check for file operations
        file_ops = [
            r'created file',
            r'modified file',
            r'deleted file',
            r'updated file',
            r'added file',
            r'removed file'
        ]
        for pattern in file_ops:
            if re.search(pattern, text_lower):
                return ChunkContentType.FILE_OPERATION
        
        # Check for code blocks
        code_matches = self.code_pattern.findall(text)
        has_code = bool(code_matches)
        total_code_length = sum(len(match[1]) for match in code_matches)
        
        # Classify based on code presence
        if has_code:
            if total_code_length > 200 and len(text) < total_code_length * 2:
                # Mostly code
                return ChunkContentType.CODE_BLOCK
            else:
                # Mix of code and text
                return ChunkContentType.MIXED
        
        # Check if user query (starts with [user])
        if text.startswith('[user]') or text.startswith('[user]:'):
            return ChunkContentType.USER_QUERY
        
        # Default to assistant response
        return ChunkContentType.ASSISTANT_RESPONSE
    
    def _detect_code_language(self, text: str) -> Optional[str]:
        """
        Detect primary code language in chunk.
        
        Args:
            text: Text containing code
            
        Returns:
            Language identifier or None
        """
        match = self.code_pattern.search(text)
        if match:
            return match.group(1) or "text"
        return None
    
    def _extract_summary(self, messages: List[ConversationMessage]) -> Optional[str]:
        """
        Extract a summary from messages.
        
        Priority:
        1. First user question (first sentence)
        2. First assistant response (first sentence)
        3. First message (truncated)
        
        Args:
            messages: List of messages
            
        Returns:
            Summary string or None
        """
        # Prefer first user question
        for msg in messages:
            if msg.role == "user" and msg.content:
                # Get first sentence
                first_sentence = msg.content.split('.')[0].strip()
                # Limit length
                if len(first_sentence) > 150:
                    first_sentence = first_sentence[:150] + "..."
                return first_sentence
        
        # Fallback to first message
        if messages and messages[0].content:
            content = messages[0].content[:150]
            if len(messages[0].content) > 150:
                content += "..."
            return content
        
        return None
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract important keywords from text.
        
        Uses frequency-based extraction with stop word filtering.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of keywords sorted by frequency
        """
        # Find all words
        words = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', text)
        
        # Filter: length > 3, not stop word, not purely numeric
        filtered = [
            w for w in words 
            if len(w) > 3 
            and w.lower() not in self.stop_words
            and not w.isdigit()
        ]
        
        # Count frequencies
        word_counts = Counter(w.lower() for w in filtered)
        
        # Return top 10 by frequency
        return [word for word, count in word_counts.most_common(10)]


class CodeAwareChunker(MessageBoundaryChunker):
    """
    Extends MessageBoundaryChunker with special handling for code blocks.
    
    When a chunk contains a substantial code block, this chunker can:
    1. Extract the code block as its own chunk
    2. Keep the surrounding text as a separate chunk
    3. Link them via metadata
    
    This ensures code blocks are:
    - Never split across chunks
    - Easily retrievable by code-specific searches
    - Kept with their surrounding context
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None, 
                 min_code_chunk_size: int = 100):
        super().__init__(config)
        self.min_code_chunk_size = min_code_chunk_size
    
    def chunk_session(self, session: TaskSession) -> List[EnhancedChunk]:
        """
        Chunk with special code block handling.
        
        Post-processes base chunks to extract large code blocks.
        """
        # First, get base chunks from parent
        base_chunks = super().chunk_session(session)
        
        if not self.config.preserve_code_blocks:
            return base_chunks
        
        # Post-process to extract large code blocks
        enhanced_chunks = []
        
        for chunk in base_chunks:
            # Check if chunk contains substantial code
            code_blocks = self.code_pattern.findall(chunk.content)
            
            if (len(code_blocks) == 1 and 
                len(code_blocks[0][1]) > self.min_code_chunk_size):
                
                # Single large code block - extract it
                lang = code_blocks[0][0] or ""
                code_content = code_blocks[0][1]
                code_text = f"```{lang}\n{code_content}\n```"
                
                # Create code-specific chunk
                code_chunk = EnhancedChunk(
                    id=f"{chunk.id}_code",
                    content=code_text,
                    metadata=EnhancedChunkMetadata(
                        session_id=chunk.metadata.session_id,
                        task_id=chunk.metadata.task_id,
                        chunk_index=chunk.metadata.chunk_index,
                        message_range=chunk.metadata.message_range,
                        primary_role="assistant",
                        content_type=ChunkContentType.CODE_BLOCK,
                        summary=f"Code: {lang or 'text'}",
                        code_language=lang,
                        files_mentioned=chunk.metadata.files_mentioned,
                        token_count=self._estimate_tokens(code_text),
                        char_count=len(code_text),
                        last_active=chunk.metadata.last_active,
                        keywords=chunk.metadata.keywords,
                        entities=chunk.metadata.entities
                    )
                )
                enhanced_chunks.append(code_chunk)
                
                # Also keep the original chunk but with code replaced by reference
                text_without_code = self.code_pattern.sub(
                    '[Code block extracted above]', 
                    chunk.content, 
                    count=1
                )
                
                if len(text_without_code.strip()) > 50:
                    chunk.content = text_without_code
                    chunk.metadata.content_type = ChunkContentType.MIXED
                    chunk.id = f"{chunk.id}_text"
                    enhanced_chunks.append(chunk)
            else:
                enhanced_chunks.append(chunk)
        
        # Re-index chunk indices
        for i, chunk in enumerate(enhanced_chunks):
            chunk.metadata.chunk_index = i
            chunk.id = f"{chunk.metadata.session_id}_{i}"
        
        return enhanced_chunks


# Factory function for creating appropriate chunker
def create_chunker(
    strategy: str = "message_boundary",
    **kwargs
) -> MessageBoundaryChunker:
    """
    Factory function to create the appropriate chunker.
    
    Args:
        strategy: Chunking strategy name
        **kwargs: Configuration parameters
        
    Returns:
        Configured chunker instance
        
    Raises:
        ValueError: If strategy is unknown
    """
    config = ChunkingConfig(**kwargs)
    
    if strategy == "message_boundary":
        return MessageBoundaryChunker(config)
    elif strategy == "code_aware":
        return CodeAwareChunker(config)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
