"""Smart fork generator with query-aware context extraction for SmartFork.

This module provides intelligent context extraction and markdown generation
based on user queries. It uses the ChunkSearchEngine from Phase 2 to find
relevant chunks and formats them into clean, token-budgeted markdown output.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from loguru import logger

from ..database.chroma_db import ChromaDatabase
from ..database.chunk_models import (
    ChunkSearchResult,
    TokenBudget,
    EnhancedChunkMetadata,
    ChunkContentType
)
from ..search.chunk_search import ChunkSearchEngine


@dataclass
class ContextExtractionConfig:
    """Configuration for context extraction operations.
    
    This dataclass defines the parameters that control how context
    is extracted from sessions, including token budgets and formatting
    preferences.
    
    Attributes:
        max_total_tokens: Maximum tokens to include in output (default: 2000)
        max_chunks: Maximum number of chunks to retrieve (default: 5)
        min_chunk_score: Minimum relevance score threshold (default: 0.3)
        include_summary: Whether to include a summary section (default: True)
        include_file_list: Whether to include the file list section (default: True)
        include_code_snippets: Whether to extract code snippets (default: True)
        format: Output format, currently only "markdown" supported (default: "markdown")
    """
    max_total_tokens: int = 2000
    max_chunks: int = 5
    min_chunk_score: float = 0.3
    include_summary: bool = True
    include_file_list: bool = True
    include_code_snippets: bool = True
    format: str = "markdown"
    
    def to_token_budget(self) -> TokenBudget:
        """Convert to TokenBudget for search operations.
        
        Returns:
            TokenBudget instance with equivalent settings
        """
        return TokenBudget(
            max_total_tokens=self.max_total_tokens,
            max_chunks=self.max_chunks,
            min_chunk_score=self.min_chunk_score,
            prioritize_recent=True,
            include_code=self.include_code_snippets,
            summary_ratio=0.3
        )


class SmartContextExtractor:
    """Extracts relevant context from sessions based on queries and token budgets.
    
    This class uses the ChunkSearchEngine to find chunks relevant to a query,
    then formats them into structured context suitable for inclusion in a
    fork.md file. It enforces token budgets and provides intelligent filtering.
    
    Attributes:
        db: ChromaDatabase instance for data access
        search_engine: ChunkSearchEngine for finding relevant chunks
    
    Example:
        >>> db = ChromaDatabase(Path("./chroma_db"))
        >>> extractor = SmartContextExtractor(db)
        >>> 
        >>> # Extract context for a query
        >>> context = extractor.extract_context(
        ...     "JWT authentication",
        ...     session_id="task_abc123",
        ...     config=ContextExtractionConfig(max_chunks=3)
        ... )
        >>> 
        >>> # Generate markdown content
        >>> markdown = extractor.generate_fork_content(
        ...     "JWT authentication",
        ...     session_id="task_abc123"
        ... )
    """
    
    def __init__(self, db: ChromaDatabase):
        """Initialize the context extractor.
        
        Args:
            db: ChromaDatabase instance for data access
        """
        self.db = db
        self.search_engine = ChunkSearchEngine(db, enable_cache=True)
        logger.debug("SmartContextExtractor initialized")
    
    def extract_context(
        self,
        query: str,
        session_id: Optional[str] = None,
        config: Optional[ContextExtractionConfig] = None
    ) -> Dict[str, Any]:
        """Extract relevant context from sessions based on query.
        
        This method searches for chunks relevant to the query, applies
        token budget constraints, and returns structured context data.
        
        Args:
            query: The search query string
            session_id: Optional session ID to scope search to. If None,
                       searches across all sessions.
            config: Optional configuration for extraction. Uses defaults if None.
        
        Returns:
            Dictionary containing:
                - found: bool - Whether any relevant chunks were found
                - query: str - The original query
                - session_id: str - The session ID (or "global" if not specified)
                - chunks: List[ChunkSearchResult] - The relevant chunks found
                - total_tokens: int - Estimated total token count
                - files_mentioned: List[str] - Unique files across all chunks
                - summary: Optional[str] - Generated summary of the context
        
        Example:
            >>> context = extractor.extract_context(
            ...     "authentication middleware",
            ...     session_id="task_123"
            ... )
            >>> if context['found']:
            ...     print(f"Found {len(context['chunks'])} chunks")
            ...     print(f"Files: {context['files_mentioned']}")
        """
        if not config:
            config = ContextExtractionConfig()
        
        # Use "global" as default session identifier when not specified
        effective_session_id = session_id or "global"
        
        logger.debug(f"Extracting context for query: '{query[:50]}...' "
                    f"(session: {effective_session_id})")
        
        try:
            # Create token budget from config
            token_budget = config.to_token_budget()
            
            # Search for relevant chunks
            results = self.search_engine.search(
                query=query,
                session_id=session_id,
                n_results=config.max_chunks,
                token_budget=token_budget
            )
            
            if not results:
                logger.info(f"No relevant chunks found for query: '{query[:50]}...'")
                return {
                    "found": False,
                    "query": query,
                    "session_id": effective_session_id,
                    "chunks": [],
                    "total_tokens": 0,
                    "files_mentioned": [],
                    "summary": None
                }
            
            # Calculate total tokens
            total_tokens = sum(r.get_token_estimate() for r in results)
            
            # Collect unique files mentioned
            files_mentioned = self._collect_files(results)
            
            # Generate summary
            summary = None
            if config.include_summary:
                summary = self._generate_summary(results, query)
            
            logger.info(f"Extracted context: {len(results)} chunks, "
                       f"{total_tokens} tokens, {len(files_mentioned)} files")
            
            return {
                "found": True,
                "query": query,
                "session_id": effective_session_id,
                "chunks": results,
                "total_tokens": total_tokens,
                "files_mentioned": files_mentioned,
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Error extracting context: {e}")
            return {
                "found": False,
                "query": query,
                "session_id": effective_session_id,
                "chunks": [],
                "total_tokens": 0,
                "files_mentioned": [],
                "summary": None,
                "error": str(e)
            }
    
    def generate_fork_content(
        self,
        query: str,
        session_id: Optional[str] = None,
        config: Optional[ContextExtractionConfig] = None
    ) -> str:
        """Generate markdown formatted context for fork.md file.
        
        This method extracts context and formats it into clean markdown
        with sections for overview, summary, files, exchanges, and code snippets.
        
        Args:
            query: The search query string
            session_id: Optional session ID to scope search to
            config: Optional configuration for extraction
        
        Returns:
            Markdown formatted string ready for fork.md file
        
        Example:
            >>> markdown = extractor.generate_fork_content(
            ...     "database connection pooling",
            ...     session_id="task_456"
            ... )
            >>> Path("fork_database.md").write_text(markdown)
        """
        if not config:
            config = ContextExtractionConfig()
        
        # Extract context
        context = self.extract_context(query, session_id, config)
        
        if not context["found"]:
            return self._generate_empty_result(query, context.get("session_id", "unknown"))
        
        # Build markdown sections
        sections = [
            self._generate_header(query, context),
        ]
        
        # Add summary if enabled and available
        if config.include_summary and context.get("summary"):
            sections.append(self._generate_summary_section(context["summary"]))
        
        # Add file list if enabled
        if config.include_file_list and context.get("files_mentioned"):
            sections.append(self._generate_files_section(context["files_mentioned"]))
        
        # Add exchanges section (always included if chunks found)
        sections.append(self._generate_exchanges_section(context["chunks"]))
        
        # Add code snippets if enabled
        if config.include_code_snippets:
            code_chunks = self._extract_code_chunks(context["chunks"])
            if code_chunks:
                sections.append(self._generate_code_section(code_chunks))
        
        # Add footer
        sections.append(self._generate_footer(context))
        
        return "\n\n".join(sections)
    
    def _collect_files(self, chunks: List[ChunkSearchResult]) -> List[str]:
        """Collect unique files mentioned across all chunks.
        
        Args:
            chunks: List of chunk search results
        
        Returns:
            Sorted list of unique file paths
        """
        all_files = set()
        
        for chunk in chunks:
            # Add files mentioned in this chunk
            all_files.update(chunk.metadata.files_mentioned or [])
            # Also add files modified in this chunk
            all_files.update(chunk.metadata.files_modified or [])
        
        return sorted(list(all_files))
    
    def _generate_summary(self, chunks: List[ChunkSearchResult], query: str) -> str:
        """Generate a summary text from the top chunks.
        
        Args:
            chunks: List of chunk search results
            query: The original query
        
        Returns:
            Generated summary string
        """
        if not chunks:
            return "No relevant content found."
        
        # Get the highest scoring chunk's summary if available
        top_chunk = chunks[0]
        if top_chunk.metadata.summary:
            return f"Most relevant: {top_chunk.metadata.summary}"
        
        # Otherwise, generate from primary topic
        if top_chunk.metadata.primary_topic:
            return f"Topic: {top_chunk.metadata.primary_topic}"
        
        # Fallback to content type description
        content_type = top_chunk.metadata.content_type
        role = top_chunk.metadata.primary_role
        
        return f"Relevant {role} content about {query[:50]}..."
    
    def _generate_header(self, query: str, context: Dict[str, Any]) -> str:
        """Generate the header section with overview information.
        
        Args:
            query: The search query
            context: The extraction context dictionary
        
        Returns:
            Markdown header section
        """
        session_id = context.get("session_id", "unknown")
        chunks_count = len(context.get("chunks", []))
        total_tokens = context.get("total_tokens", 0)
        
        return f"""# Context Fork: {query}

## Overview
- **Query**: "{query}"
- **Source Session**: `{session_id}`
- **Chunks Retrieved**: {chunks_count}
- **Estimated Tokens**: {total_tokens}"""
    
    def _generate_summary_section(self, summary: str) -> str:
        """Generate the summary section.
        
        Args:
            summary: The summary text
        
        Returns:
            Markdown summary section
        """
        return f"""## Summary
{summary}"""
    
    def _generate_files_section(self, files: List[str]) -> str:
        """Generate the relevant files section.
        
        Args:
            files: List of file paths
        
        Returns:
            Markdown files section
        """
        if not files:
            return "## Relevant Files\n\nNo files mentioned in this context."
        
        files_md = "\n".join([f"- `{f}`" for f in files])
        
        return f"""## Relevant Files
{files_md}"""
    
    def _generate_exchanges_section(self, chunks: List[ChunkSearchResult]) -> str:
        """Generate the relevant exchanges section with chunk content.
        
        Args:
            chunks: List of chunk search results
        
        Returns:
            Markdown exchanges section
        """
        if not chunks:
            return "## Relevant Exchanges\n\nNo exchanges found."
        
        exchanges_md = []
        
        for i, chunk in enumerate(chunks, 1):
            # Format relevance as percentage
            relevance_pct = int(chunk.score * 100)
            
            # Get role for labeling
            role = chunk.metadata.primary_role.capitalize()
            
            # Format the exchange
            exchange = f"""### Exchange {i} (Relevance: {relevance_pct}%)
**{role}**: {chunk.content}"""
            exchanges_md.append(exchange)
        
        return "## Relevant Exchanges\n\n" + "\n\n".join(exchanges_md)
    
    def _extract_code_chunks(self, chunks: List[ChunkSearchResult]) -> List[ChunkSearchResult]:
        """Extract chunks that contain code blocks.
        
        Args:
            chunks: List of chunk search results
        
        Returns:
            List of chunks containing code
        """
        code_chunks = []
        
        for chunk in chunks:
            # Check if content type indicates code
            if chunk.metadata.content_type == ChunkContentType.CODE_BLOCK:
                code_chunks.append(chunk)
                continue
            
            # Check content for code blocks
            if "```" in chunk.content:
                code_chunks.append(chunk)
        
        return code_chunks
    
    def _extract_code_from_content(self, content: str) -> List[Dict[str, str]]:
        """Extract code blocks from chunk content.
        
        Args:
            content: The chunk content string
        
        Returns:
            List of dictionaries with 'language' and 'code' keys
        """
        snippets = []
        
        # Match code blocks with optional language specifier
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for lang, code in matches:
            if len(code.strip()) > 20:  # Only substantial snippets
                snippets.append({
                    "language": lang or "text",
                    "code": code.strip()
                })
        
        return snippets
    
    def _generate_code_section(self, code_chunks: List[ChunkSearchResult]) -> str:
        """Generate the key code snippets section.
        
        Args:
            code_chunks: List of chunks containing code
        
        Returns:
            Markdown code section
        """
        if not code_chunks:
            return ""
        
        snippets_md = []
        seen_codes = set()  # Avoid duplicates
        
        for chunk in code_chunks:
            snippets = self._extract_code_from_content(chunk.content)
            
            for snippet in snippets:
                code_hash = hash(snippet["code"][:100])  # Hash first 100 chars for dedup
                if code_hash in seen_codes:
                    continue
                seen_codes.add(code_hash)
                
                lang = snippet["language"]
                code = snippet["code"]
                
                # Truncate if too long
                if len(code) > 800:
                    code = code[:800] + "\n# ... (truncated)"
                
                snippets_md.append(f"```{lang}\n{code}\n```")
                
                # Limit number of snippets
                if len(snippets_md) >= 5:
                    break
            
            if len(snippets_md) >= 5:
                break
        
        if not snippets_md:
            return ""
        
        return "## Key Code Snippets\n\n" + "\n\n".join(snippets_md)
    
    def _generate_footer(self, context: Dict[str, Any]) -> str:
        """Generate the footer section.
        
        Args:
            context: The extraction context dictionary
        
        Returns:
            Markdown footer section
        """
        return """---

*This context was extracted using SmartFork's query-aware retrieval.*"""
    
    def _generate_empty_result(self, query: str, session_id: str) -> str:
        """Generate markdown for when no context is found.
        
        Args:
            query: The search query
            session_id: The session ID
        
        Returns:
            Markdown indicating no results
        """
        return f"""# Context Fork: {query}

## Overview
- **Query**: "{query}"
- **Source Session**: `{session_id}`
- **Chunks Retrieved**: 0
- **Estimated Tokens**: 0

## Summary
No relevant context found for this query.

---

*This context was extracted using SmartFork's query-aware retrieval.*"""


class SmartForkMDGenerator:
    """Drop-in replacement for ForkMDGenerator with query-aware context extraction.
    
    This class provides the same interface as the original ForkMDGenerator
    but uses SmartContextExtractor internally to provide query-aware,
    token-budgeted context extraction.
    
    Attributes:
        db: ChromaDatabase instance
        extractor: SmartContextExtractor for context retrieval
    
    Example:
        >>> db = ChromaDatabase(Path("./chroma_db"))
        >>> generator = SmartForkMDGenerator(db)
        >>> 
        >>> # Generate fork.md content
        >>> markdown = generator.generate(
        ...     session_id="task_abc123",
        ...     query="authentication error handling"
        ... )
    """
    
    def __init__(self, db: ChromaDatabase):
        """Initialize the smart fork markdown generator.
        
        Args:
            db: ChromaDatabase instance for data access
        """
        self.db = db
        self.extractor = SmartContextExtractor(db)
        logger.debug("SmartForkMDGenerator initialized")
    
    def generate(
        self,
        session_id: str,
        query: str,
        current_dir: Optional[str] = None,
        max_tokens: int = 2000
    ) -> str:
        """Generate a fork.md file for a session based on a query.
        
        This is the main method that provides a drop-in replacement for
        ForkMDGenerator.generate(). It extracts relevant context based on
        the query and formats it into markdown.
        
        Args:
            session_id: Session ID to generate fork for
            query: Search query to find relevant context
            current_dir: Optional current working directory (for file path hints)
            max_tokens: Maximum tokens to include (default: 2000)
        
        Returns:
            Markdown formatted content for fork.md file
        
        Example:
            >>> markdown = generator.generate(
            ...     session_id="task_123",
            ...     query="database migration errors"
            ... )
            >>> print(markdown)
        """
        # Create config with specified token budget
        config = ContextExtractionConfig(
            max_total_tokens=max_tokens,
            max_chunks=5,
            include_summary=True,
            include_file_list=True,
            include_code_snippets=True
        )
        
        # Generate content using the extractor
        content = self.extractor.generate_fork_content(
            query=query,
            session_id=session_id,
            config=config
        )
        
        # Add directory overlap note if current_dir provided
        if current_dir:
            content = self._add_directory_overlap(content, current_dir)
        
        return content
    
    def _add_directory_overlap(self, content: str, current_dir: str) -> str:
        """Add directory overlap information to the content.
        
        Args:
            content: The generated markdown content
            current_dir: The current working directory
        
        Returns:
            Updated content with overlap information
        """
        try:
            current_path = Path(current_dir)
            dir_name = current_path.name
            
            # Add overlap note before the footer
            overlap_note = f"\n- **Current Directory**: `{dir_name}`"
            
            # Insert before the footer separator
            if "---" in content:
                content = content.replace(
                    "---\n\n*This context",
                    f"{overlap_note}\n\n---\n\n*This context"
                )
            
            return content
            
        except Exception as e:
            logger.warning(f"Could not add directory overlap: {e}")
            return content
    
    def save(
        self,
        session_id: str,
        query: str,
        output_path: Optional[Path] = None,
        current_dir: Optional[str] = None,
        max_tokens: int = 2000
    ) -> Path:
        """Generate and save fork.md file.
        
        Args:
            session_id: Session ID
            query: Original search query
            output_path: Optional output path (defaults to fork_<session_id>.md)
            current_dir: Optional current working directory
            max_tokens: Maximum tokens to include
        
        Returns:
            Path to saved file
        """
        content = self.generate(
            session_id=session_id,
            query=query,
            current_dir=current_dir,
            max_tokens=max_tokens
        )
        
        if not output_path:
            short_id = session_id[:8]
            output_path = Path(f"fork_{short_id}.md")
        
        output_path.write_text(content, encoding="utf-8")
        logger.info(f"Fork.md saved to {output_path}")
        
        return output_path
