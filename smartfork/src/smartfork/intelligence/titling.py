"""Session auto-titling using LLM/AI for human-readable session titles."""

import json
import re
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
from loguru import logger

from ..database.models import TaskSession, SessionMetadata


class TitleGenerator:
    """Generates human-readable titles for Kilo Code sessions.
    
    Uses a hybrid approach:
    1. First tries to extract a meaningful title from the first user message
    2. Falls back to LLM-based generation if available
    3. Uses heuristic-based generation as final fallback
    """
    
    # Common stop words and patterns to filter out
    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
        'please', 'help', 'need', 'want', 'like', 'would', 'could', 'should'
    }
    
    # Task patterns to detect in messages
    TASK_PATTERNS = [
        (r'(?:implement|create|build|add|setup|configure)\s+([a-zA-Z_\-\s]+?)(?:\s+(?:for|in|using|with)|\?|$)', 'Implementation'),
        (r'(?:fix|debug|resolve|solve)\s+(?:the\s+)?([a-zA-Z_\-\s]+?)(?:\s+(?:in|with|error|issue)|\?|$)', 'Bug Fix'),
        (r'(?:refactor|rewrite|reorganize|clean\s+up)\s+(?:the\s+)?([a-zA-Z_\-\s]+?)(?:\s+(?:in|code)|\?|$)', 'Refactoring'),
        (r'(?:test|testing|write\s+tests?\s+for)\s+(?:the\s+)?([a-zA-Z_\-\s]+?)(?:\s|$)', 'Testing'),
        (r'(?:optimize|improve|enhance|performance)\s+(?:the\s+)?([a-zA-Z_\-\s]+?)(?:\s|$)', 'Optimization'),
        (r'(?:migrate|upgrade|update)\s+(?:the\s+)?([a-zA-Z_\-\s]+?)(?:\s+(?:to|from)|\?|$)', 'Migration'),
        (r'(?:analyze|review|audit)\s+(?:the\s+)?([a-zA-Z_\-\s]+?)(?:\s|$)', 'Analysis'),
        (r'(?:deploy|release|publish)\s+(?:the\s+)?([a-zA-Z_\-\s]+?)(?:\s|$)', 'Deployment'),
    ]
    
    def __init__(self):
        """Initialize the title generator."""
        pass
    
    def generate_title(
        self, 
        session: TaskSession, 
        max_length: int = 60
    ) -> str:
        """Generate a human-readable title for a session.
        
        Uses a free heuristic-based approach to extract or generate titles
        from session content without requiring external APIs.
        
        Args:
            session: The TaskSession to generate a title for
            max_length: Maximum length of the generated title
            
        Returns:
            Generated title string
        """
        # Try extraction from first user message
        title = self._extract_from_first_message(session)
        if title and len(title) >= 10:
            return self._truncate(title, max_length)
        
        # Fall back to heuristic generation
        title = self._generate_heuristic(session)
        return self._truncate(title, max_length)
    
    def _extract_from_first_message(self, session: TaskSession) -> Optional[str]:
        """Try to extract title from the first user message.
        
        Args:
            session: TaskSession to analyze
            
        Returns:
            Extracted title or None
        """
        # Find first user message
        first_user_msg = None
        for msg in session.conversation:
            if msg.role == 'user' and msg.content:
                first_user_msg = msg.content
                break
        
        if not first_user_msg:
            return None
        
        # Clean up the message
        content = first_user_msg.strip()
        
        # Remove common prefixes
        prefixes = [
            r'^hi\s*,?\s*',
            r'^hello\s*,?\s*',
            r'^hey\s*,?\s*',
            r'^please\s+',
            r'^can\s+you\s+',
            r'^could\s+you\s+',
            r'^would\s+you\s+',
            r'^i\s+need\s+',
            r'^i\s+want\s+',
            r'^help\s+me\s+',
        ]
        
        for prefix in prefixes:
            content = re.sub(prefix, '', content, flags=re.IGNORECASE)
        
        content = content.strip()
        
        # If it's a question, extract the subject
        if content.endswith('?'):
            content = content[:-1].strip()
        
        # Take first sentence if multiple
        if '. ' in content:
            content = content.split('. ')[0]
        
        # Clean up
        content = content.strip()
        
        # Must be reasonable length
        if len(content) < 10 or len(content) > 200:
            return None
        
        return content
    
    def _generate_heuristic(self, session: TaskSession) -> str:
        """Generate title using heuristics when LLM is not available.
        
        Args:
            session: TaskSession to analyze
            
        Returns:
            Generated title
        """
        # Get all text from the session
        full_text = session.get_full_text()
        
        # Try to detect task type and subject
        task_type = None
        task_subject = None
        
        for pattern, detected_type in self.TASK_PATTERNS:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                task_type = detected_type
                task_subject = match.group(1).strip()
                break
        
        # Get languages from file extensions
        languages = self._detect_languages(session.metadata.files_in_context)
        
        # Build title components
        components = []
        
        if task_type:
            components.append(task_type)
        
        if task_subject:
            # Clean up subject
            subject = task_subject.replace('_', ' ').replace('-', ' ')
            subject = ' '.join(word for word in subject.split() if len(word) > 2)
            components.append(subject)
        
        if languages:
            # Add primary language
            lang = languages[0]
            if task_type:
                components.append(f"in {lang}")
            else:
                components.append(lang)
        
        # Get files context if no other info
        if not components and session.metadata.files_in_context:
            # Extract meaningful file names
            files = session.metadata.files_in_context[:3]
            file_names = []
            for f in files:
                name = Path(f).stem
                if name not in ['index', 'main', 'app', 'utils']:
                    file_names.append(name)
            
            if file_names:
                components.append(f"Working on {', '.join(file_names[:2])}")
        
        # Final fallback
        if not components:
            # Use timestamp
            timestamp = session.get_last_timestamp()
            if timestamp:
                dt = datetime.fromtimestamp(timestamp / 1000)
                return f"Session from {dt.strftime('%Y-%m-%d %H:%M')}"
            else:
                return f"Session {session.task_id[:8]}"
        
        return ' '.join(components)
    
    def _detect_languages(self, files: List[str]) -> List[str]:
        """Detect programming languages from file extensions only.
        
        This avoids false positives from text-based keyword matching.
        
        Args:
            files: List of file paths
            
        Returns:
            List of detected languages
        """
        lang_scores: Dict[str, int] = {}
        
        # Check file extensions only - no text keyword matching
        lang_extensions = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.jsx': 'React',
            '.tsx': 'React/TypeScript',
            '.vue': 'Vue',
            '.rb': 'Ruby',
            '.go': 'Go',
            '.rs': 'Rust',
            '.java': 'Java',
            '.kt': 'Kotlin',
            '.swift': 'Swift',
            '.cpp': 'C++',
            '.c': 'C',
            '.cs': 'C#',
            '.php': 'PHP',
            '.html': 'HTML',
            '.css': 'CSS',
            '.scss': 'SCSS',
            '.sql': 'SQL',
            '.r': 'R',
            '.scala': 'Scala',
            '.sh': 'Shell',
            '.ps1': 'PowerShell',
        }
        
        for file_path in files:
            ext = Path(file_path).suffix.lower()
            if ext in lang_extensions:
                lang = lang_extensions[ext]
                lang_scores[lang] = lang_scores.get(lang, 0) + 1
        
        # Sort by frequency
        sorted_langs = sorted(lang_scores.items(), key=lambda x: x[1], reverse=True)
        return [lang for lang, _ in sorted_langs[:3]]
    
    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text to max length gracefully.
        
        Args:
            text: Text to truncate
            max_length: Maximum allowed length
            
        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        
        # Try to break at word boundary
        truncated = text[:max_length - 3]
        last_space = truncated.rfind(' ')
        
        if last_space > max_length * 0.5:  # Only break if we keep at least half
            truncated = truncated[:last_space]
        
        return truncated.rstrip() + '...'


class TitleManager:
    """Manages session titles in the database."""
    
    def __init__(self, db: Any, title_generator: Optional[TitleGenerator] = None):
        """Initialize the title manager.
        
        Args:
            db: ChromaDatabase instance
            title_generator: Optional TitleGenerator instance
        """
        self.db = db
        self.generator = title_generator or TitleGenerator()
    
    def generate_and_store_title(self, session: TaskSession) -> str:
        """Generate a title for a session and return it.
        
        Note: Titles are stored in chunk metadata, so this generates
        the title to be used during indexing.
        
        Args:
            session: TaskSession to generate title for
            
        Returns:
            Generated title
        """
        title = self.generator.generate_title(session)
        logger.debug(f"Generated title for {session.task_id[:8]}: {title}")
        return title
    
    def batch_generate_titles(
        self, 
        sessions: List[TaskSession],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, str]:
        """Generate titles for multiple sessions.
        
        Args:
            sessions: List of TaskSessions
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict mapping session_id to title
        """
        titles = {}
        
        for i, session in enumerate(sessions):
            title = self.generate_and_store_title(session)
            titles[session.task_id] = title
            
            if progress_callback:
                progress_callback(i + 1, len(sessions), session.task_id, title)
        
        return titles
    
    def update_session_title_in_db(self, session_id: str, title: str) -> bool:
        """Update the title for an existing session in the database.
        
        Note: Since ChromaDB doesn't support updating metadata directly,
        this would require re-indexing the session. This method is a placeholder
        for when we implement title updates via re-indexing.
        
        Args:
            session_id: Session ID to update
            title: New title
            
        Returns:
            True if successful
        """
        # ChromaDB doesn't support metadata updates without re-adding
        # This would require getting all chunks, updating metadata, and re-adding
        # For now, this is handled by re-indexing
        logger.info(f"Title update for {session_id} requires re-indexing: {title}")
        return False


def generate_title_for_session(session: TaskSession) -> str:
    """Convenience function to generate a title for a session.
    
    Args:
        session: TaskSession to generate title for
        
    Returns:
        Generated title
    """
    generator = TitleGenerator()
    return generator.generate_title(session)