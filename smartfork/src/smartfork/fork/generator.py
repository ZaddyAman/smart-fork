"""Fork.md generator for SmartFork."""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from loguru import logger

from ..database.chroma_db import ChromaDatabase
from ..database.models import Chunk


class SessionAnalyzer:
    """Analyzes session content to extract key information."""
    
    DEPENDENCY_PATTERNS = [
        r'pip install ([\w\-\[\]]+)',
        r'npm install ([\w\-@/]+)',
        r'yarn add ([\w\-@/]+)',
        r'poetry add ([\w\-]+)',
        r'requirements\.txt[\s\S]*?([\w\-]+)(?:[=<>!~])',
        r'package\.json[\s\S]*?"([\w\-]+)":\s*"[\^~]?[\d\.]+"',
        r'from\s+([\w\.]+)\s+import',
        r'import\s+([\w\.]+)',
    ]
    
    NEXT_STEP_PATTERNS = [
        r'(?:TODO|FIXME|XXX|HACK)[\s:]*(.+?)(?:\n|$)',
        r'next[\s:]+(.+?)(?:\n|$)',
        r'should[\s:]+(.+?)(?:\n|$)',
        r'need[s]?[\s:]+to[\s:]+(.+?)(?:\n|$)',
    ]
    
    def analyze(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Analyze session chunks and extract key information.
        
        Args:
            chunks: List of session chunks
            
        Returns:
            Dictionary with analysis results
        """
        all_text = " ".join([c.content for c in chunks])
        
        # Extract code snippets
        code_snippets = self._extract_code_snippets(all_text)
        
        # Extract dependencies
        dependencies = self._extract_dependencies(all_text)
        
        # Extract next steps
        next_steps = self._extract_next_steps(all_text)
        
        # Identify key topics
        key_topics = self._identify_topics(chunks)
        
        # Get date range
        date_range = self._get_date_range(chunks)
        
        return {
            "key_topics": key_topics,
            "code_snippets": code_snippets[:5],  # Top 5 snippets
            "dependencies": list(set(dependencies))[:10],  # Unique deps
            "next_steps": next_steps[:5],  # Top 5 next steps
            "date_range": date_range,
            "files_in_context": chunks[0].metadata.files_in_context if chunks else []
        }
    
    def _extract_code_snippets(self, text: str) -> List[Dict[str, str]]:
        """Extract code snippets from text."""
        snippets = []
        
        # Match code blocks
        pattern = r'```(\w+)?\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for lang, code in matches:
            if len(code.strip()) > 50:  # Only substantial snippets
                snippets.append({
                    "language": lang or "python",
                    "code": code.strip()
                })
        
        return snippets
    
    def _extract_dependencies(self, text: str) -> List[str]:
        """Extract mentioned dependencies."""
        deps = []
        
        for pattern in self.DEPENDENCY_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            deps.extend(matches)
        
        # Clean and filter
        deps = [d.strip() for d in deps if len(d.strip()) > 1]
        
        return deps
    
    def _extract_next_steps(self, text: str) -> List[str]:
        """Extract next steps / TODOs."""
        steps = []
        
        for pattern in self.NEXT_STEP_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            steps.extend([m.strip() for m in matches if len(m.strip()) > 10])
        
        return steps
    
    def _identify_topics(self, chunks: List[Chunk]) -> List[str]:
        """Identify key topics from chunks."""
        # Extract topics from file names and extensions
        topics = set()
        file_extensions = {
            'py': 'Python', 'js': 'JavaScript', 'ts': 'TypeScript',
            'tsx': 'React/TSX', 'jsx': 'React/JSX', 'java': 'Java',
            'go': 'Go', 'rs': 'Rust', 'cpp': 'C++', 'c': 'C',
            'h': 'Header', 'hpp': 'C++ Header', 'cs': 'C#',
            'rb': 'Ruby', 'php': 'PHP', 'swift': 'Swift',
            'kt': 'Kotlin', 'scala': 'Scala', 'r': 'R',
            'md': 'Markdown', 'json': 'JSON', 'yaml': 'YAML',
            'yml': 'YAML', 'toml': 'TOML', 'ini': 'Config',
            'sql': 'SQL', 'sh': 'Shell', 'ps1': 'PowerShell',
            'dockerfile': 'Docker', 'tf': 'Terraform'
        }
        
        for chunk in chunks:
            for file_path in chunk.metadata.files_in_context:
                # Get file extension
                ext = file_path.split('.')[-1].lower() if '.' in file_path else ''
                if ext in file_extensions:
                    topics.add(file_extensions[ext])
                # Get file name (last part of path)
                file_name = file_path.split('/')[-1].split('.')[0]
                if file_name and file_name not in ['__init__', 'index', 'main', 'app']:
                    topics.add(file_name)
        
        return list(topics)[:10]
    
    def _get_date_range(self, chunks: List[Chunk]) -> Dict[str, Optional[str]]:
        """Get date range from chunks."""
        timestamps = []
        
        for chunk in chunks:
            if chunk.metadata.timestamp:
                timestamps.append(chunk.metadata.timestamp)
            if chunk.metadata.last_active:
                timestamps.append(chunk.metadata.last_active)
        
        if timestamps:
            return {
                "start": min(timestamps),
                "end": max(timestamps)
            }
        
        return {"start": None, "end": None}


class ForkMDGenerator:
    """Generates fork.md context files."""
    
    def __init__(self, db: ChromaDatabase):
        """Initialize the generator.
        
        Args:
            db: ChromaDatabase instance
        """
        self.db = db
        self.analyzer = SessionAnalyzer()
    
    def generate(self, session_id: str, query: str, current_dir: Optional[str] = None) -> str:
        """Generate a fork.md file for a session.
        
        Args:
            session_id: Session ID to generate fork for
            query: Original search query
            current_dir: Current working directory
            
        Returns:
            Markdown content
        """
        # Load session chunks
        chunks = self.db.get_session_chunks(session_id)
        
        if not chunks:
            logger.warning(f"No chunks found for session {session_id}")
            return f"# Context Fork: Session {session_id}\n\nNo content available."
        
        # Analyze session
        analysis = self.analyzer.analyze(chunks)
        
        # Build sections
        sections = [
            self._generate_header(session_id, analysis),
            self._generate_summary(analysis),
            self._generate_file_details(analysis),
            self._generate_code_snippets(analysis),
            self._generate_dependencies(analysis),
            self._generate_next_steps(analysis),
            self._generate_relevance_note(query, analysis, current_dir)
        ]
        
        return '\n\n'.join(sections)
    
    def _generate_header(self, session_id: str, analysis: Dict[str, Any]) -> str:
        """Generate the header section."""
        date_range = analysis.get("date_range", {})
        start = date_range.get("start", "Unknown")
        end = date_range.get("end", "Unknown")
        
        topics = ", ".join(analysis.get("key_topics", [])[:5]) or "None identified"
        files_count = len(analysis.get("files_in_context", []))
        
        return f"""# Context Fork: Session {session_id}

## Session Overview
- **Session ID**: {session_id}
- **Date Range**: {start} to {end}
- **Files Discussed**: {files_count} files
- **Key Topics**: {topics}"""
    
    def _generate_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate context summary."""
        topics = analysis.get("key_topics", [])
        
        if topics:
            topics_str = ", ".join(topics[:5])
            summary = f"This session covered work related to {topics_str}."
        else:
            summary = "This session covered various implementation details."
        
        return f"""## Context Summary
{summary}

### Key Topics Covered
{chr(10).join([f'- {t}' for t in topics[:8]])}"""
    
    def _generate_file_details(self, analysis: Dict[str, Any]) -> str:
        """Generate file details section."""
        files = analysis.get("files_in_context", [])
        
        if not files:
            return "## Key Files\n\nNo files were tracked in this session."
        
        files_md = "\n".join([f"- `{f}`" for f in files[:15]])
        
        if len(files) > 15:
            files_md += f"\n- ... and {len(files) - 15} more files"
        
        return f"""## Key Files
{files_md}"""
    
    def _generate_code_snippets(self, analysis: Dict[str, Any]) -> str:
        """Generate code snippets section."""
        snippets = analysis.get("code_snippets", [])
        
        if not snippets:
            return "## Relevant Code Snippets\n\nNo substantial code snippets found."
        
        snippets_md = []
        for i, snippet in enumerate(snippets[:3], 1):
            lang = snippet.get("language", "")
            code = snippet.get("code", "")
            
            # Truncate if too long
            if len(code) > 500:
                code = code[:500] + "\n# ... (truncated)"
            
            snippets_md.append(f"### Snippet {i}\n```{lang}\n{code}\n```")
        
        return "## Relevant Code Snippets\n\n" + "\n\n".join(snippets_md)
    
    def _generate_dependencies(self, analysis: Dict[str, Any]) -> str:
        """Generate dependencies section."""
        deps = analysis.get("dependencies", [])
        
        if not deps:
            return "## Dependencies\n\nNo specific dependencies identified."
        
        deps_md = "\n".join([f"- `{d}`" for d in deps])
        
        return f"""## Dependencies Mentioned
{deps_md}"""
    
    def _generate_next_steps(self, analysis: Dict[str, Any]) -> str:
        """Generate next steps section."""
        steps = analysis.get("next_steps", [])
        
        if not steps:
            return "## Next Steps from Original Session\n\nNo explicit next steps identified."
        
        steps_md = "\n".join([f"- [ ] {s}" for s in steps[:5]])
        
        return f"""## Next Steps from Original Session
{steps_md}"""
    
    def _generate_relevance_note(
        self,
        query: str,
        analysis: Dict[str, Any],
        current_dir: Optional[str]
    ) -> str:
        """Generate relevance note section."""
        files = analysis.get("files_in_context", [])
        
        overlap_note = ""
        if current_dir and files:
            # Count files in current directory
            current_path = Path(current_dir)
            matching = sum(1 for f in files if current_path.name in f or str(current_path) in f)
            if matching > 0:
                overlap_note = f"\n- **Directory Overlap**: {matching} files from this session are in your current workspace"
        
        return f"""## How This Relates to Your Current Work
- **Current Query**: "{query}"
- **Topics Match**: {len(analysis.get('key_topics', []))} topics identified{overlap_note}

---

*Generated by SmartFork - AI Session Intelligence*"""
    
    def save(self, session_id: str, query: str, output_path: Optional[Path] = None) -> Path:
        """Generate and save fork.md file.
        
        Args:
            session_id: Session ID
            query: Original search query
            output_path: Optional output path (defaults to fork_<session_id>.md)
            
        Returns:
            Path to saved file
        """
        content = self.generate(session_id, query)
        
        if not output_path:
            short_id = session_id[:8]
            output_path = Path(f"fork_{short_id}.md")
        
        output_path.write_text(content, encoding="utf-8")
        logger.info(f"Fork.md saved to {output_path}")
        
        return output_path
