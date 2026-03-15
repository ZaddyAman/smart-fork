"""Session directory scanner for Kilo Code tasks (v2).

Walks the Kilo Code sessions directory, identifies new/changed sessions,
and orchestrates the indexing pipeline.
"""

import os
import time
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
from loguru import logger

from ..database.metadata_store import MetadataStore
from .session_parser import SessionParser


@dataclass
class ScanResult:
    """Result of a directory scan."""
    total_found: int = 0
    new_sessions: int = 0
    changed_sessions: int = 0
    unchanged_sessions: int = 0
    failed_sessions: int = 0
    new_session_paths: List[Path] = field(default_factory=list)
    changed_session_paths: List[Path] = field(default_factory=list)


class SessionScanner:
    """Scans Kilo Code session directories and identifies sessions to index.
    
    Compares session folders against the MetadataStore to find:
    - New sessions (not yet indexed)
    - Changed sessions (files modified since last index)
    - Unchanged sessions (skip)
    
    Usage:
        scanner = SessionScanner(sessions_path, metadata_store)
        result = scanner.scan()
        for path in result.new_session_paths + result.changed_session_paths:
            doc = parser.parse_session(path)
            store.upsert_session(doc)
    """
    
    def __init__(self, sessions_path: Path, metadata_store: MetadataStore):
        """Initialize scanner.
        
        Args:
            sessions_path: Path to Kilo Code tasks directory
            metadata_store: MetadataStore instance to check indexed status
        """
        self.sessions_path = Path(sessions_path)
        self.store = metadata_store
    
    def scan(self) -> ScanResult:
        """Scan all session folders and classify them.
        
        Returns:
            ScanResult with counts and lists of new/changed session paths
        """
        result = ScanResult()
        
        if not self.sessions_path.exists():
            logger.warning(f"Sessions directory does not exist: {self.sessions_path}")
            return result
        
        # Get all indexed session IDs
        indexed_ids = set(self.store.get_all_session_ids())
        
        # Walk session directories
        for entry in self.sessions_path.iterdir():
            if not entry.is_dir():
                continue
            
            # Skip hidden/system directories
            if entry.name.startswith('.'):
                continue
            
            # Verify it has at least one of the required files
            if not self._is_valid_session(entry):
                continue
            
            result.total_found += 1
            session_id = entry.name
            
            if session_id not in indexed_ids:
                # New session
                result.new_sessions += 1
                result.new_session_paths.append(entry)
            elif self._is_session_changed(entry, session_id):
                # Changed session
                result.changed_sessions += 1
                result.changed_session_paths.append(entry)
            else:
                # Unchanged
                result.unchanged_sessions += 1
        
        logger.info(
            f"Scan complete: {result.total_found} total, "
            f"{result.new_sessions} new, {result.changed_sessions} changed, "
            f"{result.unchanged_sessions} unchanged"
        )
        
        return result
    
    def get_unindexed_sessions(self) -> List[Path]:
        """Get paths of sessions not yet in the metadata store.
        
        Convenience method for getting only new sessions.
        
        Returns:
            List of session directory paths
        """
        result = self.scan()
        return result.new_session_paths
    
    def get_all_session_paths(self) -> List[Path]:
        """Get paths of all valid session directories.
        
        Returns:
            List of session directory paths, sorted by name (timestamp)
        """
        if not self.sessions_path.exists():
            return []
        
        paths = []
        for entry in sorted(self.sessions_path.iterdir()):
            if entry.is_dir() and not entry.name.startswith('.') and self._is_valid_session(entry):
                paths.append(entry)
        
        return paths
    
    def _is_valid_session(self, session_path: Path) -> bool:
        """Check if a directory is a valid Kilo Code session.
        
        A valid session must have at least api_conversation_history.json.
        
        Args:
            session_path: Path to potential session directory
        
        Returns:
            True if the directory contains valid session files
        """
        return (session_path / "api_conversation_history.json").exists()
    
    def _is_session_changed(self, session_path: Path, session_id: str) -> bool:
        """Check if session files have been modified since last index.
        
        Compares file modification timestamps against the indexed_at timestamp
        in the metadata store.
        
        Args:
            session_path: Path to session directory
            session_id: Session ID to check against store
        
        Returns:
            True if any session file has been modified since last index
        """
        doc = self.store.get_session(session_id)
        if not doc or not doc.indexed_at:
            return True
        
        indexed_at_seconds = doc.indexed_at / 1000.0  # Convert ms to seconds
        
        # Check mod times of all 3 possible files
        for filename in ["task_metadata.json", "api_conversation_history.json", "ui_messages.json"]:
            filepath = session_path / filename
            if filepath.exists():
                file_mtime = filepath.stat().st_mtime
                if file_mtime > indexed_at_seconds:
                    return True
        
        return False
