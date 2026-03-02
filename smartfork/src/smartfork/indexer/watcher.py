"""File watcher for Kilo Code transcript changes."""

import time
from pathlib import Path
from typing import Callable, List, Set, Optional
from loguru import logger

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent


class TranscriptWatcher:
    """Watches Kilo Code task directories for changes."""
    
    def __init__(
        self,
        tasks_path: Path,
        on_session_changed: Callable[[str, Path], None],
        poll_interval: float = 5.0
    ):
        """Initialize the watcher.
        
        Args:
            tasks_path: Path to Kilo Code tasks directory
            on_session_changed: Callback function(session_id, session_path)
            poll_interval: Interval in seconds for polling
        """
        self.tasks_path = Path(tasks_path)
        self.on_session_changed = on_session_changed
        self.poll_interval = poll_interval
        self.known_sessions: Set[str] = set()
        self.observer: Optional[Observer] = None
    
    def start(self) -> None:
        """Start watching for changes."""
        # Initial scan
        self._scan_existing_sessions()
        
        # Set up file system watcher
        if self.tasks_path.exists():
            event_handler = SessionEventHandler(self._on_file_changed)
            self.observer = Observer()
            self.observer.schedule(event_handler, str(self.tasks_path), recursive=True)
            self.observer.start()
            logger.info(f"Watching {self.tasks_path} for changes...")
        else:
            logger.warning(f"Tasks path does not exist: {self.tasks_path}")
    
    def stop(self) -> None:
        """Stop watching."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("Watcher stopped")
    
    def _scan_existing_sessions(self) -> None:
        """Scan for existing task directories."""
        if not self.tasks_path.exists():
            return
        
        for item in self.tasks_path.iterdir():
            if item.is_dir():
                self.known_sessions.add(item.name)
        
        logger.info(f"Found {len(self.known_sessions)} existing sessions")
    
    def _on_file_changed(self, event_path: str) -> None:
        """Handle file modification event.
        
        Args:
            event_path: Path of the changed file
        """
        path = Path(event_path)
        
        # Only process JSON files
        if not path.suffix == '.json':
            return
        
        # Extract task ID from path
        # Path format: tasks/<task_id>/api_conversation_history.json
        try:
            relative = path.relative_to(self.tasks_path)
            task_id = relative.parts[0]
            session_path = self.tasks_path / task_id
            
            if task_id in self.known_sessions:
                self.on_session_changed(task_id, session_path)
            else:
                # New session
                self.known_sessions.add(task_id)
                self.on_session_changed(task_id, session_path)
                logger.info(f"New session detected: {task_id}")
        except (ValueError, IndexError):
            # Not a task file
            pass
    
    def get_all_sessions(self) -> List[Path]:
        """Get paths to all task directories.
        
        Returns:
            List of paths to session directories
        """
        if not self.tasks_path.exists():
            return []
        
        sessions = []
        for item in self.tasks_path.iterdir():
            if item.is_dir():
                # Check if it's a valid task directory
                if (item / "api_conversation_history.json").exists():
                    sessions.append(item)
        
        return sessions
    
    def is_valid_session(self, task_id: str) -> bool:
        """Check if a session exists and is valid.
        
        Args:
            task_id: Session ID to check
            
        Returns:
            True if session exists and has conversation history
        """
        session_path = self.tasks_path / task_id
        return (
            session_path.exists() and
            session_path.is_dir() and
            (session_path / "api_conversation_history.json").exists()
        )


class SessionEventHandler(FileSystemEventHandler):
    """Handles file system events for session files."""
    
    def __init__(self, callback: Callable[[str], None]):
        """Initialize handler.
        
        Args:
            callback: Function to call with file path
        """
        self.callback = callback
    
    def on_modified(self, event):
        """Handle file modification."""
        if not event.is_directory:
            self.callback(event.src_path)
    
    def on_created(self, event):
        """Handle file creation."""
        if not event.is_directory:
            self.callback(event.src_path)
