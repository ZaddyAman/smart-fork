# SmartFork Phase 0: Foundation Implementation Plan

## Overview

Phase 0 establishes the core infrastructure for SmartFork. This phase focuses on building a solid foundation with offline-first architecture, basic transcript watching, and vector database integration. The goal is to create a working system that can index Kilo Code sessions and perform simple searches.

**Prerequisites**: Python 3.13 installed, Kilo Code extension in Cursor IDE
**Timeline**: 2 weeks
**Success Criteria**: Indexes 100 sessions, full re-index <5 seconds, CLI functional

---

## 1. Project Structure

```
smartfork/
├── src/
│   └── smartfork/
│       ├── __init__.py
│       ├── cli.py                    # CLI entry point
│       ├── config.py                 # Configuration management
│       ├── database/
│       │   ├── __init__.py
│       │   ├── chroma_db.py         # ChromaDB integration
│       │   └── models.py            # Data models
│       ├── indexer/
│       │   ├── __init__.py
│       │   ├── watcher.py           # Transcript watcher
│       │   └── parser.py            # Kilo Code JSON parser
│       ├── search/
│       │   ├── __init__.py
│       │   └── semantic.py          # Basic semantic search
│       └── utils/
│           ├── __init__.py
│           └── logger.py
├── data/                             # Local data storage
│   ├── chroma_db/                   # Vector database files
│   └── cache/                       # Temporary files
├── tests/
│   ├── __init__.py
│   ├── test_parser.py
│   ├── test_watcher.py
│   └── test_search.py
├── requirements.txt
├── README.md
└── setup.py
```

---

## 2. Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Python | CPython | 3.13 |
| Vector DB | ChromaDB | ^0.4.x |
| Embeddings | sentence-transformers | ^2.2.x |
| CLI | Typer | ^0.9.x |
| Config | Pydantic | ^2.0.x |
| Testing | pytest | ^7.4.x |
| Logging | loguru | ^0.7.x |

---

## 3. Module Specifications

### 3.1 Configuration Module (`config.py`)

```python
from pydantic import BaseSettings, Field
from pathlib import Path

class SmartForkConfig(BaseSettings):
    """Configuration for SmartFork."""
    
    # Kilo Code paths
    kilo_code_tasks_path: Path = Field(
        default=Path.home() / "AppData/Roaming/Cursor/User/globalStorage/kilocode.kilo-code/tasks",
        description="Path to Kilo Code task storage"
    )
    
    # Database paths
    chroma_db_path: Path = Field(
        default=Path.home() / ".smartfork/chroma_db",
        description="Path to ChromaDB storage"
    )
    
    # Indexing settings
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    chunk_size: int = 512
    chunk_overlap: int = 128
    
    # Search settings
    default_search_results: int = 10
    
    class Config:
        env_prefix = "SMARTFORK_"
        env_file = ".env"

# Global config instance
config = SmartForkConfig()
```

### 3.2 Database Module (`chroma_db.py`)

```python
import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Any
import hashlib

class ChromaDatabase:
    """Manages ChromaDB connection and operations."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """Get or create the sessions collection."""
        return self.client.get_or_create_collection(
            name="sessions",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add conversation chunks to the database."""
        if not chunks:
            return
        
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]
        metadatas = [chunk.metadata.dict() for chunk in chunks]
        
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    def search(self, query_embedding: List[float], n_results: int = 10) -> List[SearchResult]:
        """Search for similar chunks."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        return self._format_results(results)
    
    def delete_session(self, session_id: str) -> None:
        """Delete all chunks for a session."""
        self.collection.delete(
            where={"session_id": session_id}
        )
    
    def get_session_count(self) -> int:
        """Get total number of indexed chunks."""
        return self.collection.count()
    
    def reset(self) -> None:
        """Clear all data (use with caution)."""
        self.client.reset()
        self.collection = self._get_or_create_collection()


class Chunk(BaseModel):
    """Represents a chunk of conversation."""
    id: str
    content: str
    embedding: List[float]
    metadata: ChunkMetadata


class ChunkMetadata(BaseModel):
    """Metadata for a chunk."""
    session_id: str
    task_id: str
    chunk_index: int
    timestamp: Optional[str] = None
    files_in_context: List[str] = []
    message_type: str  # "user" | "assistant" | "tool"
```

### 3.3 Parser Module (`parser.py`)

```python
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel

class KiloCodeParser:
    """Parses Kilo Code transcript files."""
    
    def parse_task_directory(self, task_path: Path) -> Optional[TaskSession]:
        """Parse a Kilo Code task directory."""
        if not task_path.is_dir():
            return None
        
        task_id = task_path.name
        
        # Load metadata
        metadata_path = task_path / "task_metadata.json"
        metadata = self._parse_metadata(metadata_path) if metadata_path.exists() else {}
        
        # Load conversation history
        history_path = task_path / "api_conversation_history.json"
        conversation = self._parse_conversation(history_path) if history_path.exists() else []
        
        # Load UI messages
        ui_path = task_path / "ui_messages.json"
        ui_messages = self._parse_ui_messages(ui_path) if ui_path.exists() else []
        
        return TaskSession(
            task_id=task_id,
            metadata=metadata,
            conversation=conversation,
            ui_messages=ui_messages
        )
    
    def _parse_metadata(self, path: Path) -> TaskMetadata:
        """Parse task_metadata.json."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        files_in_context = []
        if 'files_in_context' in data:
            for file_record in data['files_in_context']:
                if file_record.get('record_state') == 'active':
                    files_in_context.append(file_record['path'])
        
        return TaskMetadata(
            files_in_context=files_in_context,
            raw_data=data
        )
    
    def _parse_conversation(self, path: Path) -> List[ConversationMessage]:
        """Parse api_conversation_history.json."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        messages = []
        for msg in data:
            messages.append(ConversationMessage(
                role=msg.get('role', 'unknown'),
                content=self._extract_content(msg),
                timestamp=msg.get('ts'),
                type=msg.get('type')
            ))
        
        return messages
    
    def _parse_ui_messages(self, path: Path) -> List[UIMessage]:
        """Parse ui_messages.json."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        messages = []
        for msg in data:
            messages.append(UIMessage(
                type=msg.get('type'),
                say=msg.get('say'),
                ask=msg.get('ask'),
                text=msg.get('text'),
                ts=msg.get('ts')
            ))
        
        return messages
    
    def _extract_content(self, msg: Dict[str, Any]) -> str:
        """Extract text content from a message."""
        if 'content' in msg:
            if isinstance(msg['content'], str):
                return msg['content']
            elif isinstance(msg['content'], list):
                # Handle array of content parts
                texts = []
                for part in msg['content']:
                    if isinstance(part, dict) and 'text' in part:
                        texts.append(part['text'])
                return ' '.join(texts)
        return ''


class TaskSession(BaseModel):
    """Represents a parsed Kilo Code session."""
    task_id: str
    metadata: TaskMetadata
    conversation: List[ConversationMessage]
    ui_messages: List[UIMessage]
    
    def get_full_text(self) -> str:
        """Get concatenated text of all messages."""
        texts = []
        for msg in self.conversation:
            if msg.content:
                texts.append(f"[{msg.role}]: {msg.content}")
        return '\n\n'.join(texts)


class TaskMetadata(BaseModel):
    """Task metadata from task_metadata.json."""
    files_in_context: List[str]
    raw_data: Dict[str, Any]


class ConversationMessage(BaseModel):
    """A message from api_conversation_history.json."""
    role: str
    content: str
    timestamp: Optional[int] = None
    type: Optional[str] = None


class UIMessage(BaseModel):
    """A message from ui_messages.json."""
    type: Optional[str] = None
    say: Optional[str] = None
    ask: Optional[str] = None
    text: Optional[str] = None
    ts: Optional[int] = None
```

### 3.4 Watcher Module (`watcher.py`)

```python
from pathlib import Path
from typing import Callable, List, Set
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

class TranscriptWatcher:
    """Watches Kilo Code task directories for changes."""
    
    def __init__(
        self,
        tasks_path: Path,
        on_session_changed: Callable[[str, Path], None],
        poll_interval: float = 5.0
    ):
        self.tasks_path = tasks_path
        self.on_session_changed = on_session_changed
        self.poll_interval = poll_interval
        self.known_sessions: Set[str] = set()
        self.observer: Optional[Observer] = None
    
    def start(self) -> None:
        """Start watching for changes."""
        # Initial scan
        self._scan_existing_sessions()
        
        # Set up file system watcher
        event_handler = SessionEventHandler(self._on_file_changed)
        self.observer = Observer()
        self.observer.schedule(event_handler, str(self.tasks_path), recursive=True)
        self.observer.start()
        
        console.print(f"[green]Watching {self.tasks_path} for changes...[/green]")
    
    def stop(self) -> None:
        """Stop watching."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
    
    def _scan_existing_sessions(self) -> None:
        """Scan for existing task directories."""
        if not self.tasks_path.exists():
            return
        
        for item in self.tasks_path.iterdir():
            if item.is_dir():
                self.known_sessions.add(item.name)
    
    def _on_file_changed(self, event: FileModifiedEvent) -> None:
        """Handle file modification event."""
        if not event.src_path.endswith('.json'):
            return
        
        # Extract task ID from path
        path = Path(event.src_path)
        task_id = path.parent.name
        
        if task_id in self.known_sessions:
            self.on_session_changed(task_id, path.parent)
        else:
            # New session
            self.known_sessions.add(task_id)
            self.on_session_changed(task_id, path.parent)
    
    def get_all_sessions(self) -> List[Path]:
        """Get paths to all task directories."""
        if not self.tasks_path.exists():
            return []
        
        return [
            item for item in self.tasks_path.iterdir()
            if item.is_dir() and (item / "api_conversation_history.json").exists()
        ]


class SessionEventHandler(FileSystemEventHandler):
    """Handles file system events for session files."""
    
    def __init__(self, callback: Callable[[FileModifiedEvent], None]):
        self.callback = callback
    
    def on_modified(self, event):
        if not event.is_directory:
            self.callback(event)
```

### 3.5 Indexer Module (`indexer.py`)

```python
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer
from .parser import KiloCodeParser, TaskSession
from .chroma_db import ChromaDatabase, Chunk, ChunkMetadata

class FullIndexer:
    """Performs full re-indexing of sessions."""
    
    def __init__(
        self,
        db: ChromaDatabase,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    ):
        self.db = db
        self.parser = KiloCodeParser()
        self.model = SentenceTransformer(model_name)
    
    def index_all_sessions(self, tasks_path: Path) -> IndexingResult:
        """Index all sessions in the tasks directory."""
        sessions = list(tasks_path.iterdir())
        indexed = 0
        failed = 0
        
        for session_path in sessions:
            if not session_path.is_dir():
                continue
            
            try:
                self.index_session(session_path)
                indexed += 1
            except Exception as e:
                logger.error(f"Failed to index {session_path.name}: {e}")
                failed += 1
        
        return IndexingResult(indexed=indexed, failed=failed)
    
    def index_session(self, session_path: Path) -> None:
        """Index a single session."""
        task_id = session_path.name
        
        # Delete existing data for this session
        self.db.delete_session(task_id)
        
        # Parse session
        session = self.parser.parse_task_directory(session_path)
        if not session:
            return
        
        # Chunk the conversation
        chunks = self._create_chunks(session)
        
        # Generate embeddings
        for chunk in chunks:
            chunk.embedding = self.model.encode(chunk.content).tolist()
        
        # Store in database
        self.db.add_chunks(chunks)
        
        logger.info(f"Indexed {len(chunks)} chunks for session {task_id[:8]}")
    
    def _create_chunks(self, session: TaskSession) -> List[Chunk]:
        """Create chunks from session conversation."""
        chunks = []
        full_text = session.get_full_text()
        
        # Simple chunking strategy
        chunk_size = 512
        overlap = 128
        
        text_chunks = self._split_text(full_text, chunk_size, overlap)
        
        for idx, text in enumerate(text_chunks):
            chunk_id = f"{session.task_id}_{idx}"
            chunks.append(Chunk(
                id=chunk_id,
                content=text,
                embedding=[],  # Will be filled later
                metadata=ChunkMetadata(
                    session_id=session.task_id,
                    task_id=session.task_id,
                    chunk_index=idx,
                    files_in_context=session.metadata.files_in_context,
                    message_type="mixed"
                )
            ))
        
        return chunks
    
    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap
        
        return chunks


class IndexingResult(BaseModel):
    """Result of indexing operation."""
    indexed: int
    failed: int
```

### 3.6 Search Module (`semantic.py`)

```python
from sentence_transformers import SentenceTransformer
from .chroma_db import ChromaDatabase

class SemanticSearchEngine:
    """Basic semantic search using embeddings."""
    
    def __init__(
        self,
        db: ChromaDatabase,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    ):
        self.db = db
        self.model = SentenceTransformer(model_name)
    
    def search(self, query: str, n_results: int = 10) -> List[SearchResult]:
        """Search for sessions matching the query."""
        # Embed query
        query_embedding = self.model.encode(query).tolist()
        
        # Search database
        results = self.db.search(query_embedding, n_results)
        
        return results


class SearchResult(BaseModel):
    """Result from semantic search."""
    session_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
```

### 3.7 CLI Module (`cli.py`)

```python
import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="SmartFork - AI Session Intelligence")
console = Console()

@app.command()
def index(
    force: bool = typer.Option(False, "--force", "-f", help="Force full re-index"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Index all Kilo Code sessions."""
    from .config import config
    from .chroma_db import ChromaDatabase
    from .indexer import FullIndexer
    
    console.print("[bold blue]SmartFork Indexer[/bold blue]")
    console.print(f"Tasks path: {config.kilo_code_tasks_path}")
    console.print(f"Database: {config.chroma_db_path}")
    
    # Initialize components
    db = ChromaDatabase(config.chroma_db_path)
    
    if force:
        console.print("[yellow]Resetting database...[/yellow]")
        db.reset()
    
    indexer = FullIndexer(db)
    
    # Perform indexing
    with console.status("[bold green]Indexing sessions..."):
        result = indexer.index_all_sessions(config.kilo_code_tasks_path)
    
    console.print(f"[green]Indexed {result.indexed} sessions[/green]")
    if result.failed > 0:
        console.print(f"[red]Failed: {result.failed} sessions[/red]")
    
    console.print(f"Total chunks in database: {db.get_session_count()}")

@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    n_results: int = typer.Option(5, "--results", "-n", help="Number of results"),
):
    """Search indexed sessions."""
    from .config import config
    from .chroma_db import ChromaDatabase
    from .search.semantic import SemanticSearchEngine
    
    db = ChromaDatabase(config.chroma_db_path)
    engine = SemanticSearchEngine(db)
    
    console.print(f"[bold]Searching for:[/bold] {query}\n")
    
    results = engine.search(query, n_results)
    
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return
    
    # Display results
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Session ID", style="dim")
    table.add_column("Score")
    table.add_column("Content Preview")
    
    for r in results:
        preview = r.content[:100] + "..." if len(r.content) > 100 else r.content
        score_pct = f"{r.score:.1%}"
        table.add_row(r.session_id[:8], score_pct, preview)
    
    console.print(table)

@app.command()
def status():
    """Show indexing status."""
    from .config import config
    from .chroma_db import ChromaDatabase
    from .indexer.watcher import TranscriptWatcher
    
    db = ChromaDatabase(config.chroma_db_path)
    watcher = TranscriptWatcher(config.kilo_code_tasks_path, lambda x, y: None)
    
    all_sessions = watcher.get_all_sessions()
    indexed_count = db.get_session_count()
    
    console.print("[bold]SmartFork Status[/bold]\n")
    console.print(f"Kilo Code tasks path: {config.kilo_code_tasks_path}")
    console.print(f"Total task directories: {len(all_sessions)}")
    console.print(f"Indexed chunks: {indexed_count}")
    console.print(f"Database path: {config.chroma_db_path}")

@app.command()
def config_show():
    """Show current configuration."""
    from .config import config
    
    console.print("[bold]SmartFork Configuration[/bold]\n")
    for key, value in config.dict().items():
        console.print(f"{key}: {value}")

if __name__ == "__main__":
    app()
```

---

## 4. Testing Plan

### 4.1 Unit Tests

```python
# tests/test_parser.py
class TestKiloCodeParser:
    def test_parse_valid_task_directory(self, tmp_path):
        # Create mock Kilo Code structure
        task_dir = tmp_path / "test-task-123"
        task_dir.mkdir()
        
        # Create metadata file
        metadata = {
            "files_in_context": [
                {"path": "src/auth.py", "record_state": "active"},
                {"path": "README.md", "record_state": "stale"}
            ]
        }
        (task_dir / "task_metadata.json").write_text(json.dumps(metadata))
        
        # Create conversation file
        conversation = [
            {"role": "user", "content": "How do I implement JWT?", "ts": 1234567890},
            {"role": "assistant", "content": "Here's how...", "ts": 1234567891}
        ]
        (task_dir / "api_conversation_history.json").write_text(json.dumps(conversation))
        
        # Parse
        parser = KiloCodeParser()
        session = parser.parse_task_directory(task_dir)
        
        assert session is not None
        assert session.task_id == "test-task-123"
        assert len(session.metadata.files_in_context) == 1  # Only active
        assert session.metadata.files_in_context[0] == "src/auth.py"
        assert len(session.conversation) == 2

# tests/test_watcher.py
class TestTranscriptWatcher:
    def test_scan_existing_sessions(self, tmp_path):
        # Create mock task directories
        (tmp_path / "task-1").mkdir()
        (tmp_path / "task-2").mkdir()
        
        watcher = TranscriptWatcher(tmp_path, lambda x, y: None)
        watcher._scan_existing_sessions()
        
        assert "task-1" in watcher.known_sessions
        assert "task-2" in watcher.known_sessions
```

### 4.2 Integration Tests

```python
# tests/test_integration.py
class TestEndToEndIndexing:
    def test_full_indexing_workflow(self, tmp_path):
        # Setup
        tasks_path = tmp_path / "tasks"
        db_path = tmp_path / "db"
        tasks_path.mkdir()
        db_path.mkdir()
        
        # Create test session
        task_dir = tasks_path / "test-session"
        task_dir.mkdir()
        
        metadata = {"files_in_context": []}
        (task_dir / "task_metadata.json").write_text(json.dumps(metadata))
        
        conversation = [
            {"role": "user", "content": "Implement authentication"},
            {"role": "assistant", "content": "Here's the code..."}
        ]
        (task_dir / "api_conversation_history.json").write_text(json.dumps(conversation))
        
        # Index
        db = ChromaDatabase(db_path)
        indexer = FullIndexer(db)
        result = indexer.index_all_sessions(tasks_path)
        
        assert result.indexed == 1
        assert db.get_session_count() > 0
        
        # Search
        engine = SemanticSearchEngine(db)
        results = engine.search("authentication")
        
        assert len(results) > 0
```

---

## 5. Success Criteria

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Index 100 sessions | <5 seconds | Time `smartfork index` command |
| CLI functional | All commands work | Manual testing of index, search, status |
| Parse Kilo Code format | 100% success | Test with real task directories |
| Search returns results | Relevant sessions | Manual evaluation of top-5 results |
| Offline operation | No network required | Disconnect and test |
| Database persistence | Survives restart | Index, restart, search |

---

## 6. Week-by-Week Breakdown

### Week 1

**Days 1-2: Project Setup**
- Create project structure
- Set up virtual environment
- Install dependencies (requirements.txt)
- Configure logging

**Days 3-4: Parser Implementation**
- Implement `KiloCodeParser`
- Test with real Kilo Code task directories
- Handle edge cases (missing files, malformed JSON)

**Days 5-7: Database and Indexer**
- Implement `ChromaDatabase`
- Implement `FullIndexer`
- Basic chunking strategy
- Test end-to-end indexing

### Week 2

**Days 8-10: Search and CLI**
- Implement `SemanticSearchEngine`
- Build CLI with Typer
- Create `index`, `search`, `status` commands
- Add Rich console output

**Days 11-12: Watcher**
- Implement `TranscriptWatcher`
- File system event handling
- Integration with indexer

**Days 13-14: Testing and Polish**
- Write unit tests
- Write integration tests
- Documentation (README)
- Bug fixes and optimization

---

## 7. Dependencies (requirements.txt)

```
# Core
chromadb>=0.4.18
sentence-transformers>=2.2.2
pydantic>=2.5.0
pydantic-settings>=2.1.0

# CLI
 typer>=0.9.0
rich>=13.7.0

# File watching
watchdog>=3.0.0

# Utilities
loguru>=0.7.0
python-dotenv>=1.0.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
```

---

## 8. Next Steps After Phase 0

After completing Phase 0:

1. **Phase 1**: Add hybrid search (BM25 + recency + path matching)
2. **Phase 2**: Fork.md generation and context diff
3. **Phase 3**: Pre-compaction hooks and deduplication
4. **Phase 4**: Advanced features (proactive suggestions, privacy vault)
