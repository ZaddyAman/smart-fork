"""Data models for SmartFork.

Contains both v1 (original) and v2 (architecture upgrade) data models.
v1 models are preserved for backward compatibility.
v2 models (prefixed with section comments) support the new structured
session intelligence pipeline.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class TaskMetadata(BaseModel):
    """Metadata from task_metadata.json."""
    files_in_context: List[str] = []
    raw_data: Dict[str, Any] = Field(default_factory=dict)


class ConversationMessage(BaseModel):
    """A message from api_conversation_history.json."""
    role: str
    content: str
    timestamp: Optional[int] = None
    type: Optional[str] = None


class UIMessage(BaseModel):
    """A message from ui_messages.json."""
    message_type: Optional[str] = Field(None, alias="type")
    say: Optional[str] = None
    ask: Optional[str] = None
    text: Optional[str] = None
    ts: Optional[int] = None
    
    class Config:
        populate_by_name = True


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
    
    def get_code_blocks(self) -> List[str]:
        """Extract code blocks from conversation."""
        import re
        code_blocks = []
        full_text = self.get_full_text()
        
        # Match code blocks
        pattern = r'```(?:\w+)?\n(.*?)\n```'
        matches = re.findall(pattern, full_text, re.DOTALL)
        code_blocks.extend(matches)
        
        return code_blocks
    
    def get_last_timestamp(self) -> Optional[int]:
        """Get the last timestamp from the conversation."""
        for msg in reversed(self.conversation):
            if msg.timestamp:
                return msg.timestamp
        return None


class ChunkMetadata(BaseModel):
    """Metadata for a conversation chunk."""
    session_id: str
    task_id: str
    chunk_index: int
    timestamp: Optional[str] = None
    files_in_context: List[str] = []
    message_type: str = "mixed"  # "user", "assistant", "tool", "mixed"
    technologies: List[str] = []
    last_active: Optional[str] = None  # ISO timestamp for recency scoring
    session_title: Optional[str] = None  # Human-readable session title


class Chunk(BaseModel):
    """Represents a chunk of conversation."""
    id: str
    content: str
    embedding: List[float] = Field(default_factory=list)
    metadata: ChunkMetadata


class SearchResult(BaseModel):
    """Result from a search query."""
    session_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class HybridResult(BaseModel):
    """Result from hybrid search with score breakdown."""
    session_id: str
    score: float
    breakdown: Dict[str, float]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "score": self.score,
            "breakdown": self.breakdown,
            "metadata": self.metadata
        }


class IndexingResult(BaseModel):
    """Result of indexing operation."""
    indexed: int = 0
    failed: int = 0
    chunks_created: int = 0


class SessionMetadata(BaseModel):
    """Enriched session metadata."""
    session_id: str
    technologies: List[str] = []
    files_in_context: List[str] = []
    code_languages: List[str] = []
    estimated_tokens: int = 0
    has_errors: bool = False
    has_tests: bool = False
    last_active: Optional[datetime] = None
    session_title: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# v2 DATA MODELS — Structured Session Intelligence Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


class ForkIntent(str, Enum):
    """How the user wants to use forked context.
    
    CONTINUE: Pick up exactly where the last session left off.
    REFERENCE: Reuse the approach/decisions in new work.
    DEBUG: Hit the same problem again, need the fix.
    """
    CONTINUE = "continue"
    REFERENCE = "reference"
    DEBUG = "debug"
    SYNTHESIZE = "synthesize"


class TaskMetadataV2(BaseModel):
    """Structured metadata from task_metadata.json (v2).
    
    Unlike v1 TaskMetadata which only kept 'active' files,
    v2 categorizes ALL files by their record_source and record_state.
    """
    files_edited: List[str] = []       # record_source = "roo_edited"
    files_read: List[str] = []         # record_source = "read_tool"
    files_mentioned: List[str] = []    # record_source = "file_mentioned"
    files_user_edited: List[str] = []  # record_source = "user_edited"
    edit_count: int = 0                # count of roo_edited entries
    user_edit_count: int = 0           # count of user_edited entries
    final_files: List[str] = []        # record_state = "active" AND roo_edit_date not null
    raw_data: Dict[str, Any] = Field(default_factory=dict)


class ConversationDataV2(BaseModel):
    """Extracted signals from api_conversation_history.json (v2).
    
    v2 extracts specific signals from specific locations instead of
    concatenating all message text. It SKIPS <file_content> tags and
    <environment_details> blocks.
    """
    task_raw: str = ""                 # Text inside <task> tag in first user turn
    workspace_dir: str = ""            # From <environment_details> → Current Workspace Directory
    session_start: int = 0             # ts of first entry (milliseconds)
    session_end: int = 0               # ts of last entry (milliseconds)
    duration_minutes: float = 0.0      # computed from start/end
    model_used: Optional[str] = None   # From <environment_details> → Current Mode
    open_tabs: List[str] = []          # From <environment_details> → VSCode Open Tabs
    reasoning_turns: List[str] = []    # Assistant turns that are NOT tool calls
    tool_call_sequence: List[str] = [] # For session pattern classification


class SessionDocument(BaseModel):
    """Complete structured representation of a Kilo Code session (v2).
    
    This is the primary data object that flows through the entire v2 pipeline:
    Parser → MetadataStore (SQLite) → Embedder → VectorIndex → Search → ResultCard → Fork
    
    All fields are populated during indexing. No LLM is needed for any field
    except summary_doc (Phase 3) and propositions (Phase 3, optional).
    """
    session_id: str
    project_name: str = "unknown_project"
    project_root: str = ""
    session_start: int = 0             # Unix timestamp in ms
    session_end: int = 0               # Unix timestamp in ms
    duration_minutes: float = 0.0
    model_used: Optional[str] = None
    
    # File signals (from TaskMetadataV2)
    files_edited: List[str] = []
    files_read: List[str] = []
    files_mentioned: List[str] = []
    edit_count: int = 0
    user_edit_count: int = 0
    final_files: List[str] = []
    
    # Derived from file paths (pure string parsing, zero LLM)
    domains: List[str] = []            # ["rag", "auth", "frontend", ...]
    languages: List[str] = []          # ["python", "typescript", ...]
    layers: List[str] = []             # ["backend", "frontend"]
    session_pattern: str = "standard_implementation"
    
    # Embeddable text content
    task_raw: str = ""                 # Raw task description
    task_doc: str = ""                 # task_raw with contextual prefix (for embedding)
    summary_doc: str = ""              # LLM-generated 3-sentence summary (Phase 3)
    reasoning_docs: List[str] = []     # Contextually chunked reasoning blocks
    propositions: List[str] = []       # Atomic fact statements (Phase 3, optional)
    
    # Index management
    indexed_at: int = 0                # When this session was last indexed
    schema_version: int = 2
    
    # RAPTOR clustering (Phase 9)
    cluster_id: Optional[str] = None   # UUID for cross-session topic clustering
    
    # Supersession detection (v2.1)
    resolution_status: str = "unknown"  # "solved", "partial", "ongoing", "unknown"
    had_errors: int = 0                 # Count of error signals in task_raw
    supersedes_ids: List[str] = []      # IDs of sessions this one corrects


class QueryDecomposition(BaseModel):
    """Structured output from LLM-powered query decomposition (v2).
    
    One LLM call per query produces this object. All downstream retrieval
    stages (metadata filter, BM25, vector search, re-ranking) use it.
    
    Intent classes:
        decision_hunting:       "why", "decided", "chose", "approach"
        implementation_lookup:  "how did I", "code for", "implement"
        error_recall:           "bug", "error", "fix", "broke"
        file_lookup:            specific filename mentioned
        temporal_lookup:        "last week", "yesterday", "3 days ago"
        pattern_hunting:        "all sessions", "every time", "whenever"
        vague_memory:           short queries, no clear topic
    """
    intent: str = "vague_memory"
    topic: Optional[str] = None
    project: Optional[str] = None
    file_hint: Optional[str] = None
    time_hint: Optional[str] = None    # "last_week", "yesterday", "3_days_ago", "last_month"
    tech_terms: List[str] = []
    is_temporal_only: bool = False


class ResultCard(BaseModel):
    """Rich result card for display in CLI (v2).
    
    Each card is 5 lines of information maximum. Zero LLM calls needed
    to assemble — all data pulled from the existing index.
    
    Format:
        📁 Project — Task Title
        🕐 3 days ago (47 min) | ⚡ 94% match
        "Snippet from best matching reasoning/summary chunk..."
        Why: Contains auth decision | Files: auth.py, models.py
        [1] Fork  [2] Preview  [3] Skip
    """
    session_id: str
    project_name: str
    task_short: str                    # task_raw truncated to 50 chars
    relative_time: str                 # "3 days ago"
    duration_minutes: float
    match_score: float                 # 0.0 - 1.0
    snippet: str                       # Best matching reasoning/summary chunk
    why_matched: str                   # Rule-based explanation
    files_changed: List[str] = []
    matched_doc_type: str = "summary_doc"  # Which doc type matched best


class VectorResult(BaseModel):
    """Result from a vector search on a specific collection (v2)."""
    session_id: str
    doc_type: str                      # "task_doc", "summary_doc", "reasoning_doc"
    content: str
    score: float                       # Cosine similarity
    chunk_index: int = 0               # Index within the session's docs of this type
    parent_id: Optional[str] = None    # UUID for joining full parent text from SQLite

