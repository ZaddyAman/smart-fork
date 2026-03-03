"""Data models for SmartFork."""

from typing import List, Dict, Any, Optional
from datetime import datetime
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
