"""SQLite metadata store for structured session metadata (v2).

Provides hard filtering before vector search begins — this is what makes
SmartFork fast at scale. Signal A (metadata filter) eliminates 80-90% of
the index before any embedding computation.

Schema stores all structured fields from SessionDocument as queryable
columns with JSON arrays for list fields.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from loguru import logger

from .models import SessionDocument


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id       TEXT PRIMARY KEY,
    project_name     TEXT NOT NULL,
    project_root     TEXT,
    session_start    INTEGER,
    session_end      INTEGER,
    duration_minutes REAL,
    model_used       TEXT,
    
    -- File signals (stored as JSON arrays)
    files_edited     TEXT DEFAULT '[]',
    files_read       TEXT DEFAULT '[]',
    files_mentioned  TEXT DEFAULT '[]',
    edit_count       INTEGER DEFAULT 0,
    user_edit_count  INTEGER DEFAULT 0,
    final_files      TEXT DEFAULT '[]',
    
    -- Derived fields (stored as JSON arrays)  
    domains          TEXT DEFAULT '[]',
    languages        TEXT DEFAULT '[]',
    layers           TEXT DEFAULT '[]',
    session_pattern  TEXT DEFAULT 'standard_implementation',
    
    -- Text content for BM25
    task_raw         TEXT DEFAULT '',
    
    -- LLM-generated (populated in Phase 3)
    summary_doc      TEXT DEFAULT '',
    
    -- Reasoning docs (JSON array of strings)
    reasoning_docs   TEXT DEFAULT '[]',
    
    -- Cluster ID for RAPTOR Cross-Session Retrieval (Phase 9)
    cluster_id       TEXT,
    
    -- Index management
    indexed_at       INTEGER,
    schema_version   INTEGER DEFAULT 2
);

CREATE INDEX IF NOT EXISTS idx_project ON sessions(project_name);
CREATE INDEX IF NOT EXISTS idx_start ON sessions(session_start);
CREATE INDEX IF NOT EXISTS idx_pattern ON sessions(session_pattern);
CREATE INDEX IF NOT EXISTS idx_indexed_at ON sessions(indexed_at);

CREATE TABLE IF NOT EXISTS parent_chunks (
    parent_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    full_raw_text TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_parent_session ON parent_chunks(session_id);
"""


class MetadataStore:
    """SQLite-backed metadata store for v2 structured session data.
    
    This is the foundation for Signal A (metadata filtering) in the
    3-stage retrieval pipeline. It enables exact-match queries on
    structured fields before any vector computation begins.
    
    Usage:
        store = MetadataStore(Path("~/.smartfork/metadata.db"))
        store.upsert_session(session_doc)
        candidates = store.filter_sessions(project="BharatLawAI", domains=["auth"])
    """
    
    def __init__(self, db_path: Path):
        """Initialize SQLite connection and create tables if needed.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read perf
        self.conn.execute("PRAGMA synchronous=NORMAL")
        
        # Create schema
        self.conn.executescript(SCHEMA_SQL)
        try:
            self.conn.execute("ALTER TABLE sessions ADD COLUMN cluster_id TEXT")
        except sqlite3.OperationalError:
            pass
        self.conn.commit()
        
        logger.debug(f"MetadataStore initialized at {self.db_path}")
    
    def upsert_session(self, doc: SessionDocument) -> None:
        """Insert or update a session document.
        
        Uses SQLite's INSERT OR REPLACE to handle both new sessions
        and re-indexed sessions.
        
        Args:
            doc: SessionDocument with all fields populated
        """
        self.conn.execute("""
            INSERT OR REPLACE INTO sessions (
                session_id, project_name, project_root,
                session_start, session_end, duration_minutes, model_used,
                files_edited, files_read, files_mentioned,
                edit_count, user_edit_count, final_files,
                domains, languages, layers, session_pattern,
                task_raw, summary_doc, reasoning_docs,
                cluster_id, indexed_at, schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            doc.session_id,
            doc.project_name,
            doc.project_root,
            doc.session_start,
            doc.session_end,
            doc.duration_minutes,
            doc.model_used,
            json.dumps(doc.files_edited),
            json.dumps(doc.files_read),
            json.dumps(doc.files_mentioned),
            doc.edit_count,
            doc.user_edit_count,
            json.dumps(doc.final_files),
            json.dumps(doc.domains),
            json.dumps(doc.languages),
            json.dumps(doc.layers),
            doc.session_pattern,
            doc.task_raw,
            doc.summary_doc,
            json.dumps(doc.reasoning_docs),
            doc.cluster_id,
            doc.indexed_at or int(time.time() * 1000),
            doc.schema_version,
        ))
        self.conn.commit()
    
    def get_session(self, session_id: str) -> Optional[SessionDocument]:
        """Retrieve a session document by ID.
        
        Args:
            session_id: Session ID to look up
        
        Returns:
            SessionDocument or None if not found
        """
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        
        if not row:
            return None
        
        return self._row_to_document(row)
    
    def filter_sessions(self, project: str = None, domains: list = None,
                        time_after: int = None, file_hint: str = None,
                        pattern: str = None, limit: int = 100) -> List[str]:
        """Filter sessions by structured metadata fields.
        
        This is Signal A in the retrieval pipeline. It runs FIRST and
        gates all downstream signals (BM25, vector search).
        
        Args:
            project: Exact project name match
            domains: List of domains — session must match at least one
            time_after: Unix timestamp (ms) — sessions after this time only
            file_hint: Filename to search for in files_edited and files_read
            pattern: Session pattern to match
            limit: Maximum number of session IDs to return
        
        Returns:
            List of matching session_id strings
        """
        query = "SELECT session_id FROM sessions WHERE 1=1"
        params: List[Any] = []
        
        if project:
            query += " AND LOWER(project_name) LIKE LOWER(?)"
            params.append(f"%{project}%")
        
        if domains:
            domain_clauses = []
            for d in domains:
                domain_clauses.append("domains LIKE ?")
                params.append(f'%"{d}"%')
            query += f" AND ({' OR '.join(domain_clauses)})"
        
        if time_after:
            query += " AND session_start >= ?"
            params.append(time_after)
        
        if file_hint:
            query += " AND (files_edited LIKE ? OR files_read LIKE ? OR files_mentioned LIKE ?)"
            params.append(f'%{file_hint}%')
            params.append(f'%{file_hint}%')
            params.append(f'%{file_hint}%')
        
        if pattern:
            query += " AND session_pattern = ?"
            params.append(pattern)
        
        query += " ORDER BY session_start DESC LIMIT ?"
        params.append(limit)
        
        rows = self.conn.execute(query, params).fetchall()
        return [row['session_id'] for row in rows]
    
    def get_all_session_ids(self) -> List[str]:
        """Get all indexed session IDs.
        
        Returns:
            List of all session_id strings
        """
        rows = self.conn.execute(
            "SELECT session_id FROM sessions ORDER BY session_start DESC"
        ).fetchall()
        return [row['session_id'] for row in rows]
    
    def get_all_sessions(self, limit: int = 1000) -> List[SessionDocument]:
        """Get all session documents.
        
        Args:
            limit: Maximum number of sessions to return
        
        Returns:
            List of SessionDocument objects
        """
        rows = self.conn.execute(
            "SELECT * FROM sessions ORDER BY session_start DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [self._row_to_document(row) for row in rows]
    
    def get_session_count(self) -> int:
        """Get total number of indexed sessions.
        
        Returns:
            Count of sessions in the store
        """
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM sessions").fetchone()
        return row['cnt'] if row else 0
    
    def get_project_list(self) -> List[Dict[str, Any]]:
        """Get list of unique projects with session counts.
        
        Returns:
            List of dicts: [{"project_name": "X", "session_count": N}, ...]
        """
        rows = self.conn.execute("""
            SELECT project_name, COUNT(*) as session_count
            FROM sessions
            GROUP BY project_name
            ORDER BY session_count DESC
        """).fetchall()
        return [{"project_name": row['project_name'], "session_count": row['session_count']}
                for row in rows]
    
    def get_domain_breakdown(self) -> Dict[str, int]:
        """Get breakdown of domains across all sessions.
        
        Returns:
            Dict mapping domain name to count of sessions containing it
        """
        rows = self.conn.execute("SELECT domains FROM sessions").fetchall()
        domain_counts: Dict[str, int] = {}
        for row in rows:
            try:
                domains = json.loads(row['domains'])
                for d in domains:
                    domain_counts[d] = domain_counts.get(d, 0) + 1
            except (json.JSONDecodeError, TypeError):
                continue
        return dict(sorted(domain_counts.items(), key=lambda x: x[1], reverse=True))
    
    def update_summary(self, session_id: str, summary_doc: str) -> None:
        """Update the LLM-generated summary for a session.
        
        Called during Phase 3 when session summaries are generated.
        
        Args:
            session_id: Session to update
            summary_doc: 3-sentence LLM-generated summary
        """
        self.conn.execute(
            "UPDATE sessions SET summary_doc = ? WHERE session_id = ?",
            (summary_doc, session_id)
        )
        self.conn.commit()
    
    def get_project_list(self) -> List[Dict[str, Any]]:
        """Get list of distinct project names with counts.
        
        Returns:
            List of dicts with 'project_name' and 'count' keys
        """
        rows = self.conn.execute(
            "SELECT project_name, COUNT(*) as count FROM sessions "
            "GROUP BY project_name ORDER BY count DESC"
        ).fetchall()
        return [{"project_name": r["project_name"], "session_count": r["count"]} for r in rows]

    def insert_parent_chunk(self, parent_id: str, session_id: str, chunk_index: int, full_raw_text: str) -> None:
        """Insert a parent chunk into the store.
        
        Args:
            parent_id: Unique UUID for this parent reasoning block
            session_id: The session this block belongs to
            chunk_index: Chronological index of this block in the session
            full_raw_text: The entire unprocessed reasoning text
        """
        self.conn.execute("""
            INSERT OR REPLACE INTO parent_chunks (
                parent_id, session_id, chunk_index, full_raw_text
            ) VALUES (?, ?, ?, ?)
        """, (parent_id, session_id, chunk_index, full_raw_text))
        self.conn.commit()

    def get_parent_chunk(self, parent_id: str) -> Optional[str]:
        """Retrieve the full raw text of a parent chunk by its UUID.
        
        Args:
            parent_id: The UUID stored in ChromaDB metadata
            
        Returns:
            The full raw text if found, else None
        """
        row = self.conn.execute(
            "SELECT full_raw_text FROM parent_chunks WHERE parent_id = ?", (parent_id,)
        ).fetchone()
        if row:
            return row['full_raw_text']
        return None
    
    def delete_session(self, session_id: str) -> None:
        """Delete a session from the store.
        
        Args:
            session_id: Session ID to delete
        """
        self.conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        self.conn.commit()
    
    def reset(self) -> None:
        """Clear all data from the store. Use with caution."""
        self.conn.execute("DELETE FROM sessions")
        self.conn.commit()
        logger.warning("MetadataStore reset — all session data deleted")
    
    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
    
    def _row_to_document(self, row: sqlite3.Row) -> SessionDocument:
        """Convert a SQLite row back to a SessionDocument.
        
        Args:
            row: sqlite3.Row object from a query
        
        Returns:
            SessionDocument with all fields populated
        """
        return SessionDocument(
            session_id=row['session_id'],
            project_name=row['project_name'],
            project_root=row['project_root'] or "",
            session_start=row['session_start'] or 0,
            session_end=row['session_end'] or 0,
            duration_minutes=row['duration_minutes'] or 0.0,
            model_used=row['model_used'],
            files_edited=json.loads(row['files_edited'] or '[]'),
            files_read=json.loads(row['files_read'] or '[]'),
            files_mentioned=json.loads(row['files_mentioned'] or '[]'),
            edit_count=row['edit_count'] or 0,
            user_edit_count=row['user_edit_count'] or 0,
            final_files=json.loads(row['final_files'] or '[]'),
            domains=json.loads(row['domains'] or '[]'),
            languages=json.loads(row['languages'] or '[]'),
            layers=json.loads(row['layers'] or '[]'),
            session_pattern=row['session_pattern'] or 'standard_implementation',
            task_raw=row['task_raw'] or '',
            summary_doc=row['summary_doc'] or '',
            reasoning_docs=json.loads(row['reasoning_docs'] or '[]'),
            cluster_id=row['cluster_id'] if 'cluster_id' in row.keys() else None,
            indexed_at=row['indexed_at'] or 0,
            schema_version=row['schema_version'] or 2,
        )
