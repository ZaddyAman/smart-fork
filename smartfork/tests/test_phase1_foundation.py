"""Phase 1 verification tests for v2 foundation components.

Tests:
- project_extractor: project names, domains, languages, layers, patterns
- session_parser: structured 3-file parsing
- metadata_store: SQLite CRUD and filtering
- session_scanner: directory walking and change detection
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.smartfork.indexer.project_extractor import (
    derive_project_name, extract_domains, extract_languages,
    extract_layers, classify_session_pattern
)
from src.smartfork.indexer.session_parser import SessionParser
from src.smartfork.database.metadata_store import MetadataStore
from src.smartfork.database.models import SessionDocument


# ═══════════════════════════════════════════════════════════════════════════════
# PROJECT EXTRACTOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestProjectExtractor:
    
    def test_derive_project_name_from_workspace(self):
        assert derive_project_name("d:/Indian Legal Assistant") == "Indian Legal Assistant"
        assert derive_project_name("d:/SmartFork") == "SmartFork"
        assert derive_project_name("C:\\Users\\dev\\BharatLawAI") == "BharatLawAI"
    
    def test_derive_project_name_trailing_slash(self):
        assert derive_project_name("d:/MyProject/") == "MyProject"
        assert derive_project_name("d:/MyProject\\") == "MyProject"
    
    def test_derive_project_name_generic_dirs_rejected(self):
        # "src" is too generic, should fallback
        result = derive_project_name("d:/src", ["backend/main.py", "backend/models.py"])
        assert result != "src"
    
    def test_derive_project_name_fallback(self):
        assert derive_project_name("", []) == "unknown_project"
    
    def test_extract_domains(self):
        files = [
            "backend/rag/query_engine.py",
            "frontend/src/components/ChatPanel.tsx",
            "backend/api/auth.py",
        ]
        domains = extract_domains(files)
        assert "rag" in domains
        assert "frontend" in domains
        assert "auth" in domains
        assert "backend" in domains
    
    def test_extract_domains_empty(self):
        assert extract_domains([]) == []
    
    def test_extract_languages(self):
        files = ["main.py", "App.tsx", "index.html", "schema.sql"]
        langs = extract_languages(files)
        assert "python" in langs
        assert "typescript" in langs
        assert "html" in langs
        assert "sql" in langs
    
    def test_extract_languages_empty(self):
        assert extract_languages([]) == []
    
    def test_extract_layers(self):
        files = ["backend/main.py", "frontend/src/App.tsx"]
        layers = extract_layers(files)
        assert "backend" in layers
        assert "frontend" in layers
    
    def test_classify_pure_review(self):
        assert classify_session_pattern(["read_file", "read_file"], 0, 0) == "pure_review"
    
    def test_classify_debugging(self):
        pattern = classify_session_pattern(
            ["write_to_file", "read_file"], user_edit_count=3, edit_count=5
        )
        assert pattern == "iterative_debugging"
    
    def test_classify_refactoring(self):
        pattern = classify_session_pattern(
            ["write_to_file"] * 15, user_edit_count=0, edit_count=15
        )
        assert pattern == "refactoring"
    
    def test_classify_investigation(self):
        pattern = classify_session_pattern(
            ["read_file"] * 10 + ["write_to_file"] * 3,
            user_edit_count=0, edit_count=3
        )
        assert pattern == "investigation_then_implementation"


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION PARSER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSessionParser:
    
    @pytest.fixture
    def parser(self):
        return SessionParser()
    
    @pytest.fixture
    def sample_session_dir(self, tmp_path):
        """Create a realistic sample session directory."""
        session_dir = tmp_path / "1773250127795"
        session_dir.mkdir()
        
        # task_metadata.json
        metadata = {
            "files_in_context": [
                {
                    "path": "backend/rag/query_engine.py",
                    "record_state": "active",
                    "record_source": "roo_edited",
                    "roo_read_date": 1757164011601,
                    "roo_edit_date": 1757164262785,
                    "user_edit_date": None
                },
                {
                    "path": "frontend/src/ChatPanel.tsx",
                    "record_state": "stale",
                    "record_source": "roo_edited",
                    "roo_read_date": 1757164052942,
                    "roo_edit_date": 1757164262785,
                    "user_edit_date": 1757164261670
                },
                {
                    "path": "backend/main.py",
                    "record_state": "active",
                    "record_source": "read_tool",
                    "roo_read_date": 1757164011601,
                    "roo_edit_date": None,
                    "user_edit_date": None
                },
                {
                    "path": "PROJECT_HANDOFF.md",
                    "record_state": "active",
                    "record_source": "file_mentioned",
                    "roo_read_date": None,
                    "roo_edit_date": None,
                    "user_edit_date": None
                },
                {
                    "path": "backend/api/auth.py",
                    "record_state": "active",
                    "record_source": "user_edited",
                    "roo_read_date": 1757164011601,
                    "roo_edit_date": None,
                    "user_edit_date": 1757164300000
                },
            ]
        }
        (session_dir / "task_metadata.json").write_text(json.dumps(metadata), encoding='utf-8')
        
        # api_conversation_history.json
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "<task>\nreview and fix the auth module\n</task>\n\n<environment_details>\n# VSCode Open Tabs\nbackend/main.py\n# Current Workspace Directory\nd:/Indian Legal Assistant\n</environment_details>"}],
                "ts": 1757163992341
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "I'll review the auth module. The main concern is that JWT tokens need proper expiry handling. Let me look at the existing implementation to understand the current approach before making changes."}],
                "ts": 1757164001117
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<read_file>\n<path>backend/api/auth.py</path>\n</read_file>"}],
                "ts": 1757164010000
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "After reviewing the code, I've decided to use JWT with 1-hour expiry and refresh tokens stored in httpOnly cookies. This approach is more secure than session-based auth because it's stateless."}],
                "ts": 1757164800416
            },
        ]
        (session_dir / "api_conversation_history.json").write_text(json.dumps(conversation), encoding='utf-8')
        
        # ui_messages.json
        ui_messages = [
            {"ts": 1757163990226, "type": "say", "say": "text", "text": "review auth module"},
            {"ts": 1757163995042, "type": "say", "say": "reasoning", "text": "First, I need to understand the existing auth setup. The user wants me to review and fix it, so I should look at the JWT implementation and check for common security issues."},
            {"ts": 1757164005000, "type": "say", "say": "api_req_started", "text": "{\"model\": \"grok\"}"},
            {"ts": 1757164100000, "type": "say", "say": "reasoning", "text": "The JWT implementation has a bug: tokens don't expire. I need to add expiry and a refresh mechanism."},
        ]
        (session_dir / "ui_messages.json").write_text(json.dumps(ui_messages), encoding='utf-8')
        
        return session_dir
    
    def test_parse_session_basic(self, parser, sample_session_dir):
        doc = parser.parse_session(sample_session_dir)
        assert doc is not None
        assert doc.session_id == "1773250127795"
        assert doc.schema_version == 2
    
    def test_parse_task_raw(self, parser, sample_session_dir):
        doc = parser.parse_session(sample_session_dir)
        assert "review and fix the auth module" in doc.task_raw
    
    def test_parse_project_name(self, parser, sample_session_dir):
        doc = parser.parse_session(sample_session_dir)
        assert doc.project_name == "Indian Legal Assistant"
    
    def test_parse_workspace_dir(self, parser, sample_session_dir):
        doc = parser.parse_session(sample_session_dir)
        assert doc.project_root == "d:/Indian Legal Assistant"
    
    def test_parse_files_categorized(self, parser, sample_session_dir):
        doc = parser.parse_session(sample_session_dir)
        assert "backend/rag/query_engine.py" in doc.files_edited
        assert "backend/main.py" in doc.files_read
        assert "PROJECT_HANDOFF.md" in doc.files_mentioned
        assert doc.edit_count == 2  # Two roo_edited files
        assert doc.user_edit_count == 1  # One user_edited file
    
    def test_parse_final_files(self, parser, sample_session_dir):
        doc = parser.parse_session(sample_session_dir)
        # Only active state + has roo_edit_date
        assert "backend/rag/query_engine.py" in doc.final_files
    
    def test_parse_domains(self, parser, sample_session_dir):
        doc = parser.parse_session(sample_session_dir)
        assert "auth" in doc.domains
        assert "backend" in doc.domains
        assert "frontend" in doc.domains
    
    def test_parse_languages(self, parser, sample_session_dir):
        doc = parser.parse_session(sample_session_dir)
        assert "python" in doc.languages
        assert "typescript" in doc.languages
    
    def test_parse_reasoning_blocks(self, parser, sample_session_dir):
        doc = parser.parse_session(sample_session_dir)
        # Should have reasoning from both api_conversation and ui_messages
        assert len(doc.reasoning_docs) >= 2
        # Should contain decision reasoning
        found_jwt = any("JWT" in r for r in doc.reasoning_docs)
        assert found_jwt, "Should find JWT reasoning in reasoning_docs"
    
    def test_parse_timestamps(self, parser, sample_session_dir):
        doc = parser.parse_session(sample_session_dir)
        assert doc.session_start == 1757163992341
        assert doc.session_end == 1757164800416
        assert doc.duration_minutes > 0
    
    def test_parse_skips_tool_calls(self, parser, sample_session_dir):
        doc = parser.parse_session(sample_session_dir)
        # Reasoning docs should NOT contain read_file tool call
        for r in doc.reasoning_docs:
            assert "<read_file>" not in r
    
    def test_parse_nonexistent_dir(self, parser, tmp_path):
        result = parser.parse_session(tmp_path / "nonexistent")
        assert result is None
    
    def test_parse_empty_session(self, parser, tmp_path):
        session_dir = tmp_path / "empty_session"
        session_dir.mkdir()
        (session_dir / "api_conversation_history.json").write_text("[]", encoding='utf-8')
        
        doc = parser.parse_session(session_dir)
        assert doc is not None
        assert doc.task_raw == ""
        assert doc.project_name == "unknown_project"


# ═══════════════════════════════════════════════════════════════════════════════
# METADATA STORE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestMetadataStore:
    
    @pytest.fixture
    def store(self, tmp_path):
        db_path = tmp_path / "test_metadata.db"
        s = MetadataStore(db_path)
        yield s
        s.close()
    
    @pytest.fixture
    def sample_doc(self):
        return SessionDocument(
            session_id="test_session_001",
            project_name="BharatLawAI",
            project_root="d:/Indian Legal Assistant",
            session_start=1757163992341,
            session_end=1757164800416,
            duration_minutes=13.5,
            model_used="grok-code-fast-1",
            files_edited=["backend/api/auth.py", "backend/db/models.py"],
            files_read=["backend/main.py"],
            files_mentioned=["PROJECT_HANDOFF.md"],
            edit_count=2,
            user_edit_count=1,
            final_files=["backend/api/auth.py"],
            domains=["auth", "backend", "database"],
            languages=["python"],
            layers=["backend"],
            session_pattern="investigation_then_implementation",
            task_raw="implement JWT authentication for the API",
            reasoning_docs=["Decided to use JWT for stateless auth"],
            indexed_at=1757165000000,
            schema_version=2,
        )
    
    def test_upsert_and_get(self, store, sample_doc):
        store.upsert_session(sample_doc)
        retrieved = store.get_session("test_session_001")
        assert retrieved is not None
        assert retrieved.project_name == "BharatLawAI"
        assert retrieved.task_raw == "implement JWT authentication for the API"
        assert "backend/api/auth.py" in retrieved.files_edited
        assert "auth" in retrieved.domains
    
    def test_upsert_replaces(self, store, sample_doc):
        store.upsert_session(sample_doc)
        sample_doc.task_raw = "updated task"
        store.upsert_session(sample_doc)
        retrieved = store.get_session("test_session_001")
        assert retrieved.task_raw == "updated task"
    
    def test_get_nonexistent(self, store):
        assert store.get_session("nonexistent") is None
    
    def test_filter_by_project(self, store, sample_doc):
        store.upsert_session(sample_doc)
        # Add another session with different project
        other = sample_doc.model_copy()
        other.session_id = "test_session_002"
        other.project_name = "SmartFork"
        store.upsert_session(other)
        
        results = store.filter_sessions(project="BharatLawAI")
        assert len(results) == 1
        assert results[0] == "test_session_001"
    
    def test_filter_by_domain(self, store, sample_doc):
        store.upsert_session(sample_doc)
        results = store.filter_sessions(domains=["auth"])
        assert "test_session_001" in results
        
        results = store.filter_sessions(domains=["frontend"])
        assert "test_session_001" not in results
    
    def test_filter_by_time(self, store, sample_doc):
        store.upsert_session(sample_doc)
        # Filter for sessions after the start time
        results = store.filter_sessions(time_after=1757163992340)
        assert "test_session_001" in results
        
        # Filter for sessions after a later time
        results = store.filter_sessions(time_after=9999999999999)
        assert len(results) == 0
    
    def test_filter_by_file_hint(self, store, sample_doc):
        store.upsert_session(sample_doc)
        results = store.filter_sessions(file_hint="auth.py")
        assert "test_session_001" in results
        
        results = store.filter_sessions(file_hint="nonexistent.py")
        assert len(results) == 0
    
    def test_filter_combined(self, store, sample_doc):
        store.upsert_session(sample_doc)
        results = store.filter_sessions(project="BharatLawAI", domains=["auth"])
        assert "test_session_001" in results
    
    def test_get_session_count(self, store, sample_doc):
        assert store.get_session_count() == 0
        store.upsert_session(sample_doc)
        assert store.get_session_count() == 1
    
    def test_get_project_list(self, store, sample_doc):
        store.upsert_session(sample_doc)
        projects = store.get_project_list()
        assert len(projects) == 1
        assert projects[0]["project_name"] == "BharatLawAI"
        assert projects[0]["session_count"] == 1
    
    def test_delete_session(self, store, sample_doc):
        store.upsert_session(sample_doc)
        store.delete_session("test_session_001")
        assert store.get_session("test_session_001") is None
    
    def test_reset(self, store, sample_doc):
        store.upsert_session(sample_doc)
        store.reset()
        assert store.get_session_count() == 0
    
    def test_update_summary(self, store, sample_doc):
        store.upsert_session(sample_doc)
        store.update_summary("test_session_001", "This session implemented JWT auth for BharatLawAI.")
        retrieved = store.get_session("test_session_001")
        assert retrieved.summary_doc == "This session implemented JWT auth for BharatLawAI."
