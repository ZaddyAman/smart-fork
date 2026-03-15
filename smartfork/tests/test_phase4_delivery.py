"""Phase 4 verification tests — result cards, fork assembler, MCP server.

All tests run without external dependencies (no Ollama, no MCP SDK).
"""

import asyncio
import pytest
import tempfile
from pathlib import Path

from src.smartfork.database.models import SessionDocument, ForkIntent, ResultCard
from src.smartfork.ui.result_card import build_result_card, _format_relative_time
from src.smartfork.fork.fork_assembler import ForkAssembler, assemble_fork_context
from src.smartfork.mcp.mcp_server import SmartForkMCPServer, file_drop_context
from src.smartfork.database.metadata_store import MetadataStore


@pytest.fixture
def sample_doc():
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
        domains=["auth", "backend"],
        languages=["python"],
        layers=["backend"],
        session_pattern="investigation_then_implementation",
        task_raw="implement JWT authentication for the API with refresh tokens and httponly cookies",
        summary_doc="Implemented JWT auth for BharatLawAI. Chose stateless tokens with refresh mechanism. Modified auth.py and models.py.",
        reasoning_docs=[
            "Decided to use JWT for stateless auth because ChromaDB lacks session support. This approach is more scalable.",
            "Rejected session-based auth because it requires server-side state. JWT with httpOnly cookies is the industry standard.",
            "Found a bug where token expiry was not being checked. Fixed by adding middleware validation.",
        ],
        edit_count=2,
        user_edit_count=0,
        schema_version=2,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT CARD TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestResultCard:
    
    def test_build_result_card(self, sample_doc):
        card = build_result_card(sample_doc, match_score=0.94, snippet="JWT auth decision")
        assert card.project_name == "BharatLawAI"
        assert card.match_score == 0.94
        assert "JWT" in card.snippet
        assert len(card.task_short) <= 55  # 50 + "..."
    
    def test_card_has_files(self, sample_doc):
        card = build_result_card(sample_doc, match_score=0.8)
        assert "auth.py" in card.files_changed
    
    def test_relative_time_days(self):
        import time
        # 3 days ago
        ts_ms = int((time.time() - 86400 * 3) * 1000)
        result = _format_relative_time(ts_ms)
        assert "3 day" in result
    
    def test_relative_time_hours(self):
        import time
        ts_ms = int((time.time() - 7200) * 1000)  # 2 hours ago
        result = _format_relative_time(ts_ms)
        assert "hour" in result
    
    def test_relative_time_zero(self):
        result = _format_relative_time(0)
        assert "unknown" in result


# ═══════════════════════════════════════════════════════════════════════════════
# FORK ASSEMBLER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestForkAssembler:
    
    @pytest.fixture
    def assembler(self):
        return ForkAssembler()
    
    def test_continue_intent(self, assembler, sample_doc):
        context = assembler.assemble(sample_doc, ForkIntent.CONTINUE)
        assert "Continue" in context
        assert "BharatLawAI" in context
        assert "JWT" in context
        # CONTINUE includes reasoning trail
        assert "Reasoning Trail" in context or "Step 1:" in context
    
    def test_reference_intent(self, assembler, sample_doc):
        context = assembler.assemble(sample_doc, ForkIntent.REFERENCE)
        assert "Reference" in context
        assert "BharatLawAI" in context
        # REFERENCE includes key decisions
        assert "Decisions" in context or "Summary" in context
    
    def test_debug_intent(self, assembler, sample_doc):
        context = assembler.assemble(sample_doc, ForkIntent.DEBUG)
        assert "Debug" in context
        assert "BharatLawAI" in context
        # DEBUG should include error context
        assert "Error" in context or "bug" in context.lower()
    
    def test_convenience_function(self, sample_doc):
        context = assemble_fork_context(sample_doc, "reference")
        assert "BharatLawAI" in context
    
    def test_context_not_too_long(self, assembler, sample_doc):
        for intent in ForkIntent:
            context = assembler.assemble(sample_doc, intent)
            estimated_tokens = len(context.split()) * 1.3
            assert estimated_tokens < 700, f"{intent.value} exceeded token budget: ~{int(estimated_tokens)}"


# ═══════════════════════════════════════════════════════════════════════════════
# MCP SERVER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestMCPServer:
    
    @pytest.fixture
    def store_with_data(self, tmp_path, sample_doc):
        store = MetadataStore(tmp_path / "test.db")
        store.upsert_session(sample_doc)
        yield store
        store.close()
    
    @pytest.fixture
    def server(self, store_with_data):
        return SmartForkMCPServer(metadata_store=store_with_data)
    
    def test_tool_definitions(self, server):
        tools = server.get_tool_definitions()
        assert len(tools) == 4
        tool_names = [t["name"] for t in tools]
        assert "smartfork/search" in tool_names
        assert "smartfork/fork" in tool_names
    
    def test_status_handler(self, server):
        result = asyncio.run(server.handle_tool_call("smartfork/status", {}))
        assert result["indexed_sessions"] == 1
        assert result["schema_version"] == 2
    
    def test_search_handler(self, server):
        result = asyncio.run(server.handle_tool_call(
            "smartfork/search", {"query": "JWT auth"}
        ))
        assert "results" in result
        assert len(result["results"]) >= 1
    
    def test_fork_handler(self, server):
        result = asyncio.run(server.handle_tool_call(
            "smartfork/fork", {"session_id": "test_session_001", "intent": "reference"}
        ))
        assert "context" in result
        assert "BharatLawAI" in result["context"]
    
    def test_fork_nonexistent(self, server):
        result = asyncio.run(server.handle_tool_call(
            "smartfork/fork", {"session_id": "nonexistent"}
        ))
        assert "error" in result
    
    def test_unknown_tool(self, server):
        result = asyncio.run(server.handle_tool_call("unknown/tool", {}))
        assert "error" in result


# ═══════════════════════════════════════════════════════════════════════════════
# FILE DROP TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFileDrop:
    
    def test_file_drop(self, tmp_path):
        drop_path = tmp_path / "context.md"
        result = file_drop_context("# Test Context\nHello world", drop_path)
        assert result == drop_path
        assert drop_path.exists()
        assert "Test Context" in drop_path.read_text(encoding="utf-8")
