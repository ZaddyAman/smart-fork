"""MCP (Model Context Protocol) server for SmartFork (v2).

Exposes SmartFork tools to Kilo Code for direct IDE integration:
- search: Search past sessions
- fork: Fork context from a session  
- detect-fork: Auto-detect relevant sessions
- status: Get index statistics

Delivery methods (fallback chain):
1. MCP response (primary) — direct injection into Kilo Code
2. File drop — writes to .smartfork/context.md
3. Clipboard — copies context to system clipboard

Start via CLI: smartfork mcp-server --port 8765
"""

import asyncio
import json
from pathlib import Path
from typing import Optional, List
from loguru import logger


class SmartForkMCPServer:
    """MCP server exposing SmartFork tools for Kilo Code integration.
    
    This is the primary delivery mechanism for SmartFork context.
    Instead of the user manually copying context, Kilo Code calls
    SmartFork tools directly through MCP.
    
    Tools exposed:
        smartfork/search: Search sessions with full decomposition + retrieval
        smartfork/fork: Get fork context for a specific session
        smartfork/detect-fork: Auto-detect relevant sessions for current task
        smartfork/status: Get SmartFork index statistics
    """
    
    def __init__(self, metadata_store=None, vector_index=None,
                 bm25_index=None, embedder=None, query_decomposer=None):
        """Initialize MCP server with service dependencies.
        
        All dependencies are optional for deferred initialization.
        The server will validate required services before handling requests.
        """
        self.metadata_store = metadata_store
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.embedder = embedder
        self.query_decomposer = query_decomposer
        
        # MCP tool definitions
        self.tools = {
            "smartfork/search": {
                "name": "smartfork/search",
                "description": "Search past coding sessions. Returns session cards with project, task, reasoning, and files.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language search query"},
                        "project": {"type": "string", "description": "Optional project name filter"},
                        "limit": {"type": "integer", "description": "Max results (default: 5)", "default": 5},
                    },
                    "required": ["query"],
                },
            },
            "smartfork/fork": {
                "name": "smartfork/fork",
                "description": "Get fork context from a specific session. Intent: continue (resume work), reference (reuse approach), debug (get error fix).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session ID to fork from"},
                        "intent": {"type": "string", "enum": ["continue", "reference", "debug"], "default": "continue"},
                    },
                    "required": ["session_id"],
                },
            },
            "smartfork/detect-fork": {
                "name": "smartfork/detect-fork",
                "description": "Auto-detect relevant past sessions for the current task context.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_description": {"type": "string", "description": "Current task or query"},
                        "files_open": {"type": "array", "items": {"type": "string"}, "description": "Currently open files"},
                    },
                    "required": ["task_description"],
                },
            },
            "smartfork/status": {
                "name": "smartfork/status",
                "description": "Get SmartFork index statistics — session count, projects, domains.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
        }
    
    async def handle_tool_call(self, tool_name: str, arguments: dict) -> dict:
        """Route MCP tool calls to the appropriate handler.
        
        Args:
            tool_name: Tool identifier (e.g., "smartfork/search")
            arguments: Tool input arguments
        
        Returns:
            MCP-formatted response dict
        """
        handlers = {
            "smartfork/search": self._handle_search,
            "smartfork/fork": self._handle_fork,
            "smartfork/detect-fork": self._handle_detect_fork,
            "smartfork/status": self._handle_status,
        }
        
        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}
        
        try:
            return await handler(arguments)
        except Exception as e:
            logger.error(f"MCP tool {tool_name} failed: {e}")
            return {"error": str(e)}
    
    async def _handle_search(self, args: dict) -> dict:
        """Handle search tool call."""
        query = args.get("query", "")
        project = args.get("project")
        limit = args.get("limit", 5)
        
        if not self.metadata_store:
            return {"error": "SmartFork index not initialized"}
        
        # Step 1: Decompose query
        decomposition = None
        if self.query_decomposer:
            decomposition = self.query_decomposer.decompose(query)
        
        # Step 2: Metadata filter (Signal A)
        candidates = self.metadata_store.filter_sessions(
            project=project or (decomposition.project if decomposition else None),
        )
        
        if not candidates:
            return {"results": [], "message": "No matching sessions found"}
        
        # Step 3: Get session details for results
        results = []
        for sid in candidates[:limit]:
            doc = self.metadata_store.get_session(sid)
            if doc:
                results.append({
                    "session_id": doc.session_id,
                    "project": doc.project_name,
                    "task": doc.task_raw[:100] if doc.task_raw else "",
                    "files_edited": doc.files_edited[:5],
                    "domains": doc.domains,
                    "duration_minutes": doc.duration_minutes,
                })
        
        return {"results": results, "total": len(candidates)}
    
    async def _handle_fork(self, args: dict) -> dict:
        """Handle fork tool call."""
        session_id = args.get("session_id", "")
        intent = args.get("intent", "continue")
        
        if not self.metadata_store:
            return {"error": "SmartFork index not initialized"}
        
        doc = self.metadata_store.get_session(session_id)
        if not doc:
            return {"error": f"Session {session_id} not found"}
        
        from ..fork.fork_assembler import assemble_fork_context
        context = assemble_fork_context(doc, intent)
        
        return {
            "context": context,
            "session_id": session_id,
            "intent": intent,
            "project": doc.project_name,
        }
    
    async def _handle_detect_fork(self, args: dict) -> dict:
        """Handle auto-detect fork tool call."""
        task_description = args.get("task_description", "")
        files_open = args.get("files_open", [])
        
        if not self.metadata_store:
            return {"error": "SmartFork index not initialized"}
        
        # Build search hints from open files
        file_hint = None
        if files_open:
            # Use the most specific open file
            file_hint = files_open[0].split("/")[-1] if "/" in files_open[0] else files_open[0]
        
        candidates = self.metadata_store.filter_sessions(
            file_hint=file_hint,
            limit=3,
        )
        
        results = []
        for sid in candidates:
            doc = self.metadata_store.get_session(sid)
            if doc:
                results.append({
                    "session_id": doc.session_id,
                    "project": doc.project_name,
                    "task": doc.task_raw[:80] if doc.task_raw else "",
                    "relevance": "file_match" if file_hint else "recent",
                })
        
        return {"suggestions": results}
    
    async def _handle_status(self, args: dict) -> dict:
        """Handle status tool call."""
        if not self.metadata_store:
            return {"error": "SmartFork index not initialized"}
        
        session_count = self.metadata_store.get_session_count()
        projects = self.metadata_store.get_project_list()
        domains = self.metadata_store.get_domain_breakdown()
        
        return {
            "indexed_sessions": session_count,
            "projects": projects[:10],
            "domains": domains,
            "schema_version": 2,
        }
    
    def get_tool_definitions(self) -> list:
        """Get MCP-formatted tool definitions for registration.
        
        Returns:
            List of tool definition dicts for MCP server registration
        """
        return list(self.tools.values())


# ═══════════════════════════════════════════════════════════════════════════════
# FILE DROP DELIVERY (FALLBACK)
# ═══════════════════════════════════════════════════════════════════════════════


def file_drop_context(context: str, drop_path: Path = None) -> Path:
    """Write fork context to a file for manual inclusion.
    
    Fallback delivery method when MCP is not available.
    
    Args:
        context: Fork context markdown string
        drop_path: Target file path (default: .smartfork/context.md)
    
    Returns:
        Path to the written file
    """
    if drop_path is None:
        drop_path = Path.home() / ".smartfork" / "context.md"
    
    drop_path.parent.mkdir(parents=True, exist_ok=True)
    drop_path.write_text(context, encoding="utf-8")
    
    logger.info(f"Context dropped to {drop_path}")
    return drop_path


def clipboard_context(context: str) -> bool:
    """Copy fork context to system clipboard.
    
    Fallback delivery method for copy-paste workflow.
    
    Args:
        context: Fork context string
    
    Returns:
        True if successful, False if pyperclip unavailable
    """
    try:
        import pyperclip
        pyperclip.copy(context)
        logger.info("Context copied to clipboard")
        return True
    except ImportError:
        logger.warning("pyperclip not installed — clipboard copy unavailable")
        return False
    except Exception as e:
        logger.warning(f"Clipboard copy failed: {e}")
        return False
