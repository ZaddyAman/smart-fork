"""Project name, domain, language, and session pattern extraction.

Derives structured metadata from file paths and tool call sequences.
All extraction is pure Python string parsing — zero LLM needed.
"""

import os
import re
from typing import List, Set


# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN PATTERNS — Maps file path fragments to semantic domains
# ═══════════════════════════════════════════════════════════════════════════════

DOMAIN_PATTERNS = {
    "rag":       ["rag/", "vector", "embed", "retriev", "chroma", "pinecone"],
    "auth":      ["auth", "login", "jwt", "token", "session", "password"],
    "frontend":  ["frontend/", "src/components", "src/pages", ".tsx", ".jsx", ".vue"],
    "backend":   ["backend/", "api/", "routes/", "controllers/", ".py", "fastapi", "django"],
    "database":  ["db/", "models.", "migrations/", "crud.", "schema."],
    "ingest":    ["ingest/", "scrape", "embed_", "load_", "process_"],
    "devops":    ["docker", "k8s", "ci/", ".yml", "deploy"],
    "testing":   ["test_", "_test.", "spec.", "fixtures/"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# LANGUAGE EXTENSION MAP
# ═══════════════════════════════════════════════════════════════════════════════

EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".r": "r",
    ".vue": "vue",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".sql": "sql",
    ".sh": "shell",
    ".ps1": "powershell",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".md": "markdown",
    ".dart": "dart",
    ".lua": "lua",
    ".zig": "zig",
}

# Directories that are NOT valid project names
GENERIC_DIR_NAMES = {"src", "app", "code", "projects", "workspace", "work", "dev", "home", ""}


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def derive_project_name(workspace_dir: str, file_paths: list[str] = None) -> str:
    """Derive project name from workspace directory path.
    
    Priority:
        1. Last segment of workspace directory path
        2. Most common top-level directory in file paths
        3. Fallback to 'unknown_project'
    
    Args:
        workspace_dir: Workspace directory path (e.g., "d:/Indian Legal Assistant")
        file_paths: Optional list of file paths from the session
    
    Returns:
        Human-readable project name
    """
    if workspace_dir:
        # Priority 1: Last segment of workspace directory
        name = os.path.basename(workspace_dir.rstrip("/\\"))
        if name and name.lower() not in GENERIC_DIR_NAMES:
            return name
    
    if file_paths:
        # Priority 2: Most common top-level directory in file paths
        top_dirs = []
        for fp in file_paths:
            parts = fp.replace("\\", "/").split("/")
            if len(parts) > 1 and parts[0]:
                top_dirs.append(parts[0])
        
        if top_dirs:
            # Find most common
            from collections import Counter
            most_common = Counter(top_dirs).most_common(1)
            if most_common:
                candidate = most_common[0][0]
                if candidate.lower() not in GENERIC_DIR_NAMES:
                    return candidate
    
    return "unknown_project"


def extract_domains(file_paths: list[str]) -> list[str]:
    """Extract semantic domains from file paths using pattern matching.
    
    Args:
        file_paths: List of file paths from the session
    
    Returns:
        List of domain strings (e.g., ["rag", "frontend", "auth"])
    """
    if not file_paths:
        return []
    
    domains: Set[str] = set()
    all_text = " ".join(file_paths).lower()
    
    for domain, patterns in DOMAIN_PATTERNS.items():
        if any(p in all_text for p in patterns):
            domains.add(domain)
    
    return sorted(domains)


def extract_languages(file_paths: list[str]) -> list[str]:
    """Extract programming languages from file extensions.
    
    Args:
        file_paths: List of file paths from the session
    
    Returns:
        Sorted list of unique language names
    """
    if not file_paths:
        return []
    
    languages: Set[str] = set()
    for fp in file_paths:
        ext = os.path.splitext(fp)[1].lower()
        if ext in EXTENSION_TO_LANGUAGE:
            languages.add(EXTENSION_TO_LANGUAGE[ext])
    
    return sorted(languages)


def extract_layers(file_paths: list[str]) -> list[str]:
    """Extract architectural layers (backend/frontend) from file paths.
    
    Args:
        file_paths: List of file paths from the session
    
    Returns:
        List of layer strings (e.g., ["backend", "frontend"])
    """
    if not file_paths:
        return []
    
    layers: Set[str] = set()
    all_text = " ".join(file_paths).lower()
    
    # Backend indicators
    backend_patterns = ["backend/", "api/", "routes/", "controllers/", "server/",
                       "main.py", "app.py", "manage.py", "wsgi", "asgi"]
    if any(p in all_text for p in backend_patterns):
        layers.add("backend")
    
    # Frontend indicators
    frontend_patterns = ["frontend/", "src/components", "src/pages", "src/views",
                        ".tsx", ".jsx", ".vue", "public/", "static/", "client/"]
    if any(p in all_text for p in frontend_patterns):
        layers.add("frontend")
    
    return sorted(layers)


def classify_session_pattern(tool_call_sequence: list[str], user_edit_count: int,
                              edit_count: int = 0) -> str:
    """Classify the session's work pattern from tool calls and edit counts.
    
    Possible patterns:
        - pure_review: Only reads, no edits
        - iterative_debugging: Alternating AI edit → user edit
        - investigation_then_implementation: Lots of reads then edits
        - refactoring: Many edits across many files
        - standard_implementation: Default
    
    Args:
        tool_call_sequence: List of tool call identifiers from conversation
        user_edit_count: Number of user-edited files
        edit_count: Number of AI-edited files
    
    Returns:
        Session pattern string
    """
    if edit_count == 0:
        return "pure_review"
    
    # Count reads vs edits in tool call sequence
    read_count = sum(1 for t in tool_call_sequence if "read" in t.lower())
    write_count = sum(1 for t in tool_call_sequence 
                      if any(w in t.lower() for w in ["write", "edit", "replace", "insert"]))
    
    # Check for alternating edit patterns (debugging)
    if user_edit_count > 2:
        return "iterative_debugging"
    
    # Investigation → implementation
    if read_count > write_count * 2 and write_count > 0:
        return "investigation_then_implementation"
    
    # Heavy refactoring
    if edit_count > 10:
        return "refactoring"
    
    return "standard_implementation"
