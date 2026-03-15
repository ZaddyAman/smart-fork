"""Interactive shell for SmartFork CLI.

Provides a persistent REPL-like environment for running SmartFork commands
without typing the 'smartfork' prefix each time.
"""

import cmd
import sys
import shlex
from pathlib import Path
from typing import List, Optional, Any
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

# Windows compatibility for readline
try:
    import readline
except ImportError:
    # Windows doesn't have readline by default
    try:
        import pyreadline3 as readline
    except ImportError:
        readline = None

from ..config import get_config
from ..database.chroma_db import ChromaDatabase
from ..search.hybrid import HybridSearchEngine
from ..fork.generator import ForkMDGenerator
from ..fork.smart_generator import SmartForkMDGenerator, ContextExtractionConfig
from ..ui.progress import (
    DEFAULT_THEME, get_theme_colors, get_semantic_color,
    display_discovery_phase, SmartForkProgress,
)


class SmartForkShell(cmd.Cmd):
    """Interactive shell for SmartFork commands.
    
    Features:
    - Command shortcuts (s, i, f, df, st, t, c, q)
    - Quick fork with number keys after search
    - Command history with up/down arrows
    - Tab completion for session IDs and commands
    - Persistent session state (remembers last search results)
    - Full CLI command set: config, theme, reset, compact, cluster, tree, vault,
      test, metrics, abtest, titles, watch, and more
    """
    
    # Use ASCII-only characters for Windows compatibility
    intro = """\n+--------------------------------------------------------------+
|  SmartFork Interactive Shell                                  |
|                                                               |
|  Type 'help' for commands, 'exit' to quit                    |
|  Quick tip: After search, press [1-9] to fork that result    |
+--------------------------------------------------------------+\n"""
    
    prompt = "SmartFork> "
    
    def __init__(self):
        # Disable readline if not available (Windows compatibility)
        if readline is None:
            self.use_rawinput = False
        
        super().__init__()
        self.console = Console()
        self.config = get_config()
        self.db = None
        self.search_engine = None
        self.last_results: List[Any] = []
        self.last_sessions: List[dict] = []
        self.fork_generator = None
        
        # Theme support
        self.theme_name = getattr(self.config, "theme", DEFAULT_THEME)
        self.theme = get_theme_colors(self.theme_name)
        self.semantic = self.theme.get("semantic", {})
        self.info_color = self.semantic.get("info", self.theme["text_primary"])
        self.success_color = self.semantic.get("success", self.theme["done_color"])
        self.warning_color = self.semantic.get("warning", "#F59E0B")
        self.error_color = self.semantic.get("error", "#EF4444")
        self.accent_color = self.semantic.get("accent", self.theme["text_primary"])
        
        self._init_database()
    
    def _refresh_theme(self):
        """Refresh theme colors from current config."""
        self.theme_name = getattr(self.config, "theme", DEFAULT_THEME)
        self.theme = get_theme_colors(self.theme_name)
        self.semantic = self.theme.get("semantic", {})
        self.info_color = self.semantic.get("info", self.theme["text_primary"])
        self.success_color = self.semantic.get("success", self.theme["done_color"])
        self.warning_color = self.semantic.get("warning", "#F59E0B")
        self.error_color = self.semantic.get("error", "#EF4444")
        self.accent_color = self.semantic.get("accent", self.theme["text_primary"])
        
    def _init_database(self):
        """Initialize database connection."""
        try:
            self.db = ChromaDatabase(self.config.chroma_db_path)
            self.search_engine = HybridSearchEngine(self.db)
            self.fork_generator = ForkMDGenerator(self.db)
        except Exception as e:
            self.console.print(f"[{self.warning_color}]Warning: Could not initialize database: {e}[/{self.warning_color}]")
            self.console.print(f"[{self.warning_color}]Run 'index' command to create database.[/{self.warning_color}]")
    
    def _ensure_db(self) -> bool:
        """Ensure database is initialized, return True if ready."""
        if self.db is None or self.search_engine is None:
            self.console.print(f"[{self.error_color}]Database not initialized. Run 'index' first.[/{self.error_color}]")
            return False
        return True
    
    def _display_welcome(self):
        """Display welcome message with colorful styling."""
        welcome_text = Text()
        welcome_text.append("SmartFork Interactive Shell\n", style=f"bold {self.info_color}")
        welcome_text.append("Type 'help' for commands, 'exit' to quit\n", style=f"dim {self.theme['text_muted']}")
        welcome_text.append("Quick tip: After search, press [1-9] to fork that result", style=f"dim {self.theme['text_muted']}")
        
        self.console.print(Panel(
            welcome_text,
            box=box.DOUBLE,
            border_style=self.theme["panel_border"]
        ))
    
    def preloop(self):
        """Called before the command loop starts."""
        self._display_welcome()
    
    def default(self, line: str):
        """Handle unknown commands - check for quick fork numbers."""
        stripped = line.strip()
        
        # Check if it's a number for quick fork
        if stripped.isdigit():
            num = int(stripped)
            if 1 <= num <= 9:
                self.do_fork(str(num))
                return
            else:
                self.console.print(f"[{self.error_color}]Quick fork only supports 1-9. Use 'fork <n>' for larger numbers.[/{self.error_color}]")
                return
        
        # Try to parse as a command with arguments
        parts = shlex.split(line)
        if parts:
            self.console.print(f"[{self.error_color}]Unknown command: {parts[0]}[/{self.error_color}]")
            self.console.print(f"[{self.theme['text_muted']}]Type 'help' for available commands[/{self.theme['text_muted']}]")
    
    def emptyline(self):
        """Do nothing on empty line."""
        pass
    
    def do_exit(self, arg: str):
        """Exit the interactive shell."""
        self.console.print(f"[{self.success_color}]Goodbye![/{self.success_color}]")
        return True
    
    def do_quit(self, arg: str):
        """Alias for exit."""
        return self.do_exit(arg)
    
    def do_EOF(self, arg: str):
        """Handle Ctrl+D (EOF)."""
        print()  # New line after Ctrl+D
        return self.do_exit(arg)
    
    def do_clear(self, arg: str):
        """Clear the screen."""
        self.console.clear()
    
    def do_status(self, arg: str):
        """Show indexing status."""
        if not self._ensure_db():
            return
        
        try:
            total_chunks = self.db.get_session_count()
            unique_sessions = len(self.db.get_unique_sessions())
        except Exception:
            total_chunks = 0
            unique_sessions = 0
        
        # Check tasks directory
        if self.config.kilo_code_tasks_path.exists():
            task_dirs = [d for d in self.config.kilo_code_tasks_path.iterdir() if d.is_dir()]
            total_tasks = len(task_dirs)
        else:
            total_tasks = 0
        
        # Display status
        table = Table(show_header=False, box=box.SIMPLE)
        table.add_column("Property", style=self.info_color)
        table.add_column("Value", style=self.success_color)
        
        table.add_row("Kilo Code Tasks Path", str(self.config.kilo_code_tasks_path))
        table.add_row("Database Path", str(self.config.chroma_db_path))
        table.add_row("Total Task Directories", str(total_tasks))
        table.add_row("Indexed Sessions", str(unique_sessions))
        table.add_row("Total Chunks", str(total_chunks))
        
        if total_tasks > 0:
            coverage = unique_sessions / max(1, total_tasks) * 100
            table.add_row("Index Coverage", f"{unique_sessions}/{total_tasks} ({coverage:.1f}%)")
        
        self.console.print(Panel(
            table,
            title="Status",
            border_style=self.theme["panel_border"]
        ))
        
        if unique_sessions < total_tasks:
            self.console.print(f"[{self.warning_color}]Tip: Run 'index' to index remaining sessions[/{self.warning_color}]")
    
    def do_search(self, arg: str):
        """Search indexed sessions.
        
        Usage: search <query> [--results N]
        Alias: s
        """
        if not self._ensure_db():
            return
        
        if not arg.strip():
            self.console.print(f"[{self.error_color}]Usage: search <query>[/{self.error_color}]")
            return
        
        # Parse arguments
        parts = shlex.split(arg)
        query = parts[0]
        n_results = 5
        
        # Check for --results flag
        for i, part in enumerate(parts):
            if part in ("--results", "-n") and i + 1 < len(parts):
                try:
                    n_results = int(parts[i + 1])
                except ValueError:
                    pass
                break
        
        try:
            current_dir = str(Path.cwd())
            results = self.search_engine.search(query, current_dir=current_dir, n_results=n_results)
            
            if not results:
                self.console.print(f"[{self.warning_color}]No results found.[/{self.warning_color}]")
                self.last_results = []
                return
            
            self.last_results = results
            
            self.console.print(f"[{self.theme['text_muted']}]Found {len(results)} results (using hybrid search)[/{self.theme['text_muted']}]\n")
            
            for i, r in enumerate(results, 1):
                score_pct = f"{r.score:.1%}"
                breakdown = r.breakdown
                
                # Get session title if available
                session_title = r.metadata.get("session_title")
                if session_title:
                    title_text = f"[{i}] {session_title}"
                else:
                    title_text = f"[{i}] Session {r.session_id[:16]}..."
                
                # Build breakdown string
                breakdown_str = " | ".join([
                    f"sem:{breakdown.get('semantic', 0):.2f}",
                    f"bm25:{breakdown.get('bm25', 0):.2f}",
                    f"rec:{breakdown.get('recency', 0):.2f}",
                    f"path:{breakdown.get('path', 0):.2f}"
                ])
                
                # Get last active
                last_active = r.metadata.get("last_active", "Unknown")
                if last_active and last_active != "Unknown":
                    try:
                        dt = datetime.fromisoformat(last_active)
                        last_active = dt.strftime("%Y-%m-%d")
                    except:
                        pass
                
                # Theme-aware border colors based on score
                if r.score > 0.7:
                    border = self.success_color
                elif r.score > 0.4:
                    border = self.warning_color
                else:
                    border = self.error_color
                
                panel_content = f"[bold {self.info_color}]Score:[/bold {self.info_color}] {score_pct}\n"
                panel_content += f"[{self.theme['text_muted']}]{breakdown_str}[/{self.theme['text_muted']}]\n"
                panel_content += f"[{self.theme['text_muted']}]Last active: {last_active}[/{self.theme['text_muted']}]"
                
                self.console.print(Panel(
                    panel_content,
                    title=title_text,
                    border_style=border
                ))
            
            self.console.print(f"\n[{self.theme['text_muted']}]Tip: Type a number [1-9] to fork that result[/{self.theme['text_muted']}]")
            
        except Exception as e:
            self.console.print(f"[{self.error_color}]Search error: {e}[/{self.error_color}]")
    
    # Shortcut for search
    do_s = do_search
    
    def do_fork(self, arg: str):
        """Generate a fork.md context file from a session.
        
        Usage: fork <session_id_or_number> [query] [--smart]
        Alias: f
        
        If a number is provided (1-9), forks the corresponding result from last search or sessions list.
        If a query is provided with --smart flag, uses query-aware smart fork generation.
        """
        if not self._ensure_db():
            return
        
        arg = arg.strip()
        if not arg:
            self.console.print(f"[{self.error_color}]Usage: fork <session_id_or_number> [query] [--smart][/{self.error_color}]")
            return
        
        # Parse arguments
        parts = shlex.split(arg)
        
        # Check for --smart flag
        use_smart = "--smart" in parts
        if use_smart:
            parts.remove("--smart")
        
        # First part is the session identifier (ID or number)
        session_arg = parts[0]
        # Rest could be a query for smart mode
        query = ' '.join(parts[1:]) if len(parts) > 1 else None
        
        # Check if it's a number referring to last search results or sessions list
        session_id = session_arg
        if session_arg.isdigit():
            num = int(session_arg)
            # Check last_results first (from search/detect-fork)
            if self.last_results:
                if num < 1 or num > len(self.last_results):
                    self.console.print(f"[{self.error_color}]Invalid result number. Last search has {len(self.last_results)} results.[/{self.error_color}]")
                    return
                session_id = self.last_results[num - 1].session_id
                self.console.print(f"[{self.theme['text_muted']}]Selected result #{num}: {session_id[:20]}...[/{self.theme['text_muted']}]")
            # Then check last_sessions (from sessions command)
            elif self.last_sessions:
                if num < 1 or num > len(self.last_sessions):
                    self.console.print(f"[{self.error_color}]Invalid session number. Last sessions list has {len(self.last_sessions)} sessions.[/{self.error_color}]")
                    return
                session_id = self.last_sessions[num - 1]["id"]
                self.console.print(f"[{self.theme['text_muted']}]Selected session #{num}: {session_id[:20]}...[/{self.theme['text_muted']}]")
            else:
                self.console.print(f"[{self.error_color}]No search results or sessions list. Run 'search' or 'sessions' first.[/{self.error_color}]")
                return
        
        try:
            # Check if session exists
            chunks = self.db.get_session_chunks(session_id)
            if not chunks:
                self.console.print(f"[{self.error_color}]Session not found: {session_id}[/{self.error_color}]")
                return
            
            # Get session title if available
            session_title = chunks[0].metadata.session_title if chunks else None
            if session_title:
                self.console.print(f"[bold {self.info_color}]Generating fork.md for:[/bold {self.info_color}] {session_title}")
            else:
                self.console.print(f"[bold {self.info_color}]Generating fork.md for session:[/bold {self.info_color}] {session_id[:20]}...")
            
            current_dir = str(Path.cwd())
            
            # Use smart fork if --smart flag is set and query is provided
            if use_smart:
                if not query:
                    self.console.print(f"[{self.error_color}]Error: --smart requires a query. Usage: fork <session_id> <query> --smart[/{self.error_color}]")
                    return
                
                self.console.print(f"[{self.theme['text_muted']}]Using smart query-aware generation[/{self.theme['text_muted']}]")
                generator = SmartForkMDGenerator(self.db)
                content = generator.generate(session_id, query, current_dir)
            else:
                # Use standard fork generator
                content = self.fork_generator.generate(session_id, query or "forked from interactive shell", current_dir)
            
            # Save to file
            short_id = session_id[:8]
            output = Path(f"fork_{short_id}.md")
            output.write_text(content, encoding="utf-8")
            
            self.console.print(f"[{self.success_color}]✓ Fork saved to:[/{self.success_color}] {output.absolute()}")
            
        except Exception as e:
            self.console.print(f"[{self.error_color}]Fork error: {e}[/{self.error_color}]")
    
    # Shortcut for fork
    do_f = do_fork
    
    def do_smartfork(self, arg: str):
        """Generate a smart context fork based on a query.
        
        Usage: smartfork <session_id> [query] [--max-tokens N] [--output path]
        
        Generates context-aware fork.md using query-based chunk retrieval.
        If output path is not provided, shows preview and prompts to save.
        """
        if not self._ensure_db():
            return
        
        if not arg.strip():
            self.console.print(f"[{self.error_color}]Usage: smartfork <session_id> [query] [--max-tokens N] [--output path][/{self.error_color}]")
            return
        
        # Parse arguments: session is first word, rest is query until -- flags
        parts = shlex.split(arg)
        if len(parts) < 2:
            self.console.print(f"[{self.error_color}]Error: Both session_id and query are required[/{self.error_color}]")
            return
        
        session_id = parts[0]
        
        # Find -- flags, treat rest as query
        query_parts = []
        max_tokens = 2000
        output_path = None
        
        i = 1
        while i < len(parts):
            if parts[i] == '--max-tokens' and i + 1 < len(parts):
                try:
                    max_tokens = int(parts[i + 1])
                except ValueError:
                    self.console.print(f"[{self.warning_color}]Warning: Invalid max-tokens value, using default 2000[/{self.warning_color}]")
                i += 2
            elif parts[i] == '--output' and i + 1 < len(parts):
                output_path = Path(parts[i + 1])
                i += 2
            else:
                query_parts.append(parts[i])
                i += 1
        
        query = ' '.join(query_parts)
        if not query:
            self.console.print(f"[{self.error_color}]Error: Query is required[/{self.error_color}]")
            return
        
        # Check if session exists
        try:
            chunks = self.db.get_session_chunks(session_id)
            if not chunks:
                self.console.print(f"[{self.error_color}]Session not found: {session_id}[/{self.error_color}]")
                return
            
            # Get session title if available
            session_title = chunks[0].metadata.session_title if chunks else None
            if session_title:
                self.console.print(f"[bold {self.info_color}]Generating smart fork for:[/bold {self.info_color}] {session_title}")
            else:
                self.console.print(f"[bold {self.info_color}]Generating smart fork for session:[/bold {self.info_color}] {session_id[:20]}...")
            
            self.console.print(f"[{self.theme['text_muted']}]Query: {query}[/{self.theme['text_muted']}]")
            
        except Exception as e:
            self.console.print(f"[{self.error_color}]Error checking session: {e}[/{self.error_color}]")
            return
        
        # Generate smart fork
        try:
            current_dir = str(Path.cwd())
            generator = SmartForkMDGenerator(self.db)
            content = generator.generate(session_id, query, current_dir, max_tokens=max_tokens)
            
            # Preview
            self.console.print(f"\n[bold {self.info_color}]{'='*60}[/bold {self.info_color}]")
            self.console.print(f"[bold {self.info_color}]SMART FORK PREVIEW[/bold {self.info_color}]")
            self.console.print(f"[bold {self.info_color}]{'='*60}[/bold {self.info_color}]")
            
            preview = content[:1000] + "..." if len(content) > 1000 else content
            self.console.print(Panel(
                preview,
                border_style=self.theme["panel_border"]
            ))
            
            self.console.print(f"[bold {self.info_color}]{'='*60}[/bold {self.info_color}]")
            self.console.print(f"\n[{self.theme['text_muted']}]Total length: {len(content)} chars, ~{len(content)//4} tokens[/{self.theme['text_muted']}]")
            
            # Save
            if output_path:
                generator.save(session_id, query, output_path, current_dir, max_tokens=max_tokens)
                self.console.print(f"[{self.success_color}]✓ Saved to:[/{self.success_color}] {output_path}")
            else:
                save = input("Save to file? (y/n): ").lower()
                if save == 'y':
                    path_input = input("Enter file path: ").strip()
                    if path_input:
                        save_path = Path(path_input)
                        generator.save(session_id, query, save_path, current_dir, max_tokens=max_tokens)
                        self.console.print(f"[{self.success_color}]✓ Saved to:[/{self.success_color}] {save_path}")
                    else:
                        self.console.print(f"[{self.warning_color}]No path provided, file not saved[/{self.warning_color}]")
                        
        except Exception as e:
            self.console.print(f"[{self.error_color}]Smart fork error: {e}[/{self.error_color}]")
    
    def do_resume(self, arg: str):
        """Quick resume with query - shortcut for smartfork.
        
        Usage: resume <session_id> <query>
        
        Generates smart context fork with default settings.
        Equivalent to: smartfork <session_id> <query>
        """
        if not self._ensure_db():
            return
        
        if not arg.strip():
            self.console.print(f"[{self.error_color}]Usage: resume <session_id> <query>[/{self.error_color}]")
            return
        
        # Parse: session is first word, rest is query
        parts = shlex.split(arg)
        if len(parts) < 2:
            self.console.print(f"[{self.error_color}]Error: Both session_id and query are required[/{self.error_color}]")
            return
        
        session_id = parts[0]
        query = ' '.join(parts[1:])
        
        # Delegate to smartfork with defaults
        self.console.print(f"[{self.theme['text_muted']}]Resuming session with query-aware extraction...[/{self.theme['text_muted']}]")
        self.do_smartfork(f"{session_id} {query}")
    
    def help_smartfork(self):
        """Display help for smartfork command."""
        help_text = f"""[bold {self.info_color}]smartfork - Generate smart context fork based on query[/bold {self.info_color}]

Usage:
  smartfork <session_id> [query] [--max-tokens N] [--output path]

Arguments:
  session_id    The session ID to fork context from
  query         Search query for context extraction (required)

Options:
  --max-tokens N    Maximum tokens to include (default: 2000)
  --output path     Output file path (optional, prompts if not provided)

Examples:
  smartfork task_abc123 JWT authentication
  smartfork task_abc123 database connection pooling --max-tokens 3000
  smartfork task_abc123 API error handling --output my_fork.md

Description:
  Uses query-aware chunk retrieval to find the most relevant
  context from the session, creating a focused fork.md file."""
        
        self.console.print(Panel(
            help_text,
            title="Help: smartfork",
            border_style=self.theme["panel_border"]
        ))
    
    def help_resume(self):
        """Display help for resume command."""
        help_text = f"""[bold {self.info_color}]resume - Quick resume with query[/bold {self.info_color}]

Usage:
  resume <session_id> <query>

Arguments:
  session_id    The session ID to resume from
  query         Search query describing what you want to retrieve

Examples:
  resume task_abc123 authentication middleware
  resume task_xyz789 database migration issues

Description:
  A convenience shortcut for 'smartfork' with default settings.
  Generates a smart context fork to resume work on a previous task."""
        
        self.console.print(Panel(
            help_text,
            title="Help: resume",
            border_style=self.theme["panel_border"]
        ))
    
    def do_detect_fork(self, arg: str):
        """Find relevant past sessions to fork context from.
        
        Usage: detect-fork <query> [--results N]
        Alias: df
        """
        if not self._ensure_db():
            return
        
        if not arg.strip():
            self.console.print(f"[{self.error_color}]Usage: detect-fork <query>[/{self.error_color}]")
            return
        
        # Parse arguments
        parts = shlex.split(arg)
        query = parts[0]
        n_results = 5
        
        # Check for --results flag
        for i, part in enumerate(parts):
            if part in ("--results", "-n") and i + 1 < len(parts):
                try:
                    n_results = int(parts[i + 1])
                except ValueError:
                    pass
                break
        
        try:
            current_dir = str(Path.cwd())
            results = self.search_engine.search(query, current_dir=current_dir, n_results=n_results * 2)
            
            if not results:
                self.console.print(f"[{self.warning_color}]No relevant sessions found.[/{self.warning_color}]")
                self.last_results = []
                return
            
            results = results[:n_results]
            self.last_results = results
            
            self.console.print(f"[{self.theme['text_muted']}]Found {len(results)} relevant session(s):[/{self.theme['text_muted']}]\n")
            
            for i, r in enumerate(results, 1):
                score_pct = f"{r.score:.1%}"
                breakdown = r.breakdown
                
                # Get session title
                session_title = r.metadata.get("session_title")
                if session_title:
                    title_text = f"[{i}] {session_title}"
                else:
                    title_text = f"[{i}] {r.session_id[:20]}..."
                
                # Get files
                files = r.metadata.get("files_in_context", [])
                files_preview = f"\n[{self.theme['text_muted']}]Files:[/{self.theme['text_muted']}] {', '.join(files[:3])}..." if len(files) > 3 else f"\n[{self.theme['text_muted']}]Files:[/{self.theme['text_muted']}] {', '.join(files)}" if files else ""
                
                # Last active
                last_active = r.metadata.get("last_active", "Unknown")
                if last_active and last_active != "Unknown":
                    try:
                        dt = datetime.fromisoformat(last_active)
                        last_active = dt.strftime("%Y-%m-%d")
                    except:
                        pass
                
                # Match reasons
                match_reasons = []
                if breakdown.get('semantic', 0) > 0.5:
                    match_reasons.append("semantic")
                if breakdown.get('bm25', 0) > 0.5:
                    match_reasons.append("keyword")
                if breakdown.get('recency', 0) > 0.8:
                    match_reasons.append("recent")
                if breakdown.get('path', 0) > 0.3:
                    match_reasons.append("same project")
                
                match_str = f" ({', '.join(match_reasons)})" if match_reasons else ""
                
                panel_content = f"[bold {self.success_color}]{score_pct}[/bold {self.success_color}] relevance{match_str}{files_preview}\n\n"
                panel_content += f"[{self.theme['text_muted']}]Last active: {last_active}[/{self.theme['text_muted']}]"
                
                self.console.print(Panel(
                    panel_content,
                    title=title_text,
                    border_style=self.theme["panel_border"]
                ))
            
            self.console.print(f"\n[{self.theme['text_muted']}]Tip: Type a number [1-9] to fork that result[/{self.theme['text_muted']}]")
            
        except Exception as e:
            self.console.print(f"[{self.error_color}]Detect-fork error: {e}[/{self.error_color}]")
    
    # Shortcut for detect-fork
    do_df = do_detect_fork
    
    def do_index(self, arg: str):
        """Index all Kilo Code sessions.
        
        Usage: index [--force]
        """
        from ..indexer.indexer import FullIndexer
        
        # Check if tasks path exists
        if not self.config.kilo_code_tasks_path.exists():
            self.console.print(f"[{self.error_color}]Error: Tasks path does not exist: {self.config.kilo_code_tasks_path}[/{self.error_color}]")
            return
        
        # Parse arguments
        parts = shlex.split(arg)
        force = "--force" in parts or "-f" in parts
        
        # Initialize database if needed
        if self.db is None:
            self.db = ChromaDatabase(self.config.chroma_db_path)
        
        if force:
            self.console.print(f"[{self.warning_color}]Resetting database...[/{self.warning_color}]")
            self.db.reset()
        
        indexer = FullIndexer(self.db, chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap)
        
        # Discover all sessions
        all_sessions = [
            item for item in self.config.kilo_code_tasks_path.iterdir()
            if item.is_dir() and (item / "api_conversation_history.json").exists()
        ]
        
        if not all_sessions:
            self.console.print(f"[{self.warning_color}]No sessions found to index.[/{self.warning_color}]")
            return
        
        # Discovery phase
        db_session_ids = set()
        try:
            db_session_ids = set(self.db.get_unique_sessions())
        except Exception:
            pass
        
        # Use standalone function instead of method
        all_sessions, new_count, _ = display_discovery_phase(
            tasks_path=self.config.kilo_code_tasks_path,
            db_session_ids=db_session_ids,
            console=self.console,
            theme_name=self.theme_name,
        )
        
        # Filter to only new sessions
        sessions_to_index = [s for s in all_sessions if s.name not in db_session_ids]
        
        if not sessions_to_index:
            self.console.print(f"\n[{self.success_color}]All sessions already indexed. No new sessions to process.[/{self.success_color}]")
            total_db_sessions = len(db_session_ids)
            # Re-initialize search engine with updated db
            self.search_engine = HybridSearchEngine(self.db)
            self.fork_generator = ForkMDGenerator(self.db)
            return
        
        self.console.print(f"[{self.info_color}]-> Indexing {len(sessions_to_index)} new sessions...[/{self.info_color}]\n")
        
        # Use animated progress display
        with SmartForkProgress(total_sessions=len(sessions_to_index), theme_name=self.theme_name, console=self.console) as progress:
            for i, session_dir in enumerate(sessions_to_index):
                progress.set_session(session_dir.name)
                progress.set_phase("Parsing", 0.0)
                
                # Index the session
                try:
                    chunks = indexer.index_session(session_dir)
                    progress.set_phase("Parsing", 1.0)
                    progress.set_phase("Embedding", 0.0)
                    
                    # Get title from DB
                    title = None
                    try:
                        session_chunks = self.db.get_session_chunks(session_dir.name)
                        if session_chunks and len(session_chunks) > 0:
                            title = session_chunks[0].metadata.session_title
                            progress.set_phase("Embedding", 1.0)
                    except Exception:
                        progress.set_phase("Embedding", 1.0)
                    
                    progress.add_chunks(chunks)
                    progress.set_bm25((i + 1) / len(sessions_to_index))
                    progress.advance()
                except Exception as e:
                    progress.add_error()
                    self.console.print(f"[{self.error_color}]Error indexing {session_dir.name}: {e}[/{self.error_color}]")
            
            progress.finish()
        
        # CRITICAL: Finalize to flush any remaining pending chunks to database
        indexer.finalize()
        
        # Get final stats
        total_db_sessions = 0
        try:
            total_db_sessions = len(self.db.get_unique_sessions())
        except Exception:
            pass
        
        # Display completion summary
        self.console.print(f"\n[{self.success_color}]✓ Indexing complete! {total_db_sessions} sessions indexed.[/{self.success_color}]")
        
        # Re-initialize search engine with updated db
        self.search_engine = HybridSearchEngine(self.db)
        self.fork_generator = ForkMDGenerator(self.db)
    
    # Shortcut for index
    do_i = do_index
    
    # Shortcut for status
    do_st = do_status
    
    # Shortcut for quit
    do_q = do_quit
    
    def do_sessions(self, arg: str):
        """List all indexed sessions with IDs and titles.
        
        Usage: sessions [--limit N]
        Alias: sl
        
        Options:
            --limit, -n N    Maximum number of sessions to show (default: 20)
        """
        if not self._ensure_db():
            return
        
        try:
            total_chunks = self.db.get_session_count()
            if total_chunks == 0:
                self.console.print(f"[{self.warning_color}]No sessions indexed. Run 'index' first.[/{self.warning_color}]")
                return
            
            # Get all unique sessions
            session_ids = self.db.get_unique_sessions()
            
            if not session_ids:
                self.console.print(f"[{self.warning_color}]No sessions found in database.[/{self.warning_color}]")
                return
            
            # Parse limit argument
            parts = shlex.split(arg) if arg else []
            limit = 20
            for i, part in enumerate(parts):
                if part in ("--limit", "-n", "-l") and i + 1 < len(parts):
                    try:
                        limit = int(parts[i + 1])
                    except ValueError:
                        pass
                    break
            
            # Gather session info with titles
            sessions_info = []
            for session_id in session_ids:
                try:
                    chunks = self.db.get_session_chunks(session_id)
                    if chunks:
                        # Get title from first chunk's metadata
                        title = chunks[0].metadata.session_title or "Untitled"
                        # Get last active timestamp
                        last_active = chunks[0].metadata.last_active or "Unknown"
                        # Count chunks for this session
                        chunk_count = len(chunks)
                        sessions_info.append({
                            "id": session_id,
                            "title": title,
                            "last_active": last_active,
                            "chunks": chunk_count
                        })
                except Exception as e:
                    sessions_info.append({
                        "id": session_id,
                        "title": "Error loading",
                        "last_active": "Unknown",
                        "chunks": 0
                    })
            
            # Sort by last active (most recent first)
            sessions_info.sort(key=lambda x: x["last_active"] if x["last_active"] != "Unknown" else "", reverse=True)
            
            # Store sessions for fork-by-number support
            self.last_sessions = sessions_info[:limit]
            
            # Display header
            self.console.print(Panel.fit(
                f"[bold {self.info_color}]{len(sessions_info)} indexed sessions[/bold {self.info_color}]",
                title="Session List",
                border_style=self.theme["panel_border"]
            ))
            
            # Create table
            table = Table(show_header=True)
            table.add_column("#", style=self.accent_color, justify="right", width=4)
            table.add_column("Session ID", style=self.info_color, width=12)
            table.add_column("Title", style=self.success_color, min_width=30)
            table.add_column("Last Active", style=self.theme["text_muted"], width=12)
            table.add_column("Chunks", style=self.accent_color, justify="right", width=8)
            
            for i, session in enumerate(sessions_info[:limit], 1):
                # Format short ID (first 8 chars)
                short_id = session["id"][:8]
                full_id = session["id"]
                
                # Create clickable text using Rich hyperlink
                # Clicking copies the full ID (supported in modern terminals)
                clickable_id = f"[link=copy:{full_id}]{short_id}[/link]"
                
                # Format last active
                last_active = session["last_active"]
                if last_active and last_active != "Unknown":
                    try:
                        dt = datetime.fromisoformat(last_active)
                        last_active = dt.strftime("%Y-%m-%d")
                    except:
                        pass
                
                table.add_row(
                    str(i),
                    clickable_id,
                    session["title"],
                    last_active,
                    str(session["chunks"])
                )
            
            self.console.print(table)
            
            if len(sessions_info) > limit:
                self.console.print(f"\n[{self.theme['text_muted']}]... and {len(sessions_info) - limit} more sessions (use --limit to show more)[/{self.theme['text_muted']}]")
            
            self.console.print(f"\n[{self.theme['text_muted']}]Tip: Click the Session ID to copy full ID, or use 'fork <number>'[/{self.theme['text_muted']}]")
            
        except Exception as e:
            self.console.print(f"[{self.error_color}]Error listing sessions: {e}[/{self.error_color}]")
    
    # Shortcut for sessions
    do_sl = do_sessions
    
    def do_results(self, arg: str):
        """Show last search results again.
        
        Usage: results
        """
        if not self.last_results:
            self.console.print(f"[{self.warning_color}]No previous search results.[/{self.warning_color}]")
            return
        
        self.console.print(f"[{self.theme['text_muted']}]Last search results ({len(self.last_results)} items):[/{self.theme['text_muted']}]\n")
        
        for i, r in enumerate(self.last_results, 1):
            score_pct = f"{r.score:.1%}"
            
            session_title = r.metadata.get("session_title")
            if session_title:
                title_text = f"[{i}] {session_title}"
            else:
                title_text = f"[{i}] Session {r.session_id[:16]}..."
            
            # Theme-aware border colors based on score
            if r.score > 0.7:
                border = self.success_color
            elif r.score > 0.4:
                border = self.warning_color
            else:
                border = self.error_color
            
            self.console.print(Panel(
                f"[bold {self.info_color}]Score:[/bold {self.info_color}] {score_pct}",
                title=title_text,
                border_style=border
            ))
        
        self.console.print(f"\n[{self.theme['text_muted']}]Tip: Type a number [1-9] to fork that result[/{self.theme['text_muted']}]")
    
    def do_config(self, arg: str):
        """Show current configuration.
        
        Usage: config
        Alias: c
        """
        from ..config import CONFIG_FILE
        
        table = Table(show_header=False, box=box.SIMPLE)
        table.add_column("Setting", style=self.info_color)
        table.add_column("Value", style=self.success_color)
        
        for key, value in self.config.model_dump().items():
            table.add_row(key, str(value))
        
        self.console.print(Panel(
            table,
            title="Configuration",
            border_style=self.theme["panel_border"]
        ))
        
        self.console.print(f"\n[{self.theme['text_muted']}]Config file: {CONFIG_FILE}[/{self.theme['text_muted']}]")
    
    # Shortcut for config
    do_c = do_config
    
    def do_theme(self, arg: str):
        """Set or view the color theme.
        
        Usage: theme [name] [--list]
        Alias: t
        
        Examples:
            theme              # Show current theme
            theme obsidian     # Set theme to obsidian
            theme --list       # List all available themes
        """
        from ..ui.progress import THEMES, DEFAULT_THEME
        from ..config import reload_config
        
        parts = shlex.split(arg) if arg else []
        list_themes = "--list" in parts or "-l" in parts
        
        current = getattr(self.config, "theme", DEFAULT_THEME)
        
        # Extract theme name if provided (not a flag)
        theme_name = None
        for part in parts:
            if not part.startswith("-"):
                theme_name = part
                break
        
        if list_themes or theme_name is None:
            # Show theme list
            theme_table = Table(show_header=True, box=box.SIMPLE)
            theme_table.add_column("Theme", style="bold", width=12)
            theme_table.add_column("Description", style="dim", width=40)
            theme_table.add_column("", width=10)
            
            for tid, td in THEMES.items():
                c0, c1, c2 = [b["color"] for b in td["bars"]]
                swatch = f"[{c0}]▪[/{c0}][{c1}]▪[/{c1}][{c2}]▪[/{c2}] {td['name']}"
                status = "[green]● active[/green]" if tid == current else ""
                theme_table.add_row(swatch, td["desc"], status)
            
            self.console.print(Panel(
                theme_table,
                title="[bold]SmartFork Themes[/bold]",
                box=box.ROUNDED,
                border_style=self.theme["panel_border"]
            ))
            
            if theme_name is None:
                self.console.print(f"\n  Current: [bold]{current}[/bold]")
                self.console.print(f"  Set with: [dim]theme <name>[/dim]\n")
            return
        
        # Set theme
        theme_name = theme_name.lower()
        if theme_name not in THEMES:
            self.console.print(f"[{self.error_color}]Unknown theme '{theme_name}'[/{self.error_color}]")
            self.console.print(f"[{self.theme['text_muted']}]Valid: {', '.join(THEMES.keys())}[/{self.theme['text_muted']}]")
            return
        
        # Update and save config
        self.config.theme = theme_name
        self.config.save()
        
        # Refresh theme colors
        self._refresh_theme()
        
        td = THEMES[theme_name]
        c = td["bars"][1]["color"]
        self.console.print(f"\n  [{c}]✓[/{c}] Theme → [bold]{td['name']}[/bold] — {td['desc']}")
        self.console.print(f"  [{self.theme['text_muted']}]Saved to config[/{self.theme['text_muted']}]")
    
    # Shortcut for theme
    do_t = do_theme
    
    def do_reset(self, arg: str):
        """Reset the database (WARNING: deletes all indexed data).
        
        Usage: reset [--force]
        
        Options:
            --force, -f    Skip confirmation prompt
        """
        if not self.db:
            self.console.print(f"[{self.error_color}]Database not initialized.[/{self.error_color}]")
            return
        
        parts = shlex.split(arg) if arg else []
        force = "--force" in parts or "-f" in parts
        
        if not force:
            self.console.print(f"[{self.warning_color}]Warning: This will delete all indexed data![/{self.warning_color}]")
            confirm = input("Are you sure? (yes/no): ").strip().lower()
            if confirm not in ("yes", "y"):
                self.console.print(f"[{self.warning_color}]Aborted.[/{self.warning_color}]")
                return
        
        try:
            self.db.reset()
            self.console.print(f"[{self.success_color}]✓ Database reset complete.[/{self.success_color}]")
            # Re-initialize
            self._init_database()
        except Exception as e:
            self.console.print(f"[{self.error_color}]Error resetting database: {e}[/{self.error_color}]")
    
    def do_compact(self, arg: str):
        """Check for sessions at risk of compaction or export them.
        
        Usage: compact [check|export] [options]
        
        Subcommands:
            check              Check for at-risk sessions
            export             Export at-risk sessions
        
        Options:
            --messages, -m N   Message count threshold (default: 100)
            --days, -d N       Age threshold in days (default: 7)
            --dry-run          Show what would be exported
            --auto, -a         Export all at-risk sessions automatically
        """
        from ..intelligence.pre_compaction import PreCompactionHook, CompactionManager
        
        parts = shlex.split(arg) if arg else []
        subcommand = parts[0] if parts else "check"
        
        # Parse options
        threshold_messages = 100
        threshold_days = 7
        dry_run = "--dry-run" in parts
        auto = "--auto" in parts or "-a" in parts
        
        for i, part in enumerate(parts):
            if part in ("--messages", "-m") and i + 1 < len(parts):
                try:
                    threshold_messages = int(parts[i + 1])
                except ValueError:
                    pass
            elif part in ("--days", "-d") and i + 1 < len(parts):
                try:
                    threshold_days = int(parts[i + 1])
                except ValueError:
                    pass
        
        try:
            if subcommand == "check":
                hook = PreCompactionHook(threshold_messages, threshold_days)
                at_risk = hook.check_sessions(self.config.kilo_code_tasks_path)
                
                if not at_risk:
                    self.console.print(f"[{self.success_color}]✓ No sessions at risk of compaction.[/{self.success_color}]")
                    return
                
                self.console.print(Panel.fit(
                    f"[bold {self.warning_color}]{len(at_risk)} sessions at risk[/bold {self.warning_color}]",
                    title="Compaction Check",
                    border_style=self.theme["panel_border"]
                ))
                
                table = Table(show_header=True)
                table.add_column("Session", style=self.info_color)
                table.add_column("Messages", style=self.warning_color, justify="right")
                table.add_column("Age (days)", style=self.accent_color, justify="right")
                table.add_column("Risk", style=self.error_color)
                
                for session in at_risk[:20]:
                    table.add_row(
                        session["session_id"][:20],
                        str(session["message_count"]),
                        str(session["age_days"]),
                        session["risk_level"]
                    )
                
                self.console.print(table)
                
                if len(at_risk) > 20:
                    self.console.print(f"\n[{self.theme['text_muted']}]... and {len(at_risk) - 20} more[/{self.theme['text_muted']}]")
                    
            elif subcommand == "export":
                manager = CompactionManager()
                results = manager.run_auto_export(dry_run=dry_run)
                
                if dry_run:
                    self.console.print(Panel.fit(
                        f"[bold {self.info_color}]{results['at_risk']} sessions would be exported[/bold {self.info_color}]",
                        title="Dry Run",
                        border_style=self.theme["panel_border"]
                    ))
                else:
                    self.console.print(Panel.fit(
                        f"[bold {self.success_color}]Exported {results['exported']} sessions[/bold {self.success_color}]\n"
                        f"Failed: {results['failed']}",
                        title="Compaction Export",
                        border_style=self.theme["panel_border"]
                    ))
                
                if results['sessions']:
                    table = Table(show_header=True)
                    table.add_column("Session", style=self.info_color)
                    table.add_column("Action", style=self.success_color)
                    
                    for session in results['sessions'][:10]:
                        table.add_row(
                            session["session_id"][:20],
                            session["action"]
                        )
                    
                    self.console.print(table)
            else:
                self.console.print(f"[{self.error_color}]Unknown subcommand: {subcommand}[/{self.error_color}]")
                self.console.print(f"[{self.theme['text_muted']}]Use 'check' or 'export'[/{self.theme['text_muted']}]")
                
        except Exception as e:
            self.console.print(f"[{self.error_color}]Error: {e}[/{self.error_color}]")
    
    def do_cluster(self, arg: str):
        """Analyze session clusters and find duplicates.
        
        Usage: cluster
        """
        from ..intelligence.clustering import SessionClusterer
        
        try:
            clusterer = SessionClusterer()
            
            with self.console.status(f"[bold {self.info_color}]Analyzing clusters..."):
                analysis = clusterer.analyze_clusters()
            
            self.console.print(Panel.fit(
                f"[bold {self.info_color}]{analysis['total_clusters']} clusters found[/bold {self.info_color}]\n"
                f"{analysis['noise_sessions']} unclustered sessions\n"
                f"{analysis['potential_duplicates']} potential duplicates",
                title="Cluster Analysis",
                border_style=self.theme["panel_border"]
            ))
            
            if analysis['clusters']:
                table = Table(show_header=True)
                table.add_column("Cluster", style=self.info_color, justify="right")
                table.add_column("Sessions", style=self.success_color, justify="right")
                table.add_column("Top Topics", style=self.accent_color)
                
                for cluster in analysis['clusters'][:10]:
                    topics = ", ".join(cluster.get('common_topics', [])[:5]) if cluster.get('common_topics') else "-"
                    table.add_row(
                        str(cluster['cluster_id']),
                        str(cluster['session_count']),
                        topics
                    )
                
                self.console.print(f"\n[bold {self.info_color}]Top Clusters:[/bold {self.info_color}]")
                self.console.print(table)
            
            if analysis['duplicate_pairs']:
                self.console.print(f"\n[bold {self.warning_color}]Potential Duplicates:[/bold {self.warning_color}]")
                for a, b, sim in analysis['duplicate_pairs'][:5]:
                    self.console.print(f"  • {a[:16]}... ↔ {b[:16]}... ({sim:.1%})")
                    
        except Exception as e:
            self.console.print(f"[{self.error_color}]Error analyzing clusters: {e}[/{self.error_color}]")
    
    def do_tree(self, arg: str):
        """Build, visualize, or export conversation branching tree.
        
        Usage: tree [build|visualize|export] [options]
        
        Subcommands:
            build              Build conversation tree
            visualize          Visualize tree (default)
            export             Export as interactive HTML
        
        Options:
            --session, -s ID   Root session to visualize
            --expanded, -e     Show expanded view
            --output, -o FILE  Output file for export
            --open, -b         Open in browser after export
        """
        from ..intelligence.branching import BranchingTree
        
        parts = shlex.split(arg) if arg else []
        subcommand = parts[0] if parts and not parts[0].startswith("-") else "visualize"
        
        # Parse options
        session_id = None
        expanded = "--expanded" in parts or "-e" in parts
        output = None
        open_browser = "--open" in parts or "-b" in parts
        
        for i, part in enumerate(parts):
            if part in ("--session", "-s") and i + 1 < len(parts):
                session_id = parts[i + 1]
            elif part in ("--output", "-o") and i + 1 < len(parts):
                output = Path(parts[i + 1])
        
        try:
            tree = BranchingTree()
            
            if subcommand == "build":
                with self.console.status(f"[bold {self.info_color}]Building tree..."):
                    tree.auto_build_tree(self.config.kilo_code_tasks_path)
                
                stats = tree.get_stats()
                
                self.console.print(Panel.fit(
                    f"[bold {self.info_color}]{stats['total_sessions']} sessions[/bold {self.info_color}] in tree\n"
                    f"{stats['root_sessions']} roots, {stats['leaf_sessions']} leaves\n"
                    f"Max depth: {stats['max_depth']}",
                    title="Tree Built",
                    border_style=self.theme["panel_border"]
                ))
                
            elif subcommand == "visualize":
                stats = tree.get_stats()
                tree_text = tree.visualize_tree(session_id, compact=not expanded)
                
                self.console.print(Panel.fit(
                    tree_text,
                    title=f"Conversation Tree ({stats['total_sessions']} sessions, {stats['root_sessions']} roots)",
                    border_style=self.theme["panel_border"]
                ))
                
                self.console.print(f"\n[{self.theme['text_muted']}]Stats: {stats['leaf_sessions']} leaves, max depth {stats['max_depth']}[/{self.theme['text_muted']}]")
                
            elif subcommand == "export":
                with self.console.status(f"[bold {self.info_color}]Generating HTML..."):
                    html_path = tree.export_html(output)
                
                self.console.print(f"[{self.success_color}]✓ Tree exported to:[/{self.success_color}] {html_path}")
                
                stats = tree.get_stats()
                self.console.print(Panel.fit(
                    f"[bold {self.info_color}]{stats['total_sessions']}[/bold {self.info_color}] sessions\n"
                    f"[bold {self.info_color}]{stats['root_sessions']}[/bold {self.info_color}] root sessions\n"
                    f"[bold {self.info_color}]{stats['leaf_sessions']}[/bold {self.info_color}] leaf sessions\n"
                    f"Max depth: [bold {self.info_color}]{stats['max_depth']}[/bold {self.info_color}]",
                    title="Tree Statistics",
                    border_style=self.theme["panel_border"]
                ))
                
                if open_browser:
                    import webbrowser
                    webbrowser.open(f"file://{html_path.absolute()}")
                    self.console.print(f"[{self.theme['text_muted']}]Opened in browser[/{self.theme['text_muted']}]")
            else:
                self.console.print(f"[{self.error_color}]Unknown subcommand: {subcommand}[/{self.error_color}]")
                
        except Exception as e:
            self.console.print(f"[{self.error_color}]Error with tree: {e}[/{self.error_color}]")
    
    def do_vault(self, arg: str):
        """Vault operations: add, list, restore, search.
        
        Usage: vault <operation> [options]
        
        Operations:
            add <session_id>       Add session to vault
            list                   List vaulted sessions
            restore <session_id>   Restore session from vault
            search <query>         Search within vaulted sessions
        
        Options:
            --password, -p PASS    Password for encryption/decryption
            --output, -o DIR       Output directory for restore
        """
        from ..intelligence.privacy import PrivacyVault
        
        parts = shlex.split(arg) if arg else []
        if not parts:
            self.console.print(f"[{self.error_color}]Usage: vault <add|list|restore|search> [options][/{self.error_color}]")
            return
        
        operation = parts[0]
        
        # Parse options
        password = None
        output = None
        
        for i, part in enumerate(parts):
            if part in ("--password", "-p") and i + 1 < len(parts):
                password = parts[i + 1]
            elif part in ("--output", "-o") and i + 1 < len(parts):
                output = Path(parts[i + 1])
        
        try:
            if operation == "add":
                if len(parts) < 2 or parts[1].startswith("-"):
                    self.console.print(f"[{self.error_color}]Usage: vault add <session_id>[/{self.error_color}]")
                    return
                
                session_id = parts[1]
                if not password:
                    password = input("Enter vault password: ")
                
                task_dir = self.config.kilo_code_tasks_path / session_id
                if not task_dir.exists():
                    self.console.print(f"[{self.error_color}]Session not found: {session_id}[/{self.error_color}]")
                    return
                
                vault = PrivacyVault(password)
                
                with self.console.status(f"[bold {self.info_color}]Adding to vault..."):
                    success = vault.add_to_vault(session_id, task_dir)
                
                if success:
                    self.console.print(f"[{self.success_color}]✓ Session {session_id[:16]}... added to vault[/{self.success_color}]")
                else:
                    self.console.print(f"[{self.error_color}]Failed to add to vault[/{self.error_color}]")
                    
            elif operation == "list":
                vault = PrivacyVault()
                sessions = vault.list_vaulted_sessions()
                
                if not sessions:
                    self.console.print(f"[{self.theme['text_muted']}]No sessions in vault[/{self.theme['text_muted']}]")
                    return
                
                self.console.print(Panel.fit(
                    f"[bold {self.info_color}]{len(sessions)} vaulted sessions[/bold {self.info_color}]",
                    title="Privacy Vault",
                    border_style=self.theme["panel_border"]
                ))
                
                table = Table(show_header=True)
                table.add_column("Session", style=self.info_color)
                table.add_column("Vaulted At", style=self.success_color)
                table.add_column("Files", style=self.accent_color, justify="right")
                
                for session in sessions:
                    table.add_row(
                        session["session_id"][:20],
                        session.get("vaulted_at", "unknown"),
                        str(session.get("file_count", 0))
                    )
                
                self.console.print(table)
                
            elif operation == "restore":
                if len(parts) < 2 or parts[1].startswith("-"):
                    self.console.print(f"[{self.error_color}]Usage: vault restore <session_id>[/{self.error_color}]")
                    return
                
                session_id = parts[1]
                if not password:
                    password = input("Enter vault password: ")
                
                vault = PrivacyVault(password)
                
                with self.console.status(f"[bold {self.info_color}]Restoring from vault..."):
                    result = vault.restore_from_vault(session_id, output)
                
                if result:
                    self.console.print(f"[{self.success_color}]✓ Session restored to: {result}[/{self.success_color}]")
                else:
                    self.console.print(f"[{self.error_color}]Failed to restore session[/{self.error_color}]")
                    
            elif operation == "search":
                if len(parts) < 2 or parts[1].startswith("-"):
                    self.console.print(f"[{self.error_color}]Usage: vault search <query>[/{self.error_color}]")
                    return
                
                query = parts[1]
                if not password:
                    password = input("Enter vault password: ")
                
                vault = PrivacyVault(password)
                
                with self.console.status(f"[bold {self.info_color}]Searching vault..."):
                    results = vault.search_vault(query)
                
                self.console.print(Panel.fit(
                    f"[bold {self.info_color}]{len(results)} results[/bold {self.info_color}]",
                    title="Vault Search",
                    border_style=self.theme["panel_border"]
                ))
                
                for r in results[:10]:
                    self.console.print(Panel(
                        f"[{self.theme['text_muted']}]{r['preview']}[/{self.theme['text_muted']}]",
                        title=f"{r['session_id'][:16]}... / {r['file']}",
                        border_style=self.theme["panel_border"]
                    ))
            else:
                self.console.print(f"[{self.error_color}]Unknown operation: {operation}[/{self.error_color}]")
                
        except Exception as e:
            self.console.print(f"[{self.error_color}]Error: {e}[/{self.error_color}]")
    
    def do_test(self, arg: str):
        """Run SmartFork tests.
        
        Usage: test [--suite SUITE]
        
        Options:
            --suite, -s SUITE    Test suite to run (indexer, search, database, fork)
        """
        from ..testing.test_runner import create_default_test_runner
        
        parts = shlex.split(arg) if arg else []
        suite = None
        
        for i, part in enumerate(parts):
            if part in ("--suite", "-s") and i + 1 < len(parts):
                suite = parts[i + 1]
                break
        
        try:
            runner = create_default_test_runner()
            
            if suite:
                with self.console.status(f"[bold {self.info_color}]Running {suite} tests..."):
                    result = runner.run_suite(suite)
                suites = [result]
            else:
                with self.console.status(f"[bold {self.info_color}]Running all tests..."):
                    suites = runner.run_all()
            
            # Display results
            for suite_result in suites:
                suite_color = self.success_color if suite_result.failed_count == 0 else self.error_color
                self.console.print(Panel.fit(
                    f"[bold {suite_color}]{suite_result.passed_count}/{len(suite_result.tests)} passed[/bold {suite_color}]\n"
                    f"Duration: {suite_result.total_duration_ms:.0f}ms",
                    title=f"Test Suite: {suite_result.name}",
                    border_style=self.theme["panel_border"]
                ))
                
                if suite_result.failed_count > 0:
                    table = Table(show_header=True)
                    table.add_column("Test", style=self.info_color)
                    table.add_column("Status", style=self.error_color)
                    table.add_column("Error", style=self.theme["text_muted"])
                    
                    for test in suite_result.tests:
                        if not test.passed:
                            error_msg = test.error_message or ""
                            table.add_row(
                                test.name,
                                "FAILED",
                                error_msg[:50] + "..." if len(error_msg) > 50 else error_msg
                            )
                    
                    self.console.print(table)
            
            summary = runner.get_summary()
            self.console.print(f"\n[{self.theme['text_muted']}]Total: {summary['passed']}/{summary['total_tests']} passed "
                              f"({summary['pass_rate']:.1%})[/{self.theme['text_muted']}]")
            
        except Exception as e:
            self.console.print(f"[{self.error_color}]Error running tests: {e}[/{self.error_color}]")
    
    def do_metrics(self, arg: str):
        """Show success metrics dashboard.
        
        Usage: metrics [--days N]
        
        Options:
            --days, -d N    Number of days to show (default: 7)
        """
        from ..testing.metrics_tracker import MetricsTracker
        
        parts = shlex.split(arg) if arg else []
        days = 7
        
        for i, part in enumerate(parts):
            if part in ("--days", "-d") and i + 1 < len(parts):
                try:
                    days = int(parts[i + 1])
                except ValueError:
                    pass
                break
        
        try:
            tracker = MetricsTracker()
            data = tracker.get_dashboard_data(days)
            
            self.console.print(Panel.fit(
                f"[bold {self.info_color}]Success Metrics[/bold {self.info_color}] (last {data['period_days']} days)",
                title="Metrics Dashboard",
                border_style=self.theme["panel_border"]
            ))
            
            # Key metrics
            table = Table(show_header=True)
            table.add_column("Metric", style=self.info_color)
            table.add_column("Value", style=self.success_color)
            
            km = data['key_metrics']
            table.add_row("Unique Sessions", str(data['unique_sessions']))
            table.add_row("Avg Fork Gen Time", f"{km['avg_fork_generation_time_ms']:.0f}ms")
            table.add_row("Context Recovered", f"{km['total_context_recovered_mb']:.1f}MB")
            table.add_row("Sessions/Day", f"{km['sessions_per_day']:.1f}")
            
            self.console.print(table)
            
            # Metric summaries
            if data['metric_summaries']:
                self.console.print(f"\n[bold {self.info_color}]Metric Trends:[/bold {self.info_color}]")
                for name, summary in data['metric_summaries'].items():
                    trend_color_map = {
                        'improving': self.success_color,
                        'stable': self.warning_color,
                        'degrading': self.error_color,
                        'insufficient_data': self.theme["text_muted"]
                    }
                    trend_color = trend_color_map.get(summary['trend'], self.theme["text_primary"])
                    
                    self.console.print(f"  {name}: {summary['mean']:.2f} "
                                      f"([{trend_color}]{summary['trend']}[/{trend_color}])")
            
        except Exception as e:
            self.console.print(f"[{self.error_color}]Error getting metrics: {e}[/{self.error_color}]")
    
    def do_abtest(self, arg: str):
        """Show A/B test status.
        
        Usage: abtest
        """
        from ..testing.ab_testing import ABTestManager
        
        try:
            manager = ABTestManager()
            summary = manager.get_test_summary()
            
            self.console.print(Panel.fit(
                f"[bold {self.info_color}]{summary['total_tests']}[/bold {self.info_color}] active tests\n"
                f"[bold {self.info_color}]{summary['total_sessions']}[/bold {self.info_color}] test sessions",
                title="A/B Testing",
                border_style=self.theme["panel_border"]
            ))
            
            if summary['active_tests']:
                table = Table(show_header=True)
                table.add_column("Test", style=self.info_color)
                table.add_column("Sessions", style=self.success_color, justify="right")
                table.add_column("Control", style=self.accent_color, justify="right")
                table.add_column("Treatment", style=self.warning_color, justify="right")
                table.add_column("Result", style=self.info_color)
                
                for test in summary['active_tests']:
                    result_str = "No data"
                    if test['result']:
                        sig = "sig" if test['result']['significant'] else "not sig"
                        result_str = f"{test['result']['improvement_pct']:+.1f}% ({sig})"
                    
                    table.add_row(
                        test['name'],
                        str(test['total_sessions']),
                        str(test['control']),
                        str(test['treatment']),
                        result_str
                    )
                
                self.console.print(table)
                
        except Exception as e:
            self.console.print(f"[{self.error_color}]Error getting A/B test status: {e}[/{self.error_color}]")
    
    def do_titles(self, arg: str):
        """Generate or update session titles for all indexed sessions.
        
        Usage: titles [--force] [--dry-run]
        
        Options:
            --force, -f     Force regeneration of all titles
            --dry-run, -d   Show what would be generated without updating
        """
        from ..intelligence.titling import TitleManager, TitleGenerator
        from ..indexer.parser import KiloCodeParser
        from ..indexer.indexer import FullIndexer
        import time
        
        if not self._ensure_db():
            return
        
        parts = shlex.split(arg) if arg else []
        force = "--force" in parts or "-f" in parts
        dry_run = "--dry-run" in parts or "-d" in parts
        
        try:
            # Get all unique sessions
            session_ids = self.db.get_unique_sessions()
            
            if not session_ids:
                self.console.print(f"[{self.warning_color}]No sessions found in database.[/{self.warning_color}]")
                return
            
            self.console.print(Panel.fit(
                f"[bold {self.info_color}]Update Session Titles[/bold {self.info_color}]\n"
                f"Found {len(session_ids)} sessions to process",
                title="SmartFork",
                border_style=self.theme["panel_border"]
            ))
            
            # Initialize title generator
            title_gen = TitleGenerator()
            title_manager = TitleManager(self.db, title_gen)
            parser = KiloCodeParser()
            
            # Track results
            updated = 0
            skipped = 0
            failed = 0
            
            # Process each session
            with self.console.status(f"[bold {self.info_color}]Generating titles...") as status:
                for i, session_id in enumerate(session_ids):
                    try:
                        # Check if session already has a title (unless force)
                        if not force:
                            chunks = self.db.get_session_chunks(session_id)
                            if chunks and chunks[0].metadata.session_title:
                                skipped += 1
                                continue
                        
                        # Parse the session to get full content
                        task_dir = self.config.kilo_code_tasks_path / session_id
                        if not task_dir.exists():
                            failed += 1
                            continue
                        
                        session = parser.parse_task_directory(task_dir)
                        if not session:
                            failed += 1
                            continue
                        
                        # Generate title
                        title = title_manager.generate_and_store_title(session)
                        
                        if dry_run:
                            self.console.print(f"[{self.theme['text_muted']}]{session_id[:16]}...[/{self.theme['text_muted']}] -> {title}")
                        else:
                            # Re-index the session to store the new title
                            indexer = FullIndexer(
                                self.db,
                                chunk_size=self.config.chunk_size,
                                chunk_overlap=self.config.chunk_overlap,
                                batch_size=self.config.batch_size
                            )
                            indexer.index_session(task_dir)
                            # CRITICAL: Finalize to flush pending chunks immediately
                            indexer.finalize()
                        
                        updated += 1
                        
                        if (i + 1) % 10 == 0:
                            status.update(f"[bold {self.info_color}]Processed {i + 1}/{len(session_ids)} sessions...")
                            
                        # Small delay in lite mode
                        if self.config.lite_mode and i % 5 == 0:
                            time.sleep(0.1)
                            
                    except Exception as e:
                        failed += 1
                
                # Display results
            self.console.print(f"\n[bold {self.info_color}]Results:[/bold {self.info_color}]")
            self.console.print(f"  [{self.success_color}]Updated:[/{self.success_color}] {updated}")
            self.console.print(f"  [{self.warning_color}]Skipped:[/{self.warning_color}] {skipped}")
            if failed > 0:
                self.console.print(f"  [{self.error_color}]Failed:[/{self.error_color}] {failed}")
            
            if dry_run:
                self.console.print(f"\n[{self.theme['text_muted']}]This was a dry run. Use without --dry-run to apply changes.[/{self.theme['text_muted']}]")
            else:
                self.console.print(f"\n[{self.success_color}]✓ Title update complete![/{self.success_color}]")
                
        except Exception as e:
            self.console.print(f"[{self.error_color}]Error updating titles: {e}[/{self.error_color}]")
    
    def do_watch(self, arg: str):
        """Watch for session changes and index incrementally.
        
        Usage: watch
        
        Press Ctrl+C to stop watching.
        """
        from ..indexer.watcher import TranscriptWatcher
        from ..indexer.indexer import IncrementalIndexer
        import time
        
        if not self.config.kilo_code_tasks_path.exists():
            self.console.print(f"[{self.error_color}]Error: Tasks path does not exist: {self.config.kilo_code_tasks_path}[/{self.error_color}]")
            return
        
        if self.db is None:
            self.db = ChromaDatabase(self.config.chroma_db_path)
        
        self.console.print(f"[bold {self.info_color}]Starting watcher... Press Ctrl+C to stop.[/{self.info_color}]\n")
        
        # Use longer poll interval in lite mode
        poll_interval = 10.0 if self.config.lite_mode else 5.0
        if self.config.lite_mode:
            self.console.print(f"  [dim]Lite mode: using {poll_interval}s poll interval[/dim]\n")
        
        incremental = IncrementalIndexer(self.db)
        watcher = TranscriptWatcher(
            self.config.kilo_code_tasks_path,
            incremental.on_session_changed,
            poll_interval=poll_interval
        )
        
        watcher.start()
        
        try:
            while True:
                # Longer sleep in lite mode to reduce CPU
                sleep_interval = 2.0 if self.config.lite_mode else 1.0
                time.sleep(sleep_interval)
        except KeyboardInterrupt:
            self.console.print(f"\n[{self.warning_color}]Stopping watcher...[/{self.warning_color}]")
            watcher.stop()
            self.console.print(f"[{self.success_color}]✓ Watcher stopped.[/{self.success_color}]")
    
    def do_history(self, arg: str):
        """Show command history."""
        # cmd module stores history internally, but we can display a message
        self.console.print(f"[{self.theme['text_muted']}]Command history is available using Up/Down arrow keys[/{self.theme['text_muted']}]")
        self.console.print(f"[{self.theme['text_muted']}]Previous commands:[/{self.theme['text_muted']}]")
        
        # Access cmd's history (limited)
        if hasattr(self, 'lastcmd') and self.lastcmd:
            self.console.print(f"  [{self.info_color}]Most recent:[/{self.info_color}] {self.lastcmd}")
    
    def complete_search(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Tab completion for search command."""
        return []
    
    def complete_fork(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Tab completion for fork command - suggest session IDs."""
        if not self.db:
            return []
        
        try:
            sessions = self.db.get_unique_sessions()
            return [s for s in sessions if s.startswith(text)]
        except Exception:
            return []
    
    def complete_smartfork(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Tab completion for smartfork command - suggest session IDs."""
        if not self.db:
            return []
        
        try:
            sessions = self.db.get_unique_sessions()
            return [s for s in sessions if s.startswith(text)]
        except Exception:
            return []
    
    def complete_resume(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Tab completion for resume command - suggest session IDs."""
        if not self.db:
            return []
        
        try:
            sessions = self.db.get_unique_sessions()
            return [s for s in sessions if s.startswith(text)]
        except Exception:
            return []
    
    def completedefault(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Default tab completion."""
        return []
    
    def do_help(self, arg: str):
        """Show help information."""
        if arg:
            # Show help for specific command
            super().do_help(arg)
            return
        
        # Show custom help panel
        help_text = f"""[bold {self.info_color}]Available Commands:[/bold {self.info_color}]

[bold {self.theme['text_primary']}]Core Commands:[/bold {self.theme['text_primary']}]
  search <query>           Search indexed sessions (alias: s)
  sessions [--limit N]     List all indexed sessions with IDs and titles (alias: sl)
  fork <id|num> [q] [--smart]  Fork a session (alias: f, supports --smart with query)
  smartfork <id> <query>   Generate smart context fork based on query
  resume <id> <query>      Quick resume with smart fork (shortcut)
  detect-fork <q>          Find relevant sessions to fork (alias: df)
  index [--force]          Index all Kilo Code sessions (alias: i)

[bold {self.theme['text_primary']}]Database & Configuration:[/bold {self.theme['text_primary']}]
  status                   Show indexing status (alias: st)
  config                   Show current configuration (alias: c)
  theme [name] [--list]    Set or view color theme (alias: t)
  reset [--force]          Reset the database (WARNING: deletes all data)

[bold {self.theme['text_primary']}]Intelligence Features:[/bold {self.theme['text_primary']}]
  compact [check|export]   Check for compaction risk or export sessions
  cluster                  Analyze session clusters and find duplicates
  tree [build|vis|export]  Build, visualize, or export conversation tree
  vault <op> [options]     Vault operations: add/list/restore/search
  titles [--force]         Generate/update session titles
  watch                    Watch for changes and index incrementally

[bold {self.theme['text_primary']}]Testing & Metrics:[/bold {self.theme['text_primary']}]
  test [--suite NAME]      Run SmartFork tests
  metrics [--days N]       Show success metrics dashboard
  abtest                   Show A/B test status

[bold {self.theme['text_primary']}]Utility Commands:[/bold {self.theme['text_primary']}]
  results                  Show last search results again
  history                  Show command history info
  clear                    Clear the screen
  help                     Show this help message
  exit, quit, q            Exit the interactive shell

[bold {self.theme['text_primary']}]Quick Tips:[/bold {self.theme['text_primary']}]
  • After search/detect-fork, type [1-9] to quickly fork that result
  • Use Up/Down arrows for command history
  • Press Tab for command and session ID completion
  • Type 'help <command>' for detailed help on a specific command
"""
        
        self.console.print(Panel(
            help_text,
            title="SmartFork Interactive Shell Help",
            border_style=self.theme["panel_border"]
        ))


def start_interactive_shell():
    """Start the interactive shell."""
    shell = SmartForkShell()
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\n")
        shell.console.print(f"[{shell.success_color}]Goodbye![/{shell.success_color}]")
    except Exception as e:
        if "readline" in str(e).lower() or "backend" in str(e).lower():
            # Fallback for Windows readline issues
            shell.console.print(f"[{shell.warning_color}]Note: Advanced editing features not available on this terminal.[/{shell.warning_color}]")
            shell.console.print(f"[{shell.theme['text_muted']}]Basic interactive mode starting...[/{shell.theme['text_muted']}]\n")
            # Run in basic mode without readline
            shell.use_rawinput = False
            try:
                shell.cmdloop()
            except KeyboardInterrupt:
                print("\n")
                shell.console.print(f"[{shell.success_color}]Goodbye![/{shell.success_color}]")
        else:
            raise


if __name__ == "__main__":
    start_interactive_shell()
