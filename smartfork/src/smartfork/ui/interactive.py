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


class SmartForkShell(cmd.Cmd):
    """Interactive shell for SmartFork commands.
    
    Features:
    - Command shortcuts (s, f, df)
    - Quick fork with number keys after search
    - Command history with up/down arrows
    - Tab completion for session IDs and commands
    - Persistent session state (remembers last search results)
    - Special commands: help, exit, status, clear
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
        self.fork_generator = None
        self._init_database()
        
    def _init_database(self):
        """Initialize database connection."""
        try:
            self.db = ChromaDatabase(self.config.chroma_db_path)
            self.search_engine = HybridSearchEngine(self.db)
            self.fork_generator = ForkMDGenerator(self.db)
        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not initialize database: {e}[/yellow]")
            self.console.print("[yellow]Run 'index' command to create database.[/yellow]")
    
    def _ensure_db(self) -> bool:
        """Ensure database is initialized, return True if ready."""
        if self.db is None or self.search_engine is None:
            self.console.print("[red]Database not initialized. Run 'index' first.[/red]")
            return False
        return True
    
    def _display_welcome(self):
        """Display welcome message with colorful styling."""
        welcome_text = Text()
        welcome_text.append("SmartFork Interactive Shell\n", style="bold cyan")
        welcome_text.append("Type 'help' for commands, 'exit' to quit\n", style="dim")
        welcome_text.append("Quick tip: After search, press [1-9] to fork that result", style="dim")
        
        self.console.print(Panel(
            welcome_text,
            box=box.DOUBLE,
            border_style="cyan"
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
                self.console.print(f"[red]Quick fork only supports 1-9. Use 'fork <n>' for larger numbers.[/red]")
                return
        
        # Try to parse as a command with arguments
        parts = shlex.split(line)
        if parts:
            self.console.print(f"[red]Unknown command: {parts[0]}[/red]")
            self.console.print("[dim]Type 'help' for available commands[/dim]")
    
    def emptyline(self):
        """Do nothing on empty line."""
        pass
    
    def do_exit(self, arg: str):
        """Exit the interactive shell."""
        self.console.print("[green]Goodbye![/green]")
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
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
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
            border_style="blue"
        ))
        
        if unique_sessions < total_tasks:
            self.console.print("[yellow]Tip: Run 'index' to index remaining sessions[/yellow]")
    
    def do_search(self, arg: str):
        """Search indexed sessions.
        
        Usage: search <query> [--results N]
        Alias: s
        """
        if not self._ensure_db():
            return
        
        if not arg.strip():
            self.console.print("[red]Usage: search <query>[/red]")
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
                self.console.print("[yellow]No results found.[/yellow]")
                self.last_results = []
                return
            
            self.last_results = results
            
            self.console.print(f"[dim]Found {len(results)} results (using hybrid search)[/dim]\n")
            
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
                
                # Get technologies
                techs = r.metadata.get("technologies", [])
                tech_str = f"\n[dim]Tech: {', '.join(techs[:3])}[/dim]" if techs else ""
                
                # Get last active
                last_active = r.metadata.get("last_active", "Unknown")
                if last_active and last_active != "Unknown":
                    try:
                        dt = datetime.fromisoformat(last_active)
                        last_active = dt.strftime("%Y-%m-%d")
                    except:
                        pass
                
                border = "green" if r.score > 0.7 else "yellow" if r.score > 0.4 else "red"
                
                panel_content = f"[bold]Score:[/bold] {score_pct}\n"
                panel_content += f"[dim]{breakdown_str}[/dim]{tech_str}\n"
                panel_content += f"[dim]Last active: {last_active}[/dim]"
                
                self.console.print(Panel(
                    panel_content,
                    title=title_text,
                    border_style=border
                ))
            
            self.console.print("\n[dim]Tip: Type a number [1-9] to fork that result[/dim]")
            
        except Exception as e:
            self.console.print(f"[red]Search error: {e}[/red]")
    
    # Shortcut for search
    do_s = do_search
    
    def do_fork(self, arg: str):
        """Generate a fork.md context file from a session.
        
        Usage: fork <session_id_or_number>
        Alias: f
        
        If a number is provided (1-9), forks the corresponding result from last search.
        """
        if not self._ensure_db():
            return
        
        arg = arg.strip()
        if not arg:
            self.console.print("[red]Usage: fork <session_id_or_number>[/red]")
            return
        
        # Check if it's a number referring to last search results
        session_id = arg
        if arg.isdigit():
            num = int(arg)
            if not self.last_results:
                self.console.print("[red]No search results. Run 'search' first.[/red]")
                return
            if num < 1 or num > len(self.last_results):
                self.console.print(f"[red]Invalid result number. Last search has {len(self.last_results)} results.[/red]")
                return
            session_id = self.last_results[num - 1].session_id
            self.console.print(f"[dim]Selected result #{num}: {session_id[:20]}...[/dim]")
        
        try:
            # Check if session exists
            chunks = self.db.get_session_chunks(session_id)
            if not chunks:
                self.console.print(f"[red]Session not found: {session_id}[/red]")
                return
            
            # Get session title if available
            session_title = chunks[0].metadata.session_title if chunks else None
            if session_title:
                self.console.print(f"[bold]Generating fork.md for:[/bold] {session_title}")
            else:
                self.console.print(f"[bold]Generating fork.md for session:[/bold] {session_id[:20]}...")
            
            # Generate fork.md
            current_dir = str(Path.cwd())
            content = self.fork_generator.generate(session_id, "forked from interactive shell", current_dir)
            
            # Save to file
            short_id = session_id[:8]
            output = Path(f"fork_{short_id}.md")
            output.write_text(content, encoding="utf-8")
            
            self.console.print(f"[green]✓ Fork saved to:[/green] {output.absolute()}")
            
        except Exception as e:
            self.console.print(f"[red]Fork error: {e}[/red]")
    
    # Shortcut for fork
    do_f = do_fork
    
    def do_detect_fork(self, arg: str):
        """Find relevant past sessions to fork context from.
        
        Usage: detect-fork <query> [--results N]
        Alias: df
        """
        if not self._ensure_db():
            return
        
        if not arg.strip():
            self.console.print("[red]Usage: detect-fork <query>[/red]")
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
                self.console.print("[yellow]No relevant sessions found.[/yellow]")
                self.last_results = []
                return
            
            results = results[:n_results]
            self.last_results = results
            
            self.console.print(f"[dim]Found {len(results)} relevant session(s):[/dim]\n")
            
            for i, r in enumerate(results, 1):
                score_pct = f"{r.score:.1%}"
                breakdown = r.breakdown
                
                # Get session title
                session_title = r.metadata.get("session_title")
                if session_title:
                    title_text = f"[{i}] {session_title}"
                else:
                    title_text = f"[{i}] {r.session_id[:20]}..."
                
                # Get technologies
                techs = r.metadata.get("technologies", [])
                tech_str = f"\n[dim]Tech:[/dim] {', '.join(techs[:5])}" if techs else ""
                
                # Get files
                files = r.metadata.get("files_in_context", [])
                files_preview = f"\n[dim]Files:[/dim] {', '.join(files[:3])}..." if len(files) > 3 else f"\n[dim]Files:[/dim] {', '.join(files)}" if files else ""
                
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
                
                panel_content = f"[bold green]{score_pct}[/bold green] relevance{match_str}{tech_str}{files_preview}\n\n"
                panel_content += f"[dim]Last active: {last_active}[/dim]"
                
                self.console.print(Panel(
                    panel_content,
                    title=title_text
                ))
            
            self.console.print("\n[dim]Tip: Type a number [1-9] to fork that result[/dim]")
            
        except Exception as e:
            self.console.print(f"[red]Detect-fork error: {e}[/red]")
    
    # Shortcut for detect-fork
    do_df = do_detect_fork
    
    def do_index(self, arg: str):
        """Index all Kilo Code sessions.
        
        Usage: index [--force]
        """
        from ..indexer.indexer import FullIndexer
        from ..ui.progress import AnimatedProgressDisplay
        
        # Check if tasks path exists
        if not self.config.kilo_code_tasks_path.exists():
            self.console.print(f"[red]Error: Tasks path does not exist: {self.config.kilo_code_tasks_path}[/red]")
            return
        
        # Parse arguments
        parts = shlex.split(arg)
        force = "--force" in parts or "-f" in parts
        
        # Initialize database if needed
        if self.db is None:
            self.db = ChromaDatabase(self.config.chroma_db_path)
        
        if force:
            self.console.print("[yellow]Resetting database...[/yellow]")
            self.db.reset()
        
        indexer = FullIndexer(self.db, chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap)
        
        # Discover all sessions
        all_sessions = [
            item for item in self.config.kilo_code_tasks_path.iterdir()
            if item.is_dir() and (item / "api_conversation_history.json").exists()
        ]
        
        if not all_sessions:
            self.console.print("[yellow]No sessions found to index.[/yellow]")
            return
        
        # Use animated progress display
        progress_display = AnimatedProgressDisplay(self.console)
        
        # Discovery phase
        db_session_ids = set()
        try:
            db_session_ids = set(self.db.get_unique_sessions())
        except Exception:
            pass
        
        progress_display.display_discovery_phase(all_sessions, db_session_ids)
        
        # Filter to only new sessions
        sessions_to_index = [s for s in all_sessions if s.name not in db_session_ids]
        
        if not sessions_to_index:
            self.console.print("\n[green]All sessions already indexed. No new sessions to process.[/green]")
            total_db_sessions = len(db_session_ids)
            progress_display.display_completion_summary(total_db_sessions)
            
            # Re-initialize search engine with updated db
            self.search_engine = HybridSearchEngine(self.db)
            self.fork_generator = ForkMDGenerator(self.db)
            return
        
        self.console.print(f"[cyan]-> Indexing {len(sessions_to_index)} new sessions...[/cyan]\n")
        
        # Indexing phase
        def index_one(session_dir: Path):
            """Index a single session and return chunks + title."""
            chunks = indexer.index_session(session_dir)
            
            # Try to get title from chunks in DB
            title = None
            try:
                session_chunks = self.db.get_session_chunks(session_dir.name)
                if session_chunks and len(session_chunks) > 0:
                    title = session_chunks[0].metadata.session_title
            except Exception:
                pass
            
            return chunks, title
        
        results = progress_display.display_indexing_progress(sessions_to_index, index_one)
        
        # Get final stats
        total_db_sessions = 0
        try:
            total_db_sessions = len(self.db.get_unique_sessions())
        except Exception:
            pass
        
        # Display completion summary
        progress_display.display_completion_summary(total_db_sessions)
        
        # Re-initialize search engine with updated db
        self.search_engine = HybridSearchEngine(self.db)
        self.fork_generator = ForkMDGenerator(self.db)
    
    def do_results(self, arg: str):
        """Show last search results again.
        
        Usage: results
        """
        if not self.last_results:
            self.console.print("[yellow]No previous search results.[/yellow]")
            return
        
        self.console.print(f"[dim]Last search results ({len(self.last_results)} items):[/dim]\n")
        
        for i, r in enumerate(self.last_results, 1):
            score_pct = f"{r.score:.1%}"
            
            session_title = r.metadata.get("session_title")
            if session_title:
                title_text = f"[{i}] {session_title}"
            else:
                title_text = f"[{i}] Session {r.session_id[:16]}..."
            
            border = "green" if r.score > 0.7 else "yellow" if r.score > 0.4 else "red"
            
            self.console.print(Panel(
                f"[bold]Score:[/bold] {score_pct}",
                title=title_text,
                border_style=border
            ))
        
        self.console.print("\n[dim]Tip: Type a number [1-9] to fork that result[/dim]")
    
    def do_history(self, arg: str):
        """Show command history."""
        # cmd module stores history internally, but we can display a message
        self.console.print("[dim]Command history is available using Up/Down arrow keys[/dim]")
        self.console.print("[dim]Previous commands:[/dim]")
        
        # Access cmd's history (limited)
        if hasattr(self, 'lastcmd') and self.lastcmd:
            self.console.print(f"  [cyan]Most recent:[/cyan] {self.lastcmd}")
    
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
        help_text = """[bold cyan]Available Commands:[/bold cyan]

[bold]Core Commands:[/bold]
  search <query>     Search indexed sessions (alias: s)
  fork <id|num>      Fork a session by ID or result number (alias: f)
  detect-fork <q>    Find relevant sessions to fork (alias: df)
  index [--force]    Index all Kilo Code sessions

[bold]Utility Commands:[/bold]
  status             Show indexing status
  results            Show last search results again
  history            Show command history info
  clear              Clear the screen
  help               Show this help message
  exit, quit         Exit the interactive shell

[bold]Quick Tips:[/bold]
  • After search/detect-fork, type [1-9] to quickly fork that result
  • Use Up/Down arrows for command history
  • Press Tab for command and session ID completion
  • Type 'help <command>' for detailed help on a specific command
"""
        
        self.console.print(Panel(
            help_text,
            title="SmartFork Interactive Shell Help",
            border_style="cyan"
        ))


def start_interactive_shell():
    """Start the interactive shell."""
    shell = SmartForkShell()
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\n")
        shell.console.print("[green]Goodbye![/green]")
    except Exception as e:
        if "readline" in str(e).lower() or "backend" in str(e).lower():
            # Fallback for Windows readline issues
            shell.console.print("[yellow]Note: Advanced editing features not available on this terminal.[/yellow]")
            shell.console.print("[dim]Basic interactive mode starting...[/dim]\n")
            # Run in basic mode without readline
            shell.use_rawinput = False
            try:
                shell.cmdloop()
            except KeyboardInterrupt:
                print("\n")
                shell.console.print("[green]Goodbye![/green]")
        else:
            raise


if __name__ == "__main__":
    start_interactive_shell()
