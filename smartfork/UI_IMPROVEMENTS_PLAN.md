# SmartFork UI/UX Improvements Implementation Plan

## Overview

Based on the comprehensive analysis, here are the detailed implementation plans for the remaining UI/UX improvements. These focus on CLI enhancements that dramatically improve user experience before any web UI is considered.

---

## Phase 1: Rich Terminal Output (Week 1)

### 1.1 Enhanced Search Results Display

**Current State:** Basic panel output with raw scores
**Target State:** Visual progress bars, color-coded relevance, emoji indicators

#### Implementation

```python
# New file: smartfork/src/smartfork/ui/rich_output.py

from rich.console import Console
from rich.panel import Panel
from rich.progress import ProgressBar
from rich.text import Text
from rich.table import Table
from typing import List, Optional
from datetime import datetime

class SearchResultsRenderer:
    """Renders beautiful search results with visual indicators."""
    
    def __init__(self):
        self.console = Console()
    
    def render_score_bar(self, score: float, width: int = 20) -> str:
        """Render a visual score bar.
        
        94% -> ████████████████████░
        45% -> █████████░░░░░░░░░░░
        """
        filled = int(score * width)
        empty = width - filled
        
        if score >= 0.8:
            color = "green"
        elif score >= 0.5:
            color = "yellow"
        else:
            color = "red"
            
        return f"[{color}]{'█' * filled}[/][dim]{'░' * empty}[/]"
    
    def render_search_header(self, query: str, result_count: int, duration_ms: float):
        """Render the search header panel."""
        header = Text()
        header.append("🔍 ", style="bold")
        header.append(f"SmartFork Results for: ", style="bold")
        header.append(f'"{query}"\n', style="cyan")
        header.append(f"Found {result_count} relevant session", style="dim")
        header.append("s" if result_count != 1 else "", style="dim")
        header.append(f" in {duration_ms:.1f}s", style="dim")
        
        self.console.print(Panel(header, border_style="blue"))
    
    def render_result_card(
        self,
        rank: int,
        title: str,
        session_id: str,
        score: float,
        last_active: datetime,
        project_path: Optional[str],
        technologies: List[str],
        outcome: str,  # "solved", "partial", "abandoned"
        is_compacted: bool,
        preview_text: str,
    ):
        """Render a single result card."""
        # Score bar and percentage
        score_pct = f"{score:.0%}"
        score_bar = self.render_score_bar(score)
        
        # Build metadata line
        meta_parts = []
        
        # Date with calendar emoji
        date_str = last_active.strftime("%b %d, %Y")
        meta_parts.append(f"📅 {date_str}")
        
        # Project path with folder emoji
        if project_path:
            project_name = Path(project_path).name
            meta_parts.append(f"📁 {project_name}")
        
        # Technologies with appropriate emoji
        tech_emojis = {
            "Python": "🐍",
            "JavaScript": "📜",
            "TypeScript": "📘",
            "React": "⚛️",
            "FastAPI": "🚀",
            "Docker": "🐳",
            "Kubernetes": "☸️",
            "Rust": "🦀",
            "Go": "🐹",
            "PostgreSQL": "🐘",
            "Redis": "🔴",
        }
        tech_strs = []
        for tech in technologies[:3]:
            emoji = tech_emojis.get(tech, "🔧")
            tech_strs.append(f"{emoji} {tech}")
        if tech_strs:
            meta_parts.append(" ".join(tech_strs))
        
        # Outcome indicator
        outcome_emojis = {
            "solved": "✅ Solved",
            "partial": "⚠️ Partial",
            "abandoned": "❌ Abandoned",
        }
        outcome_str = outcome_emojis.get(outcome, "❓ Unknown")
        
        # Compacted indicator
        compacted_str = "  |  ⚠️ Session was compacted" if is_compacted else ""
        
        # Build the card content
        content = Text()
        content.append(f"#{rank}  ", style="bold")
        content.append(score_bar)
        content.append(f"  {score_pct}\n", style="bold green" if score > 0.7 else "yellow")
        content.append(f"{title}\n", style="bold cyan")
        content.append(f"{meta_parts[0]}", style="dim")
        for part in meta_parts[1:]:
            content.append(f"  {part}", style="dim")
        content.append(f"\n{outcome_str}{compacted_str}\n", style="dim")
        
        # Preview text (truncated)
        if preview_text:
            preview = preview_text[:100] + "..." if len(preview_text) > 100 else preview_text
            content.append(f'"{preview}"\n', style="italic dim")
        
        # Action hint
        content.append(f"\n→ smartfork fork {session_id[:8]}...", style="dim")
        
        # Border color based on score
        border = "green" if score > 0.7 else "yellow" if score > 0.4 else "red"
        
        self.console.print(Panel(content, border_style=border))
```

#### CLI Changes

```python
# In cli.py search command:

@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    n_results: int = typer.Option(5, "--results", "-n", help="Number of results"),
    path: Optional[Path] = typer.Option(None, "--path", "-p", help="Current directory"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Search indexed sessions with beautiful visual output."""
    import time
    from .ui.rich_output import SearchResultsRenderer
    
    config = get_config()
    db = ChromaDatabase(config.chroma_db_path)
    engine = HybridSearchEngine(db)
    
    if db.get_session_count() == 0:
        console.print("[yellow]No sessions indexed. Run 'smartfork index' first.[/yellow]")
        raise typer.Exit(1)
    
    # Time the search
    start_time = time.time()
    current_dir = str(path) if path else str(Path.cwd())
    results = engine.search(query, current_dir=current_dir, n_results=n_results)
    duration_ms = (time.time() - start_time) * 1000
    
    if json_output:
        output = [r.to_dict() for r in results]
        console.print(json.dumps(output, indent=2, default=str))
        return
    
    # Use new rich renderer
    renderer = SearchResultsRenderer()
    renderer.render_search_header(query, len(results), duration_ms)
    
    for i, r in enumerate(results, 1):
        # Extract metadata
        session_title = r.metadata.get("session_title", f"Session {r.session_id[:16]}...")
        last_active = datetime.fromisoformat(r.metadata.get("last_active", "2024-01-01"))
        technologies = r.metadata.get("technologies", [])
        project_path = r.metadata.get("project_path")
        outcome = r.metadata.get("outcome", "unknown")
        is_compacted = r.metadata.get("is_compacted", False)
        
        # Get preview text from first chunk
        preview = r.chunks[0][:200] if hasattr(r, 'chunks') and r.chunks else ""
        
        renderer.render_result_card(
            rank=i,
            title=session_title,
            session_id=r.session_id,
            score=r.score,
            last_active=last_active,
            project_path=project_path,
            technologies=technologies,
            outcome=outcome,
            is_compacted=is_compacted,
            preview_text=preview,
        )
```

**Time Estimate:** 6 hours
**Files Modified:** [`cli.py`](smartfork/src/smartfork/cli.py), new [`ui/rich_output.py`](smartfork/src/smartfork/ui/rich_output.py)

---

## Phase 2: Interactive Mode (Week 2)

### 2.1 Interactive Shell

**Current State:** Each command requires `smartfork` prefix
**Target State:** Persistent interactive session with shortcuts

#### Implementation

```python
# New file: smartfork/src/smartfork/ui/interactive.py

import cmd
import shlex
from typing import List, Optional
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

class SmartForkInteractive(cmd.Cmd):
    """Interactive shell for SmartFork."""
    
    intro = """
╔══════════════════════════════════════════════════════════════╗
║  ⚡ SmartFork Interactive Shell                               ║
║                                                               ║
║  Type 'help' for commands, 'exit' to quit                    ║
║  Quick tip: After search, press [1-9] to fork that result    ║
╚══════════════════════════════════════════════════════════════╝
    """
    prompt = "SmartFork> "
    
    def __init__(self):
        super().__init__()
        self.console = Console()
        self.last_results: List[dict] = []  # Store last search results
        self.history: List[str] = []
        
    def do_search(self, arg):
        """Search for sessions: search <query> [--results N]"""
        if not arg:
            self.console.print("[red]Usage: search <query>[/red]")
            return
        
        # Parse arguments
        parts = shlex.split(arg)
        query = parts[0]
        n_results = 5
        
        for i, part in enumerate(parts):
            if part in ("-n", "--results") and i + 1 < len(parts):
                n_results = int(parts[i + 1])
        
        # Run search
        self._run_search(query, n_results)
    
    def do_s(self, arg):
        """Alias for search"""
        self.do_search(arg)
    
    def _run_search(self, query: str, n_results: int):
        """Execute search and store results."""
        from ..search.hybrid import HybridSearchEngine
        from ..database.chroma_db import ChromaDatabase
        from ..config import get_config
        import time
        
        config = get_config()
        db = ChromaDatabase(config.chroma_db_path)
        engine = HybridSearchEngine(db)
        
        start_time = time.time()
        results = engine.search(query, n_results=n_results)
        duration_ms = (time.time() - start_time) * 1000
        
        # Store results for quick forking
        self.last_results = [
            {
                "session_id": r.session_id,
                "title": r.metadata.get("session_title", r.session_id[:16]),
                "score": r.score,
            }
            for r in results
        ]
        
        # Display with numbers for quick selection
        self.console.print(f"\n[dim]Found {len(results)} results in {duration_ms:.0f}ms[/dim]\n")
        
        for i, r in enumerate(results, 1):
            title = r.metadata.get("session_title", r.session_id[:16] + "...")
            score_pct = f"{r.score:.0%}"
            
            self.console.print(
                f"[{i}] [bold]{title}[/bold] "
                f"([green]{score_pct}[/green])"
            )
        
        if results:
            self.console.print(f"\n[dim]Press [1-{len(results)}] to fork, or type a command[/dim]")
    
    def do_fork(self, arg):
        """Fork a session: fork <session_id_or_number>"""
        if not arg:
            self.console.print("[red]Usage: fork <session_id_or_number>[/red]")
            return
        
        # Check if it's a number referring to last search results
        if arg.isdigit() and self.last_results:
            idx = int(arg) - 1
            if 0 <= idx < len(self.last_results):
                session_id = self.last_results[idx]["session_id"]
                self.console.print(f"[dim]Forking result #{arg}: {self.last_results[idx]['title'][:40]}...[/dim]")
            else:
                self.console.print(f"[red]Invalid result number. Last search had {len(self.last_results)} results.[/red]")
                return
        else:
            session_id = arg
        
        # Run fork command
        self._run_fork(session_id)
    
    def do_f(self, arg):
        """Alias for fork"""
        self.do_fork(arg)
    
    def _run_fork(self, session_id: str):
        """Execute fork command."""
        from ..fork.generator import ForkMDGenerator
        from ..database.chroma_db import ChromaDatabase
        from ..config import get_config
        from pathlib import Path
        
        config = get_config()
        db = ChromaDatabase(config.chroma_db_path)
        generator = ForkMDGenerator(db)
        
        content = generator.generate(session_id, "forked from interactive shell", str(Path.cwd()))
        
        output_path = Path(f"fork_{session_id[:8]}.md")
        output_path.write_text(content, encoding="utf-8")
        
        self.console.print(f"[green]✓ Fork saved to:[/green] {output_path.absolute()}")
    
    def do_status(self, arg):
        """Show indexing status"""
        from ..database.chroma_db import ChromaDatabase
        from ..config import get_config
        
        config = get_config()
        db = ChromaDatabase(config.chroma_db_path)
        
        total_chunks = db.get_session_count()
        unique_sessions = len(db.get_unique_sessions())
        
        self.console.print(Panel(
            f"[bold]📊 SmartFork Status[/bold]\n\n"
            f"Indexed Sessions: [cyan]{unique_sessions}[/cyan]\n"
            f"Total Chunks: [cyan]{total_chunks}[/cyan]\n"
            f"Database: [dim]{config.chroma_db_path}[/dim]",
            border_style="blue"
        ))
    
    def do_index(self, arg):
        """Index all sessions: index [--force]"""
        from ..indexer.indexer import FullIndexer
        from ..database.chroma_db import ChromaDatabase
        from ..config import get_config
        
        config = get_config()
        force = "--force" in arg or "-f" in arg
        
        db = ChromaDatabase(config.chroma_db_path)
        
        if force:
            db.reset()
            self.console.print("[yellow]Database reset.[/yellow]")
        
        indexer = FullIndexer(db)
        
        with self.console.status("[bold]Indexing sessions..."):
            result = indexer.index_all_sessions(config.kilo_code_tasks_path)
        
        self.console.print(
            f"[green]✓ Indexed {result.indexed} sessions "
            f"({result.chunks_created} chunks)[/green]"
        )
    
    def do_exit(self, arg):
        """Exit the interactive shell"""
        self.console.print("[dim]Goodbye! 👋[/dim]")
        return True
    
    def do_quit(self, arg):
        """Exit the interactive shell"""
        return self.do_exit(arg)
    
    def do_EOF(self, arg):
        """Handle Ctrl+D"""
        return self.do_exit(arg)
    
    def default(self, line):
        """Handle unknown commands - check if it's a number for quick fork."""
        if line.isdigit() and self.last_results:
            self.do_fork(line)
        else:
            self.console.print(f"[red]Unknown command: {line}[/red]")
            self.console.print("[dim]Type 'help' for available commands[/dim]")
    
    def emptyline(self):
        """Do nothing on empty line"""
        pass
    
    def do_help(self, arg):
        """Show help"""
        help_text = """
[bold]Available Commands:[/bold]

  [cyan]search <query>[/cyan]        Search indexed sessions
  [cyan]s <query>[/cyan]             Short alias for search
  
  [cyan]fork <id_or_num>[/cyan]      Fork a session (use number from last search)
  [cyan]f <id_or_num>[/cyan]         Short alias for fork
  
  [cyan]status[/cyan]                Show indexing status
  [cyan]index[/cyan]                 Index all sessions
  
  [cyan]detect-fork <query>[/cyan]   Find relevant sessions to fork
  [cyan]df <query>[/cyan]            Short alias for detect-fork
  
  [cyan]help[/cyan]                  Show this help
  [cyan]exit/quit[/cyan]             Exit the shell

[bold]Quick Tips:[/bold]
• After search, press [1-9] to immediately fork that result
• Use tab for command completion
• Up/down arrows navigate command history
        """
        self.console.print(help_text)


def start_interactive():
    """Entry point for interactive mode."""
    shell = SmartForkInteractive()
    shell.cmdloop()
```

#### CLI Integration

```python
# In cli.py:

@app.command(name="interactive")
def interactive_cmd():
    """Start interactive shell mode."""
    from .ui.interactive import start_interactive
    start_interactive()

# Alias
@app.command(name="i", hidden=True)
def interactive_alias():
    """Alias for interactive mode."""
    interactive_cmd()
```

**Time Estimate:** 8 hours
**Files Created:** [`ui/interactive.py`](smartfork/src/smartfork/ui/interactive.py)
**Files Modified:** [`cli.py`](smartfork/src/smartfork/cli.py)

---

## Phase 3: Progress Indicators (Week 2-3)

### 3.1 Enhanced Indexing Progress

**Current State:** Silent operation or basic spinner
**Target State:** Detailed progress bars with per-session status

#### Implementation

```python
# New file: smartfork/src/smartfork/ui/progress.py

from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
)
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from typing import Optional, Callable
from dataclasses import dataclass

@dataclass
class IndexingStats:
    """Statistics for indexing operation."""
    sessions_found: int = 0
    sessions_new: int = 0
    sessions_modified: int = 0
    sessions_processed: int = 0
    chunks_created: int = 0
    current_session: Optional[str] = None


class IndexingProgressDisplay:
    """Beautiful progress display for indexing operations."""
    
    def __init__(self):
        self.console = Console()
        self.stats = IndexingStats()
    
    def create_progress_bar(self) -> Progress:
        """Create a Rich Progress instance with custom columns."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(complete_style="green", finished_style="green"),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
    
    def display_discovery(self, total_dirs: int, new: int, modified: int):
        """Display the discovery phase results."""
        self.console.print("\n[bold]🔍 Scanning Kilo Code sessions directory...[/bold]")
        self.console.print(f"   Found [cyan]{new}[/cyan] new sessions, "
                          f"[yellow]{modified}[/yellow] modified since last index\n")
    
    def display_indexing_progress(
        self,
        sessions_to_process: List[Path],
        index_callback: Callable[[Path], tuple[int, str]],
    ) -> dict:
        """Display progress while indexing sessions.
        
        Args:
            sessions_to_process: List of session directories
            index_callback: Function that indexes a session and returns (chunks, title)
            
        Returns:
            Statistics dict
        """
        total = len(sessions_to_process)
        chunks_total = 0
        processed = 0
        
        with self.create_progress_bar() as progress:
            task = progress.add_task(
                "📦 Processing sessions...",
                total=total,
            )
            
            for session_dir in sessions_to_process:
                # Update description with current session
                short_name = session_dir.name[:20]
                progress.update(task, description=f"📦 Processing: {short_name}...")
                
                # Index the session
                try:
                    chunks, title = index_callback(session_dir)
                    chunks_total += chunks
                    processed += 1
                    
                    # Update progress
                    progress.update(task, advance=1)
                    
                except Exception as e:
                    progress.console.print(
                        f"[red]✗ Failed: {short_name} - {e}[/red]"
                    )
                    progress.update(task, advance=1)
        
        return {
            "processed": processed,
            "chunks": chunks_total,
        }
    
    def display_completion_summary(self, stats: dict):
        """Display the final summary panel."""
        summary = Panel(
            f"[bold green]✅ Index updated[/bold green]\n\n"
            f"  • {stats['sessions_added']} sessions added\n"
            f"  • {stats['sessions_updated']} sessions updated\n"
            f"  • {stats['chunks']:,} new chunks embedded\n"
            f"  • Index size: {stats['index_size_mb']:.1f}MB\n"
            f"\n[dim]Ready to search[/dim]",
            border_style="green",
        )
        self.console.print(summary)


# Integration with indexer

def index_with_progress(indexer, sessions: List[Path], config):
    """Index sessions with beautiful progress display."""
    from .progress import IndexingProgressDisplay
    
    display = IndexingProgressDisplay()
    
    # Discovery phase
    # (calculate new/modified before this call)
    display.display_discovery(len(sessions), len(new_sessions), len(modified_sessions))
    
    # Indexing phase
    def index_one(session_dir: Path) -> tuple[int, str]:
        result = indexer.index_session(session_dir)
        return result.chunks_created, result.session_title or session_dir.name
    
    results = display.display_indexing_progress(sessions, index_one)
    
    # Summary
    stats = {
        "sessions_added": results["processed"],
        "sessions_updated": 0,  # Calculate separately
        "chunks": results["chunks"],
        "index_size_mb": calculate_index_size(),
    }
    display.display_completion_summary(stats)
```

**Time Estimate:** 4 hours
**Files Created:** [`ui/progress.py`](smartfork/src/smartfork/ui/progress.py)

---

## Phase 4: Contextual Help (Week 3)

### 4.1 Smart Suggestions

**Current State:** Static help output
**Target State:** Context-aware tips based on user state

#### Implementation

```python
# New file: smartfork/src/smartfork/ui/contextual_help.py

from rich.console import Console
from rich.panel import Panel
from enum import Enum
from typing import Optional, List

class UserState(Enum):
    """States a user can be in."""
    FRESH_INSTALL = "fresh_install"
    NO_INDEX = "no_index"
    INDEXED_NO_SEARCH = "indexed_no_search"
    SEARCHED_NO_RESULTS = "searched_no_results"
    HAS_RESULTS = "has_results"
    RECENTLY_COMPACTED = "recently_compacted"


class ContextualHelpManager:
    """Provides context-aware help and suggestions."""
    
    def __init__(self):
        self.console = Console()
        self.user_actions: List[str] = []  # Track recent actions
    
    def detect_state(self, db) -> UserState:
        """Detect current user state from database."""
        session_count = db.get_session_count()
        
        if session_count == 0:
            # Check if it's truly fresh or index was reset
            return UserState.NO_INDEX
        
        if len(self.user_actions) == 0:
            return UserState.INDEXED_NO_SEARCH
        
        last_action = self.user_actions[-1]
        if last_action == "search_no_results":
            return UserState.SEARCHED_NO_RESULTS
        
        if last_action.startswith("search_"):
            return UserState.HAS_RESULTS
        
        return UserState.INDEXED_NO_SEARCH
    
    def show_welcome(self):
        """Show welcome message for first-time users."""
        welcome = Panel(
            "[bold]👋 Welcome to SmartFork![/bold]\n\n"
            "SmartFork helps you recover context from past Kilo Code sessions.\n\n"
            "[bold]Quick Start:[/bold]\n"
            "  1. [cyan]smartfork index[/cyan] - Index your Kilo Code sessions\n"
            "  2. [cyan]smartfork search \"your task\"[/cyan] - Find relevant sessions\n"
            "  3. [cyan]smartfork fork <session_id>[/cyan] - Generate context file\n\n"
            "[dim]Run 'smartfork interactive' for a guided experience[/dim]",
            border_style="green",
            title="Getting Started"
        )
        self.console.print(welcome)
    
    def after_index(self, sessions_indexed: int):
        """Show contextual help after indexing."""
        self.user_actions.append("index")
        
        tip = Panel(
            f"[bold green]✓ Successfully indexed {sessions_indexed} sessions![/bold green]\n\n"
            "💡 [bold]Next step:[/bold] Try searching for something:\n"
            f"   [cyan]smartfork search \"your current task\"[/cyan]\n\n"
            "[dim]Or launch interactive mode: smartfork interactive[/dim]",
            border_style="blue"
        )
        self.console.print(tip)
    
    def after_search_no_results(self, query: str):
        """Show help when search returns nothing."""
        self.user_actions.append("search_no_results")
        
        suggestions = [
            f"[cyan]smartfork index[/cyan] (if you have recent sessions)",
            f'[cyan]smartfork search "{" ".join(query.split()[:2])}"[/cyan] (broader query)',
            f"[cyan]smartfork status[/cyan] (check index health)",
        ]
        
        help_panel = Panel(
            f"[bold]😕 No results for \"{query}\"[/bold]\n\n"
            "💡 [bold]Try:[/bold]\n" +
            "\n".join(f"  • {s}" for s in suggestions),
            border_style="yellow"
        )
        self.console.print(help_panel)
    
    def after_search_with_results(self, query: str, result_count: int):
        """Show helpful tips after successful search."""
        self.user_actions.append(f"search_{result_count}")
        
        tip = Panel(
            f"[dim]💡 Found {result_count} results. "
            f"Fork the most relevant one with: "
            f"[cyan]smartfork fork <session_id>[/cyan][/dim]",
            border_style="dim"
        )
        self.console.print(tip)
    
    def suggest_compaction_check(self, at_risk_count: int):
        """Warn about sessions at risk of compaction."""
        if at_risk_count > 0:
            warning = Panel(
                f"[bold yellow]⚠️ {at_risk_count} sessions at risk of compaction[/bold yellow]\n\n"
                "These sessions may be auto-deleted by Kilo Code.\n"
                "💡 Run [cyan]smartfork compaction-export[/cyan] to preserve them",
                border_style="yellow"
            )
            self.console.print(warning)


# Integration in CLI

@app.command()
def index(...):
    # ... indexing logic ...
    
    # Show contextual help after completion
    from .ui.contextual_help import ContextualHelpManager
    help_mgr = ContextualHelpManager()
    help_mgr.after_index(result.indexed)


@app.command()
def search(...):
    # ... search logic ...
    
    from .ui.contextual_help import ContextualHelpManager
    help_mgr = ContextualHelpManager()
    
    if not results:
        help_mgr.after_search_no_results(query)
    else:
        help_mgr.after_search_with_results(query, len(results))
```

**Time Estimate:** 4 hours
**Files Created:** [`ui/contextual_help.py`](smartfork/src/smartfork/ui/contextual_help.py)

---

## Implementation Timeline

```
Week 1: Foundation
├── Day 1-2: Rich Search Output
│   ├── Create ui/rich_output.py
│   ├── Update search command
│   └── Update detect-fork command
│
└── Day 3-4: Indexing Progress Bars
    ├── Create ui/progress.py
    └── Update index command

Week 2: Interactive Features
├── Day 1-3: Interactive Shell
│   ├── Create ui/interactive.py
│   ├── Add cmd-based shell
│   └── Test all commands in interactive mode
│
└── Day 4-5: Contextual Help
    ├── Create ui/contextual_help.py
    ├── Add state detection
    └── Integrate into all commands

Week 3: Polish & Testing
├── Day 1-2: Cross-platform testing
│   ├── Windows CMD/PowerShell
│   ├── macOS Terminal
│   └── Linux terminals
│
├── Day 3: Unicode handling
│   ├── Detect terminal capabilities
│   └── Fallback to ASCII-only
│
└── Day 4-5: Documentation & Examples
    ├── Update README with new UI
    ├── Create demo GIFs
    └── Add troubleshooting guide
```

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Time to first search | 15-30 min | 5 min |
| Command discoverability | Low (must read docs) | High (contextual help) |
| Visual appeal | Plain text | Rich panels, progress bars |
| Interactive efficiency | Low (full commands) | High (shortcuts, quick fork) |
| Error recovery | Manual | Guided suggestions |

---

## Files Summary

### New Files
1. [`smartfork/src/smartfork/ui/__init__.py`](smartfork/src/smartfork/ui/__init__.py) - UI package
2. [`smartfork/src/smartfork/ui/rich_output.py`](smartfork/src/smartfork/ui/rich_output.py) - Rich search display
3. [`smartfork/src/smartfork/ui/interactive.py`](smartfork/src/smartfork/ui/interactive.py) - Interactive shell
4. [`smartfork/src/smartfork/ui/progress.py`](smartfork/src/smartfork/ui/progress.py) - Progress indicators
5. [`smartfork/src/smartfork/ui/contextual_help.py`](smartfork/src/smartfork/ui/contextual_help.py) - Smart suggestions

### Modified Files
1. [`smartfork/src/smartfork/cli.py`](smartfork/src/smartfork/cli.py) - Integrate all UI features

### Total Time Estimate: 22 hours (3 weeks at 7 hrs/week)
